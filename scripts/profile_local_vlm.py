#!/usr/bin/env python3
"""Offline single-GPU profiler for a local Hugging Face vision-language model."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor


MIB = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure VLM prefill time, vision-encoder time, reusable encoder "
            "output size, KV size, and peak allocated GPU memory."
        )
    )
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument(
        "--prompt",
        default="Describe the document image briefly and identify its main topic.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(item) for item in value)
    if hasattr(value, "to_legacy_cache"):
        try:
            return tensor_bytes(value.to_legacy_cache())
        except (AttributeError, NotImplementedError, TypeError):
            pass
    if hasattr(value, "__dict__"):
        return tensor_bytes(vars(value))
    return 0


def largest_tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return max((largest_tensor_bytes(item) for item in value.values()), default=0)
    if isinstance(value, (tuple, list)):
        return max((largest_tensor_bytes(item) for item in value), default=0)
    if hasattr(value, "__dict__"):
        return largest_tensor_bytes(vars(value))
    return 0


def load_model(model_dir: Path, device: str):
    try:
        from transformers import AutoModelForImageTextToText

        model_cls = AutoModelForImageTextToText
    except ImportError:
        from transformers import AutoModelForVision2Seq

        model_cls = AutoModelForVision2Seq

    model = model_cls.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    model.eval()
    model.to(device)
    return model


def find_visual_module(model: torch.nn.Module) -> tuple[str, torch.nn.Module]:
    modules = dict(model.named_modules())
    preferred = (
        "model.visual",
        "visual",
        "model.vision_tower",
        "vision_tower",
        "model.vision_model",
        "vision_model",
    )
    for name in preferred:
        if name in modules:
            return name, modules[name]

    candidates = [
        (name, module)
        for name, module in modules.items()
        if name and any(token in name.lower() for token in ("visual", "vision"))
    ]
    if not candidates:
        raise RuntimeError(
            "No visual module was found. Print model.named_modules() and add its "
            "top-level vision module name to find_visual_module()."
        )
    return min(candidates, key=lambda item: (item[0].count("."), len(item[0])))


def prepare_inputs(processor, image_path: Path, prompt: str, device: str):
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    moved = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            value = value.to(device)
            if value.is_floating_point():
                value = value.to(torch.float16)
        moved[key] = value
    return image, moved


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * fraction + 0.999) - 1))
    return ordered[index]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this profiler requires one CUDA GPU.")
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("--warmup must be >= 0 and --repeats must be > 0")
    if not args.model_dir.is_dir():
        raise FileNotFoundError(args.model_dir)
    if not args.image.is_file():
        raise FileNotFoundError(args.image)

    processor = AutoProcessor.from_pretrained(
        str(args.model_dir),
        local_files_only=True,
        trust_remote_code=True,
    )
    model = load_model(args.model_dir, args.device)
    visual_name, visual_module = find_visual_module(model)
    image, inputs = prepare_inputs(processor, args.image, args.prompt, args.device)

    visual_measurement: dict[str, Any] = {}

    def visual_pre_hook(_module, _inputs):
        visual_measurement["start"] = torch.cuda.Event(enable_timing=True)
        visual_measurement["start"].record()

    def visual_post_hook(_module, _inputs, output):
        visual_measurement["end"] = torch.cuda.Event(enable_timing=True)
        visual_measurement["end"].record()
        visual_measurement["output_bytes"] = largest_tensor_bytes(output)

    pre_handle = visual_module.register_forward_pre_hook(visual_pre_hook)
    post_handle = visual_module.register_forward_hook(visual_post_hook)
    rows: list[dict[str, Any]] = []

    try:
        with torch.inference_mode():
            for iteration in range(args.warmup + args.repeats):
                visual_measurement.clear()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(args.device)
                torch.cuda.synchronize(args.device)

                started = time.perf_counter()
                outputs = model(**inputs, use_cache=True, return_dict=True)
                torch.cuda.synchronize(args.device)
                forward_ms = (time.perf_counter() - started) * 1000

                if "start" not in visual_measurement or "end" not in visual_measurement:
                    raise RuntimeError(
                        f"Visual module {visual_name!r} was not executed during forward."
                    )
                encoder_ms = visual_measurement["start"].elapsed_time(
                    visual_measurement["end"]
                )
                kv_bytes = tensor_bytes(getattr(outputs, "past_key_values", None))
                if kv_bytes == 0:
                    raise RuntimeError(
                        "The model returned no measurable past_key_values with use_cache=True."
                    )

                if iteration >= args.warmup:
                    rows.append(
                        {
                            "iteration": iteration - args.warmup,
                            "model_dir": str(args.model_dir.resolve()),
                            "image": str(args.image.resolve()),
                            "image_width": image.width,
                            "image_height": image.height,
                            "input_tokens": int(inputs["input_ids"].shape[-1]),
                            "visual_module": visual_name,
                            "forward_ms": round(forward_ms, 4),
                            "encoder_ms": round(encoder_ms, 4),
                            "prefill_ms_estimate": round(max(0.0, forward_ms - encoder_ms), 4),
                            "encoder_size_mb": round(
                                visual_measurement["output_bytes"] / MIB, 4
                            ),
                            "kv_size_mb": round(kv_bytes / MIB, 4),
                            "peak_gpu_memory_mb": round(
                                torch.cuda.max_memory_allocated(args.device) / MIB, 4
                            ),
                            "dtype": "float16",
                            "attention": "eager",
                            "gpu": torch.cuda.get_device_name(args.device),
                        }
                    )
                del outputs
    finally:
        pre_handle.remove()
        post_handle.remove()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"model: {args.model_dir}")
    print(f"gpu: {torch.cuda.get_device_name(args.device)}")
    print(f"visual module: {visual_name}")
    for field in ("forward_ms", "encoder_ms", "prefill_ms_estimate"):
        values = [float(row[field]) for row in rows]
        print(
            f"{field}: median={statistics.median(values):.3f} ms, "
            f"p95={percentile(values, 0.95):.3f} ms"
        )
    print(f"encoder_size_mb: {rows[0]['encoder_size_mb']}")
    print(f"kv_size_mb: {rows[0]['kv_size_mb']}")
    print(f"peak_gpu_memory_mb: {max(row['peak_gpu_memory_mb'] for row in rows)}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
