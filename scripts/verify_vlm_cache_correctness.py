#!/usr/bin/env python3
"""Verify that a cached Qwen3-VL visual state is complete and lossless.

This is a targeted P0 check. It runs one cold path and one encoder-hit path with
the same image and prompt, then checks the complete cached tensor structure,
logits, greedy generation, and the visual module invocation count on the hit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from slackmaint.document_service import TransformersQwen3VLBackend, tensor_bytes


MIB = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check complete visual-state caching against a cold Qwen3-VL path."
    )
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--prompt",
        default="请简要说明该文档页面的主要内容，并给出一个可核验的细节。",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="float16"
    )
    parser.add_argument("--attention", default="eager")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--logit-atol", type=float, default=1e-3)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when any required correctness check fails.",
    )
    return parser.parse_args()


def tensor_leaves(value: Any, path: str = "features") -> Iterator[tuple[str, Any]]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - backend import needs torch first
        raise RuntimeError("PyTorch is required") from exc
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from tensor_leaves(item, f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from tensor_leaves(item, f"{path}[{index}]")


def summarize_tensors(value: Any) -> list[dict[str, Any]]:
    return [
        {
            "path": path,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "device": str(tensor.device),
            "size_mb": round(tensor.numel() * tensor.element_size() / MIB, 6),
        }
        for path, tensor in tensor_leaves(value)
    ]


def deepstack_count(features: Any) -> int:
    """Return the number of DeepStack layers in the validated Qwen3-VL tuple."""
    if not isinstance(features, (tuple, list)) or len(features) < 2:
        return 0
    deepstack = features[1]
    return len(deepstack) if isinstance(deepstack, (tuple, list)) else 0


def find_visual_module(model: Any) -> tuple[str, Any] | None:
    modules = dict(model.named_modules())
    for name in (
        "model.visual",
        "visual",
        "model.vision_tower",
        "vision_tower",
        "model.vision_model",
        "vision_model",
    ):
        if name in modules:
            return name, modules[name]
    candidates = [
        (name, module)
        for name, module in modules.items()
        if name and any(token in name.lower() for token in ("visual", "vision"))
    ]
    return min(candidates, key=lambda item: (item[0].count("."), len(item[0]))) if candidates else None


@contextmanager
def injected_features(backend: TransformersQwen3VLBackend, features: Any):
    """Temporarily bypass the visual encoder with a cached feature tuple."""
    original = backend.feature_owner.get_image_features
    calls = {"injected_get_image_features_calls": 0}

    def inject(*_args: Any, **_kwargs: Any) -> Any:
        calls["injected_get_image_features_calls"] += 1
        return features

    backend.feature_owner.get_image_features = inject
    try:
        yield calls
    finally:
        backend.feature_owner.get_image_features = original


def synchronized_forward(backend: TransformersQwen3VLBackend, inputs: dict[str, Any]) -> Any:
    backend._sync()
    with backend.torch.inference_mode():
        output = backend.model(**inputs, use_cache=False, return_dict=True)
    backend._sync()
    return output


def synchronized_generate(
    backend: TransformersQwen3VLBackend, inputs: dict[str, Any], max_new_tokens: int
) -> Any:
    backend._sync()
    with backend.torch.inference_mode():
        generated = backend.model.generate(
            **inputs,
            do_sample=False,
            use_cache=True,
            max_new_tokens=max_new_tokens,
        )
    backend._sync()
    return generated


def main() -> None:
    args = parse_args()
    if not args.model_dir.is_dir():
        raise FileNotFoundError(args.model_dir)
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")

    backend = TransformersQwen3VLBackend(
        args.model_dir,
        device=args.device,
        dtype=args.dtype,
        attention=args.attention,
    )
    state, encoder_ms = backend.encode(args.image, args.prompt)
    features = state["features"]
    cached_state = backend.to_cache(state)
    restored_state, restore_ms = backend.from_cache(cached_state)
    restored_features = restored_state["features"]
    inputs = backend._prepare(args.image, args.prompt)

    cold_started = time.perf_counter_ns()
    cold_output = synchronized_forward(backend, inputs)
    cold_forward_ms = (time.perf_counter_ns() - cold_started) / 1_000_000
    cold_logits = cold_output.logits.detach().float().cpu()
    del cold_output
    cold_generated = synchronized_generate(backend, inputs, args.max_new_tokens)

    visual_module = find_visual_module(backend.model)
    visual_calls = 0
    hook = None
    if visual_module is not None:
        _name, module = visual_module

        def count_visual_calls(*_args: Any, **_kwargs: Any) -> None:
            nonlocal visual_calls
            visual_calls += 1

        hook = module.register_forward_hook(count_visual_calls)

    try:
        with injected_features(backend, restored_features) as injection:
            hit_started = time.perf_counter_ns()
            hit_output = synchronized_forward(backend, inputs)
            hit_forward_ms = (time.perf_counter_ns() - hit_started) / 1_000_000
            hit_logits = hit_output.logits.detach().float().cpu()
            del hit_output
            hit_generated = synchronized_generate(backend, inputs, args.max_new_tokens)
    finally:
        if hook is not None:
            hook.remove()

    max_abs_logit_error = float((cold_logits - hit_logits).abs().max().item())
    greedy_output_equal = bool(backend.torch.equal(cold_generated, hit_generated))
    required_deepstack_count = 3
    complete_state = deepstack_count(features) == required_deepstack_count
    hit_skipped_visual_encoder = visual_module is None or visual_calls == 0
    passed = (
        complete_state
        and hit_skipped_visual_encoder
        and max_abs_logit_error <= args.logit_atol
        and greedy_output_equal
    )
    image_hash = hashlib.sha256(args.image.read_bytes()).hexdigest()
    result = {
        "schema": "slackmaint.vlm-cache-correctness.v1",
        "passed": passed,
        "model_dir": str(args.model_dir.resolve()),
        "image": str(args.image.resolve()),
        "image_sha256": image_hash,
        "device": args.device,
        "dtype": args.dtype,
        "attention": args.attention,
        "encoder_ms": round(encoder_ms, 3),
        "restore_ms": round(restore_ms, 3),
        "cold_forward_ms": round(cold_forward_ms, 3),
        "hit_forward_ms": round(hit_forward_ms, 3),
        "state_size_mb": round(tensor_bytes(cached_state) / MIB, 6),
        "cached_deepstack_count": deepstack_count(cached_state["features"]),
        "complete_state": complete_state,
        "visual_module": visual_module[0] if visual_module is not None else None,
        "visual_module_calls_on_hit": visual_calls if visual_module is not None else None,
        "encoder_calls_on_hit": 0 if hit_skipped_visual_encoder else visual_calls,
        "injected_get_image_features_calls": injection["injected_get_image_features_calls"],
        "max_abs_logit_error": max_abs_logit_error,
        "logit_atol": args.logit_atol,
        "greedy_output_equal": greedy_output_equal,
        "cold_generation_sha256": hashlib.sha256(
            cold_generated.detach().cpu().numpy().tobytes()
        ).hexdigest(),
        "hit_generation_sha256": hashlib.sha256(
            hit_generated.detach().cpu().numpy().tobytes()
        ).hexdigest(),
        "cached_tensor_summary": summarize_tensors(cached_state["features"]),
        "restored_tensor_summary": summarize_tensors(restored_features),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.strict and not passed:
        raise SystemExit("visual-state correctness validation failed")


if __name__ == "__main__":
    main()
