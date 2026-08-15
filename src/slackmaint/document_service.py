"""Runtime support for OpenCode-driven visual document experiments."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


MIB = 1024 * 1024


def tensor_bytes(value: Any) -> int:
    """Count tensor payload bytes in a nested reusable vision state."""
    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(item) for item in value)
    return 0


def move_nested(value: Any, device: str, *, pin_memory: bool = False) -> Any:
    """Move nested tensors while preserving the model's return structure."""
    import torch

    if isinstance(value, torch.Tensor):
        moved = value.detach().to(device)
        if pin_memory and device == "cpu" and torch.cuda.is_available():
            moved = moved.pin_memory()
        return moved
    if isinstance(value, tuple):
        return tuple(move_nested(item, device, pin_memory=pin_memory) for item in value)
    if isinstance(value, list):
        return [move_nested(item, device, pin_memory=pin_memory) for item in value]
    if isinstance(value, dict):
        return {
            key: move_nested(item, device, pin_memory=pin_memory)
            for key, item in value.items()
        }
    return value


class JsonlEventWriter:
    """Append complete JSON lines safely across service threads."""

    def __init__(self, path: Path | None):
        self.path = path
        self.lock = threading.Lock()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: dict[str, Any]) -> None:
        if self.path is None:
            return
        payload = json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
        with self.lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()


@dataclass
class CacheEntry:
    state: Any
    size_bytes: int
    created_unix_ns: int
    last_access_unix_ns: int


class LRUStateCache:
    """Capacity-bounded CPU cache for complete reusable visual states."""

    def __init__(self, capacity_bytes: int):
        if capacity_bytes < 0:
            raise ValueError("capacity_bytes must be non-negative")
        self.capacity_bytes = capacity_bytes
        self.entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self.used_bytes = 0
        self.evictions = 0
        self.lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self.lock:
            entry = self.entries.get(key)
            if entry is None:
                return None
            entry.last_access_unix_ns = time.time_ns()
            self.entries.move_to_end(key)
            return entry.state

    def put(self, key: str, state: Any, size_bytes: int) -> bool:
        if size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")
        if size_bytes > self.capacity_bytes:
            return False
        now = time.time_ns()
        with self.lock:
            old = self.entries.pop(key, None)
            if old is not None:
                self.used_bytes -= old.size_bytes
            while self.entries and self.used_bytes + size_bytes > self.capacity_bytes:
                _, evicted = self.entries.popitem(last=False)
                self.used_bytes -= evicted.size_bytes
                self.evictions += 1
            self.entries[key] = CacheEntry(state, size_bytes, now, now)
            self.used_bytes += size_bytes
            return True

    def clear(self) -> int:
        with self.lock:
            count = len(self.entries)
            self.entries.clear()
            self.used_bytes = 0
            return count

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "capacity_mb": round(self.capacity_bytes / MIB, 6),
                "used_mb": round(self.used_bytes / MIB, 6),
                "entry_count": len(self.entries),
                "evictions": self.evictions,
            }


class DocumentBackend(Protocol):
    model_config_hash: str
    backend_name: str

    def encode(self, image_path: Path, prompt: str) -> tuple[Any, float]: ...

    def generate(
        self,
        image_path: Path,
        prompt: str,
        state: Any,
        max_new_tokens: int,
    ) -> tuple[str, float]: ...

    def to_cache(self, state: Any) -> Any: ...

    def from_cache(self, state: Any) -> tuple[Any, float]: ...

    def state_size_bytes(self, state: Any) -> int: ...

    def begin_request(self) -> None: ...

    def runtime_metrics(self) -> dict[str, float]: ...


class MockDocumentBackend:
    """Dependency-free backend for protocol and OpenCode smoke tests."""

    backend_name = "mock"

    def __init__(self, encoder_delay_ms: float = 0.0, restore_delay_ms: float = 0.0):
        self.encoder_delay_ms = encoder_delay_ms
        self.restore_delay_ms = restore_delay_ms
        self.model_config_hash = "mock-v1"
        self.encoder_calls = 0

    def encode(self, image_path: Path, prompt: str) -> tuple[dict[str, str], float]:
        started = time.perf_counter_ns()
        if self.encoder_delay_ms:
            time.sleep(self.encoder_delay_ms / 1000)
        self.encoder_calls += 1
        state = {
            "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        }
        return state, (time.perf_counter_ns() - started) / 1_000_000

    def generate(
        self,
        image_path: Path,
        prompt: str,
        state: dict[str, str],
        max_new_tokens: int,
    ) -> tuple[str, float]:
        started = time.perf_counter_ns()
        answer = (
            f"mock analysis for {image_path.name}: {prompt[:120]} "
            f"[{state['image_sha256'][:12]}]"
        )
        return answer, (time.perf_counter_ns() - started) / 1_000_000

    def to_cache(self, state: dict[str, str]) -> dict[str, str]:
        return dict(state)

    def from_cache(self, state: dict[str, str]) -> tuple[dict[str, str], float]:
        started = time.perf_counter_ns()
        if self.restore_delay_ms:
            time.sleep(self.restore_delay_ms / 1000)
        return dict(state), (time.perf_counter_ns() - started) / 1_000_000

    def state_size_bytes(self, state: dict[str, str]) -> int:
        return len(json.dumps(state, sort_keys=True).encode("utf-8"))

    def begin_request(self) -> None:
        return None

    def runtime_metrics(self) -> dict[str, float]:
        return {}


class TransformersQwen3VLBackend:
    """Qwen3-VL backend that reuses image embeddings and DeepStack features."""

    backend_name = "transformers-qwen3-vl"

    def __init__(
        self,
        model_dir: Path,
        *,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        attention: str = "eager",
        pin_memory: bool = True,
    ):
        import torch
        from transformers import AutoProcessor

        if not torch.cuda.is_available() and device.startswith("cuda"):
            raise RuntimeError("CUDA is unavailable")
        self.torch = torch
        self.device = device
        self.pin_memory = pin_memory
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if dtype not in dtype_map:
            raise ValueError(f"unsupported dtype: {dtype}")
        self.dtype = dtype_map[dtype]
        self.processor = AutoProcessor.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=True
        )
        try:
            from transformers import AutoModelForImageTextToText

            model_cls = AutoModelForImageTextToText
        except ImportError:
            from transformers import AutoModelForVision2Seq

            model_cls = AutoModelForVision2Seq
        self.model = model_cls.from_pretrained(
            str(model_dir),
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            attn_implementation=attention,
        )
        self.model.eval()
        self.model.to(device)
        self.feature_owner = self._find_feature_owner()
        config_names = (
            "config.json",
            "preprocessor_config.json",
            "processor_config.json",
            "tokenizer_config.json",
        )
        config_parts = []
        for name in config_names:
            path = model_dir / name
            if path.is_file():
                config_parts.extend(
                    (name.encode("utf-8"), b"\0", path.read_bytes(), b"\0")
                )
        identity = (
            str(model_dir.resolve()).encode("utf-8")
            + b"\0"
            + b"".join(config_parts)
        )
        self.model_config_hash = hashlib.sha256(identity).hexdigest()

    def _find_feature_owner(self):
        candidates = [self.model, getattr(self.model, "model", None)]
        for candidate in candidates:
            if candidate is not None and callable(
                getattr(candidate, "get_image_features", None)
            ):
                return candidate
        raise RuntimeError(
            "Model does not expose get_image_features; the installed Transformers "
            "model interface is incompatible with the validated Qwen3-VL path."
        )

    def _prepare(self, image_path: Path, prompt: str) -> dict[str, Any]:
        from PIL import Image

        with Image.open(image_path) as source:
            image = source.convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[image], padding=True, return_tensors="pt"
        )
        moved: dict[str, Any] = {}
        for key, value in inputs.items():
            if isinstance(value, self.torch.Tensor):
                value = value.to(self.device)
                if value.is_floating_point():
                    value = value.to(self.dtype)
            moved[key] = value
        return moved

    def _sync(self) -> None:
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize(self.device)

    def encode(self, image_path: Path, prompt: str) -> tuple[dict[str, Any], float]:
        inputs = self._prepare(image_path, prompt)
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")
        if pixel_values is None or image_grid_thw is None:
            raise RuntimeError("processor did not return pixel_values and image_grid_thw")
        self._sync()
        started = time.perf_counter_ns()
        with self.torch.inference_mode():
            features = self.feature_owner.get_image_features(
                pixel_values, image_grid_thw
            )
        self._sync()
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        return {"features": features, "inputs": inputs}, elapsed_ms

    def generate(
        self,
        image_path: Path,
        prompt: str,
        state: dict[str, Any],
        max_new_tokens: int,
    ) -> tuple[str, float]:
        inputs = state.get("inputs")
        if inputs is None:
            inputs = self._prepare(image_path, prompt)
        features = state["features"]
        original = self.feature_owner.get_image_features

        def inject_features(*_args, **_kwargs):
            return features

        self.feature_owner.get_image_features = inject_features
        self._sync()
        started = time.perf_counter_ns()
        try:
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            self._sync()
        finally:
            self.feature_owner.get_image_features = original
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        prompt_length = int(inputs["input_ids"].shape[-1])
        generated_only = generated[:, prompt_length:]
        answer = self.processor.batch_decode(
            generated_only,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return answer, elapsed_ms

    def to_cache(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "features": move_nested(
                state["features"], "cpu", pin_memory=self.pin_memory
            )
        }

    def from_cache(self, state: dict[str, Any]) -> tuple[dict[str, Any], float]:
        self._sync()
        started = time.perf_counter_ns()
        restored = {"features": move_nested(state["features"], self.device)}
        self._sync()
        return restored, (time.perf_counter_ns() - started) / 1_000_000

    def state_size_bytes(self, state: dict[str, Any]) -> int:
        return tensor_bytes(state["features"])

    def begin_request(self) -> None:
        if self.device.startswith("cuda"):
            self._sync()
            self.torch.cuda.reset_peak_memory_stats(self.device)

    def runtime_metrics(self) -> dict[str, float]:
        if not self.device.startswith("cuda"):
            return {}
        self._sync()
        return {
            "current_gpu_memory_mb": round(
                self.torch.cuda.memory_allocated(self.device) / MIB, 6
            ),
            "peak_gpu_memory_mb": round(
                self.torch.cuda.max_memory_allocated(self.device) / MIB, 6
            ),
        }


class DocumentAnalysisService:
    """Cache policy and trace layer shared by real and mock backends."""

    POLICIES = {"no_cache", "task_local", "shared_cpu"}

    def __init__(
        self,
        backend: DocumentBackend,
        *,
        cache_policy: str,
        cache_capacity_mb: float,
        event_writer: JsonlEventWriter,
    ):
        if cache_policy not in self.POLICIES:
            raise ValueError(f"unsupported cache policy: {cache_policy}")
        self.backend = backend
        self.cache_policy = cache_policy
        self.cache = LRUStateCache(int(cache_capacity_mb * MIB))
        self.event_writer = event_writer
        self.model_lock = threading.Lock()
        self.metrics_lock = threading.Lock()
        self.request_count = 0
        self.cache_hits = 0
        self.encoder_calls = 0
        self.errors = 0

    def _key(self, request: dict[str, Any], image_sha256: str) -> str:
        parts = [
            image_sha256,
            str(request["document_version"]),
            self.backend.model_config_hash,
        ]
        if self.cache_policy == "task_local":
            parts.append(str(request["workflow_id"]))
        return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()

    def analyze(self, request: dict[str, Any], image_path: Path) -> dict[str, Any]:
        required = (
            "run_id",
            "workflow_id",
            "task_id",
            "turn_id",
            "document_id",
            "page_id",
            "document_version",
            "question",
        )
        missing = [key for key in required if key not in request]
        if missing:
            raise ValueError(f"missing required fields: {', '.join(missing)}")
        image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
        cache_key = self._key(request, image_sha256)
        request_start_ns = time.time_ns()
        started = time.perf_counter_ns()
        encoder_called = False
        cache_hit = False
        restore_ms = 0.0
        encoder_ms = 0.0
        generation_ms = 0.0
        state_size_bytes = 0
        runtime_metrics: dict[str, float] = {}
        error: str | None = None

        try:
            with self.model_lock:
                self.backend.begin_request()
                cached = None
                if self.cache_policy != "no_cache":
                    cached = self.cache.get(cache_key)
                if cached is not None:
                    cache_hit = True
                    state, restore_ms = self.backend.from_cache(cached)
                else:
                    encoder_called = True
                    state, encoder_ms = self.backend.encode(
                        image_path, str(request["question"])
                    )
                state_size_bytes = self.backend.state_size_bytes(state)
                answer, generation_ms = self.backend.generate(
                    image_path,
                    str(request["question"]),
                    state,
                    int(request.get("max_new_tokens", 128)),
                )
                if not cache_hit and self.cache_policy != "no_cache":
                    cached_state = self.backend.to_cache(state)
                    cached_size = self.backend.state_size_bytes(cached_state)
                    self.cache.put(cache_key, cached_state, cached_size)
                runtime_metrics = self.backend.runtime_metrics()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            with self.metrics_lock:
                self.errors += 1
            raise
        finally:
            end_ns = time.time_ns()
            total_ms = (time.perf_counter_ns() - started) / 1_000_000
            event = {
                "schema": "slackmaint.visual-access.v1",
                "event": "visual_access",
                "backend": self.backend.backend_name,
                "cache_policy": self.cache_policy,
                "cache_key": cache_key,
                "cache_hit": cache_hit,
                "hit_tier": "cpu" if cache_hit else None,
                "encoder_called": encoder_called,
                "encoder_ms": round(encoder_ms, 3),
                "restore_ms": round(restore_ms, 3),
                "generation_ms": round(generation_ms, 3),
                "total_ms": round(total_ms, 3),
                "state_size_mb": round(state_size_bytes / MIB, 6),
                "h2d_state_transfer_mb": round(state_size_bytes / MIB, 6)
                if cache_hit
                else 0.0,
                "image_sha256": image_sha256,
                "start_unix_ns": request_start_ns,
                "end_unix_ns": end_ns,
                "error": error,
                **runtime_metrics,
                **{key: request.get(key) for key in required if key != "question"},
            }
            self.event_writer.append(event)

        with self.metrics_lock:
            self.request_count += 1
            self.cache_hits += int(cache_hit)
            self.encoder_calls += int(encoder_called)
        return {
            "ok": True,
            "answer": answer,
            "document_id": request["document_id"],
            "page_id": request["page_id"],
            "document_version": request["document_version"],
            "image_sha256": image_sha256,
            "cache_policy": self.cache_policy,
            "cache_hit": cache_hit,
            "hit_tier": "cpu" if cache_hit else None,
            "encoder_called": encoder_called,
            "encoder_ms": round(encoder_ms, 3),
            "restore_ms": round(restore_ms, 3),
            "generation_ms": round(generation_ms, 3),
            "total_ms": round((time.perf_counter_ns() - started) / 1_000_000, 3),
            "state_size_mb": round(state_size_bytes / MIB, 6),
            "h2d_state_transfer_mb": round(state_size_bytes / MIB, 6)
            if cache_hit
            else 0.0,
            **runtime_metrics,
        }

    def clear_cache(self) -> dict[str, Any]:
        return {"ok": True, "cleared_entries": self.cache.clear()}

    def metrics(self) -> dict[str, Any]:
        with self.metrics_lock:
            requests = self.request_count
            result = {
                "ok": True,
                "backend": self.backend.backend_name,
                "cache_policy": self.cache_policy,
                "request_count": requests,
                "cache_hits": self.cache_hits,
                "cache_hit_rate": self.cache_hits / requests if requests else 0.0,
                "encoder_calls": self.encoder_calls,
                "errors": self.errors,
            }
        result["cache"] = self.cache.snapshot()
        return result
