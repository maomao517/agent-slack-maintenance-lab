#!/usr/bin/env python3
"""Local HTTP service for reusable Qwen3-VL document states."""

from __future__ import annotations

import argparse
import json
import signal
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from slackmaint.document_service import (
    DocumentAnalysisService,
    JsonlEventWriter,
    MockDocumentBackend,
    TransformersQwen3VLBackend,
)


MAX_REQUEST_BYTES = 4 * 1024 * 1024


def resolve_image(data_root: Path, requested: str) -> Path:
    path = Path(requested)
    if path.is_absolute():
        candidate = path.resolve()
    else:
        candidate = (data_root / path).resolve()
    root = data_root.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("image path escapes data root") from exc
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


class DocumentRequestHandler(BaseHTTPRequestHandler):
    server_version = "SlackMaintDocumentService/0.1"

    @property
    def app(self) -> "DocumentHTTPServer":
        return self.server  # type: ignore[return-value]

    def log_message(self, fmt: str, *args: Any) -> None:
        if self.app.verbose:
            super().log_message(fmt, *args)

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ValueError("missing Content-Length")
        length = int(raw_length)
        if length < 0 or length > MAX_REQUEST_BYTES:
            raise ValueError("request body is too large")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/health":
            self._write_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "backend": self.app.service.backend.backend_name,
                    "cache_policy": self.app.service.cache_policy,
                },
            )
            return
        if path == "/metrics":
            self._write_json(HTTPStatus.OK, self.app.service.metrics())
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
            if path == "/analyze":
                image_path = resolve_image(
                    self.app.data_root, str(payload.get("image", ""))
                )
                result = self.app.service.analyze(payload, image_path)
                self._write_json(HTTPStatus.OK, result)
                return
            if path == "/cache/clear":
                self._write_json(HTTPStatus.OK, self.app.service.clear_cache())
                return
            self._write_json(
                HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"}
            )
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
            )
        except Exception as exc:
            self._write_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
            )


class DocumentHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        service: DocumentAnalysisService,
        data_root: Path,
        *,
        verbose: bool,
    ):
        super().__init__(address, DocumentRequestHandler)
        self.service = service
        self.data_root = data_root.resolve()
        self.verbose = verbose


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve cached visual document analysis over localhost HTTP"
    )
    parser.add_argument("--backend", choices=("mock", "transformers"), default="transformers")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30200)
    parser.add_argument(
        "--cache-policy",
        choices=("no_cache", "task_local", "shared_cpu"),
        default="no_cache",
    )
    parser.add_argument("--cache-capacity-mb", type=float, default=1024)
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--attention", default="eager")
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--mock-encoder-ms", type=float, default=0.0)
    parser.add_argument("--mock-restore-ms", type=float, default=0.0)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data_root.is_dir():
        raise FileNotFoundError(args.data_root)
    if args.cache_capacity_mb < 0:
        raise ValueError("--cache-capacity-mb must be non-negative")

    if args.backend == "mock":
        backend = MockDocumentBackend(
            encoder_delay_ms=args.mock_encoder_ms,
            restore_delay_ms=args.mock_restore_ms,
        )
    else:
        if args.model_dir is None or not args.model_dir.is_dir():
            raise ValueError("--model-dir must be an existing directory")
        backend = TransformersQwen3VLBackend(
            args.model_dir,
            device=args.device,
            dtype=args.dtype,
            attention=args.attention,
            pin_memory=not args.no_pin_memory,
        )

    service = DocumentAnalysisService(
        backend,
        cache_policy=args.cache_policy,
        cache_capacity_mb=args.cache_capacity_mb,
        event_writer=JsonlEventWriter(args.trace_output),
    )
    server = DocumentHTTPServer(
        (args.host, args.port), service, args.data_root, verbose=args.verbose
    )
    stop = threading.Event()

    def request_shutdown(_signum, _frame) -> None:
        if not stop.is_set():
            stop.set()
            threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)
    print(
        json.dumps(
            {
                "ok": True,
                "url": f"http://{args.host}:{args.port}",
                "backend": backend.backend_name,
                "cache_policy": args.cache_policy,
                "data_root": str(args.data_root.resolve()),
                "trace_output": str(args.trace_output) if args.trace_output else None,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
