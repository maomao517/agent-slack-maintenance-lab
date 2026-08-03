#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
import uuid
from pathlib import Path
from urllib.parse import urljoin

from aiohttp import ClientSession, ClientTimeout, web


HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


class TraceProxy:
    def __init__(self, upstream: str, output: Path):
        self.upstream = upstream.rstrip("/") + "/"
        self.output = output
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.context: dict[str, object] = {}
        self.write_lock = asyncio.Lock()
        self.session: ClientSession | None = None

    async def start(self) -> None:
        self.session = ClientSession(timeout=ClientTimeout(total=None))

    async def close(self) -> None:
        if self.session is not None:
            await self.session.close()

    async def append(self, event: dict[str, object]) -> None:
        line = json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
        async with self.write_lock:
            with self.output.open("a", encoding="utf-8") as handle:
                handle.write(line)

    async def set_context(self, request: web.Request) -> web.Response:
        payload = await request.json()
        allowed = {"scenario", "arm", "trial", "turn", "session_id"}
        self.context = {key: payload[key] for key in allowed if key in payload}
        return web.json_response({"ok": True, "context": self.context})

    async def health(self, request: web.Request) -> web.Response:
        return web.json_response(
            {"ok": True, "upstream": self.upstream, "output": str(self.output)}
        )

    async def forward(self, request: web.Request) -> web.StreamResponse:
        request_id = uuid.uuid4().hex
        body = await request.read()
        start_ns = time.time_ns()
        labels = dict(self.context)
        target = urljoin(self.upstream, request.match_info["tail"])
        if request.query_string:
            target = f"{target}?{request.query_string}"
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in HOP_BY_HOP_HEADERS and key.lower() != "host"
        }
        response_bytes = 0
        status = 502
        error = None

        try:
            if self.session is None:
                raise RuntimeError("proxy client session is not initialized")
            async with self.session.request(
                request.method,
                target,
                data=body,
                headers=headers,
            ) as upstream_response:
                status = upstream_response.status
                response_headers = {
                    key: value
                    for key, value in upstream_response.headers.items()
                    if key.lower() not in HOP_BY_HOP_HEADERS
                }
                downstream = web.StreamResponse(
                    status=upstream_response.status,
                    headers=response_headers,
                )
                await downstream.prepare(request)
                async for chunk in upstream_response.content.iter_any():
                    response_bytes += len(chunk)
                    await downstream.write(chunk)
                await downstream.write_eof()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            downstream = web.json_response(
                {"error": "trace_proxy_upstream_failure", "detail": error},
                status=502,
            )

        end_ns = time.time_ns()
        await self.append(
            {
                "schema": "slackmaint.trace.v1",
                "event": "llm_call",
                "request_id": request_id,
                "method": request.method,
                "path": request.path,
                "status": status,
                "start_unix_ns": start_ns,
                "end_unix_ns": end_ns,
                "duration_ms": round((end_ns - start_ns) / 1_000_000, 3),
                "request_bytes": len(body),
                "response_bytes": response_bytes,
                "request_sha256": hashlib.sha256(body).hexdigest(),
                "error": error,
                **labels,
            }
        )
        return downstream


def build_app(upstream: str, output: Path) -> web.Application:
    app = web.Application(client_max_size=256 * 1024**2)
    proxy = TraceProxy(upstream, output)
    app.router.add_post("/_trace/context", proxy.set_context)
    app.router.add_get("/_trace/health", proxy.health)
    app.router.add_route("*", "/{tail:.*}", proxy.forward)
    app.on_startup.append(lambda _app: proxy.start())
    app.on_cleanup.append(lambda _app: proxy.close())
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace an OpenAI-compatible API")
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30100)
    args = parser.parse_args()
    web.run_app(build_app(args.upstream, args.output), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
