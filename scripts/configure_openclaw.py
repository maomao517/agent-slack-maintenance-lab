#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def configure(
    config: dict,
    *,
    provider: str,
    model: str,
    base_url: str,
    context_window: int,
    max_tokens: int,
) -> dict:
    agents = config.setdefault("agents", {})
    defaults = agents.setdefault("defaults", {})
    defaults["model"] = {"primary": f"{provider}/{model}"}

    models = config.setdefault("models", {})
    models["mode"] = "merge"
    providers = models.setdefault("providers", {})
    providers[provider] = {
        "baseUrl": base_url,
        "apiKey": "EMPTY",
        "api": "openai-completions",
        "headers": {"X-ContextPilot-Scope": "all"},
        "models": [
            {
                "id": model,
                "name": "Qwen3 4B via SGLang",
                "reasoning": False,
                "input": ["text"],
                "contextWindow": context_window,
                "maxTokens": max_tokens,
            }
        ],
    }
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Configure a local OpenClaw provider")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--provider", default="sglang")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--base-url", default="http://127.0.0.1:30100/v1")
    parser.add_argument("--context-window", type=int, default=65536)
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()

    args.config = args.config.expanduser()
    args.config.parent.mkdir(parents=True, exist_ok=True)
    if args.config.exists():
        backup = args.config.with_suffix(args.config.suffix + ".before-slackmaint")
        shutil.copy2(args.config, backup)
        config = json.loads(args.config.read_text(encoding="utf-8"))
        print(f"Backup: {backup}")
    else:
        config = {}

    configure(
        config,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        context_window=args.context_window,
        max_tokens=args.max_tokens,
    )
    args.config.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Configured: {args.config}")


if __name__ == "__main__":
    main()
