from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .generator import generate_experiment
from .lease_simulator import (
    LeaseExperimentSpec,
    LeasePolicy,
    LeaseSimulator,
)
from .models import ExperimentSpec, PolicyKind
from .simulator import Simulator
from .trace_conversion import convert_trace_file


def _load_spec(path: Path) -> ExperimentSpec:
    return ExperimentSpec.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _parse_policies(value: str) -> list[PolicyKind]:
    if value == "all":
        return list(PolicyKind)
    return [PolicyKind(item.strip()) for item in value.split(",") if item.strip()]


def _print_results(results: list[dict[str, object]]) -> None:
    headers = (
        "policy",
        "average_jct_ms",
        "p95_jct_ms",
        "freshness_violations",
        "freshness_block_ms",
        "maintenance_overlap_ratio",
        "maintenance_backlog_ms",
    )
    print(" | ".join(headers))
    print("-" * 112)
    for result in results:
        values = []
        for header in headers:
            value = result[header]
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        print(" | ".join(values))


def _run(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    results = [
        Simulator(spec, policy).run().to_dict()
        for policy in _parse_policies(args.policies)
    ]
    _print_results(results)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote {args.output}")
    return 0


def _generate(args: argparse.Namespace) -> int:
    config = generate_experiment(args.seed, args.workflows, args.turns)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")
    return 0


def _convert_trace(args: argparse.Namespace) -> int:
    spec = convert_trace_file(
        args.input,
        arm=args.arm,
        maintenance_ms=args.maintenance_ms,
        tick_ms=args.tick_ms,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(spec, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")
    return 0


def _parse_lease_policies(value: str) -> list[LeasePolicy]:
    if value == "all":
        return list(LeasePolicy)
    return [LeasePolicy(item.strip()) for item in value.split(",") if item.strip()]


def _leasebench(args: argparse.Namespace) -> int:
    spec = LeaseExperimentSpec.from_dict(
        json.loads(args.config.read_text(encoding="utf-8"))
    )
    if args.capacity_mb is not None:
        spec = replace(spec, retention_capacity_mb=args.capacity_mb)
    if args.fixed_kv_ttl_ms is not None:
        spec = replace(spec, fixed_kv_ttl_ms=args.fixed_kv_ttl_ms)
    results = [
        LeaseSimulator(spec, policy).run().to_dict()
        for policy in _parse_lease_policies(args.policies)
    ]
    headers = (
        "policy",
        "average_jct_ms",
        "p95_jct_ms",
        "total_recompute_ms",
        "kv_hits",
        "encoder_hits",
        "cache_misses",
        "demotions",
        "forced_evictions",
        "peak_retained_mb",
    )
    print(" | ".join(headers))
    print("-" * 150)
    for result in results:
        print(
            " | ".join(
                f"{result[header]:.3f}"
                if isinstance(result[header], float)
                else str(result[header])
                for header in headers
            )
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="slackmaint")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run a trace experiment")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--policies", default="all")
    run_parser.add_argument("--output", type=Path)
    run_parser.set_defaults(handler=_run)

    generate_parser = subparsers.add_parser(
        "generate", help="generate a deterministic synthetic trace"
    )
    generate_parser.add_argument("--output", type=Path, required=True)
    generate_parser.add_argument("--seed", type=int, default=42)
    generate_parser.add_argument("--workflows", type=int, default=8)
    generate_parser.add_argument("--turns", type=int, default=4)
    generate_parser.set_defaults(handler=_generate)

    trace_parser = subparsers.add_parser(
        "convert-trace", help="convert traced LLM calls into a simulator config"
    )
    trace_parser.add_argument("--input", type=Path, required=True)
    trace_parser.add_argument("--output", type=Path, required=True)
    trace_parser.add_argument("--arm", default="Direct")
    trace_parser.add_argument("--maintenance-ms", type=int, required=True)
    trace_parser.add_argument("--tick-ms", type=int, default=1)
    trace_parser.set_defaults(handler=_convert_trace)

    lease_parser = subparsers.add_parser(
        "leasebench",
        help="compare KV and multimodal encoder state lease policies",
    )
    lease_parser.add_argument("--config", type=Path, required=True)
    lease_parser.add_argument("--policies", default="all")
    lease_parser.add_argument("--output", type=Path)
    lease_parser.add_argument("--capacity-mb", type=int)
    lease_parser.add_argument("--fixed-kv-ttl-ms", type=int)
    lease_parser.set_defaults(handler=_leasebench)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
