#!/usr/bin/env python3
"""InferenceX benchmark runner CLI.

IMPORTANT: The InferenceX server must be running before executing this script.
Start it with:
    INFERENCE_X_LOADED_MODELS=<model> ./scripts/dev.sh serve
or:
    make playground  (then open a second terminal)

Usage:
    uv run python scripts/benchmark.py --model qwen2.5-0.5b
    uv run python scripts/benchmark.py --model qwen2.5-0.5b --output-dir benchmarks/results
    make benchmark MODEL=qwen2.5-0.5b
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src/ to path when running as a script (not installed package)
_src = Path(__file__).parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import httpx


def _check_server_reachable(base_url: str) -> bool:
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{base_url}/health")
            return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


def _check_model_loaded(base_url: str, model: str) -> tuple[bool, list[str]]:
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{base_url}/health")
            resp.raise_for_status()
            loaded = resp.json().get("loaded_models") or []
            return model in loaded, loaded
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError):
        return False, []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the InferenceX benchmark suite against a loaded model.",
        epilog="Server must be running before invoking this script.",
    )
    parser.add_argument("--model", required=True, help="Model name to benchmark")
    parser.add_argument(
        "--suite",
        default="benchmarks/prompts/standard.json",
        help="Path to prompt suite JSON (default: benchmarks/prompts/standard.json)",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Server base URL (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent requests (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/results",
        help="Directory to save result JSON (default: benchmarks/results)",
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")

    if not _check_server_reachable(base_url):
        print(
            f"Error: Server not reachable at {base_url}.\n"
            "Start the server first:\n"
            "  INFERENCE_X_LOADED_MODELS=<model> ./scripts/dev.sh serve\n"
            "or:\n"
            "  make playground",
            file=sys.stderr,
        )
        sys.exit(1)

    loaded_ok, loaded_models = _check_model_loaded(base_url, args.model)
    if not loaded_ok:
        loaded_str = loaded_models[0] if len(loaded_models) == 1 else str(loaded_models)
        print(
            f"Error: Model '{args.model}' is not loaded on the server.\n"
            f"Server has loaded: {loaded_str}\n"
            f"Restart with:\n"
            f"  INFERENCE_X_DEFAULT_MODEL={args.model} ./scripts/dev.sh serve\n"
            f"or:\n"
            f"  INFERENCE_X_LOADED_MODELS={args.model} ./scripts/dev.sh serve",
            file=sys.stderr,
        )
        sys.exit(1)

    suite_path = Path(args.suite)
    if not suite_path.exists():
        print(f"Error: Suite file not found: {suite_path}", file=sys.stderr)
        sys.exit(1)

    from inference_x.benchmarks.runner import BenchmarkRunner
    from inference_x.benchmarks.storage import ResultStore

    print(f"Benchmarking model '{args.model}' using suite '{args.suite}'...")
    runner = BenchmarkRunner()
    try:
        result = runner.run(
            model_name=args.model,
            suite_path=str(suite_path),
            base_url=base_url,
            concurrency=args.concurrency,
        )
    except Exception as exc:
        print(f"Error: Benchmark failed: {exc}", file=sys.stderr)
        sys.exit(1)

    store = ResultStore()
    output_path = store.save(result, output_dir=args.output_dir)
    print(f"Results saved to: {output_path}")

    print(f"\n=== {args.model} ===")
    print(f"  Mean throughput : {result.mean_throughput_tps:.1f} tok/s")
    print(f"  p50 latency     : {result.p50_latency_ms:.0f} ms")
    print(f"  p95 latency     : {result.p95_latency_ms:.0f} ms")
    print(f"  p99 latency     : {result.p99_latency_ms:.0f} ms")
    print(f"  Peak VRAM delta : {result.peak_vram_delta_gb:.2f} GB")
    print(f"  Prompts run     : {len(result.prompt_results)}")


if __name__ == "__main__":
    main()
