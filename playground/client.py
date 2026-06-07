#!/usr/bin/env python3
"""InferenceX playground client.

Sends prompts to a running InferenceX server and prints responses.
Supports single-model and side-by-side comparison mode.

Usage examples:
    # Single prompt, default model
    python playground/client.py "What is the capital of France?"

    # Specific model
    python playground/client.py --model qwen2.5-0.5b "Explain attention in one sentence."

    # Compare two models on the same prompt (requires two server instances)
    python playground/client.py --compare qwen2.5-0.5b tinyllama-chat \\
        --base-url-a http://localhost:8000 --base-url-b http://localhost:8001 \\
        "Write a haiku about GPUs."

    # Run all sample prompts
    python playground/client.py --prompts-file playground/prompts/sample_prompts.json

    # Compare on single GPU (restart server between models)
    python playground/client.py --compare qwen2.5-0.5b tinyllama-chat --sequential \\
        --prompts-file playground/prompts/sample_prompts.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_MODEL = "qwen2.5-0.5b"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 512
COMPARE_COL_WIDTH = 60
SEPARATOR = "─" * (COMPARE_COL_WIDTH * 2 + 3)
_LOADED_MODEL_RE = re.compile(r"loaded model is '([^']+)'")


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def chat_completion(
    prompt: str,
    model: str,
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    system: str | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    """POST /v1/chat/completions and return the parsed JSON response dict."""
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    ).encode()

    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from server: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot reach server at {base_url}: {exc.reason}\n"
            "Is the server running? Try: ./scripts/dev.sh serve"
        ) from exc


def list_models(base_url: str = DEFAULT_BASE_URL, timeout: int = 30) -> list[str]:
    """GET /v1/models and return a list of model IDs."""
    req = urllib.request.Request(f"{base_url}/v1/models", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return [m["id"] for m in data.get("data", [])]
    except Exception:
        return []


def health_check(base_url: str = DEFAULT_BASE_URL, timeout: int = 10) -> bool:
    """Return True if /health returns 200."""
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=timeout):
            return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Response formatting
# ---------------------------------------------------------------------------

def extract_text(response: dict[str, Any]) -> str:
    """Pull the assistant message content out of a chat completion response."""
    try:
        return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return "(no content in response)"


def format_usage(response: dict[str, Any]) -> str:
    """Format token usage as a compact summary string."""
    try:
        u = response["usage"]
        return (
            f"tokens: {u['prompt_tokens']} prompt + "
            f"{u['completion_tokens']} completion = "
            f"{u['total_tokens']} total"
        )
    except (KeyError, TypeError):
        return "tokens: unknown"


def wrap_column(text: str, width: int) -> list[str]:
    """Wrap text into lines of at most *width* characters."""
    lines: list[str] = []
    for paragraph in text.splitlines():
        if paragraph:
            lines.extend(textwrap.wrap(paragraph, width))
        else:
            lines.append("")
    return lines or [""]


def format_single(prompt: str, response: dict[str, Any], model: str) -> str:
    """Format a single-model response for terminal display."""
    text = extract_text(response)
    usage = format_usage(response)
    latency = response.get("_latency_ms")
    lat_str = f"  latency: {latency:.0f}ms" if latency is not None else ""
    lines = [
        f"┌─ Model: {model}",
        f"│  Prompt: {prompt}",
        "│",
    ]
    for line in text.splitlines():
        lines.append(f"│  {line}")
    lines += [
        "│",
        f"│  {usage}{lat_str}",
        "└" + "─" * 60,
    ]
    return "\n".join(lines)


def format_compare(
    prompt: str,
    model_a: str,
    response_a: dict[str, Any],
    model_b: str,
    response_b: dict[str, Any],
) -> str:
    """Format two responses side-by-side for comparison."""
    w = COMPARE_COL_WIDTH
    text_a = extract_text(response_a)
    text_b = extract_text(response_b)
    usage_a = format_usage(response_a)
    usage_b = format_usage(response_b)

    lat_a = response_a.get("_latency_ms")
    lat_b = response_b.get("_latency_ms")
    lat_str_a = f"{lat_a:.0f}ms" if lat_a is not None else "?"
    lat_str_b = f"{lat_b:.0f}ms" if lat_b is not None else "?"

    header_a = f" {model_a} ({lat_str_a})"
    header_b = f" {model_b} ({lat_str_b})"

    lines_a = wrap_column(text_a, w - 2)
    lines_b = wrap_column(text_b, w - 2)
    max_rows = max(len(lines_a), len(lines_b))
    lines_a += [""] * (max_rows - len(lines_a))
    lines_b += [""] * (max_rows - len(lines_b))

    out: list[str] = [
        "",
        f"Prompt: {prompt}",
        SEPARATOR,
        f"{header_a:<{w}} │ {header_b}",
        SEPARATOR,
    ]
    for la, lb in zip(lines_a, lines_b):
        out.append(f"{la:<{w}} │ {lb}")
    out += [
        SEPARATOR,
        f"{usage_a:<{w}} │ {usage_b}",
        SEPARATOR,
        "",
    ]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

@dataclass
class Prompt:
    text: str
    system: str | None = None
    label: str | None = None


def load_prompts(path: str) -> list[Prompt]:
    """Load prompts from a JSON file.

    Expected format:
        [{"text": "...", "system": "...", "label": "..."}, ...]
    or a simple list of strings:
        ["...", "..."]
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    prompts: list[Prompt] = []
    for item in data:
        if isinstance(item, str):
            prompts.append(Prompt(text=item))
        else:
            prompts.append(
                Prompt(
                    text=item["text"],
                    system=item.get("system"),
                    label=item.get("label"),
                )
            )
    return prompts


# ---------------------------------------------------------------------------
# Compare preflight (single-model-per-server constraint)
# ---------------------------------------------------------------------------

def preflight_compare(
    model_a: str,
    model_b: str,
    base_url_a: str,
    base_url_b: str,
) -> str | None:
    """Return an error message if compare cannot proceed, else None.

    InferenceX loads one model per server process (single GPU). Comparing two
    different models on the same base URL always fails for the non-loaded model.
    """
    if base_url_a != base_url_b:
        return None  # dual-server setup — user configured separate endpoints

    if model_a == model_b:
        return None  # same model twice is valid (e.g. temperature A/B later)

    # Probe the second model on the shared server — expect a loaded-model mismatch.
    try:
        chat_completion("ping", model=model_b, base_url=base_url_b, max_tokens=1)
        return None  # unexpected success — allow compare to proceed
    except RuntimeError as exc:
        msg = str(exc)
        if "loaded model is" not in msg:
            return msg
        m = _LOADED_MODEL_RE.search(msg)
        loaded = m.group(1) if m else "unknown"
        return (
            f"Cannot compare {model_a!r} and {model_b!r} on a single server.\n"
            f"  Server at {base_url_b} has {loaded!r} loaded (one model per GPU).\n\n"
            "Option 1 — single GPU (restart server between models):\n"
            f"  python playground/client.py --compare {model_a} {model_b} --sequential \\\n"
            '    "your prompt"\n\n'
            "Option 2 — two server instances (if GPU memory allows):\n"
            f"  Terminal 1: INFERENCE_X_DEFAULT_MODEL={model_a} ./scripts/dev.sh serve\n"
            f"  Terminal 2: INFERENCE_X_DEFAULT_MODEL={model_b} "
            "uv run uvicorn inference_x.api.main:app --host 0.0.0.0 --port 8001\n"
            f"  python playground/client.py --compare {model_a} {model_b} \\\n"
            "    --base-url-a http://localhost:8000 --base-url-b http://localhost:8001 \\\n"
            '    "your prompt"'
        )


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------

def run_single(
    prompt: Prompt,
    model: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
) -> None:
    label = prompt.label or prompt.text[:60]
    print(f"\nRunning: {label}")
    t0 = time.perf_counter()
    try:
        resp = chat_completion(
            prompt.text,
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            system=prompt.system,
        )
        resp["_latency_ms"] = (time.perf_counter() - t0) * 1000
        print(format_single(prompt.text, resp, model))
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)


def fetch_completion(
    prompt: Prompt,
    model: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    """Run one chat completion and return response with _latency_ms attached."""
    t0 = time.perf_counter()
    resp = chat_completion(
        prompt.text,
        model=model,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        system=prompt.system,
    )
    resp["_latency_ms"] = (time.perf_counter() - t0) * 1000
    return resp


def wait_for_loaded_model(
    base_url: str,
    expected_model: str,
    timeout: float = 600,
    poll_interval: float = 3.0,
) -> bool:
    """Poll until *expected_model* is the loaded model on *base_url*."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if not health_check(base_url):
            time.sleep(poll_interval)
            continue
        try:
            chat_completion("ping", model=expected_model, base_url=base_url, max_tokens=1)
            return True
        except RuntimeError as exc:
            if "loaded model is" in str(exc):
                time.sleep(poll_interval)
                continue
            raise
    return False


def run_compare_sequential(
    prompts: list[Prompt],
    model_a: str,
    model_b: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
) -> int:
    """Compare two models on one GPU by running A first, then B after server restart."""
    print(f"\n=== Sequential compare: {model_a} vs {model_b} ===")
    print(f"Server: {base_url}\n")

    # Phase 1 — model A (must be loaded now)
    print(f"Phase 1: querying {model_a!r} ({len(prompts)} prompt(s))...")
    responses_a: list[dict[str, Any]] = []
    for prompt in prompts:
        label = prompt.label or prompt.text[:60]
        try:
            responses_a.append(
                fetch_completion(prompt, model_a, base_url, temperature, max_tokens)
            )
            print(f"  ✓ {label}")
        except RuntimeError as exc:
            print(f"  ✗ {label}: {exc}", file=sys.stderr)
            return 1

    # Phase 2 — wait for model B
    print(
        f"\n>>> Stop the server and restart with:\n"
        f"    INFERENCE_X_DEFAULT_MODEL={model_b} ./scripts/dev.sh serve\n"
        f">>> Waiting for {model_b!r} on {base_url} (up to 10 min)..."
    )
    if not wait_for_loaded_model(base_url, model_b):
        print(
            f"ERROR: Timed out waiting for {model_b!r} on {base_url}.",
            file=sys.stderr,
        )
        return 1
    print(f"  ✓ {model_b!r} is ready\n")

    # Phase 3 — model B + display
    print(f"Phase 2: querying {model_b!r} and printing comparisons...")
    for prompt, resp_a in zip(prompts, responses_a):
        label = prompt.label or prompt.text[:60]
        try:
            resp_b = fetch_completion(
                prompt, model_b, base_url, temperature, max_tokens
            )
            print(f"\n--- {label} ---")
            print(format_compare(prompt.text, model_a, resp_a, model_b, resp_b))
        except RuntimeError as exc:
            print(f"ERROR on {label}: {exc}", file=sys.stderr)
            return 1

    return 0


def run_compare(
    prompt: Prompt,
    model_a: str,
    model_b: str,
    base_url_a: str,
    base_url_b: str,
    temperature: float,
    max_tokens: int,
) -> None:
    label = prompt.label or prompt.text[:60]
    print(f"\nComparing: {label}")
    try:
        resp_a = fetch_completion(
            prompt, model_a, base_url_a, temperature, max_tokens
        )
        resp_b = fetch_completion(
            prompt, model_b, base_url_b, temperature, max_tokens
        )
        print(format_compare(prompt.text, model_a, resp_a, model_b, resp_b))
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="client",
        description="InferenceX playground client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "prompt",
        nargs="?",
        help="Prompt text. Omit to use --prompts-file.",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        metavar="NAME",
        help=f"Model name to query (default: {DEFAULT_MODEL})",
    )
    p.add_argument(
        "--compare",
        nargs=2,
        metavar=("MODEL_A", "MODEL_B"),
        help=(
            "Compare two models side-by-side. Requires two server instances "
            "when models differ (one model per GPU). Use --base-url-a / --base-url-b."
        ),
    )
    p.add_argument(
        "--prompts-file",
        metavar="PATH",
        help="JSON file of prompts to run (overrides inline prompt arg).",
    )
    p.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        metavar="URL",
        help=f"Server base URL (default: {DEFAULT_BASE_URL})",
    )
    p.add_argument(
        "--base-url-a",
        default=None,
        metavar="URL",
        help="Server URL for MODEL_A in --compare mode (default: --base-url).",
    )
    p.add_argument(
        "--base-url-b",
        default=None,
        metavar="URL",
        help="Server URL for MODEL_B in --compare mode (default: --base-url).",
    )
    p.add_argument(
        "--sequential",
        action="store_true",
        help=(
            "With --compare: run model A first, wait for server restart with model B, "
            "then compare. Use on single-GPU setups."
        ),
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        metavar="FLOAT",
        help=f"Sampling temperature 0–2 (default: {DEFAULT_TEMPERATURE})",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        metavar="N",
        help=f"Max tokens to generate (default: {DEFAULT_MAX_TOKENS})",
    )
    p.add_argument(
        "--list-models",
        action="store_true",
        help="List available models from the server and exit.",
    )
    p.add_argument(
        "--health",
        action="store_true",
        help="Check server health and exit.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    base_url = args.base_url.rstrip("/")

    if args.health:
        ok = health_check(base_url)
        print("healthy" if ok else "unreachable")
        return 0 if ok else 1

    if args.list_models:
        models = list_models(base_url)
        if models:
            print("Available models:")
            for m in models:
                print(f"  {m}")
        else:
            print("Could not fetch model list. Is the server running?")
        return 0

    # Collect prompts
    if args.prompts_file:
        try:
            prompts = load_prompts(args.prompts_file)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            print(f"ERROR loading prompts file: {exc}", file=sys.stderr)
            return 1
    elif args.prompt:
        prompts = [Prompt(text=args.prompt)]
    else:
        parser.print_help()
        return 1

    compare_models = args.compare  # [model_a, model_b] or None

    if compare_models and args.sequential:
        return run_compare_sequential(
            prompts,
            model_a=compare_models[0],
            model_b=compare_models[1],
            base_url=base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

    if compare_models:
        base_url_a = (args.base_url_a or base_url).rstrip("/")
        base_url_b = (args.base_url_b or base_url).rstrip("/")
        err = preflight_compare(compare_models[0], compare_models[1], base_url_a, base_url_b)
        if err:
            print(f"ERROR: {err}", file=sys.stderr)
            return 1

    for prompt in prompts:
        if compare_models:
            run_compare(
                prompt,
                model_a=compare_models[0],
                model_b=compare_models[1],
                base_url_a=base_url_a,
                base_url_b=base_url_b,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        else:
            run_single(
                prompt,
                model=args.model,
                base_url=base_url,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
