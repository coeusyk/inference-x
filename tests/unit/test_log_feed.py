"""Unit tests for playground/log_feed.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "playground"))

import log_feed as lf  # noqa: E402


def test_format_log_line_structured_info():
    line = "2026-06-24 08:00:04 [INFO] inference_x.engines: Loading engine for model=tinyllama"
    assert lf.format_log_line(line) == "· Loading engine: tinyllama"


def test_format_log_line_structured_error():
    line = "2026-06-24 08:00:04 [ERROR] uvicorn.error: Application startup failed"
    assert lf.format_log_line(line) == "✗ Application startup failed"


def test_format_log_line_critical_startup_failure():
    line = (
        "2026-06-24 08:19:14 [CRITICAL] inference_x.api.main: "
        "Startup initialization failed: Insufficient GPU memory to start qwen2.5-0.5b."
    )
    formatted = lf.format_log_line(line)
    assert formatted is not None
    assert formatted.startswith("✗")
    assert "Insufficient GPU memory" in formatted


def test_format_log_line_skips_traceback_frames():
    assert lf.format_log_line('  File "/tmp/foo.py", line 1, in bar') is None
    assert lf.format_log_line('File "/tmp/foo.py", line 1, in bar') is None


def test_format_log_line_vllm_load_weights():
    line = "(EngineCore pid=1) INFO 06-24 08:10:40 [default_loader.py:397] Loading weights took 1.10 seconds"
    assert lf.format_log_line(line) == "· Weights loaded (1.10 seconds)"


def test_format_log_line_interesting_raw_line():
    line = "WARNING: We must use the `spawn` multiprocessing start method"
    assert lf.format_log_line(line) is not None
    assert lf.format_log_line(line).startswith("!")


def test_extract_error_summary_from_tail(tmp_path: Path):
    log = tmp_path / "server.log"
    log.write_text(
        "\n".join(
            [
                "2026-06-24 08:00:01 [INFO] inference_x: starting",
                "2026-06-24 08:00:04 [ERROR] uvicorn.error: Cannot fit model on GPU",
            ]
        ),
        encoding="utf-8",
    )
    summary = lf.extract_error_summary(log)
    assert "Cannot fit model on GPU" in summary


def test_extract_error_summary_vllm_kv_cache_negative(tmp_path: Path):
    log = tmp_path / "server.log"
    log.write_text(
        "\n".join(
            [
                "2026-06-24 08:00:01 [INFO] inference_x: starting",
                "(EngineCore pid=1) ERROR 06-24 08:10:40 [gpu_worker.py:99] "
                "Available KV cache memory: -0.12 GiB",
            ]
        ),
        encoding="utf-8",
    )
    summary = lf.extract_error_summary(log)
    assert "No KV cache memory" in summary
    assert "see log lines above" not in summary


def test_extract_error_summary_insufficient_gpu_memory_line(tmp_path: Path):
    log = tmp_path / "server.log"
    log.write_text(
        "2026-06-24 08:19:14 [CRITICAL] inference_x.api.main: "
        "Startup initialization failed: Insufficient GPU memory to start qwen2.5-0.5b.\n",
        encoding="utf-8",
    )
    summary = lf.extract_error_summary(log)
    assert "Insufficient GPU memory" in summary


def test_extract_error_summary_fallback_message(tmp_path: Path):
    log = tmp_path / "server.log"
    log.write_text("2026-06-24 08:00:01 [INFO] inference_x: still starting\n", encoding="utf-8")
    summary = lf.extract_error_summary(log)
    assert summary == "Startup failed — see logs/playground-server.log for details"


def test_extract_error_summary_does_not_cut_off_mid_sentence(tmp_path: Path):
    """Regression for a real make playground failure (2026-07-03): a
    validate_pool_fits ValueError long enough to exceed the old hard [:200]
    slice got cut off mid-sentence ("...capped at" with nothing after),
    hiding the actionable "Load fewer models..." clause at the end."""
    long_message = (
        "Models [qwen2.5-0.5b, qwen2.5-1.5b] cannot load sequentially on a 8 GiB GPU: "
        "Model qwen2.5-0.5b cannot fit in the remaining GPU memory for this pool. "
        "Need gpu_memory_utilization >= 0.116 but capped at -0.032. Load fewer models "
        "or use a GPU with more VRAM."
    )
    log = tmp_path / "server.log"
    log.write_text(
        f"2026-07-03 08:00:04 [CRITICAL] inference_x.api.main: "
        f"Startup initialization failed: {long_message}\n",
        encoding="utf-8",
    )
    summary = lf.extract_error_summary(log)
    assert summary.endswith("Load fewer models or use a GPU with more VRAM.")
    assert not summary.endswith("capped at")


def test_extract_error_summary_timeout_on_loading_weights(tmp_path: Path):
    log = tmp_path / "server.log"
    log.write_text(
        "\n".join(
            [
                "2026-06-24 08:00:01 [INFO] inference_x: starting",
                "2026-06-24 08:00:04 [INFO] inference_x.engines: Loading weights for model=tiny",
            ]
        ),
        encoding="utf-8",
    )
    summary = lf.extract_error_summary(log)
    assert "timed out" in summary.lower()


def test_prepare_log_session_overwrites(tmp_path: Path):
    log = tmp_path / "server.log"
    log.write_text("stale content\n", encoding="utf-8")
    lf.prepare_log_session(log)
    text = log.read_text(encoding="utf-8")
    assert "stale" not in text
    assert lf._SESSION_MARKER in text


def test_startup_failed_in_log(tmp_path: Path):
    log = tmp_path / "server.log"
    log.write_text("2026-06-24 08:19:14 [CRITICAL] inference_x.api.main: Startup initialization failed\n")
    assert lf.startup_failed_in_log(log)


@pytest.mark.asyncio
async def test_log_tailer_emits_formatted_lines(tmp_path: Path):
    log = tmp_path / "server.log"
    lf.prepare_log_session(log)

    seen: list[str] = []

    tailer = lf.LogTailer(seen.append, log_path=log, poll_interval_s=0.05)
    tailer.start(from_offset=0)

    with log.open("a", encoding="utf-8") as fh:
        fh.write("2026-06-24 08:00:04 [INFO] inference_x.api.deps: Engine pool ready: ['m']\n")

    for _ in range(30):
        if seen:
            break
        await __import__("asyncio").sleep(0.05)

    await tailer.stop()
    assert any("Engine pool ready" in line for line in seen)
