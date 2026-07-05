"""ResultStore: saves and loads BenchmarkResult JSON files from disk."""
from __future__ import annotations

import json
import re
from pathlib import Path

from inference_x.benchmarks.schemas import BenchmarkResult

DEFAULT_RESULTS_DIR = "benchmarks/results"


class ResultStore:
    """Reads and writes BenchmarkResult JSON to/from a directory."""

    def save(
        self,
        result: BenchmarkResult,
        output_dir: str = DEFAULT_RESULTS_DIR,
    ) -> Path:
        """Serialize *result* to a timestamped JSON file and return its path."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Sanitize model name for use in filename
        safe_model = re.sub(r"[^\w.\-]", "_", result.model_name)
        ts = result.timestamp.replace(":", "-").replace("+", "").replace(".", "-")
        filename = f"results-{safe_model}-{ts}.json"
        path = out / filename
        path.write_text(
            json.dumps(result.model_dump(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    def all_results(self, output_dir: str = DEFAULT_RESULTS_DIR) -> list[BenchmarkResult]:
        """Return all BenchmarkResult objects found in *output_dir*."""
        out = Path(output_dir)
        if not out.exists():
            return []
        results = []
        for p in sorted(out.glob("results-*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                results.append(BenchmarkResult.model_validate(data))
            except Exception:
                continue
        return results

    def latest_per_model(
        self, output_dir: str = DEFAULT_RESULTS_DIR
    ) -> dict[str, BenchmarkResult]:
        """Return the most recent BenchmarkResult for each model name."""
        all_r = self.all_results(output_dir)
        latest: dict[str, BenchmarkResult] = {}
        for result in all_r:
            existing = latest.get(result.model_name)
            if existing is None or result.timestamp > existing.timestamp:
                latest[result.model_name] = result
        return latest
