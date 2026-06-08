# Tasks: add-benchmark-advisor

## 1. Schemas and data model
- 1.1 Add `src/inference_x/benchmarks/schemas.py`
  - `HardwareProfile`: gpu_name, vram_total_gb, vram_free_gb, cpu_cores, ram_total_gb
  - `BenchmarkResult`: model_name, hardware, prompt_suite_id, metrics dict, timestamp
  - `AdvisorResult`: model_name, score, rank, throughput_toks, ttft_ms, vram_gb,
    viable, recommendation_str
- 1.2 Validate all schemas with Pydantic

## 2. Hardware profiler
- 2.1 Add `src/inference_x/benchmarks/hardware.py`
  - Try pynvml first; fall back to nvidia-smi subprocess; fall back to CPU-only
  - Return `HardwareProfile`
- 2.2 Unit tests with mocked pynvml (fixture profiles: 6GB, 10GB, 24GB, CPU-only)
- 2.3 Add `pynvml` and `psutil` to pyproject.toml dev/optional deps

## 3. Benchmark prompt suite
- 3.1 Create `benchmarks/prompts/standard.json` with 10 prompts:
  short factual (×3), long generation (×2), code (×2), reasoning (×2), chat (×1)
- 3.2 Add `prompt_suite_id` field — SHA256 of suite file for versioning

## 4. Benchmark runner
- 4.1 Add `src/inference_x/benchmarks/runner.py`
  - `BenchmarkRunner.run(model_name, prompt_suite_path) -> BenchmarkResult`
  - Measures: tokens/sec, TTFT, p50/p95/p99 latency, peak VRAM (via hardware.py)
  - Runs concurrency=1 by default; concurrency=4/8 as optional flag
- 4.2 Add `scripts/benchmark.py` as CLI entry point
- 4.3 Add Makefile targets: `make benchmark MODEL=<name>`, `make benchmark-all`

## 5. Result store
- 5.1 Add `src/inference_x/benchmarks/storage.py`
  - `ResultStore.save(result)` → writes to `docs/benchmarks/results-{model}-{ts}.json`
  - `ResultStore.latest_per_model()` → returns most recent result per model name
  - `ResultStore.all()` → returns all stored results
- 5.2 Unit tests for save/load round-trip

## 6. Model advisor
- 6.1 Add `src/inference_x/benchmarks/advisor.py`
  - `ModelAdvisor.rank(hardware, results) -> list[AdvisorResult]`
  - Hard VRAM gate: score=0 and viable=False if model VRAM > hardware.vram_free_gb
  - Weighted scoring: throughput 40%, TTFT 30%, VRAM headroom 20%, quantization 10%
  - Plain-language recommendation string generated per model
- 6.2 Add Makefile target: `make advise`
- 6.3 Unit tests: fixture results + 6/10/24GB VRAM hardware → assert rank order

## 7. API routes
- 7.1 Add `src/inference_x/api/routes/benchmark.py`
  - `GET /v1/benchmark/results` — returns ResultStore.all() + current hardware profile
  - `GET /v1/benchmark/advise` — returns advisor output + hardware profile
- 7.2 Register routes in `src/inference_x/api/main.py` under prefix `/v1/benchmark`
- 7.3 Unit tests for both routes (mock ResultStore)

## 8. Playground benchmark tab
- 8.1 Add `playground/benchmark_tab.py` — Textual `BenchmarkTab` widget
  - Hardware info panel: GPU name, VRAM free/total, CPU, RAM
  - Per-model throughput bar (scaled to fastest model = 100%)
  - Advisor ranked list with recommendation strings
  - "Run benchmark" button: triggers `scripts/benchmark.py` for selected model
    as a subprocess, updates display when complete
- 8.2 Wire `BenchmarkTab` into `playground/app.py` as a new tab

## 9. Tests and validation
- 9.1 All Phase 1–5 tests must still pass
- 9.2 `make benchmark MODEL=opt-125m` produces valid result JSON on WSL2 + CUDA
- 9.3 `make advise` prints ranked output to terminal
- 9.4 Article notes updated with benchmark methodology, sample output, hardware used

## 10. Docs and decisions
- 10.1 Add DEC-027: Benchmark scoring function design and weight rationale
- 10.2 Add DEC-028: Hardware profiler fallback chain (pynvml → nvidia-smi → CPU-only)
- 10.3 Update ARTICLENOTES.md with benchmark findings for the article
- 10.4 Archive change to `openspec/changes/archive/` when complete