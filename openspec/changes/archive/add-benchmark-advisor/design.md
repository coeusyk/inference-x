# Design: add-benchmark-advisor

## Overview
Add benchmarking and hardware-aware model recommendation as a self-contained module
under `src/inference_x/benchmarks/`. The module integrates with the existing server
via new read-only API routes and with the playground via a new Benchmark tab.

## Module structure

```
src/inference_x/benchmarks/
├── __init__.py
├── hardware.py        # HardwareProfile detection (pynvml + psutil)
├── runner.py          # BenchmarkRunner — runs prompt suite, records metrics
├── schemas.py         # BenchmarkResult, HardwareProfile, AdvisorResult dataclasses
├── advisor.py         # ModelAdvisor — scores and ranks models from results
└── storage.py         # ResultStore — reads/writes JSON from docs/benchmarks/

src/inference_x/api/routes/
└── benchmark.py       # GET /v1/benchmark/results, GET /v1/benchmark/advise

benchmarks/
└── prompts/
    └── standard.json  # Fixed 10-prompt suite, versioned

scripts/
└── benchmark.py       # CLI entry point for make benchmark / make benchmark-all

playground/
└── benchmark_tab.py   # Textual BenchmarkTab widget
```

## Scoring function (advisor.py)

The advisor ranks models using a weighted score across four dimensions:

| Dimension | Weight | Rationale |
|---|---|---|
| Throughput (tok/s) | 40% | Most visible performance signal |
| TTFT (ms, inverted) | 30% | Latency matters for interactive use |
| VRAM headroom | 20% | Model must fit in available VRAM |
| Quantization fit | 10% | INT8/FP8 preferred on low-VRAM hardware |

VRAM headroom check is a hard gate: any model requiring more VRAM than is free
gets score 0 and is marked "not viable" regardless of other metrics.

## Hardware profiler fallback chain

1. `pynvml` Python bindings — fastest, most reliable on CUDA systems
2. `nvidia-smi --query-gpu` subprocess — fallback if pynvml unavailable
3. CPU-only profile — if no GPU detected, VRAM-dependent models all marked not viable

## API routes (additive, no contract changes)

```
GET /v1/benchmark/results
  Response: { results: BenchmarkResult[], hardware: HardwareProfile }

GET /v1/benchmark/advise
  Response: { ranked: AdvisorResult[], hardware: HardwareProfile, generated_at: str }
```

## Validation
- Unit tests for advisor scoring with fixture hardware profiles (6GB, 10GB, 24GB VRAM)
- Unit tests for ResultStore read/write
- Unit tests for hardware profiler (mock pynvml)
- Integration: `make benchmark MODEL=opt-125m` produces valid result JSON
  (opt-125m used in CI because it is the smallest available model)
