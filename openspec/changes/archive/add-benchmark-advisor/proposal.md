# Proposal: add-benchmark-advisor

## Summary
Add a benchmark runner, hardware profiler, and model advisor so InferenceX can measure
model performance on the operator's hardware and recommend which model to use.

## Why
Users with constrained hardware (low VRAM, no GPU) need guidance on which models will
actually run on their system before committing to a download. Running every model manually
and reading logs to extract tokens/sec is not acceptable UX. InferenceX should do this
automatically and present a ranked recommendation with reasoning.

## In scope
- Benchmark runner: measures throughput, TTFT, p50/p95/p99 latency, VRAM usage
- Hardware profiler: detects GPU, VRAM, CPU, RAM at runtime
- Model advisor: scores and ranks models from benchmark results against hardware profile
- CLI surface: `make benchmark`, `make benchmark-all`, `make advise`
- API surface: GET /v1/benchmark/results, GET /v1/benchmark/advise
- Playground benchmark tab: hardware info, throughput chart, ranked recommendation
- Benchmark prompt suite: fixed 10-prompt set for reproducible comparisons

## Out of scope
- Quality benchmarks (MMLU, HumanEval, etc.)
- Automated model downloads
- Cloud API benchmarking
- Multi-GPU distributed benchmarking
- Fine-tuning or quantization tooling