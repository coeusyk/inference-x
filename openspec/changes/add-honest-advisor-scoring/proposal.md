# OS-6 — Honest advisor scoring and device-occupancy VRAM

## Why

The advisor still embeds a constant `quant_score = 1.0` at 10% weight. That
constant is an accidental floor: a viable-but-worst model scores at least `10.0`,
while a VRAM-gated model scores `0.0`. Consumers can mistake `score > 0` for
viability. The remaining weights are presented as reasoned without being
calibrated, and the persisted VRAM field name (`peak_vram_delta_gb`) describes
neither a peak delta nor a per-model footprint — the measurement is whole-device
occupancy (`total − min(free)`).

This change makes the score honest (measured quantities only), freezes
Compatibility Invariants already agreed for Phase A, and renames the VRAM metric
to match its actual referent — with aliases through Phase A. It does not migrate
history, redesign normalization, or calibrate weights.

## What Changes

- Delete `quant_score`. Advisor score uses exact weights:
  - throughput = `4/9`
  - TTFT = `1/3`
  - VRAM = `2/9`
- Freeze: `viable` is the sole viability signal; score is a within-report ordinal
  used only to rank viable models produced from the same benchmark suite
  (DEC-056). Score is NOT portable across benchmark suites, NOT portable across
  hardware, NOT portable across future weighting changes, and NOT an absolute
  quality metric.
- Rename the persisted/canonical benchmark VRAM field to
  `vram_device_occupied_gib` (DEC-057). Measurement remains
  `total − min(free)` (device occupancy).
- Keep compatibility aliases through Phase A:
  - `peak_vram_delta_gb` → `vram_device_occupied_gib` on stored results
  - `vram_gb` → `vram_device_occupied_gib` on advisor response
- Leave `HardwareProfile.vram_total_gb` / `vram_free_gb` **unchanged**.
- Canonical fields always take precedence over deprecated aliases during
  deserialization. If both are present with conflicting values, the canonical
  field wins (permanent unless superseded by a future ADR).
- Record **DEC-056** and **DEC-057** only. No additional ADRs.

**Benchmark compatibility (corrected):**

- Historical benchmark JSON remains readable without migration.
- Historical benchmark JSON is never rewritten.
- New benchmark JSON uses the canonical field `vram_device_occupied_gib`.
- Deprecated aliases remain readable throughout Phase A.

## Capabilities

### New Capabilities

- _(none — honesty and metric naming extend the existing `platform` capability)_

### Modified Capabilities

- `platform`: Advisor scoring uses measured quantities only with exact frozen
  weights; Compatibility Invariants for score/viable/weights/aliases; canonical
  VRAM field `vram_device_occupied_gib` with Phase-A aliases; HardwareProfile
  names unchanged.

## Impact

| Area | Ownership | Impact |
|---|---|---|
| `src/inference_x/benchmarks/advisor.py` | **OS-6-owned** | Delete `quant_score`; exact weights; read canonical VRAM; expose alias on wire |
| `src/inference_x/benchmarks/schemas.py` | **OS-6-owned** | Canonical field + aliases; canonical-wins precedence |
| `src/inference_x/benchmarks/runner.py` | **Shared with OS-5** | Construct/write `vram_device_occupied_gib` (measurement unchanged). **Conflict: High** — rebase on OS-5 |
| `tests/unit/test_advisor.py` | **OS-6-owned** | Floor collapse, ordinal, ranking invariance, alias reads |
| `tests/unit/test_benchmark_runner.py` | **Shared** | VRAM construction / field name at write site |
| `tests/unit/test_storage.py` | **Shared with OS-5** | Alias corpus against stored results. **Conflict: Medium** — rebase on OS-5 |
| `docs/DECISIONS.md` | **Shared (append-only)** | DEC-056, DEC-057 only |
| `benchmarks/results/*.json` | **Read-only** | Alias test corpus; never rewrite |
| `src/inference_x/benchmarks/storage.py` | **Out of bounds** | OS-5 owns suite filtering — do not change |
| `src/inference_x/benchmarks/suite_identity.py` | **Out of bounds** | OS-5 owns suite identity — do not change |
| `HardwareProfile` / `hardware.py` | **Out of bounds** | Field names unchanged by freeze |
| Chat / streaming / admission | **None** | |

### ADRs

- **DEC-056** — Advisor score reflects measured quantities only *(this change)*.
- **DEC-057** — The VRAM number is device occupancy, not model footprint *(this change)*.
- Do **not** author any additional ADRs.

### Sequencing

Land **after OS-5**. Shared-file conflicts: `runner.py`, `test_storage.py`,
`DECISIONS.md`. Do not reopen suite identity or storage filtering.

## Non-Goals

Explicitly forbidden:

- Phase B work
- Phase C work
- Suite identity changes
- Storage filtering changes
- Benchmark migration / rewriting historical JSON
- Normalization redesign
- Calibration procedure or weight tuning beyond the frozen exact fractions
- Component-score API on `AdvisorResult`
- TTFT redesign (`_warm_ttft_ms` unchanged)
- Changing `_check_vram_budget`'s comparison logic (known coupling recorded, not fixed)
- Renaming `HardwareProfile.vram_*_gb`
- Removing aliases without a future ADR
