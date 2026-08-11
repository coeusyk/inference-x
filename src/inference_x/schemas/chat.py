from __future__ import annotations

import time
import uuid
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., max_length=32_000)


class StreamOptions(BaseModel):
    """OpenAI-compatible `stream_options` object (DEC-049).

    Only `include_usage` is supported. An absent `stream_options` is equivalent
    to `include_usage=False`.
    """

    include_usage: bool = Field(
        default=False,
        description="When true, the stream emits a final chunk with an empty "
        "choices array carrying engine-accounted usage, before data: [DONE].",
    )


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(..., min_length=1, max_length=50)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=512, ge=1, le=4096)
    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0)
    stream: bool = False
    max_context_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Prompt-token ceiling enforced pre-dispatch by AdmissionController; "
        "None means the model's configured max_model_len is the only ceiling.",
    )
    max_output_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        le=4096,
        description="Preferred alias for max_tokens; takes precedence over max_tokens "
        "when both are set. Kept max_tokens for OpenAI-client compatibility.",
    )
    priority: Literal["interactive", "batch"] = Field(
        default="interactive",
        description="'interactive' requests get output clamped to fit available context/"
        "KV budget when possible; 'batch' requests are rejected (429) instead of clamped "
        "when the engine is saturated. Client-supplied and unauthenticated — do not treat "
        "as a trust boundary.",
    )
    stream_options: Optional[StreamOptions] = Field(
        default=None,
        description="OpenAI-compatible streaming options. Absent means "
        "include_usage=False — no usage chunk is emitted.",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Optional sampling seed forwarded unchanged to the live "
        "backend sampler when set (DEC-051). Absent/null omits seed from "
        "SamplingParams. Does not claim end-to-end determinism.",
    )
    strict: bool = Field(
        default=False,
        description="When true, a request the server would have executed with a "
        "substituted parameter is rejected with 400 instead (DEC-052). Strict "
        "only converts substitution into rejection — it never changes the "
        "substitution itself, and it never rejects on degraded warnings, which "
        "would turn admission fail-closed (DEC-047 §4).",
    )
    deterministic: bool = Field(
        default=False,
        description="When true, serve under vLLM batch-invariant mode "
        "(Phase C C3) — but only if the process was already started with "
        "determinism enabled (INFERENCE_X_DETERMINISTIC=1, requires SM ≥ "
        "8.0). Otherwise returns 400 rather than running non-deterministically "
        "or activating it too late to matter. Distinct from strict (DEC-052).",
    )


class ResponseWarning(BaseModel):
    """One thing the server did that the client did not ask for (OS-4).

    `type` is the axis a client branches on, and it has exactly two members:

    - `substituted` — the server ran something different from what was asked.
      This is the set `strict` rejects on (DEC-052).
    - `degraded` — the server could not verify something and proceeded anyway.
      This never rejects, under either `strict` value: rejecting here would
      convert admission's fail-open posture into fail-closed (DEC-047 §4).

    `code` is the stable machine identifier — what `strict` keys on and what
    tests assert. The closed set is registered in DEC-053. `message` is human
    text and the only field free to change without a spec change. `field` names
    the affected request field, or None for a gate that maps to no single field.
    """

    type: Literal["substituted", "degraded"]
    code: str
    message: str
    field: Optional[str] = None


class ResolvedRequest(BaseModel):
    """The request the server actually executed — the Effective Request (OS-4).

    Membership follows a derivability rule rather than curation: a field appears
    here **iff** it exists on `ChatCompletionRequest`, minus `messages` (content,
    not a parameter) and minus the transport/policy controls `stream`,
    `stream_options`, `strict`, and `deterministic`. `tests/unit/test_schemas.py`
    enforces the rule, so a field added to the request later forces an explicit
    decision here rather than silently omitting itself.

    It is a full echo rather than a diff: a diff needs the original to be
    interpretable, and only a self-contained echo can be re-submitted as-is with
    the caller's own messages.

    Load-time engine and tier settings do not appear here. They are process-level
    state, and a per-request block that reports them conflates two lifetimes. A
    tier cap that *caused* a clamp surfaces as that clamp's warning instead.
    """

    model: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    max_context_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    priority: Literal["interactive", "batch"] = "interactive"
    seed: Optional[int] = None


class ChatCompletionMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: Literal["stop", "length", "error"]


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class EngineTiming(BaseModel):
    """Per-request engine-internal timing (Phase B3).

    Derived exclusively from the engine's own process-clock timestamps —
    never from a frontend wall-clock or HTTP-boundary timestamp (see
    `expose-per-request-engine-timing` design.md, "clock domain"). This is
    deliberately distinct from `GET /v1/metrics`'s HTTP-boundary latency/TTFT
    and from `GET /metrics`'s process-wide Prometheus series: this block
    answers "how was *this* request served," not "how is the engine doing."

    `inference_time_ms` is `prefill_time_ms + decode_time_ms`, reported
    directly (mirroring vLLM's own internal tracing) rather than requiring
    the client to re-derive it.
    """

    queue_time_ms: float
    prefill_time_ms: float
    decode_time_ms: float
    inference_time_ms: float


class ManifestEngine(BaseModel):
    """`engine` block of the run manifest (Phase C, C1 — add-run-manifest, §8.3).

    Identifies the InferenceX software itself, not the model. `version` and
    `backend_version` come from installed package metadata; `git_sha` from
    the checkout or `INFERENCE_X_GIT_SHA`. Any field the platform cannot
    determine is `None` — never fabricated.
    """

    name: str = "inference-x"
    version: Optional[str] = None
    git_sha: Optional[str] = None
    backend: str = "vllm"
    backend_version: Optional[str] = None


class ManifestModel(BaseModel):
    """`model` block of the run manifest.

    `hf_revision` and `weights_sha256` are always `None` in this capability
    — their completeness/degradation policy is explicitly open
    (add-run-manifest design.md D4) — present in the schema, never
    fabricated, deferred rather than guessed.
    """

    registry_name: str
    hf_repo: Optional[str] = None
    hf_revision: Optional[str] = None
    weights_sha256: Optional[str] = None
    quantization: Optional[str] = None
    dtype: Optional[str] = None


class ManifestRuntime(BaseModel):
    """`runtime` block of the run manifest.

    `batch_invariant` reports whatever `VLLM_BATCH_INVARIANT` is set to
    right now — honestly. C3 (`deterministic: true` / startup
    `INFERENCE_X_DETERMINISTIC`) is what sets the env on supported GPUs;
    operators may also set it directly. `speculative` is always `None`: no
    speculative-decoding feature exists yet (Phase D3).
    """

    attention_backend: Optional[str] = None
    cuda_graphs: Optional[bool] = None
    enforce_eager: Optional[bool] = None
    kv_cache_dtype: Optional[str] = None
    block_size: Optional[int] = None
    max_model_len: Optional[int] = None
    kv_capacity_tokens: Optional[int] = None
    prefix_caching: Optional[bool] = None
    prefix_cache_hash_algo: Optional[str] = None
    batch_invariant: bool = False
    speculative: Optional[str] = None


class ManifestSampling(BaseModel):
    """`sampling` block of the run manifest.

    `seed` is the requested seed only (the same value `ResolvedRequest.seed`
    carries) — never a backend-derived effective seed (the seed-echo
    amendment; see the MODIFIED "Optional request seed reaches the sampler"
    requirement). `max_tokens` is what the client asked for;
    `resolved_max_tokens` is what admission actually used.
    """

    temperature: Optional[float] = None
    top_p: Optional[float] = None
    seed: Optional[int] = None
    max_tokens: Optional[int] = None
    resolved_max_tokens: Optional[int] = None


class ManifestRequestInfo(BaseModel):
    """`request` block of the run manifest.

    `prompt_sha256` hashes the structured messages, not an engine-rendered
    prompt string (see `utils/ids.py::compute_prompt_sha256`) — this never
    crosses the Engine Boundary. `prompt_tokens` is read from the response's
    own engine-accounted `usage.prompt_tokens` (DEC-050), never recomputed.
    """

    prompt_sha256: Optional[str] = None
    prompt_tokens: Optional[int] = None
    chat_template_sha256: Optional[str] = None


class ManifestBatch(BaseModel):
    """`batch` block of the run manifest.

    `co_batched_request_ids` is reserved but never populated in this
    capability (C2 is deferred until C5 establishes the value justifies the
    vLLM-internals coupling needed to extract it) — always an empty list,
    never fabricated composition data.
    """

    cold_start: Optional[bool] = None
    prefix_cache_hit_tokens: Optional[int] = None
    co_batched_request_ids: list[str] = Field(default_factory=list)
    max_batch_size_observed: Optional[int] = None


class ManifestHardware(BaseModel):
    """`hardware` block of the run manifest: a request-time snapshot (§8.2)."""

    gpu: Optional[str] = None
    vram_total_gib: Optional[float] = None
    driver: Optional[str] = None
    cuda: Optional[str] = None
    cpu: Optional[str] = None
    ram_gib: Optional[float] = None
    wsl2: Optional[bool] = None


class RunManifest(BaseModel):
    """The run manifest (Phase C, C1 — add-run-manifest, §8.3).

    `run_id` is a content hash of the `engine`/`model`/`runtime`/`sampling`/
    `request` blocks plus each warning's `type`/`code`/`field` (add-run-
    manifest design.md D2) — `timing`, `batch`, and `hardware` are excluded
    entirely (speed/observational, and execution-environment rather than
    configuration identity, respectively), and a warning's free-text
    `message` never participates. Content-addressing only: no cryptographic
    signature is produced.

    Assembled by `ChatService`, never by an engine (DEC-047 — same rule as
    `resolved`/`warnings`), from signals the platform already produces.
    Asserts comparability (equal `run_id` implies the same recorded
    configuration), not reproducibility.
    """

    manifest_version: int = 1
    run_id: str
    engine: ManifestEngine
    model: ManifestModel
    runtime: ManifestRuntime
    sampling: ManifestSampling
    request: ManifestRequestInfo
    timing: Optional[EngineTiming] = None
    batch: ManifestBatch = Field(default_factory=ManifestBatch)
    hardware: ManifestHardware
    warnings: list[ResponseWarning] = Field(default_factory=list)


class ChatStreamChunk(BaseModel):
    """One event on the engine streaming channel (DEC-049).

    This is the `BaseEngine.generate_stream` element type — the Engine Boundary's
    streaming vocabulary. It deliberately lives in `schemas.chat` beside the wire
    models: DEC-047 forbids a backend-neutral execution package, and a second
    type system for the same information is exactly what that prohibits.

    Content events set `content` and leave `finish_reason`, `usage`, and
    `timing` None. The terminal event sets `finish_reason`, and sets `usage`
    and `timing` when the backend can account them — an engine that cannot
    supply per-request timing leaves `timing` None rather than estimating it
    (Phase B3, mirroring DEC-049's usage-absence rule).
    """

    content: str = ""
    finish_reason: Optional[Literal["stop", "length", "error"]] = None
    usage: Optional[ChatCompletionUsage] = None
    timing: Optional[EngineTiming] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage
    timing: Optional[EngineTiming] = Field(
        default=None,
        description="Per-request engine-internal timing (Phase B3): queue, "
        "prefill, decode, and total inference time, sourced from the "
        "engine's own process-clock timestamps. None when the backend "
        "cannot supply it — never estimated.",
    )
    resolved: Optional[ResolvedRequest] = Field(
        default=None,
        description="The request the server actually executed (OS-4). Attached by "
        "ChatService, never by an engine — the Engine Boundary does not learn "
        "about admission. Defaults to None so direct construction keeps working; "
        "on the service path it is always populated.",
    )
    warnings: list[ResponseWarning] = Field(
        default_factory=list,
        description="Every substitution the server made and every check it could "
        "not perform (OS-4). Empty, never absent, when nothing was substituted.",
    )
    run_id: Optional[str] = Field(
        default=None,
        description="Content-addressed identity of this request's run manifest "
        "(Phase C, C1). Also carried on the X-Run-Id response header for "
        "non-streaming requests. Attached by ChatService, never by an engine "
        "(DEC-047). Two run_ids are equal iff their manifests agree on "
        "engine/model/runtime/sampling/request and each warning's "
        "type/code/field (not hardware, and not a warning's free-text "
        "message) — a comparability guarantee, not a reproducibility one.",
    )
    manifest: Optional[RunManifest] = Field(
        default=None,
        description="The full run manifest whose run_id is carried above (Phase C, C5 — "
        "integrate-varex-manifest, resolving add-run-manifest design.md D5). Non-streaming "
        "only; null whenever run_id is null. A client can recompute run_id from this "
        "object's own engine/model/runtime/sampling/request/warnings fields to verify "
        "comparability mechanically rather than trusting the string.",
    )
