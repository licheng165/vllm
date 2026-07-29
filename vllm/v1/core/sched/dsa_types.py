# SPDX-License-Identifier: Apache-2.0
"""DSA (Dynamic Sparse Attention) request-level threshold routing types.

This module is the single authoritative source for the data structures that
implement the design described in
``03_GLM51_DSA提示词阈值分流详细设计.md``.

Design highlights captured here:

* ``DSARouteState`` -- the per-request NPU execution mode state machine
  (LEGACY / RESIDENT / PROMOTING / SPARSE / RECOVERING / FALLBACK_RESIDENT /
  FINISHED).
* ``RequestKey`` -- process-incarnation-aware request identity that defeats
  stale completion / finish / preemption / pin reuse across request ID reuse
  and scheduler restarts.
* ``DSATransferPlan`` -- the four orthogonal transfer obligations
  (must_import_prefix / must_export_prefill / must_persist_decode_windows /
  remote_handoff_required).  These are *not* a single role enum: a PD Decoder
  long request simultaneously imports the P prefix and persists its own decode
  windows.
* ``DSARouteSnapshot`` -- the immutable per-step route snapshot the Scheduler
  publishes to workers/connectors.  Workers/connectors must consume the
  snapshot and MUST NOT re-derive sparse mode from ``prompt_len``.
* Operation / receipt / bundle / command / event types for the
  Scheduler-issued command protocol with exact-set quorum aggregation.
* ``DSADataCompatibility`` / ``DSAInstanceCapabilities`` for the capability
  handshake and versioned v2 cache namespace.

All value types are ``@dataclass(frozen=True)`` so they can be safely shared
between the Scheduler thread and worker threads, used as dict keys and placed
into immutable snapshots.  ``DSARequestState`` is the single mutable,
request-scoped state object owned by the Scheduler.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, replace
from typing import Literal, Optional


# ---------------------------------------------------------------------------
# Route state machine
# ---------------------------------------------------------------------------


class DSARouteState(str, enum.Enum):
    """Per-request NPU execution mode.

    See design section 6.1.  The route state describes *how the latent/indexer
    KV is laid out on the NPU for this request*.  It is orthogonal to the PD
    transfer obligations captured by :class:`DSATransferPlan`.
    """

    # threshold == 0 bypass.  The current feature classifier / metadata is
    # retained, but completion-safety state (RequestKey, operation registry,
    # quorum, failure cleanup) is still maintained.
    LEGACY = "legacy"
    # latent + indexer fully resident; absolute top-k indices.
    RESIDENT = "resident"
    # crossed the threshold; still fully resident while sparse source is being
    # saved / sealed / activated.  Absolute indices.
    PROMOTING = "promoting"
    # latent prefix released; scratch + null holes + live tail; indexer fully
    # resident; sparse remap active.
    SPARSE = "sparse"
    # restoring the dense layout (scratch payload + null holes) prior to a
    # streaming append or explicit rollback.
    RECOVERING = "recovering"
    # promotion failed before any release; this request stays resident and is
    # not retried by default.
    FALLBACK_RESIDENT = "fallback_resident"
    # request finished / aborted / failed; blocks released.
    FINISHED = "finished"


# Scheduler-level waiting sub-states.  These are *not* part of the public
# RequestStatus enum because they are DSA-internal gating states, but they map
# onto the scheduler waiting decision: a request in any of these states must
# not enter a normal model forward.  They are represented as strings so they
# can be carried on the Request alongside the canonical RequestStatus.
WAITING_FOR_PD_EXPORT = "waiting_for_pd_export"
WAITING_FOR_PD_IMPORT = "waiting_for_pd_import"
WAITING_FOR_SOURCE_ACTIVATION = "waiting_for_source_activation"
WAITING_FOR_DSA_RECOVERY = "waiting_for_dsa_recovery"
WAITING_FOR_PREEMPTION_QUIESCE = "waiting_for_preemption_quiesce"
WAITING_FOR_ROUTE_DRAIN = "waiting_for_route_drain"

DSA_WAITING_STATES = frozenset({
    WAITING_FOR_PD_EXPORT,
    WAITING_FOR_PD_IMPORT,
    WAITING_FOR_SOURCE_ACTIVATION,
    WAITING_FOR_DSA_RECOVERY,
    WAITING_FOR_PREEMPTION_QUIESCE,
    WAITING_FOR_ROUTE_DRAIN,
})


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RequestKey:
    """Process-incarnation-aware local request identity.

    ``process_instance_id`` is generated once per Scheduler process; ``scope_id``
    monotonically increases within that process.  Only the full tuple uniquely
    identifies a local request lifecycle -- a reused user request_id does NOT
    identify the same lifecycle.
    """

    process_instance_id: str
    request_id: str
    scope_id: int

    def __str__(self) -> str:
        return f"{self.process_instance_id}:{self.request_id}:{self.scope_id}"


@dataclass(frozen=True)
class ParticipantIdentity:
    """A single engine/rank/process participant in the DSA transfer protocol."""

    engine_id: str
    process_instance_id: str
    worker_rank: int
    dp_rank: int
    pp_rank: int
    tp_rank: int


# ---------------------------------------------------------------------------
# Transfer plan (orthogonal to route state)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DSATransferPlan:
    """A set of simultaneously-possible transfer obligations.

    See design section 2.3.  These are NOT a single role: a PD Decoder long
    request has ``must_import_prefix`` and ``must_persist_decode_windows`` both
    true.  Using ``kv_both`` / a single role enum to express node role is
    explicitly rejected by the design.
    """

    must_import_prefix: bool = False
    must_export_prefill: bool = False
    must_persist_decode_windows: bool = False
    remote_handoff_required: bool = False


# ---------------------------------------------------------------------------
# PD transfer context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PDTransferIdentity:
    """Cross-process PD transfer identity, agreed by the Router."""

    transfer_id: str
    trace_id: str
    data_compatibility_fingerprint: str


@dataclass(frozen=True)
class HandoffLeaseDescriptor:
    """Lease over a remote handoff object.

    Lease descriptors are sensitive -- the renew/release handles must not enter
    ordinary logs.  They are carried here as opaque ids so the Scheduler /
    Router can reference them without exposing secrets.
    """

    lease_id: str
    manifest_key_set_digest: str
    expires_at_ms: int
    policy: Literal["hard", "qualified_soft_ttl"]
    lease_owner: str
    renew_handle_id: str
    release_handle_id: str


@dataclass(frozen=True)
class ProducerAttemptBinding:
    """Router -> P binding, established before Prefill starts."""

    source_epoch: int
    prefill_endpoint: str
    attempt_id: str
    deadline_ms: int


@dataclass(frozen=True)
class ProducerSourceBinding:
    """P -> Router binding produced after export ready."""

    producer_request_key: RequestKey
    attempt: ProducerAttemptBinding
    source_manifest_id: str
    source_generation_id: str
    receipt_bundle_id: str
    lease: HandoffLeaseDescriptor


@dataclass(frozen=True)
class ConsumerBinding:
    """Router -> D binding after decode dispatch."""

    consumer_request_key: RequestKey
    dispatch_epoch: int
    import_operation_id: str


@dataclass(frozen=True)
class DSATransferContext:
    """The full staged PD transfer context for one request."""

    identity: PDTransferIdentity
    producer_attempt: Optional[ProducerAttemptBinding] = None
    producer: Optional[ProducerSourceBinding] = None
    consumer: Optional[ConsumerBinding] = None


# ---------------------------------------------------------------------------
# Operation protocol (Scheduler-issued command, exact-set quorum)
# ---------------------------------------------------------------------------


# Obligation kinds a single physical operation can serve simultaneously.
# A long P store can serve both ``promotion`` and ``prefill_export``.
ObligationKind = Literal[
    "promotion",
    "window",
    "prefill_export",
    "import",
    "source_activation",
    "recovery",
    "preemption_quiesce",
]

# Physical task kind.
OperationKind = Literal[
    "store",
    "import",
    "source_activation",
    "recovery",
    "preemption_quiesce",
]


@dataclass(frozen=True)
class DSAOperationRef:
    """Reference to a single physical operation.

    ``obligations`` is the set of *logical* consumers of this physical task.
    Each obligation has its own consumer terminal/refcount; the physical
    record is only tombstoned once ALL consumers terminate.  A failure fans
    out to every obligation.
    """

    operation_id: str
    parent_operation_id: Optional[str]
    kind: OperationKind
    obligations: frozenset[ObligationKind]
    range_start: int
    range_end: int
    input_generation_id: Optional[str]
    output_generation_id: Optional[str]
    route_epoch: int


@dataclass(frozen=True)
class DSASourceLease:
    """Worker-held source lease binding one (execution_seq, route_epoch).

    Workers acquire a unique lease per ``(RequestKey, execution_seq,
    participant)`` exact-set.  All layer loads and NPU stream fences must
    complete before the lease is released; a new generation may only unpin an
    old MemoryObj after the old lease refcount reaches zero.
    """

    source_lease_id: str
    request_key: RequestKey
    execution_seq: int
    route_epoch: int
    source_generation_id: str


ReceiptKind = Literal[
    "storage",
    "source_seal",
    "npu_materialization",
    "route_epoch_complete",
    "quiesce_complete",
]

StorageTier = Literal["npu", "local_cpu", "mooncake"]

Guarantee = Literal[
    "npu_materialized",
    "local_cpu_pinned",
    "remote_put_fenced",
    "remote_handoff_leased",
    "qualified_soft_ttl",
    "fault_domain_replicated",
]


@dataclass(frozen=True)
class DSAReceiptExpectation:
    """One atomic expectation a command requires before it can be satisfied.

    The command builder expands multi-group/tier requirements into single
    ``kv_group`` expectations; one receipt satisfies exactly one expectation
    key.  A group-0 receipt can never cover a ``(0, 1)`` set entry.
    """

    participant: ParticipantIdentity
    receipt_kind: ReceiptKind
    kv_group: int
    layers: tuple[int, ...]
    chunks: tuple[tuple[int, int], ...]
    storage_tier: StorageTier
    minimum_guarantee: str


@dataclass(frozen=True)
class DSAOperationCommand:
    """Scheduler-issued command driving one physical operation."""

    request_key: RequestKey
    operation: DSAOperationRef
    accepted_end_at_issue: int
    token_prefix_digest: str
    cache_namespace_fingerprint: str
    expected_receipts: tuple[DSAReceiptExpectation, ...]


@dataclass(frozen=True)
class DSAOperationReceipt:
    """An individual worker-emitted receipt.

    Individual workers only emit receipts; the executor aggregates into
    bundles and emits canonical ready/frontier events.  No single worker
    adapter may declare a global publish.
    """

    receipt_id: str
    request_key: RequestKey
    operation_id: str
    receipt_kind: ReceiptKind
    route_epoch: int
    input_generation_id: Optional[str]
    output_generation_id: Optional[str]
    accepted_end_at_seal: int
    token_prefix_digest: str
    cache_namespace_fingerprint: str
    range_start: int
    range_end: int
    kv_group: int
    participant: ParticipantIdentity
    covered_layers: tuple[int, ...]
    covered_chunks: tuple[tuple[int, int], ...]
    storage_tier: StorageTier
    status: Literal["complete", "failed"]
    lease_descriptor_id: Optional[str]
    guarantee: Guarantee


@dataclass(frozen=True)
class DSAReceiptBundle:
    """An executor-aggregated bundle of receipts for one operation.

    Aggregation is exact-set, never a bare ``max``: a bundle is READY only if
    every required participant/group/layer/chunk/tier expectation is covered
    exactly, and FAILED if any required participant/group fails.
    """

    bundle_id: str
    request_key: RequestKey
    operation_id: str
    route_epoch: int
    input_generation_id: Optional[str]
    output_generation_id: Optional[str]
    source_manifest_id: Optional[str]
    token_prefix_digest: str
    data_compatibility_fingerprint: str
    receipt_ids: tuple[str, ...]
    aggregate_status: Literal["complete", "failed"]
    raw_source_end: int
    sparse_source_end: int
    materialized_end: int
    lease_descriptor_id: Optional[str]


@dataclass
class DSAOperationRecord:
    """Scheduler-side mutable record tracking one in-flight operation.

    Lifecycle: ``issued`` -> ``collecting`` -> ``ready`` -> ``activating`` ->
    ``committed`` (success) OR ``failed`` / ``superseded``.  Once ``ready`` or
    ``committed``, late failures are rejected as stale events.
    """

    command: DSAOperationCommand
    receipts_by_expectation: dict[tuple, DSAOperationReceipt] = field(
        default_factory=dict
    )
    status: Literal[
        "issued", "collecting", "ready", "activating",
        "committed", "failed", "superseded",
    ] = "issued"
    deadline_ms: int = 0
    consumer_status: dict[str, Literal["pending", "committed", "failed"]] = field(
        default_factory=dict
    )
    bundle_id: Optional[str] = None
    error_code: Optional[str] = None

    @property
    def is_terminal(self) -> bool:
        return self.status in ("committed", "failed", "superseded")


ControlEventKind = Literal[
    "promotion_ready",
    "window_ready",
    "export_ready",
    "import_ready",
    "source_activation_ready",
    "store_failed",
    "source_revoked",
    "recovery_ready",
    "preemption_quiesce_ready",
]


@dataclass(frozen=True)
class DSAControlEvent:
    """A control event emitted by the executor / Router to the Scheduler."""

    request_key: RequestKey
    operation_id: str
    route_epoch: int
    input_generation_id: Optional[str]
    output_generation_id: Optional[str]
    kind: ControlEventKind
    range_start: int = 0
    range_end: int = 0
    raw_source_end: int = 0
    sparse_source_end: int = 0
    receipt_bundle_id: Optional[str] = None
    error_code: Optional[str] = None


# ---------------------------------------------------------------------------
# Source activation / connector-only import / recovery / preemption
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DSAConnectorOnlyImport:
    """Connector-only dense import command (no attn_metadata dependency).

    Used by PD Decode to dense-bootstrap both groups without a model forward.
    The Scheduler first allocates full target blocks for both groups, then
    issues this command.  The worker calls a dedicated full-layer dense import
    API; after all get / D2H-H2D stream fences and participant materialization
    receipts complete, the executor aggregates ``import_ready``.
    """

    command: DSAOperationCommand
    transfer_context: DSATransferContext
    token_count: int
    external_computed_end: int
    latent_block_table: tuple[int, ...]
    indexer_block_table: tuple[int, ...]
    latent_slot_mapping: tuple[int, ...]
    indexer_slot_mapping: tuple[int, ...]


@dataclass(frozen=True)
class DSARecoveryCommand:
    """Recovery command restoring dense layout prior to streaming append."""

    operation_id: str
    source_request_key: RequestKey
    target_request_key: RequestKey
    restore_start: int
    restore_end: int
    source_generation_id: str
    token_prefix_digest: str
    source_receipt_bundle_id: str


@dataclass(frozen=True)
class DSARecoveryReceipt:
    operation_id: str
    target_request_key: RequestKey
    participant: ParticipantIdentity
    kv_group: int
    covered_layers: tuple[int, ...]
    covered_chunks: tuple[tuple[int, int], ...]
    restored_range: tuple[int, int]
    status: Literal["complete", "failed"]


@dataclass(frozen=True)
class DSAPreemptionAction:
    """Typed preemption action replacing the bare request-id preemption.

    Old RequestKey scheduling is stopped and put into
    WAITING_FOR_PREEMPTION_QUIESCE; the Ascend runner / connector stops old
    retrieve/store/DMA and waits for stream fence; once all required
    participant quiesce receipts are collected, both groups' old blocks are
    freed and a new RequestKey is created for full dense rebuild.
    """

    old_request_key: RequestKey
    preemption_seq: int
    rebuild_mode: Literal["dense_recompute", "pd_dense_import"]
    operation: DSAOperationRef
    expected_receipts: tuple[DSAReceiptExpectation, ...]


# ---------------------------------------------------------------------------
# Capability handshake
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DSADataCompatibility:
    """Immutable data compatibility descriptor.

    ``data_compatibility_fingerprint`` is computed from this object and enters
    the v2 cache namespace.  All participants in a PD cluster MUST share an
    identical fingerprint; the Router checks ``/health/config`` equality of
    this fingerprint across P and D before dispatch.
    """

    model_weight_content_digest: str
    model_config_digest: str
    quantization_schema: str
    kv_layout_schema: str
    kv_dtype: str
    execution_kv_abi_version: str
    block_size: int
    chunk_size: int
    window_size: int
    index_topk: int
    query_width: int
    scratch_capacity: int
    kv_groups: tuple[int, ...]
    required_layers: tuple[int, ...]
    token_hash_schema: str
    cache_salt_schema: str
    cache_namespace_version: str

    def fingerprint(self) -> str:
        # Stable, order-independent encoding of the fields that affect KV
        # bytes / layout.  Excludes nothing load-bearing.
        import hashlib
        import json

        payload = json.dumps(
            {
                "w": self.model_weight_content_digest,
                "c": self.model_config_digest,
                "q": self.quantization_schema,
                "l": self.kv_layout_schema,
                "d": self.kv_dtype,
                "abi": self.execution_kv_abi_version,
                "bs": self.block_size,
                "cs": self.chunk_size,
                "ws": self.window_size,
                "tk": self.index_topk,
                "qw": self.query_width,
                "sc": self.scratch_capacity,
                "g": list(self.kv_groups),
                "rl": list(self.required_layers),
                "th": self.token_hash_schema,
                "ss": self.cache_salt_schema,
                "cnv": self.cache_namespace_version,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DSAInstanceCapabilities:
    """Per-instance capability descriptor (role/topology/storage/lease).

    ``data_compatibility_fingerprint`` must match across P/D; the instance
    capability digest only identifies this instance.  Role / participant
    topology / storage / lease policy MAY differ and are validated against the
    P/D role matrix rather than by whole-object hash equality.
    """

    participant_ids: tuple[ParticipantIdentity, ...]
    authoritative_writer: Optional[ParticipantIdentity]
    storage_tiers: tuple[str, ...]
    deployment_mode: str
    node_role: str
    remote_guarantees: tuple[str, ...]
    lease_policy: str
    native_boundary_validation: bool


# ---------------------------------------------------------------------------
# Mutable request-scoped Scheduler state
# ---------------------------------------------------------------------------


@dataclass
class DSARequestState:
    """The Scheduler-owned mutable per-request DSA state.

    All frontier bookkeeping lives here.  Workers/connectors never mutate this
    directly; they consume :class:`DSARouteSnapshot` instances derived from it.

    Frontier invariant (design section 7.2)::

        0 <= release_end <= remap_end <= min(sparse_source_end,
                                              completed_canonical_end,
                                              resident_tail_start) <= accepted_end
        0 <= sparse_source_end <= raw_source_end
        remap_end == 0 or remap_end >= scratch_capacity
        release_end / remap_end / sparse_source_end monotonically non-decreasing
          within one RequestKey lifecycle
        all release boundaries block-aligned
        sparse_source_end chunk-aligned
    """

    request_key: RequestKey
    route_state: DSARouteState
    route_epoch: int
    threshold: int
    threshold_crossed_at: Optional[int] = None
    completed_canonical_end: int = 0
    external_computed_end: int = 0
    initial_prefill_complete: bool = False
    raw_source_end: int = 0
    sparse_source_end: int = 0
    remap_end: int = 0
    release_end: int = 0
    promotion_desired_end: int = 0
    promotion_inflight: Optional[DSAOperationRef] = None
    export_inflight: Optional[DSAOperationRef] = None
    import_inflight: Optional[DSAOperationRef] = None
    source_activation_inflight: Optional[DSAOperationRef] = None
    window_inflight: Optional[DSAOperationRef] = None
    recovery_inflight: Optional[DSAOperationRef] = None
    preemption_inflight: Optional[DSAOperationRef] = None
    latest_sealed_generation_id: Optional[str] = None
    latest_sealed_receipt_bundle_id: Optional[str] = None
    latest_sealed_raw_source_end: int = 0
    latest_sealed_sparse_source_end: int = 0
    active_source_generation_id: Optional[str] = None
    active_source_receipt_bundle_id: Optional[str] = None
    active_token_prefix_digest: Optional[str] = None
    window_anchor: int = 0
    next_window_start: int = 0
    recovery_target_end: int = 0
    transfer_plan: DSATransferPlan = field(default_factory=DSATransferPlan)
    transfer_context: Optional[DSATransferContext] = None
    fallback_reason: Optional[str] = None

    def has_inflight_operation(self) -> bool:
        """True if any lane has a single in-flight operation.

        Each lane allows at most one in-flight operation (design section 9.1).
        """
        return any(
            ref is not None
            for ref in (
                self.promotion_inflight,
                self.export_inflight,
                self.import_inflight,
                self.source_activation_inflight,
                self.window_inflight,
                self.recovery_inflight,
                self.preemption_inflight,
            )
        )

    def is_waiting(self) -> bool:
        """True when this request must not enter a normal model forward."""
        return self.route_state in (
            DSARouteState.RECOVERING,
        ) or bool(self.waiting_reason())

    def waiting_reason(self) -> Optional[str]:
        if self.preemption_inflight is not None:
            return WAITING_FOR_PREEMPTION_QUIESCE
        if self.recovery_inflight is not None:
            return WAITING_FOR_DSA_RECOVERY
        if self.source_activation_inflight is not None:
            return WAITING_FOR_SOURCE_ACTIVATION
        if self.import_inflight is not None:
            return WAITING_FOR_PD_IMPORT
        if self.export_inflight is not None:
            return WAITING_FOR_PD_EXPORT
        return None


# ---------------------------------------------------------------------------
# Immutable per-step snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DSARouteSnapshot:
    """Immutable route snapshot published to workers/connectors each step.

    Workers/connectors MUST consume this snapshot and MUST NOT re-derive sparse
    mode from ``prompt_len``.  Only a snapshot carrying a unique
    :class:`DSASourceLease` authorizes a SPARSE forward to release latent.
    """

    request_key: RequestKey
    execution_seq: int
    route_epoch: int
    route_state: DSARouteState
    accepted_end: int
    completed_canonical_end: int
    external_computed_end: int
    initial_prefill_complete: bool
    raw_source_end: int
    sparse_source_end: int
    remap_end: int
    release_end: int
    promotion_desired_end: int
    active_source_generation_id: Optional[str]
    active_source_receipt_bundle_id: Optional[str]
    active_token_prefix_digest: Optional[str]
    source_lease: Optional[DSASourceLease]
    window_anchor: int
    next_window_start: int
    transfer_plan: DSATransferPlan
    transfer_context: Optional[DSATransferContext]


def build_snapshot(
    state: DSARequestState,
    *,
    execution_seq: int,
    accepted_end: int,
    source_lease: Optional[DSASourceLease] = None,
) -> DSARouteSnapshot:
    """Derive an immutable snapshot from mutable request state."""
    return DSARouteSnapshot(
        request_key=state.request_key,
        execution_seq=execution_seq,
        route_epoch=state.route_epoch,
        route_state=state.route_state,
        accepted_end=accepted_end,
        completed_canonical_end=state.completed_canonical_end,
        external_computed_end=state.external_computed_end,
        initial_prefill_complete=state.initial_prefill_complete,
        raw_source_end=state.raw_source_end,
        sparse_source_end=state.sparse_source_end,
        remap_end=state.remap_end,
        release_end=state.release_end,
        promotion_desired_end=state.promotion_desired_end,
        active_source_generation_id=state.active_source_generation_id,
        active_source_receipt_bundle_id=state.active_source_receipt_bundle_id,
        active_token_prefix_digest=state.active_token_prefix_digest,
        source_lease=source_lease,
        window_anchor=state.window_anchor,
        next_window_start=state.next_window_start,
        transfer_plan=state.transfer_plan,
        transfer_context=state.transfer_context,
    )


def derive_transfer_plan_for_promotion(
    previous: DSATransferPlan,
    deployment_role: str,
) -> DSATransferPlan:
    """Return a new frozen transfer plan for a request that just crossed the
    threshold.

    PD Decoder must at least set ``must_persist_decode_windows`` and retain the
    bound transfer context; it must NOT just flip the route enum and keep using
    the admission-time import-only plan.
    """
    must_persist = previous.must_persist_decode_windows
    if deployment_role == "decode":
        must_persist = True
    return replace(
        previous,
        must_persist_decode_windows=must_persist,
    )
