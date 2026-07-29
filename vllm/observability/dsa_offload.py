"""DSA offload structured diagnostic logging (``dsa_offload.v1``).

This module implements the *pure observation layer* of the GLM-5.1 DSA feature
flow log enhancement. Every structured event is a single line JSON object
emitted behind the ``[DSA_OFFLOAD] `` marker so it can be extracted with
``rg``/``jq`` and correlated across the four repos (vLLM, vLLM-Ascend, LMCache,
LMCache-Ascend).

Design invariants (see ``02_GLM51_DSA特性流程日志增强详细设计.md``):

* ``diag=off`` (default) must not touch the hot path: callers guard with
  ``enabled()`` and the helper returns before evaluating any (lazy) fields.
* Normal/sampled events never read NPU tensor values. ``emit()`` rejects any
  ``torch.Tensor`` value (``__class__`` from the ``torch`` package) immediately,
  at every non-off level. Deep tensor probes use caller-supplied, already-safe
  scalar summaries behind an ``enabled(DiagLevel.DEEP)`` guard.
* All expensive fields are lazy: pass ``fields=lambda: build()`` so that when
  the level/sample gate fails the supplier is never called.
* Failures are never dropped by normal sampling, but repeat failures are
  throttled with a bounded window.
* Sampling/dedup tables are bounded LRU (4096 entries) and cleared on request
  finish.

This file is intentionally dependency-light (only ``os``/``json``/``time``/
``uuid``/``threading``/``collections`` + the host logger factory) so the same
contract can be mirrored inside LMCache without forcing generic storage
backends to depend on vLLM or vLLM-Ascend.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Union

from vllm.logger import init_logger

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DSA_SCHEMA = "dsa_offload.v1"
DSA_MARKER = "[DSA_OFFLOAD] "

# Bounded sizes. JSON lines should stay well under a log collector's line cap.
_MAX_STRING_LEN = 512
_MAX_LIST_LEN = 64
_MAX_JSON_BYTES = 4096
_MAX_SAMPLE_ENTRIES = 4096
_MAX_DEDUP_ENTRIES = 4096
_FAILURE_THROTTLE_WINDOW_NS = 5 * 1_000_000_000  # 5s


class DiagLevel:
    """Diagnostic verbosity levels (ordered, higher == more verbose)."""

    OFF = 0
    LIFECYCLE = 1
    SAMPLED = 2
    DEEP = 3

    @staticmethod
    def parse(raw: str) -> int:
        v = (raw or "off").strip().lower()
        if v in ("0", "off", ""):
            return DiagLevel.OFF
        if v in ("1", "lifecycle"):
            return DiagLevel.LIFECYCLE
        if v in ("2", "sampled"):
            return DiagLevel.SAMPLED
        if v in ("3", "deep"):
            return DiagLevel.DEEP
        # Unknown value falls back to off rather than enabling noisy defaults.
        return DiagLevel.OFF


# Authoritative event name registry (§6.6). Emitters MUST use one of these; the
# helper does not hard-enforce membership (to stay cheap), but tests assert it.
DSA_EVENTS = frozenset({
    # Startup
    "runtime.config", "cache.layout", "cache.host.ready", "remote.ready",
    "capability.report", "capability.accepted",
    # Request / prefill
    "request.admit", "route.observed", "route.classify", "cache.lookup",
    "request.allocate", "allocation.blocked", "prefill.chunk.plan",
    "prefill.complete", "forward.plan",
    # Store / source / frontier / release
    "store.command", "store.group.submit", "store.group.local_ready",
    "remote.api_call", "remote.put.wait_timeout", "remote.put.fenced",
    "remote.get.fenced", "remote.handoff.leased", "remote.handoff.released",
    "remote.handoff.renewed", "store.batch.fenced", "frontier.candidate",
    "source.generation.sealed", "frontier.publish", "frontier.consume",
    "source.activation.command", "source.activation.participant_ready",
    "source.activation.commit", "latent.release.commit",
    # Decode
    "decode.step.begin", "index.select", "sparse.plan",
    "sparse.retrieve.submit", "sparse.retrieve.enqueued",
    "sparse.retrieve.fenced", "source.generation.in_use",
    "source.generation.idle", "window.command", "window.ready",
    "decode.step.complete", "execution.fatal",
    # PD
    "pd.prefill.dispatch", "pd.prefill.ready.submit",
    "pd.prefill.ready.accepted", "pd.route.dispatch", "pd.rank.materialized",
    "pd.decode.materialized", "pd.decode.quiesce.command", "pd.decode.quiesced",
    # Preemption / finish
    "preemption.quiesce.command", "preemption.quiesce.participant_ready",
    "preemption.quiesce.commit", "request.finish", "request.cleanup",
    # Safety
    "invariant.violation", "failure.suppressed",
})


# ---------------------------------------------------------------------------
# Typed correlation structures (§5.4, §7.7)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RequestKey:
    """Process-incarnation-aware request identity.

    ``process_instance_id`` is the Scheduler incarnation that created the scope;
    an emitter (worker) must NOT substitute its own ``process_instance`` here.
    """

    process_instance_id: str
    request_id: str
    scope_id: int

    def as_log_dict(self) -> dict[str, Any]:
        return {
            "request_process_instance": self.process_instance_id,
            "scope_id": self.scope_id,
        }


@dataclass(frozen=True)
class ParticipantIdentity:
    engine_id: str
    process_instance_id: str
    worker_rank: int
    dp_rank: int
    pp_rank: int
    tp_rank: int


@dataclass(frozen=True)
class LookupResult:
    """Tier-aware cache lookup result.

    Replaces the bare ``int`` previously returned by the lookup client so that
    ``cache.lookup`` can report the true source instead of guessing from config.
    """

    hit_tokens: int
    local_cpu_tokens: int = 0
    remote_tokens: int = 0
    mixed_tier: bool = False
    duration_ns: int = 0

    @property
    def source(self) -> Literal["local", "remote", "mixed", "unknown"]:
        if self.mixed_tier:
            return "mixed"
        if self.remote_tokens and not self.local_cpu_tokens:
            return "remote"
        if self.local_cpu_tokens and not self.remote_tokens:
            return "local"
        if self.remote_tokens and self.local_cpu_tokens:
            return "mixed"
        return "unknown"


@dataclass(frozen=True)
class DSAOperationReceipt:
    receipt_id: str
    request_key: RequestKey
    operation_id: str
    receipt_kind: Literal[
        "storage", "source_seal", "npu_materialization",
        "route_epoch_complete", "quiesce_complete",
    ]
    route_epoch: int
    input_generation_id: str | None
    output_generation_id: str | None
    accepted_end_at_seal: int
    token_prefix_digest: str
    cache_namespace_fingerprint: str
    range_start: int
    range_end: int
    kv_group: int
    participant: ParticipantIdentity
    covered_layers: tuple[int, ...]
    covered_chunks: tuple[tuple[int, int], ...]
    storage_tier: Literal["npu", "local_cpu", "mooncake"]
    status: Literal["complete", "failed"]
    lease_descriptor_id: str | None
    guarantee: Literal[
        "npu_materialized", "local_cpu_pinned", "remote_put_fenced",
        "remote_handoff_leased", "qualified_soft_ttl",
        "fault_domain_replicated",
    ]


@dataclass(frozen=True)
class DSAReceiptBundle:
    bundle_id: str
    request_key: RequestKey
    operation_id: str
    route_epoch: int
    input_generation_id: str | None
    output_generation_id: str | None
    source_manifest_id: str | None
    token_prefix_digest: str
    data_compatibility_fingerprint: str
    receipt_ids: tuple[str, ...]
    aggregate_status: Literal["complete", "failed"]
    raw_source_end: int
    sparse_source_end: int
    materialized_end: int
    lease_descriptor_id: str | None


# ---------------------------------------------------------------------------
# Runtime state (one per process; shared by all logger views)
# ---------------------------------------------------------------------------

# Sentinel quorum / scope strings used while the causal-verification protocol is
# not yet in place. Emitters must use these rather than claiming "proven".
QUORUM_UNPROVEN_MAX_MERGE = "unproven_current_max_merge"
SCOPE_UNSCOPED = "unscoped_current_protocol"
SOURCE_UNKNOWN_AT_CONNECTOR = "unknown_at_scheduler_connector"


@dataclass
class _ScopeSampleState:
    decode_steps: int = 0
    last_frontiers: dict[str, int] = field(default_factory=dict)
    last_mode: str | None = None
    last_window_end: int | None = None


class _DSAState:
    """Process-global diagnostic state, created once and lazily."""

    def __init__(self) -> None:
        # Configuration (captured once at process startup so that
        # ``process_instance`` is stable for the incarnation).
        self.level = self._resolve_level()
        self.sample_every = max(
            int(os.getenv("VLLM_ASCEND_DSA_DIAG_SAMPLE_EVERY", "128")), 1)
        self.trace_filter = os.getenv(
            "VLLM_ASCEND_DSA_DIAG_TRACE_ID", "").strip() or None
        self.node_tag = os.getenv(
            "VLLM_ASCEND_DSA_DIAG_NODE_TAG", "").strip() or None
        self.include_request_id = bool(int(
            os.getenv("VLLM_ASCEND_DSA_DIAG_INCLUDE_REQUEST_ID", "0")))
        self.stats_interval_s = max(
            float(os.getenv("VLLM_ASCEND_DSA_STATS_INTERVAL_SECONDS", "0")),
            0.0)
        self.process_instance = uuid.uuid4().hex
        self.pid = os.getpid()

        self._seq = 0
        self._lock = threading.Lock()
        # Per-scope sampling state, LRU.
        self._sample: OrderedDict[tuple, _ScopeSampleState] = OrderedDict()
        # Dedup set, LRU.
        self._dedup: OrderedDict[tuple, None] = OrderedDict()
        # Failure throttle: (event, reason) -> (window_start_ns, count).
        self._failure: dict[tuple, list] = {}

    # -- level resolution with legacy-compat mapping ------------------------
    @staticmethod
    def _resolve_level() -> int:
        raw = os.getenv("VLLM_ASCEND_DSA_DIAG_LEVEL")
        if raw is not None and raw.strip() != "":
            return DiagLevel.parse(raw)
        # Compat mapping (one version cycle): only the new variable selects the
        # level; the old flags merely map onto it. The old emitters MUST be
        # atomically disabled by the new protocol when this maps non-off.
        if os.getenv("VLLM_ASCEND_MTP_DW_DEEP_DIAG", "0") not in ("0", "", "false"):
            return DiagLevel.DEEP
        if os.getenv("VLLM_ASCEND_MTP_DW_DIAG", "0") not in ("0", "", "false"):
            return DiagLevel.SAMPLED
        return DiagLevel.OFF

    # -- sequence -----------------------------------------------------------
    def next_seq(self) -> int:
        with self._lock:
            self._seq += 1
            return self._seq

    # -- scope lifecycle ----------------------------------------------------
    def clear_scope(self, scope_key: tuple) -> None:
        with self._lock:
            self._sample.pop(scope_key, None)

    # -- sampling (§8.1) ----------------------------------------------------
    def should_sample(
        self,
        scope_key: tuple | None,
        fields: Mapping[str, Any],
        *,
        is_failure: bool,
        level: int,
    ) -> bool:
        # Only SAMPLED-level request events are subject to decode-step sampling.
        if level != DiagLevel.SAMPLED:
            return True
        if is_failure:
            return True
        # Non request-scoped sampled events (e.g. forward.plan) are low volume.
        if scope_key is None:
            return True
        # Frontiers present? If so, frontier/mode/window transitions force a
        # sample; otherwise apply the bounded decode-step cadence.
        schedule_id = fields.get("schedule_id")
        with self._lock:
            st = self._sample.get(scope_key)
            if st is None:
                st = _ScopeSampleState()
                self._sample[scope_key] = st
                self._sample.move_to_end(scope_key)
                self._evict_sample()
            force = False
            for fk in ("raw_source_end", "sparse_source_end", "remap_end",
                       "release_end"):
                v = fields.get(fk)
                if isinstance(v, int) and st.last_frontiers.get(fk) != v:
                    st.last_frontiers[fk] = v
                    force = True
            mode = fields.get("mode")
            if isinstance(mode, str) and st.last_mode != mode:
                st.last_mode = mode
                force = True
            we = fields.get("window_end")
            if isinstance(we, int) and st.last_window_end != we:
                st.last_window_end = we
                force = True
            # Route/preemption/recovery events are emitted at lifecycle level;
            # a sampled event carrying an explicit reason still records once.
            if fields.get("reason") in (
                    "preemption", "route_change", "recovery"):
                force = True
            if force:
                return True
            # Decode-step cadence (schedule_id marks a decode step).
            if schedule_id is not None:
                st.decode_steps += 1
                if st.decode_steps <= 3:
                    return True
                if st.decode_steps % self.sample_every == 0:
                    return True
                return False
            return True

    def _evict_sample(self) -> None:
        while len(self._sample) > _MAX_SAMPLE_ENTRIES:
            self._sample.popitem(last=False)

    # -- dedup (§8.3) ------------------------------------------------------
    def seen_dedup(self, key: tuple) -> bool:
        with self._lock:
            if key in self._dedup:
                self._dedup.move_to_end(key)
                return True
            self._dedup[key] = None
            self._dedup.move_to_end(key)
            while len(self._dedup) > _MAX_DEDUP_ENTRIES:
                self._dedup.popitem(last=False)
            return False

    # -- failure throttle ---------------------------------------------------
    def failure_should_emit(self, key: tuple) -> tuple[bool, int]:
        now = time.monotonic_ns()
        with self._lock:
            entry = self._failure.get(key)
            if entry is None or now - entry[0] > _FAILURE_THROTTLE_WINDOW_NS:
                self._failure[key] = [now, 1]
                return True, 0
            entry[1] += 1
            if entry[1] == 1:
                return True, 0
            # Emit a single suppression summary at the top of each subsequent
            # burst boundary.
            if entry[1] == 2:
                return True, entry[1] - 1
            return False, entry[1] - 1


_STATE: _DSAState | None = None
_STATE_LOCK = threading.Lock()


def _get_state() -> _DSAState:
    global _STATE
    if _STATE is not None:
        return _STATE
    with _STATE_LOCK:
        if _STATE is None:
            _STATE = _DSAState()
        return _STATE


def get_dsa_diag_level() -> int:
    """Return the resolved diagnostic level (creates state lazily)."""
    return _get_state().level


# ---------------------------------------------------------------------------
# Value sanitization / tensor guard
# ---------------------------------------------------------------------------

def _is_tensor(value: Any) -> bool:
    mod = type(value).__module__ or ""
    return mod.split(".", 1)[0] == "torch"


def _reject_tensors(payload: Mapping[str, Any]) -> None:
    bad = [k for k, v in payload.items() if _is_tensor(v)]
    if bad:
        raise TypeError(
            "DSA_OFFLOAD normal/sampled/deep events must not carry torch.Tensor "
            f"values (offending keys: {bad}). Pass already-summarized scalars "
            "behind an enabled(DiagLevel.DEEP) guard instead; the logger never "
            "calls .item()/.cpu()/.tolist() on your behalf.")


def _clamp_scalar(value: Any) -> Any:
    if isinstance(value, str) and len(value) > _MAX_STRING_LEN:
        return value[:_MAX_STRING_LEN - 3] + "..."
    return value


def _sanitize(value: Any) -> Any:
    value = _clamp_scalar(value)
    if isinstance(value, (list, tuple)):
        out = [_sanitize(v) for v in value[:_MAX_LIST_LEN]]
        if len(value) > _MAX_LIST_LEN:
            out.append(f"...<{len(value) - _MAX_LIST_LEN} more>")
        return out
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    return value


def _drop_none(payload: dict[str, Any]) -> dict[str, Any]:
    # Required envelope fields (§5.2) are always present, even when unset, so
    # consumers can rely on a stable schema; only caller-supplied optional
    # fields with a None value are removed.
    return {k: v for k, v in payload.items()
            if v is not None or k in _AUTO_FIELDS}


# ---------------------------------------------------------------------------
# Logger view
# ---------------------------------------------------------------------------

# Fields that are always auto-filled by the helper.
_AUTO_FIELDS = frozenset({
    "schema", "event", "outcome", "ts_ns", "mono_ns", "event_seq",
    "process_instance", "event_id", "emitter", "node_tag", "pid",
})


class DSAOffloadLogger:
    """A thin view over the shared process state for one ``emitter``.

    Multiple components create their own view via :func:`dsa_logger_for`; all
    views share the same ``process_instance``/sequence counters and sampling
    tables so events stay globally correlatable within the process.
    """

    __slots__ = (
        "emitter", "engine_id", "kv_role", "dp_rank", "pp_rank", "tp_rank",
        "_state",
    )

    def __init__(
        self,
        emitter: str,
        *,
        engine_id: str | None = None,
        kv_role: str | None = None,
        dp_rank: int | None = None,
        pp_rank: int | None = None,
        tp_rank: int | None = None,
        state: _DSAState | None = None,
    ) -> None:
        self.emitter = emitter
        self.engine_id = engine_id
        self.kv_role = kv_role
        self.dp_rank = dp_rank
        self.pp_rank = pp_rank
        self.tp_rank = tp_rank
        # Bind to the shared state lazily so importing this module never
        # triggers env parsing / uuid generation.
        self._state = state

    # -- introspection ------------------------------------------------------
    @property
    def state(self) -> _DSAState:
        if self._state is None:
            self._state = _get_state()
        return self._state

    def enabled(self, level: int = DiagLevel.LIFECYCLE) -> bool:
        return self.state.level >= level

    @property
    def node_tag(self) -> str | None:
        return self.state.node_tag

    @property
    def include_request_id(self) -> bool:
        return self.state.include_request_id

    def child(
        self,
        emitter: str,
        *,
        engine_id: str | None = None,
        kv_role: str | None = None,
        dp_rank: int | None = None,
        pp_rank: int | None = None,
        tp_rank: int | None = None,
    ) -> "DSAOffloadLogger":
        return DSAOffloadLogger(
            emitter,
            engine_id=engine_id if engine_id is not None else self.engine_id,
            kv_role=kv_role if kv_role is not None else self.kv_role,
            dp_rank=dp_rank if dp_rank is not None else self.dp_rank,
            pp_rank=pp_rank if pp_rank is not None else self.pp_rank,
            tp_rank=tp_rank if tp_rank is not None else self.tp_rank,
            state=self._state,
        )

    # -- emission -----------------------------------------------------------
    def emit(
        self,
        event: str,
        *,
        outcome: str = "ok",
        level: int = DiagLevel.LIFECYCLE,
        fields: Union[Mapping[str, Any], Callable[[], Mapping[str, Any]], None]
        = None,
        **cheap_fields: Any,
    ) -> None:
        state = self.state
        if state.level == DiagLevel.OFF:
            return
        if state.level < level:
            return

        # Evaluate the lazy supplier only after the level gate passed.
        merged: dict[str, Any] = {}
        if fields is not None:
            supplied = fields() if callable(fields) else fields
            if supplied:
                merged.update(supplied)
        if cheap_fields:
            merged.update(cheap_fields)

        # ``request_id`` is suppressed by default (privacy); opt-in only.
        if "request_id" in merged and not state.include_request_id:
            merged.pop("request_id")

        # Trace filtering: when a target trace is configured, drop request
        # events whose trace_id does not match. Non request-scoped events
        # (config/startup) are always emitted.
        trace_id = merged.get("trace_id")
        if (state.trace_filter is not None
                and trace_id is not None
                and trace_id != state.trace_filter):
            return

        is_failure = outcome in ("error", "degraded") or event in (
            "invariant.violation", "execution.fatal", "remote.put.wait_timeout")

        # Failure throttle: never dropped by sampling, but repeats are bounded.
        if is_failure:
            fkey = (event, str(merged.get("reason")), str(trace_id))
            emit_now, suppressed = state.failure_should_emit(fkey)
            if not emit_now:
                return
            if suppressed:
                logger.info(
                    DSA_MARKER + json.dumps(
                        {
                            "schema": DSA_SCHEMA,
                            "event": "failure.suppressed",
                            "outcome": "ok",
                            "reason": str(merged.get("reason") or "same_error"),
                            "count": suppressed,
                            "ts_ns": time.time_ns(),
                            "mono_ns": time.monotonic_ns(),
                            "emitter": self.emitter,
                        },
                        separators=(",", ":"),
                        default=str,
                    )
                )

        # Sampling for SAMPLED-level request events.
        scope_key = self._scope_key(merged)
        if not state.should_sample(scope_key, merged,
                                   is_failure=is_failure, level=level):
            return

        # Tensor guard: the logger never serializes NPU tensors.
        _reject_tensors(merged)

        # Dedup (only for the cardinality-prone lifecycle events listed in §8.3
        # when they carry a stable natural key).
        if scope_key is not None and event in (
                "source.generation.in_use", "source.generation.idle"):
            dkey = (trace_id, merged.get("request_process_instance"),
                    merged.get("scope_id"), merged.get("route_epoch"), event,
                    merged.get("schedule_id"), merged.get("operation_id"),
                    merged.get("source_generation_id"), merged.get("kv_group"),
                    self.tp_rank)
            if state.seen_dedup(dkey):
                return

        seq = state.next_seq()
        ts_ns = time.time_ns()
        payload: dict[str, Any] = {
            "schema": DSA_SCHEMA,
            "event": event,
            "outcome": outcome,
            "ts_ns": ts_ns,
            "mono_ns": time.monotonic_ns(),
            "event_seq": seq,
            "process_instance": state.process_instance,
            "event_id": f"{state.process_instance}:{seq}",
            "emitter": self.emitter,
            "node_tag": state.node_tag,
            "pid": state.pid,
        }
        if self.engine_id is not None:
            payload["engine_id"] = self.engine_id
        if self.kv_role is not None:
            payload["kv_role"] = self.kv_role
        if self.dp_rank is not None:
            payload["dp_rank"] = self.dp_rank
        if self.pp_rank is not None:
            payload["pp_rank"] = self.pp_rank
        if self.tp_rank is not None:
            payload["tp_rank"] = self.tp_rank
        # Caller fields override nothing structural (auto fields are fixed).
        for k, v in merged.items():
            if k in _AUTO_FIELDS:
                continue
            payload[k] = _sanitize(v)
        payload = _drop_none(payload)

        line = json.dumps(payload, separators=(",", ":"), default=str)
        if len(line.encode("utf-8", errors="replace")) > _MAX_JSON_BYTES:
            # Truncate the ``reason``/longest field rather than the envelope.
            payload["truncated"] = True
            line = json.dumps(payload, separators=(",", ":"), default=str)
            if len(line.encode("utf-8", errors="replace")) > _MAX_JSON_BYTES:
                line = line[:_MAX_JSON_BYTES - 1]
        logger.info(DSA_MARKER + line)

    # -- request lifecycle hooks -------------------------------------------
    def request_finished(self, scope_key: tuple | None) -> None:
        if scope_key is not None:
            self.state.clear_scope(scope_key)

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _scope_key(merged: Mapping[str, Any]) -> tuple | None:
        trace_id = merged.get("trace_id")
        rpi = merged.get("request_process_instance")
        scope_id = merged.get("scope_id")
        if rpi is None or scope_id is None:
            return None
        return (trace_id, rpi, scope_id)


# Module-level convenience: a single root view used by call sites that do not
# need per-component rank context (rank context can be supplied per-call via
# the cheap fields ``tp_rank``/``dp_rank``/...).
_root_logger = DSAOffloadLogger("vllm.dsa")


def dsa_logger_for(
    emitter: str,
    *,
    engine_id: str | None = None,
    kv_role: str | None = None,
    dp_rank: int | None = None,
    pp_rank: int | None = None,
    tp_rank: int | None = None,
) -> DSAOffloadLogger:
    """Return a :class:`DSAOffloadLogger` view bound to ``emitter``."""
    return _root_logger.child(
        emitter,
        engine_id=engine_id,
        kv_role=kv_role,
        dp_rank=dp_rank,
        pp_rank=pp_rank,
        tp_rank=tp_rank,
    )
