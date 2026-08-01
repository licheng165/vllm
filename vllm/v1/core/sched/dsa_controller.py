# SPDX-License-Identifier: Apache-2.0
"""DSA threshold-routing controller driven by the vLLM v1 Scheduler.

This module concentrates the request-level state machine logic described in
design sections 6, 7, 9 so the Scheduler only has to call a small number of
hooks at well-defined points:

* :meth:`DSAController.initialize_state` -- at request admission.
* :meth:`DSAController.consume_completed_execution` -- after a model output is
  accepted and ``all_token_ids`` is updated; refreshes ``C``/``E`` and detects
  threshold crossing.
* :meth:`DSAController.maybe_advance` -- each step; submits / advances
  promotion / window operations subject to the single-in-flight-per-lane rule.
* :meth:`DSAController.consume_event` -- consume a DSA control event
  (promotion_ready / source_activation_ready / window_ready / store_failed /
  preemption_quiesce_ready / import_ready / export_ready / recovery_ready).
* :meth:`DSAController.build_route_snapshots` -- produce the immutable
  ``dsa_routes`` map for the SchedulerOutput.
* :meth:`DSAController.begin_preemption` / :meth:`DSAController.finish_request`
  -- typed lifecycle transitions.

The controller deliberately does NOT touch the kv_cache_manager directly for
release; it produces a validated :class:`DSALatentReleaseTransaction` (or
None) that the Scheduler commits.  This keeps the no-fail-commit contract in
one place (kv_cache_manager) and the policy in another (here).
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from typing import Optional

from vllm.logger import init_logger
from vllm.v1.core.sched.dsa_operation_registry import (
    DSAOperationError,
    DSAOperationRegistry,
    align_down,
    align_up,
    safe_remap_end,
    validate_aligned_sparse_coverage,
    validate_monotonic_raw_source_coverage,
)
from vllm.v1.core.sched.dsa_types import (
    DSAControlEvent,
    DSAOperationRef,
    DSARequestState,
    DSARouteState,
    DSARouteSnapshot,
    DSASourceLease,
    DSATransferPlan,
    RequestKey,
    WAITING_FOR_DSA_RECOVERY,
    WAITING_FOR_PD_EXPORT,
    WAITING_FOR_PD_IMPORT,
    WAITING_FOR_PREEMPTION_QUIESCE,
    WAITING_FOR_SOURCE_ACTIVATION,
    build_snapshot,
    derive_transfer_plan_for_promotion,
)

logger = init_logger(__name__)


@dataclass
class ExecutionReceipt:
    """Worker-confirmed execution outcome for one request in one step."""

    request_key: RequestKey
    execution_seq: int
    completed_canonical_end: int
    external_computed_end: int
    initial_prefill_complete: bool
    route_epoch: int


@dataclass
class DSAControllerConfig:
    """Normalized DSA controller configuration.

    ``threshold == 0`` disables the threshold state machine: requests get the
    LEGACY route but still carry a RequestKey/DSARequestState for operation
    registry / quorum / failure-cleanup safety (design section 8.3).
    """

    threshold: int
    max_model_len: int
    block_size: int
    chunk_size: int
    window_size: int
    index_topk: int
    query_width: int
    scratch_capacity: int
    node_role: str  # standalone | prefill | decode
    deployment_mode: str  # standalone | pd
    data_compatibility_fingerprint: str
    instance_capability_digest: str

    @property
    def enabled(self) -> bool:
        return self.threshold > 0

    @property
    def minimum_valid_boundary(self) -> int:
        return self.scratch_capacity

    @property
    def first_reclaiming_frontier(self) -> int:
        return align_up(self.scratch_capacity + 1, self.chunk_size)

    @classmethod
    def disabled(
        cls,
        *,
        block_size: int,
        chunk_size: int,
        node_role: str = "standalone",
    ) -> "DSAControllerConfig":
        return cls(
            threshold=0,
            max_model_len=0,
            block_size=block_size,
            chunk_size=chunk_size,
            window_size=0,
            index_topk=0,
            query_width=1,
            scratch_capacity=0,
            node_role=node_role,
            deployment_mode="standalone",
            data_compatibility_fingerprint="",
            instance_capability_digest="",
        )


@dataclass
class DSAController:
    """Owns per-request DSA state and the operation registry."""

    config: DSAControllerConfig
    process_instance_id: str = field(
        default_factory=lambda: secrets.token_hex(8)
    )
    operations: DSAOperationRegistry = field(default_factory=DSAOperationRegistry)
    _scope_counter: int = 0
    # execution_seq is allocated per actual model execution.
    _execution_seq: int = 0
    _event_seq: int = field(default=0, init=False, repr=False)
    # request_id (string) -> RequestKey, so finish/preempt can locate state.
    _id_to_key: dict[str, RequestKey] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Identity / admission
    # ------------------------------------------------------------------

    def _new_request_key(self, request_id: str) -> RequestKey:
        self._scope_counter += 1
        key = RequestKey(
            process_instance_id=self.process_instance_id,
            request_id=request_id,
            scope_id=self._scope_counter,
        )
        self._id_to_key[request_id] = key
        return key

    def initialize_state(
        self,
        request_id: str,
        num_tokens: int,
        transfer_plan: Optional[DSATransferPlan] = None,
    ) -> DSARequestState:
        """Initialize DSA state for a newly admitted request.

        Implements design section 9.1.  threshold == 0 -> LEGACY; accepted >=
        threshold -> PROMOTING (but ``promotion_desired_end`` starts from the
        worker-confirmed C, NOT the full prompt length); else RESIDENT.
        """
        threshold = self.config.threshold
        key = self._new_request_key(request_id)
        plan = transfer_plan or DSATransferPlan()

        if threshold == 0:
            reason = "threshold_disabled"
            request_state = DSARequestState(
                request_key=key,
                route_state=DSARouteState.LEGACY,
                route_epoch=0,
                threshold=0,
                transfer_plan=plan,
            )
        else:
            if num_tokens >= threshold:
                route_state = DSARouteState.PROMOTING
                crossed_at = num_tokens
                reason = "at_or_above_threshold"
            else:
                route_state = DSARouteState.RESIDENT
                crossed_at = None
                reason = "below_threshold"
            request_state = DSARequestState(
                request_key=key,
                route_state=route_state,
                route_epoch=0,
                threshold=threshold,
                threshold_crossed_at=crossed_at,
                transfer_plan=plan,
            )

        self._emit(
            request_state,
            previous_state=None,
            reason=reason,
            accepted_end=num_tokens,
        )
        return request_state

    # ------------------------------------------------------------------
    # Execution consumption / threshold crossing
    # ------------------------------------------------------------------

    def consume_completed_execution(
        self, request_id: str, receipt: ExecutionReceipt
    ) -> None:
        """Refresh ``C`` / ``E`` and detect threshold crossing.

        Must be called AFTER the model output is accepted and
        ``all_token_ids`` is updated, NOT in ``_update_after_schedule``'s
        optimistic counter.  See design section 9.2.
        """
        key = self._id_to_key.get(request_id)
        if key is None:
            return
        state = self._lookup_state(request_id)
        if state is None:
            return
        if receipt.request_key != key:
            # Stale completion for an old incarnation: ignore (do not mutate).
            return

        prev_c = state.completed_canonical_end
        state.completed_canonical_end = max(
            state.completed_canonical_end, receipt.completed_canonical_end
        )
        state.external_computed_end = max(
            state.external_computed_end, receipt.external_computed_end
        )
        state.initial_prefill_complete |= receipt.initial_prefill_complete

        if state.route_state == DSARouteState.PROMOTING:
            state.promotion_desired_end = max(
                state.promotion_desired_end,
                align_down(state.completed_canonical_end, self.config.chunk_size),
            )

        # Threshold crossing: RESIDENT -> PROMOTING (design 9.2).
        # ``accepted_context_len`` is the current scope's token count, derived
        # from completed canonical + newly accepted output; the caller passes
        # it via receipt.completed_canonical_end relative to this scope.  We
        # compare the *accepted* length (prompt + accepted output) which is
        # >= completed_canonical_end; use the receipt's canonical end as the
        # authoritative lower bound plus the live tail.
        if (
            state.route_state == DSARouteState.RESIDENT
            and self.config.enabled
        ):
            accepted = max(receipt.completed_canonical_end, prev_c)
            if accepted >= state.threshold:
                previous_state = state.route_state
                state.transfer_plan = derive_transfer_plan_for_promotion(
                    previous=state.transfer_plan,
                    deployment_role=self.config.node_role,
                )
                state.route_state = DSARouteState.PROMOTING
                state.threshold_crossed_at = accepted
                state.promotion_desired_end = max(
                    state.promotion_desired_end,
                    align_down(
                        state.completed_canonical_end, self.config.chunk_size
                    ),
                )
                state.route_epoch += 1
                self._emit(
                    state,
                    previous_state=previous_state,
                    reason="context_threshold_crossed",
                    accepted_end=accepted,
                )

    def _lookup_state(self, request_id: str) -> Optional[DSARequestState]:
        # The caller (Scheduler) holds the authoritative DSARequestState on
        # the Request object; the controller mirrors the RequestKey mapping.
        # State objects are passed in by reference from the Scheduler via the
        # dedicated methods below.
        return self._states.get(request_id)

    # Scheduler registers the mutable state object so the controller can read
    # it without re-deriving from the Request.
    _states: dict[str, DSARequestState] = field(default_factory=dict)

    def attach_state(self, state: DSARequestState) -> None:
        self._states[state.request_key.request_id] = state

    @staticmethod
    def _selected_path(route_state: DSARouteState) -> str:
        if route_state == DSARouteState.LEGACY:
            return "legacy"
        if route_state in (
            DSARouteState.RESIDENT,
            DSARouteState.FALLBACK_RESIDENT,
        ):
            return "resident"
        if route_state == DSARouteState.RECOVERING:
            return "recovery"
        if route_state == DSARouteState.FINISHED:
            return "finished"
        return "dsa_sparse"

    @staticmethod
    def _execution_path(route_state: DSARouteState) -> str:
        if route_state == DSARouteState.LEGACY:
            return "legacy_classifier"
        if route_state == DSARouteState.SPARSE:
            return "sparse_remap"
        if route_state == DSARouteState.RECOVERING:
            return "blocked_recovery"
        if route_state == DSARouteState.FINISHED:
            return "none"
        return "resident_absolute"

    def _emit(
        self,
        state: DSARequestState,
        previous_state: DSARouteState | None,
        reason: str,
        accepted_end: int | None = None,
    ) -> None:
        """Log one authoritative route decision per request state change."""
        if not logger.isEnabledFor(logging.INFO):
            return
        self._event_seq += 1
        event_seq = self._event_seq
        route_state = state.route_state
        previous_mode = previous_state.value if previous_state is not None else None
        previous_selected_path = (
            self._selected_path(previous_state) if previous_state is not None else None
        )
        previous_execution_path = (
            self._execution_path(previous_state) if previous_state is not None else None
        )
        transfer_context = state.transfer_context
        # On first activation the latest sealed source becomes active before
        # the rolling SPARSE window frontiers are copied into raw/sparse state.
        raw_source_end = max(state.raw_source_end, state.latest_sealed_raw_source_end)
        sparse_source_end = max(
            state.sparse_source_end, state.latest_sealed_sparse_source_end
        )
        payload = {
            "schema": "dsa_offload.v1",
            "event": "route.classify",
            "outcome": (
                "degraded"
                if route_state
                in (DSARouteState.RECOVERING, DSARouteState.FALLBACK_RESIDENT)
                else "ok"
            ),
            "ts_ns": time.time_ns(),
            "mono_ns": time.monotonic_ns(),
            "event_seq": event_seq,
            "process_instance": self.process_instance_id,
            "event_id": f"{self.process_instance_id}:{event_seq}",
            "emitter": "vllm.scheduler",
            "node_tag": os.getenv(
                "VLLM_ASCEND_DSA_DIAG_NODE_TAG", self.config.node_role
            )[:64],
            "pid": os.getpid(),
            "request_process_instance": state.request_key.process_instance_id,
            "scope_id": state.request_key.scope_id,
            "trace_id": (
                transfer_context.identity.trace_id[:128]
                if transfer_context is not None
                else None
            ),
            "transfer_id": (
                transfer_context.identity.transfer_id[:128]
                if transfer_context is not None
                else None
            ),
            "node_role": self.config.node_role,
            "deployment_mode": self.config.deployment_mode,
            "route_epoch": state.route_epoch,
            "previous_mode": previous_mode,
            "mode": route_state.value,
            "previous_selected_path": previous_selected_path,
            "selected_path": self._selected_path(route_state),
            "previous_execution_path": previous_execution_path,
            "execution_path": self._execution_path(route_state),
            "reason": reason[:128],
            "threshold": self.config.threshold,
            "accepted_end": accepted_end,
            "completed_canonical_end": state.completed_canonical_end,
            "external_computed_end": state.external_computed_end,
            "initial_prefill_complete": state.initial_prefill_complete,
            "raw_source_end": raw_source_end,
            "sparse_source_end": sparse_source_end,
            "remap_end": state.remap_end,
            "release_end": state.release_end,
            "must_import_prefix": state.transfer_plan.must_import_prefix,
            "must_export_prefill": state.transfer_plan.must_export_prefill,
            "must_persist_decode_windows": (
                state.transfer_plan.must_persist_decode_windows
            ),
            "remote_handoff_required": (
                state.transfer_plan.remote_handoff_required
            ),
            "source_generation_id": state.active_source_generation_id,
            "receipt_bundle_id": state.active_source_receipt_bundle_id,
            "route_authority": "scheduler_state",
        }
        if os.getenv("VLLM_ASCEND_DSA_DIAG_INCLUDE_REQUEST_ID", "0") == "1":
            payload["request_id"] = state.request_key.request_id[:128]
        logger.info(
            "[DSA_OFFLOAD] %s",
            json.dumps(
                {key: value for key, value in payload.items() if value is not None},
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

    # ------------------------------------------------------------------
    # Promotion / window advancement (single in-flight per lane)
    # ------------------------------------------------------------------

    def maybe_advance(self, request_id: str) -> None:
        """Drive promotion submission / activation per step.

        Enforces one promotion operation in flight and one source-activation
        operation in flight.  See design section 9.1.
        """
        state = self._lookup_state(request_id)
        if state is None:
            return
        if state.route_state == DSARouteState.PROMOTING:
            self._maybe_advance_promotion(state)
        elif state.route_state == DSARouteState.SPARSE:
            self._maybe_advance_window(state)

    def _maybe_advance_promotion(self, state: DSARequestState) -> None:
        if (
            state.promotion_inflight is not None
            or state.source_activation_inflight is not None
        ):
            return
        if state.promotion_desired_end <= state.latest_sealed_sparse_source_end:
            return
        save_start = align_down(
            state.latest_sealed_raw_source_end, self.config.chunk_size
        )
        op_id = self._new_operation_id("promotion")
        ref = DSAOperationRef(
            operation_id=op_id,
            parent_operation_id=None,
            kind="store",
            obligations=frozenset({"promotion"}),
            range_start=save_start,
            range_end=state.promotion_desired_end,
            input_generation_id=state.latest_sealed_generation_id,
            output_generation_id=None,
            route_epoch=state.route_epoch,
        )
        state.promotion_inflight = ref

        # If promotion has already sealed a usable candidate and prefill is
        # complete, we can begin source activation once a remap > scratch is
        # achievable.
        if (
            state.initial_prefill_complete
            and state.latest_sealed_generation_id is not None
            and state.source_activation_inflight is None
        ):
            remap = safe_remap_end(
                sparse_source_end=state.latest_sealed_sparse_source_end,
                completed_canonical_end=state.completed_canonical_end,
                initial_prefill_complete=True,
                scratch_capacity=self.config.scratch_capacity,
                block_size=self.config.block_size,
                window_anchor=state.window_anchor,
                window_size=self.config.window_size,
            )
            if remap > self.config.scratch_capacity:
                self._begin_source_activation(
                    state,
                    candidate_generation_id=state.latest_sealed_generation_id,
                    candidate_bundle_id=state.latest_sealed_receipt_bundle_id,
                    proposed_remap_end=remap,
                    transition_kind="promotion",
                )

    def _maybe_advance_window(self, state: DSARequestState) -> None:
        if (
            state.window_inflight is not None
            or state.source_activation_inflight is not None
        ):
            return
        if self.config.window_size <= 0:
            return
        # Window command strictly from next_window_start along
        # window_anchor + k*window_size.
        target = state.next_window_start + self.config.window_size
        target = min(target, state.completed_canonical_end)
        if target <= state.next_window_start:
            return
        save_start = align_down(
            state.latest_sealed_raw_source_end, self.config.chunk_size
        )
        op_id = self._new_operation_id("window")
        ref = DSAOperationRef(
            operation_id=op_id,
            parent_operation_id=None,
            kind="store",
            obligations=frozenset({"window"}),
            range_start=save_start,
            range_end=target,
            input_generation_id=state.latest_sealed_generation_id,
            output_generation_id=None,
            route_epoch=state.route_epoch,
        )
        state.window_inflight = ref

    # ------------------------------------------------------------------
    # Source activation two-phase barrier
    # ------------------------------------------------------------------

    def _begin_source_activation(
        self,
        state: DSARequestState,
        *,
        candidate_generation_id: str,
        candidate_bundle_id: Optional[str],
        proposed_remap_end: int,
        transition_kind: str,
        next_window_start: Optional[int] = None,
    ) -> None:
        """Create the activation operation and put the request into
        WAITING_FOR_SOURCE_ACTIVATION.

        At this point we do NOT modify active generation / M / R / block
        table; that only happens after activation quorum (design 9.3)."""
        op_id = self._new_operation_id("source_activation")
        ref = DSAOperationRef(
            operation_id=op_id,
            parent_operation_id=None,
            kind="source_activation",
            obligations=frozenset({"source_activation"}),
            range_start=0,
            range_end=proposed_remap_end,
            input_generation_id=candidate_generation_id,
            output_generation_id=candidate_generation_id,
            route_epoch=state.route_epoch + 1,
        )
        state.source_activation_inflight = ref
        if next_window_start is not None:
            state.next_window_start = next_window_start

    # ------------------------------------------------------------------
    # Control event consumption
    # ------------------------------------------------------------------

    def consume_event(
        self, request_id: str, event: DSAControlEvent
    ) -> None:
        state = self._lookup_state(request_id)
        if state is None:
            return
        if event.request_key != state.request_key:
            # Stale event for an old incarnation: log-and-ignore.
            return
        try:
            if event.kind == "promotion_ready":
                self._consume_promotion_ready(state, event)
            elif event.kind == "source_activation_ready":
                self._consume_source_activation_ready(state, event)
            elif event.kind == "window_ready":
                self._consume_window_ready(state, event)
            elif event.kind == "store_failed":
                self._consume_store_failed(state, event)
            elif event.kind == "import_ready":
                self._consume_import_ready(state, event)
            elif event.kind == "export_ready":
                self._consume_export_ready(state, event)
            elif event.kind == "preemption_quiesce_ready":
                self._consume_preemption_quiesce_ready(state, event)
            elif event.kind == "recovery_ready":
                self._consume_recovery_ready(state, event)
            elif event.kind == "source_revoked":
                self._consume_source_revoked(state, event)
        except DSAOperationError:
            # Fail closed: a mismatched/unknown operation or conflicting late
            # receipt is an engine-fatal invariant violation.  The Scheduler
            # wraps consume_event and translates raised DSAOperationError into
            # its fatal-execution policy.
            raise

    def _consume_promotion_ready(
        self, state: DSARequestState, event: DSAControlEvent
    ) -> None:
        if state.route_state != DSARouteState.PROMOTING:
            raise DSAOperationError(
                f"promotion_ready in unexpected state {state.route_state}"
            )
        if state.promotion_inflight is None:
            raise DSAOperationError("promotion_ready with no promotion_inflight")
        if state.promotion_inflight.operation_id != event.operation_id:
            raise DSAOperationError("promotion_ready operation mismatch")

        source_end = validate_monotonic_raw_source_coverage(
            event.raw_source_end,
            previous=state.latest_sealed_raw_source_end,
        )
        sparse_source = validate_aligned_sparse_coverage(
            event.sparse_source_end,
            source_end,
            previous=state.latest_sealed_sparse_source_end,
            chunk_size=self.config.chunk_size,
        )
        remap = safe_remap_end(
            sparse_source_end=sparse_source,
            completed_canonical_end=state.completed_canonical_end,
            initial_prefill_complete=state.initial_prefill_complete,
            scratch_capacity=self.config.scratch_capacity,
            block_size=self.config.block_size,
            window_anchor=state.window_anchor,
            window_size=self.config.window_size,
        )

        state.promotion_inflight = None
        if event.output_generation_id is not None:
            state.latest_sealed_generation_id = event.output_generation_id
        state.latest_sealed_receipt_bundle_id = event.receipt_bundle_id
        state.latest_sealed_raw_source_end = source_end
        state.latest_sealed_sparse_source_end = sparse_source

        # Equality with scratch is structurally legal but reclaims no
        # post-scratch block; keep promoting until a later rolling goal
        # creates actual benefit.
        if remap <= self.config.scratch_capacity:
            return

        self._begin_source_activation(
            state,
            candidate_generation_id=(
                event.output_generation_id
                or state.latest_sealed_generation_id
                or ""
            ),
            candidate_bundle_id=event.receipt_bundle_id,
            proposed_remap_end=remap,
            transition_kind="promotion",
        )

    def _consume_source_activation_ready(
        self, state: DSARequestState, event: DSAControlEvent
    ) -> None:
        if state.source_activation_inflight is None:
            raise DSAOperationError(
                "source_activation_ready with no source_activation_inflight"
            )
        if state.source_activation_inflight.operation_id != event.operation_id:
            raise DSAOperationError("source_activation_ready operation mismatch")

        next_epoch = state.route_epoch + 1
        proposed_remap = safe_remap_end(
            sparse_source_end=state.latest_sealed_sparse_source_end,
            completed_canonical_end=state.completed_canonical_end,
            initial_prefill_complete=state.initial_prefill_complete,
            scratch_capacity=self.config.scratch_capacity,
            block_size=self.config.block_size,
            window_anchor=state.window_anchor,
            window_size=self.config.window_size,
        )
        # No-fail commit section: activation succeeded, so install the new
        # active generation + remap/release + route epoch atomically.  The
        # block-table release is delegated to the kv_cache_manager via the
        # release transaction the Scheduler commits.
        state.source_activation_inflight = None
        if event.output_generation_id is not None:
            state.active_source_generation_id = event.output_generation_id
        state.active_source_receipt_bundle_id = event.receipt_bundle_id
        state.remap_end = proposed_remap
        # release_end is set by the Scheduler after committing the release
        # transaction (it may be <= remap).
        state.route_epoch = next_epoch

        if state.route_state == DSARouteState.PROMOTING:
            # First transition to SPARSE: initialize window cursor.
            previous_state = state.route_state
            state.route_state = DSARouteState.SPARSE
            state.window_anchor = state.latest_sealed_sparse_source_end
            state.next_window_start = state.latest_sealed_sparse_source_end
            self._emit(
                state,
                previous_state=previous_state,
                reason="source_activation_ready",
            )

    def _consume_window_ready(
        self, state: DSARequestState, event: DSAControlEvent
    ) -> None:
        if state.route_state != DSARouteState.SPARSE:
            raise DSAOperationError(
                f"window_ready in unexpected state {state.route_state}"
            )
        if state.window_inflight is None:
            raise DSAOperationError("window_ready with no window_inflight")
        if state.window_inflight.operation_id != event.operation_id:
            raise DSAOperationError("window_ready operation mismatch")

        new_source = validate_monotonic_raw_source_coverage(
            event.raw_source_end,
            previous=max(
                state.raw_source_end, state.latest_sealed_raw_source_end
            ),
        )
        new_sparse_source = validate_aligned_sparse_coverage(
            event.sparse_source_end,
            new_source,
            previous=max(
                state.sparse_source_end,
                state.latest_sealed_sparse_source_end,
            ),
            chunk_size=self.config.chunk_size,
        )
        new_remap = safe_remap_end(
            sparse_source_end=new_sparse_source,
            completed_canonical_end=state.completed_canonical_end,
            initial_prefill_complete=True,
            scratch_capacity=self.config.scratch_capacity,
            block_size=self.config.block_size,
            window_anchor=state.window_anchor,
            window_size=self.config.window_size,
        )
        state.window_inflight = None
        if event.output_generation_id is not None:
            state.latest_sealed_generation_id = event.output_generation_id
        state.latest_sealed_receipt_bundle_id = event.receipt_bundle_id
        state.latest_sealed_raw_source_end = new_source
        state.latest_sealed_sparse_source_end = new_sparse_source
        self._begin_source_activation(
            state,
            candidate_generation_id=(
                event.output_generation_id
                or state.latest_sealed_generation_id
                or ""
            ),
            candidate_bundle_id=event.receipt_bundle_id,
            proposed_remap_end=new_remap,
            transition_kind="window",
            next_window_start=event.range_end,
        )

    def _consume_store_failed(
        self, state: DSARequestState, event: DSAControlEvent
    ) -> None:
        # Fail the matching lane and roll back per obligation (design 9.5).
        record = self.operations.fail_operation(
            state.request_key, event.operation_id, event.error_code or "store_failed"
        )
        for obl in record.command.operation.obligations:
            if obl == "promotion":
                previous_state = state.route_state
                state.promotion_inflight = None
                state.route_state = DSARouteState.FALLBACK_RESIDENT
                state.fallback_reason = event.error_code or "promotion_failed"
                self._emit(
                    state,
                    previous_state=previous_state,
                    reason=state.fallback_reason,
                )
            elif obl == "window":
                state.window_inflight = None
                # Keep old active source/frontier/cursor; do not regress.
            elif obl == "source_activation":
                state.source_activation_inflight = None
                # Keep pre-activation status/source.

    def _consume_import_ready(
        self, state: DSARequestState, event: DSAControlEvent
    ) -> None:
        if state.import_inflight is None:
            raise DSAOperationError("import_ready with no import_inflight")
        if state.import_inflight.operation_id != event.operation_id:
            raise DSAOperationError("import_ready operation mismatch")
        imported_end = event.sparse_source_end or event.raw_source_end
        external_end = max(0, imported_end - 1)
        state.import_inflight = None
        state.initial_prefill_complete = True
        state.completed_canonical_end = max(
            state.completed_canonical_end, imported_end
        )
        state.external_computed_end = max(
            state.external_computed_end, external_end
        )
        state.latest_sealed_raw_source_end = max(
            state.latest_sealed_raw_source_end, imported_end
        )
        state.latest_sealed_sparse_source_end = max(
            state.latest_sealed_sparse_source_end,
            align_down(imported_end, self.config.chunk_size),
        )
        if state.route_state == DSARouteState.PROMOTING:
            remap = safe_remap_end(
                sparse_source_end=state.latest_sealed_sparse_source_end,
                completed_canonical_end=state.completed_canonical_end,
                initial_prefill_complete=True,
                scratch_capacity=self.config.scratch_capacity,
                block_size=self.config.block_size,
                window_anchor=state.window_anchor,
                window_size=self.config.window_size,
            )
            if remap > self.config.scratch_capacity:
                self._begin_source_activation(
                    state,
                    candidate_generation_id=(
                        event.output_generation_id
                        or state.latest_sealed_generation_id
                        or ""
                    ),
                    candidate_bundle_id=event.receipt_bundle_id,
                    proposed_remap_end=remap,
                    transition_kind="promotion",
                )

    def _consume_export_ready(
        self, state: DSARequestState, event: DSAControlEvent
    ) -> None:
        if state.export_inflight is None:
            raise DSAOperationError("export_ready with no export_inflight")
        if state.export_inflight.operation_id != event.operation_id:
            raise DSAOperationError("export_ready operation mismatch")
        state.export_inflight = None

    def _consume_preemption_quiesce_ready(
        self, state: DSARequestState, event: DSAControlEvent
    ) -> None:
        if state.preemption_inflight is None:
            raise DSAOperationError(
                "preemption_quiesce_ready with no preemption_inflight"
            )
        state.preemption_inflight = None
        state.route_state = DSARouteState.FINISHED

    def _consume_recovery_ready(
        self, state: DSARequestState, event: DSAControlEvent
    ) -> None:
        if state.recovery_inflight is None:
            raise DSAOperationError("recovery_ready with no recovery_inflight")
        state.recovery_inflight = None
        # After dense restore, the request re-enters PROMOTING.
        previous_state = state.route_state
        state.route_state = DSARouteState.PROMOTING
        state.route_epoch += 1
        self._emit(
            state,
            previous_state=previous_state,
            reason="recovery_ready",
        )

    def _consume_source_revoked(
        self, state: DSARequestState, event: DSAControlEvent
    ) -> None:
        # If revoked before any forward, enter recovery waiting; first version
        # disables async so the drain condition is executable.
        previous_state = state.route_state
        state.route_state = DSARouteState.RECOVERING
        state.route_epoch += 1
        self._emit(
            state,
            previous_state=previous_state,
            reason="source_revoked",
        )

    # ------------------------------------------------------------------
    # Snapshot / lifecycle
    # ------------------------------------------------------------------

    def next_execution_seq(self) -> int:
        self._execution_seq += 1
        return self._execution_seq

    def build_route_snapshots(
        self, request_ids: list[str], accepted_ends: dict[str, int]
    ) -> dict[str, DSARouteSnapshot]:
        """Build the immutable ``dsa_routes`` map for the SchedulerOutput."""
        snapshots: dict[str, DSARouteSnapshot] = {}
        for rid in request_ids:
            state = self._lookup_state(rid)
            if state is None:
                continue
            lease = None
            if state.route_state == DSARouteState.SPARSE:
                lease = DSASourceLease(
                    source_lease_id=(
                        f"lease-{state.request_key}-"
                        f"{self.next_execution_seq()}"
                    ),
                    request_key=state.request_key,
                    execution_seq=self._execution_seq,
                    route_epoch=state.route_epoch,
                    source_generation_id=(
                        state.active_source_generation_id or ""
                    ),
                )
            snapshots[rid] = build_snapshot(
                state,
                execution_seq=self._execution_seq,
                accepted_end=accepted_ends.get(rid, state.completed_canonical_end),
                source_lease=lease,
            )
        return snapshots

    def begin_preemption(self, request_id: str) -> Optional[DSAOperationRef]:
        """Begin a typed preemption quiesce for the request.

        Returns the quiesce operation ref (to be carried in a
        DSAPreemptionAction), or None if the request has no DSA state.
        """
        state = self._lookup_state(request_id)
        if state is None:
            return None
        op_id = self._new_operation_id("preemption_quiesce")
        ref = DSAOperationRef(
            operation_id=op_id,
            parent_operation_id=None,
            kind="preemption_quiesce",
            obligations=frozenset({"preemption_quiesce"}),
            range_start=0,
            range_end=state.completed_canonical_end,
            input_generation_id=state.active_source_generation_id,
            output_generation_id=None,
            route_epoch=state.route_epoch,
        )
        state.preemption_inflight = ref
        return ref

    def finish_request(self, request_id: str) -> None:
        """Mark a request finished and drop its controller-side mappings.

        The kv_cache_manager free is performed by the Scheduler; here we only
        retire the DSA state so late events are rejected as stale.
        """
        self._id_to_key.pop(request_id, None)
        state = self._states.pop(request_id, None)
        if state is not None:
            state.route_state = DSARouteState.FINISHED

    def waiting_reason(self, request_id: str) -> Optional[str]:
        state = self._lookup_state(request_id)
        if state is None:
            return None
        if state.route_state == DSARouteState.RECOVERING:
            return WAITING_FOR_DSA_RECOVERY
        if state.preemption_inflight is not None:
            return WAITING_FOR_PREEMPTION_QUIESCE
        if state.recovery_inflight is not None:
            return WAITING_FOR_DSA_RECOVERY
        if state.source_activation_inflight is not None:
            return WAITING_FOR_SOURCE_ACTIVATION
        if state.import_inflight is not None:
            return WAITING_FOR_PD_IMPORT
        if state.export_inflight is not None:
            return WAITING_FOR_PD_EXPORT
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _new_operation_id(self, kind: str) -> str:
        return f"{kind}-{self.process_instance_id}-{secrets.token_hex(6)}"
