# SPDX-License-Identifier: Apache-2.0
"""Scheduler-owned DSA command, receipt, activation, and release lifecycle."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass, field

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
    WAITING_FOR_SOURCE_ACTIVATION,
    DSAControlEvent,
    DSAExecutionExpectation,
    DSAExecutionReceipt,
    DSALatentReleasePlan,
    DSAOperationCommand,
    DSAOperationReceipt,
    DSAOperationRecord,
    DSAOperationRef,
    DSAReceiptBundle,
    DSAReceiptExpectation,
    DSARequestState,
    DSARouteSnapshot,
    DSARouteState,
    DSASourceLease,
    DSATransferPlan,
    ParticipantIdentity,
    RequestKey,
    build_snapshot,
    derive_transfer_plan_for_admission,
    derive_transfer_plan_for_promotion,
)

logger = init_logger(__name__)


def _default_participant() -> ParticipantIdentity:
    return ParticipantIdentity(
        engine_id="engine",
        process_instance_id="worker-0",
        worker_rank=0,
        dp_rank=0,
        pp_rank=0,
        tp_rank=0,
    )


@dataclass
class DSAControllerConfig:
    """Normalized DSA controller and exact-receipt configuration."""

    threshold: int
    max_model_len: int
    block_size: int
    chunk_size: int
    window_size: int
    index_topk: int
    query_width: int
    scratch_capacity: int
    node_role: str
    deployment_mode: str
    data_compatibility_fingerprint: str
    instance_capability_digest: str
    participants: tuple[ParticipantIdentity, ...] = field(
        default_factory=lambda: (_default_participant(),)
    )
    # ``participant_ids`` accepts the capability descriptor's field name. If
    # present it is authoritative over the test-friendly ``participants``.
    participant_ids: tuple[ParticipantIdentity, ...] = ()
    required_layers: tuple[int, ...] = (0,)
    kv_groups: tuple[int, ...] = (0,)
    operation_timeout_ms: int = 30_000

    def __post_init__(self) -> None:
        participants = self.receipt_participants
        if not participants:
            raise ValueError("DSA requires at least one receipt participant")
        if len(set(participants)) != len(participants):
            raise ValueError("DSA receipt participants must be unique")
        if not self.required_layers or any(
            type(layer) is not int or layer < 0 for layer in self.required_layers
        ):
            raise ValueError("DSA required_layers must be non-empty and non-negative")
        if len(set(self.required_layers)) != len(self.required_layers):
            raise ValueError("DSA required_layers must be unique")
        if not self.kv_groups or any(
            type(group) is not int or group < 0 for group in self.kv_groups
        ):
            raise ValueError("DSA kv_groups must be non-empty and non-negative")
        if len(set(self.kv_groups)) != len(self.kv_groups):
            raise ValueError("DSA kv_groups must be unique")
        if self.operation_timeout_ms <= 0:
            raise ValueError("DSA operation_timeout_ms must be positive")
        if self.enabled:
            if self.block_size <= 0 or self.chunk_size <= 0:
                raise ValueError("positive-threshold DSA requires block/chunk sizes")
            if self.chunk_size % self.block_size != 0:
                raise ValueError("DSA chunk_size must be block-aligned")
            if self.window_size < 0 or (
                self.window_size > 0
                and self.window_size % self.chunk_size != 0
            ):
                raise ValueError(
                    "DSA window_size must be zero or chunk-aligned"
                )
            if self.scratch_capacity < 0:
                raise ValueError("DSA scratch_capacity must be non-negative")
            if not self.data_compatibility_fingerprint:
                raise ValueError(
                    "positive-threshold DSA requires a cache namespace fingerprint"
                )

    @property
    def receipt_participants(self) -> tuple[ParticipantIdentity, ...]:
        return self.participant_ids or self.participants

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
        deployment_mode: str = "standalone",
    ) -> DSAControllerConfig:
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
            deployment_mode=deployment_mode,
            data_compatibility_fingerprint="",
            instance_capability_digest="",
        )


@dataclass
class DSAController:
    """Own all mutable DSA request state and registered operations."""

    config: DSAControllerConfig
    process_instance_id: str = field(default_factory=lambda: secrets.token_hex(8))
    operations: DSAOperationRegistry = field(default_factory=DSAOperationRegistry)
    _scope_counter: int = 0
    _execution_seq: int = 0
    _operation_seq: int = 0
    _generation_seq: int = 0
    _event_seq: int = field(default=0, init=False, repr=False)
    _id_to_key: dict[str, RequestKey] = field(default_factory=dict)
    _states: dict[str, DSARequestState] = field(default_factory=dict)
    _command_outbox: list[DSAOperationCommand] = field(default_factory=list)
    _pending_release_plans: dict[
        tuple[RequestKey, str], DSALatentReleasePlan
    ] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Identity and request admission
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
        transfer_plan: DSATransferPlan | None = None,
    ) -> DSARequestState:
        if num_tokens < 0:
            raise ValueError("DSA request token count must be non-negative")
        key = self._new_request_key(request_id)
        plan = transfer_plan or derive_transfer_plan_for_admission(
            self.config.deployment_mode,
            self.config.node_role,
        )

        if not self.config.enabled:
            route_state = DSARouteState.LEGACY
            crossed_at = None
            reason = "threshold_disabled"
        elif num_tokens >= self.config.threshold:
            route_state = DSARouteState.PROMOTING
            crossed_at = num_tokens
            reason = "at_or_above_threshold"
            plan = derive_transfer_plan_for_promotion(
                previous=plan,
                deployment_role=self.config.node_role,
            )
        else:
            route_state = DSARouteState.RESIDENT
            crossed_at = None
            reason = "below_threshold"

        state = DSARequestState(
            request_key=key,
            route_state=route_state,
            route_epoch=0,
            threshold=self.config.threshold,
            initial_prompt_end=num_tokens,
            accepted_end=num_tokens,
            threshold_crossed_at=crossed_at,
            transfer_plan=plan,
        )
        self._emit(
            state,
            previous_state=None,
            reason=reason,
            accepted_end=num_tokens,
        )
        return state

    def attach_state(self, state: DSARequestState) -> None:
        request_id = state.request_key.request_id
        if self._id_to_key.get(request_id) != state.request_key:
            raise DSAOperationError("cannot attach a stale DSA request state")
        self._states[request_id] = state

    def _lookup_state(self, request_id: str) -> DSARequestState | None:
        return self._states.get(request_id)

    def _state_for_key(self, request_key: RequestKey) -> DSARequestState:
        state = self._states.get(request_key.request_id)
        if state is None or state.request_key != request_key:
            raise DSAOperationError(f"unknown or stale DSA request key {request_key}")
        return state

    # ------------------------------------------------------------------
    # Model execution receipt protocol
    # ------------------------------------------------------------------

    @staticmethod
    def compute_token_prefix_digest(
        token_ids: list[int], accepted_end: int
    ) -> str:
        """Compute the canonical digest used by snapshots and commands."""
        if accepted_end < 0 or accepted_end > len(token_ids):
            raise DSAOperationError(
                f"accepted token end {accepted_end} is outside token history "
                f"of length {len(token_ids)}"
            )
        digest = hashlib.sha256()
        digest.update(accepted_end.to_bytes(8, byteorder="big", signed=False))
        for token_id in token_ids[:accepted_end]:
            if type(token_id) is not int:
                raise DSAOperationError("token prefix contains a non-integer token")
            digest.update(token_id.to_bytes(8, byteorder="big", signed=True))
        return digest.hexdigest()

    @staticmethod
    def _validate_frontier(value: int, label: str) -> None:
        if type(value) is not int or value < 0:
            raise DSAOperationError(f"{label} must be a non-negative integer")

    def consume_completed_execution(
        self,
        request_id: str,
        receipt: DSAExecutionReceipt,
        expected_token_prefix_digest: str | None = None,
    ) -> None:
        """Validate and consume one worker-confirmed model execution."""
        state = self._lookup_state(request_id)
        if state is None:
            raise DSAOperationError(
                f"execution receipt for unknown request {request_id}"
            )
        if receipt.request_key != state.request_key:
            raise DSAOperationError("execution receipt request key mismatch")

        for value, label in (
            (receipt.execution_seq, "execution_seq"),
            (receipt.route_epoch, "route_epoch"),
            (receipt.accepted_end_at_execution, "accepted_end_at_execution"),
            (receipt.completed_canonical_end, "completed_canonical_end"),
            (receipt.external_computed_end, "external_computed_end"),
            (
                receipt.min_position_of_next_target_rows,
                "min_position_of_next_target_rows",
            ),
        ):
            self._validate_frontier(value, label)
        if type(receipt.initial_prefill_complete) is not bool:
            raise DSAOperationError("initial_prefill_complete must be a bool")
        if receipt.execution_seq <= state.last_completed_execution_seq:
            raise DSAOperationError(
                "execution receipt sequence did not advance monotonically"
            )

        expectation = state.pending_executions.get(receipt.execution_seq)
        if expectation is None:
            raise DSAOperationError(
                f"execution receipt sequence {receipt.execution_seq} was not issued"
            )
        earlier_pending = [
            seq for seq in state.pending_executions if seq < receipt.execution_seq
        ]
        if earlier_pending:
            raise DSAOperationError(
                "execution receipt arrived ahead of an earlier issued execution"
            )
        if receipt.route_epoch != expectation.route_epoch:
            raise DSAOperationError("execution receipt route epoch mismatch")
        if (
            receipt.accepted_end_at_execution
            != expectation.accepted_end_at_execution
        ):
            raise DSAOperationError("execution receipt accepted frontier mismatch")
        if expected_token_prefix_digest is None:
            expected_token_prefix_digest = expectation.token_prefix_digest
        if not expected_token_prefix_digest:
            raise DSAOperationError("expected execution token digest is empty")
        if receipt.token_prefix_digest != expected_token_prefix_digest:
            raise DSAOperationError("execution receipt token digest mismatch")
        if receipt.released_source_lease_id != expectation.source_lease_id:
            raise DSAOperationError("execution receipt source lease mismatch")
        if receipt.completed_canonical_end < state.completed_canonical_end:
            raise DSAOperationError("completed canonical frontier regressed")
        if (
            receipt.completed_canonical_end
            > expectation.maximum_completed_canonical_end
        ):
            raise DSAOperationError(
                "completed canonical frontier exceeds scheduled execution bound"
            )
        if receipt.completed_canonical_end > state.accepted_end:
            raise DSAOperationError(
                "completed canonical frontier exceeds accepted frontier"
            )
        if receipt.external_computed_end < state.external_computed_end:
            raise DSAOperationError("external computed frontier regressed")
        if receipt.external_computed_end > receipt.completed_canonical_end:
            raise DSAOperationError(
                "external computed frontier exceeds canonical completion"
            )
        if state.initial_prefill_complete and not receipt.initial_prefill_complete:
            raise DSAOperationError("initial prefill completion regressed")
        if (
            receipt.initial_prefill_complete
            and receipt.completed_canonical_end < state.initial_prompt_end
        ):
            raise DSAOperationError(
                "initial prefill completed before the initial prompt frontier"
            )
        if (
            receipt.completed_canonical_end >= state.initial_prompt_end
            and not receipt.initial_prefill_complete
        ):
            raise DSAOperationError(
                "initial prefill completion is missing at the prompt frontier"
            )
        if (
            receipt.min_position_of_next_target_rows
            > receipt.completed_canonical_end
        ):
            raise DSAOperationError(
                "next target-row position exceeds canonical completion"
            )
        if receipt.min_position_of_next_target_rows < state.remap_end:
            raise DSAOperationError(
                "next target-row position is behind the active remap frontier"
            )

        state.completed_canonical_end = receipt.completed_canonical_end
        state.external_computed_end = receipt.external_computed_end
        state.initial_prefill_complete = receipt.initial_prefill_complete
        state.min_position_of_next_target_rows = (
            receipt.min_position_of_next_target_rows
        )
        state.last_completed_execution_seq = receipt.execution_seq
        state.pending_executions.pop(receipt.execution_seq)
        if state.route_state == DSARouteState.PROMOTING:
            state.promotion_desired_end = max(
                state.promotion_desired_end,
                align_down(state.completed_canonical_end, self.config.chunk_size),
            )

    def consume_accepted_end(
        self,
        request_id: str,
        accepted_end: int,
        token_prefix_digest: str | None = None,
    ) -> None:
        """Consume accepted A separately from the execution C/E receipt."""
        state = self._lookup_state(request_id)
        if state is None:
            return
        self._validate_frontier(accepted_end, "accepted_end")
        if accepted_end < state.accepted_end:
            raise DSAOperationError("accepted frontier regressed")
        if token_prefix_digest is not None and not token_prefix_digest:
            raise DSAOperationError("accepted token prefix digest is empty")
        state.accepted_end = accepted_end
        if token_prefix_digest is not None:
            state.token_prefix_digest = token_prefix_digest

        if (
            state.route_state == DSARouteState.RESIDENT
            and self.config.enabled
            and accepted_end >= state.threshold
        ):
            previous_state = state.route_state
            state.transfer_plan = derive_transfer_plan_for_promotion(
                previous=state.transfer_plan,
                deployment_role=self.config.node_role,
            )
            state.route_state = DSARouteState.PROMOTING
            state.threshold_crossed_at = accepted_end
            state.promotion_desired_end = max(
                state.promotion_desired_end,
                align_down(state.completed_canonical_end, self.config.chunk_size),
            )
            state.route_epoch += 1
            self._emit(
                state,
                previous_state=previous_state,
                reason="context_threshold_crossed",
                accepted_end=accepted_end,
            )

    # ------------------------------------------------------------------
    # Registered command construction and advancement
    # ------------------------------------------------------------------

    def _new_operation_id(self, state: DSARequestState, kind: str) -> str:
        self._operation_seq += 1
        return (
            f"{kind}-{self.process_instance_id}-"
            f"{state.request_key.scope_id}-{self._operation_seq}"
        )

    def _new_generation_id(self, state: DSARequestState) -> str:
        self._generation_seq += 1
        return (
            f"generation-{self.process_instance_id}-"
            f"{state.request_key.scope_id}-{self._generation_seq}"
        )

    def _chunks_for_range(
        self, range_start: int, range_end: int
    ) -> tuple[tuple[int, int], ...]:
        chunks: list[tuple[int, int]] = []
        start = range_start
        while start < range_end:
            end = min(start + self.config.chunk_size, range_end)
            chunks.append((start, end))
            start = end
        return tuple(chunks)

    def _build_expectations(
        self, operation: DSAOperationRef
    ) -> tuple[DSAReceiptExpectation, ...]:
        chunks = self._chunks_for_range(
            operation.range_start,
            operation.range_end,
        )
        if operation.kind == "store":
            receipt_specs = (
                ("storage", "local_cpu", "local_cpu_pinned"),
                ("source_seal", "local_cpu", "local_cpu_pinned"),
            )
        elif operation.kind == "source_activation":
            receipt_specs = (("route_epoch_complete", "npu", "npu_materialized"),)
        else:
            raise DSAOperationError(
                f"unsupported command kind for focused DSA flow: {operation.kind}"
            )

        expectations: list[DSAReceiptExpectation] = []
        for participant in self.config.receipt_participants:
            for kv_group in self.config.kv_groups:
                for receipt_kind, storage_tier, guarantee in receipt_specs:
                    expectations.append(
                        DSAReceiptExpectation(
                            participant=participant,
                            receipt_kind=receipt_kind,  # type: ignore[arg-type]
                            kv_group=kv_group,
                            layers=self.config.required_layers,
                            chunks=chunks,
                            storage_tier=storage_tier,  # type: ignore[arg-type]
                            minimum_guarantee=guarantee,  # type: ignore[arg-type]
                        )
                    )
        return tuple(expectations)

    def build_command(
        self,
        state: DSARequestState,
        operation: DSAOperationRef,
        *,
        accepted_end_at_issue: int,
        token_prefix_digest: str,
    ) -> DSAOperationCommand:
        """Build a command whose expectations are fully config-derived."""
        if not token_prefix_digest:
            raise DSAOperationError("cannot issue a DSA command without a token digest")
        return DSAOperationCommand(
            request_key=state.request_key,
            operation=operation,
            accepted_end_at_issue=accepted_end_at_issue,
            token_prefix_digest=token_prefix_digest,
            cache_namespace_fingerprint=(
                self.config.data_compatibility_fingerprint
            ),
            expected_receipts=self._build_expectations(operation),
        )

    def _register_and_publish(self, command: DSAOperationCommand) -> None:
        deadline_ms = int(time.time() * 1000) + self.config.operation_timeout_ms
        # Registration must happen before a worker can observe the command.
        self.operations.register(command, deadline_ms=deadline_ms)
        self._command_outbox.append(command)

    def take_pending_commands(self) -> tuple[DSAOperationCommand, ...]:
        commands = tuple(self._command_outbox)
        self._command_outbox.clear()
        return commands

    def maybe_advance(self, request_id: str) -> None:
        state = self._lookup_state(request_id)
        if (
            state is None
            or not self.config.enabled
            or state.route_state
            in (
                DSARouteState.LEGACY,
                DSARouteState.RESIDENT,
                DSARouteState.FALLBACK_RESIDENT,
                DSARouteState.RECOVERING,
                DSARouteState.FINISHED,
            )
            or not state.transfer_plan.must_persist_decode_windows
        ):
            return
        if state.source_activation_inflight is not None:
            return
        # A sealed candidate gets first chance to activate. Do not race it with
        # a new store merely because C advanced while the candidate was delayed.
        if self._maybe_activate_latest_candidate(state):
            return

        desired_end = align_down(
            state.completed_canonical_end,
            self.config.chunk_size,
        )
        if state.route_state == DSARouteState.PROMOTING:
            state.promotion_desired_end = max(
                state.promotion_desired_end,
                desired_end,
            )
            if state.promotion_inflight is not None:
                return
            target_end = state.promotion_desired_end
            obligation = "promotion"
        elif state.route_state == DSARouteState.SPARSE:
            if state.window_inflight is not None or self.config.window_size <= 0:
                return
            target_end = state.next_window_start + self.config.window_size
            if desired_end < target_end:
                return
            if (
                target_end - state.window_anchor
            ) % self.config.window_size != 0:
                raise DSAOperationError("DSA window cursor is not anchor-aligned")
            obligation = "window"
        else:
            return

        if target_end <= state.latest_sealed_sparse_source_end:
            return
        save_start = align_down(
            state.latest_sealed_raw_source_end,
            self.config.chunk_size,
        )
        if save_start >= target_end:
            raise DSAOperationError("DSA store range is empty or regressive")
        self._issue_store(state, obligation, save_start, target_end)

    def _issue_store(
        self,
        state: DSARequestState,
        obligation: str,
        range_start: int,
        range_end: int,
    ) -> None:
        if state.token_prefix_digest is None:
            raise DSAOperationError("no accepted token digest is available for store")
        operation = DSAOperationRef(
            operation_id=self._new_operation_id(state, obligation),
            parent_operation_id=None,
            kind="store",
            obligations=frozenset({obligation}),  # type: ignore[arg-type]
            range_start=range_start,
            range_end=range_end,
            input_generation_id=state.latest_sealed_generation_id,
            output_generation_id=self._new_generation_id(state),
            route_epoch=state.route_epoch,
        )
        command = self.build_command(
            state,
            operation,
            accepted_end_at_issue=state.accepted_end,
            token_prefix_digest=state.token_prefix_digest,
        )
        self._register_and_publish(command)
        if obligation == "promotion":
            state.promotion_inflight = operation
        elif obligation == "window":
            state.window_inflight = operation
        else:
            raise AssertionError(f"unexpected store obligation {obligation}")

    def _candidate_remap_end(self, state: DSARequestState) -> int:
        return safe_remap_end(
            sparse_source_end=state.latest_sealed_sparse_source_end,
            completed_canonical_end=state.completed_canonical_end,
            initial_prefill_complete=state.initial_prefill_complete,
            scratch_capacity=self.config.scratch_capacity,
            block_size=self.config.block_size,
            window_anchor=state.window_anchor,
            window_size=self.config.window_size,
            min_position_of_next_target_rows=(
                state.min_position_of_next_target_rows
            ),
        )

    def _maybe_activate_latest_candidate(self, state: DSARequestState) -> bool:
        if (
            state.promotion_inflight is not None
            or state.window_inflight is not None
            or bool(state.pending_executions)
            or not state.initial_prefill_complete
            or state.latest_sealed_generation_id is None
            or state.latest_sealed_operation_id is None
            or state.latest_sealed_receipt_bundle_id is None
            or state.latest_sealed_token_prefix_digest is None
        ):
            return False
        proposed_remap_end = self._candidate_remap_end(state)
        if proposed_remap_end <= state.remap_end:
            return False

        parent_key = (state.request_key, state.latest_sealed_operation_id)
        parent = self.operations.records.get(parent_key)
        if parent is None or parent.status != "ready":
            return False
        obligations = parent.command.operation.obligations
        if obligations == frozenset({"promotion"}):
            transition_kind = "promotion"
        elif obligations == frozenset({"window"}):
            transition_kind = "window"
        else:
            raise DSAOperationError(
                "sealed candidate has unsupported activation obligations"
            )
        self._begin_source_activation(
            state,
            parent,
            transition_kind=transition_kind,
            proposed_remap_end=proposed_remap_end,
        )
        return True

    def _begin_source_activation(
        self,
        state: DSARequestState,
        parent: DSAOperationRecord,
        *,
        transition_kind: str,
        proposed_remap_end: int,
    ) -> None:
        if state.source_activation_inflight is not None:
            raise DSAOperationError("source activation lane is already occupied")
        parent_operation = parent.command.operation
        generation_id = state.latest_sealed_generation_id
        digest = state.latest_sealed_token_prefix_digest
        if generation_id is None or digest is None:
            raise DSAOperationError("source activation candidate is incomplete")
        operation = DSAOperationRef(
            operation_id=self._new_operation_id(state, "source_activation"),
            parent_operation_id=parent_operation.operation_id,
            kind="source_activation",
            obligations=frozenset({"source_activation"}),
            range_start=0,
            range_end=proposed_remap_end,
            input_generation_id=generation_id,
            output_generation_id=generation_id,
            route_epoch=state.route_epoch + 1,
        )
        command = self.build_command(
            state,
            operation,
            accepted_end_at_issue=parent.command.accepted_end_at_issue,
            token_prefix_digest=digest,
        )
        self._register_and_publish(command)
        parent.status = "activating"
        state.source_activation_inflight = operation

    # ------------------------------------------------------------------
    # Individual receipt consumption
    # ------------------------------------------------------------------

    def consume_operation_receipt(
        self, receipt: DSAOperationReceipt
    ) -> DSALatentReleasePlan | None:
        """Submit one receipt and advance only after an exact-set bundle."""
        state = self._lookup_state(receipt.request_key.request_id)
        if state is None or state.request_key != receipt.request_key:
            return None
        key = (receipt.request_key, receipt.operation_id)
        was_live = key in self.operations.records
        bundle = self.operations.submit_receipt(receipt)
        if bundle is None or not was_live:
            return None
        record = self.operations.records.get(key)
        if record is None:
            raise DSAOperationError("live DSA receipt lost its operation record")
        if not self._lane_matches(state, record.command.operation):
            return None
        if bundle.aggregate_status == "failed":
            self._rollback_failed_operation(state, record, bundle)
            return None
        if record.command.operation.kind == "store":
            self._consume_complete_store_bundle(state, record, bundle)
            return None
        if record.command.operation.kind == "source_activation":
            return self._consume_complete_activation_bundle(state, record, bundle)
        raise DSAOperationError(
            f"unsupported completed DSA operation {record.command.operation.kind}"
        )

    @staticmethod
    def _lane_matches(
        state: DSARequestState, operation: DSAOperationRef
    ) -> bool:
        """Return True when the operation still holds its focused lane.

        Unlike :meth:`_matching_lane`, this returns ``False`` instead of
        raising when the lane was cleared (bundle already consumed) or
        taken over by a newer operation.  Callers use the ``False``
        return to treat an identical-duplicate receipt as an idempotent
        no-op.
        """
        if operation.kind == "source_activation":
            lane = "source_activation_inflight"
        elif "promotion" in operation.obligations:
            lane = "promotion_inflight"
        elif "window" in operation.obligations:
            lane = "window_inflight"
        else:
            return False
        lane_operation = getattr(state, lane)
        return (
            lane_operation is not None
            and lane_operation.operation_id == operation.operation_id
        )

    @staticmethod
    def _matching_lane(
        state: DSARequestState, operation: DSAOperationRef
    ) -> str:
        if operation.kind == "source_activation":
            lane = "source_activation_inflight"
        elif "promotion" in operation.obligations:
            lane = "promotion_inflight"
        elif "window" in operation.obligations:
            lane = "window_inflight"
        else:
            raise DSAOperationError("operation does not belong to a focused DSA lane")
        lane_operation = getattr(state, lane)
        if (
            lane_operation is None
            or lane_operation.operation_id != operation.operation_id
        ):
            raise DSAOperationError(f"receipt does not match the active {lane} lane")
        return lane

    def _consume_complete_store_bundle(
        self,
        state: DSARequestState,
        record: DSAOperationRecord,
        bundle: DSAReceiptBundle,
    ) -> None:
        operation = record.command.operation
        lane = self._matching_lane(state, operation)
        self.operations.consume_ready_bundle(
            bundle.bundle_id,
            state.request_key,
            operation.operation_id,
        )
        if operation.route_epoch != state.route_epoch:
            raise DSAOperationError("store completed for a stale route epoch")
        if not operation.output_generation_id:
            raise DSAOperationError("store command has no output generation")
        if bundle.output_generation_id != operation.output_generation_id:
            raise DSAOperationError("store bundle output generation mismatch")
        if bundle.token_prefix_digest != record.command.token_prefix_digest:
            raise DSAOperationError("store bundle token digest mismatch")
        raw_source_end = validate_monotonic_raw_source_coverage(
            bundle.raw_source_end,
            previous=state.latest_sealed_raw_source_end,
        )
        sparse_source_end = validate_aligned_sparse_coverage(
            bundle.sparse_source_end,
            raw_source_end,
            previous=state.latest_sealed_sparse_source_end,
            chunk_size=self.config.chunk_size,
        )
        if raw_source_end != operation.range_end:
            raise DSAOperationError("store bundle does not cover its command range")
        if sparse_source_end > state.completed_canonical_end:
            raise DSAOperationError("sealed source exceeds canonical completion")

        previous_candidate_operation_id = state.latest_sealed_operation_id
        if (
            previous_candidate_operation_id is not None
            and previous_candidate_operation_id != operation.operation_id
            and (
                state.request_key,
                previous_candidate_operation_id,
            )
            in self.operations.records
        ):
            self.operations.supersede(
                state.request_key,
                previous_candidate_operation_id,
            )
        setattr(state, lane, None)
        state.latest_sealed_operation_id = operation.operation_id
        state.latest_sealed_generation_id = operation.output_generation_id
        state.latest_sealed_receipt_bundle_id = bundle.bundle_id
        state.latest_sealed_token_prefix_digest = bundle.token_prefix_digest
        state.latest_sealed_raw_source_end = raw_source_end
        state.latest_sealed_sparse_source_end = sparse_source_end
        self._maybe_activate_latest_candidate(state)

    def _consume_complete_activation_bundle(
        self,
        state: DSARequestState,
        record: DSAOperationRecord,
        bundle: DSAReceiptBundle,
    ) -> DSALatentReleasePlan:
        operation = record.command.operation
        self._matching_lane(state, operation)
        self.operations.consume_ready_bundle(
            bundle.bundle_id,
            state.request_key,
            operation.operation_id,
        )
        if operation.route_epoch != state.route_epoch + 1:
            raise DSAOperationError("activation does not target the next route epoch")
        if operation.parent_operation_id != state.latest_sealed_operation_id:
            raise DSAOperationError("activation parent is not the sealed candidate")
        if operation.input_generation_id != state.latest_sealed_generation_id:
            raise DSAOperationError("activation candidate generation mismatch")
        if bundle.output_generation_id != state.latest_sealed_generation_id:
            raise DSAOperationError("activation bundle generation mismatch")
        if bundle.token_prefix_digest != state.latest_sealed_token_prefix_digest:
            raise DSAOperationError("activation bundle token digest mismatch")
        if self._candidate_remap_end(state) < operation.range_end:
            raise DSAOperationError("activation frontier is no longer safe to release")

        parent_key = (state.request_key, operation.parent_operation_id)
        parent = self.operations.records.get(parent_key)
        if parent is None or parent.status != "activating":
            raise DSAOperationError("activation parent is not active")
        if "promotion" in parent.command.operation.obligations:
            transition_kind = "promotion"
            next_window_start = state.latest_sealed_sparse_source_end
        elif "window" in parent.command.operation.obligations:
            transition_kind = "window"
            next_window_start = parent.command.operation.range_end
        else:
            raise DSAOperationError("activation parent obligation is unsupported")

        plan = DSALatentReleasePlan(
            request_key=state.request_key,
            activation_operation_id=operation.operation_id,
            activation_bundle_id=bundle.bundle_id,
            store_operation_id=parent.command.operation.operation_id,
            store_bundle_id=state.latest_sealed_receipt_bundle_id or "",
            transition_kind=transition_kind,
            source_generation_id=state.latest_sealed_generation_id or "",
            token_prefix_digest=state.latest_sealed_token_prefix_digest or "",
            raw_source_end=state.latest_sealed_raw_source_end,
            sparse_source_end=state.latest_sealed_sparse_source_end,
            remap_end=operation.range_end,
            release_end=operation.range_end,
            next_route_epoch=operation.route_epoch,
            next_window_start=next_window_start,
        )
        key = (state.request_key, operation.operation_id)
        existing = self._pending_release_plans.get(key)
        if existing is not None and existing != plan:
            raise DSAOperationError("activation produced a conflicting release plan")
        self._pending_release_plans[key] = plan
        return plan

    def _rollback_failed_operation(
        self,
        state: DSARequestState,
        record: DSAOperationRecord,
        bundle: DSAReceiptBundle,
    ) -> None:
        operation = record.command.operation
        error_code = bundle.error_code or "receipt_failed"
        if operation.kind == "store":
            lane = self._matching_lane(state, operation)
            setattr(state, lane, None)
            if "promotion" in operation.obligations:
                previous_state = state.route_state
                state.route_state = DSARouteState.FALLBACK_RESIDENT
                state.route_epoch += 1
                state.fallback_reason = error_code
                self._emit(
                    state,
                    previous_state=previous_state,
                    reason=error_code,
                )
        elif operation.kind == "source_activation":
            self._matching_lane(state, operation)
            state.source_activation_inflight = None
            parent_id = operation.parent_operation_id
            if parent_id is None:
                raise DSAOperationError("failed activation has no parent")
            parent = self.operations.records.get((state.request_key, parent_id))
            if parent is None:
                raise DSAOperationError("failed activation parent is missing")
            self.operations.mark_consumer_terminal(
                state.request_key,
                parent_id,
                next(iter(parent.command.operation.obligations)),
                "failed",
            )
            if state.route_state == DSARouteState.PROMOTING:
                previous_state = state.route_state
                state.route_state = DSARouteState.FALLBACK_RESIDENT
                state.route_epoch += 1
                state.fallback_reason = error_code
                self._emit(
                    state,
                    previous_state=previous_state,
                    reason=error_code,
                )
            else:
                # A rolling activation failure must leave the old active source
                # and cursor authoritative. The next store gets a fresh ID and
                # generation rather than retrying this failed candidate.
                state.latest_sealed_operation_id = None
                state.latest_sealed_generation_id = state.active_source_generation_id
                state.latest_sealed_receipt_bundle_id = (
                    state.active_source_receipt_bundle_id
                )
                state.latest_sealed_token_prefix_digest = (
                    state.active_token_prefix_digest
                )
                state.latest_sealed_raw_source_end = state.raw_source_end
                state.latest_sealed_sparse_source_end = state.sparse_source_end
        else:
            raise DSAOperationError("unsupported failed DSA operation")
        self.operations.fail_operation(
            state.request_key,
            operation.operation_id,
            error_code,
        )

    def validate_release_plan(self, plan: DSALatentReleasePlan) -> None:
        """Validate every controller binding before the KV no-fail commit."""
        state = self._state_for_key(plan.request_key)
        key = (plan.request_key, plan.activation_operation_id)
        if self._pending_release_plans.get(key) != plan:
            raise DSAOperationError("release plan is not pending or was replaced")
        activation = state.source_activation_inflight
        if (
            activation is None
            or activation.operation_id != plan.activation_operation_id
        ):
            raise DSAOperationError("release plan no longer owns activation lane")
        if activation.parent_operation_id != plan.store_operation_id:
            raise DSAOperationError("release plan parent operation mismatch")
        if activation.route_epoch != plan.next_route_epoch:
            raise DSAOperationError("release plan route epoch mismatch")
        if activation.range_end != plan.remap_end or plan.release_end != plan.remap_end:
            raise DSAOperationError("release plan frontier mismatch")
        if plan.remap_end <= state.remap_end:
            raise DSAOperationError("release plan does not advance remap frontier")
        if plan.source_generation_id != state.latest_sealed_generation_id:
            raise DSAOperationError("release plan source generation is stale")
        if plan.store_bundle_id != state.latest_sealed_receipt_bundle_id:
            raise DSAOperationError("release plan store bundle is stale")
        if plan.token_prefix_digest != state.latest_sealed_token_prefix_digest:
            raise DSAOperationError("release plan token digest is stale")
        activation_record = self.operations.consume_ready_bundle(
            plan.activation_bundle_id,
            plan.request_key,
            plan.activation_operation_id,
        )
        parent = self.operations.records.get(
            (plan.request_key, plan.store_operation_id)
        )
        if parent is None or parent.status != "activating":
            raise DSAOperationError("release plan parent is not activating")
        if activation_record.status != "ready":
            raise DSAOperationError("release plan activation is not ready")

    def commit_release_plan(
        self,
        plan: DSALatentReleasePlan,
        committed_release_end: int,
    ) -> None:
        """Atomically install a released generation after KV commit."""
        self.validate_release_plan(plan)
        if committed_release_end != plan.release_end:
            raise DSAOperationError(
                "KV transaction committed a different release frontier"
            )
        state = self._state_for_key(plan.request_key)
        previous_state = state.route_state

        # Both records were fully validated above. These terminal transitions
        # are non-failing under the Scheduler's single-owner execution model.
        self.operations.mark_consumer_terminal(
            plan.request_key,
            plan.store_operation_id,
            plan.transition_kind,
            "committed",
        )
        self.operations.mark_consumer_terminal(
            plan.request_key,
            plan.activation_operation_id,
            "source_activation",
            "committed",
        )

        state.active_source_generation_id = plan.source_generation_id
        state.active_source_receipt_bundle_id = plan.store_bundle_id
        state.active_token_prefix_digest = plan.token_prefix_digest
        state.raw_source_end = plan.raw_source_end
        state.sparse_source_end = plan.sparse_source_end
        state.remap_end = plan.remap_end
        state.release_end = committed_release_end
        state.route_epoch = plan.next_route_epoch
        state.route_state = DSARouteState.SPARSE
        state.source_activation_inflight = None
        if plan.transition_kind == "promotion":
            state.window_anchor = plan.sparse_source_end
        state.next_window_start = plan.next_window_start
        self._pending_release_plans.pop(
            (plan.request_key, plan.activation_operation_id),
            None,
        )
        self._emit(
            state,
            previous_state=previous_state,
            reason="source_activation_committed",
        )

    # ------------------------------------------------------------------
    # Worker control events: revocation is the only unsolicited event.
    # ------------------------------------------------------------------

    def consume_event(self, request_id: str, event: DSAControlEvent) -> None:
        if event.kind != "source_revoked":
            raise DSAOperationError(
                f"worker-produced DSA ready event {event.kind!r} is forbidden"
            )
        state = self._lookup_state(request_id)
        if state is None:
            return
        if event.request_key != state.request_key:
            raise DSAOperationError("source revocation request key mismatch")
        if event.route_epoch != state.route_epoch:
            raise DSAOperationError("source revocation route epoch mismatch")
        if (
            event.input_generation_id is not None
            and event.input_generation_id != state.active_source_generation_id
        ):
            raise DSAOperationError("source revocation generation mismatch")
        # Core does not yet have a qualified dense rehydrate implementation.
        # Once holes exist, continuing would be unsafe, so fail closed.
        if state.release_end > 0:
            raise DSAOperationError(
                "active DSA source was revoked after release; recovery unsupported"
            )
        previous_state = state.route_state
        state.route_state = DSARouteState.FALLBACK_RESIDENT
        state.route_epoch += 1
        state.fallback_reason = "source_revoked"
        self._emit(
            state,
            previous_state=previous_state,
            reason="source_revoked",
        )

    # ------------------------------------------------------------------
    # Route snapshots and lifecycle
    # ------------------------------------------------------------------

    def next_execution_seq(self) -> int:
        self._execution_seq += 1
        return self._execution_seq

    def build_route_snapshots(
        self,
        request_ids: list[str],
        accepted_ends: dict[str, int],
        *,
        scheduled_token_counts: dict[str, int] | None = None,
        execution_seq: int | None = None,
        token_prefix_digests: dict[str, str] | None = None,
    ) -> dict[str, DSARouteSnapshot]:
        """Build snapshots sharing one sequence for this model batch."""
        if not request_ids:
            return {}
        if execution_seq is None:
            execution_seq = self.next_execution_seq()
        if execution_seq <= 0:
            raise DSAOperationError("route snapshot execution_seq must be positive")

        snapshots: dict[str, DSARouteSnapshot] = {}
        if scheduled_token_counts is None:
            scheduled_token_counts = {
                request_id: accepted_ends[request_id]
                for request_id in request_ids
            }
        for request_id in request_ids:
            state = self._lookup_state(request_id)
            if state is None:
                continue
            if state.source_activation_inflight is not None:
                raise DSAOperationError(
                    "source-activating request cannot receive a model snapshot"
                )
            accepted_end = accepted_ends.get(request_id)
            if accepted_end is None:
                raise DSAOperationError("scheduled DSA request has no accepted end")
            if accepted_end != state.accepted_end:
                raise DSAOperationError(
                    "route snapshot accepted end was not consumed by controller"
                )
            scheduled_tokens = scheduled_token_counts.get(request_id)
            if scheduled_tokens is None or scheduled_tokens <= 0:
                raise DSAOperationError(
                    "scheduled DSA request has no positive scheduled token count"
                )
            digest = (
                token_prefix_digests.get(request_id)
                if token_prefix_digests is not None
                else None
            )
            if digest is None:
                digest = hashlib.sha256(
                    f"{state.request_key}\0{accepted_end}".encode()
                ).hexdigest()
            if not digest:
                raise DSAOperationError("route snapshot token digest is empty")
            if execution_seq <= state.last_issued_execution_seq:
                raise DSAOperationError("route execution sequence was reused")

            source_lease = None
            if state.route_state == DSARouteState.SPARSE:
                if not state.active_source_generation_id:
                    raise DSAOperationError("SPARSE route has no active source")
                source_lease = DSASourceLease(
                    source_lease_id=(
                        f"lease-{self.process_instance_id}-"
                        f"{state.request_key.scope_id}-{execution_seq}-"
                        f"{state.route_epoch}"
                    ),
                    request_key=state.request_key,
                    execution_seq=execution_seq,
                    route_epoch=state.route_epoch,
                    source_generation_id=state.active_source_generation_id,
                )
            if self.config.enabled:
                if execution_seq in state.pending_executions:
                    raise DSAOperationError("duplicate pending DSA execution")
                state.pending_executions[execution_seq] = DSAExecutionExpectation(
                    execution_seq=execution_seq,
                    route_epoch=state.route_epoch,
                    accepted_end_at_execution=accepted_end,
                    maximum_completed_canonical_end=(
                        state.completed_canonical_end + scheduled_tokens
                    ),
                    token_prefix_digest=digest,
                    source_lease_id=(
                        source_lease.source_lease_id
                        if source_lease is not None
                        else None
                    ),
                )
            state.last_issued_execution_seq = execution_seq
            state.token_prefix_digest = digest
            snapshots[request_id] = build_snapshot(
                state,
                execution_seq=execution_seq,
                accepted_end=accepted_end,
                source_lease=source_lease,
            )
        return snapshots

    def blocks_model_execution(self, request_id: str) -> bool:
        state = self._lookup_state(request_id)
        return state is not None and state.source_activation_inflight is not None

    def requires_preemption_quiesce(self, request_id: str) -> bool:
        """Whether preemption must wait for the unimplemented receipt quorum."""
        return self.config.enabled and self._lookup_state(request_id) is not None

    def requires_import_frontier_receipt(
        self,
        request_id: str,
        external_computed_tokens: int,
    ) -> bool:
        """Whether an external prefix needs a typed import frontier proof."""
        state = self._lookup_state(request_id)
        return bool(
            self.config.enabled
            and external_computed_tokens > 0
            and state is not None
            and state.transfer_plan.must_import_prefix
        )

    def begin_preemption(self, request_id: str) -> DSAOperationRef | None:
        state = self._lookup_state(request_id)
        if state is None:
            return None
        operation = DSAOperationRef(
            operation_id=self._new_operation_id(state, "preemption_quiesce"),
            parent_operation_id=None,
            kind="preemption_quiesce",
            obligations=frozenset({"preemption_quiesce"}),
            range_start=0,
            range_end=state.completed_canonical_end,
            input_generation_id=state.active_source_generation_id,
            output_generation_id=None,
            route_epoch=state.route_epoch,
        )
        state.preemption_inflight = operation
        return operation

    def finish_request(self, request_id: str) -> None:
        key = self._id_to_key.pop(request_id, None)
        state = self._states.pop(request_id, None)
        if key is not None:
            for request_key, operation_id in tuple(self.operations.records):
                if request_key == key:
                    self.operations.supersede(request_key, operation_id)
            self._command_outbox = [
                command
                for command in self._command_outbox
                if command.request_key != key
            ]
            for plan_key in tuple(self._pending_release_plans):
                if plan_key[0] == key:
                    self._pending_release_plans.pop(plan_key, None)
        if state is not None:
            state.route_state = DSARouteState.FINISHED
            state.pending_executions.clear()

    def waiting_reason(self, request_id: str) -> str | None:
        return (
            WAITING_FOR_SOURCE_ACTIVATION
            if self.blocks_model_execution(request_id)
            else None
        )

    # ------------------------------------------------------------------
    # Route decision logging
    # ------------------------------------------------------------------

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
        if not logger.isEnabledFor(logging.INFO):
            return
        self._event_seq += 1
        event_seq = self._event_seq
        route_state = state.route_state
        transfer_context = state.transfer_context
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
            "previous_mode": (
                previous_state.value if previous_state is not None else None
            ),
            "mode": route_state.value,
            "previous_selected_path": (
                self._selected_path(previous_state)
                if previous_state is not None
                else None
            ),
            "selected_path": self._selected_path(route_state),
            "previous_execution_path": (
                self._execution_path(previous_state)
                if previous_state is not None
                else None
            ),
            "execution_path": self._execution_path(route_state),
            "reason": reason[:128],
            "threshold": self.config.threshold,
            "accepted_end": accepted_end,
            "completed_canonical_end": state.completed_canonical_end,
            "external_computed_end": state.external_computed_end,
            "initial_prefill_complete": state.initial_prefill_complete,
            "raw_source_end": state.raw_source_end,
            "sparse_source_end": state.sparse_source_end,
            "remap_end": state.remap_end,
            "release_end": state.release_end,
            "must_import_prefix": state.transfer_plan.must_import_prefix,
            "must_export_prefill": state.transfer_plan.must_export_prefill,
            "must_persist_decode_windows": (
                state.transfer_plan.must_persist_decode_windows
            ),
            "remote_handoff_required": state.transfer_plan.remote_handoff_required,
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
