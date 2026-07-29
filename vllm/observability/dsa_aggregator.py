# SPDX-License-Identifier: Apache-2.0
"""Executor exact-set receipt aggregator for the DSA offload protocol.

Owns the ``source.generation.sealed`` and ``frontier.publish`` events (§6.2/§7.7).
It replaces the legacy ``max()`` merge semantics with a *trusted exact-set*
contract: an operation's frontier is only published once every required
participant/group/layer/chunk/tier has reported a matching complete receipt,
and the common frontier is the minimum across required participants (never
``max``).

Until every caller is migrated to register expected slots, callers that still
rely on the old ``max`` merge MUST keep emitting ``route.observed`` with
``quorum_status=unproven_current_max_merge`` (see ``dsa_offload``); only this
aggregator is permitted to emit a canonical ``frontier.publish`` carrying
``quorum_status="proven"``.

The aggregator is pure-Python and dependency-light so it can be unit-tested in
isolation and shared by vLLM-Ascend/LMCache through the vLLM helper.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Iterable

from vllm.observability.dsa_offload import (
    DSAOffloadLogger,
    DSAOperationReceipt,
    DSAReceiptBundle,
    DiagLevel,
    RequestKey,
    dsa_logger_for,
)


@dataclass(frozen=True)
class OperationSlot:
    """A single required receipt slot for an operation.

    A slot is satisfied by exactly one matching :class:`DSAOperationReceipt`
    (same participant, group and kind). ``authoritative_writer`` slots must be
    filled by a ``storage`` receipt; view slots (e.g. non-writer ranks under
    ``save_only_first_rank``) are filled by a ``route_epoch_complete`` receipt.
    """

    participant_id: str
    kv_group: int
    receipt_kind: str
    authoritative_writer: bool = False


@dataclass
class OperationSpec:
    """The trusted expected receipt set for one store operation.

    Built by the Scheduler from trusted startup config (engine/DP/PP/TP shape,
    ``save_only_first_rank``, required groups/layers/chunks/tiers) — never from
    an emitter's self-reported ``required_groups``.
    """

    operation_id: str
    request_key: RequestKey
    route_epoch: int
    input_generation_id: str | None
    token_prefix_digest: str
    cache_namespace_fingerprint: str
    data_compatibility_fingerprint: str
    expected_slots: frozenset[OperationSlot]
    # Participants whose storage receipt contributes to the common frontier
    # (min). For ``save_only_first_rank`` this is just the writer rank.
    frontier_participant_ids: frozenset[str]


@dataclass
class _OperationState:
    spec: OperationSpec
    receipts: dict[OperationSlot, DSAOperationReceipt] = field(default_factory=dict)
    failed: DSAOperationReceipt | None = None
    sealed_bundle: DSAReceiptBundle | None = None


def _slot_for(receipt: DSAOperationReceipt) -> OperationSlot:
    return OperationSlot(
        participant_id=_participant_id(receipt.participant),
        kv_group=receipt.kv_group,
        receipt_kind=receipt.receipt_kind,
        authoritative_writer=(receipt.receipt_kind == "storage"),
    )


def _participant_id(p) -> str:
    return f"{p.engine_id}:{p.worker_rank}:{p.dp_rank}:{p.pp_rank}:{p.tp_rank}"


class DSAReceiptAggregator:
    """Accumulates receipts and seals exactly-complete operations.

    Thread-safe. Emits structured events through the shared
    :class:`DSAOffloadLogger`.
    """

    def __init__(self, log: DSAOffloadLogger | None = None) -> None:
        self._log = log or dsa_logger_for("vllm.aggregator")
        self._lock = threading.Lock()
        self._ops: dict[str, _OperationState] = {}

    # -- registration -------------------------------------------------------
    def register_operation(self, spec: OperationSpec) -> None:
        with self._lock:
            self._ops[spec.operation_id] = _OperationState(spec=spec)
        if self._log.enabled(DiagLevel.LIFECYCLE):
            self._log.emit(
                "store.command",
                outcome="ok",
                level=DiagLevel.LIFECYCLE,
                trace_id=None,
                request_process_instance=spec.request_key.process_instance_id,
                scope_id=spec.request_key.scope_id,
                operation_id=spec.operation_id,
                route_epoch=spec.route_epoch,
                input_generation_id=spec.input_generation_id,
                expected_slot_count=len(spec.expected_slots),
                frontier_participant_count=len(spec.frontier_participant_ids),
            )

    def forget_operation(self, operation_id: str) -> None:
        with self._lock:
            self._ops.pop(operation_id, None)

    # -- ingestion ----------------------------------------------------------
    def add_receipt(self, receipt: DSAOperationReceipt) -> DSAReceiptBundle | None:
        """Record one receipt; return a freshly sealed bundle if this completed
        the exact set, else ``None``.

        A failed receipt fails the whole operation (any required participant
        failing -> failure) and no canonical frontier is published.
        """
        with self._lock:
            state = self._ops.get(receipt.operation_id)
            if state is None:
                # Unknown operation: log but do not fabricate a bundle.
                if self._log.enabled(DiagLevel.LIFECYCLE):
                    self._log.emit(
                        "invariant.violation",
                        outcome="error",
                        level=DiagLevel.LIFECYCLE,
                        reason="receipt_for_unknown_operation",
                        operation_id=receipt.operation_id,
                        receipt_id=receipt.receipt_id,
                    )
                return None
            slot = _slot_for(receipt)
            if receipt.status == "failed":
                if state.failed is None:
                    state.failed = receipt
                    self._emit_failure(state, receipt)
                return None
            if slot not in state.spec.expected_slots:
                # Unexpected slot: ignore silently for exactness; deep-log it.
                if self._log.enabled(DiagLevel.DEEP):
                    self._log.emit(
                        "invariant.violation",
                        outcome="degraded",
                        level=DiagLevel.DEEP,
                        reason="unexpected_receipt_slot",
                        operation_id=receipt.operation_id,
                        receipt_id=receipt.receipt_id,
                        participant_id=slot.participant_id,
                        kv_group=slot.kv_group,
                        receipt_kind=slot.receipt_kind,
                    )
                return None
            # Last-writer-wins within a slot, but a slot is only complete with
            # status=="complete"; receipt here is complete by the guard above.
            state.receipts[slot] = receipt
            if self._is_complete(state):
                bundle = self._seal(state)
                return bundle
            return None

    # -- queries ------------------------------------------------------------
    def is_complete(self, operation_id: str) -> bool:
        with self._lock:
            state = self._ops.get(operation_id)
            return state is not None and state.sealed_bundle is not None

    def bundle(self, operation_id: str) -> DSAReceiptBundle | None:
        with self._lock:
            state = self._ops.get(operation_id)
            return state.sealed_bundle if state is not None else None

    # -- internals ----------------------------------------------------------
    @staticmethod
    def _is_complete(state: _OperationState) -> bool:
        if state.failed is not None:
            return False
        return state.spec.expected_slots <= set(state.receipts.keys())

    def _seal(self, state: _OperationState) -> DSAReceiptBundle:
        spec = state.spec
        receipts = list(state.receipts.values())
        # Common frontier = min over required frontier participants' completed
        # storage receipts (§7.7). Never max.
        storage_ends = [
            r.range_end for r in receipts
            if r.receipt_kind == "storage"
            and _participant_id(r.participant) in spec.frontier_participant_ids
        ]
        sparse_source_end = min(storage_ends) if storage_ends else 0
        raw_source_end = sparse_source_end
        # Deterministic generation id from the operation + sealed frontier.
        output_generation_id = f"g-{spec.operation_id}-{sparse_source_end}"
        bundle = DSAReceiptBundle(
            bundle_id=f"b-{uuid.uuid4().hex[:12]}",
            request_key=spec.request_key,
            operation_id=spec.operation_id,
            route_epoch=spec.route_epoch,
            input_generation_id=spec.input_generation_id,
            output_generation_id=output_generation_id,
            source_manifest_id=None,
            token_prefix_digest=spec.token_prefix_digest,
            data_compatibility_fingerprint=spec.data_compatibility_fingerprint,
            receipt_ids=tuple(r.receipt_id for r in receipts),
            aggregate_status="complete",
            raw_source_end=raw_source_end,
            sparse_source_end=sparse_source_end,
            materialized_end=sparse_source_end,
            lease_descriptor_id=None,
        )
        state.sealed_bundle = bundle
        rk = spec.request_key
        if self._log.enabled(DiagLevel.LIFECYCLE):
            self._log.emit(
                "source.generation.sealed",
                outcome="ok",
                level=DiagLevel.LIFECYCLE,
                trace_id=None,
                request_process_instance=rk.process_instance_id,
                scope_id=rk.scope_id,
                operation_id=spec.operation_id,
                route_epoch=spec.route_epoch,
                source_generation_id=output_generation_id,
                receipt_id=bundle.bundle_id,
                raw_source_end=raw_source_end,
                sparse_source_end=sparse_source_end,
            )
            self._log.emit(
                "frontier.publish",
                outcome="ok",
                level=DiagLevel.LIFECYCLE,
                trace_id=None,
                request_process_instance=rk.process_instance_id,
                scope_id=rk.scope_id,
                route_epoch=spec.route_epoch,
                operation_id=spec.operation_id,
                receipt_bundle_id=bundle.bundle_id,
                source_generation_id=output_generation_id,
                raw_source_end=raw_source_end,
                sparse_source_end=sparse_source_end,
                quorum_status="proven",
                participant_quorum="proven",
            )
        return bundle

    def _emit_failure(self, state: _OperationState,
                      receipt: DSAOperationReceipt) -> None:
        if not self._log.enabled(DiagLevel.LIFECYCLE):
            return
        rk = state.spec.request_key
        self._log.emit(
            "store.batch.fenced",
            outcome="error",
            level=DiagLevel.LIFECYCLE,
            trace_id=None,
            request_process_instance=rk.process_instance_id,
            scope_id=rk.scope_id,
            operation_id=receipt.operation_id,
            route_epoch=state.spec.route_epoch,
            reason="required_participant_failed",
            failed_receipt_id=receipt.receipt_id,
            failed_participant_id=_participant_id(receipt.participant),
            kv_group=receipt.kv_group,
        )


__all__ = [
    "DSAReceiptAggregator",
    "OperationSpec",
    "OperationSlot",
]
