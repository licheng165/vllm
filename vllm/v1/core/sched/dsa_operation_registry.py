# SPDX-License-Identifier: Apache-2.0
"""DSA operation registry and frontier validation helpers.

Implements the Scheduler-side operation registry described in design sections
8.3 / 8.4 / 9.3-9.5 and the frontier invariants from section 7.2.

Key responsibilities:

* Track in-flight operations by ``(RequestKey, operation_id)``.
* Aggregate individual receipts into bundles by **exact-set** quorum (never a
  bare ``max``).  A bundle is READY only when every required
  participant/group/layer/chunk/tier expectation is covered exactly; FAILED if
  any required participant/group fails.
* Reject duplicate / conflicting receipts: identical duplicates are idempotent;
  conflicting duplicates fail closed.
* Maintain bounded tombstones so late ``ready`` events for terminal/superseded
  operations are rejected instead of unpinning active generations.
* Provide monotonic, aligned frontier validators and ``safe_remap_end`` which
  enforces ``M == 0 or M >= scratch_capacity`` and the
  ``R <= M <= min(Q, C, L) <= A`` invariant.

This module is deliberately pure-Python and free of NPU / scheduler-internal
dependencies so it can be unit tested in isolation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from vllm.v1.core.sched.dsa_types import (
    DSAOperationCommand,
    DSAOperationReceipt,
    DSAOperationRecord,
    DSAReceiptBundle,
    DSAReceiptExpectation,
    RequestKey,
)


class DSAOperationError(RuntimeError):
    """Raised when an invariant violation is detected in the operation protocol.

    The first version treats unknown/mismatched operations, unexpected state
    transitions and conflicting late receipts as fail-closed conditions.
    """


def align_down(value: int, alignment: int) -> int:
    if alignment <= 0:
        return value
    return (value // alignment) * alignment


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        return value
    return ((value + alignment - 1) // alignment) * alignment


def safe_remap_end(
    *,
    sparse_source_end: int,
    completed_canonical_end: int,
    initial_prefill_complete: bool,
    scratch_capacity: int,
    block_size: int,
    window_anchor: int,
    window_size: int,
    min_position_of_next_target_rows: Optional[int] = None,
) -> int:
    """Compute the safe remap (M) frontier.

    Implements design section 7.2::

        if not initial_prefill_complete:
            M = 0
        else:
            W = candidate_Q_for_first_activation or window_anchor
            L = W + align_down(min_position_of_next_target_rows - W, window_size)
        Q = align_down(verified_sparse_source_coverage, chunk_size)  # done by caller
        M = align_down(min(Q, C, L), block_size)

    Enforces ``M == 0 or M >= scratch_capacity``: equality with scratch
    capacity is structurally legal but reclaims no post-scratch block, so the
    caller keeps promoting until a later rolling goal creates actual benefit.
    """
    if not initial_prefill_complete:
        return 0
    if sparse_source_end <= 0:
        return 0

    c = completed_canonical_end
    q = sparse_source_end

    w = window_anchor
    if min_position_of_next_target_rows is None:
        # First version disables async; the next target row position is
        # derivable from completed output, commonly C.  MTP with multiple rows
        # MUST take the minimum position across all unfinished target rows.
        next_row = c
    else:
        next_row = min_position_of_next_target_rows
    l = w + align_down(max(next_row - w, 0), window_size)

    candidate = align_down(min(q, c, l), block_size)
    # Enforce the scratch safety floor: M == 0 or M >= scratch_capacity.
    if 0 < candidate < scratch_capacity:
        return 0
    return candidate


def validate_monotonic_raw_source_coverage(
    new_end: int,
    *,
    previous: int,
) -> int:
    """Raw source coverage is monotonic non-decreasing within a RequestKey."""
    if new_end < previous:
        raise DSAOperationError(
            f"raw source coverage regressed: new={new_end} previous={previous}"
        )
    return new_end


def validate_aligned_sparse_coverage(
    new_sparse_end: int,
    raw_source_end: int,
    *,
    previous: int,
    chunk_size: int,
) -> int:
    """Sparse source end must be chunk-aligned and ``<= raw_source_end``."""
    if new_sparse_end > raw_source_end:
        raise DSAOperationError(
            f"sparse source end {new_sparse_end} exceeds raw source end "
            f"{raw_source_end}"
        )
    if chunk_size > 0 and (new_sparse_end % chunk_size) != 0:
        raise DSAOperationError(
            f"sparse source end {new_sparse_end} not chunk-aligned "
            f"(chunk_size={chunk_size})"
        )
    if new_sparse_end < previous:
        raise DSAOperationError(
            f"sparse source coverage regressed: new={new_sparse_end} "
            f"previous={previous}"
        )
    return new_sparse_end


@dataclass
class _ExpectedKey:
    """The exact-set aggregation key for one atomic expectation."""

    participant_engine: str
    participant_process: str
    participant_rank: tuple[int, int, int, int]
    receipt_kind: str
    kv_group: int
    storage_tier: str

    @classmethod
    def from_expectation(cls, exp: DSAReceiptExpectation) -> "_ExpectedKey":
        p = exp.participant
        return cls(
            participant_engine=p.engine_id,
            participant_process=p.process_instance_id,
            participant_rank=(
                p.worker_rank, p.dp_rank, p.pp_rank, p.tp_rank,
            ),
            receipt_kind=exp.receipt_kind,
            kv_group=exp.kv_group,
            storage_tier=exp.storage_tier,
        )


@dataclass
class DSAOperationRegistry:
    """Scheduler-side registry of in-flight and recently-terminal operations.

    ``records`` maps ``(RequestKey, operation_id)`` -> record.  Terminal
    records are retained for a bounded time as tombstones
    (``tombstone_capacity``) so late ``ready`` / ``failed`` events are rejected
    rather than mutating live state.
    """

    records: dict[tuple[RequestKey, str], DSAOperationRecord] = field(
        default_factory=dict
    )
    tombstones: dict[tuple[RequestKey, str], DSAOperationRecord] = field(
        default_factory=dict
    )
    tombstone_capacity: int = 4096

    def register(self, command: DSAOperationCommand, *, deadline_ms: int) -> None:
        key = (command.request_key, command.operation.operation_id)
        existing = self.records.get(key)
        if existing is not None and existing.status != "issued":
            raise DSAOperationError(
                f"operation {key} already registered in status "
                f"{existing.status}"
            )
        self.records[key] = DSAOperationRecord(
            command=command,
            deadline_ms=deadline_ms,
            consumer_status={
                obl: "pending" for obl in command.operation.obligations
            },
        )

    def match(self, event_or_receipt) -> DSAOperationRecord:
        """Find the live record matching a control event or receipt."""
        key = (
            event_or_receipt.request_key,
            event_or_receipt.operation_id,
        )
        record = self.records.get(key)
        if record is None:
            tomb = self.tombstones.get(key)
            if tomb is not None:
                raise DSAOperationError(
                    f"event for terminal/superseded operation {key} rejected "
                    f"(status={tomb.status})"
                )
            raise DSAOperationError(f"unknown operation {key}")
        return record

    def consume_ready_bundle(
        self, bundle_id: str, request_key: RequestKey, operation_id: str
    ) -> DSAOperationRecord:
        """Fetch the record for a ready bundle and validate it is ready."""
        record = self.match(
            _BundleRef(request_key=request_key, operation_id=operation_id)
        )
        if record.status not in ("ready", "activating"):
            raise DSAOperationError(
                f"bundle {bundle_id} consumed from non-ready record "
                f"(status={record.status})"
            )
        if record.bundle_id != bundle_id:
            raise DSAOperationError(
                f"bundle id mismatch: record={record.bundle_id} event={bundle_id}"
            )
        return record

    def submit_receipt(self, receipt: DSAOperationReceipt) -> Optional[DSAReceiptBundle]:
        """Submit an individual worker receipt.

        Returns a READY bundle once exact-set quorum is achieved, ``None`` while
        still collecting, and raises on conflict / unknown operation / failed
        terminal.  Identical duplicate receipts are idempotent.
        """
        record = self.match(receipt)
        if record.status in ("ready", "committed"):
            # Terminal physical operation: late failure is rejected; identical
            # success receipt is idempotent.
            self._reject_late_failure_if_terminal(receipt, record)
            return None

        if receipt.status == "failed":
            self._handle_failed_receipt(record, receipt)
            return None

        expected_map: dict[_ExpectedKey, DSAReceiptExpectation] = {}
        for exp in record.command.expected_receipts:
            expected_map[_ExpectedKey.from_expectation(exp)] = exp

        exp_key = _ExpectedKey(
            participant_engine=receipt.participant.engine_id,
            participant_process=receipt.participant.process_instance_id,
            participant_rank=(
                receipt.participant.worker_rank,
                receipt.participant.dp_rank,
                receipt.participant.pp_rank,
                receipt.participant.tp_rank,
            ),
            receipt_kind=receipt.receipt_kind,
            kv_group=receipt.kv_group,
            storage_tier=receipt.storage_tier,
        )
        expectation = expected_map.get(exp_key)
        if expectation is None:
            raise DSAOperationError(
                f"receipt {receipt.receipt_id} does not match any expected "
                f"expectation for operation {receipt.operation_id}"
            )

        existing = record.receipts_by_expectation.get(
            self._expectation_store_key(exp_key, expectation)
        )
        if existing is not None:
            if existing.receipt_id != receipt.receipt_id:
                raise DSAOperationError(
                    f"conflicting duplicate receipt for expectation {exp_key}: "
                    f"existing={existing.receipt_id} new={receipt.receipt_id}"
                )
            return None  # idempotent duplicate

        # Validate exact layer/chunk coverage against the expectation.
        self._validate_exact_coverage(receipt, expectation)

        record.receipts_by_expectation[
            self._expectation_store_key(exp_key, expectation)
        ] = receipt
        record.status = "collecting"

        if len(record.receipts_by_expectation) >= len(
            record.command.expected_receipts
        ):
            return self._assemble_bundle(record)
        return None

    @staticmethod
    def _expectation_store_key(
        exp_key: _ExpectedKey, expectation: DSAReceiptExpectation
    ) -> tuple:
        # Layers/chunks are part of the identity so a single participant/group
        # cannot forge completeness by covering only part of the range.
        return (
            exp_key.participant_engine,
            exp_key.participant_process,
            exp_key.participant_rank,
            exp_key.receipt_kind,
            exp_key.kv_group,
            exp_key.storage_tier,
            tuple(expectation.layers),
            tuple(expectation.chunks),
        )

    @staticmethod
    def _validate_exact_coverage(
        receipt: DSAOperationReceipt, expectation: DSAReceiptExpectation
    ) -> None:
        expected_layers = set(expectation.layers)
        covered_layers = set(receipt.covered_layers)
        if covered_layers != expected_layers:
            raise DSAOperationError(
                f"receipt {receipt.receipt_id} layer coverage mismatch: "
                f"expected={sorted(expected_layers)} "
                f"covered={sorted(covered_layers)}"
            )
        expected_chunks = set(expectation.chunks)
        covered_chunks = set(receipt.covered_chunks)
        if covered_chunks != expected_chunks:
            raise DSAOperationError(
                f"receipt {receipt.receipt_id} chunk coverage mismatch: "
                f"expected={sorted(expected_chunks)} "
                f"covered={sorted(covered_chunks)}"
            )

    def _assemble_bundle(self, record: DSAOperationRecord) -> DSAReceiptBundle:
        receipts = tuple(record.receipts_by_expectation.values())
        raw_source_end = max((r.range_end for r in receipts), default=0)
        # sparse source end is chunk-aligned by the caller's validators; here
        # we take the min continuous frontier across required participants.
        sparse_source_end = min((r.range_end for r in receipts), default=0)
        materialized_end = max(
            (r.range_end for r in receipts if r.receipt_kind == "npu_materialization"),
            default=0,
        )
        bundle = DSAReceiptBundle(
            bundle_id=self._new_bundle_id(record),
            request_key=record.command.request_key,
            operation_id=record.command.operation.operation_id,
            route_epoch=record.command.operation.route_epoch,
            input_generation_id=record.command.operation.input_generation_id,
            output_generation_id=record.command.operation.output_generation_id,
            source_manifest_id=None,
            token_prefix_digest=record.command.token_prefix_digest,
            data_compatibility_fingerprint=(
                record.command.cache_namespace_fingerprint
            ),
            receipt_ids=tuple(r.receipt_id for r in receipts),
            aggregate_status="complete",
            raw_source_end=raw_source_end,
            sparse_source_end=sparse_source_end,
            materialized_end=materialized_end,
            lease_descriptor_id=None,
        )
        record.bundle_id = bundle.bundle_id
        record.status = "ready"
        return bundle

    @staticmethod
    def _new_bundle_id(record: DSAOperationRecord) -> str:
        return f"bundle-{record.command.operation.operation_id}-{int(time.time() * 1e6)}"

    def _reject_late_failure_if_terminal(
        self, receipt: DSAOperationReceipt, record: DSAOperationRecord
    ) -> None:
        if receipt.status == "failed":
            raise DSAOperationError(
                f"late failure for terminal operation {receipt.operation_id} "
                f"(status={record.status}) rejected as stale event"
            )

    def _handle_failed_receipt(
        self, record: DSAOperationRecord, receipt: DSAOperationReceipt
    ) -> None:
        record.status = "failed"
        record.error_code = (
            receipt.error_code if hasattr(receipt, "error_code") else "receipt_failed"
        )
        for obl in record.consumer_status:
            if record.consumer_status[obl] == "pending":
                record.consumer_status[obl] = "failed"

    def mark_consumer_terminal(
        self,
        request_key: RequestKey,
        operation_id: str,
        obligation: str,
        terminal_status: str,
    ) -> None:
        """Mark one logical consumer of a shared physical operation terminal.

        The physical record only enters final tombstone once ALL consumers
        terminate; a shared long-P store serving ``promotion`` and
        ``prefill_export`` keeps its bundle/candidate alive until both
        consumers terminate.
        """
        record = self.match(_BundleRef(request_key=request_key, operation_id=operation_id))
        if obligation in record.consumer_status:
            record.consumer_status[obligation] = terminal_status  # type: ignore[assignment]
        if all(s != "pending" for s in record.consumer_status.values()):
            self._tombstone(record)

    def fail_operation(
        self,
        request_key: RequestKey,
        operation_id: str,
        error_code: str,
    ) -> DSAOperationRecord:
        """Fail an entire operation, fanning out to all obligations."""
        record = self.match(_BundleRef(request_key=request_key, operation_id=operation_id))
        if record.status in ("ready", "committed"):
            raise DSAOperationError(
                f"late failure for terminal operation {operation_id} rejected"
            )
        record.status = "failed"
        record.error_code = error_code
        for obl in record.consumer_status:
            if record.consumer_status[obl] == "pending":
                record.consumer_status[obl] = "failed"
        self._tombstone(record)
        return record

    def supersede(self, request_key: RequestKey, operation_id: str) -> None:
        """Mark an operation superseded (e.g. new generation took over)."""
        record = self.match(_BundleRef(request_key=request_key, operation_id=operation_id))
        record.status = "superseded"
        self._tombstone(record)

    def _tombstone(self, record: DSAOperationRecord) -> None:
        key = (record.command.request_key, record.command.operation.operation_id)
        self.records.pop(key, None)
        self.tombstones[key] = record
        if len(self.tombstones) > self.tombstone_capacity:
            # Bounded eviction: drop the oldest by insertion order.
            oldest = next(iter(self.tombstones))
            self.tombstones.pop(oldest, None)

    def expire_deadlines(self, *, now_ms: Optional[int] = None) -> list[DSAOperationRecord]:
        """Fail operations whose deadline has passed.

        Returns the failed records so the Scheduler can run per-obligation
        rollback (design section 9.5).
        """
        now = now_ms if now_ms is not None else int(time.time() * 1000)
        expired: list[DSAOperationRecord] = []
        for key, record in list(self.records.items()):
            if record.deadline_ms and now > record.deadline_ms:
                record.status = "failed"
                record.error_code = "deadline_expired"
                for obl in record.consumer_status:
                    if record.consumer_status[obl] == "pending":
                        record.consumer_status[obl] = "failed"
                self._tombstone(record)
                expired.append(record)
        return expired


@dataclass(frozen=True)
class _BundleRef:
    """Lightweight stand-in so :meth:`match` accepts bundle-originated lookups."""

    request_key: RequestKey
    operation_id: str
