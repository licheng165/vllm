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

import hashlib
import time
from dataclasses import dataclass, field
from typing import NamedTuple

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
    min_position_of_next_target_rows: int | None = None,
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
    limit_end = w + align_down(max(next_row - w, 0), window_size)

    candidate = align_down(min(q, c, limit_end), block_size)
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


class _ExpectedKey(NamedTuple):
    """The exact-set aggregation key for one atomic expectation."""

    participant_engine: str
    participant_process: str
    participant_rank: tuple[int, int, int, int]
    receipt_kind: str
    kv_group: int
    layers: tuple[int, ...]
    chunks: tuple[tuple[int, int], ...]
    storage_tier: str

    @classmethod
    def from_expectation(cls, exp: DSAReceiptExpectation) -> _ExpectedKey:
        p = exp.participant
        return cls(
            participant_engine=p.engine_id,
            participant_process=p.process_instance_id,
            participant_rank=(
                p.worker_rank,
                p.dp_rank,
                p.pp_rank,
                p.tp_rank,
            ),
            receipt_kind=exp.receipt_kind,
            kv_group=exp.kv_group,
            layers=tuple(sorted(exp.layers)),
            chunks=tuple(sorted(exp.chunks)),
            storage_tier=exp.storage_tier,
        )

    @classmethod
    def from_receipt(cls, receipt: DSAOperationReceipt) -> _ExpectedKey:
        p = receipt.participant
        return cls(
            participant_engine=p.engine_id,
            participant_process=p.process_instance_id,
            participant_rank=(
                p.worker_rank,
                p.dp_rank,
                p.pp_rank,
                p.tp_rank,
            ),
            receipt_kind=receipt.receipt_kind,
            kv_group=receipt.kv_group,
            layers=tuple(sorted(receipt.covered_layers)),
            chunks=tuple(sorted(receipt.covered_chunks)),
            storage_tier=receipt.storage_tier,
        )


@dataclass
class DSAOperationRegistry:
    """Scheduler-side registry of in-flight and recently-terminal operations.

    ``records`` maps ``(RequestKey, operation_id)`` -> record.  Terminal
    records are retained for a bounded time as tombstones
    (``tombstone_capacity``) so late ``ready`` / ``failed`` events are rejected
    rather than mutating live state. An already accepted identical receipt is
    the sole idempotent exception.
    """

    records: dict[tuple[RequestKey, str], DSAOperationRecord] = field(
        default_factory=dict
    )
    tombstones: dict[tuple[RequestKey, str], DSAOperationRecord] = field(
        default_factory=dict
    )
    tombstone_capacity: int = 4096

    def register(
        self, command: DSAOperationCommand, *, deadline_ms: int
    ) -> DSAOperationRecord:
        key = (command.request_key, command.operation.operation_id)
        if key in self.records:
            raise DSAOperationError(f"operation {key} is already registered")
        if key in self.tombstones:
            raise DSAOperationError(
                f"operation {key} is tombstoned and cannot be reused"
            )

        self._validate_command(command, deadline_ms)
        record = DSAOperationRecord(
            command=command,
            deadline_ms=deadline_ms,
            consumer_status={obl: "pending" for obl in command.operation.obligations},
        )
        self.records[key] = record
        return record

    @classmethod
    def _validate_command(cls, command: DSAOperationCommand, deadline_ms: int) -> None:
        operation = command.operation
        if not operation.operation_id:
            raise DSAOperationError("operation_id must not be empty")
        cls._validate_range(
            operation.range_start,
            operation.range_end,
            label="operation",
            allow_empty=False,
        )
        if command.accepted_end_at_issue < 0:
            raise DSAOperationError("accepted_end_at_issue must be non-negative")
        if operation.range_end > command.accepted_end_at_issue:
            raise DSAOperationError(
                f"operation range end {operation.range_end} exceeds accepted "
                f"end {command.accepted_end_at_issue}"
            )
        if deadline_ms < 0:
            raise DSAOperationError("deadline_ms must be non-negative")
        if not command.expected_receipts:
            raise DSAOperationError("operation must have at least one expectation")

        expected_keys: set[_ExpectedKey] = set()
        for expectation in command.expected_receipts:
            cls._validate_expectation(expectation, operation)
            expectation_key = _ExpectedKey.from_expectation(expectation)
            if expectation_key in expected_keys:
                raise DSAOperationError(
                    f"duplicate receipt expectation {expectation_key}"
                )
            expected_keys.add(expectation_key)

    @classmethod
    def _validate_expectation(cls, expectation, operation) -> None:
        cls._validate_layers(
            expectation.layers,
            label="expectation layers",
        )
        cls._validate_chunks(
            expectation.chunks,
            operation.range_start,
            operation.range_end,
            label="expectation chunks",
        )

    @staticmethod
    def _validate_layers(layers: tuple[int, ...], *, label: str) -> None:
        if not layers:
            raise DSAOperationError(f"{label} must not be empty")
        if any(type(layer) is not int or layer < 0 for layer in layers):
            raise DSAOperationError(f"{label} must contain non-negative integers")
        if len(layers) != len(set(layers)):
            raise DSAOperationError(f"{label} contains duplicates")

    @classmethod
    def _validate_chunks(
        cls,
        chunks: tuple[tuple[int, int], ...],
        range_start: int,
        range_end: int,
        *,
        label: str,
    ) -> None:
        validated: list[tuple[int, int]] = []
        for chunk in chunks:
            if not isinstance(chunk, tuple) or len(chunk) != 2:
                raise DSAOperationError(f"{label} contains a malformed range")
            start, end = chunk
            cls._validate_range(start, end, label=label, allow_empty=False)
            if start < range_start or end > range_end:
                raise DSAOperationError(
                    f"{label} range {chunk} is outside operation range "
                    f"({range_start}, {range_end})"
                )
            validated.append(chunk)

        if len(validated) != len(set(validated)):
            raise DSAOperationError(f"{label} contains duplicates")
        if not validated:
            raise DSAOperationError(f"{label} must not be empty")
        previous_end = range_start
        for start, end in sorted(validated):
            if start < previous_end:
                raise DSAOperationError(f"{label} contains overlapping ranges")
            if start > previous_end:
                raise DSAOperationError(f"{label} contains a coverage gap")
            previous_end = end
        if previous_end != range_end:
            raise DSAOperationError(
                f"{label} does not cover the complete operation range"
            )

    @staticmethod
    def _validate_range(
        start: int,
        end: int,
        *,
        label: str,
        allow_empty: bool,
    ) -> None:
        if type(start) is not int or type(end) is not int:
            raise DSAOperationError(f"{label} range boundaries must be integers")
        if start < 0 or end < start or (not allow_empty and end == start):
            raise DSAOperationError(f"malformed {label} range ({start}, {end})")

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
        self._validate_stored_complete_bundle(record)
        return record

    def submit_receipt(self, receipt: DSAOperationReceipt) -> DSAReceiptBundle | None:
        """Submit an individual worker receipt.

        Returns a complete bundle once exact-set quorum is achieved, a failed
        bundle for a validated expected failure, and ``None`` while still
        collecting. Identical duplicate receipts are idempotent.
        """
        key = (receipt.request_key, receipt.operation_id)
        record = self.records.get(key)
        tombstoned = False
        if record is None:
            record = self.tombstones.get(key)
            tombstoned = record is not None
        if record is None:
            raise DSAOperationError(f"unknown operation {key}")

        if not receipt.receipt_id:
            raise DSAOperationError("receipt_id must not be empty")
        duplicate = self._find_receipt_by_id(receipt.receipt_id)
        if duplicate is not None:
            duplicate_record, duplicate_receipt = duplicate
            if duplicate_receipt != receipt:
                raise DSAOperationError(
                    f"receipt id {receipt.receipt_id} was reused with a "
                    "different payload"
                )
            if duplicate_record is not record:
                raise DSAOperationError(
                    f"receipt id {receipt.receipt_id} belongs to another operation"
                )
            return record.bundle

        if (
            record.status == "failed"
            and record.bundle is not None
            and record.bundle.aggregate_status == "failed"
        ):
            # The first expected failure has already rolled back the logical
            # operation. Other workers and receipt kinds from the same batch
            # must still be drainable without turning that fallback into an
            # engine-fatal late-receipt error.
            expectation_key = self._validate_receipt(record, receipt)
            existing = record.receipts_by_expectation.get(expectation_key)
            if existing is not None:
                raise DSAOperationError(
                    f"expectation {expectation_key} is already filled by receipt "
                    f"{existing.receipt_id}"
                )
            record.receipts_by_expectation[expectation_key] = receipt
            record.receipts_by_id[receipt.receipt_id] = receipt
            return record.bundle

        if tombstoned or record.status in (
            "ready",
            "activating",
            "committed",
            "failed",
            "superseded",
        ):
            raise DSAOperationError(
                f"new receipt for non-collecting operation {key} rejected "
                f"(status={record.status})"
            )

        expectation_key = self._validate_receipt(record, receipt)
        existing = record.receipts_by_expectation.get(expectation_key)
        if existing is not None:
            raise DSAOperationError(
                f"expectation {expectation_key} is already filled by receipt "
                f"{existing.receipt_id}"
            )

        record.receipts_by_expectation[expectation_key] = receipt
        record.receipts_by_id[receipt.receipt_id] = receipt
        if receipt.status == "failed":
            bundle = self._build_failed_bundle(record, receipt)
            record.bundle_id = bundle.bundle_id
            record.bundle = bundle
            record.status = "failed"
            record.error_code = bundle.error_code
            self._fail_pending_consumers(record)
            return bundle

        record.status = "collecting"
        expected_keys = set(self._expected_map(record))
        if set(record.receipts_by_expectation) == expected_keys:
            bundle = self._build_complete_bundle(record)
            record.bundle_id = bundle.bundle_id
            record.bundle = bundle
            record.status = "ready"
            return bundle
        return None

    def _find_receipt_by_id(
        self, receipt_id: str
    ) -> tuple[DSAOperationRecord, DSAOperationReceipt] | None:
        for record in (*self.records.values(), *self.tombstones.values()):
            receipt = record.receipts_by_id.get(receipt_id)
            if receipt is not None:
                return record, receipt
        return None

    def _validate_receipt(
        self, record: DSAOperationRecord, receipt: DSAOperationReceipt
    ) -> _ExpectedKey:
        command = record.command
        operation = command.operation
        if receipt.request_key != command.request_key:
            raise DSAOperationError("receipt request key does not match command")
        if receipt.operation_id != operation.operation_id:
            raise DSAOperationError("receipt operation id does not match command")
        if receipt.route_epoch != operation.route_epoch:
            raise DSAOperationError("receipt route epoch does not match command")
        if receipt.input_generation_id != operation.input_generation_id:
            raise DSAOperationError("receipt input generation does not match command")
        if receipt.output_generation_id != operation.output_generation_id:
            raise DSAOperationError("receipt output generation does not match command")
        if (
            receipt.range_start != operation.range_start
            or receipt.range_end != operation.range_end
        ):
            raise DSAOperationError("receipt range does not match command")
        if receipt.accepted_end_at_seal != command.accepted_end_at_issue:
            raise DSAOperationError(
                "receipt accepted end does not match command issue frontier"
            )
        if receipt.token_prefix_digest != command.token_prefix_digest:
            raise DSAOperationError("receipt token digest does not match command")
        if receipt.cache_namespace_fingerprint != command.cache_namespace_fingerprint:
            raise DSAOperationError("receipt cache namespace does not match command")
        if receipt.status not in ("complete", "failed"):
            raise DSAOperationError(f"invalid receipt status {receipt.status!r}")
        if receipt.status == "complete" and receipt.error_code is not None:
            raise DSAOperationError("complete receipt must not have an error code")

        self._validate_layers(
            receipt.covered_layers,
            label="receipt covered_layers",
        )
        self._validate_chunks(
            receipt.covered_chunks,
            operation.range_start,
            operation.range_end,
            label="receipt covered_chunks",
        )
        expectation_key = _ExpectedKey.from_receipt(receipt)
        expectation = self._expected_map(record).get(expectation_key)
        if expectation is None:
            raise DSAOperationError(
                f"receipt {receipt.receipt_id} does not match an exact "
                f"expectation for operation {receipt.operation_id}"
            )
        if receipt.status == "complete":
            if receipt.guarantee != expectation.minimum_guarantee:
                raise DSAOperationError(
                    f"receipt {receipt.receipt_id} guarantee mismatch: "
                    f"expected={expectation.minimum_guarantee!r} "
                    f"received={receipt.guarantee!r}"
                )
        elif (
            receipt.guarantee is not None
            and receipt.guarantee != expectation.minimum_guarantee
        ):
            raise DSAOperationError(
                f"failed receipt {receipt.receipt_id} carries an unexpected "
                f"guarantee {receipt.guarantee!r}"
            )
        if receipt.status == "complete" and receipt.lease_descriptor_id is not None:
            existing_lease_ids = {
                existing.lease_descriptor_id
                for existing in record.receipts_by_expectation.values()
                if existing.lease_descriptor_id is not None
            }
            if existing_lease_ids - {receipt.lease_descriptor_id}:
                raise DSAOperationError(
                    "receipt lease descriptor conflicts with prior receipts"
                )
        return expectation_key

    @staticmethod
    def _expected_map(
        record: DSAOperationRecord,
    ) -> dict[_ExpectedKey, DSAReceiptExpectation]:
        expected: dict[_ExpectedKey, DSAReceiptExpectation] = {}
        for expectation in record.command.expected_receipts:
            key = _ExpectedKey.from_expectation(expectation)
            if key in expected:
                raise DSAOperationError(f"duplicate receipt expectation {key}")
            expected[key] = expectation
        return expected

    def _build_complete_bundle(self, record: DSAOperationRecord) -> DSAReceiptBundle:
        expected = self._expected_map(record)
        if set(record.receipts_by_expectation) != set(expected):
            raise DSAOperationError("cannot build bundle without exact quorum")
        expected_keys = tuple(sorted(expected))
        expected_receipts = tuple(
            record.receipts_by_expectation[key] for key in expected_keys
        )
        if len(record.receipts_by_id) != len(expected_receipts):
            raise DSAOperationError("receipt id index does not match exact quorum")
        for key, receipt in zip(expected_keys, expected_receipts):
            if self._validate_receipt(record, receipt) != key:
                raise DSAOperationError("receipt is stored under the wrong expectation")
            if record.receipts_by_id.get(receipt.receipt_id) != receipt:
                raise DSAOperationError("receipt id index contains conflicting payload")
        receipts = tuple(
            sorted(expected_receipts, key=lambda receipt: receipt.receipt_id)
        )
        if any(receipt.status != "complete" for receipt in receipts):
            raise DSAOperationError("complete bundle contains a failed receipt")

        raw_source_end = self._minimum_frontier(
            receipts, frozenset({"storage", "source_seal"})
        )
        sparse_source_end = self._minimum_frontier(receipts, frozenset({"source_seal"}))
        materialized_end = self._minimum_frontier(
            receipts, frozenset({"npu_materialization"})
        )
        lease_ids = {
            receipt.lease_descriptor_id
            for receipt in receipts
            if receipt.lease_descriptor_id is not None
        }
        if len(lease_ids) > 1:
            raise DSAOperationError("receipts contain conflicting lease descriptor ids")
        lease_descriptor_id = next(iter(lease_ids), None)
        return DSAReceiptBundle(
            bundle_id=self._new_bundle_id(record),
            request_key=record.command.request_key,
            operation_id=record.command.operation.operation_id,
            route_epoch=record.command.operation.route_epoch,
            input_generation_id=record.command.operation.input_generation_id,
            output_generation_id=record.command.operation.output_generation_id,
            source_manifest_id=None,
            token_prefix_digest=record.command.token_prefix_digest,
            data_compatibility_fingerprint=(record.command.cache_namespace_fingerprint),
            receipt_ids=tuple(receipt.receipt_id for receipt in receipts),
            aggregate_status="complete",
            raw_source_end=raw_source_end,
            sparse_source_end=sparse_source_end,
            materialized_end=materialized_end,
            lease_descriptor_id=lease_descriptor_id,
            error_code=None,
        )

    def _build_failed_bundle(
        self,
        record: DSAOperationRecord,
        receipt: DSAOperationReceipt,
    ) -> DSAReceiptBundle:
        return DSAReceiptBundle(
            bundle_id=self._new_bundle_id(record),
            request_key=record.command.request_key,
            operation_id=record.command.operation.operation_id,
            route_epoch=record.command.operation.route_epoch,
            input_generation_id=record.command.operation.input_generation_id,
            output_generation_id=record.command.operation.output_generation_id,
            source_manifest_id=None,
            token_prefix_digest=record.command.token_prefix_digest,
            data_compatibility_fingerprint=(record.command.cache_namespace_fingerprint),
            receipt_ids=(receipt.receipt_id,),
            aggregate_status="failed",
            raw_source_end=0,
            sparse_source_end=0,
            materialized_end=0,
            lease_descriptor_id=None,
            error_code=receipt.error_code or "receipt_failed",
        )

    @staticmethod
    def _minimum_frontier(
        receipts: tuple[DSAOperationReceipt, ...],
        receipt_kinds: frozenset[str],
    ) -> int:
        ends = tuple(
            receipt.range_end
            for receipt in receipts
            if receipt.receipt_kind in receipt_kinds
        )
        return min(ends, default=0)

    @staticmethod
    def _new_bundle_id(record: DSAOperationRecord) -> str:
        command = record.command
        identity = "\0".join(
            (
                command.request_key.process_instance_id,
                command.request_key.request_id,
                str(command.request_key.scope_id),
                command.operation.operation_id,
            )
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
        return f"bundle-{command.operation.operation_id}-{digest}"

    def _validate_stored_complete_bundle(self, record: DSAOperationRecord) -> None:
        bundle = record.bundle
        if bundle is None:
            raise DSAOperationError("ready record has no stored bundle")
        if record.bundle_id != bundle.bundle_id:
            raise DSAOperationError("record and stored bundle ids do not match")
        expected_bundle = self._build_complete_bundle(record)
        if bundle != expected_bundle:
            raise DSAOperationError("stored bundle does not match exact quorum")

    @staticmethod
    def _fail_pending_consumers(record: DSAOperationRecord) -> None:
        for obligation, status in record.consumer_status.items():
            if status == "pending":
                record.consumer_status[obligation] = "failed"

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
        record = self.match(
            _BundleRef(request_key=request_key, operation_id=operation_id)
        )
        if obligation not in record.consumer_status:
            raise DSAOperationError(f"unknown consumer obligation {obligation!r}")
        if terminal_status not in ("committed", "failed"):
            raise DSAOperationError(
                f"invalid consumer terminal status {terminal_status!r}"
            )

        current = record.consumer_status[obligation]
        if current != "pending" and current != terminal_status:
            raise DSAOperationError(
                f"consumer {obligation!r} is already terminal as {current}"
            )
        if terminal_status == "committed":
            if record.status not in ("ready", "activating"):
                raise DSAOperationError(
                    f"cannot commit consumer from operation status {record.status}"
                )
            self._validate_stored_complete_bundle(record)
        elif record.status not in ("ready", "activating", "failed"):
            raise DSAOperationError(
                f"cannot fail consumer from operation status {record.status}"
            )

        record.consumer_status[obligation] = terminal_status  # type: ignore[assignment]
        if all(status != "pending" for status in record.consumer_status.values()):
            if any(status == "failed" for status in record.consumer_status.values()):
                record.status = "failed"
                record.error_code = record.error_code or "consumer_failed"
            else:
                record.status = "committed"
            self._tombstone(record)

    def fail_operation(
        self,
        request_key: RequestKey,
        operation_id: str,
        error_code: str,
    ) -> DSAOperationRecord:
        """Fail an entire operation, fanning out to all obligations."""
        record = self.match(
            _BundleRef(request_key=request_key, operation_id=operation_id)
        )
        if record.status in ("ready", "activating", "committed"):
            raise DSAOperationError(
                f"late failure for terminal operation {operation_id} rejected"
            )
        if record.status == "failed":
            record.error_code = record.error_code or error_code
            self._fail_pending_consumers(record)
        else:
            record.status = "failed"
            record.error_code = error_code
            self._fail_pending_consumers(record)
        self._tombstone(record)
        return record

    def supersede(self, request_key: RequestKey, operation_id: str) -> None:
        """Mark an operation superseded (e.g. new generation took over)."""
        record = self.match(
            _BundleRef(request_key=request_key, operation_id=operation_id)
        )
        record.status = "superseded"
        record.error_code = "superseded"
        self._fail_pending_consumers(record)
        self._tombstone(record)

    def _tombstone(self, record: DSAOperationRecord) -> None:
        key = (record.command.request_key, record.command.operation.operation_id)
        self.records.pop(key, None)
        self.tombstones[key] = record
        if len(self.tombstones) > self.tombstone_capacity:
            # Bounded eviction: drop the oldest by insertion order.
            oldest = next(iter(self.tombstones))
            self.tombstones.pop(oldest, None)

    def expire_deadlines(
        self, *, now_ms: int | None = None
    ) -> list[DSAOperationRecord]:
        """Fail operations whose deadline has passed.

        Returns the failed records so the Scheduler can run per-obligation
        rollback (design section 9.5).
        """
        now = now_ms if now_ms is not None else int(time.time() * 1000)
        expired: list[DSAOperationRecord] = []
        for record in tuple(self.records.values()):
            if (
                record.status in ("issued", "collecting")
                and record.deadline_ms
                and now >= record.deadline_ms
            ):
                record.status = "failed"
                record.error_code = "deadline_expired"
                self._fail_pending_consumers(record)
                expired.append(record)
        return expired


@dataclass(frozen=True)
class _BundleRef:
    """Lightweight stand-in so :meth:`match` accepts bundle-originated lookups."""

    request_key: RequestKey
    operation_id: str
