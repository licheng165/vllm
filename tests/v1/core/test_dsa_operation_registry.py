# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace

import pytest

from vllm.v1.core.sched.dsa_operation_registry import (
    DSAOperationError,
    DSAOperationRegistry,
)
from vllm.v1.core.sched.dsa_types import (
    DSAOperationCommand,
    DSAOperationReceipt,
    DSAOperationRef,
    DSAReceiptExpectation,
    Guarantee,
    ObligationKind,
    ParticipantIdentity,
    ReceiptKind,
    RequestKey,
    StorageTier,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

REQUEST_KEY = RequestKey("scheduler-process", "request-1", 1)
OTHER_REQUEST_KEY = RequestKey("scheduler-process", "request-2", 2)
PARTICIPANT_0 = ParticipantIdentity("engine", "worker-process-0", 0, 0, 0, 0)
PARTICIPANT_1 = ParticipantIdentity("engine", "worker-process-1", 1, 0, 0, 0)


def _expectation(
    *,
    participant: ParticipantIdentity = PARTICIPANT_0,
    receipt_kind: ReceiptKind = "storage",
    kv_group: int = 0,
    layers: tuple[int, ...] = (0, 1),
    chunks: tuple[tuple[int, int], ...] = ((0, 128), (128, 256)),
    storage_tier: StorageTier = "local_cpu",
    minimum_guarantee: Guarantee = "local_cpu_pinned",
) -> DSAReceiptExpectation:
    return DSAReceiptExpectation(
        participant=participant,
        receipt_kind=receipt_kind,
        kv_group=kv_group,
        layers=layers,
        chunks=chunks,
        storage_tier=storage_tier,
        minimum_guarantee=minimum_guarantee,
    )


def _command(
    *,
    expectations: tuple[DSAReceiptExpectation, ...] | None = None,
    request_key: RequestKey = REQUEST_KEY,
    operation_id: str = "operation-1",
    range_start: int = 0,
    range_end: int = 256,
    accepted_end: int | None = None,
    obligations: frozenset[ObligationKind] = frozenset({"promotion"}),
) -> DSAOperationCommand:
    if expectations is None:
        expectations = (_expectation(),)
    return DSAOperationCommand(
        request_key=request_key,
        operation=DSAOperationRef(
            operation_id=operation_id,
            parent_operation_id=None,
            kind="store",
            obligations=obligations,
            range_start=range_start,
            range_end=range_end,
            input_generation_id="input-generation",
            output_generation_id="output-generation",
            route_epoch=7,
        ),
        accepted_end_at_issue=(range_end if accepted_end is None else accepted_end),
        token_prefix_digest="token-digest",
        cache_namespace_fingerprint="cache-namespace",
        expected_receipts=expectations,
    )


def _receipt(
    command: DSAOperationCommand,
    expectation: DSAReceiptExpectation,
    receipt_id: str,
    **changes: object,
) -> DSAOperationReceipt:
    receipt = DSAOperationReceipt(
        receipt_id=receipt_id,
        request_key=command.request_key,
        operation_id=command.operation.operation_id,
        receipt_kind=expectation.receipt_kind,
        route_epoch=command.operation.route_epoch,
        input_generation_id=command.operation.input_generation_id,
        output_generation_id=command.operation.output_generation_id,
        accepted_end_at_seal=command.accepted_end_at_issue,
        token_prefix_digest=command.token_prefix_digest,
        cache_namespace_fingerprint=command.cache_namespace_fingerprint,
        range_start=command.operation.range_start,
        range_end=command.operation.range_end,
        kv_group=expectation.kv_group,
        participant=expectation.participant,
        covered_layers=expectation.layers,
        covered_chunks=expectation.chunks,
        storage_tier=expectation.storage_tier,
        status="complete",
        lease_descriptor_id=None,
        guarantee=expectation.minimum_guarantee,
        error_code=None,
    )
    return replace(receipt, **changes)


def test_register_returns_record_and_keys_include_exact_coverage() -> None:
    expectations = (
        _expectation(layers=(0,)),
        _expectation(layers=(1,)),
    )
    command = _command(expectations=expectations)
    registry = DSAOperationRegistry()

    record = registry.register(command, deadline_ms=1000)

    assert record is registry.records[(REQUEST_KEY, "operation-1")]
    assert record.consumer_status == {"promotion": "pending"}
    assert registry.submit_receipt(_receipt(command, expectations[0], "r-0")) is None
    bundle = registry.submit_receipt(_receipt(command, expectations[1], "r-1"))
    assert bundle is not None
    assert len(record.receipts_by_expectation) == 2
    for key in record.receipts_by_expectation:
        assert isinstance(key, tuple)
        hash(key)


@pytest.mark.parametrize(
    "expectations",
    [
        (),
        (_expectation(), _expectation()),
        (
            _expectation(),
            _expectation(
                layers=(1, 0),
                chunks=((128, 256), (0, 128)),
            ),
        ),
        (_expectation(layers=(0, 0)),),
        (_expectation(layers=()),),
        (_expectation(chunks=()),),
        (_expectation(chunks=((0, 128), (0, 128))),),
        (_expectation(chunks=((0, 192), (128, 256))),),
        (_expectation(chunks=((0, 64), (128, 256))),),
        (_expectation(chunks=((0, 128),)),),
        (_expectation(chunks=((0, 0),)),),
        (_expectation(chunks=((0, 257),)),),
    ],
    ids=[
        "empty",
        "duplicate",
        "reordered-duplicate",
        "duplicate-layers",
        "empty-layers",
        "empty-chunks",
        "duplicate-chunks",
        "overlapping-chunks",
        "gapped-chunks",
        "incomplete-chunks",
        "empty-chunk",
        "chunk-outside-operation",
    ],
)
def test_register_rejects_invalid_expectations_atomically(
    expectations: tuple[DSAReceiptExpectation, ...],
) -> None:
    registry = DSAOperationRegistry()

    with pytest.raises(DSAOperationError):
        registry.register(_command(expectations=expectations), deadline_ms=1000)

    assert registry.records == {}


@pytest.mark.parametrize(
    ("range_start", "range_end", "accepted_end"),
    [
        (-1, 256, 256),
        (256, 256, 256),
        (257, 256, 256),
        (0, 257, 256),
    ],
)
def test_register_rejects_malformed_operation_ranges(
    range_start: int,
    range_end: int,
    accepted_end: int,
) -> None:
    registry = DSAOperationRegistry()

    with pytest.raises(DSAOperationError):
        registry.register(
            _command(
                range_start=range_start,
                range_end=range_end,
                accepted_end=accepted_end,
            ),
            deadline_ms=1000,
        )

    assert registry.records == {}


def test_register_rejects_live_and_tombstoned_duplicates() -> None:
    command = _command()
    registry = DSAOperationRegistry()
    registry.register(command, deadline_ms=1000)

    with pytest.raises(DSAOperationError):
        registry.register(command, deadline_ms=1000)

    registry.supersede(REQUEST_KEY, "operation-1")
    with pytest.raises(DSAOperationError):
        registry.register(command, deadline_ms=1000)


@pytest.mark.parametrize("status", ["complete", "failed"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("request_key", OTHER_REQUEST_KEY),
        ("operation_id", "other-operation"),
        ("route_epoch", 8),
        ("input_generation_id", "other-input"),
        ("output_generation_id", "other-output"),
        ("range_start", 1),
        ("range_end", 128),
        ("accepted_end_at_seal", 257),
        ("token_prefix_digest", "other-digest"),
        ("cache_namespace_fingerprint", "other-namespace"),
        ("participant", PARTICIPANT_1),
        ("receipt_kind", "source_seal"),
        ("kv_group", 1),
        ("storage_tier", "mooncake"),
        ("covered_layers", (0,)),
        ("covered_chunks", ((0, 128),)),
        ("guarantee", "remote_put_fenced"),
    ],
)
def test_submit_validates_every_command_binding_before_accepting(
    status: str,
    field: str,
    value: object,
) -> None:
    command = _command()
    expectation = command.expected_receipts[0]
    registry = DSAOperationRegistry()
    record = registry.register(command, deadline_ms=1000)
    receipt = _receipt(command, expectation, "receipt-1")
    if status == "failed":
        receipt = replace(
            receipt,
            status="failed",
            guarantee=None,
            error_code="worker-failed",
        )
    receipt = replace(receipt, **{field: value})

    with pytest.raises(DSAOperationError):
        registry.submit_receipt(receipt)

    assert record.status == "issued"
    assert record.receipts_by_expectation == {}
    assert record.receipts_by_id == {}
    assert record.bundle is None


@pytest.mark.parametrize(
    "changes",
    [
        {"covered_layers": (0, 0)},
        {"covered_chunks": ((0, 128), (0, 128))},
        {"receipt_id": ""},
        {"error_code": "error-on-success"},
    ],
)
def test_submit_rejects_malformed_or_internally_duplicate_receipts(
    changes: dict[str, object],
) -> None:
    command = _command()
    registry = DSAOperationRegistry()
    record = registry.register(command, deadline_ms=1000)
    receipt = replace(
        _receipt(command, command.expected_receipts[0], "receipt-1"),
        **changes,
    )

    with pytest.raises(DSAOperationError):
        registry.submit_receipt(receipt)

    assert record.status == "issued"
    assert record.receipts_by_id == {}


def test_receipt_duplicates_are_idempotent_and_conflicts_fail_closed() -> None:
    expectations = (
        _expectation(participant=PARTICIPANT_0),
        _expectation(participant=PARTICIPANT_1),
    )
    command = _command(expectations=expectations)
    registry = DSAOperationRegistry()
    record = registry.register(command, deadline_ms=1000)
    first = _receipt(command, expectations[0], "receipt-0")
    second = _receipt(command, expectations[1], "receipt-1")

    assert registry.submit_receipt(first) is None
    assert registry.submit_receipt(first) is None
    with pytest.raises(DSAOperationError):
        registry.submit_receipt(replace(first, token_prefix_digest="changed"))
    with pytest.raises(DSAOperationError):
        registry.submit_receipt(replace(first, receipt_id="receipt-other"))

    bundle = registry.submit_receipt(second)
    assert bundle is not None
    assert record.bundle is bundle
    assert registry.submit_receipt(second) is bundle

    registry.mark_consumer_terminal(
        REQUEST_KEY, "operation-1", "promotion", "committed"
    )
    assert registry.submit_receipt(second) is bundle
    with pytest.raises(DSAOperationError):
        registry.submit_receipt(replace(second, receipt_id="receipt-late"))


def test_receipt_ids_cannot_be_reused_by_another_operation() -> None:
    first_command = _command(operation_id="operation-1")
    second_command = _command(operation_id="operation-2")
    registry = DSAOperationRegistry()
    registry.register(first_command, deadline_ms=1000)
    second_record = registry.register(second_command, deadline_ms=1000)
    registry.submit_receipt(
        _receipt(first_command, first_command.expected_receipts[0], "shared-id")
    )

    with pytest.raises(DSAOperationError):
        registry.submit_receipt(
            _receipt(second_command, second_command.expected_receipts[0], "shared-id")
        )

    assert second_record.status == "issued"
    assert second_record.receipts_by_id == {}


def test_expected_failure_builds_one_failed_bundle_and_fans_out() -> None:
    command = _command(obligations=frozenset({"promotion", "prefill_export"}))
    expectation = command.expected_receipts[0]
    registry = DSAOperationRegistry()
    record = registry.register(command, deadline_ms=1000)
    failure = _receipt(
        command,
        expectation,
        "failed-receipt",
        status="failed",
        guarantee=None,
        error_code="put-failed",
    )

    bundle = registry.submit_receipt(failure)

    assert bundle is not None
    assert bundle.aggregate_status == "failed"
    assert bundle.error_code == "put-failed"
    assert bundle.receipt_ids == ("failed-receipt",)
    assert (bundle.raw_source_end, bundle.sparse_source_end) == (0, 0)
    assert record.status == "failed"
    assert record.error_code == "put-failed"
    assert record.bundle is bundle
    assert set(record.consumer_status.values()) == {"failed"}
    assert registry.submit_receipt(failure) is bundle
    with pytest.raises(DSAOperationError):
        registry.submit_receipt(replace(failure, error_code="changed"))
    with pytest.raises(DSAOperationError):
        registry.submit_receipt(replace(failure, receipt_id="another-failure"))


def test_failed_operation_drains_remaining_exact_set_receipts() -> None:
    expectations = (
        _expectation(participant=PARTICIPANT_0),
        _expectation(participant=PARTICIPANT_1),
        _expectation(
            participant=PARTICIPANT_0,
            receipt_kind="source_seal",
        ),
    )
    command = _command(expectations=expectations)
    registry = DSAOperationRegistry()
    record = registry.register(command, deadline_ms=1000)
    first_failure = _receipt(
        command,
        expectations[0],
        "failed-0",
        status="failed",
        guarantee=None,
        error_code="store-failed",
    )
    bundle = registry.submit_receipt(first_failure)
    assert bundle is not None
    registry.fail_operation(REQUEST_KEY, "operation-1", "store-failed")

    success = _receipt(command, expectations[1], "success-1")
    second_failure = _receipt(
        command,
        expectations[2],
        "failed-2",
        status="failed",
        guarantee=None,
        error_code="seal-failed",
    )

    assert registry.submit_receipt(success) is bundle
    assert registry.submit_receipt(second_failure) is bundle
    assert record.bundle is bundle
    assert set(record.receipts_by_id) == {"failed-0", "success-1", "failed-2"}


def test_failed_bundle_is_deterministic_regardless_of_prior_successes() -> None:
    expectations = (
        _expectation(participant=PARTICIPANT_0),
        _expectation(participant=PARTICIPANT_1),
    )
    command = _command(expectations=expectations)
    success = _receipt(command, expectations[0], "success-receipt")
    failure = _receipt(
        command,
        expectations[1],
        "failed-receipt",
        status="failed",
        guarantee=None,
        error_code="worker-failed",
    )
    first_registry = DSAOperationRegistry()
    second_registry = DSAOperationRegistry()
    first_registry.register(command, deadline_ms=1000)
    second_registry.register(command, deadline_ms=1000)

    first_bundle = first_registry.submit_receipt(failure)
    assert second_registry.submit_receipt(success) is None
    second_bundle = second_registry.submit_receipt(failure)

    assert first_bundle == second_bundle


def test_unexpected_failed_receipt_cannot_poison_operation() -> None:
    command = _command()
    expectation = command.expected_receipts[0]
    registry = DSAOperationRegistry()
    record = registry.register(command, deadline_ms=1000)
    unexpected_failure = _receipt(
        command,
        expectation,
        "failed-receipt",
        participant=PARTICIPANT_1,
        status="failed",
        guarantee=None,
        error_code="worker-failed",
    )

    with pytest.raises(DSAOperationError):
        registry.submit_receipt(unexpected_failure)

    assert record.status == "issued"
    assert record.error_code is None
    assert record.bundle is None
    assert record.receipts_by_id == {}
    assert registry.submit_receipt(_receipt(command, expectation, "valid")) is not None


def test_success_bundle_is_deterministic_and_requires_exact_quorum() -> None:
    expectations = (
        _expectation(participant=PARTICIPANT_0),
        _expectation(participant=PARTICIPANT_1),
        _expectation(participant=PARTICIPANT_0, receipt_kind="source_seal"),
        _expectation(participant=PARTICIPANT_1, receipt_kind="source_seal"),
        _expectation(
            participant=PARTICIPANT_0,
            receipt_kind="npu_materialization",
            storage_tier="npu",
            minimum_guarantee="npu_materialized",
        ),
        _expectation(
            participant=PARTICIPANT_1,
            receipt_kind="npu_materialization",
            storage_tier="npu",
            minimum_guarantee="npu_materialized",
        ),
    )
    command = _command(expectations=expectations)
    receipts = tuple(
        _receipt(command, expectation, f"receipt-{index}")
        for index, expectation in enumerate(expectations)
    )
    first_registry = DSAOperationRegistry()
    second_registry = DSAOperationRegistry()
    first_record = first_registry.register(command, deadline_ms=1000)
    second_record = second_registry.register(command, deadline_ms=1000)

    first_bundle = None
    for index, receipt in enumerate(receipts):
        first_bundle = first_registry.submit_receipt(receipt)
        assert (first_bundle is None) is (index < len(receipts) - 1)
    second_bundle = None
    for index, receipt in enumerate(reversed(receipts)):
        second_bundle = second_registry.submit_receipt(receipt)
        assert (second_bundle is None) is (index < len(receipts) - 1)

    assert first_bundle is not None
    assert first_bundle == second_bundle
    assert first_bundle.receipt_ids == tuple(receipt.receipt_id for receipt in receipts)
    assert first_bundle.aggregate_status == "complete"
    assert first_bundle.raw_source_end == 256
    assert first_bundle.sparse_source_end == 256
    assert first_bundle.materialized_end == 256
    assert first_record.bundle is first_bundle
    assert second_record.bundle is second_bundle


def test_frontier_aggregation_uses_minimum_for_each_receipt_kind() -> None:
    command = _command()
    expectation = command.expected_receipts[0]
    short = _receipt(command, expectation, "short", range_end=128)
    long = _receipt(command, expectation, "long", range_end=256)
    materialized = replace(long, receipt_kind="npu_materialization")

    assert (
        DSAOperationRegistry._minimum_frontier(
            (long, short, materialized), frozenset({"storage"})
        )
        == 128
    )
    assert (
        DSAOperationRegistry._minimum_frontier(
            (long, short, materialized), frozenset({"npu_materialization"})
        )
        == 256
    )
    assert (
        DSAOperationRegistry._minimum_frontier(
            (long, short), frozenset({"source_seal"})
        )
        == 0
    )


def test_consume_ready_bundle_revalidates_stored_bundle() -> None:
    command = _command()
    registry = DSAOperationRegistry()
    record = registry.register(command, deadline_ms=1000)
    bundle = registry.submit_receipt(
        _receipt(command, command.expected_receipts[0], "receipt-1")
    )
    assert bundle is not None

    assert (
        registry.consume_ready_bundle(bundle.bundle_id, REQUEST_KEY, "operation-1")
        is record
    )
    with pytest.raises(DSAOperationError):
        registry.consume_ready_bundle("wrong-bundle", REQUEST_KEY, "operation-1")

    record.bundle = replace(bundle, raw_source_end=255)
    with pytest.raises(DSAOperationError):
        registry.consume_ready_bundle(bundle.bundle_id, REQUEST_KEY, "operation-1")


def test_deadline_expiry_stays_live_until_controller_rollback() -> None:
    command = _command()
    registry = DSAOperationRegistry()
    record = registry.register(command, deadline_ms=100)

    assert registry.expire_deadlines(now_ms=99) == []
    assert registry.expire_deadlines(now_ms=100) == [record]
    assert record.status == "failed"
    assert record.error_code == "deadline_expired"
    assert set(record.consumer_status.values()) == {"failed"}
    assert (REQUEST_KEY, "operation-1") in registry.records
    assert registry.tombstones == {}
    assert registry.expire_deadlines(now_ms=101) == []

    failed = registry.fail_operation(REQUEST_KEY, "operation-1", "rollback-finished")
    assert failed is record
    assert record.error_code == "deadline_expired"
    assert (REQUEST_KEY, "operation-1") not in registry.records
    assert registry.tombstones[(REQUEST_KEY, "operation-1")] is record


def test_consumer_terminals_are_validated_and_tombstone_after_all_consumers() -> None:
    command = _command(obligations=frozenset({"promotion", "prefill_export"}))
    registry = DSAOperationRegistry()
    record = registry.register(command, deadline_ms=1000)

    with pytest.raises(DSAOperationError):
        registry.mark_consumer_terminal(
            REQUEST_KEY, "operation-1", "promotion", "committed"
        )

    bundle = registry.submit_receipt(
        _receipt(command, command.expected_receipts[0], "receipt-1")
    )
    assert bundle is not None
    with pytest.raises(DSAOperationError):
        registry.mark_consumer_terminal(
            REQUEST_KEY, "operation-1", "unknown", "committed"
        )
    with pytest.raises(DSAOperationError):
        registry.mark_consumer_terminal(
            REQUEST_KEY, "operation-1", "promotion", "pending"
        )

    registry.mark_consumer_terminal(
        REQUEST_KEY, "operation-1", "promotion", "committed"
    )
    assert record.status == "ready"
    assert (REQUEST_KEY, "operation-1") in registry.records
    with pytest.raises(DSAOperationError):
        registry.mark_consumer_terminal(
            REQUEST_KEY, "operation-1", "promotion", "failed"
        )

    registry.mark_consumer_terminal(
        REQUEST_KEY, "operation-1", "prefill_export", "committed"
    )
    assert record.status == "committed"
    assert (REQUEST_KEY, "operation-1") not in registry.records
    assert registry.tombstones[(REQUEST_KEY, "operation-1")] is record
    assert (
        registry.submit_receipt(
            _receipt(command, command.expected_receipts[0], "receipt-1")
        )
        is bundle
    )


def test_failed_consumer_makes_final_record_failed() -> None:
    command = _command(obligations=frozenset({"promotion", "prefill_export"}))
    registry = DSAOperationRegistry()
    record = registry.register(command, deadline_ms=1000)
    registry.submit_receipt(
        _receipt(command, command.expected_receipts[0], "receipt-1")
    )

    registry.mark_consumer_terminal(REQUEST_KEY, "operation-1", "promotion", "failed")
    registry.mark_consumer_terminal(
        REQUEST_KEY, "operation-1", "prefill_export", "committed"
    )

    assert record.status == "failed"
    assert record.error_code == "consumer_failed"
    assert registry.tombstones[(REQUEST_KEY, "operation-1")] is record


def test_superseded_record_rejects_new_late_receipts_but_keeps_idempotency() -> None:
    expectations = (
        _expectation(participant=PARTICIPANT_0),
        _expectation(participant=PARTICIPANT_1),
    )
    command = _command(expectations=expectations)
    registry = DSAOperationRegistry()
    record = registry.register(command, deadline_ms=1000)
    accepted = _receipt(command, expectations[0], "receipt-0")
    late = _receipt(command, expectations[1], "receipt-1")
    assert registry.submit_receipt(accepted) is None

    registry.supersede(REQUEST_KEY, "operation-1")

    assert record.status == "superseded"
    assert set(record.consumer_status.values()) == {"failed"}
    assert registry.submit_receipt(accepted) is None
    with pytest.raises(DSAOperationError):
        registry.submit_receipt(late)
