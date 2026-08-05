# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from dataclasses import replace

import pytest

from vllm.v1.core.sched import dsa_controller as dsa_controller_module
from vllm.v1.core.sched.dsa_controller import (
    DSAController,
    DSAControllerConfig,
)
from vllm.v1.core.sched.dsa_operation_registry import DSAOperationError
from vllm.v1.core.sched.dsa_types import (
    DSAControlEvent,
    DSAExecutionReceipt,
    DSAOperationCommand,
    DSAOperationReceipt,
    DSAReceiptExpectation,
    ParticipantIdentity,
    RequestKey,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _make_controller(node_role: str = "decode") -> DSAController:
    return DSAController(
        DSAControllerConfig(
            threshold=8192,
            max_model_len=32768,
            block_size=128,
            chunk_size=256,
            window_size=256,
            index_topk=2048,
            query_width=2,
            scratch_capacity=4096,
            node_role=node_role,
            deployment_mode=(
                "standalone" if node_role == "standalone" else "pd"
            ),
            data_compatibility_fingerprint="data-fingerprint",
            instance_capability_digest="instance-digest",
        ),
        process_instance_id="scheduler-process",
    )


def _make_lifecycle_controller(*, participants: int = 1) -> DSAController:
    participant_ids = tuple(
        ParticipantIdentity("engine", f"worker-{rank}", rank, 0, 0, rank)
        for rank in range(participants)
    )
    return DSAController(
        DSAControllerConfig(
            threshold=64,
            max_model_len=1024,
            block_size=16,
            chunk_size=16,
            window_size=16,
            index_topk=16,
            query_width=2,
            scratch_capacity=32,
            node_role="standalone",
            deployment_mode="standalone",
            data_compatibility_fingerprint="data-fingerprint",
            instance_capability_digest="instance-digest",
            participants=participant_ids,
            required_layers=(0, 1),
            kv_groups=(0,),
        ),
        process_instance_id="scheduler-process",
    )


def _operation_receipt(
    command: DSAOperationCommand,
    expectation: DSAReceiptExpectation,
    receipt_id: str,
    *,
    status: str = "complete",
    error_code: str | None = None,
) -> DSAOperationReceipt:
    return DSAOperationReceipt(
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
        status=status,  # type: ignore[arg-type]
        lease_descriptor_id=None,
        guarantee=(expectation.minimum_guarantee if status == "complete" else None),
        error_code=error_code,
    )


def _complete_command(
    controller: DSAController,
    command: DSAOperationCommand,
):
    result = None
    for index, expectation in enumerate(command.expected_receipts):
        result = controller.consume_operation_receipt(
            _operation_receipt(
                command,
                expectation,
                f"{command.operation.operation_id}-receipt-{index}",
            )
        )
    return result


def _issue_initial_store(
    controller: DSAController,
    request_id: str = "request",
):
    state = controller.initialize_state(request_id, 64)
    controller.attach_state(state)
    execution_seq = controller.next_execution_seq()
    controller.build_route_snapshots(
        [request_id],
        {request_id: 64},
        execution_seq=execution_seq,
        token_prefix_digests={request_id: "execution-digest"},
    )
    controller.consume_completed_execution(
        request_id,
        DSAExecutionReceipt(
            request_key=state.request_key,
            execution_seq=execution_seq,
            route_epoch=state.route_epoch,
            accepted_end_at_execution=64,
            completed_canonical_end=64,
            external_computed_end=63,
            initial_prefill_complete=True,
            min_position_of_next_target_rows=64,
            token_prefix_digest="execution-digest",
        ),
    )
    controller.consume_accepted_end(request_id, 64, "accepted-digest")
    controller.maybe_advance(request_id)
    commands = controller.take_pending_commands()
    assert len(commands) == 1
    return state, commands[0]


def _capture_route_events(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    events: list[dict] = []

    def capture(message: str, *args: object, **kwargs: object) -> None:
        rendered = message % args
        marker = "[DSA_OFFLOAD] "
        assert rendered.startswith(marker)
        events.append(json.loads(rendered[len(marker) :]))

    monkeypatch.setattr(
        dsa_controller_module.logger, "isEnabledFor", lambda level: True
    )
    monkeypatch.setattr(dsa_controller_module.logger, "info", capture)
    return events


def test_positive_threshold_request_requires_preemption_quiesce() -> None:
    controller = _make_controller()
    state = controller.initialize_state("request", 64)
    controller.attach_state(state)

    assert controller.requires_preemption_quiesce("request")
    assert not controller.requires_preemption_quiesce("unknown")

    controller.finish_request("request")
    assert not controller.requires_preemption_quiesce("request")


def test_pd_decode_external_prefix_requires_import_frontier_receipt() -> None:
    decode = _make_controller("decode")
    decode_state = decode.initialize_state("decode-request", 64)
    decode.attach_state(decode_state)
    prefill = _make_controller("prefill")
    prefill_state = prefill.initialize_state("prefill-request", 64)
    prefill.attach_state(prefill_state)

    assert decode.requires_import_frontier_receipt("decode-request", 63)
    assert not decode.requires_import_frontier_receipt("decode-request", 0)
    assert not decode.requires_import_frontier_receipt("unknown", 63)
    assert not prefill.requires_import_frontier_receipt("prefill-request", 63)


@pytest.mark.parametrize(
    ("node_role", "num_tokens", "mode", "selected_path", "reason"),
    [
        ("prefill", 8191, "resident", "resident", "below_threshold"),
        (
            "prefill",
            8192,
            "promoting",
            "dsa_sparse",
            "at_or_above_threshold",
        ),
        ("decode", 8191, "resident", "resident", "below_threshold"),
        (
            "decode",
            8192,
            "promoting",
            "dsa_sparse",
            "at_or_above_threshold",
        ),
    ],
)
def test_route_log_identifies_initial_pd_path_once(
    monkeypatch: pytest.MonkeyPatch,
    node_role: str,
    num_tokens: int,
    mode: str,
    selected_path: str,
    reason: str,
) -> None:
    events = _capture_route_events(monkeypatch)
    controller = _make_controller(node_role)
    state = controller.initialize_state("request-1", num_tokens)
    controller.attach_state(state)

    controller.build_route_snapshots(["request-1"], {"request-1": num_tokens})
    controller.build_route_snapshots(["request-1"], {"request-1": num_tokens})

    assert len(events) == 1
    event = events[0]
    assert event["schema"] == "dsa_offload.v1"
    assert event["event"] == "route.classify"
    assert "request_id" not in event
    assert event["request_process_instance"] == "scheduler-process"
    assert event["scope_id"] == 1
    assert event["node_role"] == node_role
    assert event["deployment_mode"] == "pd"
    assert event["mode"] == mode
    assert event["selected_path"] == selected_path
    assert event["execution_path"] == "resident_absolute"
    assert event["reason"] == reason
    assert event["accepted_end"] == num_tokens
    assert event["threshold"] == 8192
    assert event["route_authority"] == "scheduler_state"
    assert state.transfer_plan.must_import_prefix is (node_role == "decode")
    assert state.transfer_plan.must_export_prefill is (node_role == "prefill")
    assert state.transfer_plan.remote_handoff_required is (
        node_role == "prefill"
    )
    assert state.transfer_plan.must_persist_decode_windows is (
        node_role == "decode" and num_tokens >= 8192
    )


def test_standalone_promotion_requires_decode_window_persistence() -> None:
    controller = _make_controller("standalone")
    state = controller.initialize_state("request-standalone", 8192)

    assert state.transfer_plan.must_persist_decode_windows is True


def test_disabled_config_preserves_pd_transfer_role() -> None:
    config = DSAControllerConfig.disabled(
        block_size=128,
        chunk_size=256,
        node_role="decode",
        deployment_mode="pd",
    )
    controller = DSAController(config, process_instance_id="scheduler-process")

    state = controller.initialize_state("request-legacy-pd", 1024)

    assert config.deployment_mode == "pd"
    assert state.transfer_plan.must_import_prefix is True
    assert state.transfer_plan.must_export_prefill is False
    assert state.transfer_plan.must_persist_decode_windows is False


def test_accepted_end_crosses_threshold_before_completion_frontier() -> None:
    controller = _make_controller("decode")
    state = controller.initialize_state("request-crossing", 8191)
    controller.attach_state(state)

    controller.consume_accepted_end("request-crossing", 8192)

    assert state.route_state.value == "promoting"
    assert state.threshold_crossed_at == 8192
    assert state.completed_canonical_end == 0
    assert state.promotion_desired_end == 0
    assert state.transfer_plan.must_import_prefix is True
    assert state.transfer_plan.must_persist_decode_windows is True


def test_route_log_request_id_is_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_ASCEND_DSA_DIAG_INCLUDE_REQUEST_ID", "1")
    events = _capture_route_events(monkeypatch)
    controller = _make_controller("prefill")

    controller.initialize_state("request-visible", 8191)

    assert events[0]["request_id"] == "request-visible"


def test_route_log_does_no_work_when_info_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller("decode")
    monkeypatch.setattr(
        dsa_controller_module.logger, "isEnabledFor", lambda level: False
    )
    monkeypatch.setattr(
        dsa_controller_module.logger,
        "info",
        lambda *args, **kwargs: pytest.fail("disabled route log was emitted"),
    )

    controller.initialize_state("request-disabled", 8192)

    assert controller._event_seq == 0


def test_route_log_distinguishes_selection_from_sparse_execution_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _capture_route_events(monkeypatch)
    controller = _make_controller("decode")
    state = controller.initialize_state("request-2", 8191)
    controller.attach_state(state)
    execution_seq = controller.next_execution_seq()
    controller.build_route_snapshots(
        ["request-2"],
        {"request-2": 8191},
        execution_seq=execution_seq,
        token_prefix_digests={"request-2": "execution-digest"},
    )

    controller.consume_completed_execution(
        "request-2",
        DSAExecutionReceipt(
            request_key=state.request_key,
            execution_seq=execution_seq,
            accepted_end_at_execution=8191,
            completed_canonical_end=8191,
            external_computed_end=8190,
            initial_prefill_complete=True,
            route_epoch=0,
            min_position_of_next_target_rows=8191,
            token_prefix_digest="execution-digest",
        ),
    )
    controller.consume_accepted_end("request-2", 8192, "accepted-digest")
    controller.maybe_advance("request-2")
    (store_command,) = controller.take_pending_commands()
    assert _complete_command(controller, store_command) is None
    (activation_command,) = controller.take_pending_commands()
    release_plan = _complete_command(controller, activation_command)
    assert release_plan is not None
    assert state.route_state.value == "promoting"
    controller.commit_release_plan(
        release_plan,
        release_plan.release_end,
    )

    assert [event["mode"] for event in events] == [
        "resident",
        "promoting",
        "sparse",
    ]

    promoting = events[1]
    assert promoting["previous_selected_path"] == "resident"
    assert promoting["selected_path"] == "dsa_sparse"
    assert promoting["previous_execution_path"] == "resident_absolute"
    assert promoting["execution_path"] == "resident_absolute"
    assert promoting["reason"] == "context_threshold_crossed"

    sparse = events[2]
    assert sparse["previous_selected_path"] == "dsa_sparse"
    assert sparse["selected_path"] == "dsa_sparse"
    assert sparse["previous_execution_path"] == "resident_absolute"
    assert sparse["execution_path"] == "sparse_remap"
    assert sparse["reason"] == "source_activation_committed"
    assert sparse["sparse_source_end"] == 7936
    assert sparse["remap_end"] == 7936
    assert sparse["release_end"] == 7936
    assert sparse["source_generation_id"] == (
        store_command.operation.output_generation_id
    )
    assert sparse["receipt_bundle_id"] == release_plan.store_bundle_id


def test_store_is_registered_before_ordered_command_publication() -> None:
    controller = _make_lifecycle_controller(participants=2)
    state, store_command = _issue_initial_store(controller)

    store_key = (state.request_key, store_command.operation.operation_id)
    assert store_key in controller.operations.records
    assert store_command.operation.operation_id == (
        "promotion-scheduler-process-1-1"
    )
    assert store_command.operation.output_generation_id == (
        "generation-scheduler-process-1-1"
    )
    assert len(store_command.expected_receipts) == 4
    assert {
        expectation.receipt_kind
        for expectation in store_command.expected_receipts
    } == {"storage", "source_seal"}
    assert all(
        expectation.chunks
        == ((0, 16), (16, 32), (32, 48), (48, 64))
        for expectation in store_command.expected_receipts
    )

    assert _complete_command(controller, store_command) is None
    assert state.route_state.value == "promoting"
    assert state.release_end == 0
    (activation_command,) = controller.take_pending_commands()
    activation_key = (
        state.request_key,
        activation_command.operation.operation_id,
    )
    assert activation_key in controller.operations.records
    assert activation_command.operation.parent_operation_id == (
        store_command.operation.operation_id
    )
    assert activation_command.operation.operation_id == (
        "source_activation-scheduler-process-1-2"
    )


def test_exact_activation_returns_plan_without_early_sparse_then_commits() -> None:
    controller = _make_lifecycle_controller(participants=2)
    state, store_command = _issue_initial_store(controller)
    _complete_command(controller, store_command)
    (activation_command,) = controller.take_pending_commands()

    first = activation_command.expected_receipts[0]
    assert (
        controller.consume_operation_receipt(
            _operation_receipt(activation_command, first, "activation-first")
        )
        is None
    )
    assert state.route_state.value == "promoting"
    assert state.remap_end == state.release_end == 0

    plan = None
    for index, expectation in enumerate(
        activation_command.expected_receipts[1:],
        start=1,
    ):
        plan = controller.consume_operation_receipt(
            _operation_receipt(
                activation_command,
                expectation,
                f"activation-{index}",
            )
        )
    assert plan is not None
    assert state.route_state.value == "promoting"
    assert state.active_source_generation_id is None
    assert plan.release_end == 64

    controller.validate_release_plan(plan)
    controller.commit_release_plan(plan, 64)

    assert state.route_state.value == "sparse"
    assert state.active_source_generation_id == plan.source_generation_id
    assert state.remap_end == state.release_end == 64
    assert state.route_epoch == plan.next_route_epoch
    assert state.window_anchor == state.next_window_start == 64


def test_failed_promotion_receipt_falls_back_without_release() -> None:
    controller = _make_lifecycle_controller()
    state, store_command = _issue_initial_store(controller)
    expectation = store_command.expected_receipts[0]

    assert (
        controller.consume_operation_receipt(
            _operation_receipt(
                store_command,
                expectation,
                "store-failed",
                status="failed",
                error_code="store-io-error",
            )
        )
        is None
    )

    assert state.route_state.value == "fallback_resident"
    assert state.promotion_inflight is None
    assert state.active_source_generation_id is None
    assert state.remap_end == state.release_end == 0
    assert controller.take_pending_commands() == ()


def test_failed_promotion_drains_all_local_failure_receipts() -> None:
    controller = _make_lifecycle_controller()
    state, store_command = _issue_initial_store(controller)

    for index, expectation in enumerate(store_command.expected_receipts):
        assert (
            controller.consume_operation_receipt(
                _operation_receipt(
                    store_command,
                    expectation,
                    f"store-failed-{index}",
                    status="failed",
                    error_code="store-io-error",
                )
            )
            is None
        )

    assert state.route_state.value == "fallback_resident"
    tombstone = controller.operations.tombstones[
        (state.request_key, store_command.operation.operation_id)
    ]
    assert len(tombstone.receipts_by_id) == len(
        store_command.expected_receipts
    )


def test_failed_window_activation_preserves_old_active_source() -> None:
    controller = _make_lifecycle_controller()
    state, store_command = _issue_initial_store(controller)
    _complete_command(controller, store_command)
    (activation_command,) = controller.take_pending_commands()
    first_plan = _complete_command(controller, activation_command)
    assert first_plan is not None
    controller.commit_release_plan(first_plan, first_plan.release_end)

    old_source = state.active_source_generation_id
    old_bundle = state.active_source_receipt_bundle_id
    old_digest = state.active_token_prefix_digest
    old_frontiers = (state.remap_end, state.release_end, state.next_window_start)

    controller.consume_accepted_end("request", 80, "accepted-digest-80")
    execution_seq = controller.next_execution_seq()
    snapshots = controller.build_route_snapshots(
        ["request"],
        {"request": 80},
        execution_seq=execution_seq,
        token_prefix_digests={"request": "execution-digest-80"},
    )
    lease = snapshots["request"].source_lease
    assert lease is not None
    controller.consume_completed_execution(
        "request",
        DSAExecutionReceipt(
            request_key=state.request_key,
            execution_seq=execution_seq,
            route_epoch=state.route_epoch,
            accepted_end_at_execution=80,
            completed_canonical_end=80,
            external_computed_end=79,
            initial_prefill_complete=True,
            min_position_of_next_target_rows=80,
            token_prefix_digest="execution-digest-80",
            released_source_lease_id=lease.source_lease_id,
        ),
    )
    controller.maybe_advance("request")
    (window_command,) = controller.take_pending_commands()
    _complete_command(controller, window_command)
    (rolling_activation,) = controller.take_pending_commands()

    expectation = rolling_activation.expected_receipts[0]
    controller.consume_operation_receipt(
        _operation_receipt(
            rolling_activation,
            expectation,
            "activation-failed",
            status="failed",
            error_code="activation-fence-failed",
        )
    )

    assert state.route_state.value == "sparse"
    assert state.active_source_generation_id == old_source
    assert state.active_source_receipt_bundle_id == old_bundle
    assert state.active_token_prefix_digest == old_digest
    assert (state.remap_end, state.release_end, state.next_window_start) == (
        old_frontiers
    )


def test_one_execution_sequence_is_shared_by_all_route_snapshots() -> None:
    controller = _make_lifecycle_controller()
    resident = controller.initialize_state("resident", 32)
    promoting = controller.initialize_state("promoting", 64)
    controller.attach_state(resident)
    controller.attach_state(promoting)
    execution_seq = controller.next_execution_seq()

    snapshots = controller.build_route_snapshots(
        ["resident", "promoting"],
        {"resident": 32, "promoting": 64},
        execution_seq=execution_seq,
        token_prefix_digests={
            "resident": "resident-digest",
            "promoting": "promoting-digest",
        },
    )

    assert {snapshot.execution_seq for snapshot in snapshots.values()} == {
        execution_seq
    }
    assert execution_seq > 0
    assert snapshots["resident"].source_lease is None
    assert snapshots["promoting"].source_lease is None


def test_execution_bound_uses_prior_completion_plus_scheduled_rows() -> None:
    controller = _make_lifecycle_controller()
    state = controller.initialize_state("request", 64)
    controller.attach_state(state)
    state.completed_canonical_end = 10
    execution_seq = controller.next_execution_seq()

    controller.build_route_snapshots(
        ["request"],
        {"request": 64},
        scheduled_token_counts={"request": 3},
        execution_seq=execution_seq,
        token_prefix_digests={"request": "execution-digest"},
    )

    expectation = state.pending_executions[execution_seq]
    assert expectation.maximum_completed_canonical_end == 13

    with pytest.raises(DSAOperationError, match="scheduled execution bound"):
        controller.consume_completed_execution(
            "request",
            DSAExecutionReceipt(
                request_key=state.request_key,
                execution_seq=execution_seq,
                route_epoch=state.route_epoch,
                accepted_end_at_execution=64,
                completed_canonical_end=14,
                external_computed_end=10,
                initial_prefill_complete=False,
                min_position_of_next_target_rows=14,
                token_prefix_digest="execution-digest",
            ),
        )


@pytest.mark.parametrize(
    "change",
    [
        {"request_key": RequestKey("other", "request", 1)},
        {"route_epoch": 1},
        {"execution_seq": 2},
        {"accepted_end_at_execution": 63},
        {"completed_canonical_end": 65},
        {"external_computed_end": 65},
        {"initial_prefill_complete": False},
        {"token_prefix_digest": "wrong-digest"},
        {"released_source_lease_id": "unexpected-lease"},
    ],
)
def test_execution_receipt_mismatch_fails_without_frontier_mutation(
    change: dict[str, object],
) -> None:
    controller = _make_lifecycle_controller()
    state = controller.initialize_state("request", 64)
    controller.attach_state(state)
    execution_seq = controller.next_execution_seq()
    controller.build_route_snapshots(
        ["request"],
        {"request": 64},
        execution_seq=execution_seq,
        token_prefix_digests={"request": "execution-digest"},
    )
    receipt = DSAExecutionReceipt(
        request_key=state.request_key,
        execution_seq=execution_seq,
        route_epoch=state.route_epoch,
        accepted_end_at_execution=64,
        completed_canonical_end=64,
        external_computed_end=63,
        initial_prefill_complete=True,
        min_position_of_next_target_rows=64,
        token_prefix_digest="execution-digest",
    )

    with pytest.raises(DSAOperationError):
        controller.consume_completed_execution(
            "request",
            replace(receipt, **change),
        )

    assert state.completed_canonical_end == 0
    assert state.external_computed_end == 0
    assert state.last_completed_execution_seq == 0


def test_worker_ready_event_is_rejected() -> None:
    controller = _make_lifecycle_controller()
    state = controller.initialize_state("request", 64)
    controller.attach_state(state)

    with pytest.raises(DSAOperationError, match="ready event"):
        controller.consume_event(
            "request",
            DSAControlEvent(
                request_key=state.request_key,
                operation_id="worker-ready",
                route_epoch=state.route_epoch,
                input_generation_id=None,
                output_generation_id=None,
                kind="promotion_ready",
            ),
        )


def test_operation_receipt_after_finish_returns_none() -> None:
    controller = _make_lifecycle_controller()
    state, store_command = _issue_initial_store(controller)
    controller.finish_request("request")
    expectation = store_command.expected_receipts[0]
    receipt = _operation_receipt(store_command, expectation, "late-store-0")
    assert controller.consume_operation_receipt(receipt) is None


def test_consume_event_after_finish_is_noop() -> None:
    controller = _make_lifecycle_controller()
    state = controller.initialize_state("request", 64)
    controller.attach_state(state)
    controller.finish_request("request")
    controller.consume_event(
        "request",
        DSAControlEvent(
            request_key=state.request_key,
            operation_id="source-revoke",
            route_epoch=state.route_epoch,
            input_generation_id=None,
            output_generation_id=None,
            kind="source_revoked",
        ),
    )


def test_duplicate_store_receipt_is_idempotent() -> None:
    controller = _make_lifecycle_controller()
    state, store_command = _issue_initial_store(controller)
    _complete_command(controller, store_command)
    for index, expectation in enumerate(store_command.expected_receipts):
        receipt_id = f"{store_command.operation.operation_id}-receipt-{index}"
        receipt = _operation_receipt(store_command, expectation, receipt_id)
        assert controller.consume_operation_receipt(receipt) is None
