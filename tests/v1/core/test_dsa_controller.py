# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.v1.core.sched import dsa_controller as dsa_controller_module
from vllm.v1.core.sched.dsa_controller import (
    DSAController,
    DSAControllerConfig,
    ExecutionReceipt,
)
from vllm.v1.core.sched.dsa_types import (
    DSAControlEvent,
    DSAOperationRef,
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
            deployment_mode="pd",
            data_compatibility_fingerprint="data-fingerprint",
            instance_capability_digest="instance-digest",
        ),
        process_instance_id="scheduler-process",
    )


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
    assert state.transfer_plan.must_persist_decode_windows is (
        node_role == "decode" and num_tokens >= 8192
    )


def test_standalone_promotion_requires_decode_window_persistence() -> None:
    controller = _make_controller("standalone")
    state = controller.initialize_state("request-standalone", 8192)

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
    controller.build_route_snapshots(["request-2"], {"request-2": 8191})

    controller.consume_completed_execution(
        "request-2",
        ExecutionReceipt(
            request_key=state.request_key,
            execution_seq=1,
            completed_canonical_end=8192,
            external_computed_end=8191,
            initial_prefill_complete=True,
            route_epoch=0,
        ),
    )
    controller.build_route_snapshots(["request-2"], {"request-2": 8192})

    state.completed_canonical_end = 8448
    state.latest_sealed_raw_source_end = 8192
    state.latest_sealed_sparse_source_end = 8192
    state.latest_sealed_generation_id = "generation-1"
    state.source_activation_inflight = DSAOperationRef(
        operation_id="activation-1",
        parent_operation_id="promotion-1",
        kind="source_activation",
        obligations=frozenset({"source_activation"}),
        range_start=0,
        range_end=8192,
        input_generation_id="generation-1",
        output_generation_id="generation-1",
        route_epoch=2,
    )
    controller.consume_event(
        "request-2",
        DSAControlEvent(
            request_key=state.request_key,
            operation_id="activation-1",
            route_epoch=2,
            input_generation_id="generation-1",
            output_generation_id="generation-1",
            kind="source_activation_ready",
            receipt_bundle_id="bundle-1",
        ),
    )
    controller.build_route_snapshots(["request-2"], {"request-2": 8448})
    controller.build_route_snapshots(["request-2"], {"request-2": 8448})

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
    assert sparse["reason"] == "source_activation_ready"
    assert sparse["sparse_source_end"] == 8192
    assert sparse["remap_end"] == 8192
    assert sparse["release_end"] == 0
    assert sparse["source_generation_id"] == "generation-1"
    assert sparse["receipt_bundle_id"] == "bundle-1"
