# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from unittest.mock import patch, sentinel

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (  # noqa: E501
    ExampleConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_transfer_state import (
    ensure_kv_transfer_initialized,
    get_kv_transfer_group,
)
from vllm.forward_context import set_forward_context
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorModelRunnerMixin

# Importing utils registers TestExampleConnector with the factory
from .utils import create_vllm_config


def _make_empty_scheduler_output():
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        kv_connector_metadata=ExampleConnectorMetadata(),
    )


@pytest.fixture
def dsa_connector():
    vllm_config = create_vllm_config()
    vllm_config.kv_transfer_config.kv_connector = "TestExampleConnector"
    vllm_config.kv_transfer_config.kv_role = "kv_both"
    vllm_config.kv_transfer_config.kv_connector_extra_config["name"] = "dsa"
    ensure_kv_transfer_initialized(vllm_config)

    try:
        connector = get_kv_transfer_group()
        yield vllm_config, connector._connector
    finally:
        KVConnectorModelRunnerMixin.ensure_kv_transfer_shutdown()


def test_kv_connector_mixin_clears_metadata():
    vllm_config = create_vllm_config()
    vllm_config.kv_transfer_config.kv_connector = "TestExampleConnector"
    vllm_config.kv_transfer_config.kv_role = "kv_both"
    vllm_config.kv_transfer_config.kv_connector_extra_config["name"] = "unit"

    # Initialize the global connector instance
    ensure_kv_transfer_initialized(vllm_config)

    try:
        # Minimal scheduler output with empty metadata; mixin should still
        # bind/clear metadata even if no loads happen
        scheduler_output = _make_empty_scheduler_output()

        # Invoke the no-forward path which uses the mixin context manager
        KVConnectorModelRunnerMixin.kv_connector_no_forward(
            scheduler_output, vllm_config
        )

        # Verify clear_connector_metadata was called on the connector
        connector = get_kv_transfer_group()
        assert connector._connector_metadata is None
        # Test connector wrapper records method calls
        assert connector.call_record.get("bind_connector_metadata", 0) == 1
        assert connector.call_record.get("clear_connector_metadata", 0) == 1
    finally:
        # Ensure we clean up the global connector between tests
        KVConnectorModelRunnerMixin.ensure_kv_transfer_shutdown()


@pytest.mark.parametrize(
    "failure_method",
    ("start_load_kv", "model_forward", "wait_for_save", "get_finished"),
)
def test_kv_connector_mixin_clears_metadata_on_failure(failure_method):
    vllm_config = create_vllm_config()
    vllm_config.kv_transfer_config.kv_connector = "TestExampleConnector"
    vllm_config.kv_transfer_config.kv_role = "kv_both"
    vllm_config.kv_transfer_config.kv_connector_extra_config["name"] = "unit"
    ensure_kv_transfer_initialized(vllm_config)

    try:
        connector = get_kv_transfer_group()
        failure = (
            nullcontext()
            if failure_method == "model_forward"
            else patch.object(
                connector,
                failure_method,
                side_effect=RuntimeError("injected connector failure"),
            )
        )
        with (
            set_forward_context(None, vllm_config),
            failure,
            pytest.raises(RuntimeError, match="injected connector failure"),
            KVConnectorModelRunnerMixin._get_kv_connector_output(
                _make_empty_scheduler_output()
            ),
        ):
            if failure_method == "model_forward":
                raise RuntimeError("injected connector failure")

        assert connector._connector_metadata is None
        assert connector.call_record.get("clear_connector_metadata", 0) == 1
    finally:
        KVConnectorModelRunnerMixin.ensure_kv_transfer_shutdown()


def test_kv_connector_finalize_clears_metadata_on_failure():
    vllm_config = create_vllm_config()
    vllm_config.kv_transfer_config.kv_connector = "TestExampleConnector"
    vllm_config.kv_transfer_config.kv_role = "kv_both"
    vllm_config.kv_transfer_config.kv_connector_extra_config["name"] = "unit"
    ensure_kv_transfer_initialized(vllm_config)

    try:
        connector = get_kv_transfer_group()
        connector.bind_connector_metadata(ExampleConnectorMetadata())
        with (
            patch.object(
                connector,
                "wait_for_save",
                side_effect=RuntimeError("injected finalize failure"),
            ),
            pytest.raises(RuntimeError, match="injected finalize failure"),
        ):
            KVConnectorModelRunnerMixin.finalize_kv_connector()

        assert connector._connector_metadata is None
        assert connector.call_record.get("clear_connector_metadata", 0) == 1
    finally:
        KVConnectorModelRunnerMixin.ensure_kv_transfer_shutdown()


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
def test_kv_connector_mixin_drains_dsa_output_after_save(dsa_connector):
    vllm_config, connector = dsa_connector
    calls = []

    def wait_for_save():
        calls.append("wait_for_save")

    def get_receipts():
        calls.append("get_receipts")
        return [sentinel.receipt]

    def get_events():
        calls.append("get_events")
        return iter((sentinel.event,))

    with (
        patch.object(connector, "wait_for_save", side_effect=wait_for_save),
        patch.object(
            connector,
            "get_dsa_operation_receipts",
            side_effect=get_receipts,
        ),
        patch.object(
            connector,
            "get_dsa_control_events",
            side_effect=get_events,
        ),
        set_forward_context(None, vllm_config),
        KVConnectorModelRunnerMixin._get_kv_connector_output(
            _make_empty_scheduler_output()
        ) as output,
    ):
        pass

    assert calls == ["wait_for_save", "get_receipts", "get_events"]
    assert output.dsa_receipts == (sentinel.receipt,)
    assert output.dsa_events == (sentinel.event,)


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
def test_kv_connector_no_forward_preserves_dsa_only_output(dsa_connector):
    vllm_config, connector = dsa_connector

    with patch.object(
        connector,
        "get_dsa_operation_receipts",
        return_value=[sentinel.receipt],
    ):
        output = KVConnectorModelRunnerMixin.kv_connector_no_forward(
            _make_empty_scheduler_output(), vllm_config
        )

    assert output is not EMPTY_MODEL_RUNNER_OUTPUT
    assert output.kv_connector_output is not None
    assert output.kv_connector_output.dsa_receipts == (sentinel.receipt,)


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
def test_kv_connector_deferred_finalize_preserves_dsa_output(dsa_connector):
    vllm_config, connector = dsa_connector
    calls = []

    def wait_for_save():
        calls.append("wait_for_save")

    def get_receipts():
        calls.append("get_receipts")
        return [sentinel.receipt]

    def get_events():
        calls.append("get_events")
        return [sentinel.event]

    with (
        patch.object(connector, "wait_for_save", side_effect=wait_for_save),
        patch.object(
            connector,
            "get_dsa_operation_receipts",
            side_effect=get_receipts,
        ),
        patch.object(
            connector,
            "get_dsa_control_events",
            side_effect=get_events,
        ),
        patch.object(
            connector,
            "get_completed_decode_window_saves",
            side_effect=(
                {"early": 128, "same": 512},
                {"late": 256, "same": 256},
            ),
            create=True,
        ),
        set_forward_context(None, vllm_config),
    ):
        with KVConnectorModelRunnerMixin._get_kv_connector_output(
            _make_empty_scheduler_output(), defer_finalize=True
        ) as output:
            pass
        finalized_output = KVConnectorModelRunnerMixin.finalize_kv_connector(output)

    assert finalized_output is output
    assert calls == [
        "get_receipts",
        "get_events",
        "wait_for_save",
        "get_receipts",
        "get_events",
    ]
    assert finalized_output.dsa_receipts == (
        sentinel.receipt,
        sentinel.receipt,
    )
    assert finalized_output.dsa_events == (sentinel.event, sentinel.event)
    assert finalized_output.completed_decode_window_saves == {
        "early": 128,
        "same": 512,
        "late": 256,
    }
