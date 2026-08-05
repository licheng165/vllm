# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import cast

import pytest

from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.v1.core.sched.dsa_types import (
    DSAControlEvent,
    DSAExecutionReceipt,
    DSAOperationReceipt,
)
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput

pytestmark = pytest.mark.cpu_test


class DummyModelRunnerOutput(ModelRunnerOutput):
    def __init__(
        self,
        finished_sending: set[str] | None = None,
        finished_recving: set[str] | None = None,
        invalid_block_ids: set[int] | None = None,
        expected_finished_count: int = 0,
        dsa_receipts: tuple[DSAOperationReceipt, ...] = (),
        dsa_events: tuple[DSAControlEvent, ...] = (),
    ):
        self.kv_connector_output = KVConnectorOutput(
            finished_sending=finished_sending,
            finished_recving=finished_recving,
            invalid_block_ids=invalid_block_ids or set(),
            expected_finished_count=expected_finished_count,
            dsa_receipts=dsa_receipts,
            dsa_events=dsa_events,
        )
        self.dsa_execution_receipts = {}

    def __repr__(self):
        return (
            f"DummyModelRunnerOutput("
            f"finished_sending={self.kv_connector_output.finished_sending},"
            f"finished_recving={self.kv_connector_output.finished_recving})"
            f"invalid_block_ids={self.kv_connector_output.invalid_block_ids})"
        )


def test_aggregate_workers_output():
    aggregator = KVOutputAggregator(expected_finished_count=2)

    output1 = DummyModelRunnerOutput()
    output2 = DummyModelRunnerOutput()

    aggregated = aggregator.aggregate([output1, output2])

    assert aggregated is output1
    aggregated = aggregated.kv_connector_output
    assert aggregated.finished_sending is None
    assert aggregated.finished_recving is None
    assert not aggregated.invalid_block_ids

    output1 = DummyModelRunnerOutput(
        finished_sending={"req1"}, finished_recving={"req2"}
    )
    output2 = DummyModelRunnerOutput(invalid_block_ids={1})

    aggregated = aggregator.aggregate([output1, output2])

    assert aggregated is output1
    aggregated = aggregated.kv_connector_output
    assert aggregated.finished_sending is None
    assert aggregated.finished_recving is None
    assert aggregated.invalid_block_ids == {1}

    output1 = DummyModelRunnerOutput(invalid_block_ids={2})
    output2 = DummyModelRunnerOutput(finished_sending={"req1"})

    aggregated = aggregator.aggregate([output1, output2])

    assert aggregated is output1
    aggregated = aggregated.kv_connector_output
    assert aggregated.finished_sending == {"req1"}
    assert aggregated.finished_recving is None
    assert aggregated.invalid_block_ids == {2}

    output1 = DummyModelRunnerOutput(invalid_block_ids={3, 4})
    output2 = DummyModelRunnerOutput(
        finished_recving={"req2"}, invalid_block_ids={4, 5}
    )

    aggregated = aggregator.aggregate([output1, output2])

    assert aggregated is output1
    aggregated = aggregated.kv_connector_output
    assert aggregated.finished_sending is None
    assert aggregated.finished_recving == {"req2"}
    assert aggregated.invalid_block_ids == {3, 4, 5}


def test_aggregate_workers_output_with_expected_finished_count():
    # We create the aggregator expecting to collect from 4 workers
    aggregator = KVOutputAggregator(expected_finished_count=4)
    assert aggregator._expected_finished_count == 4
    # Some request with default expected finished requests
    output1 = DummyModelRunnerOutput(finished_sending={"req1"})
    aggregated = aggregator.aggregate([output1])
    # still expecting to collect from 4 workers
    assert aggregator._send_remaining_count["req1"] == 3
    assert not aggregated.kv_connector_output.finished_sending
    assert not aggregated.kv_connector_output.finished_recving

    # Workers discover and find that in this setup they only need to
    # collect from 2
    output1 = DummyModelRunnerOutput(
        finished_sending={"req1"}, expected_finished_count=2
    )
    output2 = DummyModelRunnerOutput(
        finished_recving={"req2"}, expected_finished_count=2
    )
    output3 = DummyModelRunnerOutput(finished_recving={"req2"})
    # Req2 only needs 2 acks
    aggregated = aggregator.aggregate([output1, output2, output3])
    assert aggregated.kv_connector_output.expected_finished_count == 2

    assert not aggregated.kv_connector_output.finished_sending

    # Req2 is finished
    assert "req2" not in aggregator._recv_remaining_count
    assert aggregated.kv_connector_output.finished_recving == {"req2"}

    # Req1 is still waiting for 2 more acks (expected_finished_count has no effect)
    # NOTE: This is to showcase dynamic update. Workers are responsible for
    # ensuring "req1" termination in this case
    assert aggregator._send_remaining_count["req1"] == 2


def test_aggregate_workers_preserves_dsa_order_and_duplicates() -> None:
    receipt_1 = cast(DSAOperationReceipt, object())
    receipt_2 = cast(DSAOperationReceipt, object())
    event_1 = cast(DSAControlEvent, object())
    event_2 = cast(DSAControlEvent, object())
    execution_receipt_1 = cast(DSAExecutionReceipt, object())
    output_1 = DummyModelRunnerOutput(
        dsa_receipts=(receipt_1, receipt_1),
        dsa_events=(event_1,),
    )
    output_2 = DummyModelRunnerOutput(
        dsa_receipts=(receipt_2, receipt_1),
        dsa_events=(event_2, event_1),
    )
    output_1.dsa_execution_receipts = {"request": execution_receipt_1}
    output_2.dsa_execution_receipts = {"request": execution_receipt_1}

    aggregated = KVOutputAggregator(expected_finished_count=2).aggregate(
        [output_1, output_2]
    )

    assert aggregated is output_1
    assert aggregated.dsa_execution_receipts == {"request": execution_receipt_1}
    assert aggregated.kv_connector_output is not None
    assert aggregated.kv_connector_output.dsa_receipts == (
        receipt_1,
        receipt_1,
        receipt_2,
        receipt_1,
    )
    assert aggregated.kv_connector_output.dsa_events == (
        event_1,
        event_2,
        event_1,
    )


def test_aggregate_workers_rejects_divergent_dsa_execution_receipts() -> None:
    receipt_1 = cast(DSAExecutionReceipt, object())
    receipt_2 = cast(DSAExecutionReceipt, object())
    output_1 = DummyModelRunnerOutput()
    output_2 = DummyModelRunnerOutput()
    output_1.dsa_execution_receipts = {"request": receipt_1}
    output_2.dsa_execution_receipts = {"request": receipt_2}

    with pytest.raises(RuntimeError, match="differ across worker ranks"):
        KVOutputAggregator(expected_finished_count=2).aggregate(
            [output_1, output_2]
        )
