# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from typing import Any, cast
from unittest import TestCase

import pytest

from vllm.v1.core.sched.dsa_types import (
    DSAControlEvent,
    DSAOperationReceipt,
    RequestKey,
)
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput, LogprobsLists, ModelRunnerOutput

pytestmark = pytest.mark.cpu_test


class TestLogprobsLists(TestCase):
    def setUp(self):
        self.logprobsLists = LogprobsLists(
            logprob_token_ids=[
                [1, 2],  # Request 0 token 0
                [3, 4],  # Request 0 token 1
                [5, 6],  # Request 1 token 0
                [7, 8],  # Request 1 token 1
                [9, 10],  # Request 1 token 2
                [11, 12],  # Request 2 token 0
                [13, 14],  # Request 2 token 1
                [15, 16],  # Request 2 token 2
                [17, 18],  # Request 2 token 3
            ],
            logprobs=[
                [0.1, 0.2],
                [0.3, 0.4],
                [0.5, 0.6],
                [0.7, 0.8],
                [0.9, 1.0],
                [1.1, 1.2],
                [1.3, 1.4],
                [1.5, 1.6],
                [1.7, 1.8],
            ],
            sampled_token_ranks=[1, 3, 5, 7, 9, 11, 13, 15, 17],
            cu_num_generated_tokens=[0, 2, 5, 9],
        )

    def test_slice_without_cu_num_generated_tokens(self):
        """Test slicing without cu_num_generated_tokens"""
        logprobsLists = LogprobsLists(
            logprob_token_ids=[[1], [2], [3]],
            logprobs=[[0.1], [0.2], [0.3]],
            sampled_token_ranks=[1, 2, 3],
            cu_num_generated_tokens=None,
        )

        sliced = logprobsLists.slice_request(1, num_positions=2)
        assert sliced.logprob_token_ids == [[2], [3]]
        assert sliced.logprobs == [[0.2], [0.3]]
        assert sliced.sampled_token_ranks == [2, 3]
        assert sliced.cu_num_generated_tokens is None

    def test_slice_from_start(self):
        """Test slicing from the start position"""
        sliced = self.logprobsLists.slice_request(0, num_positions=5)
        assert len(sliced.logprob_token_ids) == 5
        assert sliced.logprob_token_ids == [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [9, 10],
        ]
        assert sliced.cu_num_generated_tokens is None

    def test_slice_from_middle(self):
        """Test slicing from the middle position"""
        sliced = self.logprobsLists.slice_request(1, num_positions=7)
        assert len(sliced.logprob_token_ids) == 7
        assert sliced.logprob_token_ids == [
            [5, 6],
            [7, 8],
            [9, 10],
            [11, 12],
            [13, 14],
            [15, 16],
            [17, 18],
        ]
        assert sliced.cu_num_generated_tokens is None

    def test_slice_single_request(self):
        """Test slicing a single request"""
        sliced = self.logprobsLists.slice_request(1, num_positions=3)
        assert len(sliced.logprob_token_ids) == 3
        assert sliced.logprob_token_ids == [[5, 6], [7, 8], [9, 10]]
        assert sliced.cu_num_generated_tokens is None

    def test_slice_last_request(self):
        """Test slicing the last request"""
        sliced = self.logprobsLists.slice_request(2, num_positions=4)
        assert len(sliced.logprob_token_ids) == 4
        assert sliced.logprob_token_ids == [[11, 12], [13, 14], [15, 16], [17, 18]]
        assert sliced.cu_num_generated_tokens is None

    def test_slice_all_requests(self):
        """Test slicing all requests (full slice)"""
        sliced = self.logprobsLists.slice_request(0, num_positions=9)
        assert len(sliced.logprob_token_ids) == 9  # All tokens
        assert sliced.logprob_token_ids == self.logprobsLists.logprob_token_ids
        assert sliced.cu_num_generated_tokens is None


def test_kv_connector_output_dsa_only_is_not_empty() -> None:
    receipt = cast(DSAOperationReceipt, object())
    event = cast(DSAControlEvent, object())

    assert KVConnectorOutput().is_empty()
    assert not KVConnectorOutput(dsa_receipts=(receipt,)).is_empty()
    assert not KVConnectorOutput(dsa_events=(event,)).is_empty()


def test_kv_connector_output_merge_preserves_dsa_order() -> None:
    receipt_1 = cast(DSAOperationReceipt, object())
    receipt_2 = cast(DSAOperationReceipt, object())
    event_1 = cast(DSAControlEvent, object())
    event_2 = cast(DSAControlEvent, object())

    merged = KVConnectorOutput.merge(
        KVConnectorOutput(
            dsa_receipts=(receipt_1, receipt_2),
            dsa_events=(event_1,),
        ),
        KVConnectorOutput(
            dsa_receipts=(receipt_1,),
            dsa_events=(event_2, event_1),
        ),
    )

    assert merged.dsa_receipts == (receipt_1, receipt_2, receipt_1)
    assert merged.dsa_events == (event_1, event_2, event_1)


def test_model_runner_output_dsa_execution_receipts_default() -> None:
    output_1 = ModelRunnerOutput(req_ids=[], req_id_to_index={})
    output_2 = ModelRunnerOutput(req_ids=[], req_id_to_index={})

    assert output_1.dsa_execution_receipts == {}
    assert output_1.dsa_execution_receipts is not output_2.dsa_execution_receipts


def test_scheduler_output_dsa_defaults() -> None:
    output_1 = SchedulerOutput.make_empty()
    output_2 = SchedulerOutput.make_empty()

    assert output_1.dsa_routes == {}
    assert output_1.dsa_routes is not output_2.dsa_routes
    assert output_1.dsa_commands == ()
    assert output_1.dsa_connector_only_imports == ()
    assert output_1.dsa_preemption_actions == ()


def test_request_data_preserves_request_keys() -> None:
    request_key = RequestKey("process", "request", 1)
    request = SimpleNamespace(
        request_id="request",
        prompt_token_ids=[1],
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        num_computed_tokens=0,
        lora_request=None,
        prompt_embeds=None,
        dsa_state=SimpleNamespace(request_key=request_key),
    )

    new_request = NewRequestData.from_request(cast(Any, request), ([],))
    cached_request = CachedRequestData(
        req_ids=["request"],
        resumed_req_ids=set(),
        new_token_ids=[],
        all_token_ids={},
        new_block_ids=[],
        num_computed_tokens=[0],
        num_output_tokens=[0],
        dsa_request_keys={"request": request_key},
    )

    assert new_request.dsa_request_key is request_key
    assert cached_request.dsa_request_keys == {"request": request_key}
