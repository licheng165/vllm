# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.v1.core.kv_cache_manager import (
    DSALatentReleaseTransaction,
    KVCacheManager,
)
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.sched.dsa_types import DSAOperationRef, RequestKey
from vllm.v1.core.single_type_kv_cache_manager import DSALatentManager

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _release_fixture():
    request_key = RequestKey("scheduler", "request", 1)
    activation = DSAOperationRef(
        operation_id="activation",
        parent_operation_id="store",
        kind="source_activation",
        obligations=frozenset({"source_activation"}),
        range_start=0,
        range_end=64,
        input_generation_id="generation-1",
        output_generation_id="generation-1",
        route_epoch=1,
    )
    state = SimpleNamespace(
        request_key=request_key,
        source_activation_inflight=activation,
        latest_sealed_generation_id="generation-1",
        latest_sealed_sparse_source_end=64,
        completed_canonical_end=64,
        initial_prefill_complete=True,
        release_end=0,
    )
    request = SimpleNamespace(
        request_id="request",
        dsa_state=state,
    )

    latent_manager = DSALatentManager.__new__(DSALatentManager)
    latent_manager.block_size = 16
    latent_manager.scratch_blocks = 2
    latent_manager._null_block = KVCacheBlock(-1)
    latent_manager.req_to_blocks = {
        "request": [KVCacheBlock(index, ref_cnt=1) for index in range(4)]
    }
    remove_saved_blocks = Mock(return_value=2)
    manager = SimpleNamespace(
        coordinator=SimpleNamespace(single_type_managers=(latent_manager,)),
        remove_saved_decode_window_blocks=remove_saved_blocks,
    )
    transaction = DSALatentReleaseTransaction(
        request_id="request",
        request_key=request_key,
        proposed_end=64,
        source_generation_id="generation-1",
        block_size=16,
        current_request=request,
    )
    return transaction, manager, remove_saved_blocks


def test_release_transaction_returns_frontier_not_freed_block_count() -> None:
    transaction, manager, remove_saved_blocks = _release_fixture()
    transaction.validate(manager)

    committed_end = transaction.commit_no_fail(manager)

    assert committed_end == 64
    assert transaction.committed_release == 64
    assert transaction.freed_blocks == 2
    remove_saved_blocks.assert_called_once_with("request", 64)


def test_prepare_release_uses_the_latent_manager_block_size() -> None:
    transaction, manager, _ = _release_fixture()

    prepared = KVCacheManager.prepare_dsa_release(
        manager,
        request_key=transaction.request_key,
        proposed_end=transaction.proposed_end,
        source_generation_id=transaction.source_generation_id,
        request=transaction.current_request,
    )

    assert prepared.block_size == 16
    assert prepared._validated is True


def test_release_transaction_rejects_uncommitted_hole_before_mutation() -> None:
    transaction, manager, remove_saved_blocks = _release_fixture()
    latent_manager = manager.coordinator.single_type_managers[0]
    latent_manager.req_to_blocks["request"][3] = latent_manager._null_block

    with pytest.raises(ValueError, match="uncommitted hole"):
        transaction.validate(manager)

    remove_saved_blocks.assert_not_called()


def test_prepare_release_rejects_legacy_request_id_bypass() -> None:
    transaction, manager, remove_saved_blocks = _release_fixture()

    with pytest.raises(ValueError, match="requires a RequestKey"):
        KVCacheManager.prepare_dsa_release(
            manager,
            request_key="request",  # type: ignore[arg-type]
            proposed_end=transaction.proposed_end,
            source_generation_id=transaction.source_generation_id,
            request=transaction.current_request,
        )

    remove_saved_blocks.assert_not_called()


def test_release_transaction_rejects_aliased_block_before_mutation() -> None:
    transaction, manager, remove_saved_blocks = _release_fixture()
    latent_manager = manager.coordinator.single_type_managers[0]
    aliased = latent_manager.req_to_blocks["request"][2]
    latent_manager.req_to_blocks["request"][3] = aliased

    with pytest.raises(ValueError, match="aliased block"):
        transaction.validate(manager)

    assert all(block.block_id >= 0 for block in latent_manager.req_to_blocks["request"])
    remove_saved_blocks.assert_not_called()
