# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.core.block_pool import DSASharedLogicalBlockPool
from vllm.v1.core.dsa_shared_pool import (
    MAX_ALLOCATION_GENERATION,
    DSABlockAllocationMode,
    DSASharedBlockLayout,
    DSASharedBlockOwner,
    DSASharedBundleAllocator,
    PrefillLayerBundlePool,
)
from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinatorNoPrefixCache
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import (
    build_dsa_kv_topology,
    generate_scheduler_kv_cache_config,
    get_dsa_role_groups,
    get_kv_cache_config_from_groups,
    get_kv_cache_configs,
    get_layerwise_prefill_max_tokens,
    layerwise_prefill_startup_summary,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    DSAKVRegistration,
    DSAKVRow,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    layerwise_prefill_p_node_enabled,
    validate_layerwise_prefill_p_node,
)
from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
from vllm.v1.worker.gpu_model_runner import (
    GPUModelRunner,
    _validate_block_allocation_metadata,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

_BUNDLE_PAGE_BYTES = 294_912
_LATENT_PAGE_BYTES = 147_456
_INDEXER_PAGE_BYTES = 32_768
_GLM52_INDEXER_EXECUTIONS = (
    0,
    1,
    2,
    6,
    10,
    14,
    18,
    22,
    26,
    30,
    34,
    38,
    42,
    46,
    50,
    54,
    58,
    62,
    66,
    70,
    74,
    78,
)


class _CapableGPUModelRunner(GPUModelRunner):
    supports_layerwise_prefill_p_node = True


class _CapableGPUModelRunnerV2(GPUModelRunnerV2):
    supports_layerwise_prefill_p_node = True


def _enable_layerwise_prefill(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE", "true")
    monkeypatch.setenv("VLLM_ASCEND_DSA_UNBUNDLE", "1")
    monkeypatch.setenv("VLLM_ASCEND_DSA_TWO_GROUPS", "1")
    monkeypatch.setenv("VLLM_ASCEND_DSA_SHARED_POOL", "1")


def _glm52_groups_and_topology():
    specs = {
        f"latent.{execution}": MLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
            dsa_kv_registration=DSAKVRegistration(execution, 0),
        )
        for execution in range(79)
    }
    specs.update(
        {
            f"indexer.{execution}": MLAAttentionSpec(
                block_size=128,
                num_kv_heads=1,
                head_size=128,
                dtype=torch.bfloat16,
                dsa_kv_registration=DSAKVRegistration(execution, 1),
            )
            for execution in _GLM52_INDEXER_EXECUTIONS
        }
    )
    topology = build_dsa_kv_topology(specs)
    groups = [
        KVCacheGroupSpec(
            [row.layer_name for row in topology.rows_by_group[group]],
            specs[topology.rows_by_group[group][0].layer_name],
        )
        for group in range(2)
    ]
    return groups, topology, specs


def _vllm_config(parent_capacity: int, max_model_len: int = 1_000_000):
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=parent_capacity,
            gpu_memory_utilization=0.9,
            kv_cache_memory_bytes=None,
            enable_prefix_caching=False,
        ),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(index_topk=2048),
            max_model_len=max_model_len,
            original_max_model_len=max_model_len,
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=32,
            disable_hybrid_kv_cache_manager=False,
        ),
        num_speculative_tokens=0,
    )


def _make_child_pool(parent_capacity: int = 2, physical_slots: int = 3):
    parent = DSASharedBundleAllocator(
        DSASharedBlockLayout(
            latent_page_size_bytes=147_456,
            indexer_page_size_bytes=32_768,
            capacity_bundles=parent_capacity,
        )
    )
    return parent, PrefillLayerBundlePool(parent, physical_slots)


def _make_glm52_coordinator(monkeypatch: pytest.MonkeyPatch, max_model_len: int = 128):
    _enable_layerwise_prefill(monkeypatch)
    groups, topology, _ = _glm52_groups_and_topology()
    config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[
            KVCacheTensor(
                size=(79 + 1) * _BUNDLE_PAGE_BYTES,
                shared_by=[
                    row.layer_name for rows in topology.rows_by_group for row in rows
                ],
            )
        ],
        kv_cache_groups=groups,
        dsa_kv_topology=topology,
    )
    return KVCacheCoordinatorNoPrefixCache(
        config,
        max_model_len=max_model_len,
        use_eagle=False,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        hash_block_size=128,
    )


def test_feature_gate_is_strict_and_validation_is_fail_closed(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE", "yes")
    with pytest.raises(ValueError, match="must be 'true' or 'false'"):
        layerwise_prefill_p_node_enabled()

    monkeypatch.setenv("VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE", "true")
    monkeypatch.setenv("VLLM_ASCEND_DSA_SHRINK_LATENT", "2")
    config = _vllm_config(1)
    config.cache_config.enable_prefix_caching = True
    config.parallel_config.pipeline_parallel_size = 2
    with pytest.raises(ValueError) as exc_info:
        validate_layerwise_prefill_p_node(config, None)
    message = str(exc_info.value)
    assert "VLLM_ASCEND_DSA_UNBUNDLE=1" in message
    assert "VLLM_ASCEND_DSA_SHARED_POOL=1" in message
    assert "VLLM_ASCEND_DSA_SHRINK_LATENT=0" in message
    assert "pipeline_parallel_size=1" in message
    assert "enable_prefix_caching=false" in message
    assert "canonical DSA KV topology" in message


def test_topology_role_resolution_does_not_infer_from_page_size():
    groups, topology, _ = _glm52_groups_and_topology()
    inverted_geometry_groups = [
        KVCacheGroupSpec(groups[0].layer_names, groups[1].kv_cache_spec),
        KVCacheGroupSpec(groups[1].layer_names, groups[0].kv_cache_spec),
    ]

    latent, indexer = get_dsa_role_groups(inverted_geometry_groups, topology)
    assert latent is inverted_geometry_groups[0]
    assert indexer is inverted_geometry_groups[1]


def test_empty_scheduler_output_preserves_legacy_optional_metadata():
    cached = SchedulerOutput.make_empty().scheduled_cached_reqs
    assert cached.new_block_ids_by_bank is None
    assert cached.new_block_allocation_modes is None
    assert cached.allocation_generations is None


def test_child_mapping_is_dense_and_bank_allocation_rolls_back():
    parent, pool = _make_child_pool(parent_capacity=2, physical_slots=3)
    assert [
        pool.child_bundle_id(slot, parent_id)
        for slot in range(3)
        for parent_id in range(1, 3)
    ] == list(range(1, 7))
    for child_id in range(1, 7):
        assert (
            pool.child_bundle_id(
                pool.physical_slot(child_id), pool.parent_bundle_id(child_id)
            )
            == child_id
        )

    parent_small, pool_small = _make_child_pool(parent_capacity=1, physical_slots=2)
    pool_small.begin_request_allocation("req", 1)
    with pytest.raises(ValueError, match="Cannot get 2 DSA child bundles"):
        pool_small.allocate_banks(DSASharedBlockOwner.LATENT, 2)
    pool_small.end_request_allocation("req", 1)

    assert pool_small.free_bundle_count == 2
    assert pool_small.reserved_parent_count == 0
    assert parent_small.free_bundle_count == 1
    assert pool_small.get_request_arena("req", 1) is None
    assert parent.free_bundle_count == 2


def test_stale_generation_cannot_free_reused_children():
    parent, allocator = _make_child_pool(parent_capacity=1, physical_slots=4)
    pool = DSASharedLogicalBlockPool(allocator, DSASharedBlockOwner.LATENT)

    allocator.begin_request_allocation("same-id", 1)
    stale_blocks = pool.get_new_blocks(1)
    allocator.end_request_allocation("same-id", 1)
    allocation = KVCacheBlocks((stale_blocks,))
    assert allocation.get_allocation_mode() == DSABlockAllocationMode.PREFILL_CHILD
    assert allocation.get_allocation_generation() == 1
    bank_ids = allocation.get_block_ids_by_bank()
    assert bank_ids is not None
    assert bank_ids[0][0] == [block.block_id for block in stale_blocks]
    stale_bank_ids = tuple(block.bank_block_ids for block in stale_blocks)
    pool.free_blocks(stale_blocks)

    allocator.begin_request_allocation("same-id", 2)
    current_blocks = pool.get_new_blocks(1)
    allocator.end_request_allocation("same-id", 2)
    assert tuple(block.bank_block_ids for block in current_blocks) == stale_bank_ids

    with pytest.raises(ValueError, match="stale allocation generation"):
        pool.free_blocks(stale_blocks)
    assert allocator.free_bundle_count == 2

    pool.free_blocks(current_blocks)
    with pytest.raises(ValueError, match="already free"):
        pool.free_blocks(current_blocks)
    assert parent.free_bundle_count == 1


def test_source_lease_delays_parent_reclamation():
    parent, allocator = _make_child_pool(parent_capacity=1, physical_slots=4)
    pool = DSASharedLogicalBlockPool(allocator, DSASharedBlockOwner.LATENT)
    allocator.begin_request_allocation("req", 7)
    blocks = pool.get_new_blocks(1)
    allocator.end_request_allocation("req", 7)

    pool.acquire_source_lease(blocks, "req", 7)
    pool.free_blocks(blocks)
    assert allocator.reserved_parent_count == 1
    assert parent.free_bundle_count == 0

    pool.release_source_lease(blocks, "req", 7)
    assert allocator.reserved_parent_count == 0
    assert parent.free_bundle_count == 1
    with pytest.raises(ValueError, match="already free"):
        pool.release_source_lease(blocks, "req", 7)


def test_source_lease_rejects_incomplete_block_identity():
    _, allocator = _make_child_pool(parent_capacity=1, physical_slots=4)
    pool = DSASharedLogicalBlockPool(allocator, DSASharedBlockOwner.LATENT)
    allocator.begin_request_allocation("req", 7)
    blocks = pool.get_new_blocks(1)
    allocator.end_request_allocation("req", 7)

    bank_block_ids = blocks[0].bank_block_ids
    blocks[0].bank_block_ids = None
    with pytest.raises(ValueError, match="missing PREFILL_CHILD identity"):
        pool.acquire_source_lease(blocks, "req", 7)
    blocks[0].bank_block_ids = bank_block_ids
    pool.free_blocks(blocks)


def test_wrong_owner_free_is_atomic():
    parent, allocator = _make_child_pool(parent_capacity=1, physical_slots=4)
    allocator.begin_request_allocation("req", 4)
    banks = allocator.allocate_banks(DSASharedBlockOwner.LATENT, 1)
    allocator.end_request_allocation("req", 4)
    free_before = allocator.free_bundle_count

    with pytest.raises(ValueError, match="not DSASharedBlockOwner.INDEXER"):
        allocator.free(
            DSASharedBlockOwner.INDEXER,
            banks[0],
            "req",
            4,
        )
    assert allocator.free_bundle_count == free_before

    for bank in banks:
        allocator.free(DSASharedBlockOwner.LATENT, bank, "req", 4)
    assert parent.free_bundle_count == 1


def test_wrong_bank_free_is_atomic():
    _, allocator = _make_child_pool(parent_capacity=1, physical_slots=4)
    pool = DSASharedLogicalBlockPool(allocator, DSASharedBlockOwner.LATENT)
    allocator.begin_request_allocation("req", 4)
    blocks = pool.get_new_blocks(1)
    allocator.end_request_allocation("req", 4)
    original_bank_ids = blocks[0].bank_block_ids
    assert original_bank_ids is not None
    other_bank_zero_ids = blocks[1].bank_block_ids
    assert other_bank_zero_ids is not None
    blocks[0].bank_block_ids = (original_bank_ids[0], other_bank_zero_ids[0])

    with pytest.raises(ValueError, match="different bank"):
        pool.free_blocks([blocks[0]])
    assert blocks[0].ref_cnt == 1

    blocks[0].bank_block_ids = original_bank_ids
    pool.free_blocks(blocks)


def test_global_slab_capacity_boundaries_and_reconciliation(monkeypatch):
    _enable_layerwise_prefill(monkeypatch)
    groups, topology, specs = _glm52_groups_and_topology()
    assert (
        tuple(row.execution_ordinal for row in topology.rows_by_group[1])
        == _GLM52_INDEXER_EXECUTIONS
    )

    with pytest.raises(ValueError, match="C=120"):
        get_kv_cache_config_from_groups(
            _vllm_config(120),
            groups,
            available_memory=4 * 2**30,
            dsa_kv_topology=topology,
        )

    config_121 = get_kv_cache_config_from_groups(
        _vllm_config(121),
        groups,
        available_memory=4 * 2**30,
        dsa_kv_topology=topology,
    )
    config_138 = get_kv_cache_config_from_groups(
        _vllm_config(138),
        groups,
        available_memory=4 * 2**30,
        dsa_kv_topology=topology,
    )

    assert len(config_138.kv_cache_tensors) == 1
    tensor = config_138.kv_cache_tensors[0]
    assert len(tensor.shared_by) == 101
    assert tensor.size == (79 * 138 + 1) * _BUNDLE_PAGE_BYTES
    assert tensor.size / 2**30 == pytest.approx(2.995, abs=0.001)
    assert get_layerwise_prefill_max_tokens(config_138) == 1_141_632

    summary = layerwise_prefill_startup_summary(config_138)
    assert summary["residency_mode"] == "PREFILL_LAYERWISE"
    assert summary["topology_signature"]
    assert summary["latent_layers"] == 79
    assert summary["indexer_layers"] == 22
    assert summary["producer_execution_count"] == 22
    assert summary["latent_page_bytes"] == _LATENT_PAGE_BYTES
    assert summary["indexer_page_bytes"] == _INDEXER_PAGE_BYTES
    assert summary["bundle_page_bytes"] == _BUNDLE_PAGE_BYTES
    assert summary["parent_capacity"] == 138
    assert summary["child_capacity"] == 79 * 138
    assert summary["slab_bytes"] == (79 * 138 + 1) * _BUNDLE_PAGE_BYTES
    assert summary["max_tokens"] == 1_141_632

    vllm_config = _vllm_config(1)
    vllm_config.cache_config.num_gpu_blocks_override = None
    reconciled = get_kv_cache_configs(
        vllm_config,
        [specs, dict(reversed(list(specs.items())))],
        [config_138.kv_cache_tensors[0].size, config_121.kv_cache_tensors[0].size],
    )
    scheduler_config = generate_scheduler_kv_cache_config(reconciled)
    assert scheduler_config.num_blocks == 121
    assert (
        scheduler_config.kv_cache_tensors[0].size == (79 * 121 + 1) * _BUNDLE_PAGE_BYTES
    )


@pytest.mark.parametrize("failing_leg", range(1, 5))
def test_four_leg_coordinator_allocation_rolls_back(monkeypatch, failing_leg):
    coordinator = _make_glm52_coordinator(monkeypatch)
    allocator = coordinator.dsa_shared_allocator
    allocate_leg = allocator._allocate_leg
    call_count = 0

    def fail_one_leg(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        result = allocate_leg(*args, **kwargs)
        if call_count == failing_leg:
            raise RuntimeError(f"injected allocation leg {failing_leg} failure")
        return result

    monkeypatch.setattr(allocator, "_allocate_leg", fail_one_leg)
    with pytest.raises(RuntimeError, match=f"allocation leg {failing_leg}"):
        coordinator.allocate_new_blocks(
            "req",
            num_tokens=128,
            num_tokens_main_model=128,
            allocation_generation=9,
        )

    assert allocator.free_bundle_count == 79
    assert allocator.reserved_parent_count == 0
    assert coordinator.dsa_shared_parent_allocator.free_bundle_count == 1
    assert all(
        not manager.req_to_blocks["req"] for manager in coordinator.single_type_managers
    )


def test_failed_chunk_extension_preserves_committed_children(monkeypatch):
    coordinator = _make_glm52_coordinator(monkeypatch, max_model_len=2048)
    coordinator.allocate_new_blocks(
        "req",
        num_tokens=128,
        num_tokens_main_model=128,
        allocation_generation=9,
    )
    allocator = coordinator.dsa_shared_allocator
    request_blocks_before = tuple(
        tuple(manager.req_to_blocks["req"])
        for manager in coordinator.single_type_managers
    )
    free_before = allocator.free_bundle_count
    allocate_leg = allocator._allocate_leg
    call_count = 0

    def fail_last_leg(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        result = allocate_leg(*args, **kwargs)
        if call_count == 4:
            raise RuntimeError("injected extension failure")
        return result

    monkeypatch.setattr(allocator, "_allocate_leg", fail_last_leg)
    with pytest.raises(RuntimeError, match="extension failure"):
        coordinator.allocate_new_blocks(
            "req",
            num_tokens=1280,
            num_tokens_main_model=1280,
            allocation_generation=9,
        )

    assert allocator.free_bundle_count == free_before
    assert allocator.reserved_parent_count == 1
    assert coordinator.dsa_shared_parent_allocator.free_bundle_count == 0
    assert all(
        tuple(manager.req_to_blocks["req"]) == blocks_before
        for manager, blocks_before in zip(
            coordinator.single_type_managers, request_blocks_before
        )
    )


def test_scheduler_generation_is_monotonic_across_request_id_reuse():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.layerwise_prefill_p_node = True
    scheduler._last_allocation_generation = 0
    scheduler._request_allocation_generations = {}

    first = scheduler._get_or_create_allocation_generation("same-id")
    assert first == 1
    assert scheduler._get_or_create_allocation_generation("same-id") == first
    scheduler._request_allocation_generations.pop("same-id")
    assert scheduler._get_or_create_allocation_generation("same-id") == 2

    scheduler._last_allocation_generation = MAX_ALLOCATION_GENERATION
    with pytest.raises(OverflowError, match="exhausted uint64"):
        scheduler._get_or_create_allocation_generation("new-id")


def test_scheduler_rejects_mismatched_block_generation():
    _, allocator = _make_child_pool(parent_capacity=1, physical_slots=4)
    pool = DSASharedLogicalBlockPool(allocator, DSASharedBlockOwner.LATENT)
    allocator.begin_request_allocation("req", 1)
    blocks = pool.get_new_blocks(1)
    allocator.end_request_allocation("req", 1)

    scheduler = Scheduler.__new__(Scheduler)
    scheduler._request_allocation_generations = {"req": 2}
    with pytest.raises(RuntimeError, match="disagrees with allocated blocks"):
        scheduler._allocation_generation_for_output("req", KVCacheBlocks((blocks,)))
    pool.free_blocks(blocks)


def test_worker_rejects_invalid_banked_metadata_shape():
    block_ids = ([2, 3], [18])
    with pytest.raises(RuntimeError, match="exactly two physical banks"):
        _validate_block_allocation_metadata(
            block_ids,
            (([2, 3], [18]),),
            DSABlockAllocationMode.PREFILL_CHILD,
            1,
        )

    with pytest.raises(RuntimeError, match="inconsistent group lengths"):
        _validate_block_allocation_metadata(
            block_ids,
            (([2, 3], [18]), ([4], [19])),
            DSABlockAllocationMode.PREFILL_CHILD,
            1,
        )


@pytest.mark.parametrize(
    ("runner_cls", "mutation_method"),
    [
        (GPUModelRunner, "_update_states"),
        (GPUModelRunnerV2, "finish_requests"),
    ],
)
def test_worker_variants_fail_before_state_update(
    monkeypatch, runner_cls, mutation_method
):
    runner = runner_cls.__new__(runner_cls)
    runner.layerwise_prefill_p_node = True
    monkeypatch.setattr(
        runner,
        mutation_method,
        lambda *_: pytest.fail("worker state must not be mutated"),
    )

    with pytest.raises(RuntimeError, match="did not opt in"):
        runner.execute_model(SchedulerOutput.make_empty())


@pytest.mark.parametrize(
    ("runner_cls", "mutation_method"),
    [
        (_CapableGPUModelRunner, "_update_states"),
        (_CapableGPUModelRunnerV2, "finish_requests"),
    ],
)
def test_capable_worker_variants_reach_state_update(
    monkeypatch, runner_cls, mutation_method
):
    class StateUpdateReached(Exception):
        pass

    runner = runner_cls.__new__(runner_cls)
    runner.layerwise_prefill_p_node = True
    runner.execute_model_state = None
    runner.routed_experts_initialized = False
    runner.speculative_config = None
    runner.prepare_inputs_event = None

    def state_update_reached(*_args, **_kwargs):
        raise StateUpdateReached

    monkeypatch.setattr(runner, mutation_method, state_update_reached)
    with pytest.raises(StateUpdateReached):
        runner.execute_model(SchedulerOutput.make_empty())


def test_capable_worker_exposes_generation_and_canonical_bank_metadata():
    runner = _CapableGPUModelRunner.__new__(_CapableGPUModelRunner)
    runner.layerwise_prefill_p_node = True
    runner.input_batch = SimpleNamespace(req_ids=["req"])
    runner.requests = {
        "req": SimpleNamespace(
            block_ids=([1], [2]),
            block_ids_by_bank=(([1], [2]), ([11], [12])),
            block_allocation_mode=DSABlockAllocationMode.PREFILL_CHILD,
            allocation_generation=9,
        )
    }
    indexer_row = DSAKVRow(
        layer_name="indexer",
        execution_ordinal=6,
        kv_group=1,
        row_ordinal=3,
        bank=1,
    )
    execution = SimpleNamespace(latent=object(), indexer=indexer_row)
    runner.dsa_kv_rows_by_layer_name = {indexer_row.layer_name: indexer_row}
    runner.dsa_kv_executions_by_ordinal = {indexer_row.execution_ordinal: execution}

    assert runner.get_layerwise_prefill_request_generations() == (("req", 9),)
    assert runner.get_layerwise_prefill_block_ids("req", indexer_row, 9) == [12]
    with pytest.raises(RuntimeError, match="uint64 allocation generation"):
        runner.get_layerwise_prefill_block_ids("req", indexer_row, True)
    with pytest.raises(RuntimeError, match="stale.*allocation generation"):
        runner.get_layerwise_prefill_block_ids("req", indexer_row, 8)
    with pytest.raises(RuntimeError, match="absent from the runtime topology"):
        runner.get_layerwise_prefill_block_ids(
            "req",
            DSAKVRow("fabricated", 6, 1, 3, 1),
            9,
        )
    runner.requests["req"].allocation_generation = True
    with pytest.raises(RuntimeError, match="must be a uint64 value"):
        runner.get_layerwise_prefill_request_generations()


def test_common_input_boundary_guards_overriding_runners():
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.layerwise_prefill_p_node = True
    runner.prepare_inputs_event = None

    with (
        pytest.raises(RuntimeError, match="did not opt in"),
        runner.synchronize_input_prep(),
    ):
        pytest.fail("guard must run before overridden worker state updates")
