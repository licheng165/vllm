# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from math import gcd


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)


def dsa_block_pool_index(
    group_index: int,
    *,
    use_per_group_block_pools: bool,
    use_dsa_shared_block_pool: bool,
) -> int:
    """Map a KV cache group to the BlockPool/LogicalBlockPool it should use."""

    if use_per_group_block_pools or use_dsa_shared_block_pool:
        return group_index
    return 0


def dsa_scratch_blocks_for_topk(
    index_topk: int,
    block_size: int,
    num_rows: int = 1,
) -> int:
    """Number of latent blocks needed to hold sparse decode top-k rows."""

    if index_topk <= 0:
        raise ValueError("index_topk must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if num_rows <= 0:
        raise ValueError("num_rows must be positive")
    if index_topk % block_size:
        raise ValueError(
            "DSA index_topk must be an integer multiple of block_size: "
            f"index_topk={index_topk}, block_size={block_size}. Configure "
            "index_topk to N * block_size."
        )
    return num_rows * index_topk // block_size


class DSASharedBlockOwner(str, Enum):
    LATENT = "latent"
    INDEXER = "indexer"
    PREFILL_RESERVED = "prefill_reserved"


class DSABlockAllocationMode(str, Enum):
    FULL_PARENT = "full_parent"
    PREFILL_CHILD = "prefill_child"


LAYERWISE_PREFILL_BANK_COUNT = 2
MAX_ALLOCATION_GENERATION = (1 << 64) - 1


@dataclass(frozen=True)
class DSASharedBlockLayout:
    """Logical block-id mapping for the DSA shared bundle pool.

    Bundle slot 0 is reserved so block table padding can continue using block
    id 0. Real allocations start at bundle id 1.
    """

    latent_page_size_bytes: int
    indexer_page_size_bytes: int
    capacity_bundles: int
    k_nope_dim: int = 512
    k_pe_dim: int = 64
    indexer_dim: int = 128

    def __post_init__(self) -> None:
        if self.latent_page_size_bytes <= 0 or self.indexer_page_size_bytes <= 0:
            raise ValueError("page sizes must be positive")
        if self.capacity_bundles <= 0:
            raise ValueError("capacity_bundles must be positive")
        if self.latent_dim != self.k_nope_dim + self.k_pe_dim:
            raise ValueError(
                "latent_page/indexer_page ratio does not match k_nope+k_pe dims"
            )
        if self.nope_pages_per_bundle + self.pe_pages_per_bundle != (
            self.indexer_blocks_per_bundle
        ):
            raise ValueError("DSA bundle page split is inconsistent")

    @property
    def bundle_page_size_bytes(self) -> int:
        return _lcm(self.latent_page_size_bytes, self.indexer_page_size_bytes)

    @property
    def latent_blocks_per_bundle(self) -> int:
        return self.bundle_page_size_bytes // self.latent_page_size_bytes

    @property
    def indexer_blocks_per_bundle(self) -> int:
        return self.bundle_page_size_bytes // self.indexer_page_size_bytes

    @property
    def latent_dim(self) -> int:
        return (
            self.indexer_dim
            * self.latent_page_size_bytes
            // self.indexer_page_size_bytes
        )

    @property
    def slot_count(self) -> int:
        return self.capacity_bundles + 1

    @property
    def nope_pages_per_bundle(self) -> int:
        return self.latent_blocks_per_bundle * self.k_nope_dim // self.indexer_dim

    @property
    def pe_pages_per_bundle(self) -> int:
        return self.latent_blocks_per_bundle * self.k_pe_dim // self.indexer_dim

    def blocks_per_bundle(self, owner: DSASharedBlockOwner) -> int:
        if owner == DSASharedBlockOwner.LATENT:
            return self.latent_blocks_per_bundle
        if owner == DSASharedBlockOwner.INDEXER:
            return self.indexer_blocks_per_bundle
        raise AssertionError(f"Unexpected DSA owner: {owner}")

    def row_count(self, owner: DSASharedBlockOwner) -> int:
        return self.slot_count * self.blocks_per_bundle(owner)

    def block_ids_for_bundle(
        self,
        owner: DSASharedBlockOwner,
        bundle_id: int,
    ) -> tuple[int, ...]:
        if bundle_id <= 0 or bundle_id > self.capacity_bundles:
            raise ValueError(f"invalid bundle id {bundle_id}")
        if owner == DSASharedBlockOwner.LATENT:
            start = bundle_id * self.latent_blocks_per_bundle
            return tuple(start + i for i in range(self.latent_blocks_per_bundle))

        if owner == DSASharedBlockOwner.INDEXER:
            nope_start = bundle_id * self.nope_pages_per_bundle
            pe_start = (
                self.slot_count * self.nope_pages_per_bundle
                + bundle_id * self.pe_pages_per_bundle
            )
            return tuple(
                list(range(nope_start, nope_start + self.nope_pages_per_bundle))
                + list(range(pe_start, pe_start + self.pe_pages_per_bundle))
            )
        raise AssertionError(f"Unexpected DSA owner: {owner}")

    def bundle_id_for_block(
        self,
        owner: DSASharedBlockOwner,
        block_id: int,
    ) -> int:
        if block_id <= 0:
            raise ValueError(f"block id {block_id} is not a real DSA block")
        if owner == DSASharedBlockOwner.LATENT:
            return block_id // self.latent_blocks_per_bundle

        if owner == DSASharedBlockOwner.INDEXER:
            nope_slab_pages = self.slot_count * self.nope_pages_per_bundle
            if block_id < nope_slab_pages:
                return block_id // self.nope_pages_per_bundle
            return (block_id - nope_slab_pages) // self.pe_pages_per_bundle
        raise AssertionError(f"Unexpected DSA owner: {owner}")


class DSASharedBundleAllocator:
    """Physical bundle allocator shared by DSA latent and indexer wrappers."""

    def __init__(self, layout: DSASharedBlockLayout) -> None:
        self.layout = layout
        self._owners: dict[int, DSASharedBlockOwner] = {}
        self._free_ranges: list[tuple[int, int]] = [(1, layout.capacity_bundles)]

    @property
    def free_bundle_count(self) -> int:
        return sum(end - start + 1 for start, end in self._free_ranges)

    @property
    def free_range_count(self) -> int:
        return len(self._free_ranges)

    @property
    def largest_free_range(self) -> int:
        if not self._free_ranges:
            return 0
        return max(end - start + 1 for start, end in self._free_ranges)

    def owner_bundle_counts(self) -> Counter[DSASharedBlockOwner]:
        return Counter(self._owners.values())

    def bundle_count_for_blocks(
        self,
        owner: DSASharedBlockOwner,
        num_blocks: int,
    ) -> int:
        return _cdiv(num_blocks, self.layout.blocks_per_bundle(owner))

    def allocate(
        self,
        owner: DSASharedBlockOwner,
        num_bundles: int,
    ) -> tuple[int, ...]:
        if num_bundles <= 0:
            return ()
        if num_bundles > self.free_bundle_count:
            raise ValueError(
                f"Cannot get {num_bundles} DSA bundles from the shared pool"
            )
        chosen = self._take_from_best_contiguous_range(num_bundles)
        if chosen is None:
            chosen_list: list[int] = []
            while len(chosen_list) < num_bundles:
                need = num_bundles - len(chosen_list)
                start, end = self._free_ranges.pop(0)
                take = min(need, end - start + 1)
                chosen_list.extend(range(start, start + take))
                if start + take <= end:
                    self._free_ranges.insert(0, (start + take, end))
            chosen = tuple(chosen_list)
        for bundle_id in chosen:
            self._owners[bundle_id] = owner
        return chosen

    def free(
        self,
        owner: DSASharedBlockOwner,
        bundle_ids: Iterable[int],
    ) -> None:
        ids = tuple(bundle_ids)
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate DSA bundle ids in free: {ids}")
        for bundle_id in ids:
            actual = self._owners.get(bundle_id)
            if actual != owner:
                raise ValueError(
                    f"DSA bundle {bundle_id} is owned by {actual}, not {owner}"
                )
        for bundle_id in ids:
            del self._owners[bundle_id]
            self._insert_free_bundle(bundle_id)

    def _take_from_best_contiguous_range(
        self,
        num_bundles: int,
    ) -> tuple[int, ...] | None:
        best_idx: int | None = None
        best_len: int | None = None
        for idx, (start, end) in enumerate(self._free_ranges):
            length = end - start + 1
            if length >= num_bundles and (best_len is None or length < best_len):
                best_idx = idx
                best_len = length
        if best_idx is None:
            return None

        start, end = self._free_ranges[best_idx]
        chosen = tuple(range(start, start + num_bundles))
        if start + num_bundles <= end:
            self._free_ranges[best_idx] = (start + num_bundles, end)
        else:
            del self._free_ranges[best_idx]
        return chosen

    def _insert_free_bundle(self, bundle_id: int) -> None:
        self._free_ranges.append((bundle_id, bundle_id))
        self._free_ranges.sort()
        merged: list[tuple[int, int]] = []
        for start, end in self._free_ranges:
            if not merged or start > merged[-1][1] + 1:
                merged.append((start, end))
            else:
                prev_start, prev_end = merged[-1]
                merged[-1] = (prev_start, max(prev_end, end))
        self._free_ranges = merged


@dataclass
class PrefillRequestArena:
    """Child bundles owned by one request allocation generation."""

    request_id: str
    allocation_generation: int
    block_ids_by_owner: dict[DSASharedBlockOwner, tuple[list[int], ...]]

    @property
    def allocated_child_ids(self) -> tuple[int, ...]:
        return tuple(
            child_id
            for owner_banks in self.block_ids_by_owner.values()
            for bank in owner_banks
            for child_id in bank
        )


@dataclass
class _PrefillChildState:
    owner: DSASharedBlockOwner
    owner_request_id: str
    allocation_generation: int
    bank: int
    allocator_refcount: int = 1
    source_lease_count: int = 0


@dataclass(frozen=True)
class _PrefillAllocationTransaction:
    request_id: str
    allocation_generation: int
    existing_child_ids: frozenset[int]


class PrefillLayerBundlePool:
    """Layer-agnostic children backed by reserved full-cache parents."""

    def __init__(
        self,
        parent_allocator: DSASharedBundleAllocator,
        num_physical_slots: int,
        *,
        bank_count: int = LAYERWISE_PREFILL_BANK_COUNT,
    ) -> None:
        if num_physical_slots <= 0:
            raise ValueError("num_physical_slots must be positive")
        if bank_count != LAYERWISE_PREFILL_BANK_COUNT:
            raise ValueError("layerwise prefill requires exactly two physical banks")
        self.parent_allocator = parent_allocator
        self.num_physical_slots = num_physical_slots
        self.bank_count = bank_count
        parent_layout = parent_allocator.layout
        self.parent_capacity = parent_layout.capacity_bundles
        self.layout = DSASharedBlockLayout(
            latent_page_size_bytes=parent_layout.latent_page_size_bytes,
            indexer_page_size_bytes=parent_layout.indexer_page_size_bytes,
            capacity_bundles=self.parent_capacity * num_physical_slots,
            k_nope_dim=parent_layout.k_nope_dim,
            k_pe_dim=parent_layout.k_pe_dim,
            indexer_dim=parent_layout.indexer_dim,
        )
        self._children: dict[int, _PrefillChildState] = {}
        self._reserved_parents: set[int] = set()
        self._arenas: dict[tuple[str, int], PrefillRequestArena] = {}
        self._active_allocation: _PrefillAllocationTransaction | None = None

    def child_bundle_id(self, physical_slot: int, parent_bundle_id: int) -> int:
        if physical_slot < 0 or physical_slot >= self.num_physical_slots:
            raise ValueError(f"invalid physical slot {physical_slot}")
        if parent_bundle_id <= 0 or parent_bundle_id > self.parent_capacity:
            raise ValueError(f"invalid parent bundle id {parent_bundle_id}")
        return physical_slot * self.parent_capacity + parent_bundle_id

    def parent_bundle_id(self, child_bundle_id: int) -> int:
        self._validate_child_id(child_bundle_id)
        return (child_bundle_id - 1) % self.parent_capacity + 1

    def physical_slot(self, child_bundle_id: int) -> int:
        self._validate_child_id(child_bundle_id)
        return (child_bundle_id - 1) // self.parent_capacity

    def _validate_child_id(self, child_bundle_id: int) -> None:
        if child_bundle_id <= 0 or child_bundle_id > self.layout.capacity_bundles:
            raise ValueError(f"invalid child bundle id {child_bundle_id}")

    @staticmethod
    def _validate_generation(allocation_generation: int) -> None:
        if not 0 < allocation_generation <= MAX_ALLOCATION_GENERATION:
            raise ValueError(
                "allocation_generation must be a positive unsigned 64-bit "
                f"integer, got {allocation_generation}"
            )

    def begin_request_allocation(
        self, request_id: str, allocation_generation: int
    ) -> None:
        self._validate_generation(allocation_generation)
        if self._active_allocation is not None:
            raise RuntimeError("a layerwise-prefill allocation is already active")
        self._active_allocation = _PrefillAllocationTransaction(
            request_id,
            allocation_generation,
            frozenset(self._children),
        )

    def end_request_allocation(
        self, request_id: str, allocation_generation: int
    ) -> None:
        expected = (request_id, allocation_generation)
        if self.active_request_allocation != expected:
            raise RuntimeError(
                "layerwise-prefill allocation context mismatch: "
                f"active={self.active_request_allocation}, closing={expected}"
            )
        self._active_allocation = None

    def rollback_request_allocation(
        self, request_id: str, allocation_generation: int
    ) -> None:
        expected = (request_id, allocation_generation)
        if self.active_request_allocation != expected:
            raise RuntimeError(
                "layerwise-prefill allocation rollback context mismatch: "
                f"active={self.active_request_allocation}, rollback={expected}"
            )
        transaction = self._active_allocation
        assert transaction is not None
        self._rollback_children_since(
            transaction.existing_child_ids, request_id, allocation_generation
        )

    @property
    def active_request_allocation(self) -> tuple[str, int]:
        if self._active_allocation is None:
            raise RuntimeError(
                "PREFILL_CHILD allocation requires scheduler-owned request "
                "identity and allocation_generation"
            )
        return (
            self._active_allocation.request_id,
            self._active_allocation.allocation_generation,
        )

    def get_request_arena(
        self, request_id: str, allocation_generation: int
    ) -> PrefillRequestArena | None:
        return self._arenas.get((request_id, allocation_generation))

    @property
    def reserved_parent_count(self) -> int:
        return len(self._reserved_parents)

    @property
    def materialized_free_bundle_count(self) -> int:
        return self.reserved_parent_count * self.num_physical_slots - len(
            self._children
        )

    @property
    def free_bundle_count(self) -> int:
        return self.materialized_free_bundle_count + (
            self.parent_allocator.free_bundle_count * self.num_physical_slots
        )

    @property
    def free_range_count(self) -> int:
        return self.reserved_parent_count

    @property
    def largest_free_range(self) -> int:
        if not self.free_bundle_count:
            return 0
        return max(self.num_physical_slots, self.materialized_free_bundle_count)

    def owner_bundle_counts(self) -> Counter[DSASharedBlockOwner]:
        return Counter(state.owner for state in self._children.values())

    def logical_bundle_count_for_blocks(
        self, owner: DSASharedBlockOwner, num_blocks: int
    ) -> int:
        return _cdiv(num_blocks, self.layout.blocks_per_bundle(owner))

    def bundle_count_for_blocks(
        self, owner: DSASharedBlockOwner, num_blocks: int
    ) -> int:
        return self.bank_count * self.logical_bundle_count_for_blocks(owner, num_blocks)

    def allocate_banks(
        self,
        owner: DSASharedBlockOwner,
        num_logical_bundles: int,
    ) -> tuple[tuple[int, ...], ...]:
        if owner == DSASharedBlockOwner.PREFILL_RESERVED:
            raise ValueError("PREFILL_RESERVED is not a child owner")
        if num_logical_bundles <= 0:
            return tuple(() for _ in range(self.bank_count))
        request_id, generation = self.active_request_allocation
        checkpoint = frozenset(self._children)
        allocated: list[tuple[int, ...]] = []
        try:
            for bank in range(self.bank_count):
                allocated.append(
                    self._allocate_leg(
                        owner,
                        bank,
                        num_logical_bundles,
                        request_id,
                        generation,
                    )
                )
        except BaseException:
            self._rollback_children_since(checkpoint, request_id, generation)
            raise
        return tuple(allocated)

    def _allocate_leg(
        self,
        owner: DSASharedBlockOwner,
        bank: int,
        num_bundles: int,
        request_id: str,
        generation: int,
    ) -> tuple[int, ...]:
        if owner not in (DSASharedBlockOwner.LATENT, DSASharedBlockOwner.INDEXER):
            raise ValueError(f"invalid layerwise-prefill child owner {owner}")
        if bank < 0 or bank >= self.bank_count:
            raise ValueError(f"invalid layerwise-prefill bank {bank}")
        if self.active_request_allocation != (request_id, generation):
            raise RuntimeError("layerwise-prefill allocation leg context mismatch")
        if num_bundles > self.free_bundle_count:
            raise ValueError(
                f"Cannot get {num_bundles} DSA child bundles from the "
                "layerwise prefill pool"
            )

        free_children = self._materialized_free_children()
        missing = num_bundles - len(free_children)
        if missing > 0:
            parents_needed = _cdiv(missing, self.num_physical_slots)
            parent_ids = self.parent_allocator.allocate(
                DSASharedBlockOwner.PREFILL_RESERVED, parents_needed
            )
            self._reserved_parents.update(parent_ids)
            free_children = self._materialized_free_children()

        chosen = tuple(free_children[:num_bundles])
        if len(chosen) != num_bundles:
            raise AssertionError(
                "materialized DSA child capacity disagrees with admission"
            )
        arena = self._arenas.get((request_id, generation))
        if arena is None:
            arena = PrefillRequestArena(
                request_id=request_id,
                allocation_generation=generation,
                block_ids_by_owner={
                    child_owner: tuple([] for _ in range(self.bank_count))
                    for child_owner in (
                        DSASharedBlockOwner.LATENT,
                        DSASharedBlockOwner.INDEXER,
                    )
                },
            )
            self._arenas[(request_id, generation)] = arena
        owner_banks = arena.block_ids_by_owner[owner]
        owner_banks[bank].extend(chosen)
        for child_id in chosen:
            self._children[child_id] = _PrefillChildState(
                owner=owner,
                owner_request_id=request_id,
                allocation_generation=generation,
                bank=bank,
            )
        return chosen

    def _rollback_children_since(
        self,
        checkpoint: frozenset[int],
        request_id: str,
        allocation_generation: int,
    ) -> None:
        child_ids = tuple(
            child_id for child_id in self._children if child_id not in checkpoint
        )
        states = self._validate_owned_children_for_request(
            child_ids, request_id, allocation_generation
        )
        if any(state.source_lease_count for state in states):
            raise RuntimeError("cannot roll back leased DSA child bundles")

        arena = self._arenas.get((request_id, allocation_generation))
        if arena is None and child_ids:
            raise RuntimeError("layerwise-prefill rollback is missing its arena")
        for child_id, state in zip(child_ids, states):
            if arena is not None:
                arena.block_ids_by_owner[state.owner][state.bank].remove(child_id)
            del self._children[child_id]
        self._release_empty_parents(set(self._reserved_parents))
        self._drop_empty_arena(request_id, allocation_generation)

    def free(
        self,
        owner: DSASharedBlockOwner,
        bundle_ids: Iterable[int],
        request_id: str,
        allocation_generation: int,
    ) -> None:
        ids = tuple(bundle_ids)
        states = self._validate_owned_children(
            owner, ids, request_id, allocation_generation
        )
        for child_id, state in zip(ids, states):
            if state.allocator_refcount <= 0:
                raise ValueError(f"DSA child bundle {child_id} is already free")

        affected_parents = {self.parent_bundle_id(child_id) for child_id in ids}
        arena = self._arenas[(request_id, allocation_generation)]
        for child_id, state in zip(ids, states):
            state.allocator_refcount -= 1
            arena.block_ids_by_owner[owner][state.bank].remove(child_id)
            if state.source_lease_count == 0:
                del self._children[child_id]
        self._release_empty_parents(affected_parents)
        self._drop_empty_arena(request_id, allocation_generation)

    def acquire_source_lease(
        self,
        owner: DSASharedBlockOwner,
        bundle_ids: Iterable[int],
        request_id: str,
        allocation_generation: int,
    ) -> None:
        ids = tuple(bundle_ids)
        states = self._validate_owned_children(
            owner, ids, request_id, allocation_generation
        )
        if any(state.allocator_refcount <= 0 for state in states):
            raise ValueError("cannot lease an allocator-free DSA child bundle")
        for state in states:
            state.source_lease_count += 1

    def release_source_lease(
        self,
        owner: DSASharedBlockOwner,
        bundle_ids: Iterable[int],
        request_id: str,
        allocation_generation: int,
    ) -> None:
        ids = tuple(bundle_ids)
        states = self._validate_owned_children(
            owner, ids, request_id, allocation_generation
        )
        if any(state.source_lease_count <= 0 for state in states):
            raise ValueError("DSA child source lease is already released")
        affected_parents = {self.parent_bundle_id(child_id) for child_id in ids}
        for child_id, state in zip(ids, states):
            state.source_lease_count -= 1
            if state.source_lease_count == 0 and state.allocator_refcount == 0:
                del self._children[child_id]
        self._release_empty_parents(affected_parents)
        self._drop_empty_arena(request_id, allocation_generation)

    def validate_owned_children(
        self,
        owner: DSASharedBlockOwner,
        bundle_ids: Iterable[int],
        request_id: str,
        allocation_generation: int,
        bank: int | None = None,
        require_allocator_refcount: bool = True,
    ) -> None:
        states = self._validate_owned_children(
            owner,
            tuple(bundle_ids),
            request_id,
            allocation_generation,
        )
        if bank is not None:
            if bank < 0 or bank >= self.bank_count:
                raise ValueError(f"invalid layerwise-prefill bank {bank}")
            if any(state.bank != bank for state in states):
                raise ValueError("DSA child bundle belongs to a different bank")
        if require_allocator_refcount and any(
            state.allocator_refcount <= 0 for state in states
        ):
            raise ValueError("DSA child bundle is already allocator-free")

    def _validate_owned_children(
        self,
        owner: DSASharedBlockOwner,
        ids: tuple[int, ...],
        request_id: str,
        generation: int,
    ) -> tuple[_PrefillChildState, ...]:
        self._validate_generation(generation)
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate DSA child bundle ids in release: {ids}")
        states: list[_PrefillChildState] = []
        for child_id in ids:
            self._validate_child_id(child_id)
            state = self._children.get(child_id)
            if state is None:
                raise ValueError(f"DSA child bundle {child_id} is already free")
            if state.allocation_generation != generation:
                raise ValueError(
                    f"stale allocation generation for DSA child {child_id}: "
                    f"expected {state.allocation_generation}, got {generation}"
                )
            if state.owner_request_id != request_id:
                raise ValueError(
                    f"DSA child bundle {child_id} is owned by request "
                    f"{state.owner_request_id!r}, not {request_id!r}"
                )
            if state.owner != owner:
                raise ValueError(
                    f"DSA child bundle {child_id} is owned by {state.owner}, "
                    f"not {owner}"
                )
            states.append(state)
        return tuple(states)

    def _validate_owned_children_for_request(
        self,
        ids: tuple[int, ...],
        request_id: str,
        generation: int,
    ) -> tuple[_PrefillChildState, ...]:
        self._validate_generation(generation)
        states: list[_PrefillChildState] = []
        for child_id in ids:
            state = self._children[child_id]
            if (
                state.owner_request_id != request_id
                or state.allocation_generation != generation
            ):
                raise RuntimeError(
                    "layerwise-prefill transaction captured another request's child"
                )
            states.append(state)
        return tuple(states)

    def _materialized_free_children(self) -> list[int]:
        return [
            self.child_bundle_id(slot, parent_id)
            for parent_id in sorted(self._reserved_parents)
            for slot in range(self.num_physical_slots)
            if self.child_bundle_id(slot, parent_id) not in self._children
        ]

    def _release_empty_parents(self, parent_ids: set[int]) -> None:
        releasable = tuple(
            parent_id
            for parent_id in sorted(parent_ids)
            if parent_id in self._reserved_parents
            and not any(
                self.parent_bundle_id(child_id) == parent_id
                for child_id in self._children
            )
        )
        if releasable:
            self.parent_allocator.free(DSASharedBlockOwner.PREFILL_RESERVED, releasable)
            self._reserved_parents.difference_update(releasable)

    def _drop_empty_arena(self, request_id: str, generation: int) -> None:
        if not any(
            state.owner_request_id == request_id
            and state.allocation_generation == generation
            for state in self._children.values()
        ):
            self._arenas.pop((request_id, generation), None)
