# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the DSA exact-set receipt aggregator (§7.7/§13.3)."""

from __future__ import annotations

import logging

import pytest

from vllm.observability.dsa_aggregator import (
    DSAReceiptAggregator,
    OperationSlot,
    OperationSpec,
)
from vllm.observability.dsa_offload import (
    DSAOperationReceipt,
    ParticipantIdentity,
    RequestKey,
)


def _cap(monkeypatch):
    records: list[str] = []

    class _H(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    lg = logging.getLogger("vllm.observability.dsa_offload")
    h = _H()
    h.setLevel(logging.DEBUG)
    lg.addHandler(h)
    lg.setLevel(logging.DEBUG)
    return records, h, lg


def _mk_part(rank=0) -> ParticipantIdentity:
    return ParticipantIdentity("e0", "p0", rank, 0, 0, rank)


def _mk_receipt(
    op_id, participant, kv_group, range_end, *, status="complete",
    kind="storage",
) -> DSAOperationReceipt:
    return DSAOperationReceipt(
        receipt_id=f"r-{op_id}-{kv_group}-{participant.tp_rank}",
        request_key=RequestKey("p0", "req0", 1),
        operation_id=op_id,
        receipt_kind=kind,
        route_epoch=0,
        input_generation_id=None,
        output_generation_id=None,
        accepted_end_at_seal=range_end,
        token_prefix_digest="d",
        cache_namespace_fingerprint="ns",
        range_start=0,
        range_end=range_end,
        kv_group=kv_group,
        participant=participant,
        covered_layers=(0,),
        covered_chunks=((0, 0),),
        storage_tier="local_cpu",
        status=status,
        lease_descriptor_id=None,
        guarantee="local_cpu_pinned",
    )


def _mk_spec(op_id, slots, frontier_ids) -> OperationSpec:
    return OperationSpec(
        operation_id=op_id,
        request_key=RequestKey("p0", "req0", 1),
        route_epoch=0,
        input_generation_id=None,
        token_prefix_digest="d",
        cache_namespace_fingerprint="ns",
        data_compatibility_fingerprint="compat",
        expected_slots=frozenset(slots),
        frontier_participant_ids=frozenset(frontier_ids),
    )


def _reset_state(monkeypatch, level):
    """Reset diag state to pick up the requested level without reloading
    modules (reload would break frozen-dataclass identity across modules)."""
    import vllm.observability.dsa_offload as mod
    monkeypatch.setenv("VLLM_ASCEND_DSA_DIAG_LEVEL", level)
    mod._STATE = None
    # Clear any bound state on existing logger views so they re-resolve lazily.
    mod._root_logger._state = None


def test_no_publish_until_exact_set_complete(monkeypatch):
    _reset_state(monkeypatch, "lifecycle")
    records, h, lg = _cap(monkeypatch)

    p = _mk_part(0)
    slots = [
        OperationSlot("e0:0:0:0:0", 0, "storage", True),
        OperationSlot("e0:0:0:0:0", 1, "storage", True),
    ]
    agg = DSAReceiptAggregator()
    agg.register_operation(_mk_spec("op-1", slots,
                                    ["e0:0:0:0:0"]))
    # Only group 0 so far -> no seal.
    assert agg.add_receipt(_mk_receipt("op-1", p, 0, 8192)) is None
    assert not agg.is_complete("op-1")
    # Group 1 completes the exact set -> sealed.
    bundle = agg.add_receipt(_mk_receipt("op-1", p, 1, 8192))
    assert bundle is not None
    assert agg.is_complete("op-1")
    # canonical frontier.publish with proven quorum emitted.
    publishes = [r for r in records if '"frontier.publish"' in r]
    assert len(publishes) == 1
    assert '"quorum_status":"proven"' in publishes[0]
    lg.removeHandler(h)


def test_common_frontier_is_min_not_max(monkeypatch):
    _reset_state(monkeypatch, "off")

    p0 = _mk_part(0)
    p1 = _mk_part(1)
    slots = [
        OperationSlot("e0:0:0:0:0", 0, "storage", True),
        OperationSlot("e0:1:0:0:1", 0, "storage", True),
    ]
    agg = DSAReceiptAggregator()
    agg.register_operation(_mk_spec(
        "op-2", slots,
        ["e0:0:0:0:0", "e0:1:0:0:1"]))
    agg.add_receipt(_mk_receipt("op-2", p0, 0, 9000))
    bundle = agg.add_receipt(_mk_receipt("op-2", p1, 0, 7000))
    assert bundle is not None
    # Common frontier is min(9000, 7000) == 7000, never max.
    assert bundle.sparse_source_end == 7000
    assert bundle.raw_source_end == 7000


def test_failed_receipt_prevents_publish(monkeypatch):
    _reset_state(monkeypatch, "lifecycle")
    records, h, lg = _cap(monkeypatch)

    p = _mk_part(0)
    slots = [
        OperationSlot("e0:0:0:0:0", 0, "storage", True),
        OperationSlot("e0:0:0:0:0", 1, "storage", True),
    ]
    agg = DSAReceiptAggregator()
    agg.register_operation(_mk_spec("op-3", slots, ["e0:0:0:0:0"]))
    # group 0 failed -> operation fails, group 1 success cannot publish.
    assert agg.add_receipt(
        _mk_receipt("op-3", p, 0, 0, status="failed")) is None
    assert agg.add_receipt(_mk_receipt("op-3", p, 1, 8192)) is None
    assert not agg.is_complete("op-3")
    publishes = [r for r in records if '"frontier.publish"' in r]
    assert publishes == []
    fails = [r for r in records
             if '"store.batch.fenced"' in r and '"error"' in r]
    assert len(fails) == 1
    lg.removeHandler(h)


def test_unexpected_slot_ignored(monkeypatch):
    _reset_state(monkeypatch, "off")

    p = _mk_part(0)
    slots = [OperationSlot("e0:0:0:0:0", 0, "storage", True)]
    agg = DSAReceiptAggregator()
    agg.register_operation(_mk_spec("op-4", slots, ["e0:0:0:0:0"]))
    # group 1 is not expected -> ignored, no seal.
    assert agg.add_receipt(_mk_receipt("op-4", p, 1, 8192)) is None
    bundle = agg.add_receipt(_mk_receipt("op-4", p, 0, 8192))
    assert bundle is not None
