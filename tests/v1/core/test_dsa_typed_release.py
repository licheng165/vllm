# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the typed DSA commit-evidence / release-permit pipeline.

These tests pin the step-1 vLLM invariants from the "DSA按上下文长度动态切换"
design:

* raw ``DSACommitEvidence`` must never directly free latent blocks (only
  validated ``DSAReleasePermit`` may);
* aggregation across workers / connectors must preserve every reporter record
  and must NOT reduce by ``max(frontier)`` (an early or stale rank must not be
  able to release a new block table);
* ``MultiConnector`` designates a unique DSA release owner and rejects permits
  from non-owners or conflicting permits;
* the scheduler consumes only permits, and falls back to the legacy scalar
  ``completed_decode_window_saves`` map only for requests without a permit.
"""

from unittest.mock import Mock

import pytest

from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
)
from vllm.v1.outputs import (
    DSACommitEvidence,
    DSAReleasePermit,
    KVConnectorOutput,
    ModelRunnerOutput,
)

pytestmark = pytest.mark.cpu_test


# ---------------------------------------------------------------------------
# Data structure validation
# ---------------------------------------------------------------------------


def test_dsa_commit_evidence_rejects_unknown_kind_and_status():
    with pytest.raises(ValueError, match="Unknown DSA release kind"):
        DSACommitEvidence(
            req_id="r", kind="bogus", frontier=8, generation=1, reporter_rank=0
        )
    with pytest.raises(ValueError, match="Unknown DSA evidence status"):
        DSACommitEvidence(
            req_id="r",
            kind="promotion_store",
            frontier=8,
            generation=1,
            reporter_rank=0,
            status="maybe",
        )
    with pytest.raises(ValueError, match="frontier must be non-negative"):
        DSACommitEvidence(
            req_id="r",
            kind="promotion_store",
            frontier=-1,
            generation=1,
            reporter_rank=0,
        )


def test_dsa_commit_evidence_is_hashable_and_equal_for_dedup():
    a = DSACommitEvidence("r", "promotion_store", 8, 1, 0)
    b = DSACommitEvidence("r", "promotion_store", 8, 1, 0)
    c = DSACommitEvidence("r", "promotion_store", 8, 1, 1)  # different reporter
    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    assert len({a, b, c}) == 2


# ---------------------------------------------------------------------------
# KVConnectorOutput.merge: reporter preservation, dedup, permit conflict
# ---------------------------------------------------------------------------


def test_merge_preserves_all_reporters_without_max():
    # Three reporters with the SAME frontier but distinct ranks, plus a dup.
    rank0 = DSACommitEvidence("r", "promotion_store", 8192, 1, 0)
    rank1 = DSACommitEvidence("r", "promotion_store", 8192, 1, 1)
    rank0_dup = DSACommitEvidence("r", "promotion_store", 8192, 1, 0)

    merged = KVConnectorOutput.merge(
        KVConnectorOutput(dsa_commit_evidence=[rank0, rank0_dup]),
        KVConnectorOutput(dsa_commit_evidence=[rank1]),
    )
    assert len(merged.dsa_commit_evidence) == 2
    reporters = {e.reporter_rank for e in merged.dsa_commit_evidence}
    assert reporters == {0, 1}


def test_merge_does_not_reduce_conflicting_frontiers_to_max():
    # A larger frontier from a stale generation must not silently win.
    small = DSACommitEvidence("r", "promotion_store", 4096, 1, 0)
    large = DSACommitEvidence("r", "promotion_store", 8192, 1, 1)
    merged = KVConnectorOutput.merge(
        KVConnectorOutput(dsa_commit_evidence=[small]),
        KVConnectorOutput(dsa_commit_evidence=[large]),
    )
    frontiers = sorted(e.frontier for e in merged.dsa_commit_evidence)
    # Both kept verbatim; no max reduction.
    assert frontiers == [4096, 8192]


def test_merge_permits_conflict_raises():
    p1 = DSAReleasePermit("r", "promotion_store", 8192, 1)
    p2 = DSAReleasePermit("r", "promotion_store", 8192, 2)
    with pytest.raises(ValueError, match="Conflicting DSA release permits"):
        KVConnectorOutput.merge(
            KVConnectorOutput(dsa_release_permits={"r": p1}),
            KVConnectorOutput(dsa_release_permits={"r": p2}),
        )


def test_merge_permits_dedup_identical():
    p = DSAReleasePermit("r", "promotion_store", 8192, 1)
    merged = KVConnectorOutput.merge(
        KVConnectorOutput(dsa_release_permits={"r": p}),
        KVConnectorOutput(dsa_release_permits={"r": p}),
    )
    assert merged.dsa_release_permits == {"r": p}


# ---------------------------------------------------------------------------
# KVOutputAggregator: preserve reporters, no max
# ---------------------------------------------------------------------------


def _mro(evidence):
    out = ModelRunnerOutput(req_ids=["r"], req_id_to_index={"r": 0})
    out.kv_connector_output = KVConnectorOutput(dsa_commit_evidence=evidence)
    return out


def test_aggregator_preserves_reporters_and_dedupes():
    agg = KVOutputAggregator(expected_finished_count=1)
    e0 = DSACommitEvidence("r", "promotion_store", 8192, 1, 0)
    e0_dup = DSACommitEvidence("r", "promotion_store", 8192, 1, 0)
    e1 = DSACommitEvidence("r", "promotion_store", 8192, 1, 1)
    out = agg.aggregate([_mro([e0, e0_dup]), _mro([e1])])
    assert out is not None
    ev = out.kv_connector_output.dsa_commit_evidence
    assert len(ev) == 2
    assert {e.reporter_rank for e in ev} == {0, 1}


def test_aggregator_keeps_conflicting_frontiers_without_max():
    agg = KVOutputAggregator(expected_finished_count=1)
    small = DSACommitEvidence("r", "promotion_store", 4096, 1, 0)
    large = DSACommitEvidence("r", "promotion_store", 8192, 1, 1)
    out = agg.aggregate([_mro([small]), _mro([large])])
    frontiers = sorted(e.frontier for e in out.kv_connector_output.dsa_commit_evidence)
    assert frontiers == [4096, 8192]


# ---------------------------------------------------------------------------
# MultiConnector: evidence aggregation + owner-based permit arbitration
# ---------------------------------------------------------------------------


class _FakeConnector:
    def __init__(self, evidence=None, permits=None, name="c"):
        self._evidence = evidence or []
        self._permits = permits or {}
        self.name = name

    def get_dsa_commit_evidence(self):
        return list(self._evidence)

    def update_connector_output(self, connector_output):
        # Only an owner would normally return permits; non-owners return {}.
        return dict(self._permits)


class _OwnerConnector(_FakeConnector):
    """Declares arbitrate_dsa_release so MultiConnector treats it as owner."""

    def arbitrate_dsa_release(self, *args, **kwargs):  # noqa: D401
        return {}


def _make_multi(connectors):
    mc = MultiConnector.__new__(MultiConnector)
    mc._connectors = list(connectors)
    return mc


def test_multi_get_dsa_commit_evidence_aggregates_and_dedupes():
    e0 = DSACommitEvidence("r", "promotion_store", 8192, 1, 0)
    e1 = DSACommitEvidence("r", "promotion_store", 8192, 1, 1)
    mc = _make_multi([_FakeConnector([e0]), _FakeConnector([e0, e1])])
    ev = mc.get_dsa_commit_evidence()
    assert len(ev) == 2
    assert {e.reporter_rank for e in ev} == {0, 1}


def test_multi_owner_permit_accepted():
    owner = _OwnerConnector(
        permits={"r": DSAReleasePermit("r", "promotion_store", 8192, 1)},
        name="owner",
    )
    other = _FakeConnector(name="other")
    mc = _make_multi([other, owner])
    out = KVConnectorOutput()
    permits = mc.update_connector_output(out)
    assert set(permits) == {"r"}


def test_multi_non_owner_permit_rejected():
    # No connector declares arbitrate_dsa_release, yet one returns permits ->
    # no owner is designated, so this is a misconfiguration and must raise.
    offender = _FakeConnector(
        permits={"r": DSAReleasePermit("r", "promotion_store", 8192, 1)},
        name="offender",
    )
    mc = _make_multi([offender, _FakeConnector(name="other")])
    with pytest.raises(RuntimeError, match="no DSA release owner"):
        mc.update_connector_output(KVConnectorOutput())


def test_multi_owner_is_unique_among_owner_capable_connectors():
    # If two connectors both declare arbitrate_dsa_release, only the first is
    # treated as owner; the second's permits are rejected as non-owner.
    p1 = DSAReleasePermit("r", "promotion_store", 8192, 1)
    p2 = DSAReleasePermit("r", "promotion_store", 8192, 2)
    o1 = _OwnerConnector(permits={"r": p1}, name="o1")
    o2 = _OwnerConnector(permits={"r": p2}, name="o2")
    mc = _make_multi([o1, o2])
    with pytest.raises(RuntimeError, match="Non-owner sub-connector"):
        mc.update_connector_output(KVConnectorOutput())


# ---------------------------------------------------------------------------
# Scheduler: raw evidence never releases; permits release; legacy fallback
# ---------------------------------------------------------------------------


def _make_fake_request():
    req = Mock()
    req.all_token_ids = [1] * 8192
    req.spec_token_ids = []
    req.num_tokens = 8192
    req.status = "RUNNING"
    return req


def _make_fake_scheduler():
    from vllm.v1.core.sched.scheduler import Scheduler

    sched = Scheduler.__new__(Scheduler)
    sched.requests = {}
    sched.kv_cache_manager = Mock()
    sched.kv_cache_manager.remove_saved_decode_window_blocks = Mock(return_value=3)
    sched.connector = Mock()
    # By default the connector arbitrates nothing (no permits): raw evidence
    # alone must never release.
    sched.connector.update_connector_output = Mock(return_value={})
    return sched


def test_raw_evidence_alone_never_releases():
    sched = _make_fake_scheduler()
    evidence = [
        DSACommitEvidence("r", "promotion_store", 8192, 1, 0),
        DSACommitEvidence("r", "promotion_store", 8192, 1, 1),
    ]
    out = KVConnectorOutput(dsa_commit_evidence=evidence)
    sched._update_from_kv_xfer_finished(out)
    sched.kv_cache_manager.remove_saved_decode_window_blocks.assert_not_called()


def test_permits_release_only_named_requests():
    sched = _make_fake_scheduler()
    sched.requests = {"r1": _make_fake_request(), "r2": _make_fake_request()}
    permits = {
        "r1": DSAReleasePermit("r1", "promotion_store", 8192, 1),
    }
    sched.connector.update_connector_output = Mock(return_value=permits)
    # Legacy scalar also mentions r2; r1 must be skipped by the legacy path.
    out = KVConnectorOutput(completed_decode_window_saves={"r2": 4096})
    sched._update_from_kv_xfer_finished(out)

    released = {
        call.args[0]
        for call in (
            sched.kv_cache_manager.remove_saved_decode_window_blocks.call_args_list
        )
    }
    assert released == {"r1", "r2"}
    # r1 released via permit at frontier 8192; r2 via legacy scalar at 4096.
    per_req = {
        call.args[0]: call.args[1]
        for call in (
            sched.kv_cache_manager.remove_saved_decode_window_blocks.call_args_list
        )
    }
    assert per_req["r1"] == 8192
    assert per_req["r2"] == 4096


def test_permit_skips_legacy_path_for_same_request():
    sched = _make_fake_scheduler()
    sched.requests = {"r": _make_fake_request()}
    permits = {"r": DSAReleasePermit("r", "promotion_store", 8192, 1)}
    sched.connector.update_connector_output = Mock(return_value=permits)
    out = KVConnectorOutput(completed_decode_window_saves={"r": 8192})
    sched._update_from_kv_xfer_finished(out)
    # Exactly one release for r (permit path wins; legacy skipped).
    assert sched.kv_cache_manager.remove_saved_decode_window_blocks.call_count == 1
    call = sched.kv_cache_manager.remove_saved_decode_window_blocks.call_args
    assert call.args[0] == "r"
    assert call.args[1] == 8192


def test_permit_for_unknown_request_skipped():
    sched = _make_fake_scheduler()
    sched.requests = {}
    permits = {"ghost": DSAReleasePermit("ghost", "promotion_store", 8192, 1)}
    sched.connector.update_connector_output = Mock(return_value=permits)
    out = KVConnectorOutput()
    sched._update_from_kv_xfer_finished(out)
    sched.kv_cache_manager.remove_saved_decode_window_blocks.assert_not_called()
