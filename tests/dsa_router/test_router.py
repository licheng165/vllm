# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the DSA Router handoff state machine (§6.5/§15.9)."""

from __future__ import annotations

import logging

import pytest

from vllm.dsa_router import DSARouter, TransferState
from vllm.observability.dsa_offload import RequestKey


def _cap():
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


def _pk(tag):
    return RequestKey(tag, "req", 1)


def test_happy_path_dispatch_to_release(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_DSA_DIAG_LEVEL", "lifecycle")
    import vllm.observability.dsa_offload as mod
    mod._STATE = None
    mod._root_logger._state = None
    records, h, lg = _cap()

    r = DSARouter(lease_ttl_s=30, qualified_ttl_s=300)
    lid = r.prefill_dispatch("x-1", 1, _pk("p"))
    assert r.get_lease(lid).state == TransferState.P_DISPATCHED

    ok, _ = r.kv_ready("x-1", 1, _pk("p"), "b-1", lid)
    assert ok
    assert r.get_lease(lid).state == TransferState.P_READY

    ok, de = r.route_dispatch("x-1", "decoder0")
    assert ok and de == 1
    assert r.get_lease(lid).state == TransferState.D_DISPATCHED

    ok = r.decode_materialized("x-1", 1, 1, _pk("d"))
    assert ok
    assert r.get_lease(lid).state == TransferState.RELEASED

    leased = [x for x in records if '"remote.handoff.leased"' in x]
    released = [x for x in records if '"remote.handoff.released"' in x]
    assert len(leased) == 1
    assert '"remote_guarantee":"handoff_leased"' in leased[0]
    assert len(released) == 1
    lg.removeHandler(h)


def test_stale_source_epoch_rejected(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_DSA_DIAG_LEVEL", "off")
    import vllm.observability.dsa_offload as mod
    mod._STATE = None
    mod._root_logger._state = None

    r = DSARouter()
    lid = r.prefill_dispatch("x-2", 1, _pk("p"))
    # A retried/stale producer attempt (source_epoch=2) must be rejected; the
    # lease stays at its prior state and is NOT handed off.
    ok, _ = r.kv_ready("x-2", 2, _pk("p"), "b-2", lid)
    assert ok is False
    assert r.get_lease(lid).state == TransferState.P_DISPATCHED


def test_dispatch_without_ready_rejected(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_DSA_DIAG_LEVEL", "off")
    import vllm.observability.dsa_offload as mod
    mod._STATE = None
    mod._root_logger._state = None

    r = DSARouter()
    r.prefill_dispatch("x-3", 1, _pk("p"))
    ok, _ = r.route_dispatch("x-3", "decoder0")
    assert ok is False  # no kv_ready accepted yet


def test_failure_quiesces_then_ack_releases(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_DSA_DIAG_LEVEL", "off")
    import vllm.observability.dsa_offload as mod
    mod._STATE = None
    mod._root_logger._state = None

    r = DSARouter()
    lid = r.prefill_dispatch("x-4", 1, _pk("p"))
    r.kv_ready("x-4", 1, _pk("p"), "b-4", lid)
    _, de = r.route_dispatch("x-4", "decoder0")
    # Decoder fails: quiesce, do NOT release yet.
    r.transfer_failed("x-4", "decoder")
    assert r.get_lease(lid).state == TransferState.QUIESCING
    # D quiesce ack releases.
    assert r.decode_quiesced("x-4", de)
    assert r.get_lease(lid).state == TransferState.RELEASED


def test_qualified_ttl_not_reclaimed_immediately(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_DSA_DIAG_LEVEL", "off")
    import vllm.observability.dsa_offload as mod
    mod._STATE = None
    mod._root_logger._state = None

    now = [0]

    def clk():
        return now[0]

    r = DSARouter(lease_ttl_s=1.0, qualified_ttl_s=100.0, clock_ns=clk)
    lid = r.prefill_dispatch("x-5", 1, _pk("p"))
    # Renewal lapses (past renewal deadline) but qualified TTL not reached.
    now[0] = int(2 * 1e9)  # past lease_ttl
    assert r.tick() == []
    assert r.get_lease(lid).state != TransferState.RELEASED
    # Once qualified TTL also elapses, it is force-reclaimed.
    now[0] = int(101 * 1e9)
    released = r.tick()
    assert lid in released
    assert r.get_lease(lid).state == TransferState.RELEASED


def test_renewal_extends_lease(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_DSA_DIAG_LEVEL", "off")
    import vllm.observability.dsa_offload as mod
    mod._STATE = None
    mod._root_logger._state = None

    now = [0]

    def clk():
        return now[0]

    r = DSARouter(lease_ttl_s=1.0, qualified_ttl_s=100.0, clock_ns=clk)
    lid = r.prefill_dispatch("x-6", 1, _pk("p"))
    now[0] = int(0.5 * 1e9)
    rb = r.renew_lease(lid)
    assert rb is not None
    now[0] = int(0.9 * 1e9)
    # Still within renewed window.
    assert r.tick() == []
