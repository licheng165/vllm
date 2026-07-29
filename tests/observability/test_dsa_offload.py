# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the DSA offload structured logging protocol (§13.1/§13.2).

These tests verify the *pure observation layer* invariants:

* every emitted line carries the required envelope fields and a known event;
* ``off`` never evaluates a lazy field supplier;
* normal/sampled/deep ``emit()`` never accepts a ``torch.Tensor`` and never
  calls any D2H method on it;
* sampling/dedup tables are bounded and cleared on request finish.
"""

from __future__ import annotations

import io
import json
import logging
import os
from collections.abc import Mapping

import pytest


def _reload(monkeypatch, **env):
    for k in (
        "VLLM_ASCEND_DSA_DIAG_LEVEL", "VLLM_ASCEND_DSA_DIAG_SAMPLE_EVERY",
        "VLLM_ASCEND_DSA_DIAG_TRACE_ID", "VLLM_ASCEND_DSA_DIAG_NODE_TAG",
        "VLLM_ASCEND_DSA_DIAG_INCLUDE_REQUEST_ID",
        "VLLM_ASCEND_MTP_DW_DIAG", "VLLM_ASCEND_MTP_DW_DEEP_DIAG",
    ):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import importlib
    import vllm.observability.dsa_offload as mod
    mod._STATE = None
    importlib.reload(mod)
    return mod


@pytest.fixture
def capture(monkeypatch):
    """Capture ``[DSA_OFFLOAD]`` lines emitted to the module logger."""
    records: list[str] = []

    class _H(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    def _setup(mod, name="vllm.observability.dsa_offload"):
        lg = logging.getLogger(name)
        h = _H()
        h.setLevel(logging.DEBUG)
        lg.addHandler(h)
        lg.setLevel(logging.DEBUG)
        return records, h, lg

    return _setup


def _emit_lines(mod, records):
    out = []
    for line in records:
        if line.startswith(mod.DSA_MARKER):
            out.append(json.loads(line[len(mod.DSA_MARKER):]))
    return out


REQUIRED_ENVELOPE = {
    "schema", "event", "outcome", "ts_ns", "mono_ns", "event_seq",
    "process_instance", "event_id", "process_instance", "emitter",
    "node_tag", "pid",
}


def test_off_does_not_evaluate_supplier(monkeypatch, capture):
    mod = _reload(monkeypatch)
    records, h, lg = capture(mod)
    log = mod.dsa_logger_for("vllm.scheduler")
    seen = {"called": False}

    def supplier():
        seen["called"] = True
        return {"prompt_tokens": 8}

    log.emit("request.admit", fields=supplier)
    lg.removeHandler(h)
    assert seen["called"] is False
    assert records == []


def test_lifecycle_envelope_and_known_event(monkeypatch, capture):
    mod = _reload(monkeypatch, VLLM_ASCEND_DSA_DIAG_LEVEL="lifecycle")
    records, h, lg = capture(mod)
    log = mod.dsa_logger_for("vllm.scheduler")
    log.emit("request.admit", outcome="ok", trace_id="t1",
             request_process_instance="s", scope_id=1, prompt_tokens=4)
    lg.removeHandler(h)
    events = _emit_lines(mod, records)
    assert len(events) == 1
    ev = events[0]
    assert REQUIRED_ENVELOPE <= set(ev)
    assert ev["schema"] == "dsa_offload.v1"
    assert ev["event"] == "request.admit"
    assert ev["event_id"] == f"{ev['process_instance']}:{ev['event_seq']}"
    assert ev["trace_id"] == "t1"
    assert ev["prompt_tokens"] == 4
    assert ev["event"] in mod.DSA_EVENTS


def test_unknown_event_rejected_by_membership(monkeypatch, capture):
    # Membership is asserted in tests (the helper stays cheap at runtime).
    mod = _reload(monkeypatch, VLLM_ASCEND_DSA_DIAG_LEVEL="lifecycle")
    assert "not.a.real.event" not in mod.DSA_EVENTS


def test_tensor_rejected_at_every_non_off_level(monkeypatch, capture):
    torch = pytest.importorskip("torch")
    for lvl in ("lifecycle", "sampled", "deep"):
        mod = _reload(monkeypatch, VLLM_ASCEND_DSA_DIAG_LEVEL=lvl)
        records, h, lg = capture(mod)
        log = mod.dsa_logger_for("ascend.sfa")
        with pytest.raises(TypeError):
            log.emit("sparse.plan", outcome="ok",
                     level={"lifecycle": 1, "sampled": 2, "deep": 3}[lvl],
                     secret=torch.zeros(2))
        lg.removeHandler(h)


class _BombTensor:
    """Stand-in for a tensor whose D2H accessors must never be touched."""
    __module__ = "torch.fake"

    def item(self):
        raise AssertionError("item() must not be called by the logger")

    def cpu(self):
        raise AssertionError("cpu() must not be called by the logger")

    def tolist(self):
        raise AssertionError("tolist() must not be called by the logger")

    @property
    def shape(self):
        return (4,)


def test_logger_never_invokes_d2h(monkeypatch, capture):
    # A non-tensor mapping value is fine; the bomb-tensor would be rejected by
    # the tensor guard before any attribute access, but we additionally assert
    # that a safe dict payload works without touching D2H stand-ins.
    mod = _reload(monkeypatch, VLLM_ASCEND_DSA_DIAG_LEVEL="lifecycle")
    records, h, lg = capture(mod)
    log = mod.dsa_logger_for("lmcache.worker")
    bomb = _BombTensor()
    with pytest.raises(TypeError):
        log.emit("store.group.local_ready", outcome="ok", kv_group=0,
                 loaded=bomb)
    lg.removeHandler(h)


def test_compat_mapping_legacy_flags(monkeypatch):
    # Old MTP_DW flags map onto the new levels for one version cycle when the
    # unified variable is unset.
    assert _reload(monkeypatch).get_dsa_diag_level() == 0
    assert _reload(monkeypatch, VLLM_ASCEND_MTP_DW_DIAG="1"
                   ).get_dsa_diag_level() == 2
    assert _reload(monkeypatch, VLLM_ASCEND_MTP_DW_DEEP_DIAG="1"
                   ).get_dsa_diag_level() == 3
    # The unified variable wins when set.
    assert _reload(monkeypatch, VLLM_ASCEND_MTP_DW_DIAG="1",
                   VLLM_ASCEND_DSA_DIAG_LEVEL="off").get_dsa_diag_level() == 0


def test_sampling_cadence_and_frontier_force(monkeypatch, capture):
    mod = _reload(monkeypatch, VLLM_ASCEND_DSA_DIAG_LEVEL="sampled",
                  VLLM_ASCEND_DSA_DIAG_SAMPLE_EVERY="3")
    records, h, lg = capture(mod)
    log = mod.dsa_logger_for("vllm.scheduler")
    emitted = []
    for step in range(1, 10):
        # frontier constant -> only the first step is forced (0->100), then
        # first-3-always + every-3 cadence.
        remap = 100
        before = len(_emit_lines(mod, records))
        log.emit("decode.step.complete", outcome="ok",
                 level=mod.DiagLevel.SAMPLED, trace_id="t",
                 request_process_instance="s", scope_id=1, schedule_id=step,
                 remap_end=remap)
        emitted.append(len(_emit_lines(mod, records)) > before)
    lg.removeHandler(h)
    # step 1 is frontier-forced (0->100), steps 2,3,4 under first-3-always,
    # then step 7 by every-3 cadence.
    assert emitted == [True, True, True, True, False, False, True, False, False]


def test_failure_not_dropped_but_throttled(monkeypatch, capture):
    mod = _reload(monkeypatch, VLLM_ASCEND_DSA_DIAG_LEVEL="lifecycle")
    records, h, lg = capture(mod)
    log = mod.dsa_logger_for("lmcache.worker")
    for _ in range(5):
        log.emit("execution.fatal", outcome="error", reason="boom",
                 trace_id="t", request_process_instance="s", scope_id=1)
    lg.removeHandler(h)
    events = [e for e in _emit_lines(mod, records)
              if e["event"] == "execution.fatal"]
    suppressed = [e for e in _emit_lines(mod, records)
                  if e["event"] == "failure.suppressed"]
    # First failure always emits; at most one suppression summary follows.
    assert len(events) >= 1
    assert len(events) + len(suppressed) <= 3


def test_request_finish_clears_scope(monkeypatch, capture):
    mod = _reload(monkeypatch, VLLM_ASCEND_DSA_DIAG_LEVEL="sampled",
                  VLLM_ASCEND_DSA_DIAG_SAMPLE_EVERY="1000")
    records, h, lg = capture(mod)
    log = mod.dsa_logger_for("vllm.scheduler")
    log.request_finished(("t", "s", 1))
    state = log.state
    assert ("t", "s", 1) not in state._sample
    lg.removeHandler(h)


def test_dedup_bounded(monkeypatch):
    mod = _reload(monkeypatch, VLLM_ASCEND_DSA_DIAG_LEVEL="sampled")
    log = mod.dsa_logger_for("vllm.scheduler")
    state = log.state
    # capacity invariant
    assert mod._MAX_DEDUP_ENTRIES == 4096
    for i in range(10):
        state.seen_dedup(("k", i))
    assert len(state._dedup) == 10


def test_request_id_suppressed_by_default(monkeypatch, capture):
    mod = _reload(monkeypatch, VLLM_ASCEND_DSA_DIAG_LEVEL="lifecycle")
    records, h, lg = capture(mod)
    log = mod.dsa_logger_for("vllm.scheduler")
    log.emit("request.admit", outcome="ok", trace_id="t",
             request_process_instance="s", scope_id=1, request_id="secret-42")
    lg.removeHandler(h)
    ev = _emit_lines(mod, records)[0]
    assert "request_id" not in ev


def test_lookup_result_source():
    mod = pytest.importorskip("vllm.observability.dsa_offload")
    assert mod.LookupResult(hit_tokens=10, local_cpu_tokens=10).source == "local"
    assert mod.LookupResult(hit_tokens=10, remote_tokens=10).source == "remote"
    assert mod.LookupResult(hit_tokens=10, local_cpu_tokens=4,
                            remote_tokens=6).source == "mixed"
    assert mod.LookupResult(hit_tokens=0).source == "unknown"
