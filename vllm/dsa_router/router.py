"""DSA offload Router: the formal fifth control-plane component (§6.5/§15.9).

The Router owns the P/D handoff lifecycle for the DSA offload protocol. It is
the *only* component permitted to emit ``remote.handoff.leased`` /
``remote.handoff.released`` and the ``pd.*`` lifecycle events, because only it
holds the authoritative transfer state machine and lease registry.

The state machine is::

    CREATED -> P_DISPATCHED -> P_READY -> D_DISPATCHED -> D_MATERIALIZED -> RELEASED
                                       (failure) -> QUIESCING --(D ack/expiry)--> RELEASED

Key safety rules enforced here (and reflected in logs):

* ``kv-ready`` is only accepted when its source epoch matches the producer
  attempt the Router persisted at Prefill dispatch (process-incarnation-aware).
* ``remote.handoff.leased`` is emitted only after ``kv-ready`` is accepted; a
  bare ``remote.put.fenced`` (``put_fenced``) must never be relabeled
  ``handoff_leased``.
* On D unreachability the Router stops renewing the lease but waits for the
  *qualified* soft TTL before reclaiming — it does not reclaim immediately.
* ``remote.handoff.released`` is emitted only after the D ack
  (``decode-materialized``/``decode-quiesced``) or a qualified-TTL expiry.

This module is intentionally pure-Python (depends only on the lightweight
``vllm.observability.dsa_offload`` helper) so the state machine and lease TTL
semantics can be unit-tested without NPU or network. An HTTP front-end can wrap
``DSARouter`` to expose the documented endpoints.
"""

from __future__ import annotations

import enum
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from vllm.observability.dsa_offload import (
    DSAOffloadLogger,
    DiagLevel,
    RequestKey,
    dsa_logger_for,
)


class TransferState(enum.IntEnum):
    CREATED = 1
    P_DISPATCHED = 2
    P_READY = 3
    D_DISPATCHED = 4
    D_MATERIALIZED = 5
    QUIESCING = 6
    RELEASED = 7


@dataclass
class LeaseDescriptor:
    lease_descriptor_id: str
    transfer_id: str
    source_epoch: int
    dispatch_epoch: int
    created_ns: int
    renew_by_ns: int
    state: TransferState
    producer_key: Optional[RequestKey] = None
    consumer_key: Optional[RequestKey] = None
    receipt_bundle_id: Optional[str] = None
    decoder: Optional[str] = None
    # Track whether the producer reported ready so dispatch is single-shot.
    ready_seen: bool = False


_DEFAULT_LEASE_TTL_S = 30.0
_DEFAULT_QUALIFIED_TTL_S = 300.0


class DSARouter:
    """Authoritative P/D handoff state machine and lease registry."""

    def __init__(
        self,
        log: DSAOffloadLogger | None = None,
        *,
        lease_ttl_s: float = _DEFAULT_LEASE_TTL_S,
        qualified_ttl_s: float = _DEFAULT_QUALIFIED_TTL_S,
        clock_ns=time.monotonic_ns,
    ) -> None:
        self._log = log or dsa_logger_for("vllm.router")
        self._lease_ttl_s = lease_ttl_s
        self._qualified_ttl_s = qualified_ttl_s
        self._clock_ns = clock_ns
        self._lock = threading.Lock()
        # transfer_id -> lease. A transfer may have multiple dispatch epochs;
        # the latest lease descriptor id is authoritative.
        self._transfers: dict[str, LeaseDescriptor] = {}
        self._lease_by_id: dict[str, LeaseDescriptor] = {}

    # -- Prefill side -------------------------------------------------------
    def prefill_dispatch(
        self,
        transfer_id: str,
        source_epoch: int,
        producer_key: RequestKey,
        trace_id: Optional[str] = None,
    ) -> str:
        """Router -> P: persist the producer attempt/source epoch (CREATED).

        Returns the ``lease_descriptor_id``. This does NOT grant object
        protection yet; that only happens at ``kv_ready`` acceptance.
        """
        now = self._clock_ns()
        lease_id = f"lease-{uuid.uuid4().hex[:12]}"
        lease = LeaseDescriptor(
            lease_descriptor_id=lease_id,
            transfer_id=transfer_id,
            source_epoch=source_epoch,
            dispatch_epoch=0,
            created_ns=now,
            renew_by_ns=now + int(self._lease_ttl_s * 1e9),
            state=TransferState.P_DISPATCHED,
            producer_key=producer_key,
        )
        with self._lock:
            self._transfers[transfer_id] = lease
            self._lease_by_id[lease_id] = lease
        if self._log.enabled(DiagLevel.LIFECYCLE):
            self._log.emit(
                "pd.prefill.dispatch",
                outcome="ok",
                level=DiagLevel.LIFECYCLE,
                trace_id=trace_id,
                transfer_id=transfer_id,
                source_epoch=source_epoch,
                lease_descriptor_id=lease_id,
                request_process_instance=producer_key.process_instance_id,
                scope_id=producer_key.scope_id,
            )
        return lease_id

    def kv_ready(
        self,
        transfer_id: str,
        source_epoch: int,
        producer_key: RequestKey,
        receipt_bundle_id: str,
        lease_descriptor_id: str,
        trace_id: Optional[str] = None,
    ) -> tuple[bool, Optional[LeaseDescriptor]]:
        """P -> Router: producer binding + bundle + lease descriptor.

        Accepted only if the source epoch matches the attempt persisted at
        dispatch (stale/retried attempts are rejected). On acceptance the Router
        takes object-protection responsibility and emits
        ``remote.handoff.leased``.
        """
        with self._lock:
            lease = self._lease_by_id.get(lease_descriptor_id)
            if lease is None or lease.transfer_id != transfer_id:
                self._reject(trace_id, transfer_id, "unknown_lease_descriptor")
                return False, None
            if lease.state not in (TransferState.P_DISPATCHED,
                                   TransferState.P_READY):
                self._reject(trace_id, transfer_id, "unexpected_state")
                return False, None
            if lease.source_epoch != source_epoch:
                # Stale/retried producer attempt: reject, keep prior lease.
                self._reject(trace_id, transfer_id, "stale_source_epoch")
                return False, None
            lease.state = TransferState.P_READY
            lease.receipt_bundle_id = receipt_bundle_id
            lease.ready_seen = True
            lease.renew_by_ns = self._clock_ns() + int(
                self._lease_ttl_s * 1e9)
            accepted = lease
        if self._log.enabled(DiagLevel.LIFECYCLE):
            self._log.emit(
                "remote.handoff.leased",
                outcome="ok",
                level=DiagLevel.LIFECYCLE,
                trace_id=trace_id,
                transfer_id=transfer_id,
                source_epoch=source_epoch,
                lease_descriptor_id=lease_descriptor_id,
                caused_by=receipt_bundle_id,
                remote_guarantee="handoff_leased",
                request_process_instance=producer_key.process_instance_id,
                scope_id=producer_key.scope_id,
            )
            self._log.emit(
                "pd.prefill.ready.accepted",
                outcome="ok",
                level=DiagLevel.LIFECYCLE,
                trace_id=trace_id,
                transfer_id=transfer_id,
                source_epoch=source_epoch,
                lease_descriptor_id=lease_descriptor_id,
                caused_by=receipt_bundle_id,
            )
        return True, accepted

    # -- Dispatch / decode side --------------------------------------------
    def route_dispatch(
        self,
        transfer_id: str,
        decoder: str,
        trace_id: Optional[str] = None,
    ) -> tuple[bool, int]:
        """Router -> D: choose decoder endpoint, assign a new dispatch epoch."""
        with self._lock:
            lease = self._transfers.get(transfer_id)
            if lease is None or not lease.ready_seen:
                self._reject(trace_id, transfer_id, "dispatch_without_ready")
                return False, 0
            lease.dispatch_epoch += 1
            lease.state = TransferState.D_DISPATCHED
            lease.decoder = decoder
            lease.consumer_key = None
            de = lease.dispatch_epoch
        if self._log.enabled(DiagLevel.LIFECYCLE):
            self._log.emit(
                "pd.route.dispatch",
                outcome="ok",
                level=DiagLevel.LIFECYCLE,
                trace_id=trace_id,
                transfer_id=transfer_id,
                source_epoch=lease.source_epoch,
                dispatch_epoch=de,
                decoder=decoder,
            )
        return True, de

    def decode_materialized(
        self,
        transfer_id: str,
        source_epoch: int,
        dispatch_epoch: int,
        consumer_key: RequestKey,
        trace_id: Optional[str] = None,
    ) -> bool:
        """D -> Router: exact participant quorum materialized on NPU."""
        with self._lock:
            lease = self._transfers.get(transfer_id)
            if lease is None:
                return False
            if (lease.source_epoch != source_epoch
                    or lease.dispatch_epoch != dispatch_epoch):
                self._reject(trace_id, transfer_id, "stale_dispatch_epoch")
                return False
            if lease.state != TransferState.D_DISPATCHED:
                # Allow idempotent re-ack.
                if lease.state != TransferState.D_MATERIALIZED:
                    self._reject(trace_id, transfer_id, "unexpected_state")
                    return False
            lease.state = TransferState.D_MATERIALIZED
            lease.consumer_key = consumer_key
        if self._log.enabled(DiagLevel.LIFECYCLE):
            self._log.emit(
                "pd.decode.materialized",
                outcome="ok",
                level=DiagLevel.LIFECYCLE,
                trace_id=trace_id,
                transfer_id=transfer_id,
                source_epoch=source_epoch,
                dispatch_epoch=dispatch_epoch,
                request_process_instance=consumer_key.process_instance_id,
                scope_id=consumer_key.scope_id,
                participant_quorum="proven",
            )
        # On successful materialization the handoff object is no longer needed;
        # release the lease.
        self.release_lease(lease.lease_descriptor_id,
                           caused_by="decode_materialized", trace_id=trace_id)
        return True

    def decode_quiesced(
        self,
        transfer_id: str,
        dispatch_epoch: int,
        trace_id: Optional[str] = None,
    ) -> bool:
        """D -> Router: quiesce ack for a failed/expired dispatch."""
        with self._lock:
            lease = self._transfers.get(transfer_id)
            if lease is None or lease.dispatch_epoch != dispatch_epoch:
                return False
            if lease.state == TransferState.RELEASED:
                return True
            lease.state = TransferState.QUIESCING
        return self.release_lease(lease.lease_descriptor_id,
                                  caused_by="decode_quiesced",
                                  trace_id=trace_id)

    def transfer_failed(
        self,
        transfer_id: str,
        side: str,
        trace_id: Optional[str] = None,
    ) -> None:
        """P/D -> Router: failure. Enter QUIESCING; do not release until D ack
        or qualified TTL (§6.5)."""
        with self._lock:
            lease = self._transfers.get(transfer_id)
            if lease is None:
                return
            if lease.state in (TransferState.RELEASED,
                               TransferState.QUIESCING):
                return
            lease.state = TransferState.QUIESCING
        if self._log.enabled(DiagLevel.LIFECYCLE):
            self._log.emit(
                "pd.decode.quiesce.command",
                outcome="degraded",
                level=DiagLevel.LIFECYCLE,
                trace_id=trace_id,
                transfer_id=transfer_id,
                source_epoch=lease.source_epoch,
                dispatch_epoch=lease.dispatch_epoch,
                reason=f"{side}_transfer_failed",
            )

    # -- Lease lifecycle ----------------------------------------------------
    def renew_lease(
        self, lease_descriptor_id: str, trace_id: Optional[str] = None
    ) -> Optional[int]:
        with self._lock:
            lease = self._lease_by_id.get(lease_descriptor_id)
            if lease is None or lease.state == TransferState.RELEASED:
                return None
            lease.renew_by_ns = self._clock_ns() + int(
                self._lease_ttl_s * 1e9)
            renew_by = lease.renew_by_ns
        if self._log.enabled(DiagLevel.LIFECYCLE):
            self._log.emit(
                "remote.handoff.renewed",
                outcome="ok",
                level=DiagLevel.LIFECYCLE,
                trace_id=trace_id,
                transfer_id=lease.transfer_id,
                lease_descriptor_id=lease_descriptor_id,
            )
        return renew_by

    def release_lease(
        self,
        lease_descriptor_id: str,
        *,
        caused_by: str = "ack",
        trace_id: Optional[str] = None,
    ) -> bool:
        with self._lock:
            lease = self._lease_by_id.get(lease_descriptor_id)
            if lease is None or lease.state == TransferState.RELEASED:
                return False
            lease.state = TransferState.RELEASED
        if self._log.enabled(DiagLevel.LIFECYCLE):
            self._log.emit(
                "remote.handoff.released",
                outcome="ok",
                level=DiagLevel.LIFECYCLE,
                trace_id=trace_id,
                transfer_id=lease.transfer_id,
                source_epoch=lease.source_epoch,
                dispatch_epoch=lease.dispatch_epoch,
                lease_descriptor_id=lease_descriptor_id,
                caused_by=caused_by,
            )
        return True

    # -- Qualified TTL sweep ------------------------------------------------
    def tick(self, now_ns: Optional[int] = None) -> list[str]:
        """Advance lease expiry. A lease is only reclaimed after the *qualified*
        soft TTL has elapsed since creation, even if renewal stopped — never
        immediately on renewal lapse (§6.5 D-unreachable rule). Returns the
        lease ids that were force-released.
        """
        now = now_ns if now_ns is not None else self._clock_ns()
        qualified_deadline_offset = int(self._qualified_ttl_s * 1e9)
        released: list[str] = []
        with self._lock:
            for lease in list(self._lease_by_id.values()):
                if lease.state == TransferState.RELEASED:
                    continue
                # Renewal keeps it alive; once renewal lapses we still wait for
                # the qualified TTL before reclaiming.
                past_renewal = now > lease.renew_by_ns
                past_qualified = (now - lease.created_ns) > qualified_deadline_offset
                if past_renewal and past_qualified:
                    lease.state = TransferState.RELEASED
                    released.append(lease.lease_descriptor_id)
        for lid in released:
            self.release_lease(lid, caused_by="qualified_ttl_expired")
        return released

    # -- helpers ------------------------------------------------------------
    def get_lease(self, lease_descriptor_id: str) -> Optional[LeaseDescriptor]:
        with self._lock:
            return self._lease_by_id.get(lease_descriptor_id)

    def _reject(self, trace_id, transfer_id, reason) -> None:
        if self._log.enabled(DiagLevel.LIFECYCLE):
            self._log.emit(
                "invariant.violation",
                outcome="error",
                level=DiagLevel.LIFECYCLE,
                trace_id=trace_id,
                transfer_id=transfer_id,
                reason=reason,
            )


__all__ = ["DSARouter", "LeaseDescriptor", "TransferState"]
