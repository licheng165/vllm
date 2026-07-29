"""Structured observability helpers for vLLM.

Currently hosts the DSA offload diagnostic protocol (``dsa_offload.v1`` single
line JSON events emitted behind the ``[DSA_OFFLOAD]`` marker). See
``dsa_offload`` for the design contract.
"""

from vllm.observability.dsa_offload import (
    DSA_EVENTS,
    DSA_MARKER,
    DSA_SCHEMA,
    DiagLevel,
    DSAOffloadLogger,
    DSAOperationReceipt,
    DSAReceiptBundle,
    LookupResult,
    ParticipantIdentity,
    RequestKey,
    dsa_logger_for,
    get_dsa_diag_level,
)

__all__ = [
    "DSA_EVENTS",
    "DSA_MARKER",
    "DSA_SCHEMA",
    "DiagLevel",
    "DSAOffloadLogger",
    "DSAOperationReceipt",
    "DSAReceiptBundle",
    "LookupResult",
    "ParticipantIdentity",
    "RequestKey",
    "dsa_logger_for",
    "get_dsa_diag_level",
]
