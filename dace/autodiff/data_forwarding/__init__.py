# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
Data Forwarding Strategies for Automatic Differentiation.

This package manages the tradeoff between storing intermediate values and
recomputing them during the backward pass. This is a fundamental memory-time
tradeoff in automatic differentiation.
"""

from .manager import DataForwardingManager
from .store import resolve_overwrite_with_store
from .recompute import get_recomputation_nsdfg, resolve_overwrite_with_recomputation

__all__ = [
    "DataForwardingManager",
    "resolve_overwrite_with_store",
    "resolve_overwrite_with_recomputation",
    "get_recomputation_nsdfg",
]
