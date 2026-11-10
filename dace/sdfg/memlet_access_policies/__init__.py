# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Memlet access policy kinds (see :mod:`dace.sdfg.memlet_access_policy` for the base class and the protocol)."""

from dace.sdfg.memlet_access_policies.copy_on_access import CopyOnAccess
from dace.sdfg.memlet_access_policies.loop_cursor import LoopCursor

__all__ = ["CopyOnAccess", "LoopCursor"]
