# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Analysis and per-kind lowering of memlet access policies (see :mod:`dace.sdfg.memlet_access_policy`). The pass
that dispatches the lowering of every policy kind is
:class:`~dace.transformation.passes.memlet_access_policy_lowering.LowerMemletAccessPolicies`."""

from dace.transformation.passes.memlet_access_policies.assign_loop_cursors import AssignLoopCursors
from dace.transformation.passes.memlet_access_policies.loop_cursor_lowering import lower_loop_cursors

__all__ = ["AssignLoopCursors", "lower_loop_cursors"]
