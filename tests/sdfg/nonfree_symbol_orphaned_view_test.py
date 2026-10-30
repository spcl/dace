# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``is_nonfree_sym_dependent`` answers for a view with no edges, which has no view edge to inspect."""

import dace
from dace.sdfg import utils as sdutil


def test_orphaned_view_is_not_nonfree_dependent():
    sdfg = dace.SDFG("orphaned_view")
    sdfg.add_view("v", [4], dace.float64)
    state = sdfg.add_state()
    node = state.add_access("v")

    assert sdutil.is_nonfree_sym_dependent(node, sdfg.arrays["v"], state, set()) is False
