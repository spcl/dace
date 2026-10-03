# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A transient the memlets index BY the tile iter-var already carries the lane axis.

:class:`WidenAccesses` decides a lane-dependent transient must become a per-lane ``(W, ...)`` tile
from the DESCRIPTOR alone, and refuses any shape that is not scalar-like. It never looked at how
the transient is indexed, so CloudSC's ``zsolqa`` -- shape ``(nclv, nclv, klon)``, written
``zsolqa[jm, jn, jl]`` with ``jl`` the tile iter-var -- was refused outright:
"lane-dependent transient 'loop_body_zsolqa' has non-scalar shape (5, 5, klon)". An array indexed
by the lane var holds a whole lane DIMENSION, not one value per lane; widening it would add a
second lane axis. The refusal belongs to the buffer it was written for -- a sliding window indexed
by constants -- which still refuses.
"""
import os

os.environ.setdefault('MPI4PY_RC_INITIALIZE', '0')

import pytest

import dace
from dace.transformation.passes.vectorization.widen_accesses import WidenAccesses

ITER_VARS = ('i', )


def build(shape, out_subset: str) -> dace.SDFG:
    """One tasklet reading ``src[i]`` and writing ``buf[out_subset]``, so rule 2 marks ``buf``."""
    sdfg = dace.SDFG('widen_lane_indexed')
    sdfg.add_symbol('i', dace.int64)
    sdfg.add_array('src', [16], dace.float64)
    sdfg.add_transient('buf', shape, dace.float64)
    state = sdfg.add_state('main', is_start_block=True)
    # The code body names the iter-var, which is what makes the tasklet lane-dependent.
    tasklet = state.add_tasklet('write_buf', {'_in'}, {'_out'}, '_out = _in + i')
    state.add_edge(state.add_read('src'), None, tasklet, '_in', dace.Memlet('src[i]'))
    state.add_edge(tasklet, '_out', state.add_write('buf'), None, dace.Memlet(f'buf[{out_subset}]'))
    return sdfg


def test_lane_indexed_transient_is_left_alone():
    """``buf[1, i]`` binds the iter-var in the transient's own dim: no refusal, and no widening."""
    sdfg = build([2, 16], '1, i')

    lane_dep = WidenAccesses(widths=(8, ))._propagate_lane_dep(sdfg, ITER_VARS, set())

    assert 'buf' not in lane_dep, 'a transient that already carries the lane axis must not be widened'


def test_constant_indexed_window_still_refuses():
    """The buffer the refusal was written for -- indexed by constants, one value per lane."""
    sdfg = build([2], '0:2')

    with pytest.raises(NotImplementedError, match="non-scalar shape"):
        WidenAccesses(widths=(8, ))._propagate_lane_dep(sdfg, ITER_VARS, set())


def test_scalar_transient_is_still_widened():
    """The ordinary case must keep flowing: a scalar-like per-lane transient still widens."""
    sdfg = build([1], '0')

    lane_dep = WidenAccesses(widths=(8, ))._propagate_lane_dep(sdfg, ITER_VARS, set())

    assert 'buf' in lane_dep


if __name__ == '__main__':
    test_lane_indexed_transient_is_left_alone()
    test_constant_indexed_window_still_refuses()
    test_scalar_transient_is_still_widened()
