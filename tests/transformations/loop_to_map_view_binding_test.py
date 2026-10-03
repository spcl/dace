# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``LoopToMap`` and the edge that DEFINES a view.

A view access node is bound to what it looks at by one edge, and that edge moves no data -- it is
the SDFG spelling of ``flat = np.reshape(a, ...)``. Its subset nevertheless spans everything the
view covers, so read as a store it looks like a whole-array write on every iteration and refuses
any loop that touches a view used elsewhere in the program. npbench ``mandelbrot2`` is the shape:
its per-element loops read ``Xiv``/``Yiv``, reshapes of ``Xi``/``Yi``, and were refused with
``write to Xi_0 is not uniquely indexed by the iteration variable``.

The binding is skipped; traffic THROUGH the view is not, and the second test is what holds that
line -- a store into a view still lands on the viewed array and is analysed there.
"""
import numpy as np

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate import LoopToMap

M = dace.symbol('M')


def loop_labels(sdfg: dace.SDFG):
    return [c.label for c in sdfg.all_control_flow_regions(recursive=True) if isinstance(c, LoopRegion)]


def has_map(sdfg: dace.SDFG) -> bool:
    return any(isinstance(n, nodes.MapEntry) for n, _ in sdfg.all_nodes_recursive())


@dace.program
def view_read_in_two_loops(a: dace.float64[M, M], out: dace.float64[M * M], out2: dace.float64[M * M]):
    # Two consumers keep the view alive past either loop, which is what puts its binding edge
    # inside both loop bodies -- a view local to one loop is never examined here.
    flat = np.reshape(a, (M * M, ))
    for i in range(M * M):
        out[i] = flat[i] * 2.0
    for j in range(M * M):
        out2[j] = flat[j] + 1.0


@dace.program
def broadcast_write_through_view(a: dace.float64[M, M], src: dace.float64[M * M], out: dace.float64[M * M]):
    # Every iteration stores to the SAME element through the view: a genuine conflict that the
    # binding skip must not swallow.
    flat = np.reshape(a, (M * M, ))
    for i in range(M * M):
        out[i] = flat[i]
        flat[0] = src[i]


def test_a_view_binding_does_not_block_the_lift():
    """Both loops are plain elementwise reads of a reshape. Structural assert, because a lift that
    fires but leaves the LoopRegion standing is the failure mode a boolean would miss."""
    sdfg = view_read_in_two_loops.to_sdfg(simplify=True)
    sdfg.simplify()
    assert sdfg.apply_transformations_repeated([LoopToMap]) == 2
    assert loop_labels(sdfg) == []
    assert has_map(sdfg)


def test_a_write_through_a_view_still_refuses():
    """The store lands on the viewed array, so the conflict is seen there. Without this the skip
    would be a hole: any non-parallel loop could hide its writes behind a reshape."""
    sdfg = broadcast_write_through_view.to_sdfg(simplify=True)
    sdfg.simplify()
    sdfg.apply_transformations_repeated([LoopToMap])
    assert loop_labels(sdfg) != []


def test_the_lifted_view_reads_keep_their_values():
    rng = np.random.default_rng(0)
    a = rng.random((16, 16))
    ref, ref2 = a.reshape(256) * 2.0, a.reshape(256) + 1.0

    sdfg = view_read_in_two_loops.to_sdfg(simplify=True)
    sdfg.simplify()
    sdfg.apply_transformations_repeated([LoopToMap])
    out, out2 = np.zeros(256), np.zeros(256)
    sdfg(a=a, out=out, out2=out2, M=16)
    assert np.allclose(out, ref)
    assert np.allclose(out2, ref2)


if __name__ == '__main__':
    test_a_view_binding_does_not_block_the_lift()
    test_a_write_through_a_view_still_refuses()
    test_the_lifted_view_reads_keep_their_values()
