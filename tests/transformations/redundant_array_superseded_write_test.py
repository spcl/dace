# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""RedundantArray must keep a copy that sequences two writes to the same region.

``A`` is written twice in one state: once straight out of a map exit, and once by copying the
transient the same map filled. The copy is the only thing ordering the two -- it lands its value
on ``A`` after the map completed. Folding it away makes both writes direct siblings of one map
exit, and siblings in a scope have no relative order, so codegen may replay the superseded value
last. This is the vadv miscompile (``ccol[i,j,0]`` stored twice, stale store winning) in miniature.
"""
import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.dataflow import RedundantArray

N = 20


def _build_superseded_write_sdfg() -> dace.SDFG:
    """``A[i] = B[i]`` out of a map, then ``A[:] = tmp[:]`` where ``tmp[i] = 2 * B[i]``."""
    sdfg = dace.SDFG('redundant_array_superseded_write')
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    sdfg.add_transient('tmp', [N], dace.float64)

    state = sdfg.add_state('main')
    entry, exit_node = state.add_map('m', dict(i=f'0:{N}'))
    b = state.add_access('B')
    tmp = state.add_access('tmp')
    a = state.add_access('A')

    superseded = state.add_tasklet('superseded', {'inp'}, {'out'}, 'out = inp')
    final = state.add_tasklet('final', {'inp'}, {'out'}, 'out = 2.0 * inp')

    state.add_memlet_path(b, entry, superseded, dst_conn='inp', memlet=dace.Memlet('B[i]'))
    state.add_memlet_path(superseded, exit_node, a, src_conn='out', memlet=dace.Memlet('A[i]'))
    state.add_memlet_path(b, entry, final, dst_conn='inp', memlet=dace.Memlet('B[i]'))
    state.add_memlet_path(final, exit_node, tmp, src_conn='out', memlet=dace.Memlet('tmp[i]'))
    state.add_edge(tmp, None, a, None, dace.Memlet(f'tmp[0:{N}] -> [0:{N}]'))

    sdfg.validate()
    return sdfg


def _sibling_overlapping_writes(sdfg: dace.SDFG):
    """Map exits carrying two non-WCR in-edges that write the same container and may overlap."""
    from dace import subsets

    hits = []
    for sd in sdfg.all_sdfgs_recursive():
        for state in sd.states():
            for node in state.nodes():
                if not isinstance(node, nodes.MapExit):
                    continue
                writes = [e for e in state.in_edges(node) if not e.data.is_empty() and e.data.wcr is None]
                for idx, first in enumerate(writes):
                    for second in writes[idx + 1:]:
                        if first.data.data != second.data.data:
                            continue
                        if subsets.intersects(first.data.subset, second.data.subset) is not False:
                            hits.append((state.label, first.data.data))
    return hits


def test_redundant_array_refuses_superseded_write():
    """The copy carries the ordering, so the match must be refused outright."""
    sdfg = _build_superseded_write_sdfg()
    assert sdfg.apply_transformations_repeated(RedundantArray, validate=False) == 0

    state = sdfg.states()[0]
    assert any(isinstance(n, nodes.AccessNode) and n.data == 'tmp' for n in state.nodes()), \
        'the sequencing copy must survive'
    assert not _sibling_overlapping_writes(sdfg)


def test_simplify_keeps_superseded_write_ordered():
    """Simplify may rewrite freely, but never into two unordered writes of the same region."""
    sdfg = _build_superseded_write_sdfg()
    sdfg.simplify()
    assert not _sibling_overlapping_writes(sdfg)


@pytest.mark.parametrize('simplify', [False, True])
def test_superseded_write_numerics(simplify):
    """The later write wins: ``A`` must come out as ``2 * B``, never as ``B``."""
    sdfg = _build_superseded_write_sdfg()
    if simplify:
        sdfg.simplify()

    b = np.arange(1, N + 1, dtype=np.float64)
    a = np.zeros(N, dtype=np.float64)
    sdfg(A=a, B=b)

    assert np.allclose(a, 2.0 * b)


if __name__ == '__main__':
    test_redundant_array_refuses_superseded_write()
    test_simplify_keeps_superseded_write_ordered()
    test_superseded_write_numerics(False)
    test_superseded_write_numerics(True)
