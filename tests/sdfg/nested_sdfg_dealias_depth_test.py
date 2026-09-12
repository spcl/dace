# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Dealiasing chains of nested SDFGs more than one level deep.

A connector either describes the container it is connected to as a whole -- what the nested SDFG
contract asks for, where the memlet on the edge only states which part is touched -- or a window of
it, the older shape where the memlets inside are written in the window's coordinates. Both may
appear at any level of a chain, and dealiasing has to bring the innermost accesses into the top
container's coordinates either way.
"""

import itertools

import numpy as np
import pytest

import dace
from dace import symbolic
from dace.sdfg import SDFG, dealias, nodes

N = 16
STARTS = (2, 3, 1, 2)  # Where each window starts, in its parent container's coordinates
LENS = (14, 11, 9, 7)  # How long each window is
WRITES = 2  # How many elements the innermost SDFG writes
TARGET = 8  # The index in the top container where it writes them


def _build(kinds, symbolic_top: bool = False):
    """
    Builds a chain of nested SDFGs, one per entry of ``kinds`` ('whole' or 'window').

    :param kinds: What each level's connector describes, outermost first.
    :param symbolic_top: If True, the outermost window slides with the parameter of an enclosing map.
    :return: The SDFG and the subset the innermost write should end up with.
    """
    top = SDFG('depth%d_%s' % (len(kinds), ''.join(k[0] for k in kinds)))
    top.add_array('A', [N], dace.float64)
    state = top.add_state()

    # What each level's connector covers, and where its coordinates begin in the top container
    shapes, offsets, parent_shape, offset = [], [], N, 0
    for i, kind in enumerate(kinds):
        shapes.append(parent_shape if kind == 'whole' else LENS[i])
        offset += STARTS[i] if kind == 'window' else 0
        offsets.append(offset)
        parent_shape = shapes[-1]

    names = [f'l{i}' for i in range(len(kinds))]

    # The innermost SDFG writes ``WRITES`` elements at ``TARGET``, in its own coordinates
    inner = SDFG('deepest')
    inner.add_array(names[-1], [shapes[-1]], dace.float64)
    istate = inner.add_state()
    entry, exit_ = istate.add_map('w', {'k': f'0:{WRITES}'})
    tasklet = istate.add_tasklet('t', {}, {'o'}, 'o = 100.0 + k')
    istate.add_memlet_path(entry, tasklet, memlet=dace.Memlet())
    coord = TARGET - offsets[-1]
    istate.add_memlet_path(tasklet,
                           exit_,
                           istate.add_write(names[-1]),
                           src_conn='o',
                           memlet=dace.Memlet(f'{names[-1]}[k + {coord}]' if coord else f'{names[-1]}[k]'))

    def window(level: int, parent_offset: int):
        """The part of the parent container the connector of ``level`` is given."""
        if kinds[level] == 'window':
            return STARTS[level], STARTS[level] + LENS[level]
        # A whole-container connector shares the parent's coordinates; the memlet is precise about
        # the part that is touched, which is what makes the offsets worth checking.
        return TARGET - parent_offset, TARGET - parent_offset + WRITES

    child, child_name = inner, names[-1]
    for i in range(len(kinds) - 2, -1, -1):
        outer = SDFG(f'level{i}')
        outer.add_array(names[i], [shapes[i]], dace.float64)
        ostate = outer.add_state()
        node = ostate.add_nested_sdfg(child, {}, {child_name}, {})
        low, high = window(i + 1, offsets[i])
        ostate.add_edge(node, child_name, ostate.add_write(names[i]), None, dace.Memlet(f'{names[i]}[{low}:{high}]'))
        child, child_name = outer, names[i]

    low, high = window(0, 0)
    if symbolic_top:
        node = state.add_nested_sdfg(child, {}, {names[0]}, {'i': dace.symbol('i')})
        entry, exit_ = state.add_map('rows', {'i': '0:3'})
        state.add_nedge(entry, node, dace.Memlet())
        state.add_memlet_path(node,
                              exit_,
                              state.add_write('A'),
                              src_conn=names[0],
                              memlet=dace.Memlet(f'A[i + {low}:i + {high}]'))
        return top, f'k + i + {TARGET}'

    node = state.add_nested_sdfg(child, {}, {names[0]}, {})
    state.add_edge(node, names[0], state.add_write('A'), None, dace.Memlet(f'A[{low}:{high}]'))
    return top, f'k + {TARGET}'


def _innermost_write(sdfg: SDFG):
    """The memlet of the write in the deepest SDFG of the chain."""
    deepest, deepest_depth = sdfg, -1
    for nsdfg in sdfg.all_sdfgs_recursive():
        depth, cur = 0, nsdfg
        while cur.parent_sdfg is not None:
            depth, cur = depth + 1, cur.parent_sdfg
        if depth > deepest_depth:
            deepest, deepest_depth = nsdfg, depth
    for state in deepest.all_states():
        for edge in state.edges():
            if isinstance(edge.src, nodes.Tasklet) and edge.data.data is not None:
                return edge.data
    raise ValueError('No write found in the deepest SDFG')


def _check(kinds, symbolic_top: bool = False):
    sdfg, expected = _build(kinds, symbolic_top)
    dealias.dealias_sdfg_recursive(sdfg)

    memlet = _innermost_write(sdfg)
    assert memlet.data == 'A', f'{"/".join(kinds)}: writes {memlet.data}, not the top container'
    begin = memlet.subset[0][0]
    assert symbolic.simplify(begin - symbolic.pystr_to_symbolic(expected)) == 0, (
        f'{"/".join(kinds)}: writes A[{begin}], expected A[{expected}]')


@pytest.mark.parametrize('kinds', list(itertools.product(['whole', 'window'], repeat=2)))
def test_two_levels(kinds):
    _check(kinds)


@pytest.mark.parametrize('kinds', list(itertools.product(['whole', 'window'], repeat=3)))
def test_three_levels(kinds):
    _check(kinds)


@pytest.mark.parametrize('kinds', list(itertools.product(['whole', 'window'], repeat=2)))
def test_three_levels_under_a_map(kinds):
    """The outermost window slides with a map parameter, so the offsets are symbolic.

    Only a window moves with the map: a whole-container connector keeps the container's coordinates,
    and the part its memlet names has nothing to do with where the writes land, so the outermost
    level is a window here.
    """
    _check(('window', ) + kinds, symbolic_top=True)


def test_three_levels_run():
    """The contract shape, end to end: every connector is the container it is connected to."""
    sdfg, _ = _build(('whole', 'whole', 'whole'))
    sdfg.validate()

    expected = np.zeros(N)
    expected[TARGET:TARGET + WRITES] = [100.0 + k for k in range(WRITES)]

    result = np.zeros(N)
    sdfg(A=result)
    assert np.allclose(result, expected)

    dealias.dealias_sdfg_recursive(sdfg)
    sdfg.validate()
    dealiased = np.zeros(N)
    sdfg(A=dealiased)
    assert np.allclose(dealiased, expected)


if __name__ == '__main__':
    for depth in (2, 3):
        for kindset in itertools.product(['whole', 'window'], repeat=depth):
            _check(kindset)
    for kindset in itertools.product(['whole', 'window'], repeat=2):
        _check(('window', ) + kindset, symbolic_top=True)
    test_three_levels_run()
