# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for views that cross a nested-SDFG boundary in ``tree_to_sdfg``.

A view is an aliasing relationship between two containers, and its subset
generally names the enclosing map parameters. Two things follow, and both used
to be wrong:

- a view bound inside a map body must stay INSIDE it, rebuilt from its source,
  rather than crossing the boundary as a connector -- outside the map, the
  parameters its subset names do not exist;
- a view that DID cross a boundary was already bound on the other side, so
  binding it again inside would import its own source as a second connector,
  and the same memory would reach the body twice under two names.

The symptom of the second was a memlet with the view's offset applied twice
(``a[__i0 + 20]`` where the source says ``a[10:15]``), which also came out
differently from run to run: the two aliasing paths were unioned in whichever
order the propagation happened to visit them.
"""
import numpy as np

import dace
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.analysis.schedule_tree.sdfg_to_tree import as_schedule_tree


def test_view_argument_of_a_nested_program_stays_in_the_map_body():
    """``nester(A[:, i])`` inside a map: the view's subset names ``i``, so the
    binding cannot be hoisted to where the map is only entered."""

    @dace.program
    def nester(A: dace.float64[20]):
        A[:] = 12

    @dace.program
    def tester(A: dace.float64[20, 20]):
        for i in dace.map[0:20]:
            nester(A[:, i])

    sdfg = tester.to_sdfg(simplify=False)
    # The view is body-local, so it is not a container of the top-level SDFG's
    # dataflow: no viewing edge may be attached outside the map.
    for state in sdfg.states():
        for edge in state.edges():
            if edge.src_conn == 'views' or edge.dst_conn == 'views':
                assert 'i' not in {str(symbol) for symbol in edge.data.subset.free_symbols}

    A = np.random.rand(20, 20)
    tester(A)
    assert np.allclose(A, 12.0)


def test_a_view_reaching_a_nested_program_is_bound_once():
    """Two levels of nesting, each passing a view: the offsets of the four
    blocks must come out exactly once each, not compounded."""

    @dace.program
    def inner(a: dace.float64[5]):
        a += 1

    @dace.program
    def middle(a: dace.float64[5, 10]):
        for i in range(10):
            inner(a[:, i])

    @dace.program
    def outer(a: dace.float64[20, 10]):
        middle(a[0:5])
        middle(a[5:10])
        middle(a[10:15])
        middle(a[15:20])

    sdfg = outer.to_sdfg(simplify=True)
    tree = as_schedule_tree(sdfg)

    tasklets = [node for node in tree.preorder_traversal() if isinstance(node, tn.TaskletNode)]
    assert len(tasklets) == 4
    offsets = []
    for tasklet in tasklets:
        subsets = {str(memlet.subset) for memlet in tasklet.in_memlets.values()}
        subsets |= {str(memlet.subset) for memlet in tasklet.out_memlets.values()}
        # A tasklet reads and writes the same element; a second aliasing path
        # would make the two disagree.
        assert len(subsets) == 1, f'read and write disagree: {sorted(subsets)}'
        offsets.append(subsets.pop())
    assert sorted(offsets) == sorted(['__i0, i', '__i0 + 5, i', '__i0 + 10, i', '__i0 + 15, i'])

    a = np.random.rand(20, 10)
    expected = a + 1
    outer(a)
    assert np.allclose(a, expected)


def test_the_reconstructed_tree_is_deterministic():
    """The duplicate aliasing path made propagation order observable."""

    @dace.program
    def inner(a: dace.float64[5]):
        a += 1

    @dace.program
    def middle(a: dace.float64[5, 10]):
        for i in range(10):
            inner(a[:, i])

    @dace.program
    def outer(a: dace.float64[20, 10]):
        middle(a[0:5])
        middle(a[10:15])

    trees = {as_schedule_tree(outer.to_sdfg(simplify=True)).as_string() for _ in range(3)}
    assert len(trees) == 1, f'schedule tree differs between runs: {trees}'


if __name__ == '__main__':
    test_view_argument_of_a_nested_program_stays_in_the_map_body()
    test_a_view_reaching_a_nested_program_is_bound_once()
    test_the_reconstructed_tree_is_deterministic()
