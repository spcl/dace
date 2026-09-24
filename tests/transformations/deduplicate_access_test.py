# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests for the DeduplicateAccess transformation. """
import dace
import numpy as np
from dace.subsets import Range
from dace.transformation.dataflow import DeduplicateAccess
from dace.transformation.passes.consolidate_edges import ConsolidateEdges
import dace.transformation.helpers as helpers

N = dace.symbol('N')
i = dace.symbol('i')
j = dace.symbol('j')


def test_find_contiguous_subsets():
    subset_list = [
        Range([(i, i, 1), (j, j, 1)]),
        Range([(i, i, 1), (j + 3, j + 3, 1)]),
        Range([(i, i, 1), (j + 1, j + 2, 1)]),
        Range([(i - 2, i - 1, 1), (j, j + 3, 1)]),
    ]

    result = helpers.find_contiguous_subsets(subset_list)
    assert len(result) == 1
    assert list(result)[0] == Range([(i - 2, i, 1), (j, j + 3, 1)])


def test_find_contiguous_subsets_nonsquare():
    subset_list = [
        Range([(i, i, 1), (j, j, 1)]),
        Range([(i, i, 1), (j + 3, j + 3, 1)]),
        Range([(i, i, 1), (j + 1, j + 2, 1)]),
        Range([(i + 2, i + 2, 1), (j, j, 1)]),
        Range([(i + 2, i + 2, 1), (j + 3, j + 3, 1)]),
        Range([(i + 2, i + 2, 1), (j + 1, j + 2, 1)]),
        Range([(i + 1, i + 1, 1), (j - 1, j - 1, 1)]),
        Range([(i + 1, i + 1, 1), (j, j, 1)]),
        Range([(i + 1, i + 1, 1), (j + 1, j + 1, 1)]),
    ]

    # Prioritize on first dimension
    result2 = helpers.find_contiguous_subsets(subset_list, 0)
    result2 = helpers.find_contiguous_subsets(result2, None)
    assert len(result2) == 2

    # Prioritize on second dimension
    result3 = helpers.find_contiguous_subsets(subset_list, 1)
    assert len(result3) == 3
    result3 = helpers.find_contiguous_subsets(result3, None)
    assert len(result3) == 3


def test_dedup_access_simple():
    """
    Simple duplicate access.
    """

    @dace.program
    def datest(A: dace.float64[N, N], B: dace.float64[N, N]):
        for i, j in dace.map[0:N, 0:N]:
            tmp1 = np.ndarray([1], dace.float64)
            tmp2 = np.ndarray([1], dace.float64)
            with dace.tasklet:
                a << A[i, j]
                b >> tmp1
                b = 0.2 * a
            with dace.tasklet:
                a << A[i, j]
                b >> tmp2
                b = 0.2 * a
            with dace.tasklet:
                inp1 << tmp1
                inp2 << tmp2
                out >> B[i, j]
                out = inp1 + inp2

    sdfg: dace.SDFG = datest.to_sdfg(simplify=True)
    nodes_before = sdfg.node(0).number_of_nodes()
    assert sdfg.apply_transformations(DeduplicateAccess) == 1
    nodes_after = sdfg.node(0).number_of_nodes()
    # The transformation should add exactly one node
    assert nodes_after == nodes_before + 1


def test_dedup_access_plus():
    """
    A test where there is no gain by applying DeduplicateAccess, and so it
    should not be applied.
    """

    @dace.program
    def datest(A: dace.float64[N, N], B: dace.float64[N, N]):
        for i, j in dace.map[1:N - 1, 1:N - 1]:
            tmp = np.ndarray([5], dace.float64)
            with dace.tasklet:
                a << A[i, j + 1]
                b >> tmp[0]
                b = 0.2 * a
            with dace.tasklet:
                a << A[i, j - 1]
                b >> tmp[1]
                b = 0.2 * a
            with dace.tasklet:
                a << A[i, j]
                b >> tmp[2]
                b = 0.2 * a
            with dace.tasklet:
                a << A[i - 1, j]
                b >> tmp[3]
                b = 0.2 * a
            with dace.tasklet:
                a << A[i + 1, j]
                b >> tmp[4]
                b = 0.2 * a
            with dace.tasklet:
                inp << tmp
                out >> B[i, j]
                out = inp[0] + inp[1] + inp[2] + inp[3] + inp[4]

    sdfg: dace.SDFG = datest.to_sdfg(simplify=True)
    ConsolidateEdges().apply_pass(sdfg, {})
    assert sdfg.apply_transformations(DeduplicateAccess) == 1


def test_dedup_access_square():
    """
    A test where a one square load can be performed once.
    """

    @dace.program
    def datest(A: dace.float64[N, N], B: dace.float64[N, N]):
        for i, j in dace.map[3:N - 3, 3:N - 3]:
            tmp = np.ndarray([3], dace.float64)
            with dace.tasklet:
                a << A[i - 1:i + 2, j - 1:j + 2]
                b >> tmp[0]
                b = (a[0, 0] + a[0, 1] + a[0, 2] + a[1, 0] + a[1, 1] + a[1, 2] + a[2, 0] + a[2, 1] + a[2, 2]) / 9.0
            with dace.tasklet:
                a << A[i - 1:i + 2, j:j + 3]
                b >> tmp[1]
                b = (a[0, 0] + a[0, 1] + a[0, 2] + a[1, 0] + a[1, 1] + a[1, 2] + a[2, 0] + a[2, 1] + a[2, 2]) / 9.0
            with dace.tasklet:
                a << A[i - 1:i + 2, j + 1:j + 4]
                b >> tmp[2]
                b = (a[0, 0] + a[0, 1] + a[0, 2] + a[1, 0] + a[1, 1] + a[1, 2] + a[2, 0] + a[2, 1] + a[2, 2]) / 9.0
            with dace.tasklet:
                inp << tmp
                out >> B[i, j]
                out = (inp[0] + inp[1] + inp[2]) / 12.0

    sdfg: dace.SDFG = datest.to_sdfg(simplify=True)
    ConsolidateEdges().apply_pass(sdfg, {})

    nodes_before = sdfg.node(0).number_of_nodes()
    assert sdfg.apply_transformations(DeduplicateAccess) == 1
    # Check that the subset is contiguous by checking how many nodes are added
    nodes_after = sdfg.node(0).number_of_nodes()
    assert nodes_after == nodes_before + 1


def test_dedup_access_contiguous():
    """
    A test where there is a non-square shape that, based on whether contiguity
    is prioritized, might give different results.
    Subset is:
           j
     6543 21012 3456
          _____
     ____|     |____  2
    |    |     |    | 1
    |    |     |    | 0 i
    |____|     |____| 1
         |_____|     2
    A square of size 5x5, with two 4x3 squares on each side
    """

    @dace.program
    def datest(A: dace.float64[N, N], B: dace.float64[N, N]):
        for i, j in dace.map[6:N - 6, 6:N - 6]:
            tmp = np.ndarray([3], dace.float64)
            with dace.tasklet:
                a << A[i - 2:i + 3, j - 2:j + 3]
                b >> tmp[0]
                b = a[2, 2] * 5.0
            with dace.tasklet:
                a << A[i - 1:i + 2, j - 6:j + 5]
                b >> tmp[1]
                b = a[0, 0] * 4.0
            with dace.tasklet:
                a << A[i - 1:i + 2, j - 2:j + 7]
                b >> tmp[2]
                b = a[0, 0] * 3.0
            with dace.tasklet:
                inp << tmp
                out >> B[i, j]
                out = (inp[0] + inp[1] + inp[2]) / 3.0

    # j contiguous dimension
    sdfg: dace.SDFG = datest.to_sdfg(simplify=True)
    ConsolidateEdges().apply_pass(sdfg, {})

    nodes_before = sdfg.node(0).number_of_nodes()
    assert sdfg.apply_transformations(DeduplicateAccess) == 1
    nodes_after = sdfg.node(0).number_of_nodes()
    assert nodes_after == nodes_before + 2

    # i contiguous dimension
    sdfg: dace.SDFG = datest.to_sdfg(simplify=True)
    ConsolidateEdges().apply_pass(sdfg, {})

    sdfg.arrays['A'].strides = [1, N]
    nodes_before = sdfg.node(0).number_of_nodes()
    assert sdfg.apply_transformations(DeduplicateAccess) == 1
    nodes_after = sdfg.node(0).number_of_nodes()
    assert nodes_after == nodes_before + 3


LEN = 8


def _window_reader(name, code, offset):
    """A map body reading ``a[i + offset]`` and ``a[i + offset + 1]``, as a nested SDFG.

    Under the nested SDFG contract (see ``dace.sdfg.dealias.integrate_nested_sdfg``) the connector
    is the whole container and the memlets inside are written in its coordinates.
    """
    sdfg = dace.SDFG(name)
    sdfg.add_array('a', [LEN], dace.float64)
    sdfg.add_array('o', [LEN], dace.float64)
    sdfg.add_symbol('i', dace.int64)
    state = sdfg.add_state()
    tasklet = state.add_tasklet('t', {'x', 'y'}, {'z'}, 'z = %s' % code)
    state.add_edge(state.add_read('a'), None, tasklet, 'x', dace.Memlet('a[i + %d]' % offset))
    state.add_edge(state.add_read('a'), None, tasklet, 'y', dace.Memlet('a[i + %d]' % (offset + 1)))
    state.add_edge(tasklet, 'z', state.add_write('o'), None, dace.Memlet('o[i]'))
    return sdfg


def _two_overlapping_readers():
    """One map connector feeding two nested SDFGs that read overlapping windows of ``A``."""
    sdfg = dace.SDFG('dedup_nested')
    for name in ('A', 'B', 'C'):
        sdfg.add_array(name, [LEN], dace.float64)
    state = sdfg.add_state()
    entry, exit_ = state.add_map('m', dict(i='0:%d' % (LEN - 2)))
    entry.add_in_connector('IN_A')
    entry.add_out_connector('OUT_A')
    first = state.add_nested_sdfg(_window_reader('sum', 'x + y', 0), {'a'}, {'o'}, {'i': 'i'})
    second = state.add_nested_sdfg(_window_reader('product', 'x * y', 1), {'a'}, {'o'}, {'i': 'i'})
    state.add_edge(state.add_read('A'), None, entry, 'IN_A', dace.Memlet('A[0:%d]' % LEN))
    state.add_edge(entry, 'OUT_A', first, 'a', dace.Memlet('A[i:i + 2]'))
    state.add_edge(entry, 'OUT_A', second, 'a', dace.Memlet('A[i + 1:i + 3]'))
    state.add_memlet_path(first, exit_, state.add_write('B'), src_conn='o', memlet=dace.Memlet('B[i]'))
    state.add_memlet_path(second, exit_, state.add_write('C'), src_conn='o', memlet=dace.Memlet('C[i]'))
    return sdfg, state, entry, first, second


def test_dedup_access_into_nested_sdfgs():
    """The consumers read the deduplicated transient, so their connectors follow it."""
    A = np.arange(LEN, dtype=np.float64)
    expected_b = A[:LEN - 2] + A[1:LEN - 1]
    expected_c = A[1:LEN - 1] * A[2:LEN]

    sdfg, state, entry, first, second = _two_overlapping_readers()
    sdfg.validate()

    DeduplicateAccess.apply_to(sdfg, map_entry=entry, node1=first, node2=second, verify=False, save=False)

    for node in (first, second):
        edge = next(e for e in state.in_edges(node) if e.dst_conn == 'a')
        assert node.sdfg.arrays['a'].is_equivalent(sdfg.arrays[edge.data.data])
    sdfg.validate()

    B = np.zeros(LEN)
    C = np.zeros(LEN)
    sdfg(A=A, B=B, C=C)
    assert np.allclose(B[:LEN - 2], expected_b)
    assert np.allclose(C[:LEN - 2], expected_c)


if __name__ == '__main__':
    test_find_contiguous_subsets()
    test_find_contiguous_subsets_nonsquare()
    test_dedup_access_simple()
    test_dedup_access_plus()
    test_dedup_access_square()
    test_dedup_access_contiguous()
    test_dedup_access_into_nested_sdfgs()
