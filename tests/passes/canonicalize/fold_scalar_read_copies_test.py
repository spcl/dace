# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A scalar read copy folds into its tasklet exactly when no write can land between the copy and the read."""
import numpy as np

import dace
from dace import data
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize.fold_scalar_read_copies import FoldScalarReadCopies

N = dace.symbol('N')


def scalar_copies(sdfg: dace.SDFG) -> int:
    return sum(
        1 for node, state in sdfg.all_nodes_recursive()
        if isinstance(node, nodes.AccessNode) and isinstance(state.sdfg.arrays[node.data], data.Scalar) and state.sdfg.
        arrays[node.data].transient and state.in_degree(node) == 1 and state.in_edges(node)[0].data.data != node.data)


def folded(program, **arrays):
    sdfg = program.to_sdfg(simplify=True)
    before = scalar_copies(sdfg)
    FoldScalarReadCopies().apply_pass(sdfg, {})
    sdfg.validate()
    sdfg(**arrays)
    return before, scalar_copies(sdfg)


@dace.program
def doubled(A: dace.float64[N], out: dace.float64[N]):
    for i in range(N):
        out[i] = A[i] * 2.0


@dace.program
def rowsum(B: dace.float64[N, N], out: dace.float64[N]):
    for k in range(N):
        for i in range(N):
            out[i] = out[i] + B[k, i]


@dace.program
def recurrence(a: dace.float64[N], b: dace.float64[N]):
    for i in range(N - 1):
        a[i + 1] = a[i] + b[i]


@dace.program
def written_then_read(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = a[i] * 2.0
        c[i] = b[i] + 1.0


def test_a_read_only_copy_folds():
    A, out = np.random.rand(8), np.zeros(8)
    before, after = folded(doubled, A=A, out=out, N=8)
    assert before > after
    assert np.allclose(out, 2 * A)


def test_a_same_element_update_folds():
    B, out = np.random.rand(6, 6), np.zeros(6)
    before, after = folded(rowsum, B=B, out=out, N=6)
    assert before > after
    assert np.allclose(out, B.sum(axis=0))


def test_a_recurrence_whose_write_follows_the_read_folds():
    a, b = np.random.rand(8), np.random.rand(8)
    expected = a.copy()
    for i in range(7):
        expected[i + 1] = expected[i] + b[i]
    before, after = folded(recurrence, a=a, b=b, N=8)
    assert before > after
    assert np.allclose(a, expected)


def test_a_copy_out_of_the_node_its_writer_fills_folds():
    a, b, c = np.random.rand(8), np.zeros(8), np.zeros(8)
    before, after = folded(written_then_read, a=a, b=b, c=c, N=8)
    assert before > 0 and after == 0
    assert np.allclose(b, 2 * a) and np.allclose(c, 2 * a + 1)


def copy_beside_a_writer(ordered: bool) -> dace.SDFG:
    """``A[0] -> s -> t1 -> B[0]`` beside ``t2 -> A[0]``; with ``ordered`` the write follows ``t1``."""
    sdfg = dace.SDFG('copy_beside_a_writer' + ('_ordered' if ordered else ''))
    sdfg.add_array('A', [2], dace.float64)
    sdfg.add_array('B', [1], dace.float64)
    sdfg.add_scalar('s', dace.float64, transient=True)
    state = sdfg.add_state()
    s = state.add_access('s')
    state.add_edge(state.add_read('A'), None, s, None, dace.Memlet('A[0]'))
    t1 = state.add_tasklet('use', {'x'}, {'y'}, 'y = x')
    state.add_edge(s, None, t1, 'x', dace.Memlet('s[0]'))
    b_node = state.add_write('B')
    state.add_edge(t1, 'y', b_node, None, dace.Memlet('B[0]'))
    t2 = state.add_tasklet('clobber', {}, {'z'}, 'z = 7.0')
    state.add_edge(t2, 'z', state.add_write('A'), None, dace.Memlet('A[0]'))
    if ordered:
        state.add_nedge(b_node, t2, dace.Memlet())
    return sdfg


def test_a_copy_beside_an_unordered_writer_stays():
    sdfg = copy_beside_a_writer(ordered=False)
    assert FoldScalarReadCopies().apply_pass(sdfg, {}) is None


def test_a_copy_whose_writer_follows_the_read_folds():
    sdfg = copy_beside_a_writer(ordered=True)
    assert FoldScalarReadCopies().apply_pass(sdfg, {}) == 1
    sdfg.validate()


def test_a_scalar_read_again_in_another_state_stays():
    sdfg = copy_beside_a_writer(ordered=True)
    later = sdfg.add_state_after(sdfg.start_state)
    t = later.add_tasklet('again', {'x'}, {'y'}, 'y = x')
    later.add_edge(later.add_read('s'), None, t, 'x', dace.Memlet('s[0]'))
    later.add_edge(t, 'y', later.add_write('B'), None, dace.Memlet('B[0]'))
    assert FoldScalarReadCopies().apply_pass(sdfg, {}) is None


def result_copy_beside_a_reader(ordered: bool) -> dace.SDFG:
    """``t -> r -> A[0]`` beside ``A[0] -> peek -> B[0]``; with ``ordered`` the reader runs first."""
    sdfg = dace.SDFG('result_copy_beside_a_reader' + ('_ordered' if ordered else ''))
    sdfg.add_array('A', [1], dace.float64)
    sdfg.add_array('B', [1], dace.float64)
    sdfg.add_scalar('r', dace.float64, transient=True)
    state = sdfg.add_state()
    produce = state.add_tasklet('produce', {}, {'o'}, 'o = 3.0')
    r = state.add_access('r')
    state.add_edge(produce, 'o', r, None, dace.Memlet('r[0]'))
    state.add_edge(r, None, state.add_write('A'), None, dace.Memlet('r[0] -> [0]'))
    peek = state.add_tasklet('peek', {'x'}, {'y'}, 'y = x')
    state.add_edge(state.add_read('A'), None, peek, 'x', dace.Memlet('A[0]'))
    b_node = state.add_write('B')
    state.add_edge(peek, 'y', b_node, None, dace.Memlet('B[0]'))
    if ordered:
        state.add_nedge(b_node, produce, dace.Memlet())
    return sdfg


def test_a_result_copy_beside_an_unordered_reader_stays():
    sdfg = result_copy_beside_a_reader(ordered=False)
    assert FoldScalarReadCopies().apply_pass(sdfg, {}) is None


def test_a_result_copy_whose_reader_runs_first_folds():
    sdfg = result_copy_beside_a_reader(ordered=True)
    assert FoldScalarReadCopies().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    assert 'r' not in {node.data for node in sdfg.start_state.data_nodes()}


def test_a_read_modify_write_keeps_its_result_scalar():
    """``t(A[0]) -> r -> A[0]``: the result copy of a tasklet that reads the array it writes stays, the staged
    read-modify-write the WCR passes match; only the read copy folds."""
    sdfg = result_copy_beside_a_reader(ordered=False)
    state = sdfg.start_state
    produce = next(n for n in state.nodes() if isinstance(n, nodes.Tasklet) and n.label == 'produce')
    produce.add_in_connector('a')
    produce.code = dace.properties.CodeBlock('o = a + 3.0')
    state.add_edge(state.add_read('A'), None, produce, 'a', dace.Memlet('A[0]'))
    FoldScalarReadCopies().apply_pass(sdfg, {})
    assert 'r' in {node.data for node in state.data_nodes()}


if __name__ == '__main__':
    test_a_read_only_copy_folds()
    test_a_same_element_update_folds()
    test_a_recurrence_whose_write_follows_the_read_folds()
    test_a_copy_out_of_the_node_its_writer_fills_folds()
    test_a_copy_beside_an_unordered_writer_stays()
    test_a_copy_whose_writer_follows_the_read_folds()
    test_a_scalar_read_again_in_another_state_stays()
    test_a_result_copy_beside_an_unordered_reader_stays()
    test_a_result_copy_whose_reader_runs_first_folds()
    test_a_read_modify_write_keeps_its_result_scalar()
