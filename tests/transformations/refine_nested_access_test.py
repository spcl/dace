# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests for the RefineNestedAccess transformation. """
import dace
import numpy as np

from dace.transformation.interstate import RefineNestedAccess


def test_refine_dataflow():

    i = dace.symbol('i')
    j = dace.symbol('j')

    @dace.program
    def inner_sdfg(A: dace.int32[5, 5], B: dace.int32[5, 5]):
        B[i, j] = A[j, i]

    sdfg = dace.SDFG('refine_dataflow')
    sdfg.add_array('A', [5, 5], dace.int32)
    sdfg.add_array('B', [5, 5], dace.int32)

    state = sdfg.add_state()
    A = state.add_access('A')
    B = state.add_access('B')
    me, mx = state.add_map('m', dict(i='0:5', j='0:5'))
    nsdfg = state.add_nested_sdfg(inner_sdfg.to_sdfg(simplify=False), sdfg, {'A'}, {'B'}, {'i': 'i', 'j': 'j'})
    state.add_memlet_path(A, me, nsdfg, dst_conn='A', memlet=dace.Memlet.from_array('A', sdfg.arrays['A']))
    state.add_memlet_path(nsdfg, mx, B, src_conn='B', memlet=dace.Memlet.from_array('B', sdfg.arrays['B']))

    num = sdfg.apply_transformations_repeated(RefineNestedAccess)
    assert num == 1

    for edge in state.out_edges(me):
        assert edge.data.subset == dace.subsets.Range([(j, j, 1), (i, i, 1)])
    for edge in state.in_edges(mx):
        assert edge.data.subset == dace.subsets.Range([(i, i, 1), (j, j, 1)])

    A = np.arange(25, dtype=np.int32).reshape(5, 5).copy()
    B = np.empty((5, 5), dtype=np.int32)
    sdfg(A=A, B=B)
    assert np.allclose(B, A.T)


def test_refine_interstate():

    i = dace.symbol('i')
    j = dace.symbol('j')

    @dace.program
    def inner_sdfg(A: dace.int32[5, 5], B: dace.int32[5, 5], select: dace.bool[5, 5]):
        if select[i, j]:
            B[i, j] = A[j, i]
        else:
            B[i, j] = A[i, j]

    sdfg = dace.SDFG('refine_dataflow')
    sdfg.add_array('A', [5, 5], dace.int32)
    sdfg.add_array('B', [5, 5], dace.int32)
    sdfg.add_array('select', [5, 5], dace.bool)

    state = sdfg.add_state()
    A = state.add_access('A')
    B = state.add_access('B')
    select = state.add_access('select')
    me, mx = state.add_map('m', dict(i='0:5', j='0:5'))
    nsdfg = state.add_nested_sdfg(inner_sdfg.to_sdfg(simplify=False), sdfg, {'A', 'select'}, {'B'}, {
        'i': 'i',
        'j': 'j'
    })
    state.add_memlet_path(A, me, nsdfg, dst_conn='A', memlet=dace.Memlet.from_array('A', sdfg.arrays['A']))
    state.add_memlet_path(select,
                          me,
                          nsdfg,
                          dst_conn='select',
                          memlet=dace.Memlet.from_array('select', sdfg.arrays['select']))
    state.add_memlet_path(nsdfg, mx, B, src_conn='B', memlet=dace.Memlet.from_array('B', sdfg.arrays['B']))

    num = sdfg.apply_transformations_repeated(RefineNestedAccess)
    assert num == 1

    for edge in state.out_edges(me):
        if edge.data.data == 'A':
            expr = dace.symbolic.pystr_to_symbolic('Max(i, j)')
            assert edge.data.subset == dace.subsets.Range([(0, expr, 1), (0, expr, 1)])
        else:
            assert edge.data.subset == dace.subsets.Range([(i, i, 1), (j, j, 1)])
    for edge in state.in_edges(mx):
        assert edge.data.subset == dace.subsets.Range([(i, i, 1), (j, j, 1)])

    A = np.arange(25, dtype=np.int32).reshape(5, 5).copy()
    B = np.empty((5, 5), dtype=np.int32)
    select = np.empty((5, 5), dtype=np.bool_)
    select[:] = True
    upper = np.triu(select, k=0)
    sdfg(A=A, B=B, select=upper)
    lower = np.tril(A, k=0)
    diag = np.diag(np.diag(A))
    assert np.allclose(B, lower.T + lower - diag)


def test_free_symbols_only_by_indices():
    i = dace.symbol('i')
    sdfg = dace.SDFG('refine_free_symbols_only_by_indices')
    sdfg.add_array('A', [5], dace.int32)
    sdfg.add_array('B', [5, 5], dace.int32)
    sdfg.add_scalar('idx_a', dace.int64)
    sdfg.add_scalar('idx_b', dace.int64)

    @dace.program
    def inner_sdfg(A: dace.int32[5], B: dace.int32[5, 5], idx_a: int, idx_b: int):
        if A[i] > 0.5:
            B[i, idx_a] = 1
        else:
            B[i, idx_b] = 0

    state = sdfg.add_state()
    A = state.add_access('A')
    B = state.add_access('B')
    ia = state.add_access('idx_a')
    ib = state.add_access('idx_b')
    map_entry, map_exit = state.add_map('map', dict(i='0:5'))
    nsdfg = state.add_nested_sdfg(inner_sdfg.to_sdfg(simplify=False), sdfg, {'A', 'idx_a', 'idx_b'}, {'B'}, {'i': 'i'})
    state.add_memlet_path(A, map_entry, nsdfg, dst_conn='A', memlet=dace.Memlet.from_array('A', sdfg.arrays['A']))
    state.add_memlet_path(nsdfg, map_exit, B, src_conn='B', memlet=dace.Memlet.from_array('B', sdfg.arrays['B']))
    state.add_memlet_path(ia,
                          map_entry,
                          nsdfg,
                          dst_conn='idx_a',
                          memlet=dace.Memlet.from_array('idx_a', sdfg.arrays['idx_a']))
    state.add_memlet_path(ib,
                          map_entry,
                          nsdfg,
                          dst_conn='idx_b',
                          memlet=dace.Memlet.from_array('idx_b', sdfg.arrays['idx_b']))

    num = sdfg.apply_transformations_repeated(RefineNestedAccess)
    assert num == 1

    assert len(state.in_edges(map_exit)) == 1
    edge = state.in_edges(map_exit)[0]
    assert edge.data.subset == dace.subsets.Range([(i, i, 1), (0, 4, 1)])

    A = np.array([0, 1, 0, 1, 0], dtype=np.int32)
    ref = np.zeros((5, 5), dtype=np.int32)
    val = np.zeros((5, 5), dtype=np.int32)
    ia = 3
    ib = 2

    for i in range(5):
        if A[i] > 0.5:
            ref[i, ia] = 1
        else:
            ref[i, ib] = 0
    sdfg(A=A, B=val, idx_a=ia, idx_b=ib)

    assert np.allclose(ref, val)


def _make_rna_read_and_write_set_sdfg(diff_in_out: bool) -> dace.SDFG:
    """Generates the SDFG for the `test_rna_read_and_write_sets_*()` tests.

    If `diff_in_out` is `False` then the output is also used as temporary storage
    within the nested SDFG. Because of the definition of the read/write sets,
    this usage of the temporary storage is not picked up and it is only considered
    as write set.

    If `diff_in_out` is true, then a different storage container, which is classified
    as output, is used as temporary storage.

    This test was added during [PR#1678](https://github.com/spcl/dace/pull/1678).
    """

    def _make_nested_sdfg(diff_in_out: bool) -> dace.SDFG:
        sdfg = dace.SDFG("inner_sdfg")
        state = sdfg.add_state(is_start_block=True)
        sdfg.add_array("A", dtype=dace.float64, shape=(2,), transient=False)
        sdfg.add_array("T1", dtype=dace.float64, shape=(2,), transient=False)

        A = state.add_access("A")
        T1_output = state.add_access("T1")
        if diff_in_out:
            sdfg.add_array("T2", dtype=dace.float64, shape=(2,), transient=False)
            T1_input = state.add_access("T2")
        else:
            T1_input = state.add_access("T1")

        tsklt = state.add_tasklet(
            "comp",
            inputs={"__in1": None, "__in2": None},
            outputs={"__out": None},
            code="__out = __in1 + __in2",
        )

        state.add_edge(A, None, tsklt, "__in1", dace.Memlet("A[1]"))
        # An alternative would be to write to a different location here.
        #  Then, the data would be added to the access node.
        state.add_edge(A, None, T1_input, None, dace.Memlet("A[0] -> [0]"))
        state.add_edge(T1_input, None, tsklt, "__in2", dace.Memlet(T1_input.data + "[0]"))
        state.add_edge(tsklt, "__out", T1_output, None, dace.Memlet(T1_output.data + "[1]"))
        return sdfg

    sdfg = dace.SDFG("Parent_SDFG")
    state = sdfg.add_state(is_start_block=True)
    
    sdfg.add_array("A", dtype=dace.float64, shape=(2,), transient=False)
    sdfg.add_array("T1", dtype=dace.float64, shape=(2,), transient=False)
    sdfg.add_array("T2", dtype=dace.float64, shape=(2,), transient=False)
    A = state.add_access("A")
    T1 = state.add_access("T1")

    nested_sdfg = _make_nested_sdfg(diff_in_out)

    nsdfg = state.add_nested_sdfg(
        nested_sdfg,
        parent=sdfg,
        inputs={"A"},
        outputs={"T2", "T1"} if diff_in_out else {"T1"},
        symbol_mapping={},
    )

    state.add_edge(A, None, nsdfg, "A", dace.Memlet("A[0:2]"))
    state.add_edge(nsdfg, "T1", T1, None, dace.Memlet("T1[0:2]"))

    if diff_in_out:
        state.add_edge(nsdfg, "T2", state.add_access("T2"), None, dace.Memlet("T2[0:2]"))
    sdfg.validate()
    return sdfg


def test_rna_read_and_write_sets_doule_use():
    # The transformation does not apply because we access element `0` of both arrays that we
    #  pass inside the nested SDFG.
    sdfg = _make_rna_read_and_write_set_sdfg(False)
    nb_applied = sdfg.apply_transformations_repeated(
        [RefineNestedAccess],
        validate=True,
        validate_all=True,
    )
    assert nb_applied == 0


def test_rna_read_and_write_sets_different_storage():

    # There is a dedicated temporary storage used.
    sdfg = _make_rna_read_and_write_set_sdfg(True)

    nb_applied = sdfg.apply_transformations_repeated(
        [RefineNestedAccess],
        validate=True,
        validate_all=True,
    )
    assert nb_applied > 0

    args = {
        "A": np.array(np.random.rand(2), dtype=np.float64, copy=True),
        "T2": np.array(np.random.rand(2), dtype=np.float64, copy=True),
        "T1": np.zeros(2, dtype=np.float64),
    }
    ref = args["A"][0] + args["A"][1]
    sdfg(**args)
    res = args["T1"][1]
    assert np.allclose(res, ref), f"Expected '{ref}' but got '{res}'."


if __name__ == '__main__':
    test_refine_dataflow()
    test_refine_interstate()
    test_free_symbols_only_by_indices()
    test_rna_read_and_write_sets_different_storage()
    test_rna_read_and_write_sets_doule_use()
