# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.transformation.interstate import InlineSDFG
import numpy as np
import pytest

W = dace.symbol('W')
H = dace.symbol('H')


@dace.program
def transpose(input, output):
    @dace.map(_[0:H, 0:W])
    def compute(i, j):
        a << input[j, i]
        b >> output[i, j]
        b = a


@dace.program
def bla(A, B, alpha):
    @dace.tasklet
    def something():
        al << alpha
        a << A[0, 0]
        b >> B[0, 0]
        b = al * a


@dace.program
def myprogram(A, B, cst):
    transpose(A, B)
    bla(A, B, cst)


def test():
    myprogram.compile(dace.float32[W, H], dace.float32[H, W], dace.int32)


@pytest.mark.skip
def test_regression_reshape_unsqueeze():
    nsdfg = dace.SDFG("nested_reshape_node")
    nstate = nsdfg.add_state()
    nsdfg.add_array("input", [3, 3], dace.float64)
    nsdfg.add_view("view", [3, 3], dace.float64)
    nsdfg.add_array("output", [9], dace.float64)

    R = nstate.add_read("input")
    A = nstate.add_access("view")
    W = nstate.add_write("output")

    mm1 = dace.Memlet("input[0:3, 0:3] -> 0:3, 0:3")
    mm2 = dace.Memlet("view[0:3, 0:2] -> 3:9")

    nstate.add_edge(R, None, A, None, mm1)
    nstate.add_edge(A, None, W, None, mm2)

    @dace.program
    def test_reshape_unsqueeze(A: dace.float64[3, 3], B: dace.float64[9]):
        nsdfg(input=A, output=B)

    sdfg = test_reshape_unsqueeze.to_sdfg(coarsen=False)
    sdfg.coarsen_dataflow()
    sdfg.validate()

    a = np.random.rand(3, 3)
    b = np.random.rand(9)
    regb = np.copy(b)
    regb[3:9] = a[0:3, 0:2].reshape([6])
    sdfg(A=a, B=b)

    assert np.allclose(b, regb)


def test_empty_memlets():
    sdfg = dace.SDFG('test')
    state = sdfg.add_state('test_state')
    sdfg.add_array('field_a', shape=[1], dtype=float)
    sdfg.add_array('field_b', shape=[1], dtype=float)

    nsdfg1 = dace.SDFG('nsdfg1')
    nstate1 = nsdfg1.add_state('nstate1')
    tasklet1 = nstate1.add_tasklet('tasklet1',
                                   code='b=a',
                                   inputs={'a'},
                                   outputs={'b'})
    nsdfg1.add_array('field_a', shape=[1], dtype=float)
    nsdfg1.add_array('field_b', shape=[1], dtype=float)
    nstate1.add_edge(nstate1.add_read('field_a'), None, tasklet1, 'a',
                     dace.Memlet.simple('field_a', subset_str='0'))
    nstate1.add_edge(tasklet1, 'b', nstate1.add_write('field_b'), None,
                     dace.Memlet.simple('field_b', subset_str='0'))

    nsdfg2 = dace.SDFG('nsdfg2')
    nstate2 = nsdfg2.add_state('nstate2')
    tasklet2 = nstate2.add_tasklet('tasklet2',
                                   code='tmp=a;a_res=a+1',
                                   inputs={'a'},
                                   outputs={'a_res'})
    nsdfg2.add_array('field_a', shape=[1], dtype=float)
    nstate2.add_edge(nstate2.add_read('field_a'), None, tasklet2, 'a',
                     dace.Memlet.simple('field_a', subset_str='0'))
    nstate2.add_edge(tasklet2, 'a_res', nstate2.add_write('field_a'), None,
                     dace.Memlet.simple('field_a', subset_str='0'))

    nsdfg1_node = state.add_nested_sdfg(nsdfg1, None, {'field_a'}, {'field_b'})
    nsdfg2_node = state.add_nested_sdfg(nsdfg2, None, {'field_a'}, {'field_a'})

    a_read = state.add_read('field_a')
    state.add_edge(a_read, None, nsdfg1_node, 'field_a',
                   dace.Memlet.simple('field_a', subset_str='0'))
    state.add_edge(nsdfg1_node, 'field_b', state.add_write('field_b'), None,
                   dace.Memlet.simple('field_b', subset_str='0'))
    state.add_edge(a_read, None, nsdfg2_node, 'field_a',
                   dace.Memlet.simple('field_a', subset_str='0'))
    state.add_edge(nsdfg2_node, 'field_a', state.add_write('field_a'), None,
                   dace.Memlet.simple('field_a', subset_str='0'))
    state.add_edge(nsdfg1_node, None, nsdfg2_node, None, dace.Memlet())

    sdfg.validate()
    sdfg.coarsen_dataflow()


def test_multistate_inline():
    @dace.program
    def nested(A: dace.float64[20]):
        for i in range(5):
            A[i] += A[i - 1]

    @dace.program
    def outerprog(A: dace.float64[20]):
        nested(A)

    sdfg = outerprog.to_sdfg(coarsen=True)
    from dace.transformation.interstate import InlineMultistateSDFG
    sdfg.apply_transformations(InlineMultistateSDFG)
    assert sdfg.number_of_nodes() in (4, 5)

    A = np.random.rand(20)
    expected = np.copy(A)
    outerprog.f(expected)

    outerprog(A)
    assert np.allclose(A, expected)


def test_multistate_inline_samename():
    @dace.program
    def nested(A: dace.float64[20]):
        for i in range(5):
            A[i] += A[i - 1]

    @dace.program
    def outerprog(A: dace.float64[20]):
        for i in range(5):
            nested(A)

    sdfg = outerprog.to_sdfg(coarsen=True)
    from dace.transformation.interstate import InlineMultistateSDFG
    sdfg.apply_transformations(InlineMultistateSDFG)
    assert sdfg.number_of_nodes() in (7, 8)

    A = np.random.rand(20)
    expected = np.copy(A)
    outerprog.f(expected)

    outerprog(A)
    assert np.allclose(A, expected)


def test_inline_symexpr():
    nsdfg = dace.SDFG('inner')
    nsdfg.add_array('a', [20], dace.float64)
    nstate = nsdfg.add_state()
    nstate.add_mapped_tasklet('doit', {'k': '0:20'}, {},
                              '''if k < j:
    o = 2.0''', {'o': dace.Memlet('a[k]', dynamic=True)},
                              external_edges=True)

    sdfg = dace.SDFG('outer')
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_symbol('i', dace.int32)
    state = sdfg.add_state()
    w = state.add_write('A')
    nsdfg_node = state.add_nested_sdfg(nsdfg, None, {}, {'a'},
                                       {'j': 'min(i, 10)'})
    state.add_edge(nsdfg_node, 'a', w, None, dace.Memlet('A'))

    # Verify that compilation works before inlining
    sdfg.compile()

    sdfg.apply_transformations(InlineSDFG)

    # Compile and run
    a = np.random.rand(20)
    sdfg(A=a, i=15)
    assert np.allclose(a[:10], 2.0)
    assert not np.allclose(a[10:], 2.0)


if __name__ == "__main__":
    test()
    # Skipped to to bug that cannot be reproduced
    # test_regression_reshape_unsqueeze()
    test_empty_memlets()
    test_multistate_inline()
    test_multistate_inline_samename()
    test_inline_symexpr()
