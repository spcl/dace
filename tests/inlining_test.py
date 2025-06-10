# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace import nodes as dace_nodes
from dace.sdfg.state import FunctionCallRegion, NamedRegion
from dace.transformation.interstate import InlineSDFG, StateFusion
from dace.libraries import blas
from dace.library import change_default
from typing import Tuple
import numpy as np
import os
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


def _make_chain_reduction_sdfg() -> Tuple[dace.SDFG, dace.SDFGState, dace_nodes.NestedSDFG, dace_nodes.NestedSDFG]:

    def _make_nested_sdfg(name: str) -> dace.SDFG:
        sdfg = dace.SDFG(name)
        state = sdfg.add_state(is_start_block=True)

        for name in "ABC":
            sdfg.add_array(
                name=name,
                shape=(10, ),
                dtype=dace.float64,
                transient=False,
            )
        state.add_mapped_tasklet(
            "comp",
            map_ranges={"__i": "0:10"},
            inputs={
                "__in1": dace.Memlet("A[__i]"),
                "__in2": dace.Memlet("B[__i]"),
            },
            code="__out = __in1 + __in2",
            outputs={"__out": dace.Memlet("C[__i]")},
            external_edges=True,
        )
        sdfg.validate()
        return sdfg

    outer_sdfg = dace.SDFG("chain_reduction_sdfg")
    state = outer_sdfg.add_state(is_start_block=True)

    anames_s10 = ["I1", "I2", "I3", "I4", "T1", "T4", "T5"]
    anames_s20 = ["T2"]
    anames_s30 = ["T3", "O"]
    sizes = dict()
    sizes.update({name: 10 for name in anames_s10})
    sizes.update({name: 20 for name in anames_s20})
    sizes.update({name: 30 for name in anames_s30})

    for name in anames_s10 + anames_s20 + anames_s30:
        outer_sdfg.add_array(
            name=name,
            shape=(sizes[name], ),
            dtype=dace.float64,
            transient=name.startswith("T"),
        )
    T1, T2, T3, T4, T5 = (state.add_access(f"T{i}") for i in range(1, 6))

    state.add_mapped_tasklet(
        "comp1",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("I3[__i]")},
        code="__out = __in + 1.0",
        outputs={"__out": dace.Memlet("T4[__i]")},
        external_edges=True,
        output_nodes={T4},
    )

    inner_sdfg_1 = _make_nested_sdfg("first_adding")
    nsdfg_node_1 = state.add_nested_sdfg(
        sdfg=inner_sdfg_1,
        parent=outer_sdfg,
        inputs={"A", "B"},
        outputs={"C"},
        symbol_mapping={},
    )

    state.add_edge(state.add_access("I4"), None, nsdfg_node_1, "A", dace.Memlet("I4[0:10]"))
    state.add_edge(state.add_access("I1"), None, nsdfg_node_1, "B", dace.Memlet("I1[0:10]"))
    state.add_edge(nsdfg_node_1, "C", T5, None, dace.Memlet("T5[0:10]"))

    state.add_nedge(T4, T2, dace.Memlet("T4[0:10] -> [0:10]"))
    state.add_nedge(T5, T2, dace.Memlet("T5[0:10] -> [10:20]"))

    inner_sdfg_2 = _make_nested_sdfg("second_adding")
    nsdfg_node_2 = state.add_nested_sdfg(
        sdfg=inner_sdfg_2,
        parent=outer_sdfg,
        inputs={"A", "B"},
        outputs={"C"},
        symbol_mapping={},
    )

    state.add_edge(state.add_access("I1"), None, nsdfg_node_2, "A", dace.Memlet("I1[0:10]"))
    state.add_edge(state.add_access("I2"), None, nsdfg_node_2, "B", dace.Memlet("I2[0:10]"))

    state.add_edge(nsdfg_node_2, "C", T1, None, dace.Memlet("T1[0:10]"))

    state.add_nedge(T1, T3, dace.Memlet("T1[0:10] -> [0:10]"))
    state.add_nedge(T2, T3, dace.Memlet("T2[0:20] -> [10:30]"))
    state.add_nedge(T3, state.add_access("O"), dace.Memlet("T3[0:30] -> [0:30]"))
    outer_sdfg.validate()

    return outer_sdfg, state, nsdfg_node_1, nsdfg_node_2


def _perform_chain_reduction_inlining(which: int, ) -> None:
    from dace.transformation.interstate import InlineMultistateSDFG

    def count_writes(sdfg):
        nb_writes = 0
        for state in sdfg.states():
            for dnode in state.data_nodes():
                nb_writes += state.in_degree(dnode)
        return nb_writes

    def count_nsdfg(sdfg):
        nb_nsdfg = 0
        for state in sdfg.states():
            nb_nsdfg += sum(isinstance(node, dace_nodes.NestedSDFG) for node in state.nodes())
        return nb_nsdfg

    ref = {arg: np.array(np.random.rand(10), dtype=np.float64, copy=True) for arg in ["I1", "I2", "I3", "I4"]}
    ref["O"] = np.array(np.random.rand(30), dtype=np.float64, copy=True)
    res = {k: v.copy() for k, v in ref.items()}

    sdfg, state, nsdfg_node_1, nsdfg_node_2 = _make_chain_reduction_sdfg()
    initial_writes = count_writes(sdfg)
    initial_nsdfg = count_nsdfg(sdfg)

    csdfg_ref = sdfg.compile()
    csdfg_ref(**ref)

    if which == 0:
        InlineMultistateSDFG.apply_to(
            sdfg=sdfg,
            verify=True,
            nested_sdfg=nsdfg_node_1,
        )
    elif which == 1:
        InlineMultistateSDFG.apply_to(
            sdfg=sdfg,
            verify=True,
            nested_sdfg=nsdfg_node_2,
        )
    elif which == -1:
        nb_applied = sdfg.apply_transformations_repeated(InlineMultistateSDFG)
        assert nb_applied == 2
    else:
        raise ValueError(f"Unknown selector {which}")

    assert initial_writes == count_writes(sdfg)
    assert count_nsdfg(sdfg) < initial_nsdfg

    csdfg_res = sdfg.compile()
    csdfg_res(**res)

    assert all(np.allclose(ref[k], res[k]) for k in ref.keys())


@pytest.mark.skip('CI failure that cannot be reproduced outside CI')
def test_regression_reshape_unsqueeze():
    nsdfg = dace.SDFG("nested_reshape_node")
    nstate = nsdfg.add_state()
    nsdfg.add_array("input", [3, 3], dace.float64)
    nsdfg.add_view("view", [3, 3], dace.float64)
    nsdfg.add_array("output", [9], dace.float64)

    R = nstate.add_read("input")
    A = nstate.add_access("view")
    W = nstate.add_write("output")

    mm1 = dace.Memlet("input[0:3, 0:3] -> [0:3, 0:3]")
    mm2 = dace.Memlet("view[0:3, 0:2] -> [3:9]")

    nstate.add_edge(R, None, A, None, mm1)
    nstate.add_edge(A, None, W, None, mm2)

    @dace.program
    def test_reshape_unsqueeze(A: dace.float64[3, 3], B: dace.float64[9]):
        nsdfg(input=A, output=B)

    sdfg = test_reshape_unsqueeze.to_sdfg(simplify=False)
    sdfg.simplify()
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
    tasklet1 = nstate1.add_tasklet('tasklet1', code='b=a', inputs={'a'}, outputs={'b'})
    nsdfg1.add_array('field_a', shape=[1], dtype=float)
    nsdfg1.add_array('field_b', shape=[1], dtype=float)
    nstate1.add_edge(nstate1.add_read('field_a'), None, tasklet1, 'a', dace.Memlet.simple('field_a', subset_str='0'))
    nstate1.add_edge(tasklet1, 'b', nstate1.add_write('field_b'), None, dace.Memlet.simple('field_b', subset_str='0'))

    nsdfg2 = dace.SDFG('nsdfg2')
    nstate2 = nsdfg2.add_state('nstate2')
    tasklet2 = nstate2.add_tasklet('tasklet2', code='tmp=a;a_res=a+1', inputs={'a'}, outputs={'a_res'})
    nsdfg2.add_array('field_a', shape=[1], dtype=float)
    nstate2.add_edge(nstate2.add_read('field_a'), None, tasklet2, 'a', dace.Memlet.simple('field_a', subset_str='0'))
    nstate2.add_edge(tasklet2, 'a_res', nstate2.add_write('field_a'), None, dace.Memlet.simple('field_a',
                                                                                               subset_str='0'))

    nsdfg1_node = state.add_nested_sdfg(nsdfg1, None, {'field_a'}, {'field_b'})
    nsdfg2_node = state.add_nested_sdfg(nsdfg2, None, {'field_a'}, {'field_a'})

    a_read = state.add_read('field_a')
    state.add_edge(a_read, None, nsdfg1_node, 'field_a', dace.Memlet.simple('field_a', subset_str='0'))
    state.add_edge(nsdfg1_node, 'field_b', state.add_write('field_b'), None,
                   dace.Memlet.simple('field_b', subset_str='0'))
    state.add_edge(a_read, None, nsdfg2_node, 'field_a', dace.Memlet.simple('field_a', subset_str='0'))
    state.add_edge(nsdfg2_node, 'field_a', state.add_write('field_a'), None,
                   dace.Memlet.simple('field_a', subset_str='0'))
    state.add_edge(nsdfg1_node, None, nsdfg2_node, None, dace.Memlet())

    sdfg.validate()
    sdfg.simplify()


def test_multistate_inline():

    @dace.program
    def nested(A: dace.float64[20]):
        for i in range(5):
            A[i] += A[i - 1]

    @dace.program
    def outerprog(A: dace.float64[20]):
        nested(A)

    sdfg = outerprog.to_sdfg(simplify=True)

    A = np.random.rand(20)
    expected = np.copy(A)
    outerprog.f(expected)

    from dace.transformation.interstate import InlineMultistateSDFG
    sdfg.apply_transformations(InlineMultistateSDFG)
    assert sdfg.number_of_nodes() in (1, 2)

    sdfg(A)
    assert np.allclose(A, expected)


def test_multistate_inline_samename():

    @dace.program
    def nested(A: dace.float64[20]):
        for i in range(5):
            A[i + 1] += A[i]

    @dace.program
    def outerprog(A: dace.float64[20]):
        for i in range(5):
            nested(A)

    sdfg = outerprog.to_sdfg(simplify=False)

    A = np.random.rand(20)
    expected = np.copy(A)
    outerprog.f(expected)

    from dace.transformation.interstate import InlineMultistateSDFG
    sdfg.apply_transformations(InlineMultistateSDFG)
    sdfg.simplify()
    assert sdfg.number_of_nodes() == 1

    sdfg(A)
    assert np.allclose(A, expected)


def test_multistate_inline_outer_dependencies():

    @dace.program
    def nested(A: dace.float64[20]):
        for i in range(1, 20):
            A[i] += A[i - 1]

    @dace.program
    def outerprog(A: dace.float64[20], B: dace.float64[20]):
        for i in dace.map[0:20]:
            with dace.tasklet:
                a >> A[i]
                b >> B[i]

                a = 0
                b = 1

        nested(A)

        for i in dace.map[0:20]:
            with dace.tasklet:
                a << A[i]
                b >> A[i]

                b = 2 * a

    sdfg = outerprog.to_sdfg(simplify=False)
    for cf in sdfg.all_control_flow_regions():
        if isinstance(cf, (FunctionCallRegion, NamedRegion)):
            cf.inline()
    sdfg.apply_transformations_repeated((StateFusion, InlineSDFG))
    assert len(sdfg.nodes()) == 1

    A = np.random.rand(20)
    B = np.random.rand(20)
    expected_a = np.copy(A)
    expected_b = np.copy(B)
    outerprog.f(expected_a, expected_b)

    from dace.transformation.interstate import InlineMultistateSDFG
    sdfg.apply_transformations(InlineMultistateSDFG)

    sdfg(A, B)
    assert np.allclose(A, expected_a)
    assert np.allclose(B, expected_b)


def test_multistate_inline_concurrent_subgraphs():

    @dace.program
    def nested(A: dace.float64[10], B: dace.float64[10]):
        for i in range(1, 10):
            B[i] = A[i]

    @dace.program
    def outerprog(A: dace.float64[10], B: dace.float64[10], C: dace.float64[10]):
        nested(A, B)

        for i in dace.map[0:10]:
            with dace.tasklet:
                a << A[i]
                c >> C[i]

                c = 2 * a

    sdfg = outerprog.to_sdfg(simplify=False)
    for cf in sdfg.all_control_flow_regions():
        if isinstance(cf, (FunctionCallRegion, NamedRegion)):
            cf.inline()
    dace.propagate_memlets_sdfg(sdfg)
    sdfg.apply_transformations_repeated((StateFusion, InlineSDFG))
    assert len(sdfg.nodes()) == 1
    assert len([node for node in sdfg.start_state.data_nodes()]) == 3

    A = np.random.rand(10)
    B = np.random.rand(10)
    C = np.random.rand(10)
    expected_a = np.copy(A)
    expected_b = np.copy(B)
    expected_c = np.copy(C)
    outerprog.f(expected_a, expected_b, expected_c)

    from dace.transformation.interstate import InlineMultistateSDFG
    applied = sdfg.apply_transformations(InlineMultistateSDFG)
    assert applied == 1

    sdfg(A, B, C)
    assert np.allclose(A, expected_a)
    assert np.allclose(B, expected_b)
    assert np.allclose(C, expected_c)


def test_chain_reduction_1():
    _perform_chain_reduction_inlining(0)


def test_chain_reduction_2():
    _perform_chain_reduction_inlining(1)


def test_chain_reduction_all():
    _perform_chain_reduction_inlining(-1)


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
    nsdfg_node = state.add_nested_sdfg(nsdfg, None, {}, {'a'}, {'j': 'min(i, 10)'})
    state.add_edge(nsdfg_node, 'a', w, None, dace.Memlet('A'))

    # Verify that compilation works before inlining
    sdfg.compile()

    sdfg.apply_transformations(InlineSDFG)

    # Compile and run
    a = np.random.rand(20)
    sdfg(A=a, i=15)
    assert np.allclose(a[:10], 2.0)
    assert not np.allclose(a[10:], 2.0)


def test_inline_unsqueeze():

    @dace.program
    def nested_squeezed(c: dace.int32[5], d: dace.int32[5]):
        d[:] = c

    @dace.program
    def inline_unsqueeze(A: dace.int32[2, 5], B: dace.int32[5, 3]):
        nested_squeezed(A[1, :], B[:, 1])

    sdfg = inline_unsqueeze.to_sdfg()
    sdfg.apply_transformations(InlineSDFG)

    A = np.arange(10, dtype=np.int32).reshape(2, 5).copy()
    B = np.zeros((5, 3), np.int32)
    sdfg(A, B)
    for i in range(3):
        if i == 1:
            assert (np.array_equal(B[:, i], A[1, :]))
        else:
            assert (np.array_equal(B[:, i], np.zeros((5, ), np.int32)))


def test_inline_unsqueeze2():

    @dace.program
    def nested_squeezed(c, d):
        d[:] = c

    @dace.program
    def inline_unsqueeze(A: dace.int32[2, 5], B: dace.int32[5, 3]):
        for i in range(2):
            nested_squeezed(A[i, :], B[:, 1 - i])

    sdfg = inline_unsqueeze.to_sdfg()
    sdfg.apply_transformations(InlineSDFG)

    A = np.arange(10, dtype=np.int32).reshape(2, 5).copy()
    B = np.zeros((5, 3), np.int32)
    sdfg(A, B)
    for i in range(3):
        if i < 2:
            assert (np.array_equal(B[:, 1 - i], A[i, :]))
        else:
            assert (np.array_equal(B[:, i], np.zeros((5, ), np.int32)))


def test_inline_unsqueeze3():

    @dace.program
    def nested_squeezed(c, d):
        d[:] = c

    @dace.program
    def inline_unsqueeze(A: dace.int32[2, 5], B: dace.int32[5, 3]):
        for i in range(2):
            nested_squeezed(A[i, i:i + 2], B[i + 1:i + 3, 1 - i])

    sdfg = inline_unsqueeze.to_sdfg()
    sdfg.apply_transformations(InlineSDFG)

    A = np.arange(10, dtype=np.int32).reshape(2, 5).copy()
    B = np.zeros((5, 3), np.int32)
    sdfg(A, B)
    for i in range(3):
        if i < 2:
            assert (np.array_equal(B[i + 1:i + 3, 1 - i], A[i, i:i + 2]))
        else:
            assert (np.array_equal(B[:, i], np.zeros((5, ), np.int32)))


def test_inline_unsqueeze4():

    @dace.program
    def nested_squeezed(c, d):
        d[:] = c

    @dace.program
    def inline_unsqueeze(A: dace.int32[2, 5], B: dace.int32[5, 3]):
        for i in range(2):
            nested_squeezed(A[i, i:2 * i + 2], B[i + 1:2 * i + 3, 1 - i])

    sdfg = inline_unsqueeze.to_sdfg()
    sdfg.apply_transformations(InlineSDFG)

    A = np.arange(10, dtype=np.int32).reshape(2, 5).copy()
    B = np.zeros((5, 3), np.int32)
    last_value = os.environ.get('DACE_testing_serialization', '0')
    os.environ['DACE_testing_serialization'] = '0'
    sdfg(A, B)
    os.environ['DACE_testing_serialization'] = last_value
    for i in range(3):
        if i < 2:
            assert (np.array_equal(B[i + 1:2 * i + 3, 1 - i], A[i, i:2 * i + 2]))
        else:
            assert (np.array_equal(B[:, i], np.zeros((5, ), np.int32)))


def test_inline_symbol_assignment():

    def nested(a, num):
        cat = num - 1
        last_step = (cat == 0)
        if last_step is True:
            return a + 1

        return a

    @dace.program
    def tester(a: dace.float64[20], b: dace.float64[10, 20]):
        for i in range(10):
            cat = nested(a, i)
            b[i] = cat

    sdfg = tester.to_sdfg()
    sdfg.compile()


def test_regression_inline_subset():
    nsdfg = dace.SDFG("nested_sdfg")
    nstate = nsdfg.add_state()
    nsdfg.add_array("input", [96, 32], dace.float64)
    nsdfg.add_array("output", [32, 32], dace.float64)
    nstate.add_edge(nstate.add_read("input"), None, nstate.add_write("output"), None,
                    dace.Memlet("input[32:64, 0:32] -> [0:32, 0:32]"))

    @dace.program
    def test(A: dace.float64[96, 32]):
        B = dace.define_local([32, 32], dace.float64)
        nsdfg(input=A, output=B)
        return B + 1

    sdfg = test.to_sdfg(simplify=False)
    sdfg.apply_transformations_repeated(StateFusion)
    sdfg.validate()
    sdfg.simplify()
    sdfg.validate()
    data = np.random.rand(96, 32)
    out = test(data)
    assert np.allclose(out, data[32:64, :] + 1)


def test_inlining_view_input():

    @dace.program
    def test(A: dace.float64[96, 32], B: dace.float64[42, 32]):
        O = np.zeros([96 * 2, 42], dace.float64)
        for i in dace.map[0:2]:
            O[i * 96:(i + 1) * 96, :] = np.einsum("ij,kj->ik", A, B)
        return O

    sdfg = test.to_sdfg()
    with change_default(blas, "pure"):
        sdfg.expand_library_nodes()
    sdfg.simplify()

    state = sdfg.sink_nodes()[0]
    # find nested_sdfg
    nsdfg = [n for n in state.nodes() if isinstance(n, dace.sdfg.nodes.NestedSDFG)][0]
    # delete gemm initialization state
    nsdfg.sdfg.remove_node(nsdfg.sdfg.nodes()[0])

    # check that inlining the sdfg works
    sdfg.simplify()

    A = np.random.rand(96, 32)
    B = np.random.rand(42, 32)

    expected = np.concatenate([A @ B.T, A @ B.T], axis=0)
    actual = sdfg(A=A, B=B)
    np.testing.assert_allclose(expected, actual)


if __name__ == "__main__":
    test()
    # Skipped due to bug that cannot be reproduced outside CI
    # test_regression_reshape_unsqueeze()
    test_empty_memlets()
    test_multistate_inline()
    test_multistate_inline_outer_dependencies()
    test_multistate_inline_concurrent_subgraphs()
    test_multistate_inline_samename()
    test_inline_symexpr()
    test_inline_unsqueeze()
    test_inline_unsqueeze2()
    test_inline_unsqueeze3()
    test_inline_unsqueeze4()
    test_inline_symbol_assignment()
    test_regression_inline_subset()
    test_inlining_view_input()
    test_chain_reduction_1()
    test_chain_reduction_2()
    test_chain_reduction_all()
