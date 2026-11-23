# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import pytest

import dace
from dace import dtypes
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.transformation.dataflow import MapFusionVertical, TaskletFusion
from dace.transformation.optimizer import Optimizer

datatype = dace.float32
np_datatype = np.float32
M = 10
N = 2 * M


@dace.program
def map_with_tasklets(A: datatype[N], B: datatype[M]):
    C = np.zeros_like(B)
    for i in dace.map[0:M]:
        a = A[i] + B[i]
        b = a * A[2 * i]
        c = a + B[i]
        d = c / b
        C[i] = a * d
    return C


def _make_sdfg(language: str):
    lang = dtypes.Language.Python if language == "Python" else dtypes.Language.CPP
    endl = "\n" if language == "Python" else ";\n"

    sdfg = dace.SDFG("map_with_tasklets")
    sdfg.add_array("A", (N,), datatype)
    sdfg.add_array("B", (M,), datatype)
    sdfg.add_array("C", (M,), datatype)
    state = sdfg.add_state(is_start_block=True)
    A = state.add_read("A")
    B = state.add_read("B")
    C = state.add_write("C")
    me, mx = state.add_map("Map", {"i": "0:" + str(M)})
    inputs = {
        "__inp1": datatype,
        "__inp2": datatype,
    }
    outputs = {
        "__out": datatype,
    }
    ta = state.add_tasklet(
        "a",
        inputs,
        {
            "__out1": datatype,
            "__out2": datatype,
            "__out3": datatype,
        },
        f"__out1 = __inp1 + __inp2{endl}__out2 = __out1{endl}__out3 = __out1{endl}",
        lang,
    )
    tb = state.add_tasklet("b", inputs, outputs, f"__out = __inp1 * __inp2{endl}", lang)
    tc = state.add_tasklet("c", inputs, outputs, f"__out = __inp1 + __inp2{endl}", lang)
    td = state.add_tasklet("d", inputs, outputs, f"__out = __inp1 / __inp2{endl}", lang)
    te = state.add_tasklet("e", inputs, outputs, f"__out = __inp1 * __inp2{endl}", lang)
    state.add_memlet_path(A, me, ta, memlet=dace.Memlet("A[i]"), dst_conn="__inp1")
    state.add_memlet_path(B, me, ta, memlet=dace.Memlet("B[i]"), dst_conn="__inp2")
    state.add_memlet_path(A, me, tb, memlet=dace.Memlet("A[2*i]"), dst_conn="__inp2")
    state.add_memlet_path(B, me, tc, memlet=dace.Memlet("B[i]"), dst_conn="__inp2")
    sdfg.add_array("tmp1", (1,), datatype, dtypes.StorageType.Default, None, True)
    sdfg.add_array("tmp2", (1,), datatype, dtypes.StorageType.Default, None, True)
    sdfg.add_array("tmp3", (1,), datatype, dtypes.StorageType.Default, None, True)
    sdfg.add_array("tmp4", (1,), datatype, dtypes.StorageType.Default, None, True)
    sdfg.add_array("tmp5", (1,), datatype, dtypes.StorageType.Default, None, True)
    sdfg.add_array("tmp6", (1,), datatype, dtypes.StorageType.Default, None, True)
    atemp1 = state.add_access("tmp1")
    atemp2 = state.add_access("tmp2")
    atemp3 = state.add_access("tmp3")
    atemp4 = state.add_access("tmp4")
    atemp5 = state.add_access("tmp5")
    atemp6 = state.add_access("tmp6")
    state.add_edge(ta, "__out1", atemp1, None, dace.Memlet("tmp1[0]"))
    state.add_edge(atemp1, None, tb, "__inp1", dace.Memlet("tmp1[0]"))
    state.add_edge(ta, "__out2", atemp2, None, dace.Memlet("tmp2[0]"))
    state.add_edge(atemp2, None, tc, "__inp1", dace.Memlet("tmp2[0]"))
    state.add_edge(tb, "__out", atemp3, None, dace.Memlet("tmp3[0]"))
    state.add_edge(atemp3, None, td, "__inp2", dace.Memlet("tmp3[0]"))
    state.add_edge(tc, "__out", atemp4, None, dace.Memlet("tmp4[0]"))
    state.add_edge(atemp4, None, td, "__inp1", dace.Memlet("tmp4[0]"))
    state.add_edge(ta, "__out3", atemp5, None, dace.Memlet("tmp5[0]"))
    state.add_edge(atemp5, None, te, "__inp1", dace.Memlet("tmp5[0]"))
    state.add_edge(td, "__out", atemp6, None, dace.Memlet("tmp6[0]"))
    state.add_edge(atemp6, None, te, "__inp2", dace.Memlet("tmp6[0]"))
    state.add_memlet_path(te, mx, C, memlet=dace.Memlet("C[i]"), src_conn="__out")

    return sdfg


def test_basic():

    @dace.program
    def test_basic_tf(A: datatype[5, 5]):
        B = A + 1
        return B * 2

    sdfg = test_basic_tf.to_sdfg(simplify=True)

    num_map_fusions = sdfg.apply_transformations(MapFusionVertical)
    assert num_map_fusions == 1
    num_tasklet_fusions = sdfg.apply_transformations(TaskletFusion)
    assert num_tasklet_fusions == 1

    A = np.ones((5, 5), dtype=np_datatype)
    result = sdfg(A=A)
    assert np.allclose(result, 2 * (A + 1))


def test_same_name():

    @dace.program
    def test_same_name(A: datatype[5, 5]):
        B = A + 1
        C = A * 3
        return B + C

    sdfg = test_same_name.to_sdfg(simplify=True)

    num_map_fusions = sdfg.apply_transformations_repeated(MapFusionVertical)
    assert num_map_fusions == 2
    num_tasklet_fusions = sdfg.apply_transformations_repeated(TaskletFusion)
    assert num_tasklet_fusions == 2

    A = np.ones((5, 5), dtype=np_datatype)
    result = sdfg(A=A)
    assert np.allclose(result, (A + 1) + (A * 3))


def test_same_name_different_memlet():

    @dace.program
    def test_same_name_different_memlet(A: datatype[5, 5], B: datatype[5, 5]):
        C = B * 3
        D = A + 1
        return D + C

    sdfg = test_same_name_different_memlet.to_sdfg(simplify=True)

    num_map_fusions = sdfg.apply_transformations_repeated(MapFusionVertical)
    assert num_map_fusions == 2
    num_tasklet_fusions = sdfg.apply_transformations_repeated(TaskletFusion)
    assert num_tasklet_fusions == 2

    A = np.ones((5, 5), dtype=np_datatype)
    B = np.ones((5, 5), dtype=np_datatype) * 2
    result = sdfg(A=A, B=B)
    assert np.allclose(result, (A + 1) + (B * 3))


def test_tasklet_fusion_multiline():

    @dace.program
    def test_tasklet_fusion_multiline(A: datatype):
        B = A + 1
        C = dace.define_local([1], datatype)
        D = dace.define_local([1], datatype)

        with dace.tasklet:
            b << B[0]
            c >> C[0]
            d >> D[0]
            d = b * 3
            c = b + 3

        return C + D

    sdfg = test_tasklet_fusion_multiline.to_sdfg(simplify=True)

    num_tasklet_fusions = sdfg.apply_transformations(TaskletFusion)
    assert num_tasklet_fusions == 1

    result = sdfg(A=1)
    assert result[0] == 11


def test_map_param():

    @dace.program
    def map_uses_param(A: dace.float32[10], B: dace.float32[10], C: dace.float32[10]):
        for i in dace.map[0:10]:
            a = i - A[i]
            b = B[i] * i
            C[i] = a + b

    sdfg = map_uses_param.to_sdfg(simplify=True)

    num_tasklet_fusions = sdfg.apply_transformations_repeated(TaskletFusion)
    assert num_tasklet_fusions == 3

    A = np.zeros([10], dtype=np.float32)
    B = np.ones([10], dtype=np.float32)
    C = np.empty([10], dtype=np.float32)
    sdfg(A=A, B=B, C=C)

    ref = np.array(range(0, 10, 1)) * 2.0
    assert (C == ref).all()


@pytest.mark.parametrize("language", [pytest.param("CPP"), pytest.param("Python")])
def test_map_with_tasklets(language: str):
    sdfg = _make_sdfg(language)
    sdfg.compile()
    sdfg.simplify()
    num = sdfg.apply_transformations_repeated(TaskletFusion)
    assert num == 3
    func = sdfg.compile()
    A = np.arange(1, N + 1, dtype=np_datatype)
    B = np.arange(1, M + 1, dtype=np_datatype)
    C = np.zeros((M,), dtype=np_datatype)
    func(A=A, B=B, C=C)
    ref = map_with_tasklets.f(A, B)
    assert np.allclose(C, ref)


def test_none_connector():

    @dace.program
    def sdfg_none_connector(A: dace.float32[32], B: dace.float32[32]):
        tmp = dace.define_local([32], dace.float32)
        for i in dace.map[0:32]:
            with dace.tasklet:
                a >> tmp[i]
                a = 0

        tmp2 = dace.define_local([32], dace.float32)
        for i in dace.map[0:32]:
            with dace.tasklet:
                a << A[i]
                b >> tmp2[i]
                b = a + 1

        for i in dace.map[0:32]:
            with dace.tasklet:
                a << tmp[i]
                b << tmp2[i]
                c >> B[i]
                c = a + b

    sdfg = sdfg_none_connector.to_sdfg()
    sdfg.simplify()
    applied = sdfg.apply_transformations_repeated(MapFusionVertical)
    assert applied == 2

    map_entry = None
    for node in sdfg.start_state.nodes():
        if isinstance(node, dace.nodes.MapEntry):
            map_entry = node
            break

    assert map_entry is not None
    assert len([edge.src_conn for edge in sdfg.start_state.out_edges(map_entry) if edge.src_conn is None]) == 1

    applied = sdfg.apply_transformations_repeated(TaskletFusion)
    assert applied == 2

    assert sdfg.start_state.out_degree(map_entry) == 1
    assert len([edge.src_conn for edge in sdfg.start_state.out_edges(map_entry) if edge.src_conn is None]) == 0


def test_intermediate_transients():

    @dace.program
    def sdfg_intermediate_transients(A: dace.float32[10], B: dace.float32[10]):
        tmp = dace.define_local_scalar(dace.float32)

        # Use tmp twice to test removal of data
        tmp = A[0] + 1
        tmp = tmp * 2
        B[0] = tmp

    sdfg = sdfg_intermediate_transients.to_sdfg(simplify=True)
    assert len([node for node in sdfg.start_state.data_nodes() if node.data == "tmp"]) == 2

    xforms = Optimizer(sdfg=sdfg).get_pattern_matches(patterns=(TaskletFusion,))
    applied = False
    for xform in xforms:
        if xform.data.data == "tmp":
            xform.apply(sdfg.start_state, sdfg)
            applied = True
            break

    # tmp is used twice, so tasklet fusion should not apply
    assert not applied


def test_transient_in_different_state():

    @dace.program
    def sdfg_intermediate_transients(A: dace.float32[10], B: dace.float32[10]):
        tmp = dace.define_local_scalar(dace.float32)

        # Use tmp in different states to test removal of data
        if B[0] < 0:
            tmp = A[0] - 1
            B[0] = tmp
        else:
            tmp = A[0] + 1
            B[0] = tmp

    sdfg = sdfg_intermediate_transients.to_sdfg(simplify=True)
    assert len([node for state in sdfg.states() for node in state.data_nodes() if node.data == "tmp"]) == 2

    xforms = Optimizer(sdfg=sdfg).get_pattern_matches(patterns=(TaskletFusion,))
    applied = False
    for xform in xforms:
        if xform.data.data == "tmp":
            xform.apply(sdfg.start_state, sdfg)
            applied = True
            break

    # tmp is used twice, so tasklet fusion should not apply
    assert not applied


def test_tasklets_without_debuginfo_fuse():
    """Two tasklets that carry no debug info (a CPF expansion's) fused into ``None.filename`` (heat_3d, 11-22)."""

    @dace.program
    def no_debuginfo_tf(A: datatype[5, 5]):
        B = A + 1
        return B * 2

    sdfg = no_debuginfo_tf.to_sdfg(simplify=True)
    sdfg.apply_transformations(MapFusionVertical)
    for node, parent in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.Tasklet):
            node.debuginfo = None
    assert sdfg.apply_transformations(TaskletFusion) == 1
    A = np.ones((5, 5), dtype=np_datatype)
    assert np.allclose(sdfg(A=A), 2 * (A + 1))


def test_a_write_to_an_argument_is_not_folded():
    """A one-element argument one tasklet writes and another reads is read after the SDFG too (the output of a nested
    SDFG a fused map body writes, fuse_diamond, 11-23): folding the two tasklets would drop the write."""
    sdfg = dace.SDFG("argument_between_tasklets")
    sdfg.add_array("A", [1], dace.float64)
    sdfg.add_array("T", [1], dace.float64)
    sdfg.add_array("U", [1], dace.float64)
    state = sdfg.add_state()
    square = state.add_tasklet("square", {"a"}, {"t"}, "t = a * a")
    shift = state.add_tasklet("shift", {"t"}, {"u"}, "u = t + 1.0")
    middle = state.add_access("T")
    state.add_edge(state.add_read("A"), None, square, "a", dace.Memlet("A[0]"))
    state.add_edge(square, "t", middle, None, dace.Memlet("T[0]"))
    state.add_edge(middle, None, shift, "t", dace.Memlet("T[0]"))
    state.add_edge(shift, "u", state.add_write("U"), None, dace.Memlet("U[0]"))

    assert sdfg.apply_transformations(TaskletFusion) == 0
    A, T, U = np.full(1, 3.0), np.zeros(1), np.zeros(1)
    sdfg(A=A, T=T, U=U)
    assert T[0] == 9.0 and U[0] == 10.0


def test_a_scalar_a_branch_condition_reads_is_not_folded():
    """``c_den = 1.0 + den`` and then ``if abs(1.0 + den) < eps``: folding ``den`` into the tasklet computing ``c_den``
    left the branch condition reading an undeclared ``den`` (srad, 11-23)."""
    sdfg = dace.SDFG("scalar_read_by_branch")
    sdfg.add_array("A", [1], dace.float64)
    sdfg.add_array("C", [1], dace.float64)
    sdfg.add_scalar("den", dace.float64, transient=True)
    state = sdfg.add_state("compute", is_start_block=True)
    half = state.add_tasklet("half", {"a"}, {"d"}, "d = a * 0.5")
    shift = state.add_tasklet("shift", {"d"}, {"c"}, "c = 1.0 + d")
    middle = state.add_access("den")
    state.add_edge(state.add_read("A"), None, half, "a", dace.Memlet("A[0]"))
    state.add_edge(half, "d", middle, None, dace.Memlet("den[0]"))
    state.add_edge(middle, None, shift, "d", dace.Memlet("den[0]"))
    state.add_edge(shift, "c", state.add_write("C"), None, dace.Memlet("C[0]"))
    branch = ConditionalBlock("guard")
    sdfg.add_node(branch)
    sdfg.add_edge(state, branch, dace.InterstateEdge())
    body = ControlFlowRegion("small", sdfg=sdfg)
    branch.add_branch(CodeBlock("(1.0 + den) < 2.0"), body)
    clamp = body.add_state("clamp", is_start_block=True)
    write = clamp.add_tasklet("clamp", {}, {"c"}, "c = 2.0")
    clamp.add_edge(write, "c", clamp.add_write("C"), None, dace.Memlet("C[0]"))
    sdfg.validate()

    assert sdfg.apply_transformations(TaskletFusion) == 0
    for a, expected in ((1.0, 2.0), (4.0, 3.0)):
        A, C = np.full(1, a), np.zeros(1)
        sdfg(A=A, C=C)
        assert C[0] == expected


if __name__ == "__main__":
    test_basic()
    test_same_name()
    test_same_name_different_memlet()
    test_tasklet_fusion_multiline()
    test_map_param()
    test_map_with_tasklets(language="Python")
    test_map_with_tasklets(language="CPP")
    test_none_connector()
    test_intermediate_transients()
    test_transient_in_different_state()
    test_tasklets_without_debuginfo_fuse()
    test_a_write_to_an_argument_is_not_folded()
    test_a_scalar_a_branch_condition_reads_is_not_folded()
