# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Any, Union, Tuple, Optional

import numpy as np
import os
import dace
import copy
import uuid
import pytest

from dace import SDFG, SDFGState
from dace.sdfg import nodes
from dace.transformation.dataflow import MapFusion, MapExpansion


def count_node(sdfg: SDFG, node_type):
    nb_nodes = 0
    for rsdfg in sdfg.all_sdfgs_recursive():
        for state in rsdfg.states():
            for node in state.nodes():
                if isinstance(node, node_type):
                    nb_nodes += 1
    return nb_nodes


def safe_view(sdfg: SDFG):
    """Calls `sdfg.view()` but if it fails does nothing.

    Mostly needed for the CI, for whatever reason.
    """
    try:
        sdfg.view()
    except Exception:
        pass


def apply_fusion(
    sdfg: SDFG,
    removed_maps: Union[int, None] = None,
    final_maps: Union[int, None] = None,
    unspecific: bool = False,
    apply_once: bool = False,
    strict_dataflow: bool = True,
) -> SDFG:
    """Applies the Map fusion transformation.

    The function checks that the number of maps has been reduced, it is also possible
    to specify the number of removed maps. It is also possible to specify the final
    number of maps.
    If `unspecific` is set to `True` then the function will just apply the
    transformation and not check if maps were removed at all.
    If `strict_dataflow` is set to `True`, the default, then the function will perform
    the fusion in strict dataflow mode.
    """
    org_sdfg = copy.deepcopy(sdfg)
    num_maps_before = None if unspecific else count_node(sdfg, nodes.MapEntry)

    try:
        with dace.config.temporary_config():
            dace.Config.set("optimizer", "match_exception", value=True)
            if apply_once:
                sdfg.apply_transformations(MapFusion(strict_dataflow=strict_dataflow), validate=True, validate_all=True)
            else:
                sdfg.apply_transformations_repeated(MapFusion(strict_dataflow=strict_dataflow),
                                                    validate=True,
                                                    validate_all=True)
    except:
        safe_view(org_sdfg)
        safe_view(sdfg)
        raise

    if unspecific:
        return sdfg

    num_maps_after = count_node(sdfg, nodes.MapEntry)
    has_processed = False
    if removed_maps is not None:
        has_processed = True
        rm = num_maps_before - num_maps_after
        if not (rm == removed_maps):
            safe_view(sdfg)
        assert rm == removed_maps, f"Expected to remove {removed_maps} but removed {rm}"
    if final_maps is not None:
        has_processed = True
        if not (final_maps == num_maps_after):
            safe_view(sdfg)
        assert final_maps == num_maps_after, f"Expected that only {final_maps} maps remain, but there are sill {num_maps_after}."
    if not has_processed:
        if not (num_maps_after < num_maps_before):
            safe_view(sdfg)
        assert num_maps_after < num_maps_before, f"Maps after: {num_maps_after}; Maps before: {num_maps_before}"
    return sdfg


@dace.program
def fusion_simple(A: dace.float32[10, 20], B: dace.float32[10, 20], out: dace.float32[1]):
    tmp = dace.define_local([10, 20], dtype=A.dtype)
    tmp_2 = dace.define_local([10, 20], dtype=A.dtype)
    for i, j in dace.map[0:10, 0:20]:
        with dace.tasklet:
            a << A[i, j]
            b >> tmp[i, j]

            b = a * a

    for i, j in dace.map[0:10, 0:20]:
        with dace.tasklet:
            a << tmp[i, j]
            b << B[i, j]
            c >> tmp_2[i, j]

            c = a + b

    for i, j in dace.map[0:10, 0:20]:
        with dace.tasklet:
            a << tmp_2[i, j]
            b >> out(1, lambda a, b: a + b)[0]

            b = a


@dace.program
def fusion_rename(A: dace.float32[10, 20], B: dace.float32[10, 20], out: dace.float32[1]):
    tmp = dace.define_local([10, 20], dtype=A.dtype)
    tmp_2 = dace.define_local([10, 20], dtype=A.dtype)
    for i, j in dace.map[0:10, 0:20]:
        with dace.tasklet:
            a << A[i, j]
            b >> tmp[i, j]

            b = a * a

    for i, j in dace.map[0:20, 0:10]:
        with dace.tasklet:
            a << tmp[j, i]
            b << B[j, i]
            c >> tmp_2[j, i]

            c = a + b

    for m, n in dace.map[0:10, 0:20]:
        with dace.tasklet:
            a << tmp_2[m, n]
            b >> out(1, lambda a, b: a + b)[0]

            b = a


@dace.program
def multiple_fusions(A: dace.float32[10, 20], B: dace.float32[10, 20], C: dace.float32[10, 20], out: dace.float32[1]):
    A_prime = dace.define_local([10, 20], dtype=A.dtype)
    A_prime_copy = dace.define_local([10, 20], dtype=A.dtype)
    for i, j in dace.map[0:10, 0:20]:
        with dace.tasklet:
            inp << A[i, j]
            out1 >> out(1, lambda a, b: a + b)[0]
            out2 >> A_prime[i, j]
            out3 >> A_prime_copy[i, j]
            out1 = inp
            out2 = inp * inp
            out3 = inp * inp

    for i, j in dace.map[0:10, 0:20]:
        with dace.tasklet:
            inp << A_prime[i, j]
            out1 >> B[i, j]
            out1 = inp + 1

    for i, j in dace.map[0:10, 0:20]:
        with dace.tasklet:
            inp << A_prime_copy[i, j]
            out2 >> C[i, j]
            out2 = inp + 2


@dace.program
def fusion_chain(A: dace.float32[10, 20], B: dace.float32[10, 20]):
    tmp1 = A * 2
    tmp2 = tmp1 * 4
    B[:] = tmp2 + 5


@dace.program
def fusion_with_transient(A: dace.float64[2, 20]):
    res = np.ndarray([2, 20], dace.float64)
    for i in dace.map[0:20]:
        for j in dace.map[0:2]:
            with dace.tasklet:
                a << A[j, i]
                t >> res[j, i]
                t = a * a
    for i in dace.map[0:20]:
        for j in dace.map[0:2]:
            with dace.tasklet:
                t << res[j, i]
                o >> A[j, i]
                o = t * 2


@dace.program
def fusion_shared_output(A: dace.float32[10, 20], B: dace.float32[10, 20], C: dace.float32[10, 20]):
    tmp = A + 3
    B[:] = tmp * 4
    C[:] = tmp / 6


@dace.program
def fusion_indirect_access(A: dace.float32[100], B: dace.float32[100], idx: dace.int32[30], out: dace.float32[30]):
    tmp = (A + B * 2) + 3
    out[:] = tmp[idx]


def make_interstate_transient_fusion_sdfg():
    sdfg = dace.SDFG("interstate_transient_fusion")
    state1 = sdfg.add_state("state1", is_start_block=True)
    state2 = sdfg.add_state_after(state1, "state2")

    for name in ["A", "B", "C", "D"]:
        sdfg.add_array(name, shape=(20, 20), dtype=dace.float64, transient=False)
    sdfg.arrays["B"].transient = True

    A1, B1, C1 = (state1.add_access(name) for name in ["A", "B", "C"])
    state1.add_mapped_tasklet(
        "map_1_1",
        map_ranges={
            "__i0": "0:20",
            "__i1": "0:20"
        },
        inputs={"__in1": dace.Memlet("A[__i0, __i1]")},
        code="__out = __in1 + 20",
        outputs={"__out": dace.Memlet("B[__i0, __i1]")},
        input_nodes={"A": A1},
        output_nodes={"B": B1},
        external_edges=True,
    )
    state1.add_mapped_tasklet(
        "map_2_1",
        map_ranges={
            "__i3": "0:20",
            "__i4": "0:20"
        },
        inputs={"__in1": dace.Memlet("B[__i3, __i4]")},
        code="__out = __in1 + 10",
        outputs={"__out": dace.Memlet("C[__i3, __i4]")},
        input_nodes={"B": B1},
        output_nodes={"C": C1},
        external_edges=True,
    )

    B2, D2 = (state2.add_access(name) for name in ["B", "D"])
    state2.add_mapped_tasklet(
        "map_1_2",
        map_ranges={
            "__i4": "0:20",
            "__i3": "0:20"
        },
        inputs={"__in1": dace.Memlet("B[__i4, __i3]")},
        code="__out = __in1 + 6",
        outputs={"__out": dace.Memlet("D[__i4, __i3]")},
        input_nodes={"B": B2},
        output_nodes={"D": D2},
        external_edges=True,
    )

    return sdfg, state1, state2


def test_fusion_simple():
    sdfg = fusion_simple.to_sdfg(simplify=True)
    sdfg = apply_fusion(sdfg, final_maps=1)

    A = np.random.rand(10, 20).astype(np.float32)
    B = np.random.rand(10, 20).astype(np.float32)
    out = np.zeros(shape=1, dtype=np.float32)
    sdfg(A=A, B=B, out=out)

    diff = abs(np.sum(A * A + B) - out)
    print('Difference:', diff)
    assert diff <= 1e-3


def test_fusion_rename():
    sdfg = fusion_rename.to_sdfg(simplify=True)
    sdfg = apply_fusion(sdfg, final_maps=1)

    A = np.random.rand(10, 20).astype(np.float32)
    B = np.random.rand(10, 20).astype(np.float32)
    out = np.zeros(shape=1, dtype=np.float32)
    sdfg(A=A, B=B, out=out)

    diff = abs(np.sum(A * A + B) - out)
    print('Difference:', diff)
    assert diff <= 1e-3


def test_fusion_shared():
    sdfg = fusion_shared_output.to_sdfg(simplify=True)
    sdfg = apply_fusion(sdfg)

    A = np.random.rand(10, 20).astype(np.float32)
    B = np.random.rand(10, 20).astype(np.float32)
    C = np.random.rand(10, 20).astype(np.float32)

    B_res = (A + 3) * 4
    C_res = (A + 3) / 6
    sdfg(A=A, B=B, C=C)

    assert np.allclose(B_res, B)
    assert np.allclose(C_res, C)


def test_indirect_accesses():
    sdfg = fusion_indirect_access.to_sdfg(simplify=True)
    sdfg = apply_fusion(sdfg, final_maps=2)

    A = np.random.rand(100).astype(np.float32)
    B = np.random.rand(100).astype(np.float32)
    idx = ((np.random.rand(30) * 100) % 100).astype(np.int32)
    out = np.zeros(shape=30, dtype=np.float32)

    res = ((A + B * 2) + 3)[idx]
    sdfg(A=A, B=B, idx=idx, out=out)

    assert np.allclose(res, out)


def test_multiple_fusions():
    sdfg = multiple_fusions.to_sdfg(simplify=True)

    sdfg.save(os.path.join('_dacegraphs', 'before2.sdfg'))
    sdfg.simplify()
    sdfg = apply_fusion(sdfg)

    A = np.random.rand(10, 20).astype(np.float32)
    B = np.zeros_like(A)
    C = np.zeros_like(A)
    out = np.zeros(shape=1, dtype=np.float32)
    sdfg(A=A, B=B, C=C, out=out)
    diff1 = np.linalg.norm(A * A + 1 - B)
    diff2 = np.linalg.norm(A * A + 2 - C)
    print('Difference1:', diff1)
    assert diff1 <= 1e-4

    print('Difference2:', diff2)
    assert diff2 <= 1e-4


def test_fusion_chain():
    sdfg = fusion_chain.to_sdfg(simplify=True)
    sdfg.simplify()
    sdfg = apply_fusion(sdfg, final_maps=1)

    A = np.random.rand(10, 20).astype(np.float32)
    B = np.zeros_like(A)
    sdfg(A=A, B=B)
    diff = np.linalg.norm(A * 8 + 5 - B)
    print('Difference:', diff)
    assert diff <= 1e-4


def test_fusion_with_transient():
    A = np.random.rand(2, 20)
    expected = A * A * 2
    sdfg = fusion_with_transient.to_sdfg(simplify=True)
    sdfg.simplify()
    sdfg = apply_fusion(sdfg, removed_maps=2)

    sdfg(A=A)
    assert np.allclose(A, expected)


def test_fusion_with_transient_scalar():
    N = 10
    K = 4

    def build_sdfg():
        sdfg = dace.SDFG("map_fusion_with_transient_scalar")
        state = sdfg.add_state()
        sdfg.add_array("A", (N, K), dace.float64)
        sdfg.add_array("B", (N, ), dace.float64)
        sdfg.add_array("T", (N, ), dace.float64, transient=True)
        t_node = state.add_access("T")
        sdfg.add_scalar("V", dace.float64, transient=True)
        v_node = state.add_access("V")

        me1, mx1 = state.add_map("map1", dict(i=f"0:{N}"))
        tlet1 = state.add_tasklet("select", {"_v"}, {"_out"}, f"_out = _v[i, {K-1}]")
        state.add_memlet_path(state.add_access("A"),
                              me1,
                              tlet1,
                              dst_conn="_v",
                              memlet=dace.Memlet.from_array("A", sdfg.arrays["A"]))
        state.add_edge(tlet1, "_out", v_node, None, dace.Memlet("V[0]"))
        state.add_memlet_path(v_node, mx1, t_node, memlet=dace.Memlet("T[i]"))

        me2, mx2 = state.add_map("map2", dict(j=f"0:{N}"))
        tlet2 = state.add_tasklet("numeric", {"_inp"}, {"_out"}, f"_out = _inp + 1")
        state.add_memlet_path(t_node, me2, tlet2, dst_conn="_inp", memlet=dace.Memlet("T[j]"))
        state.add_memlet_path(tlet2, mx2, state.add_access("B"), src_conn="_out", memlet=dace.Memlet("B[j]"))

        return sdfg

    sdfg = build_sdfg()
    sdfg = apply_fusion(sdfg)

    A = np.random.rand(N, K)
    B = np.repeat(np.nan, N)
    sdfg(A=A, B=B)

    assert np.allclose(B, (A[:, K - 1] + 1))


def test_fusion_with_inverted_indices():

    @dace.program
    def inverted_maps(A: dace.int32[10]):
        B = np.empty_like(A)
        for i in dace.map[0:10]:
            B[i] = i
        for i in dace.map[0:10]:
            A[9 - i] = B[9 - i] + 5

    ref = np.arange(5, 15, dtype=np.int32)

    sdfg = inverted_maps.to_sdfg(simplify=True)
    val0 = np.ndarray((10, ), dtype=np.int32)
    sdfg(A=val0)
    assert np.array_equal(val0, ref)

    # This can not be fused
    apply_fusion(sdfg, removed_maps=0)

    val1 = np.ndarray((10, ), dtype=np.int32)
    sdfg(A=val1)
    assert np.array_equal(val1, ref), f"REF: {ref}; VAL: {val1}"


def test_fusion_with_empty_memlet():

    N = dace.symbol('N', positive=True)

    @dace.program
    def inner_product(A: dace.float32[N], B: dace.float32[N], out: dace.float32[1]):
        tmp = np.empty_like(A)
        for i in dace.map[0:N:128]:
            for j in dace.map[0:128]:
                tmp[i + j] = A[i + j] * B[i + j]
        for i in dace.map[0:N:128]:
            lsum = dace.float32(0)
            for j in dace.map[0:128]:
                lsum = lsum + tmp[i + j]
            out[0] += lsum

    sdfg = inner_product.to_sdfg(simplify=True)
    apply_fusion(sdfg, removed_maps=2)

    A = np.arange(1024, dtype=np.float32)
    B = np.arange(1024, dtype=np.float32)
    val = np.zeros((1, ), dtype=np.float32)
    sdfg(A=A, B=B, out=val, N=1024)
    ref = A @ B
    assert np.allclose(val[0], ref)


def test_fusion_with_nested_sdfg_0():

    def ref(A, B, C):
        tmp = np.zeros_like(A)
        for i in dace.map[0:10]:
            if C[i] < 0:
                tmp[i] = B[i] - A[i]
            else:
                tmp[i] = B[i] + A[i]
        for i in dace.map[0:10]:
            A[i] = tmp[i] * 2

    def _make_sdfg() -> dace.SDFG:
        sdfg = SDFG("fusion_with_nested_sdfg_0")
        state = sdfg.add_state(is_start_block=True)

        for name in "ABCT":
            sdfg.add_array(
                name,
                shape=(10, ),
                dtype=dace.float64,
                transient=False,
            )
        sdfg.arrays["T"].transient = True

        me1, mx1 = state.add_map("first_map", ndrange={"__i0": "0:10"})
        nsdfg = state.add_nested_sdfg(
            sdfg=_make_nested_sdfg(),
            parent=sdfg,
            inputs={"a", "b", "c"},
            outputs={"t"},
            symbol_mapping={},
        )

        for name in "ABC":
            state.add_edge(
                state.add_access(name),
                None,
                me1,
                "IN_" + name,
                dace.Memlet(f"{name}[0:10]"),
            )
            me1.add_in_connector("IN_" + name)
            state.add_edge(
                me1,
                "OUT_" + name,
                nsdfg,
                name.lower(),
                dace.Memlet(f"{name}[__i0]"),
            )
            me1.add_out_connector("OUT_" + name)
        state.add_edge(
            nsdfg,
            "t",
            mx1,
            "IN_T",
            dace.Memlet("T[__i0]"),
        )
        T = state.add_access("T")
        state.add_edge(
            mx1,
            "OUT_T",
            T,
            None,
            dace.Memlet("T[0:10]"),
        )
        mx1.add_in_connector("IN_T")
        mx1.add_out_connector("OUT_T")

        state.add_mapped_tasklet(
            "comp2",
            map_ranges={"__i4": "0:10"},
            inputs={"__in1": dace.Memlet("T[__i4]")},
            code="__out = __in1 * 2",
            outputs={"__out": dace.Memlet("A[__i4]")},
            input_nodes={T},
            external_edges=True,
        )
        sdfg.validate()
        return sdfg

    def _make_nested_sdfg() -> dace.SDFG:
        sdfg = SDFG("Nested")

        for name in "abct":
            sdfg.add_scalar(
                name,
                dtype=dace.float64,
                transient=False,
            )

        state_head = sdfg.add_state("head_state", is_start_block=True)
        state_if_guard = sdfg.add_state("state_if_guard")
        sdfg.add_edge(state_head, state_if_guard, dace.InterstateEdge(
            condition="1",
            assignments={"__tmp2": "c < 0.0"},
        ))

        def _make_branch_tasklet(
            state: dace.SDFGState,
            code: str,
        ) -> None:
            tasklet = state.add_tasklet(
                state.label + "_tasklet",
                inputs={"__in1", "__in2"},
                code=code,
                outputs={"__out"},
            )
            state.add_edge(
                state.add_access("b"),
                None,
                tasklet,
                "__in1",
                dace.Memlet("b[0]"),
            )
            state.add_edge(
                state.add_access("a"),
                None,
                tasklet,
                "__in2",
                dace.Memlet("a[0]"),
            )
            state.add_edge(
                tasklet,
                "__out",
                state.add_access("t"),
                None,
                dace.Memlet("t[0]"),
            )

        state_trueb = sdfg.add_state("true_branch")
        _make_branch_tasklet(state_trueb, "__out = __in1 - __in2")
        state_falseb = sdfg.add_state("false_branch")
        _make_branch_tasklet(state_falseb, "__out = __in1 + __in2")
        state_if_end = sdfg.add_state("if_join")

        sdfg.add_edge(state_if_guard, state_trueb, dace.InterstateEdge(condition="__tmp2"))
        sdfg.add_edge(state_if_guard, state_falseb, dace.InterstateEdge(condition="not __tmp2"))
        sdfg.add_edge(state_falseb, state_if_end, dace.InterstateEdge())
        sdfg.add_edge(state_trueb, state_if_end, dace.InterstateEdge())
        sdfg.validate()
        return sdfg

    sdfg = _make_sdfg()
    apply_fusion(sdfg)

    for sd in sdfg.all_sdfgs_recursive():
        if sd is not sdfg:
            node = sd.parent_nsdfg_node
            state = sd.parent
            for e0 in state.out_edges(node):
                for e1 in state.memlet_tree(e0):
                    dst = state.memlet_path(e1)[-1].dst
                assert isinstance(dst, dace.nodes.AccessNode)

    args_ref = {
        'A': np.array(np.random.rand(10), dtype=np.float64, copy=True),
        'B': np.array(np.random.rand(10), dtype=np.float64, copy=True),
        'C': np.array(np.random.rand(10) - 0.5, dtype=np.float64, copy=True),
    }
    args_res = copy.deepcopy(args_ref)

    ref(**args_ref)
    sdfg(**args_res)
    for arg in args_ref.keys():
        arg_ref = args_ref[arg]
        arg_res = args_res[arg]
        assert np.allclose(arg_ref, arg_res), f"Failed in {arg}"


def test_fusion_with_nested_sdfg_1():

    # As a side effect this test also ensures that dynamic consumer edges, does not
    #  impact fusing, i.e. allow that fusion can take place.
    @dace.program
    def fusion_with_nested_sdfg_1(A: dace.int32[10], B: dace.int32[10], C: dace.int32[10]):
        tmp = np.empty([10], dtype=np.int32)
        for i in dace.map[0:10]:
            with dace.tasklet:
                a << A[i]
                b << B[i]
                t >> tmp[i]
                t = b - a
        for i in dace.map[0:10]:
            if C[i] < 0:
                A[i] = tmp[i] * 2
            else:
                B[i] = tmp[i] * 2

    sdfg = fusion_with_nested_sdfg_1.to_sdfg(simplify=True)
    apply_fusion(sdfg)

    if len(sdfg.states()) != 1:
        return

    for sd in sdfg.all_sdfgs_recursive():
        if sd is not sdfg:
            node = sd.parent_nsdfg_node
            state = sd.parent
            for e0 in state.in_edges(node):
                for e1 in state.memlet_tree(e0):
                    src = state.memlet_path(e1)[0].src
                assert isinstance(src, dace.nodes.AccessNode)


def test_interstate_fusion():
    """Transient between two maps is used in another state and must become shared.
    """
    sdfg, state1, state2 = make_interstate_transient_fusion_sdfg()

    A = np.random.rand(20, 20)
    C = np.random.rand(20, 20)
    D = np.random.rand(20, 20)

    ref_C = A + 30
    ref_D = A + 26

    apply_fusion(sdfg, removed_maps=1)
    assert sdfg.number_of_nodes() == 2
    assert len([node for node in state1.data_nodes() if node.data == "B"]) == 1

    sdfg(A=A, C=C, D=D)

    assert np.allclose(C, ref_C)
    assert np.allclose(D, ref_D)


def test_fuse_indirect_accesses():

    @dace.program(auto_optimize=False)
    def inner_product(
        A: dace.float32[20],
        B: dace.float32[20],
        idx: dace.int32[20],
        out: dace.float32[20],
    ):
        tmp1 = np.empty_like(A)
        tmp2 = np.empty_like(A)
        for i in dace.map[0:20]:
            tmp1[i] = A[i] * B[i]
        for i in dace.map[0:20]:
            tmp2[i] = tmp1[i] + A[i]
        for i in dace.map[0:20]:
            with dace.tasklet:
                __arr << tmp2(1)[:]
                __idx << idx[i]
                __out >> out[i]
                __out = __arr[__idx]

    sdfg = inner_product.to_sdfg(simplify=True)
    assert sdfg.number_of_nodes() == 1
    assert count_node(sdfg, nodes.MapEntry) == 3

    apply_fusion(sdfg, final_maps=2)

    # The last map, with the indirect access, can not be fused, so check that.
    state = next(iter(sdfg.nodes()))
    assert len(list(state.sink_nodes())) == 1
    out_node = next(iter(state.sink_nodes()))
    assert out_node.data == "out"
    assert state.in_degree(out_node) == 1

    # Now find the last map and the indirect access Tasklet
    last_map_exit = next(iter(state.in_edges(out_node))).src
    last_map_entry = state.entry_node(last_map_exit)
    assert isinstance(last_map_exit, nodes.MapExit)
    assert state.in_degree(last_map_exit) == 1

    indirect_access_tasklet = next(iter(state.in_edges(last_map_exit))).src
    assert isinstance(indirect_access_tasklet, nodes.Tasklet)
    assert indirect_access_tasklet.code == "__out = __arr[__idx]"  # TODO: Regex with connectors

    # The tasklet can only be connected to a map entry.
    assert all(in_edge.src is last_map_entry for in_edge in state.in_edges(indirect_access_tasklet))


def make_correction_offset_sdfg(
    range_read: bool,
    second_read_start: int,
) -> SDFG:
    """Make the SDFGs for the `test_offset_correction_*` tests.

    Args:
        range_read: If `True` then a range is read in the second map.
            if `False` then only a scalar is read.
        second_read_start: Where the second map should start reading.
    """
    sdfg = SDFG("offset_correction_test")
    state = sdfg.add_state(is_start_block=True)
    shapes = {
        "A": (20, 10),
        "B": (20, 8),
        "C": (20, 2) if range_read else (20, 1),
    }
    descs = {}
    for name, shape in shapes.items():
        _, desc = sdfg.add_array(name, shape, dace.float64, transient=False)
        descs[name] = desc
    sdfg.arrays["B"].transient = True
    A, B, C = (state.add_access(name) for name in sorted(shapes.keys()))

    state.add_mapped_tasklet(
        "first_map",
        map_ranges={
            "i": "0:20",
            "j": "2:8"
        },
        inputs={"__in1": dace.Memlet("A[i, j]")},
        code="__out = __in1 + 1.0",
        outputs={"__out": dace.Memlet("B[i, j]")},
        input_nodes={"A": A},
        output_nodes={"B": B},
        external_edges=True,
    )
    state.add_mapped_tasklet(
        "second_map",
        map_ranges=({
            "l": "0:20",
            "k": "0:2"
        } if range_read else {
            "l": "0:20"
        }),
        inputs={"__in1": dace.Memlet(f"B[l, {second_read_start}{'+k' if range_read else ''}]")},
        code="__out = __in1",
        outputs={"__out": dace.Memlet(f"C[l, {'k' if range_read else '0'}]")},
        input_nodes={"B": B},
        output_nodes={"C": C},
        external_edges=True,
    )
    sdfg.validate()
    assert sdfg.apply_transformations_repeated(MapExpansion, validate_all=True) > 0
    return sdfg


def test_offset_correction_range_read():

    np.random.seed(42)
    A = np.random.rand(20, 10)
    C = np.zeros((20, 2))
    exp = (A + 1.0)[:, 3:5].copy()

    sdfg = make_correction_offset_sdfg(range_read=True, second_read_start=3)

    sdfg(A=A, C=C)
    assert np.allclose(C, exp)
    C[:] = 0.0

    apply_fusion(sdfg)

    sdfg(A=A, C=C)
    assert np.allclose(C, exp)


def test_offset_correction_scalar_read():

    np.random.seed(42)
    A = np.random.rand(20, 10)
    C = np.zeros((20, 1))
    exp = (A + 1.0)[:, 3].copy().reshape((-1, 1))

    sdfg = make_correction_offset_sdfg(range_read=False, second_read_start=3)

    sdfg(A=A, C=C)
    assert np.allclose(C, exp)
    C[:] = 0.0

    apply_fusion(sdfg)

    sdfg(A=A, C=C)
    assert np.allclose(C, exp)


def test_offset_correction_empty():

    # Because the second map starts reading from 1, but the second map only
    #  starts writing from 2 there is no overlap and it can not be fused.
    #  NOTE: This computation is useless.
    sdfg = make_correction_offset_sdfg(range_read=True, second_read_start=1)

    apply_fusion(sdfg, removed_maps=0)


def test_different_offsets():

    def exptected(A, B):
        N, M = A.shape
        return (A + 1) + B[1:(N + 1), 2:(M + 2)]

    def _make_sdfg(N: int, M: int) -> dace.SDFG:
        sdfg = dace.SDFG("test_different_access")
        names = ["A", "B", "__tmp", "__return"]
        def_shape = (N, M)
        sshape = {"B": (N + 1, M + 2), "__tmp": (N + 1, M + 1)}
        for name in names:
            sdfg.add_array(
                name,
                shape=sshape.get(name, def_shape),
                dtype=dace.float64,
                transient=False,
            )
        sdfg.arrays["__tmp"].transient = True

        state = sdfg.add_state(is_start_block=True)
        A, B, _tmp, _return = (state.add_access(name) for name in names)

        state.add_mapped_tasklet(
            "comp1",
            map_ranges={
                "__i0": f"0:{N}",
                "__i1": f"0:{M}"
            },
            inputs={"__in": dace.Memlet("A[__i0, __i1]")},
            code="__out = __in + 1.0",
            outputs={"__out": dace.Memlet("__tmp[__i0 + 1, __i1 + 1]")},
            input_nodes={A},
            output_nodes={_tmp},
            external_edges=True,
        )
        state.add_mapped_tasklet(
            "comp2",
            map_ranges={
                "__i0": f"0:{N}",
                "__i1": f"0:{M}"
            },
            inputs={
                "__in1": dace.Memlet("__tmp[__i0 + 1, __i1 + 1]"),
                "__in2": dace.Memlet("B[__i0 + 1, __i1 + 2]"),
            },
            code="__out = __in1 + __in2",
            outputs={"__out": dace.Memlet("__return[__i0, __i1]")},
            input_nodes={_tmp, B},
            output_nodes={_return},
            external_edges=True,
        )

        sdfg.validate()
        return sdfg

    N, M = 14, 17
    sdfg = _make_sdfg(N, M)
    apply_fusion(sdfg, final_maps=1)

    A = np.array(np.random.rand(N, M), dtype=np.float64, copy=True)
    B = np.array(np.random.rand(N + 1, M + 2), dtype=np.float64, copy=True)

    ref = exptected(A, B)
    res = sdfg(A=A, B=B)
    assert np.allclose(ref, res)


def _make_strict_dataflow_sdfg_pointwise(
    input_data: str = "A",
    intermediate_data: str = "T",
    output_data: Optional[str] = None,
    input_read: str = "__i0",
    output_write: Optional[str] = None,
) -> Tuple[dace.SDFG, dace.SDFGState]:
    """
    Creates the SDFG for the strict data flow tests.

    The SDFG will read and write into `A`, but it is pointwise, thus the Maps can
    be fused. Furthermore, this particular SDFG guarantees that no data race occurs.
    """
    if output_data is None:
        output_data = input_data
    if output_write is None:
        output_write = input_read

    sdfg = dace.SDFG(f"strict_dataflow_sdfg_pointwise_{str(uuid.uuid1()).replace('-', '_')}")
    state = sdfg.add_state(is_start_block=True)
    for name in {input_data, intermediate_data, output_data}:
        sdfg.add_array(
            name,
            shape=(10, ),
            dtype=dace.float64,
            transient=False,
        )

    if intermediate_data not in {input_data, output_data}:
        sdfg.arrays[intermediate_data].transient = True

    input_node, intermediate_node, output_node = (state.add_access(name)
                                                  for name in [input_data, intermediate_data, output_data])

    state.add_mapped_tasklet(
        "first_comp",
        map_ranges={"__i0": "0:10"},
        inputs={"__in1": dace.Memlet(f"{input_data}[{input_read}]")},
        code="__out = __in1 + 2.0",
        outputs={"__out": dace.Memlet(f"{intermediate_data}[__i0]")},
        input_nodes={input_node},
        output_nodes={intermediate_node},
        external_edges=True,
    )
    state.add_mapped_tasklet(
        "second_comp",
        map_ranges={"__i1": "0:10"},
        inputs={"__in1": dace.Memlet(f"{intermediate_data}[__i1]")},
        code="__out = __in1 + 3.0",
        outputs={"__out": dace.Memlet(f"{output_data}[{output_write}]")},
        input_nodes={intermediate_node},
        output_nodes={output_node},
        external_edges=True,
    )
    sdfg.validate()
    return sdfg, state


def test_fusion_strict_dataflow_pointwise():
    sdfg, state = _make_strict_dataflow_sdfg_pointwise(input_data="A")

    # However, if strict dataflow is disabled, then it will be able to fuse.
    apply_fusion(sdfg, removed_maps=1, strict_dataflow=False)


def test_fusion_strict_dataflow_not_pointwise():
    sdfg, state = _make_strict_dataflow_sdfg_pointwise(
        input_data="A",
        input_read="__i0",
        output_write="9 - __i0",
    )

    # Because the dependency is not pointwise even disabling strict dataflow
    #  will not make it work.
    apply_fusion(sdfg, removed_maps=0, strict_dataflow=False)


def test_fusion_dataflow_intermediate():
    sdfg, _ = _make_strict_dataflow_sdfg_pointwise(
        input_data="A",
        intermediate_data="O",
        output_data="O",
    )
    apply_fusion(sdfg, removed_maps=0, strict_dataflow=True)

    # Because the intermediate is also output of the second map it is not possible
    #  to fuse even without strict dataflow mode.
    apply_fusion(sdfg, removed_maps=0, strict_dataflow=False)


def test_fusion_dataflow_intermediate_2():
    # The transformation applies for two reasons, first reading and writing `A`
    #  is pointwise. Furthermore, there is no further access to `A` after the
    #  intermediate node. Note that if the second map would also have an output
    #  that refers to `A` then the transformation would not apply regardless
    #  of the strict dataflow mode.
    sdfg, state = _make_strict_dataflow_sdfg_pointwise(
        input_data="A",
        intermediate_data="A",
        output_data="O",
    )
    apply_fusion(sdfg, removed_maps=1, strict_dataflow=True)
    map_exit = next(iter(node for node in state.nodes() if isinstance(node, nodes.MapExit)))
    assert state.out_degree(map_exit) == 2
    assert {"A", "O"} == {edge.dst.data for edge in state.out_edges(map_exit) if isinstance(edge.dst, nodes.AccessNode)}


def test_fusion_dataflow_intermediate_3():
    # This is exactly the same situation as in `test_fusion_dataflow_intermediate_2()`
    #  with the exception that now the access to `A` is no longer pointwise, thus
    #  the transformation does not apply. Note that this SDFG is wrong, it is only
    #  here to show that the case is detected.
    sdfg, state = _make_strict_dataflow_sdfg_pointwise(
        input_data="A",
        intermediate_data="A",
        output_data="O",
        input_read="9 - __i0",
        output_write="__i0",
    )
    apply_fusion(sdfg, removed_maps=0, strict_dataflow=True)


def test_fusion_dataflow_intermediate_downstream():
    # Because the intermediate `T` is used downstream again,
    #  the transformation can not apply.
    sdfg, state = _make_strict_dataflow_sdfg_pointwise(
        input_data="A",
        intermediate_data="T",
        output_data="output_1",
    )
    sdfg.arrays["output_1"].transient = False
    sdfg.arrays["T"].transient = True
    output_1 = next(iter(dnode for dnode in state.sink_nodes()))
    assert isinstance(output_1, nodes.AccessNode) and output_1.data == "output_1"

    # Make the real output node.
    sdfg.arrays["O"] = sdfg.arrays["A"].clone()
    state.add_mapped_tasklet(
        "downstream_computation",
        map_ranges={"__i10": "0:10"},
        inputs={"__in1": dace.Memlet("output_1[__i10]")},
        code="__out = __in1 + 10.0",
        outputs={"__out": dace.Memlet("T[__i10]")},
        input_nodes={output_1},
        external_edges=True,
    )

    # Make another state where `T` is written back, such that it is not dead data flow.
    state2 = sdfg.add_state_after(state)
    sdfg.add_datadesc("output_2", sdfg.arrays["output_1"].clone())
    state2.add_nedge(
        state2.add_access("T"),
        state2.add_access("output_2"),
        sdfg.make_array_memlet("T"),
    )
    sdfg.validate()

    apply_fusion(sdfg, removed_maps=0, strict_dataflow=True)

    # However without strict dataflow, the merge is possible.
    apply_fusion(sdfg, removed_maps=1, strict_dataflow=False)
    assert state.in_degree(output_1) == 1
    assert state.out_degree(output_1) == 1
    assert all(isinstance(edge.src, nodes.MapExit) for edge in state.in_edges(output_1))
    assert all(isinstance(edge.dst, nodes.MapEntry) for edge in state.out_edges(output_1))

    upper_map_exit = next(iter(edge.src for edge in state.in_edges(output_1)))
    assert isinstance(upper_map_exit, nodes.MapExit)
    assert state.out_degree(upper_map_exit) == 2
    assert {"T", "output_1"
            } == {edge.dst.data
                  for edge in state.out_edges(upper_map_exit) if isinstance(edge.dst, nodes.AccessNode)}


def test_fusion_non_strict_dataflow_implicit_dependency():
    """
    This test simulates if the fusion respect implicit dependencies, given by access nodes.

    This test simulates a situation that could arise if non strict dataflow is enabled.
    The test ensures that the fusion does not continue fusing in this situation.
    """
    sdfg = dace.SDFG("fusion_strict_dataflow_implicit_dependency_sdfg")
    state = sdfg.add_state(is_start_block=True)
    names = ["A", "B", "T1", "T2", "C"]
    for name in names:
        sdfg.add_array(
            name,
            shape=(10, ),
            dtype=dace.float64,
            transient=False,
        )
    sdfg.arrays["T1"].transient = True
    sdfg.arrays["T2"].transient = True

    me, mx = state.add_map("first_map", ndrange={"__i0": "0:10"})
    tskl1 = state.add_tasklet("tskl1", inputs={"__in1", "__in2"}, code="__out = __in1 * __in2", outputs={"__out"})
    tskl2 = state.add_tasklet("tskl2", inputs={"__in1", "__in2"}, code="__out = (__in1 + __in2) / 2", outputs={"__out"})
    A, B, T1, T2 = (state.add_access(name) for name in names[:-1])

    state.add_edge(A, None, me, "IN_A", dace.Memlet("A[0:10]"))
    state.add_edge(B, None, me, "IN_B", dace.Memlet("B[0:10]"))
    me.add_in_connector("IN_A")
    me.add_in_connector("IN_B")

    state.add_edge(me, "OUT_A", tskl1, "__in1", dace.Memlet("A[__i0]"))
    state.add_edge(me, "OUT_B", tskl1, "__in2", dace.Memlet("B[__i0]"))
    state.add_edge(me, "OUT_A", tskl2, "__in1", dace.Memlet("A[__i0]"))
    state.add_edge(me, "OUT_B", tskl2, "__in2", dace.Memlet("B[__i0]"))
    me.add_out_connector("OUT_A")
    me.add_out_connector("OUT_B")

    state.add_edge(tskl1, "__out", mx, "IN_T1", dace.Memlet("T1[__i0]"))
    state.add_edge(tskl2, "__out", mx, "IN_T2", dace.Memlet("T2[__i0]"))
    mx.add_in_connector("IN_T1")
    mx.add_in_connector("IN_T2")

    state.add_edge(mx, "OUT_T1", T1, None, dace.Memlet("T1[0:10]"))
    state.add_edge(mx, "OUT_T2", T2, None, dace.Memlet("T2[0:10]"))
    mx.add_out_connector("OUT_T1")
    mx.add_out_connector("OUT_T2")

    state.add_mapped_tasklet(
        "second_map",
        map_ranges={"__i0": "0:10"},
        inputs={"__in1": dace.Memlet("T1[__i0]")},
        code="if __in1 < 0.5:\n\t__out = 100.",
        outputs={"__out": dace.Memlet("T2[__i0]", dynamic=True)},
        input_nodes={T1},
        external_edges=True,
    )

    state2 = sdfg.add_state_after(state)
    state2.add_edge(
        state2.add_access("T2"),
        None,
        state2.add_access("C"),
        None,
        dace.Memlet("T2[0:10] -> [0:10]"),
    )
    sdfg.validate()

    apply_fusion(sdfg, removed_maps=0, strict_dataflow=False)


def _make_inner_conflict_shared_scalar(has_conflict: bool, ) -> dace.SDFG:
    """Generate the SDFG for tests with the inner dependency.

    If `has_conflict` is `True` then a transient scalar is used inside both Map bodies.
    Therefore, `MapFusion` should not be able to fuse them.
    In case `has_conflict` is `False` then different scalars are used which allows
    fusing the two maps.
    """
    sdfg = dace.SDFG("inner_map_dependency_sdfg" if has_conflict else "inner_map_dependency_resolved_sdfg")
    state = sdfg.add_state(is_start_block=True)

    name_arrays = ["A", "T", "C"]
    for aname in name_arrays:
        sdfg.add_array(
            aname,
            shape=(10, ),
            dtype=dace.float64,
            transient=False,
        )
    sdfg.arrays["T"].transient = True

    name_scalars = ["s", "s"] if has_conflict else ["s1", "s2"]
    for sname in set(name_scalars):
        sdfg.add_scalar(
            sname,
            dtype=dace.float64,
            transient=True,
        )
    A, T, C = (state.add_access(aname) for aname in name_arrays)
    s1, s2 = (state.add_access(sname) for sname in name_scalars)

    me1, mx1 = state.add_map(
        "map_1",
        ndrange={"__i0": "0:10"},
    )
    tsklt1 = state.add_tasklet(
        "tskl1",
        inputs={"__in1"},
        outputs={"__out"},
        code="__out = __in1 + 1.0",
    )

    # Create the first map series.
    state.add_edge(A, None, me1, "IN_A", dace.Memlet("A[0:10]"))
    me1.add_in_connector("IN_A")
    state.add_edge(me1, "OUT_A", s1, None, dace.Memlet("A[__i0] -> [0]"))
    me1.add_out_connector("OUT_A")
    state.add_edge(s1, None, tsklt1, "__in1", dace.Memlet(f"{s1.data}[0]"))
    state.add_edge(tsklt1, "__out", mx1, "IN_T", dace.Memlet("T[__i0]"))
    mx1.add_in_connector("IN_T")
    state.add_edge(mx1, "OUT_T", T, None, dace.Memlet("T[0:10]"))
    mx1.add_out_connector("OUT_T")

    # Create the second map.
    me2, mx2 = state.add_map(
        "map_2",
        ndrange={"__i0": "0:10"},
    )
    tsklt2 = state.add_tasklet(
        "tskl2",
        inputs={"__in1"},
        outputs={"__out"},
        code="__out = __in1 + 3.0",
    )

    state.add_edge(T, None, me2, "IN_T", dace.Memlet("T[0:10]"))
    me2.add_in_connector("IN_T")
    state.add_edge(me2, "OUT_T", s2, None, dace.Memlet("T[__i0]"))
    me2.add_out_connector("OUT_T")
    state.add_edge(s2, None, tsklt2, "__in1", dace.Memlet(f"{s2.data}[0]"))
    state.add_edge(tsklt2, "__out", mx2, "IN_C", dace.Memlet("C[__i0]"))
    mx2.add_in_connector("IN_C")
    state.add_edge(mx2, "OUT_C", C, None, dace.Memlet("C[0:10]"))
    mx2.add_out_connector("OUT_C")
    sdfg.validate()
    return sdfg


def test_inner_map_dependency():
    # Because the scalar is not shared the maps can not be fused.
    sdfg = _make_inner_conflict_shared_scalar(has_conflict=True)
    apply_fusion(sdfg, removed_maps=0, final_maps=2)


def test_inner_map_dependency_resolved():
    # Because the scalars are different, the scalar
    sdfg = _make_inner_conflict_shared_scalar(has_conflict=False)
    apply_fusion(sdfg, removed_maps=1, final_maps=1)


def _impl_fusion_intermediate_different_access(modified_shape: bool, traditional_memlet_direction: bool):

    def ref(A, B):
        T = np.zeros((A.shape[0] + 1, 2))
        for i in range(A.shape[0]):
            T[i + 1, 0] = A[i] * 2
            T[i + 1, 1] = A[i] / 2
        for j in range(A.shape[0]):
            B[j] = np.sin(T[j + 1, 1])

    sdfg = dace.SDFG("fusion_intermediate_different_access_sdfg")
    state = sdfg.add_state(is_start_block=True)
    for name in "AB":
        sdfg.add_array(
            name,
            shape=(10, ),
            dtype=dace.float64,
            transient=False,
        )
    sdfg.add_array(
        "T",
        shape=(11, 2),
        dtype=dace.float64,
        transient=True,
    )

    # For this intermediate, which essentially represents `[A[i] * 2, A[i] / 2]` in
    #  the reference above, there are two important remarks:
    #  - It exists because one data stream, i.e. `T[i + 1, 1]` would be dead data flow
    #       and currently the transformation can not handle this.
    #  - The strange shape is because the transformation can not handle this case.
    #       This is a limitation of the implementation.
    sdfg.add_array(
        "temp",
        shape=((
            1,
            2,
        ) if modified_shape else (2, )),
        dtype=dace.float64,
        transient=True,
    )

    A, B, T, temp = (state.add_access(name) for name in ["A", "B", "T", "temp"])

    me1, mx1 = state.add_map(
        "first_map",
        ndrange={"__i0": "0:10"},
    )

    state.add_edge(A, None, me1, "IN_A", dace.Memlet("A[0:10]"))
    me1.add_in_connector("IN_A")
    me1.add_out_connector("OUT_A")

    tsklt1_1 = state.add_tasklet(
        "tsklt1_1",
        inputs={"__in1"},
        outputs={"__out"},
        code="__out = __in1 * 2.0",
    )
    state.add_edge(me1, "OUT_A", tsklt1_1, "__in1", dace.Memlet("A[__i0]"))
    state.add_edge(tsklt1_1, "__out", temp, None, dace.Memlet("temp[0, 0]" if modified_shape else "temp[0]"))

    tsklt1_2 = state.add_tasklet(
        "tsklt1_2",
        inputs={"__in1"},
        outputs={"__out"},
        code="__out = __in1 / 2.0",
    )
    state.add_edge(me1, "OUT_A", tsklt1_2, "__in1", dace.Memlet("A[__i0]"))
    state.add_edge(tsklt1_2, "__out", temp, None, dace.Memlet("temp[0, 1]" if modified_shape else "temp[1]"))

    temp_subset = ("0, 0:2" if modified_shape else "0:2")
    T_subset = "__i0 + 1, 0:2"

    if traditional_memlet_direction:
        mem_data = "T"
        mem_subset = T_subset
        mem_other_subset = temp_subset
    else:
        mem_data = "temp"
        mem_subset = temp_subset
        mem_other_subset = T_subset

    state.add_edge(temp, None, mx1, "IN_temp", dace.Memlet(f"{mem_data}[{mem_subset}] -> [{mem_other_subset}]"))
    state.add_edge(mx1, "OUT_temp", T, None, dace.Memlet("T[1:11, 0:2]"))
    mx1.add_in_connector("IN_temp")
    mx1.add_out_connector("OUT_temp")

    state.add_mapped_tasklet(
        "comp2",
        map_ranges={"__i1": "0:10"},
        inputs={"__in1": dace.Memlet("T[__i1 + 1, 1]")},
        code="__out = math.sin(__in1)",
        outputs={"__out": dace.Memlet("B[__i1]")},
        input_nodes={T},
        output_nodes={B},
        external_edges=True,
    )
    sdfg.validate()

    apply_fusion(sdfg, removed_maps=1, final_maps=1)

    args_ref = {
        'A': np.array(np.random.rand(10), dtype=np.float64, copy=True),
        'B': np.array(np.random.rand(10), dtype=np.float64, copy=True),
    }
    args_res = copy.deepcopy(args_ref)

    ref(**args_ref)
    sdfg(**args_res)
    for arg in args_ref.keys():
        arg_ref = args_ref[arg]
        arg_res = args_res[arg]
        assert np.allclose(arg_ref, arg_res)


def test_fusion_intermediate_different_access():
    _impl_fusion_intermediate_different_access(modified_shape=False, traditional_memlet_direction=False)


def test_fusion_intermediate_different_access_2():
    _impl_fusion_intermediate_different_access(modified_shape=False, traditional_memlet_direction=True)


def test_fusion_intermediate_different_access_mod_shape():
    _impl_fusion_intermediate_different_access(modified_shape=True, traditional_memlet_direction=False)


def test_fusion_intermediate_different_access_mod_shape_2():
    _impl_fusion_intermediate_different_access(modified_shape=True, traditional_memlet_direction=True)


@pytest.mark.skip(reason="This feature is not yet fully supported.")
def test_fusion_multiple_producers_consumers():
    """Multiple producer and consumer nodes.

    This test is very similar to the `test_fusion_intermediate_different_access()`
    and `test_fusion_intermediate_different_access_mod_shape()` test, with the
    exception that now full data is used in the second map.
    However, currently `MapFusion` only supports a single producer, thus this test can
    not run.
    """

    def ref(A, B):
        T = np.zeros((A.shape[0], 2))
        for i in range(A.shape[0]):
            T[i, 0] = A[i] * 2
            T[i, 1] = A[i] / 2
        for j in range(A.shape[0]):
            B[j] = np.sin(T[j, 1]) + np.cos(T[j, 0])

    sdfg = dace.SDFG("fusion_multiple_producers_consumers_sdfg")
    state = sdfg.add_state(is_start_block=True)
    for name in "AB":
        sdfg.add_array(
            name,
            shape=(10, ),
            dtype=dace.float64,
            transient=False,
        )
    sdfg.add_array(
        "T",
        shape=(10, 2),
        dtype=dace.float64,
        transient=True,
    )

    A, B, T = (state.add_access(name) for name in ["A", "B", "T"])

    me1, mx1 = state.add_map(
        "first_map",
        ndrange={"__i0": "0:10"},
    )

    state.add_edge(A, None, me1, "IN_A", dace.Memlet("A[0:10]"))
    me1.add_in_connector("IN_A")
    me1.add_out_connector("OUT_A")

    tsklt1_1 = state.add_tasklet(
        "tsklt1_1",
        inputs={"__in1"},
        outputs={"__out"},
        code="__out = __in1 * 2.0",
    )
    state.add_edge(me1, "OUT_A", tsklt1_1, "__in1", dace.Memlet("A[__i0]"))
    state.add_edge(tsklt1_1, "__out", mx1, "IN_T", dace.Memlet("T[__i0, 0]"))

    tsklt1_2 = state.add_tasklet(
        "tsklt1_2",
        inputs={"__in1"},
        outputs={"__out"},
        code="__out = __in1 / 2.0",
    )
    state.add_edge(me1, "OUT_A", tsklt1_2, "__in1", dace.Memlet("A[__i0]"))
    state.add_edge(tsklt1_2, "__out", mx1, "IN_T", dace.Memlet("T[__i0, 1]"))
    mx1.add_in_connector("IN_T")

    state.add_edge(
        mx1,
        "OUT_T",
        T,
        None,
        dace.Memlet("T[0:10, 0:2]"),
    )
    mx1.add_out_connector("OUT_T")

    state.add_mapped_tasklet(
        "comp2",
        map_ranges={"__i1": "0:10"},
        inputs={
            "__in1": dace.Memlet("T[__i1, 1]"),
            "__in2": dace.Memlet("T[__i1, 0]"),
        },
        code="__out = math.sin(__in1) + math.cos(__in2)",
        outputs={"__out": dace.Memlet("B[__i1]")},
        input_nodes={T},
        output_nodes={B},
        external_edges=True,
    )
    sdfg.validate()

    apply_fusion(sdfg, removed_maps=1, final_maps=1)

    args_ref = {
        'A': np.array(np.random.rand(10), dtype=np.float64, copy=True),
        'B': np.array(np.random.rand(10), dtype=np.float64, copy=True),
    }
    args_res = copy.deepcopy(args_ref)

    ref(**args_ref)
    sdfg(**args_res)
    for arg in args_ref.keys():
        arg_ref = args_ref[arg]
        arg_res = args_res[arg]
        assert np.allclose(arg_ref, arg_res)


def test_fusion_multiple_consumers():
    """The intermediate is consumed multiple times in the second map.
    """

    def ref(A, B, C):
        T = np.zeros_like(A)
        for i in range(A.shape[0]):
            T[i] = np.sin(A[i] * 2)
        for j in range(A.shape[0]):
            B[j] = T[j] * 3.
            C[j] = T[j] - 1.

    sdfg = dace.SDFG("fusion_multiple_consumers_sdfg")
    state = sdfg.add_state(is_start_block=True)
    for name in "ABCT":
        sdfg.add_array(
            name,
            shape=(10, ),
            dtype=dace.float64,
            transient=False,
        )
    sdfg.arrays["T"].transient = True

    A, B, C, T = (state.add_access(name) for name in ["A", "B", "C", "T"])

    state.add_mapped_tasklet(
        "comp1",
        map_ranges={"__i1": "0:10"},
        inputs={
            "__in1": dace.Memlet("A[__i1]"),
        },
        code="__out = math.sin(2 * __in1)",
        outputs={"__out": dace.Memlet("T[__i1]")},
        input_nodes={A},
        output_nodes={T},
        external_edges=True,
    )

    me2, mx2 = state.add_map(
        "second_map",
        ndrange={"__i0": "0:10"},
    )

    state.add_edge(T, None, me2, "IN_T", dace.Memlet("T[0:10]", volume=20))
    me2.add_in_connector("IN_T")
    me2.add_out_connector("OUT_T")

    tsklt2_1 = state.add_tasklet(
        "tsklt2_1",
        inputs={"__in1"},
        outputs={"__out"},
        code="__out = __in1 * 3.0",
    )
    state.add_edge(me2, "OUT_T", tsklt2_1, "__in1", dace.Memlet("T[__i0]"))
    state.add_edge(tsklt2_1, "__out", mx2, "IN_B", dace.Memlet("B[__i0]"))

    tsklt2_2 = state.add_tasklet(
        "tsklt2_2",
        inputs={"__in1"},
        outputs={"__out"},
        code="__out = __in1 - 1.0",
    )
    state.add_edge(me2, "OUT_T", tsklt2_2, "__in1", dace.Memlet("T[__i0]"))
    state.add_edge(tsklt2_2, "__out", mx2, "IN_C", dace.Memlet("C[__i0]"))
    mx2.add_in_connector("IN_B")
    mx2.add_in_connector("IN_C")

    state.add_edge(
        mx2,
        "OUT_B",
        B,
        None,
        dace.Memlet("B[0:10]"),
    )
    state.add_edge(
        mx2,
        "OUT_C",
        C,
        None,
        dace.Memlet("C[0:10]"),
    )
    mx2.add_out_connector("OUT_B")
    mx2.add_out_connector("OUT_C")
    sdfg.validate()

    apply_fusion(sdfg, removed_maps=1, final_maps=1)

    args_ref = {
        'A': np.array(np.random.rand(10), dtype=np.float64, copy=True),
        'B': np.array(np.random.rand(10), dtype=np.float64, copy=True),
        'C': np.array(np.random.rand(10), dtype=np.float64, copy=True),
    }
    args_res = copy.deepcopy(args_ref)

    ref(**args_ref)
    sdfg(**args_res)
    for arg in args_ref.keys():
        arg_ref = args_ref[arg]
        arg_res = args_res[arg]
        assert np.allclose(arg_ref, arg_res)


def test_fusion_different_global_accesses():

    def ref(A, B):
        T = np.zeros_like(A)
        for i in range(10):
            T[i] = A[i] - B[i + 1]
        for i in range(10):
            A[i] = np.sin(T[i])
            B[i + 1] = np.cos(T[i])

    sdfg = dace.SDFG("fusion_different_global_accesses_sdfg")
    state = sdfg.add_state(is_start_block=True)
    for name in "ABT":
        sdfg.add_array(
            name,
            shape=(11, ),
            dtype=dace.float64,
            transient=False,
        )
    sdfg.arrays["T"].transient = True
    T = state.add_access("T")

    state.add_mapped_tasklet(
        "first_comp",
        map_ranges={"__i0": "0:10"},
        inputs={
            "__in1": dace.Memlet("A[__i0]"),
            "__in2": dace.Memlet("B[__i0 + 1]")
        },
        code="__out = __in1 - __in2",
        outputs={"__out": dace.Memlet("T[__i0]")},
        output_nodes={T},
        external_edges=True,
    )
    state.add_mapped_tasklet(
        "second_comp",
        map_ranges={"__i4": "0:10"},
        inputs={"__in1": dace.Memlet("T[__i4]")},
        code="__out1 = math.sin(__in1)\n__out2 = math.cos(__in1)",
        outputs={
            "__out1": dace.Memlet("A[__i4]"),
            "__out2": dace.Memlet("B[__i4 + 1]"),
        },
        input_nodes={T},
        external_edges=True,
    )
    sdfg.validate()

    apply_fusion(sdfg, removed_maps=1, final_maps=1)

    args_ref = {
        'A': np.array(np.random.rand(11), dtype=np.float64, copy=True),
        'B': np.array(np.random.rand(11), dtype=np.float64, copy=True),
    }
    args_res = copy.deepcopy(args_ref)

    ref(**args_ref)
    sdfg(**args_res)
    for arg in args_ref.keys():
        arg_ref = args_ref[arg]
        arg_res = args_res[arg]
        assert np.allclose(arg_ref, arg_res)


def test_fusion_dynamic_producer():

    def ref(A, B):
        for i in range(10):
            if B[i] < 0.5:
                A[i] = 0.0
        for i in range(10):
            B[i] = np.sin(A[i])

    sdfg = dace.SDFG("fusion_dynamic_producer_sdfg")
    state = sdfg.add_state(is_start_block=True)
    for name in "AB":
        sdfg.add_array(
            name,
            shape=(10, ),
            dtype=dace.float64,
            transient=False,
        )
    B_top, B_bottom, A = (state.add_access(name) for name in "BBA")

    state.add_mapped_tasklet(
        "comp1",
        map_ranges={"__i0": "0:10"},
        inputs={"__in1": dace.Memlet("B[__i0]")},
        code="if __in1 < 0.5:\n\t__out = 0.0",
        outputs={"__out": dace.Memlet("A[__i0]", dynamic=True)},
        input_nodes={B_top},
        output_nodes={A},
        external_edges=True,
    )
    state.add_mapped_tasklet(
        "comp2",
        map_ranges={"__i3": "0:10"},
        inputs={"__in1": dace.Memlet("A[__i3]")},
        code="__out = math.sin(__in1)",
        outputs={"__out": dace.Memlet("B[__i3]")},
        input_nodes={A},
        output_nodes={B_bottom},
        external_edges=True,
    )
    sdfg.validate()

    # In case dynamic Memlets should be handled, we specify `unspecific`, i.e.
    #  only validation tests are done. However, we run a verification step to see
    #  if the transformation did the right thing.
    apply_fusion(sdfg, unspecific=True)

    args_ref = {
        'A': np.array(np.random.rand(11), dtype=np.float64, copy=True),
        'B': np.array(np.random.rand(11), dtype=np.float64, copy=True),
    }
    args_res = copy.deepcopy(args_ref)

    ref(**args_ref)
    csdfg = sdfg.compile()
    csdfg(**args_res)
    for arg in args_ref.keys():
        arg_ref = args_ref[arg]
        arg_res = args_res[arg]
        assert np.allclose(arg_ref, arg_res)


def test_fusion_intrinsic_memlet_direction():

    def ref(A, B):
        T = A + 10.0
        B[:] = np.sin(T)

    sdfg = dace.SDFG("fusion_dynamic_producer_sdfg")
    state = sdfg.add_state(is_start_block=True)

    for name in "ATB":
        sdfg.add_array(
            name,
            shape=(10, 11, 12),
            dtype=dace.float64,
            transient=False,
        )
    sdfg.arrays["T"].transient = True

    for num in "12":
        sdfg.add_scalar(
            "t" + num,
            dtype=dace.float64,
            transient=True,
        )

    A, T, B, t1, t2 = (state.add_access(name) for name in ["A", "T", "B", "t1", "t2"])

    tsklt1, me1, mx1 = state.add_mapped_tasklet(
        "comp1",
        map_ranges={
            "__i1": "0:10",
            "__i2": "0:11",
            "__i3": "0:12",
        },
        inputs={"__in1": dace.Memlet("A[__i1, __i2, __i3]")},
        code="__out = __in1 + 10.0",
        outputs={"__out": dace.Memlet("T[__i1, __i2, __i3]")},
        input_nodes={A},
        output_nodes={T},
        external_edges=True,
    )

    tsklt2, me2, mx2 = state.add_mapped_tasklet(
        "comp2",
        map_ranges={
            "__i1": "0:10",
            "__i2": "0:11",
            "__i3": "0:12",
        },
        inputs={"__in1": dace.Memlet("T[__i1, __i2, __i3]")},
        code="__out = math.sin(__in1)",
        outputs={"__out": dace.Memlet("B[__i1, __i2, __i3]")},
        input_nodes={T},
        output_nodes={B},
        external_edges=True,
    )

    for me in [me1, me2]:
        dace.transformation.dataflow.MapExpansion.apply_to(
            sdfg,
            options={"inner_schedule": dace.ScheduleType.Default},
            map_entry=me,
        )

    # Now add a transient scalar at the output of `tsklt1`.
    tsklt1_oedge = next(iter(state.out_edges(tsklt1)))
    me1_inner = tsklt1_oedge.dst
    state.add_edge(
        tsklt1,
        "__out",
        t1,
        None,
        dace.Memlet("t1[0]"),
    )
    state.add_edge(
        t1,
        None,
        me1_inner,
        tsklt1_oedge.dst_conn,
        dace.Memlet("t1[0] -> [__i1, __i2, __i3]"),
    )
    state.remove_edge(tsklt1_oedge)
    tsklt1_oedge = None

    # Now add a transient scalar in the front of `tsklt2`.
    tsklt2_iedge = next(iter(state.in_edges(tsklt2)))
    me2_inner = tsklt2_iedge.src
    state.add_edge(
        me2_inner,
        tsklt2_iedge.src_conn,
        t2,
        None,
        dace.Memlet("t2[0] -> [__i1, __i2, __i3]"),
    )
    state.add_edge(
        t2,
        None,
        tsklt2,
        "__in1",
        dace.Memlet("t2[0]"),
    )
    state.remove_edge(tsklt2_iedge)
    tsklt2_iedge = None
    sdfg.validate()

    # By Specifying `apply_once` we only perform one fusion, which will eliminate `T`.
    #  This is not efficient, we do this to make sure that the update of the Memlets
    #  has worked.
    apply_fusion(sdfg, apply_once=True)

    for edge in state.edges():
        # There should be no edge, that references `T`.
        assert edge.data.data != "T"

        # If an edge is connected to `t2` or `t1` then its data should refer to it.
        #  no other Memlet shall refer to them.
        for t in [t1, t2]:
            if edge.src is t or edge.dst is t:
                assert edge.data.data == t.data
            else:
                assert edge.data.data != t.data

    args_ref = {
        'A': np.array(np.random.rand(10, 11, 12), dtype=np.float64, copy=True),
        'B': np.array(np.random.rand(10, 11, 12), dtype=np.float64, copy=True),
    }
    args_res = copy.deepcopy(args_ref)

    ref(**args_ref)
    sdfg(**args_res)
    for arg in args_ref.keys():
        arg_ref = args_ref[arg]
        arg_res = args_res[arg]
        assert np.allclose(arg_ref, arg_res)


def _make_possible_cycle_if_fuesed_sdfg() -> Tuple[dace.SDFG, nodes.MapExit, nodes.AccessNode, nodes.MapEntry]:
    """Generate an SDFG that if two maps would be fused a cycle would be created.

    Essentially tests if the MapFusion detects this special case.
    """
    sdfg = dace.SDFG("possible_cycle_if_fuesed_sdfg")
    state = sdfg.add_state(is_start_block=True)

    names = ["A", "B", "T", "U", "V"]
    for name in names:
        sdfg.add_array(
            name,
            shape=(10, ),
            dtype=dace.float64,
            transient=True,
        )
    sdfg.arrays["A"].transient = False
    sdfg.arrays["B"].transient = False

    A, B, T, U, V = (state.add_access(name) for name in names)

    _, _, first_map_exit = state.add_mapped_tasklet(
        "map1",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("A[__i]")},
        code="__out1 = __in + 10\n__out2 = __in - 10",
        outputs={
            "__out1": dace.Memlet("T[__i]"),
            "__out2": dace.Memlet("U[__i]"),
        },
        input_nodes={A},
        output_nodes={T, U},
        external_edges=True,
    )

    state.add_mapped_tasklet(
        "map2",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("U[__i]")},
        code="__out = math.sin(__in)",
        outputs={"__out": dace.Memlet("V[__i]")},
        input_nodes={U},
        output_nodes={V},
        external_edges=True,
    )

    _, second_map_entry, _ = state.add_mapped_tasklet(
        "map3",
        map_ranges={"__i": "0:10"},
        inputs={
            "__in1": dace.Memlet("T[__i]"),
            "__in2": dace.Memlet("V[__i]"),
        },
        code="__out = __in1 + __in2",
        outputs={"__out": dace.Memlet("B[__i]")},
        input_nodes={T, V},
        output_nodes={B},
        external_edges=True,
    )
    sdfg.validate()

    return sdfg, first_map_exit, T, second_map_entry


def test_possible_cycle_if_fuesed_sdfg():
    sdfg, first_map_exit, array, second_map_entry = _make_possible_cycle_if_fuesed_sdfg()

    would_transformation_apply = MapFusion.can_be_applied_to(
        sdfg,
        first_map_exit=first_map_exit,
        array=array,
        second_map_entry=second_map_entry,
    )
    assert not would_transformation_apply


if __name__ == '__main__':
    test_fusion_intrinsic_memlet_direction()
    test_fusion_dynamic_producer()
    test_fusion_different_global_accesses()
    test_fusion_multiple_consumers()
    test_fusion_intermediate_different_access()
    test_fusion_intermediate_different_access_mod_shape()
    test_fusion_non_strict_dataflow_implicit_dependency()
    test_fusion_strict_dataflow_pointwise()
    test_fusion_strict_dataflow_not_pointwise()
    test_fusion_dataflow_intermediate()
    test_fusion_dataflow_intermediate_2()
    test_fusion_dataflow_intermediate_downstream()
    test_indirect_accesses()
    test_fusion_shared()
    test_fusion_with_transient()
    test_fusion_rename()
    test_fusion_simple()
    test_multiple_fusions()
    test_fusion_chain()
    test_fusion_with_transient_scalar()
    test_fusion_with_inverted_indices()
    test_fusion_with_empty_memlet()
    test_fusion_with_nested_sdfg_0()
    test_interstate_fusion()
    test_fusion_with_nested_sdfg_1()
    test_fuse_indirect_accesses()
    test_offset_correction_range_read()
    test_offset_correction_scalar_read()
    test_offset_correction_empty()
    test_different_offsets()
    test_inner_map_dependency()
    test_inner_map_dependency_resolved()
    test_possible_cycle_if_fuesed_sdfg()
