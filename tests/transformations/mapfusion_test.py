# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Any, Union

import numpy as np
import os
import dace
import copy

from dace import SDFG, SDFGState
from dace.sdfg import nodes
from dace.transformation.dataflow import MapFusionSerial, MapFusionParallel, MapExpansion


def count_node(sdfg: SDFG, node_type):
    nb_nodes = 0
    for rsdfg in sdfg.all_sdfgs_recursive():
        for state in sdfg.states():
            for node in state.nodes():
                if isinstance(node, node_type):
                    nb_nodes += 1
    return nb_nodes

def apply_fusion(
        sdfg: SDFG,
        removed_maps: Union[int, None] = None,
        final_maps: Union[int, None] = None,
) -> SDFG:
    """Applies the Map fusion transformation.

    The function checks that the number of maps has been reduced, it is also possible
    to specify the number of removed maps. It is also possible to specify the final
    number of maps.
    """
    num_maps_before = count_node(sdfg, nodes.MapEntry)
    org_sdfg = copy.deepcopy(sdfg)
    sdfg.apply_transformations_repeated(MapFusionSerial, validate=True, validate_all=True)
    num_maps_after = count_node(sdfg, nodes.MapEntry)

    has_processed = False
    if removed_maps is not None:
        has_processed = True
        rm = num_maps_before - num_maps_after
        if not (rm == removed_maps):
            sdfg.view()
        assert rm == removed_maps, f"Expected to remove {removed_maps} but removed {rm}"
    if final_maps is not None:
        has_processed = True
        if not (final_maps == num_maps_after):
            sdfg.view()
        assert final_maps == num_maps_after, f"Expected that only {final_maps} maps remain, but there are sill {num_maps_after}."
    if not has_processed:
        if not (num_maps_after < num_maps_before):
            sdfg.view()
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
            map_ranges={"__i0": "0:20", "__i1": "0:20"},
            inputs={"__in1": dace.Memlet("A[__i0, __i1]")},
            code="__out = __in1 + 20",
            outputs={"__out": dace.Memlet("B[__i0, __i1]")},
            input_nodes={"A": A1},
            output_nodes={"B": B1},
            external_edges=True,
    )
    state1.add_mapped_tasklet(
            "map_2_1",
            map_ranges={"__i0": "0:20", "__i1": "0:20"},
            inputs={"__in1": dace.Memlet("B[__i0, __i1]")},
            code="__out = __in1 + 10",
            outputs={"__out": dace.Memlet("C[__i0, __i1]")},
            input_nodes={"B": B1},
            output_nodes={"C": C1},
            external_edges=True,
    )

    B2, D2 = (state2.add_access(name) for name in ["B", "D"])
    state2.add_mapped_tasklet(
            "map_1_2",
            map_ranges={"__i0": "0:20", "__i1": "0:20"},
            inputs={"__in1": dace.Memlet("B[__i0, __i1]")},
            code="__out = __in1 + 6",
            outputs={"__out": dace.Memlet("D[__i0, __i1]")},
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
        sdfg.add_array("A",  (N,K), dace.float64)
        sdfg.add_array("B",  (N,), dace.float64)
        sdfg.add_array("T",  (N,), dace.float64, transient=True)
        t_node = state.add_access("T")
        sdfg.add_scalar("V",  dace.float64, transient=True)
        v_node = state.add_access("V")

        me1, mx1 = state.add_map("map1", dict(i=f"0:{N}"))
        tlet1 = state.add_tasklet("select", {"_v"}, {"_out"}, f"_out = _v[i, {K-1}]")
        state.add_memlet_path(state.add_access("A"), me1, tlet1, dst_conn="_v", memlet=dace.Memlet.from_array("A", sdfg.arrays["A"]))
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

    assert np.allclose(B, (A[:, K-1] + 1))


def test_fusion_with_inverted_indices():

    @dace.program
    def inverted_maps(A: dace.int32[10]):
        B = np.empty_like(A)
        for i in dace.map[0:10]:
            B[i] = i
        for i in dace.map[0:10]:
            A[9-i] = B[9-i] + 5
    
    ref = np.arange(5, 15, dtype=np.int32)

    sdfg = inverted_maps.to_sdfg(simplify=True)
    val0 = np.ndarray((10,), dtype=np.int32)
    sdfg(A=val0)
    assert np.array_equal(val0, ref)

    # This can not be fused
    apply_fusion(sdfg, removed_maps=0)

    val1 = np.ndarray((10,), dtype=np.int32)
    sdfg(A=val1)
    assert np.array_equal(val1, ref), f"REF: {ref}; VAL: {val1}"


def test_fusion_with_empty_memlet():

    N = dace.symbol('N', positive=True)

    @dace.program
    def inner_product(A: dace.float32[N], B: dace.float32[N], out: dace.float32[1]):
        tmp = np.empty_like(A)
        for i in dace.map[0:N:128]:
            for j in dace.map[0:128]:
                tmp[i+j] = A[i+j] * B[i+j]
        for i in dace.map[0:N:128]:
            lsum = dace.float32(0)
            for j in dace.map[0:128]:
                lsum = lsum + tmp[i+j]
            out[0] += lsum
    
    sdfg = inner_product.to_sdfg(simplify=True)
    apply_fusion(sdfg, removed_maps=2)

    A = np.arange(1024, dtype=np.float32)
    B = np.arange(1024, dtype=np.float32)
    val = np.zeros((1,), dtype=np.float32)
    sdfg(A=A, B=B, out=val, N=1024)
    ref = A @ B
    assert np.allclose(val[0], ref)


def test_fusion_with_nested_sdfg_0():
    
    @dace.program
    def fusion_with_nested_sdfg_0(A: dace.int32[10], B: dace.int32[10], C: dace.int32[10]):
        tmp = np.empty([10], dtype=np.int32)
        for i in dace.map[0:10]:
            if C[i] < 0:
                tmp[i] = B[i] - A[i]
            else:
                tmp[i] = B[i] + A[i]
        for i in dace.map[0:10]:
            A[i] = tmp[i] * 2
    
    sdfg = fusion_with_nested_sdfg_0.to_sdfg(simplify=True)

    # Because the transformation refuses to fuse dynamic edges.
    #  We have to eliminate them.
    for state in sdfg.states():
        for edge in state.edges():
            edge.data.dynamic = False
    apply_fusion(sdfg)

    for sd in sdfg.all_sdfgs_recursive():
        if sd is not sdfg:
            node = sd.parent_nsdfg_node
            state = sd.parent
            for e0 in state.out_edges(node):
                for e1 in state.memlet_tree(e0):
                    dst = state.memlet_path(e1)[-1].dst
                assert isinstance(dst, dace.nodes.AccessNode)


def test_fusion_with_nested_sdfg_1():
    
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

    # Because the transformation refuses to fuse dynamic edges.
    #  We have to eliminate them.
    for state in sdfg.states():
        for edge in state.edges():
            edge.data.dynamic = False
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

    assert sdfg.apply_transformations_repeated(MapFusionSerial, validate=True, validate_all=True) == 1
    assert sdfg.number_of_nodes() == 2
    assert len([node for node in state1.data_nodes() if node.data == "B"]) == 1

    sdfg(A=A, C=C, D=D)

    assert np.allclose(C, ref_C)
    assert np.allclose(D, ref_D)


def test_parallel_fusion_simple():
    N1, N2 = 10, 20

    def _make_sdfg():
        sdfg = dace.SDFG("simple_parallel_map")
        state = sdfg.add_state("state", is_start_block=True)
        for name in ("A", "B", "out1", "out2"):
            sdfg.add_array(name, shape=(N1, N2), transient=False, dtype=dace.float64)
        sdfg.add_scalar("dmr", dtype=dace.float64, transient=False)
        A, B, dmr, out1, out2 = (state.add_access(name) for name in ("A", "B", "dmr", "out1", "out2"))

        _, map1_entry, _ = state.add_mapped_tasklet(
                "map_with_dynamic_range",
                map_ranges={"__i0": f"0:{N1}", "__i1": f"0:{N2}"},
                inputs={"__in0": dace.Memlet("A[__i0, __i1]")},
                code="__out = __in0 + dynamic_range_value",
                outputs={"__out": dace.Memlet("out1[__i0, __i1]")},
                input_nodes={"A": A},
                output_nodes={"out1": out1},
                external_edges=True,
        )
        state.add_edge(
                dmr,
                None,
                map1_entry,
                "dynamic_range_value",
                dace.Memlet("dmr[0]"),
        )
        map1_entry.add_in_connector("dynamic_range_value")

        _, map2_entry, _ = state.add_mapped_tasklet(
                "map_without_dynamic_range",
                map_ranges={"__i2": f"0:{N1}", "__i3": f"0:{N2}"},
                inputs={
                    "__in0": dace.Memlet("A[__i2, __i3]"),
                    "__in1": dace.Memlet("B[__i2, __i3]")
                },
                code="__out = __in0 + __in1",
                outputs={"__out": dace.Memlet("out2[__i2, __i3]")},
                input_nodes={"A": A, "B": B},
                output_nodes={"out2": out2},
                external_edges=True,
        )
        sdfg.validate()
        return sdfg, map1_entry, map2_entry

    for mode in range(2):
        A = np.random.rand(N1, N2)
        B = np.random.rand(N1, N2)
        dmr = 3.1415
        out1 = np.zeros_like(A)
        out2 = np.zeros_like(B)
        res1 = A + dmr
        res2 = A + B

        sdfg, map1_entry, map2_entry = _make_sdfg()

        if mode:
            map1_entry, map2_entry = map2_entry, map1_entry

        MapFusionParallel.apply_to(
                sdfg,
                map_entry_1=map1_entry,
                map_entry_2=map2_entry,
                verify=True,
        )
        assert count_node(sdfg, dace.sdfg.nodes.MapEntry) == 1

        sdfg(A=A, B=B, dmr=dmr, out1=out1, out2=out2)
        assert np.allclose(out1, res1)
        assert np.allclose(out2, res2)


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
            map_ranges={"i": "0:20", "j": "2:8"},
            inputs={"__in1": dace.Memlet("A[i, j]")},
            code="__out = __in1 + 1.0",
            outputs={"__out": dace.Memlet("B[i, j]")},
            input_nodes={"A": A},
            output_nodes={"B": B},
            external_edges=True,
    )
    state.add_mapped_tasklet(
            "second_map",
            map_ranges=(
                {"i": "0:20", "k": "0:2"}
                if range_read
                else {"i": "0:20"}
            ),
            inputs={"__in1": dace.Memlet(f"B[i, {second_read_start}{'+k' if range_read else ''}]")},
            code="__out = __in1",
            outputs={"__out": dace.Memlet(f"C[i, {'k' if range_read else '0'}]")},
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


if __name__ == '__main__':
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
    test_parallel_fusion_simple()
    test_fuse_indirect_accesses()
    test_offset_correction_range_read()
    test_offset_correction_scalar_read()
    test_offset_correction_empty()
    print("SUCCESS")

