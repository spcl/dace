# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import re

import dace
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.interstate.loop_unroll import LoopUnroll
from dace.transformation.passes.prune_symbols import RemoveUnusedSymbols


def _get_sdfg(add_state_before: bool, l: int) -> dace.SDFG:
    sdfg = dace.SDFG("size_5_loop_sdfg")

    for_cfg = LoopRegion(label="size_5_loop",
                         condition_expr=CodeBlock(f"i < {l}"),
                         loop_var="i",
                         initialize_expr=CodeBlock("i = 0"),
                         update_expr=CodeBlock("i = i + 1"),
                         sdfg=sdfg)

    if add_state_before:
        _ps = sdfg.add_state(label="pre_s", is_start_block=True)
        sdfg.add_node(for_cfg, is_start_block=False)
        sdfg.add_edge(_ps, for_cfg, InterstateEdge(assignments={}))
    else:
        sdfg.add_node(for_cfg, is_start_block=True)

    body = ControlFlowRegion(label="for_body", sdfg=sdfg, parent=for_cfg)
    for_cfg.add_node(body, is_start_block=True)

    s1 = body.add_state(label="s1", is_start_block=True)

    c1 = ConditionalBlock(label="cond1", sdfg=sdfg, parent=body)
    c_body = ControlFlowRegion(label="if_body", sdfg=sdfg, parent=c1)

    c1.add_branch(condition=CodeBlock("a_sym > 0.0"), branch=c_body)

    body.add_node(c1, is_start_block=False)
    body.add_edge(s1, c1, InterstateEdge(assignments={"a_sym": "A[i]"}))

    s2 = c_body.add_state(label="s2", is_start_block=True)

    b_an = s2.add_access("B")
    a_an = s2.add_access("A")
    t = s2.add_tasklet(name="assign", inputs={"_in"}, outputs={"_out"}, code="_out = _in")
    s2.add_edge(t, "_out", b_an, None, Memlet(expr="B[i]"))
    s2.add_edge(a_an, None, t, "_in", Memlet(expr="A[i]"))

    sdfg.add_array("A", shape=(5, ), dtype=dace.float64)
    sdfg.add_array("B", shape=(5, ), dtype=dace.float64)

    sdfg.validate()
    return sdfg


def test_if_block_inside_for():
    sdfg = _get_sdfg(add_state_before=True, l=5)

    sdfg.apply_transformations_repeated(LoopUnroll, validate_all=True)

    loops = {n for n in sdfg.all_control_flow_regions() if isinstance(n, LoopRegion)}
    assert len(loops) == 0


def test_top_level_for():
    sdfg = _get_sdfg(add_state_before=False, l=5)

    sdfg.apply_transformations_repeated(LoopUnroll, validate_all=True)

    loops = {n for n in sdfg.all_control_flow_regions() if isinstance(n, LoopRegion)}
    assert len(loops) == 0


def test_empty_loop():
    sdfg = _get_sdfg(add_state_before=False, l=0)

    sdfg.apply_transformations_repeated(LoopUnroll, validate_all=True)

    loops = {n for n in sdfg.all_control_flow_regions() if isinstance(n, LoopRegion)}
    assert len(loops) == 0


def test_compiler_unroll_pragma():
    sdfg = _get_sdfg(add_state_before=False, l=5)
    code = sdfg.generate_code()[0].clean_code
    unroll_pragma = re.search(r'#pragma unroll', code) is None
    assert unroll_pragma, "Unroll pragma found in generated code."
    loops = {n for n in sdfg.all_control_flow_regions() if isinstance(n, LoopRegion)}
    assert len(loops) == 1
    loop = next(iter(loops))
    loop.unroll = True
    loop.unroll_factor = 5
    unrolled_loop_code = sdfg.generate_code()[0].clean_code
    unroll_pragma = re.search(r'#pragma unroll 5', unrolled_loop_code) is not None
    assert unroll_pragma, "Unroll pragma not found in generated code after setting unroll_pragma to True."


klev = dace.symbol('klev', dtype=dace.int32)
klon = dace.symbol('klon', dtype=dace.int32)
nclv = dace.symbol('nclv', dtype=dace.int32)


@dace.program
def melt_kernel(
    zqxfg: dace.float64[nclv, klon],
    zsolqa: dace.float64[nclv, nclv, klon],
    zmeltmax: dace.float64[klon],
    zicetot: dace.float64[klon],
    imelt: dace.int32[nclv],
):
    zepsec = 1e-14
    for jm in range(nclv):
        for jl in range(klon):
            if zmeltmax[jl] > zepsec and zicetot[jl] > zepsec:
                zalfa2 = zqxfg[jm, jl] / zicetot[jl]
                zmelt = min(zqxfg[jm, jl], zalfa2 * zmeltmax[jl])
                zqxfg[jm, jl] = zqxfg[jm, jl] - zmelt
                zqxfg[imelt[jm] - 1, jl] = zqxfg[imelt[jm] - 1, jl] + zmelt
                zsolqa[jm, imelt[jm] - 1, jl] = zsolqa[jm, imelt[jm] - 1, jl] + zmelt
                zsolqa[imelt[jm] - 1, jm, jl] = zsolqa[imelt[jm] - 1, jm, jl] - zmelt


@dace.program
def triang_elim_kernel(
    zqlhs: dace.float64[nclv, nclv, klon],
    kidia: dace.int32,
    kfdia: dace.int32,
):
    for jn in range(1, nclv - 1 + 1):
        for jm in range(jn + 1, nclv + 1):
            for ik in range(jn + 1, nclv + 1):
                for jl in range(kidia, kfdia + 1):
                    zqlhs[ik - 1, jm - 1, jl - 1] = (zqlhs[ik - 1, jm - 1, jl - 1] -
                                                     zqlhs[jn - 1, jm - 1, jl - 1] * zqlhs[ik - 1, jn - 1, jl - 1])


def test_triang_elim():
    KLON, NCLV = 512, 5
    KIDIA, KFDIA = 1, KLON

    sdfg = triang_elim_kernel.to_sdfg()
    sdfg.replace_dict({"nclv": NCLV})
    sdfg.validate()

    # Unroll the outer jn loop
    loops = {n for n in sdfg.nodes() if isinstance(n, LoopRegion)}
    assert len(loops) == 1
    loop = loops.pop()
    assert loop.loop_variable == "jn"

    LoopUnroll().apply_to(sdfg=sdfg, loop=loop)
    sdfg.validate()

    RemoveUnusedSymbols().apply_pass(sdfg, {})

    free_syms = {str(s) for s in sdfg.free_symbols}
    assert "jn" not in free_syms, f"Loop variable 'jn' still found in free symbols after unrolling: {free_syms}"
    sdfg.compile()


def test_melt_kernel():
    sdfg = melt_kernel.to_sdfg()
    sdfg.replace_dict({"nclv": 5})
    sdfg.validate()

    loops = {n for n in sdfg.nodes() if isinstance(n, LoopRegion)}
    assert len(loops) == 1
    loop = loops.pop()

    LoopUnroll().apply_to(sdfg=sdfg, loop=loop)
    sdfg.validate()

    # Test's aim to duplicate the case where a connection is missing
    src_nodes = {n for n in sdfg.nodes() if sdfg.in_degree(n) == 0}
    dst_nodes = {n for n in sdfg.nodes() if sdfg.out_degree(n) == 0}
    assert len(src_nodes) == 1
    assert len(dst_nodes) == 1

    other_nodes = set(sdfg.nodes()).difference((src_nodes.union(dst_nodes)))
    assert all({(sdfg.in_degree(n) == 1 and sdfg.out_degree(n) == 1) for n in other_nodes})


def test_replace_dict_inner_loop():
    nclv = dace.symbol('nclv', dtype=dace.int32)
    klon = dace.symbol('klon', dtype=dace.int32)

    @dace.program
    def nested_loop_kernel(A: dace.float64[nclv, klon], jn: dace.int32):
        for jm in range(jn, nclv + 1):
            for jl in range(klon):
                A[jm, jl] = A[jm, jl] + A[jn, jl]

    sdfg = nested_loop_kernel.to_sdfg()
    sdfg.replace_dict({"jn": "1"})
    sdfg.validate()
    sdfg.remove_data("jn")
    sdfg.validate()
    assert "jn" not in {str(s) for s in sdfg.arglist()}


if __name__ == "__main__":
    test_if_block_inside_for()
    test_empty_loop()
    test_top_level_for()
    test_compiler_unroll_pragma()
    test_melt_kernel()
    test_triang_elim()
    test_replace_dict_inner_loop()
