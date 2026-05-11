# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import math
import copy
from typing import Tuple
import dace
import pytest
import numpy
from dace import InterstateEdge
from dace import Union
from dace.properties import CodeBlock
from dace.sdfg import ControlFlowRegion
from dace.sdfg.state import ConditionalBlock
from dace.transformation.interstate import branch_elimination
from dace.transformation.passes.vectorization.tasklet_preprocessing_passes import (
    ReplaceSTDExpWithDaCeExp, ReplaceSTDLogWithDaCeLog, ReplaceSTDPowWithDaCePow,
)
from dace.transformation.passes.vectorization.vectorize_cpu import VectorizeCPU
from tests.passes.vectorization._harness import (
    run_vectorization_test,
    N, S, S1, S2, klev, kidia, kfdia, n, m, nnz,
    KLON, KLEV, NCLDQL, NCLDQI, ssym, X, Y, C,
    log, exp, pow,
    _get_disjoint_chain_sdfg, _get_disjoint_chain_sdfg_two,
    _get_cloudsc_snippet_three, _get_cloudsc_snippet_four,
    _get_map_inside_nested_map,
    _get_dependency_edge_to_unary_symbol_sdfg,
    _get_unstructured_access_cloudsc_sdfg,
)

@pytest.mark.parametrize("trivial_if_demote_symbols", [(True, True), (True, False), (False, True), (False, False)])
def test_disjoint_chain_split_branch_only(trivial_if_demote_symbols: Tuple[bool, bool], request, branch_mode):
    if branch_mode == "merge":
        request.applymarker(
            pytest.mark.xfail(reason="merge mode coverage pending follow-up; track as TODO"))
    trivial_if, demote_symbols = trivial_if_demote_symbols
    sdfg, nsdfg_parent_state = _get_disjoint_chain_sdfg(trivial_if)
    zsolqa = numpy.random.choice([0.001, 5.0], size=(C, 5, 5))
    zrainacc = numpy.random.choice([0.001, 5.0], size=(C, ))
    zrainaut = numpy.random.choice([0.001, 5.0], size=(C, ))
    ztp1 = numpy.random.choice([3.5, 5.0], size=(C, ))
    rtt = numpy.random.choice([4.0], size=(1, ))

    sdfg.name = f"{sdfg.name}_{str(trivial_if).lower()}_{str(demote_symbols).lower()}"
    copy_sdfg = copy.deepcopy(sdfg)
    copy_sdfg.name = f"{copy_sdfg.name}_{str(trivial_if).lower()}_{str(demote_symbols).lower()}_vectorized"

    arrays = {"zsolqa": zsolqa, "zrainacc": zrainacc, "zrainaut": zrainaut, "ztp1": ztp1, "rtt": rtt[0]}

    sdfg.validate()
    sdfg.save(f"disjoint_chain_{str(trivial_if).lower()}_{str(demote_symbols).lower()}.sdfg")
    out_no_fuse = {k: v.copy() for k, v in arrays.items()}
    sdfg(**out_no_fuse)

    # Run SDFG version (with transformation)
    if trivial_if:
        from dace.transformation.passes.lift_trivial_if import LiftTrivialIf
        LiftTrivialIf().apply_pass(copy_sdfg, {})
    else:
        xform = branch_elimination.BranchElimination()
        cblocks = {n for n, g in copy_sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)}
        assert len(cblocks) == 1
        cblock = cblocks.pop()

        xform.conditional = cblock
        xform.parent_nsdfg_state = nsdfg_parent_state
        xform.sequentialize_if_else_branch_if_disjoint_subsets(cblock.parent_graph)

    out_fused = {k: v.copy() for k, v in arrays.items()}

    branch_kwargs = {"use_fp_factor": False, "branch_normalization": True} if branch_mode == "merge" else {}
    vectorizer = VectorizeCPU(vector_width=8, fail_on_unvectorizable=True, **branch_kwargs)
    vectorizer.try_to_demote_symbols_in_nsdfgs = demote_symbols
    vectorizer.apply_pass(copy_sdfg, {})
    copy_sdfg.save("disjoint_chain_vectorized.sdfg")

    copy_sdfg(**out_fused)

    for name in arrays.keys():
        numpy.testing.assert_allclose(out_no_fuse[name], out_fused[name], atol=1e-12)


def test_disjoint_chain_with_overlapping_region_fusion(request, branch_mode):
    if branch_mode == "merge":
        request.applymarker(
            pytest.mark.xfail(reason="merge mode coverage pending follow-up; track as TODO"))
    sdfg, nsdfg_parent_state = _get_disjoint_chain_sdfg_two()
    sdfg.name = f"disjoint_chain_split_two_rtt_val_4_2_with_overlapping_region_fusion"
    _N = 64
    zsolqa = numpy.random.choice([0.001, 5.0], size=(5, 5, _N))
    zrainacc = numpy.random.choice([0.001, 5.0], size=(_N, ))
    zrainaut = numpy.random.choice([0.001, 5.0], size=(_N, ))
    ztp1 = numpy.random.choice([3.5, 5.0], size=(_N, ))
    rtt = numpy.array([4.2], numpy.float64)
    _N = numpy.array([64], numpy.int64)
    sdfg.validate()
    sdfg.save(f"{sdfg.name}.sdfgz", compress=True)

    copy_sdfg = copy.deepcopy(sdfg)
    copy_sdfg.name = sdfg.name + "_vectorized"
    arrays = {"zsolqa": zsolqa, "zrainacc": zrainacc, "zrainaut": zrainaut, "ztp1": ztp1, "rtt": rtt[0], "N": _N[0]}

    sdfg.validate()
    out_no_fuse = {k: v.copy() for k, v in arrays.items()}
    sdfg(**out_no_fuse)
    # Run SDFG version (with transformation)
    branch_kwargs = {"use_fp_factor": False, "branch_normalization": True} if branch_mode == "merge" else {}
    VectorizeCPU(vector_width=8, insert_copies=True, fuse_overlapping_loads=True,
                 fail_on_unvectorizable=True, **branch_kwargs).apply_pass(copy_sdfg, {})

    out_fused = {k: v.copy() for k, v in arrays.items()}
    copy_sdfg.validate()
    copy_sdfg.save(f"{copy_sdfg.name}.sdfgz", compress=True)

    # There is should be no `_union` access nodes

    access_nodes_of_unions = set()
    for state in copy_sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.AccessNode):
                if node.data.endswith("_union"):
                    access_nodes_of_unions.add(node)
    assert len(access_nodes_of_unions) == 0

    copy_sdfg(**out_fused)

    for name in arrays.keys():
        numpy.testing.assert_allclose(out_no_fuse[name], out_fused[name], atol=1e-12)


def test_disjoint_chain(request, branch_mode):
    if branch_mode == "merge":
        request.applymarker(
            pytest.mark.xfail(reason="merge mode coverage pending follow-up; track as TODO"))
    sdfg, nsdfg_parent_state = _get_disjoint_chain_sdfg_two()
    sdfg.name = f"disjoint_chain"
    _N = 64
    zsolqa = numpy.random.choice([0.001, 5.0], size=(5, 5, _N))
    zrainacc = numpy.random.choice([0.001, 5.0], size=(_N, ))
    zrainaut = numpy.random.choice([0.001, 5.0], size=(_N, ))
    ztp1 = numpy.random.choice([3.5, 5.0], size=(_N, ))
    rtt = numpy.array([4.2], numpy.float64)
    _N = numpy.array([64], numpy.int64)
    sdfg.validate()
    sdfg.save(f"{sdfg.name}.sdfgz", compress=True)

    copy_sdfg = copy.deepcopy(sdfg)
    copy_sdfg.name = sdfg.name + "_vectorized"
    arrays = {"zsolqa": zsolqa, "zrainacc": zrainacc, "zrainaut": zrainaut, "ztp1": ztp1, "rtt": rtt[0], "N": _N[0]}

    sdfg.validate()
    out_no_fuse = {k: v.copy() for k, v in arrays.items()}
    sdfg(**out_no_fuse)
    # Run SDFG version (with transformation)
    branch_kwargs = {"use_fp_factor": False, "branch_normalization": True} if branch_mode == "merge" else {}
    VectorizeCPU(vector_width=8, insert_copies=True, fuse_overlapping_loads=False,
                 fail_on_unvectorizable=True, **branch_kwargs).apply_pass(copy_sdfg, {})

    out_fused = {k: v.copy() for k, v in arrays.items()}
    copy_sdfg.validate()
    copy_sdfg.save(f"{copy_sdfg.name}.sdfgz", compress=True)

    for state in copy_sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.AccessNode):
                if "zrainacc_vec" in node.data or "zrainaut_vec" in node.data:
                    for ie in state.in_edges(node):
                        assert ie.data.subset != dace.subsets.Range([(0, N - 1, 1)])

    copy_sdfg(**out_fused)

    for name in sorted(arrays.keys()):
        print(f"Compare {name}")
        numpy.testing.assert_allclose(out_no_fuse[name], out_fused[name], atol=1e-12)
        print(f"{name} OK")

