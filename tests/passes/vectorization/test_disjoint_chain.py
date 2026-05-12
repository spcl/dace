# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import copy
from typing import Tuple
import dace
import pytest
import numpy
from dace.sdfg.state import ConditionalBlock
from dace.transformation.interstate import branch_elimination
from dace.transformation.passes.vectorization.vectorize_cpu import VectorizeCPU
from tests.passes.vectorization._harness import (
    N,
    C,
    _get_disjoint_chain_sdfg,
    _get_disjoint_chain_sdfg_two,
)


@pytest.mark.parametrize("trivial_if_demote_symbols", [(True, True), (True, False), (False, True), (False, False)])
def test_disjoint_chain_split_branch_only(trivial_if_demote_symbols: Tuple[bool, bool], branch_mode):
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

    copy_sdfg(**out_fused)

    for name in arrays.keys():
        numpy.testing.assert_allclose(out_no_fuse[name], out_fused[name], atol=1e-12)


def test_disjoint_chain_with_overlapping_region_fusion(branch_mode):
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

    copy_sdfg = copy.deepcopy(sdfg)
    copy_sdfg.name = sdfg.name + "_vectorized"
    arrays = {"zsolqa": zsolqa, "zrainacc": zrainacc, "zrainaut": zrainaut, "ztp1": ztp1, "rtt": rtt[0], "N": _N[0]}

    sdfg.validate()
    out_no_fuse = {k: v.copy() for k, v in arrays.items()}
    sdfg(**out_no_fuse)
    # Run SDFG version (with transformation)
    branch_kwargs = {"use_fp_factor": False, "branch_normalization": True} if branch_mode == "merge" else {}
    VectorizeCPU(vector_width=8,
                 insert_copies=True,
                 fuse_overlapping_loads=True,
                 fail_on_unvectorizable=True,
                 **branch_kwargs).apply_pass(copy_sdfg, {})

    out_fused = {k: v.copy() for k, v in arrays.items()}
    copy_sdfg.validate()

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


def test_disjoint_chain(branch_mode):
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

    copy_sdfg = copy.deepcopy(sdfg)
    copy_sdfg.name = sdfg.name + "_vectorized"
    arrays = {"zsolqa": zsolqa, "zrainacc": zrainacc, "zrainaut": zrainaut, "ztp1": ztp1, "rtt": rtt[0], "N": _N[0]}

    sdfg.validate()
    out_no_fuse = {k: v.copy() for k, v in arrays.items()}
    sdfg(**out_no_fuse)
    # Run SDFG version (with transformation)
    branch_kwargs = {"use_fp_factor": False, "branch_normalization": True} if branch_mode == "merge" else {}
    VectorizeCPU(vector_width=8,
                 insert_copies=True,
                 fuse_overlapping_loads=False,
                 fail_on_unvectorizable=True,
                 **branch_kwargs).apply_pass(copy_sdfg, {})

    out_fused = {k: v.copy() for k, v in arrays.items()}
    copy_sdfg.validate()

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
