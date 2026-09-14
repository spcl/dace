# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A temporary each branch recomputes before reading is block-local, so branch lowering never merges it.

Two guarded blocks both write ``t`` and read it back. Counting the second block's read as an escape of the first
block's write made the merge emit ``t = ITE(c, t_then, t)``, reading a ``t`` no path had written.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")

import copy
import warnings

import numpy as np

import dace
from dace.libraries.tileops._dispatch import detect_host_isa
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.vectorization import VectorizeCPUMultiDim
from dace.transformation.passes.vectorization.branch_normalization import BranchNormalization
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.lower_interstate_conditional_assignments_to_tasklets import (
    LowerInterstateConditionalAssignmentsToTasklets)
from dace.transformation.passes.vectorization.same_write_set_if_else_to_ite_cfg import SameWriteSetIfElseToITECFG
from tests.passes.vectorization.tile_assertions import tile_library_nodes

N = dace.symbol("N", dtype=dace.int64)
LENGTH = 67


@dace.program
def two_guarded_blocks_sharing_a_temporary(c: dace.float64[N], x: dace.float64[N], y: dace.float64[N],
                                           out: dace.float64[N]):
    for i in dace.map[0:N]:
        if c[i] > 0.5:
            t = x[i] - y[i]
            out[i] = out[i] + t * 2.0
        if c[i] < -0.5:
            t = x[i] - y[i]
            out[i] = out[i] + t * 3.0


def inputs() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(3)
    return {name: rng.uniform(-1.0, 1.0, LENGTH) for name in ("c", "x", "y", "out")}


def canonical_sdfg(name: str) -> dace.SDFG:
    sdfg = two_guarded_blocks_sharing_a_temporary.to_sdfg(simplify=False)
    sdfg.name = name
    canonicalize(sdfg, validate=True)
    return sdfg


def test_branch_lowering_merges_only_writes_that_outlive_their_block():
    sdfg = canonical_sdfg("block_local_temporary_lowering")

    for lowering in (LowerInterstateConditionalAssignmentsToTasklets(), SameWriteSetIfElseToITECFG(),
                     BranchNormalization()):
        lowering.apply_pass(sdfg, {})

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sdfg.validate()
    merged = [
        edge.dst.data for node, state in sdfg.all_nodes_recursive()
        if isinstance(node, nodes.Tasklet) and "ITE(" in node.code.as_string for edge in state.out_edges(node)
    ]
    assert len(merged) == 2 and len(set(merged)) == 1, merged


def test_vectorized_blocks_sharing_a_temporary_match_the_scalar_program():
    reference = two_guarded_blocks_sharing_a_temporary.to_sdfg(simplify=False)
    reference.name = "block_local_temporary_reference"
    want = inputs()
    reference.compile()(**want, N=LENGTH)
    sdfg = canonical_sdfg("block_local_temporary_vectorized")

    VectorizeCPUMultiDim(VectorizeConfig(widths=(8, ), target_isa=detect_host_isa(),
                                         validate=True)).apply_pass(sdfg, {})

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sdfg.validate()
    assert tile_library_nodes(sdfg), "the vectorizer emitted no tile ops"
    got = copy.deepcopy(inputs())
    sdfg.compile()(**got, N=LENGTH)
    np.testing.assert_allclose(got["out"], want["out"], rtol=1e-12, atol=0)
