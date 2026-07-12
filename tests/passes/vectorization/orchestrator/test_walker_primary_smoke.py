# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Walker-primary pipeline smoke tests (post-descent migration).

After the legacy ``PromoteNSDFGBodyToTiles`` + ``EmitTileOps`` descent
was deleted, ``VectorizeCPUMultiDim`` runs a walker-primary pipeline:
``MarkTileDims`` -> ``GenerateTileIterationMask`` ->
``StrideMapByTileWidths`` -> ``PreparePerLaneIndices`` ->
``InsertTileLoadStore`` -> lib-node expansion -> ``ClearPerLaneIndexSymbols``.

This file pins minimum-viable invariants on the walker-primary
orchestrator -- it imports, instantiates, and runs without crashing
on a trivial SDFG. Numerical equivalence end-to-end will land once
the walker handles tasklet -> TileBinop / TileITE / TileReduce
conversion (currently in scope).
"""
import dace
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import (VectorizeCPUMultiDim)
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import ISA, RemainderStrategy


def test_orchestrator_runs_on_empty_sdfg():
    """An empty SDFG triggers no pipeline rewrites; orchestrator returns cleanly."""
    sdfg = dace.SDFG("empty")
    sdfg.add_state("s")
    VectorizeCPUMultiDim(VectorizeConfig(widths=(8, ), target_isa=ISA.SCALAR)).apply_pass(sdfg, {})


def test_orchestrator_K1_runs_on_trivial_array_copy_kernel():
    """K=1 orchestrator over a minimal array-copy kernel (B[i] = A[i]) passes cleanly."""
    sdfg = dace.SDFG("copy_k1")
    sdfg.add_array("A", (16, ), dace.float64, transient=False)
    sdfg.add_array("B", (16, ), dace.float64, transient=False)
    state = sdfg.add_state("s")
    me, mx = state.add_map("k", {"ii": "0:16"})
    a = state.add_access("A")
    b = state.add_access("B")
    tasklet = state.add_tasklet("body", {"_a"}, {"_b"}, "_b = _a")
    state.add_memlet_path(a, me, tasklet, dst_conn="_a", memlet=dace.Memlet("A[ii]"))
    state.add_memlet_path(tasklet, mx, b, src_conn="_b", memlet=dace.Memlet("B[ii]"))
    VectorizeCPUMultiDim(VectorizeConfig(widths=(8, ), target_isa=ISA.SCALAR)).apply_pass(sdfg, {})


def test_orchestrator_K2_runs_on_trivial_2d_copy_kernel():
    """K=2 orchestrator over a 2-D copy kernel (B[i, j] = A[i, j]) passes cleanly."""
    sdfg = dace.SDFG("copy_k2")
    sdfg.add_array("A", (16, 32), dace.float64, transient=False)
    sdfg.add_array("B", (16, 32), dace.float64, transient=False)
    state = sdfg.add_state("s")
    me, mx = state.add_map("k", {"ii": "0:16", "jj": "0:32"})
    a = state.add_access("A")
    b = state.add_access("B")
    tasklet = state.add_tasklet("body", {"_a"}, {"_b"}, "_b = _a")
    state.add_memlet_path(a, me, tasklet, dst_conn="_a", memlet=dace.Memlet("A[ii, jj]"))
    state.add_memlet_path(tasklet, mx, b, src_conn="_b", memlet=dace.Memlet("B[ii, jj]"))
    VectorizeCPUMultiDim(VectorizeConfig(widths=(4, 8), target_isa=ISA.SCALAR)).apply_pass(sdfg, {})


def test_orchestrator_constructor_refuses_invalid_widths():
    """Pipeline-level knob validation still loud-fails on bad widths."""
    import pytest
    # K=0 (empty widths) -- the first check that fires depends on the validator's order.
    # Just verify NotImplementedError is raised.
    with pytest.raises((NotImplementedError, IndexError)):
        VectorizeCPUMultiDim(VectorizeConfig(widths=(), target_isa=ISA.SCALAR))
    with pytest.raises(NotImplementedError):
        VectorizeCPUMultiDim(VectorizeConfig(widths=(8, 8, 8, 8), target_isa=ISA.SCALAR))


def test_orchestrator_supports_branch_modes():
    """Both ``merge`` (default) and ``fp_factor`` branch modes still construct cleanly."""
    sdfg = dace.SDFG("branch_modes")
    sdfg.add_state("s")
    for branch_mode in ("merge", "fp_factor"):
        # fp_factor requires K=1 + scalar_postamble; merge accepts any combo.
        if branch_mode == "fp_factor":
            VectorizeCPUMultiDim(
                VectorizeConfig(widths=(8, ),
                                target_isa=ISA.SCALAR,
                                branch_mode=branch_mode,
                                remainder_strategy=RemainderStrategy.SCALAR_POSTAMBLE)).apply_pass(sdfg, {})
        else:
            VectorizeCPUMultiDim(VectorizeConfig(widths=(8, ), target_isa=ISA.SCALAR,
                                                 branch_mode=branch_mode)).apply_pass(sdfg, {})
