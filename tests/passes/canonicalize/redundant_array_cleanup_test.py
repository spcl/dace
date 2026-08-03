# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The canonicalization pipeline's post-fuse redundant-array cleanup stage.

heat3d's second sweep writes an ``(N-2)^3`` transient that is then copied wholesale into
``A[1:-1, 1:-1, 1:-1]``. The pipeline's last ``SimplifyPass`` runs BEFORE the terminal
LoopToMap/fuse, and ``ArrayElimination`` (Simplify's array reclaimer) refuses the shape
anyway -- its WAR-carrier guard skips any candidate whose destination is read and written
in the same state, which every in-place stencil sweep is -- so the buffer used to reach
codegen as a heap allocation plus a full-size copy loop per timestep.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

from typing import List, Tuple

import numpy as np

import dace
from dace.sdfg import nodes
from dace.transformation import pass_pipeline as ppl
from dace.transformation.dataflow.redundant_array import RedundantArray
from dace.transformation.passes.canonicalize import pipeline as canon_pipeline
from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated

N = dace.symbol('N')


@dace.program
def _heat3d(TSTEPS: dace.int64, A: dace.float64[N, N, N], B: dace.float64[N, N, N]):
    """The npbench/polybench formulation, verbatim -- see ``tests/corpus/polybench/stencils/heat_3d.py``."""
    for _ in range(1, TSTEPS):
        B[1:-1, 1:-1,
          1:-1] = (0.125 * (A[2:, 1:-1, 1:-1] - 2.0 * A[1:-1, 1:-1, 1:-1] + A[:-2, 1:-1, 1:-1]) + 0.125 *
                   (A[1:-1, 2:, 1:-1] - 2.0 * A[1:-1, 1:-1, 1:-1] + A[1:-1, :-2, 1:-1]) + 0.125 *
                   (A[1:-1, 1:-1, 2:] - 2.0 * A[1:-1, 1:-1, 1:-1] + A[1:-1, 1:-1, 0:-2]) + A[1:-1, 1:-1, 1:-1])
        A[1:-1, 1:-1,
          1:-1] = (0.125 * (B[2:, 1:-1, 1:-1] - 2.0 * B[1:-1, 1:-1, 1:-1] + B[:-2, 1:-1, 1:-1]) + 0.125 *
                   (B[1:-1, 2:, 1:-1] - 2.0 * B[1:-1, 1:-1, 1:-1] + B[1:-1, :-2, 1:-1]) + 0.125 *
                   (B[1:-1, 1:-1, 2:] - 2.0 * B[1:-1, 1:-1, 1:-1] + B[1:-1, 1:-1, 0:-2]) + B[1:-1, 1:-1, 1:-1])


def _heat3d_numpy(tsteps: int, A: np.ndarray, B: np.ndarray) -> None:
    """Independent oracle: plain numpy, same operand order, updated in place."""
    for _ in range(1, tsteps):
        B[1:-1, 1:-1,
          1:-1] = (0.125 * (A[2:, 1:-1, 1:-1] - 2.0 * A[1:-1, 1:-1, 1:-1] + A[:-2, 1:-1, 1:-1]) + 0.125 *
                   (A[1:-1, 2:, 1:-1] - 2.0 * A[1:-1, 1:-1, 1:-1] + A[1:-1, :-2, 1:-1]) + 0.125 *
                   (A[1:-1, 1:-1, 2:] - 2.0 * A[1:-1, 1:-1, 1:-1] + A[1:-1, 1:-1, 0:-2]) + A[1:-1, 1:-1, 1:-1])
        A[1:-1, 1:-1,
          1:-1] = (0.125 * (B[2:, 1:-1, 1:-1] - 2.0 * B[1:-1, 1:-1, 1:-1] + B[:-2, 1:-1, 1:-1]) + 0.125 *
                   (B[1:-1, 2:, 1:-1] - 2.0 * B[1:-1, 1:-1, 1:-1] + B[1:-1, :-2, 1:-1]) + 0.125 *
                   (B[1:-1, 1:-1, 2:] - 2.0 * B[1:-1, 1:-1, 1:-1] + B[1:-1, 1:-1, 0:-2]) + B[1:-1, 1:-1, 1:-1])


def _cleanup_stage() -> PatternMatchAndApplyRepeated:
    """A fresh instance of exactly the stage under test."""
    return PatternMatchAndApplyRepeated([RedundantArray()])


def _is_cleanup_stage(unit: ppl.Pass) -> bool:
    if not isinstance(unit, PatternMatchAndApplyRepeated):
        return False
    return any(isinstance(t, RedundantArray) for t in unit.transformations)


def _canonicalize(sdfg: dace.SDFG, with_cleanup: bool) -> dace.SDFG:
    """Run the real recipe, optionally with the cleanup stage skipped (the A/B reference)."""
    canon_pipeline.disable_openmp_sections(sdfg)
    for _label, unit in canon_pipeline._build_stages():
        if not with_cleanup and _is_cleanup_stage(unit):
            continue
        unit.apply_pass(sdfg, {})
    canon_pipeline.disable_openmp_sections(sdfg)
    sdfg.validate()
    return sdfg


def _heap_buffers(sdfg: dace.SDFG) -> List[Tuple[str, str]]:
    """Transients that are a real allocation, not a register (total size 1)."""
    return sorted((name, str(desc.shape)) for nested in sdfg.all_sdfgs_recursive()
                  for name, desc in nested.arrays.items() if desc.transient and desc.total_size != 1)


def _bulk_copies(sdfg: dace.SDFG) -> int:
    """AccessNode -> AccessNode edges: a container-to-container copy loop at codegen."""
    return sum(1 for nested in sdfg.all_sdfgs_recursive() for state in nested.states() for e in state.edges()
               if isinstance(e.src, nodes.AccessNode) and isinstance(e.dst, nodes.AccessNode))


def _shape(sdfg: dace.SDFG) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, str, str, str]]]:
    """Everything the cleanup could legally touch, for an exact no-op comparison."""
    arrays = sorted((nested.label, name) for nested in sdfg.all_sdfgs_recursive() for name in nested.arrays)
    edges = sorted((nested.label, state.label, str(e.src), str(e.dst), str(e.data))
                   for nested in sdfg.all_sdfgs_recursive() for state in nested.states() for e in state.edges())
    return arrays, edges


def test_the_stage_is_wired_in_once_after_the_terminal_fuse():
    labels = [label for label, unit in canon_pipeline._build_stages() if _is_cleanup_stage(unit)]
    assert labels == ['end'], labels


def test_heat3d_loses_the_copy_out_buffer():
    """A/B in one test, so it can never go vacuous: without the stage the buffer is there."""
    reference = _canonicalize(_heat3d.to_sdfg(simplify=True), with_cleanup=False)
    assert len(_heap_buffers(reference)) == 1, _heap_buffers(reference)
    assert _bulk_copies(reference) == 1

    cleaned = _canonicalize(_heat3d.to_sdfg(simplify=True), with_cleanup=True)
    assert _heap_buffers(cleaned) == [], _heap_buffers(cleaned)
    assert _bulk_copies(cleaned) == 0


def test_the_stage_is_idempotent():
    """Re-entering the stage must find nothing: canonicalize is already short of a fixed point."""
    sdfg = _canonicalize(_heat3d.to_sdfg(simplify=True), with_cleanup=True)
    before = _shape(sdfg)
    assert _cleanup_stage().apply_pass(sdfg, {}) is None
    assert _shape(sdfg) == before
    sdfg.validate()


def test_heat3d_is_bit_exact():
    """A cleanup that changes results is a miscompile: check against the same pipeline without the
    stage AND against an independent numpy oracle."""
    size, tsteps = 24, 5
    rng = np.random.default_rng(4711)
    base_a, base_b = rng.random((size, ) * 3), rng.random((size, ) * 3)

    results = []
    for with_cleanup in (False, True):
        sdfg = _canonicalize(_heat3d.to_sdfg(simplify=True), with_cleanup)
        sdfg.name = 'heat3d_rra_cleanup' if with_cleanup else 'heat3d_rra_reference'
        a, b = base_a.copy(), base_b.copy()
        sdfg.compile()(TSTEPS=tsteps, A=a, B=b, N=size)
        results.append((a, b))

    oracle_a, oracle_b = base_a.copy(), base_b.copy()
    _heat3d_numpy(tsteps, oracle_a, oracle_b)

    for i, name in ((0, 'A'), (1, 'B')):
        assert np.array_equal(results[1][i].view(np.uint64), results[0][i].view(np.uint64)), \
            'vs the same pipeline without the stage: ' + name
        assert np.array_equal(results[1][i].view(np.uint64), (oracle_a, oracle_b)[i].view(np.uint64)), \
            'vs numpy oracle: ' + name


if __name__ == '__main__':
    test_the_stage_is_wired_in_once_after_the_terminal_fuse()
    test_heat3d_loses_the_copy_out_buffer()
    test_the_stage_is_idempotent()
    test_heat3d_is_bit_exact()
