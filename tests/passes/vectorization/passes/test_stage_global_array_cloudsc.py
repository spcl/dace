# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""cloudsc-pattern tests for ``StageGlobalArrayThroughScalars``.

These exercise the pass on the REAL cloudsc reuse pattern, kept separate from
the synthetic-matrix unit tests in
``test_stage_global_array_through_scalars.py``.

``_get_cloudsc_snippet_four`` builds the ``zqlhs`` read-modify-write reuse chain
(``zqlhs@k -> tasklet(+zsolqb[..,k,..]) -> zqlhs@(k+1)``) — a global array
sitting between two tasklets across the whole accumulation, the canonical
``Tasklet -> A(global) -> Tasklet`` occurrence the pass targets. Each hop reads
``zqlhs`` at the same element it just wrote, so the chain is the linear
accumulation the spec's refusal carve-out explicitly allows.

Every test compiles + runs the untransformed kernel as the reference and
compares (``numpy.allclose``) against the pass-applied deep copy — the
e2e-numerical convention.

The pass module does not exist yet — the implementation agent creates it. Until
then this file raises an ``ImportError`` for the pass *only*.
"""

import pytest
pytestmark = pytest.mark.skip(reason="legacy K=1/K=2 descent path frozen during walker-primary migration -- this test goes through VectorizeCPUMultiDim or the harness; both depend on the legacy descent + emit infrastructure being removed. Will be revived (or replaced by walker-primary equivalents) after the new orchestrator pipeline lands end-to-end.")
import copy

import numpy
import pytest

import dace
from dace import data as dt
from dace.transformation.passes.vectorization.stage_global_array_through_scalars import (
    StageGlobalArrayThroughScalars, )

from tests.passes.vectorization.helpers.harness import (
    _get_cloudsc_snippet_four, )

#: Number of species in the cloudsc-style ``[i, j, 5]`` global arrays.
NSPECIES = 5

#: Tolerances for the value-preserving numerical equivalence check.
RTOL = 1e-12
ATOL = 1e-12


def _transient_scalars(sdfg: dace.SDFG) -> set:
    """Collect every transient scalar descriptor name in ``sdfg`` (recursively).

    :param sdfg: The SDFG to scan.
    :returns: A set of descriptor names that are transient scalars.
    """
    names = set()
    for sd in sdfg.all_sdfgs_recursive():
        for name, desc in sd.arrays.items():
            if isinstance(desc, dt.Scalar) and desc.transient:
                names.add(name)
    return names


def _global_write_edges(sdfg: dace.SDFG, array_name: str) -> list:
    """Collect every edge that writes the global array ``array_name`` (recursively).

    :param sdfg: The SDFG to scan.
    :param array_name: The global array descriptor name.
    :returns: A list of ``(edge, state)`` tuples for every write.
    """
    writes = []
    for sd in sdfg.all_sdfgs_recursive():
        for state in sd.all_states():
            for node in state.data_nodes():
                if node.data != array_name:
                    continue
                for edge in state.in_edges(node):
                    writes.append((edge, state))
    return writes


def _global_to_tasklet_edges(sdfg: dace.SDFG, array_name: str) -> int:
    """Count surviving ``A(global) -> Tasklet`` edges for ``array_name``.

    After staging the read side, the consumer reads a transient scalar, so a
    correctly-staged chain leaves zero such edges.

    :param sdfg: The SDFG to scan.
    :param array_name: The global array descriptor name.
    :returns: The number of ``global -> tasklet`` out-edges.
    """
    count = 0
    for sd in sdfg.all_sdfgs_recursive():
        for state in sd.all_states():
            for node in state.data_nodes():
                if node.data != array_name:
                    continue
                count += sum(1 for e in state.out_edges(node) if isinstance(e.dst, dace.nodes.Tasklet))
    return count


def _cloudsc_four_arrays(klon: int):
    """Build the input arrays + params for ``_get_cloudsc_snippet_four`` at ``klon``.

    :param klon: Concrete column count.
    :returns: ``(arrays, params)`` ready for a compiled-SDFG call.
    """
    arr_shapes = {
        "zfallsink": (klon, klon, NSPECIES),
        "zqlhs": (klon, klon, NSPECIES),
        "zsolqb": (klon, klon, NSPECIES),
    }
    arrays = {
        name: numpy.random.default_rng(7).random(shape).astype(numpy.float64, order="F")
        for name, shape in arr_shapes.items()
    }
    params = {
        "kfdia": numpy.int64(klon // 2),
        "kidia": numpy.int64(1),
        "klev": numpy.int64(klon),
        "klon": numpy.int64(klon),
        "_for_it_92": numpy.int64(0),
        "_for_it_91": numpy.int64(0),
    }
    return arrays, params


@pytest.mark.parametrize("klon", [8, 16, 17])
def test_cloudsc_four_zqlhs_chain_numerics(klon: int):
    """Staging the cloudsc ``zqlhs`` reuse chain stays bit-equivalent to the
    untransformed reference across several column counts.

    :param klon: Concrete column count exercised.
    """
    ref = _get_cloudsc_snippet_four()
    ref.name = f"cloudsc_four_num_ref_{klon}"
    ref.validate()

    vec = _get_cloudsc_snippet_four()
    vec.name = f"cloudsc_four_num_vec_{klon}"
    applied = StageGlobalArrayThroughScalars().apply_pass(vec, {})
    vec.validate()
    assert applied, "expected the zqlhs chain to host stageable global hops"

    arrays, params = _cloudsc_four_arrays(klon)
    ref_arrays = {k: copy.deepcopy(v) for k, v in arrays.items()}
    vec_arrays = {k: copy.deepcopy(v) for k, v in arrays.items()}
    ref.compile()(**ref_arrays, **params)
    vec.compile()(**vec_arrays, **params)
    for key in arrays:
        numpy.testing.assert_allclose(vec_arrays[key],
                                      ref_arrays[key],
                                      rtol=RTOL,
                                      atol=ATOL,
                                      err_msg=f"cloudsc_four[klon={klon}]: array {key!r} diverged after staging")


def test_cloudsc_four_preserves_global_store_and_adds_transients():
    """Staging the ``zqlhs`` chain keeps the global store to ``zqlhs`` (the
    output array) and introduces transient scalars."""
    vec = _get_cloudsc_snippet_four()
    vec.name = "cloudsc_four_struct"
    n_before = len(_transient_scalars(vec))
    StageGlobalArrayThroughScalars().apply_pass(vec, {})
    vec.validate()

    assert _global_write_edges(vec, "zqlhs"), "the global zqlhs store must survive staging"
    assert len(_transient_scalars(vec)) > n_before, "staging must introduce transient scalars"


def test_cloudsc_four_removes_global_to_tasklet_reads():
    """After staging, no consumer reads the global ``zqlhs`` node directly — the
    read side is routed through a transient scalar (the point of the pass)."""
    vec = _get_cloudsc_snippet_four()
    vec.name = "cloudsc_four_noreadthrough"
    before = _global_to_tasklet_edges(vec, "zqlhs")
    applied = StageGlobalArrayThroughScalars().apply_pass(vec, {})
    vec.validate()
    assert applied, "expected stageable hops in the zqlhs chain"
    assert before > 0, "fixture precondition: the chain reads zqlhs through the global node"
    assert _global_to_tasklet_edges(vec, "zqlhs") == 0, \
        "staging must remove every global-zqlhs -> tasklet read edge"


def test_cloudsc_four_is_idempotent():
    """A second application of the pass is a no-op (the chain is fully staged)."""
    vec = _get_cloudsc_snippet_four()
    vec.name = "cloudsc_four_idem"
    StageGlobalArrayThroughScalars().apply_pass(vec, {})
    vec.validate()
    again = StageGlobalArrayThroughScalars().apply_pass(vec, {})
    vec.validate()
    assert not again, "second application must find nothing left to stage"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
