# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The vectorizer must not lift an outer product to a degenerate GEMM.

An outer product ``A[i, j] = u[i] * v[j]`` (einsum ``i,j->ij``) has no contracted
index -- every output element is independent. Lifting it to an Einsum / GEMM
node makes a K=1 matrix multiply, which is no BLAS win and which the tile
widener mis-lowers (unbound inner dimensions -> illegal DGEMM leading
dimension, gemver's rank-2 update). ``LiftEinsum(contraction_only=True)`` refuses
it so it tiles per-lane like any other elementwise broadcast.
"""
import copy

import numpy as np
import pytest

from dace.libraries.blas.nodes.matmul import MatMul
from dace.sdfg.nodes import LibraryNode
from dace.transformation.dataflow.lift_einsum import LiftEinsum
from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated

N = 16


def _gemm_libnodes(sdfg):
    """Count Gemm / Einsum library nodes (a lifted contraction)."""
    return [
        type(n).__name__ for s in sdfg.all_sdfgs_recursive() for st in s.states() for n in st.nodes()
        if isinstance(n, LibraryNode) and type(n).__name__ in ('Gemm', 'Einsum')
    ]


def test_default_mode_lifts_outer_products_contraction_only_does_not():
    """On gemver's rank-2 update, the default lift produces GEMM/Einsum nodes
    from the outer products; the contraction-only mode produces none."""
    try:
        import tests.corpus.measure_parallelization as mp
    except Exception:
        pytest.skip('corpus harness unavailable')
    from dace.transformation.passes.parallelize import parallelize

    base, _ = mp.CORPORA['poly'][1]('gemver')

    default = copy.deepcopy(base)
    parallelize(default, validate=True, validate_all=False, peel_limit=4)
    PatternMatchAndApplyRepeated([LiftEinsum()]).apply_pass(default, {})

    guarded = copy.deepcopy(base)
    parallelize(guarded, validate=True, validate_all=False, peel_limit=4)
    PatternMatchAndApplyRepeated([LiftEinsum(contraction_only=True)]).apply_pass(guarded, {})

    # The default lift turns the outer products into GEMM/Einsum nodes; the
    # contraction-only mode leaves them as elementwise maps (no new contraction
    # node beyond the two GEMV MatMuls already present).
    assert len(_gemm_libnodes(default)) > len(_gemm_libnodes(guarded))


def test_existing_matmul_contractions_are_untouched():
    """The two GEMV MatMuls gemver already carries survive either mode -- the
    guard only affects the map-level outer-product lift, not real contractions."""
    try:
        import tests.corpus.measure_parallelization as mp
    except Exception:
        pytest.skip('corpus harness unavailable')
    from dace.transformation.passes.parallelize import parallelize

    base, _ = mp.CORPORA['poly'][1]('gemver')
    sd = copy.deepcopy(base)
    parallelize(sd, validate=True, validate_all=False, peel_limit=4)
    matmuls_before = sum(1 for s in sd.all_sdfgs_recursive() for st in s.states() for n in st.nodes()
                         if isinstance(n, MatMul))
    PatternMatchAndApplyRepeated([LiftEinsum(contraction_only=True)]).apply_pass(sd, {})
    matmuls_after = sum(1 for s in sd.all_sdfgs_recursive() for st in s.states() for n in st.nodes()
                        if isinstance(n, MatMul))
    assert matmuls_before == matmuls_after == 2, 'the existing GEMV MatMuls must be untouched'


def test_gemver_vectorizes_correctly_both_pipelines():
    """End-to-end: gemver (a rank-2 update + two GEMVs) is value-correct under
    both canon+vec and parallelize+vec once the outer-product lift is refused.

    Skipped if the corpus harness is unavailable.
    """
    import copy
    try:
        import tests.corpus.measure_parallelization as mp
    except Exception:
        pytest.skip('corpus harness unavailable')
    from dace.transformation.passes.canonicalize.finalize import finalize_for_target

    base, checker = mp.CORPORA['poly'][1]('gemver')
    for config in ('canon+vec', 'parallelize+vec'):
        sd = copy.deepcopy(base)
        mp.apply_config(sd, config, mp.cpu_params(4))
        fin = finalize_for_target(copy.deepcopy(sd), 'cpu')
        fin.name = 'gemver_' + config.replace('+', '_')
        assert bool(checker(fin)), f'gemver must be correct under {config}'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
