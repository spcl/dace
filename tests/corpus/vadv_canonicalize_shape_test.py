# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shape integrity of the npbench ``vadv`` kernel across canonicalization.

``vadv`` is the corpus's only kernel with an ASYMMETRIC leading dimension:
``wcon`` is declared ``[I + 1, J, K]`` while every other array is ``[I, J, K]``,
and the kernel reads ``wcon[1:, :, k]`` / ``wcon[:-1, :, k]`` -- two ``I``-long
windows into an ``I + 1``-long axis. Any pass that confuses a window length with
its parent axis length turns an ``(I, J)`` extent into ``(I + 1, J)``, which at
``I = 255`` reads as a ``(255, 256) -> (256, 256)`` off-by-one. These tests pin
that no such drift happens.

``test_scalarized_single_element_scratch_validates`` additionally guards the
``other_subset`` collapse in :class:`ConvertLengthOneArraysToScalars`: canonicalize
leaves ``(1, 1)`` MapFusion scratch buffers that ``single_element=True``
scalarizes. A copy edge names only ONE side in ``Memlet.data``; the opposite side
is addressed by ``other_subset``. Without collapsing that opposite side the edge
keeps its pre-scalarization RANK and validation rejects it with
"Memlet other_subset does not match node dimension (expected 1, got 2)".
"""
import numpy as np
import pytest

from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.length_one_array_scalar_conversion import ConvertLengthOneArraysToScalars
from tests.corpus.npbench.structured_grids import vadv

#: The numerical corpus gate's canonicalize configuration (canonicalize_numerical_corpus_test._CPU).
_CPU = dict(target='cpu',
            peel_limit=4,
            break_anti_dependence=True,
            interchange_carry_with_map=True,
            scatter_to_guarded_maps=True)

#: Kernel argument order, matching ``vadv.initialize``'s positional return.
_ARGS = ('utens_stage', 'u_stage', 'wcon', 'u_pos', 'utens', 'dtr_stage')

#: fp64 corpus tolerance (npbench._tol_for). vadv is FP-reassociation sensitive
#: (the inner k-sweep is a Thomas solve), so the gate's own criterion is used
#: rather than a bit-exact compare.
_RTOL, _ATOL = 1e-9, 1e-11


def _inputs(I, J, K):
    """``(arrays_dict, reference_utens_stage)`` for one dataset size."""
    arrays = dict(zip(_ARGS, vadv.CORPUS['initialize'](I, J, K)))
    work = {n: (v.copy() if isinstance(v, np.ndarray) else v) for n, v in arrays.items()}
    vadv.CORPUS['reference'](**work)
    return arrays, work['utens_stage']


def _canonicalized(tag):
    sdfg = vadv.CORPUS['program'].to_sdfg(simplify=True)
    sdfg.name = f'vadv_{tag}'
    canonicalize(sdfg, validate=True, **_CPU)
    return sdfg


def test_canonicalize_preserves_array_shapes():
    """Canonicalize changes no surviving array's shape -- in particular ``wcon``
    keeps ``I + 1`` and the ``(I, J)`` slice transients do not grow to ``(I + 1, J)``."""
    sdfg = vadv.CORPUS['program'].to_sdfg(simplify=True)
    before = {name: tuple(str(d) for d in desc.shape) for name, desc in sdfg.arrays.items()}
    sdfg.name = 'vadv_shape_preserved'
    canonicalize(sdfg, validate=True, **_CPU)
    after = {name: tuple(str(d) for d in desc.shape) for name, desc in sdfg.arrays.items()}

    drifted = {name: (before[name], after[name]) for name in before if name in after and before[name] != after[name]}
    assert not drifted, f'canonicalize changed vadv array shapes: {drifted}'
    assert after['wcon'] == ('I + 1', 'J', 'K'), f"wcon lost its asymmetric leading dim: {after['wcon']}"


def test_canonicalize_validates_and_leaves_no_oversized_subset():
    """The canonicalized SDFG validates: every memlet subset stays inside its array,
    so no ``wcon`` window was propagated with the parent ``I + 1`` extent."""
    _canonicalized('validates').validate()


def test_scalarized_single_element_scratch_validates():
    """canonicalize + ``single_element`` scalarization validates.

    FAILS without the ``other_subset`` collapse in ``ConvertLengthOneArraysToScalars``
    with ``InvalidSDFGEdgeError: Memlet other_subset does not match node dimension
    (expected 1, got 2)`` on the ``(1, 1)`` MapFusion scratch.
    """
    sdfg = _canonicalized('single_element')
    scalarized = ConvertLengthOneArraysToScalars(single_element=True, transient_only=True).apply_pass(sdfg, {})
    assert scalarized, 'expected canonicalize to leave single-element scratch for scalarization'
    sdfg.validate()


@pytest.mark.parametrize('size', [(32, 32, 32), (33, 32, 30)])
def test_canonicalize_matches_reference(size):
    """End-to-end value check at an even and an ODD ``I`` (an odd ``I`` pairs a
    255-style window with a 256-style parent axis)."""
    I, J, K = size
    arrays, ref = _inputs(I, J, K)
    sdfg = _canonicalized(f'ref_{I}_{J}_{K}')
    finalize_for_target(sdfg, 'cpu')

    work = {n: (v.copy() if isinstance(v, np.ndarray) else v) for n, v in arrays.items()}
    sdfg.compile()(**work, I=I, J=J, K=K)
    assert np.allclose(work['utens_stage'], ref, rtol=_RTOL, atol=_ATOL)
