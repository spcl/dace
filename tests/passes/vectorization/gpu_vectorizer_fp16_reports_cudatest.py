# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The two GPU-vectorizer reports that arrived against float16 input, end to end and numerically.

Both were compile-time failures, and both were fixed against the emitted text rather than a number:

  * ``np.where(x > 0, 0.0, x)`` on a ``float16`` array reaches codegen as ``ITE(c, 0.0, x)``. The
    literal arm carries no edge, so nothing cast it, and the scalar tail of a vectorized map emits a
    ternary whose ``std::common_type<double, dace::float16>`` does not exist.
  * ``vadv`` with one ``float16`` field takes the tile path, where ``MoveArrayOutOfKernel`` prefixed
    per-iteration indices onto memlets in nested SDFGs whose descriptor it never reshaped, and
    validation rejected a rank-3 memlet on a rank-2 descriptor.

What neither had was a RESULT. A cast placed at the wrong precision, or a lifted array whose slice
selects the wrong iteration, compiles and validates perfectly and answers with the wrong numbers, so
these run the programs and check what came back. Each is compared against the same computation
carried out on the host in the same precisions -- numpy's float16 is the CPU emulation of the
storage type the device path uses natively.
"""
import numpy as np
import pytest

import dace
from dace.transformation.auto import auto_optimize as aopt
from dace.transformation.dataflow import MapFusion
from dace.transformation.interstate import LoopToMap
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.vectorize_gpu import VectorizeGPU
from tests.npbench.weather_stencils.vadv_test import ground_truth, initialize

#: Both programs are about float16, so the module carries the marker the dedicated fp16 CI leg
#: selects on, next to the ``gpu`` marker the GPU legs select on.
pytestmark = pytest.mark.fp16

#: One unit in the last place of the storage type, relative. A select copies its operand across
#: unchanged, so the tolerance only has to absorb a rounding that the two paths could disagree on
#: at all -- anything larger is a narrowing, which is the class of bug this file exists for.
FP16_ULP = float(np.finfo(np.float16).eps)

#: vadv accumulates in float64 and reads ``u_pos`` in float16. The device path fuses and vectorizes
#: what the host reference evaluates one statement at a time, so the two disagree by float64
#: reassociation over a Thomas sweep of K=16 -- a few ulps carried down the column, not a precision
#: difference.
VADV_RTOL = 1e-11

N = dace.symbol('N')
I, J, K = (dace.symbol(s, dtype=dace.int64) for s in ('I', 'J', 'K'))


@dace.program
def where_over_fp16(x: dace.float16[N], y: dace.float16[N]):
    y[:] = np.where(x > 0, 0.0, x)


@dace.program
def vadv_with_an_fp16_field(utens_stage: dace.float64[I, J, K], u_stage: dace.float64[I, J, K],
                            wcon: dace.float64[I + 1, J, K], u_pos: dace.float16[I, J, K], utens: dace.float64[I, J,
                                                                                                               K]):
    """``vadv``, verbatim from the report, with ``u_pos`` in float16 -- the field that sends the
    vectorizer down the tile path. ``dtr_stage`` is the constant the reference uses (3 / 20)."""
    dtr_stage = 3.0 / 20.0

    ccol = np.ndarray((I, J, K), dtype=utens_stage.dtype)
    dcol = np.ndarray((I, J, K), dtype=utens_stage.dtype)
    data_col = np.ndarray((I, J), dtype=utens_stage.dtype)

    for k in range(1):
        gcv = 0.25 * (wcon[1:, :, k + 1] + wcon[:-1, :, k + 1])
        cs = gcv * 0.5

        ccol[:, :, k] = gcv * 0.5
        bcol = dtr_stage - ccol[:, :, k]

        correction_term = -cs * (u_stage[:, :, k + 1] - u_stage[:, :, k])
        dcol[:, :, k] = (dtr_stage * u_pos[:, :, k] + utens[:, :, k] + utens_stage[:, :, k] + correction_term)

        divided = 1.0 / bcol
        ccol[:, :, k] = ccol[:, :, k] * divided
        dcol[:, :, k] = dcol[:, :, k] * divided

    for k in range(1, K - 1):
        gav = -0.25 * (wcon[1:, :, k] + wcon[:-1, :, k])
        gcv[:] = 0.25 * (wcon[1:, :, k + 1] + wcon[:-1, :, k + 1])

        as_ = gav * 0.5
        cs[:] = gcv * 0.5

        acol = gav * 0.5
        ccol[:, :, k] = gcv * 0.5
        bcol[:] = dtr_stage - acol - ccol[:, :, k]

        correction_term[:] = -as_ * (u_stage[:, :, k - 1] - u_stage[:, :, k]) - cs * (u_stage[:, :, k + 1] -
                                                                                      u_stage[:, :, k])
        dcol[:, :, k] = (dtr_stage * u_pos[:, :, k] + utens[:, :, k] + utens_stage[:, :, k] + correction_term)

        divided[:] = 1.0 / (bcol - ccol[:, :, k - 1] * acol)
        ccol[:, :, k] = ccol[:, :, k] * divided
        dcol[:, :, k] = (dcol[:, :, k] - (dcol[:, :, k - 1]) * acol) * divided

    for k in range(K - 1, K):
        gav[:] = -0.25 * (wcon[1:, :, k] + wcon[:-1, :, k])
        as_[:] = gav * 0.5
        acol[:] = gav * 0.5
        bcol[:] = dtr_stage - acol

        correction_term[:] = -as_ * (u_stage[:, :, k - 1] - u_stage[:, :, k])
        dcol[:, :, k] = (dtr_stage * u_pos[:, :, k] + utens[:, :, k] + utens_stage[:, :, k] + correction_term)

        divided[:] = 1.0 / (bcol - ccol[:, :, k - 1] * acol)
        dcol[:, :, k] = (dcol[:, :, k] - (dcol[:, :, k - 1]) * acol) * divided

    for k in range(K - 1, K - 2, -1):
        datacol = dcol[:, :, k]
        data_col[:] = datacol
        utens_stage[:, :, k] = dtr_stage * (datacol - u_pos[:, :, k])

    for k in range(K - 2, -1, -1):
        datacol[:] = dcol[:, :, k] - ccol[:, :, k] * data_col[:, :]
        data_col[:] = datacol
        utens_stage[:, :, k] = dtr_stage * (datacol - u_pos[:, :, k])


def vectorized_where(n: int) -> dace.SDFG:
    """``where_over_fp16`` specialized to ``n``, offloaded, and vectorized at half2 width.

    :param n: The extent to specialize to.
    :returns: The vectorized SDFG, ready to compile.
    """
    sdfg = where_over_fp16.to_sdfg(simplify=True)
    sdfg.specialize({'N': n})
    sdfg.apply_gpu_transformations()
    sdfg.simplify()
    VectorizeGPU(VectorizeConfig(widths=(2, ), remainder_strategy='branched_tail')).apply_pass(sdfg, {})
    return sdfg


def vectorized_vadv(extents: dict) -> dace.SDFG:
    """``vadv_with_an_fp16_field`` through the pipeline the report used, ending in the vectorizer.

    :param extents: ``I`` / ``J`` / ``K`` to specialize to.
    :returns: The vectorized SDFG, ready to compile.
    """
    sdfg = vadv_with_an_fp16_field.to_sdfg(simplify=True)
    sdfg.specialize(extents)
    sdfg = aopt.auto_optimize(sdfg, dace.DeviceType.CPU)
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.apply_transformations_repeated(MapFusion)
    sdfg.simplify()
    sdfg.apply_gpu_transformations(simplify=False)
    sdfg.simplify()
    VectorizeGPU(VectorizeConfig(widths=(2, ), remainder_strategy='branched_tail')).apply_pass(sdfg, {})
    return sdfg


@pytest.mark.gpu
@pytest.mark.parametrize('n', [1024, 1025])
def test_a_where_over_fp16_selects_the_same_values_the_host_does(n):
    """The blend keeps every operand it selects, at both an even extent and an odd one.

    The odd extent is the point: the remainder lane is the scalar tail, which is the arm that failed
    to compile and the only arm the tile assertions never look at.
    """
    x = np.linspace(-4.0, 4.0, n).astype(np.float16)
    y = np.zeros(n, dtype=np.float16)

    vectorized_where(n)(x=x, y=y)

    reference = np.where(x > np.float16(0), np.float16(0), x)
    assert np.allclose(y.astype(np.float64), reference.astype(np.float64), rtol=FP16_ULP,
                       atol=0.0), (f'{int((y != reference).sum())} of {n} lanes differ from the host select')


@pytest.mark.gpu
def test_vadv_with_an_fp16_field_matches_the_host_reference():
    """The whole stencil, against the numpy reference the npbench vadv test is checked with.

    The reference reads ``u_pos`` at the values the kernel reads -- the float16 field widened, not
    the float64 field it was rounded from -- so the only difference left to measure is the lowering.
    """
    extents = {'I': 32, 'J': 32, 'K': 16}
    _, utens_stage, u_stage, wcon, u_pos, utens = initialize(extents['I'], extents['J'], extents['K'])
    u_pos16 = u_pos.astype(np.float16)

    reference = np.copy(utens_stage)
    ground_truth(reference, u_stage, wcon, u_pos16.astype(np.float64), utens, 3.0 / 20.0)

    result = np.copy(utens_stage)
    vectorized_vadv(extents)(utens_stage=result, u_stage=u_stage, wcon=wcon, u_pos=u_pos16, utens=utens)

    assert np.allclose(result, reference, rtol=VADV_RTOL, atol=0.0), (
        f'max relative error {np.max(np.abs(result - reference) / np.abs(reference)):.3e} exceeds {VADV_RTOL:.0e}')


if __name__ == '__main__':
    test_a_where_over_fp16_selects_the_same_values_the_host_does(1024)
    test_a_where_over_fp16_selects_the_same_values_the_host_does(1025)
    test_vadv_with_an_fp16_field_matches_the_host_reference()
