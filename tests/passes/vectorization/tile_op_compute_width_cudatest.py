# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The CUDA tile ops must COMPUTE in the element type, not in ``float``.

``dace/runtime/include/dace/tile_ops/cuda.h`` used to funnel every element type through a single
``float`` compute type -- the conversion fp8 genuinely needs, applied to all of them. A ``double``
tile then carried 24 bits of mantissa instead of 53, and a 64-bit integer tile lost every value
past ``2^24``. Nothing about that shows up in the emitted C++: the declarations, the memlets and
the ``tile_binop<double, ...>`` template argument are all correct, and only the RESULT is wrong.

It is not a rounding difference either. vadv over a Thomas sweep came back ``8.7e-03`` relative
against its own unvectorized lowering, and the wrong values were exactly float-representable.

The cases below are chosen so ``float`` is not merely less accurate but ANSWERS DIFFERENTLY: an
addend below the fp32 epsilon of its partner vanishes entirely, and an integer past ``2^24`` moves
to the nearest even. The scalar backend (``tile_ops/scalar.h``) has always computed in ``T``, so
these also pin the two backends to the same answer.
"""
import numpy as np
import pytest

import dace
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.vectorize_gpu import VectorizeGPU

#: Even, so every lane is a full tile and the remainder arm plays no part.
N = 256

#: Below ``np.finfo(np.float32).eps`` (1.2e-07) relative to 1.0, so a float32 add of the two drops
#: it and returns 1.0 exactly. Well above the fp64 epsilon, so the double answer keeps it.
TINY = 1e-12


def vectorized(program, **symbols) -> dace.SDFG:
    """``program`` offloaded and vectorized at width 2, which is what puts it on the tile ops."""
    sdfg = program.to_sdfg(simplify=True)
    if symbols:
        sdfg.specialize(symbols)
    sdfg.apply_gpu_transformations()
    sdfg.simplify()
    VectorizeGPU(VectorizeConfig(widths=(2, ), remainder_strategy='branched_tail')).apply_pass(sdfg, {})
    return sdfg


@dace.program
def add_then_scale(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    c[:] = (a + b) * a


@dace.program
def add_int64(a: dace.int64[N], b: dace.int64[N], c: dace.int64[N]):
    c[:] = a + b


@dace.program
def divide_doubles(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    c[:] = a / b


@pytest.mark.gpu
def test_a_double_tile_keeps_the_bits_float_would_have_dropped():
    """``(1 + 1e-12) * 1`` is ``1.0`` in float32 and ``1.000000000001`` in float64."""
    a = np.ones(N, dtype=np.float64)
    b = np.full(N, TINY, dtype=np.float64)
    c = np.zeros(N, dtype=np.float64)

    vectorized(add_then_scale)(a=a, b=b, c=c)

    reference = (a + b) * a
    assert np.array_equal(c, reference), (f'the double tile answered {c[0]!r}, not {reference[0]!r}; '
                                          f'{"a float32 compute type" if c[0] == 1.0 else "some other narrowing"}')


@pytest.mark.gpu
def test_a_double_division_is_not_a_float_division():
    """Division is where a narrowed compute type does the most damage: the vadv report divided by a
    ``0.15 - ccol`` that cancels to 1e-04, which turns a last-bit float32 error into a 1e-04 one."""
    a = np.linspace(1.0, 2.0, N)
    b = np.full(N, 0.15) - a * 0.0749999999999
    c = np.zeros(N, dtype=np.float64)

    vectorized(divide_doubles)(a=a, b=b, c=c)

    assert np.allclose(c, a / b, rtol=1e-15, atol=0.0), \
        f'max relative error {np.max(np.abs(c - a / b) / np.abs(a / b)):.3e} against the fp64 quotient'


@pytest.mark.gpu
def test_an_int64_tile_survives_two_to_the_twenty_fourth():
    """Every value here is exactly representable in int64 and NONE of them in float32, so a float
    compute type answers with a neighbouring even number rather than the sum."""
    a = np.arange(N, dtype=np.int64) + (1 << 40)
    b = np.ones(N, dtype=np.int64)
    c = np.zeros(N, dtype=np.int64)

    vectorized(add_int64)(a=a, b=b, c=c)

    assert np.array_equal(c, a + b), f'{int((c != a + b).sum())} of {N} lanes differ from the integer sum'


if __name__ == '__main__':
    test_a_double_tile_keeps_the_bits_float_would_have_dropped()
    test_a_double_division_is_not_a_float_division()
    test_an_int64_tile_survives_two_to_the_twenty_fourth()
