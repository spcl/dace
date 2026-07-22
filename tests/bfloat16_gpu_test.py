# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``dace.bfloat16`` on the GPU target, where it is the vendor's native type.

Inside a ``.cu`` translation unit ``dace::bfloat16`` is ``__nv_bfloat16`` (and
``__hip_bfloat16`` under HIP), not the host emulation. The contract that has to hold
across that boundary is LAYOUT, not arithmetic: the host side of a GPU program copies
bytes, so a 2-byte, 2-byte-aligned, standard-layout, trivially-copyable type on both
sides is what makes every transfer bit-exact. ``<dace/types.h>`` asserts that
statically in each branch; these tests check it end to end through real transfers.

Marked ``gpu`` -- collected and run by the GPU CI selection (``-m gpu``), not skipped.
"""

import ml_dtypes
import numpy as np
import pytest

import dace
from dace.transformation.interstate import GPUTransformSDFG

BF16 = ml_dtypes.bfloat16


def bfloat16_half_ulp(values):
    """Half an ulp of bfloat16 at the magnitude of each value.

    ``frexp`` gives ``|v| = m * 2**e`` with ``m`` in [0.5, 1), so ``|v|`` lies in
    ``[2**(e-1), 2**e)``. bfloat16 keeps 8 significant bits, making one ulp there
    ``2**(e-8)`` and half an ulp ``2**(e-9)`` -- exact powers of two, so the bound
    carries no rounding slop of its own.
    """
    _, exp = np.frexp(np.asarray(values, dtype=np.float64))
    return 2.0**(exp.astype(np.int64) - 9)


def gpu_elementwise_sdfg(name, size, code):
    """A GPU-scheduled elementwise kernel over bfloat16 arrays."""
    sdfg = dace.SDFG(name)
    for arr in ('A', 'B', 'C'):
        sdfg.add_array(arr, [size], dace.bfloat16)
    state = sdfg.add_state()
    state.add_mapped_tasklet('kern',
                             dict(i=f'0:{size}'),
                             dict(x=dace.Memlet('A[i]'), y=dace.Memlet('B[i]')),
                             code,
                             dict(z=dace.Memlet('C[i]')),
                             external_edges=True)
    sdfg.apply_transformations(GPUTransformSDFG)
    return sdfg


@pytest.mark.gpu
def test_bfloat16_host_device_roundtrip_is_bit_exact():
    """Every one of the 2^16 bfloat16 bit patterns must survive host -> device -> host.

    This is the guarantee the whole GPU story rests on. ``dace::bfloat16`` is a
    different C++ type on each side of the copy -- the host emulation struct and
    ``__nv_bfloat16`` -- so if their size, alignment or layout ever diverged, arrays
    would be silently mangled rather than failing loudly. Compared on raw bits, so
    NaN payloads are included instead of being swallowed by ``nan != nan``.
    """
    size = 1 << 16
    values = np.arange(size, dtype=np.uint16).view(BF16)

    sdfg = dace.SDFG('bf16_gpu_copy')
    sdfg.add_array('A', [size], dace.bfloat16)
    sdfg.add_array('B', [size], dace.bfloat16)
    state = sdfg.add_state()
    state.add_mapped_tasklet('cp',
                             dict(i=f'0:{size}'),
                             dict(x=dace.Memlet('A[i]')),
                             'y = x',
                             dict(y=dace.Memlet('B[i]')),
                             external_edges=True)
    sdfg.apply_transformations(GPUTransformSDFG)

    out = np.zeros(size, dtype=BF16)
    sdfg.compile()(A=values.copy(), B=out)
    np.testing.assert_array_equal(out.view(np.uint16), values.view(np.uint16))


@pytest.mark.gpu
def test_bfloat16_gpu_single_operation_matches_float32():
    """A SINGLE bfloat16 operation must equal the float32 result rounded once.

    One operation is the case where host and device genuinely agree: both compute in
    (at least) float precision and round the single result to bfloat16. Asserted bit
    for bit -- no tolerance.

    A *fused* expression is deliberately not asserted this way; see the test below.
    """
    size = 4096
    rng = np.random.default_rng(3)
    a = (rng.standard_normal(size) * 4.0).astype(BF16)
    b = (rng.standard_normal(size) * 4.0).astype(BF16)

    sdfg = gpu_elementwise_sdfg('bf16_gpu_mul', size, 'z = x * y')
    out = np.zeros(size, dtype=BF16)
    sdfg.compile()(A=a.copy(), B=b.copy(), C=out)

    expected = (a.astype(np.float32) * b.astype(np.float32)).astype(BF16)
    np.testing.assert_array_equal(out.view(np.uint16), expected.view(np.uint16))


@pytest.mark.gpu
def test_bfloat16_gpu_fused_expression_rounds_per_operation():
    """``x * y + x`` on the device rounds TWICE, and that is expected behaviour.

    The host emulation evaluates a whole expression in ``float`` and rounds once on
    the store, because every operator goes through ``operator float()``. The device
    uses ``__nv_bfloat16``'s own operators, so the product is rounded to bfloat16
    before the add. The two therefore disagree on fused expressions -- this is the
    ordinary consequence of arithmetic in a narrow type, not a bug, and it is
    asserted here so nobody "fixes" it into a silent surprise.

    Both bounds matter: the result must match the round-per-operation reference
    EXACTLY (so the device really is doing bfloat16 arithmetic), and the gap to the
    round-once reference must be no larger than that one extra rounding explains
    (so it is not merely wrong).
    """
    size = 4096
    rng = np.random.default_rng(3)
    a = (rng.standard_normal(size) * 4.0).astype(BF16)
    b = (rng.standard_normal(size) * 4.0).astype(BF16)

    sdfg = gpu_elementwise_sdfg('bf16_gpu_fma', size, 'z = x * y + x')
    out = np.zeros(size, dtype=BF16)
    sdfg.compile()(A=a.copy(), B=b.copy(), C=out)

    af, bf = a.astype(np.float32), b.astype(np.float32)
    round_per_op = ((af * bf).astype(BF16).astype(np.float32) + af).astype(BF16)
    np.testing.assert_array_equal(out.view(np.uint16), round_per_op.view(np.uint16))

    # ...and the distance from the EXACT value is no more than those two roundings
    # allow, so the device is not merely wrong. Rounding ``x * y`` to bfloat16 costs
    # at most half an ulp of ``|x * y|``, that absolute error passes through the
    # addition unchanged, and storing the sum costs at most half an ulp of the result.
    #
    # Measured against the exact float64 value rather than against the round-once
    # bfloat16 answer, and NOT as a ulp count on the result: ``x * y + x`` cancels
    # catastrophically when ``x * y`` is near ``-x``, so a correct half-ulp error on
    # the intermediate shows up as 104 ulps of the (tiny) result. The bound below is
    # tight -- the worst case in this data sits exactly at 1.0x it.
    exact = af.astype(np.float64) * bf.astype(np.float64) + af.astype(np.float64)
    allowed = bfloat16_half_ulp(af.astype(np.float64) * bf.astype(np.float64)) + bfloat16_half_ulp(out)
    delta = np.abs(out.astype(np.float64) - exact)
    assert np.all(delta <= allowed), f'worst overshoot {(delta / allowed).max()}x the two-rounding bound'


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'gpu'])
