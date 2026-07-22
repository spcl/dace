# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""End-to-end ``dace.bfloat16`` on the CPU target: conversions and OpenMP reductions.

The counterpart of ``tests/half_cpu_test.py``. Everything here goes through the real
code generator and the real C++ compiler, and every expectation is exact -- see the
per-test docstrings for why a tolerance would be meaningless at 8 significant bits.

``ml_dtypes.bfloat16`` is the reference throughout: it is the NumPy scalar the
typeclass is registered against, so agreeing with it is what "``dace.bfloat16`` is
bfloat16" means.
"""

import os
import subprocess
import sys

import ml_dtypes
import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes.reduce import Reduce

BF16 = ml_dtypes.bfloat16


def bits(arr):
    """Raw 16-bit storage of a bfloat16 array, for bit-exact comparison."""
    return arr.view(np.uint16)


def all_bfloat16_patterns():
    """All 2^16 bfloat16 bit patterns, as a bfloat16 array."""
    return np.arange(1 << 16, dtype=np.uint16).view(BF16)


def test_bfloat16_widening_is_exhaustively_exact():
    """bfloat16 -> float32 must be exact for every one of the 2^16 bit patterns.

    bfloat16 IS the top half of a float32, so widening is a pure 16-bit shift and
    cannot lose anything -- NaNs, infinities and subnormals included. Asserted on raw
    bits rather than values so that NaN payloads are compared too (``==`` would call
    every NaN unequal and quietly pass a broken conversion).
    """
    values = all_bfloat16_patterns()

    sdfg = dace.SDFG('bf16_widen')
    sdfg.add_array('A', [1 << 16], dace.bfloat16)
    sdfg.add_array('B', [1 << 16], dace.float32)
    state = sdfg.add_state()
    state.add_mapped_tasklet('widen',
                             dict(i=f'0:{1 << 16}'),
                             dict(inp=dace.Memlet('A[i]')),
                             'out = float(inp)',
                             dict(out=dace.Memlet('B[i]')),
                             external_edges=True)
    out = np.zeros(1 << 16, dtype=np.float32)
    sdfg.compile()(A=values.copy(), B=out)

    expected = values.astype(np.float32)
    np.testing.assert_array_equal(out.view(np.uint32), expected.view(np.uint32))


def test_bfloat16_narrowing_is_exhaustively_exact():
    """float32 -> bfloat16 must match ml_dtypes for every representable value.

    Feeds back the exact float32 image of all 2^16 bfloat16 patterns: narrowing them
    is the identity, so any rounding-direction or NaN-handling bug shows up as a
    changed bit pattern. The rounding rule itself (round-to-nearest-even, including
    exact ties and NaNs whose payload lives in the discarded low bits) is covered
    exhaustively over all 2^32 float inputs by ``tests/cpp/bfloat16_ops_test.cpp``;
    what this adds is that DaCe's *generated code* takes that same path.
    """
    values = all_bfloat16_patterns().astype(np.float32)

    sdfg = dace.SDFG('bf16_narrow')
    sdfg.add_array('A', [1 << 16], dace.float32)
    sdfg.add_array('B', [1 << 16], dace.bfloat16)
    state = sdfg.add_state()
    state.add_mapped_tasklet('narrow',
                             dict(i=f'0:{1 << 16}'),
                             dict(inp=dace.Memlet('A[i]')),
                             'out = inp',
                             dict(out=dace.Memlet('B[i]')),
                             external_edges=True)
    out = np.zeros(1 << 16, dtype=BF16)
    sdfg.compile()(A=values.copy(), B=out)

    # NaN payloads are excluded from the bit comparison and checked separately.
    # IEEE-754 leaves the payload of a converted NaN unspecified, and the two
    # implementations legitimately differ: DaCe quiets in place (0x7f81 -> 0x7fc1),
    # which is what CUDA's __float2bfloat16 does and therefore keeps the host and
    # device paths of one program agreeing, while ml_dtypes collapses every NaN to
    # the canonical 0x7fc0. Both are conforming; only "stays a NaN" is asserted.
    nan = np.isnan(values)
    assert nan.sum() == 254, 'expected 2 signs * 127 payloads of NaN among the patterns'
    np.testing.assert_array_equal(bits(out)[~nan], bits(values[~nan].astype(BF16)))
    assert np.isnan(out[nan].astype(np.float32)).all()


def test_bfloat16_narrowing_rounds_like_ml_dtypes_on_random_floats():
    """Random float32 -> bfloat16 must agree with ml_dtypes bit for bit.

    The exhaustive test above only feeds values that are already representable. This
    one feeds floats that genuinely have to round, drawn across the whole exponent
    range (bfloat16's range is float32's, so large and tiny magnitudes are in-range
    rather than saturating).
    """
    rng = np.random.default_rng(5)
    raw = rng.integers(0, 1 << 32, size=1 << 16, dtype=np.uint64).astype(np.uint32)
    values = raw.view(np.float32)
    finite = values[np.isfinite(values)]

    sdfg = dace.SDFG('bf16_narrow_rnd')
    sdfg.add_array('A', [finite.size], dace.float32)
    sdfg.add_array('B', [finite.size], dace.bfloat16)
    state = sdfg.add_state()
    state.add_mapped_tasklet('narrow',
                             dict(i=f'0:{finite.size}'),
                             dict(inp=dace.Memlet('A[i]')),
                             'out = inp',
                             dict(out=dace.Memlet('B[i]')),
                             external_edges=True)
    out = np.zeros(finite.size, dtype=BF16)
    sdfg.compile()(A=finite.copy(), B=out)

    np.testing.assert_array_equal(bits(out), bits(finite.astype(BF16)))


def openmp_reduce_sdfg(name, wcr, identity, size, dtype=dace.bfloat16):
    """A Reduce node pinned to the OpenMP expansion, bfloat16 unless told otherwise.

    Lowers to exactly the pattern this exercises::

        #pragma omp parallel for reduction(<op>: _out[0])
        for (int _i0 = 0; _i0 < size; ++_i0)
            _out[0] <op>= _in[_i0 * 1];

    ``dace::bfloat16`` is a class type on the host, so the clause needs both an
    ``operator<op>=`` and a ``#pragma omp declare reduction`` to be accepted at all.
    """
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [size], dtype)
    sdfg.add_array('s', [1], dtype)
    state = sdfg.add_state()
    red = Reduce('reduce', wcr, None, identity=identity)
    red.implementation = 'OpenMP'
    state.add_node(red)
    state.add_edge(state.add_read('A'), None, red, None, dace.Memlet(f'A[0:{size}]'))
    state.add_edge(red, None, state.add_write('s'), None, dace.Memlet('s[0]'))
    return sdfg


def run_reduction(sdfg, values, npdtype=BF16):
    """Compile and run ``sdfg`` over ``values``."""
    out = np.zeros(1, dtype=npdtype)
    sdfg.compile()(A=values.copy(), s=out)
    return out[0]


#: bfloat16 keeps 8 significant bits, so every integer up to 256 is exact and every
#: partial sum any thread chunking can produce is exact too. That is what makes the
#: sum tests below assertable with no tolerance -- and it is why they use 256
#: elements where the binary16 equivalents use 2048.
EXACT_INT_LIMIT = 256


def test_bfloat16_openmp_reduction_sum_is_exact():
    """Sum of 256 ones is exactly 256 regardless of how threads split it.

    A wrong reduction identity, a dropped chunk, or one combined twice all change the
    integer and are caught. A tolerance-based check would be worthless here: the
    worst-case bound for round-to-nearest summation is ``n*u/(1 - n*u)`` with
    ``u = 2**-8`` for bfloat16, already vacuous (>= 1) at n = 256.
    """
    size = EXACT_INT_LIMIT
    sdfg = openmp_reduce_sdfg('bf16_omp_sum', 'lambda a, b: a + b', BF16(0), size)
    assert run_reduction(sdfg, np.ones(size, dtype=BF16)) == BF16(size)


def test_bfloat16_openmp_reduction_sum_nonuniform_is_exact():
    """Same, with data a "just returns n" bug could not fake."""
    rng = np.random.default_rng(7)
    size = EXACT_INT_LIMIT
    values = rng.integers(-1, 3, size=size).astype(BF16)
    expected = BF16(values.astype(np.float64).sum())
    assert abs(float(expected)) <= EXACT_INT_LIMIT

    sdfg = openmp_reduce_sdfg('bf16_omp_sum_nu', 'lambda a, b: a + b', BF16(0), size)
    assert run_reduction(sdfg, values) == expected


def test_bfloat16_openmp_reduction_product_is_exact():
    """Products of powers of two are exact in bfloat16 while they stay in range."""
    size = EXACT_INT_LIMIT
    values = np.where(np.arange(size) % 2 == 0, np.float32(0.5), np.float32(2.0)).astype(BF16)
    sdfg = openmp_reduce_sdfg('bf16_omp_prod', 'lambda a, b: a * b', BF16(1), size)
    # Equal counts of 0.5 and 2.0, so the exact product is 1.0 in any order.
    assert run_reduction(sdfg, values) == BF16(1.0)


@pytest.mark.parametrize('kind, wcr, reference', [('min', 'lambda a, b: min(a, b)', np.min),
                                                  ('max', 'lambda a, b: max(a, b)', np.max)])
def test_bfloat16_openmp_reduction_minmax(kind, wcr, reference):
    """min/max select an existing element, so they never round: assert exactly.

    ``identity=None`` on purpose, so the identity comes from
    ``dace.dtypes.reduction_identity`` -- which for bfloat16 goes through
    ``max_value`` / ``min_value``. Those used to raise ``TypeError`` for every
    ml_dtypes-backed float, because NumPy does not classify bfloat16 as inexact and
    ``numpy.finfo`` rejects it.
    """
    rng = np.random.default_rng(11)
    size = 2048
    values = (rng.standard_normal(size) * 100.0).astype(BF16)
    sdfg = openmp_reduce_sdfg(f'bf16_omp_{kind}', wcr, None, size)
    assert run_reduction(sdfg, values) == reference(values)


def test_bfloat16_openmp_reduction_div_matches_sequential():
    """Div lowers to a product reduction over the divisors, then one divide.

    ``reduction(/: x)`` does not exist -- ``/`` is not an OpenMP reduction identifier
    and the clause is rejected for every dtype, ``float`` included. Division is not
    associative, but ``a / b / c / d == a / (b * c * d)``, so the divisors reduce with
    ``*`` and the output is divided once.

    Inputs are powers of two, so the product (2^4) and the quotient (1024 / 16) are
    both exactly representable and the result is independent of how the runtime
    chunks the loop.
    """
    size = EXACT_INT_LIMIT
    values = np.ones(size, dtype=BF16)
    values[[3, 9, 77, 200]] = BF16(2.0)

    sdfg = openmp_reduce_sdfg('bf16_div', 'lambda a, b: a / b', BF16(1024.0), size)
    assert run_reduction(sdfg, values) == BF16(64.0)


@pytest.mark.parametrize('kind, wcr, identity, reference',
                         [('and', 'lambda a, b: a and b', BF16(1), np.logical_and.reduce),
                          ('or', 'lambda a, b: a or b', BF16(0), np.logical_or.reduce)])
def test_bfloat16_openmp_reduction_logical(kind, wcr, identity, reference):
    """&& and || reduce on truthiness and normalize to exactly 1.0 / 0.0.

    Exact by nature -- the result is only ever 0.0 or 1.0 -- so no tolerance is
    involved. Includes a truthy non-1.0 value (5.0) to pin that it normalizes rather
    than propagating, which is what ``float`` does.
    """
    size = EXACT_INT_LIMIT
    for values in (np.ones(size, dtype=BF16), np.zeros(size, dtype=BF16),
                   np.concatenate([np.ones(size - 1, dtype=BF16),
                                   np.zeros(1, dtype=BF16)]),
                   np.concatenate(
                       [np.zeros(7, dtype=BF16),
                        np.full(1, 5.0, dtype=BF16),
                        np.zeros(size - 8, dtype=BF16)])):
        sdfg = openmp_reduce_sdfg(f'bf16_logical_{kind}', wcr, identity, size)
        expected = BF16(1.0) if reference(values.astype(bool)) else BF16(0.0)
        assert run_reduction(sdfg, values) == expected


@pytest.mark.parametrize('wcr', ['lambda a, b: a & b', 'lambda a, b: a | b', 'lambda a, b: a ^ b'])
def test_bfloat16_openmp_bitwise_reduction_is_refused(wcr):
    """A bitwise reduction over bfloat16 must be refused, exactly as for float32.

    This is the asymmetry that must NOT be introduced: C++ has no ``&``/``|``/``^`` on
    a floating-point operand and OpenMP has no reduction identifier for one, so
    declaring these for bfloat16 would make it accept programs float32 rejects.
    """
    sdfg = openmp_reduce_sdfg('bf16_bitwise', wcr, 1, 256)
    with pytest.raises(ValueError, match='not defined for non-integral data type'):
        sdfg.compile()


#: Runs one bfloat16 OpenMP sum in a fresh interpreter. ``OMP_NUM_THREADS`` is only
#: read when the OpenMP runtime first initializes, which has already happened by the
#: time a test body executes -- so varying it genuinely requires a new process.
THREAD_RUNNER = """
import ml_dtypes, numpy as np, dace
from dace.sdfg import SDFG
sdfg = SDFG.from_file({path!r})
out = np.zeros(1, dtype=ml_dtypes.bfloat16)
sdfg.compile()(A=np.ones({size}, dtype=ml_dtypes.bfloat16), s=out)
print(float(out[0]))
"""


def test_bfloat16_openmp_reduction_thread_counts(tmp_path):
    """The same exact answer for 1, 2, 4, 8 and 16 threads.

    A reduction can be wrong in ways that only appear at a particular thread count: a
    private copy left uninitialized shows up once there is more than one chunk, and a
    bad combiner shows up once the chunks are merged. Because bfloat16 sums 256 ones
    exactly in any order, every thread count must give exactly 256.
    """
    size = EXACT_INT_LIMIT
    sdfg = openmp_reduce_sdfg('bf16_omp_threads', 'lambda a, b: a + b', BF16(0), size)
    path = str(tmp_path / 'bf16_omp_threads.sdfg')
    sdfg.save(path)
    sdfg.compile()  # populate the build cache so the subprocesses do not each rebuild

    script = THREAD_RUNNER.format(path=path, size=size)
    for threads in (1, 2, 4, 8, 16):
        env = dict(os.environ, OMP_NUM_THREADS=str(threads))
        proc = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, env=env)
        assert proc.returncode == 0, f'OMP_NUM_THREADS={threads}:\n{proc.stdout}\n{proc.stderr}'
        assert float(proc.stdout.strip().splitlines()[-1]) == float(size), \
            f'OMP_NUM_THREADS={threads} gave {proc.stdout.strip()}'


def test_bfloat16_arithmetic_matches_float32():
    """Arithmetic evaluates in float32 and rounds back once, like ``dace::half``.

    Asserted bit-exactly against a float32 reference that rounds to bfloat16 at the
    same single point. If generated code ever evaluated bfloat16 arithmetic *in*
    bfloat16, intermediates would round twice and this would break.
    """
    size = 1024
    rng = np.random.default_rng(17)
    a = (rng.standard_normal(size) * 10.0).astype(BF16)
    b = (rng.standard_normal(size) * 10.0).astype(BF16)

    sdfg = dace.SDFG('bf16_arith')
    sdfg.add_array('A', [size], dace.bfloat16)
    sdfg.add_array('B', [size], dace.bfloat16)
    sdfg.add_array('C', [size], dace.bfloat16)
    state = sdfg.add_state()
    state.add_mapped_tasklet('fma',
                             dict(i=f'0:{size}'),
                             dict(x=dace.Memlet('A[i]'), y=dace.Memlet('B[i]')),
                             'z = x * y + x',
                             dict(z=dace.Memlet('C[i]')),
                             external_edges=True)
    out = np.zeros(size, dtype=BF16)
    sdfg.compile()(A=a.copy(), B=b.copy(), C=out)

    af, bf = a.astype(np.float32), b.astype(np.float32)
    expected = (af * bf + af).astype(BF16)
    np.testing.assert_array_equal(bits(out), bits(expected))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
