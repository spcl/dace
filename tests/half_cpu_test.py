# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import os
import subprocess
import sys

import dace
import numpy as np
import pytest

from dace.libraries.standard.nodes.reduce import Reduce

N = dace.symbol("N")


def _roundtrip_program():
    """A program that stores float32 input as float16 and reads it back as float32.

    This forces both conversion directions of the host ``dace::half``:
    ``half(float)`` on the store and ``operator float()`` on the load.
    """

    @dace.program
    def f32_to_f16_to_f32(inp: dace.float32[N], out: dace.float32[N]):
        tmp = np.ndarray([N], dace.float16)
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << inp[i]
                t >> tmp[i]
                t = a  # float32 -> float16
        for i in dace.map[0:N]:
            with dace.tasklet:
                t << tmp[i]
                o >> out[i]
                o = t  # float16 -> float32

    return f32_to_f16_to_f32


def test_float16_cpu_roundtrip_matches_numpy():
    """Host float32<->float16 conversion must agree with IEEE-754 (NumPy float16).

    Covers zero, exact-rounding, small/subnormal magnitudes, inf, and NaN.
    """
    prog = _roundtrip_program()
    inp = np.array(
        [
            0.0,
            -0.0,
            1.0,
            -1.0,
            2.0,
            3.0,
            0.5,
            -0.5,
            1.25,
            -2.5,
            100.0,
            1024.0,
            65504.0,
            1e-3,
            6e-5,
            6e-8,
            -6e-8,
            np.inf,
            -np.inf,
            np.nan,
        ],
        dtype=np.float32,
    )
    out = np.full(inp.shape, np.nan, dtype=np.float32)
    prog(inp=inp, out=out, N=inp.size)

    # NumPy's float16 is IEEE-754 round-to-nearest-even -- the reference.
    expected = inp.astype(np.float16).astype(np.float32)
    np.testing.assert_array_equal(out, expected)


def test_float16_cpu_random_roundtrip_matches_numpy():
    """Randomized sweep of the host conversion against NumPy float16."""
    rng = np.random.default_rng(0)
    inp = (rng.standard_normal(4096) * 10.0).astype(np.float32)
    out = np.full(inp.shape, np.nan, dtype=np.float32)

    prog = _roundtrip_program()
    prog(inp=inp, out=out, N=inp.size)

    expected = inp.astype(np.float16).astype(np.float32)
    np.testing.assert_array_equal(out, expected)


def openmp_reduce_sdfg(name, wcr, identity, size, dtype=dace.float16):
    """A Reduce node pinned to the OpenMP expansion, float16 unless told otherwise.

    Lowers to exactly the pattern this exercises::

        #pragma omp parallel for reduction(<op>: _out[0])
        for (int _i0 = 0; _i0 < size; ++_i0)
            _out[0] <op>= _in[_i0 * 1];

    ``dace::float16`` is a class type on the host, so the clause needs both a
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


def run_reduction(sdfg, values):
    """Compile and run ``sdfg`` over ``values``.

    The OpenMP runtime is already initialized by the time a test body runs, so the
    thread count is whatever the pytest process was started with (all cores unless
    ``OMP_NUM_THREADS`` was exported beforehand) and cannot be changed from here --
    ``test_float16_openmp_reduction_thread_counts`` re-runs the same reduction in
    fresh processes to vary it.
    """
    out = np.zeros(1, dtype=np.float16)
    sdfg.compile()(A=values.copy(), s=out)
    return out[0]


def test_float16_openmp_reduction_sum_is_exact():
    """Sum of 2048 ones is exact in binary16 regardless of how threads split it.

    binary16 represents every integer up to 2048 exactly, so every partial sum any
    chunking can produce is exact and the result is order-independent. That makes
    this assertable with no tolerance at all -- a wrong reduction identity, a chunk
    that is dropped, or one that is combined twice all change the integer and are
    caught. A tolerance-based check would be worthless at this size: the standard
    worst-case bound for round-to-nearest summation is ``n*u/(1 - n*u)`` with
    ``u = 2**-11`` for binary16, which is already vacuous (>= 1) for n = 2048.
    """
    size = 2048
    sdfg = openmp_reduce_sdfg('half_omp_sum', 'lambda a, b: a + b', np.float16(0), size)
    result = run_reduction(sdfg, np.ones(size, dtype=np.float16))
    assert result == np.float16(size)


def test_float16_openmp_reduction_sum_nonuniform_is_exact():
    """Same, with data that a "just returns n" bug could not fake.

    Values are drawn from {-1, 0, 1, 2} and the array is sized so that the running
    total stays within the exactly-representable integer range of binary16, keeping
    the expected result exact.
    """
    rng = np.random.default_rng(7)
    size = 2048
    values = rng.integers(-1, 3, size=size).astype(np.float16)
    expected = np.float16(values.astype(np.float64).sum())
    assert abs(float(expected)) <= 2048.0

    sdfg = openmp_reduce_sdfg('half_omp_sum_nu', 'lambda a, b: a + b', np.float16(0), size)
    assert run_reduction(sdfg, values) == expected


def test_float16_openmp_reduction_product_is_exact():
    """Products of powers of two are exact in binary16 while they stay in range."""
    size = 2048
    values = np.where(np.arange(size) % 2 == 0, np.float16(0.5), np.float16(2.0)).astype(np.float16)
    sdfg = openmp_reduce_sdfg('half_omp_prod', 'lambda a, b: a * b', np.float16(1), size)
    # Equal counts of 0.5 and 2.0, so the exact product is 1.0 in any order.
    assert run_reduction(sdfg, values) == np.float16(1.0)


@pytest.mark.parametrize('kind, wcr, identity, reference',
                         [('min', 'lambda a, b: min(a, b)', np.float16(np.inf), np.min),
                          ('max', 'lambda a, b: max(a, b)', np.float16(-np.inf), np.max)])
def test_float16_openmp_reduction_minmax(kind, wcr, identity, reference):
    """min/max select an existing element, so they never round: assert exactly."""
    rng = np.random.default_rng(11)
    size = 2048
    values = (rng.standard_normal(size) * 100.0).astype(np.float16)
    sdfg = openmp_reduce_sdfg(f'half_omp_{kind}', wcr, identity, size)
    assert run_reduction(sdfg, values) == reference(values)


@pytest.mark.parametrize('dtype, npdtype', [(dace.float16, np.float16), (dace.float32, np.float32)])
def test_openmp_reduction_div_matches_sequential(dtype, npdtype):
    """Div lowers to a product reduction over the divisors, then one divide.

    ``reduction(/: x)`` does not exist -- ``/`` is not an OpenMP reduction identifier
    and the clause is rejected for every dtype, ``float`` included. Division is not
    associative, but ``a / b / c / d == a / (b * c * d)``, so the divisors reduce with
    ``*`` and the output is divided once.

    Inputs are powers of two: the product (2^4) and the quotient (1024 / 16) are both
    exactly representable in binary16, so the result is exact and independent of how
    the runtime chunks the loop, and is compared against a sequential reference rather
    than a tolerance. Run for float32 as well, since the old lowering was broken for
    every dtype and not just fp16.
    """
    size = 2048
    values = np.ones(size, dtype=npdtype)
    values[[3, 9, 77, 900]] = npdtype(2.0)
    identity = npdtype(1024.0)

    sdfg = openmp_reduce_sdfg(f'div_{np.dtype(npdtype).name}', 'lambda a, b: a / b', identity, size, dtype=dtype)
    out = np.zeros(1, dtype=npdtype)
    sdfg.compile()(A=values.copy(), s=out)

    reference = np.float64(identity)
    for v in values:
        reference = reference / np.float64(v)
    assert reference == 64.0  # 1024 / 2^4, exact
    assert out[0] == npdtype(reference)


@pytest.mark.parametrize('kind, wcr, identity, reference',
                         [('and', 'lambda a, b: a and b', np.float16(1), np.logical_and.reduce),
                          ('or', 'lambda a, b: a or b', np.float16(0), np.logical_or.reduce)])
def test_float16_openmp_reduction_logical(kind, wcr, identity, reference):
    """&& and || reduce on truthiness and normalize to exactly 1.0 / 0.0.

    Exact by nature -- the result is only ever 0.0 or 1.0, both exactly representable
    -- so no tolerance is involved. Includes a truthy non-1.0 value (5.0) to pin that
    it normalizes rather than propagating, which is what ``float`` does.
    """
    size = 2048
    for values in (np.ones(size, dtype=np.float16), np.zeros(size, dtype=np.float16),
                   np.concatenate([np.ones(size - 1, dtype=np.float16),
                                   np.zeros(1, dtype=np.float16)]),
                   np.concatenate([np.zeros(7, dtype=np.float16),
                                   np.full(1, 5.0, dtype=np.float16),
                                   np.zeros(size - 8, dtype=np.float16)])):
        sdfg = openmp_reduce_sdfg(f'logical_{kind}', wcr, identity, size)
        out = np.zeros(1, dtype=np.float16)
        sdfg.compile()(A=values.copy(), s=out)
        expected = np.float16(1.0) if reference(values.astype(bool)) else np.float16(0.0)
        assert out[0] == expected, f'{kind} over {values[:10]}...: got {out[0]}, want {expected}'


@pytest.mark.parametrize('wcr', ['lambda a, b: a & b', 'lambda a, b: a | b', 'lambda a, b: a ^ b'])
@pytest.mark.parametrize('dtype', [dace.float16, dace.float32, dace.float64])
def test_openmp_bitwise_reduction_on_float_is_refused(wcr, dtype):
    """A bitwise reduction over a floating-point dtype must be refused, clearly.

    C++ has no ``&`` / ``|`` / ``^`` on a floating-point operand and OpenMP has no
    reduction identifier for one either, so this combination can never be lowered. It
    used to reach the C++ compiler and fail there with "user defined reduction not
    found for '*(float (*)[1])_out'", which says nothing about the real cause.
    """
    size = 256
    sdfg = openmp_reduce_sdfg('bitwise', wcr, 1, size, dtype=dtype)
    with pytest.raises(ValueError, match='not defined for non-integral data type'):
        sdfg.compile()


def test_openmp_bitwise_reduction_on_integers_still_works():
    """The refusal must not catch the case bitwise reductions are actually for."""
    size = 2048
    rng = np.random.default_rng(3)
    values = rng.integers(0, 2**31 - 1, size=size).astype(np.int32)

    sdfg = openmp_reduce_sdfg('bitwise_int', 'lambda a, b: a & b', -1, size, dtype=dace.int32)
    out = np.zeros(1, dtype=np.int32)
    sdfg.compile()(A=values.copy(), s=out)
    assert out[0] == np.bitwise_and.reduce(values)


#: Runs one float16 OpenMP sum in a fresh interpreter. ``OMP_NUM_THREADS`` is only
#: read when the OpenMP runtime first initializes, which has already happened by the
#: time a test body executes -- so varying it genuinely requires a new process.
THREAD_RUNNER = """
import numpy as np, dace
from dace.sdfg import SDFG
sdfg = SDFG.from_file({path!r})
out = np.zeros(1, dtype=np.float16)
sdfg.compile()(A=np.ones({size}, dtype=np.float16), s=out)
print(float(out[0]))
"""


def test_float16_openmp_reduction_thread_counts(tmp_path):
    """The same exact answer for 1, 2, 4, 8 and 16 threads.

    A reduction can be wrong in ways that only appear at a particular thread count:
    a private copy left uninitialized shows up once there is more than one chunk, and
    a bad combiner shows up once the chunks are merged. Because binary16 sums 2048
    ones exactly in any order (see above), every thread count must give exactly 2048.
    """
    size = 2048
    sdfg = openmp_reduce_sdfg('half_omp_threads', 'lambda a, b: a + b', np.float16(0), size)
    path = str(tmp_path / 'half_omp_threads.sdfg')
    sdfg.save(path)
    sdfg.compile()  # populate the build cache so the subprocesses do not each rebuild

    script = THREAD_RUNNER.format(path=path, size=size)
    for threads in (1, 2, 4, 8, 16):
        env = dict(os.environ, OMP_NUM_THREADS=str(threads))
        proc = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, env=env)
        assert proc.returncode == 0, f'OMP_NUM_THREADS={threads}:\n{proc.stdout}\n{proc.stderr}'
        assert float(proc.stdout.strip().splitlines()[-1]) == float(size), \
            f'OMP_NUM_THREADS={threads} gave {proc.stdout.strip()}'


if __name__ == "__main__":
    test_float16_cpu_roundtrip_matches_numpy()
    test_float16_cpu_random_roundtrip_matches_numpy()
    test_float16_openmp_reduction_sum_is_exact()
    test_float16_openmp_reduction_sum_nonuniform_is_exact()
    test_float16_openmp_reduction_product_is_exact()
    test_float16_openmp_reduction_minmax('min', 'lambda a, b: min(a, b)', np.float16(np.inf), np.min)
    test_float16_openmp_reduction_minmax('max', 'lambda a, b: max(a, b)', np.float16(-np.inf), np.max)
    test_float16_openmp_reduction_scales_with_threads()
