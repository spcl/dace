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


def openmp_reduce_sdfg(name, wcr, identity, size):
    """A float16 Reduce node pinned to the OpenMP expansion.

    Lowers to exactly the pattern this exercises::

        #pragma omp parallel for reduction(<op>: _out[0])
        for (int _i0 = 0; _i0 < size; ++_i0)
            _out[0] <op>= _in[_i0 * 1];

    ``dace::float16`` is a class type on the host, so the clause needs both a
    ``operator<op>=`` and a ``#pragma omp declare reduction`` to be accepted at all.
    """
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [size], dace.float16)
    sdfg.add_array('s', [1], dace.float16)
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
