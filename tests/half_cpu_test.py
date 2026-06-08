# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

import dace
import numpy as np

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


if __name__ == "__main__":
    test_float16_cpu_roundtrip_matches_numpy()
    test_float16_cpu_random_roundtrip_matches_numpy()
