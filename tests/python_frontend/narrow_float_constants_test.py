# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Constants of NumPy types narrower than double keep their type in tasklets (a bare literal is a double in C++)."""

import math

import numpy as np

import dace

N = dace.symbol("N")
SCALE = np.float32(math.sqrt(2 / math.pi))
HALF = np.float32(0.5)


@dace.program
def scale(a: dace.float32[N], b: dace.float32[N]):
    for i in dace.map[0:N]:
        b[i] = HALF * a[i] + a[i] * SCALE


def test_float32_constants_stay_single_precision():
    sdfg = scale.to_sdfg(simplify=True)
    code = sdfg.generate_code()[0].clean_code
    assert "float(0.5)" in code and f"float({float(SCALE)!r})" in code
    assert "(0.5 *" not in code

    a = np.random.default_rng(0).random(16).astype(np.float32)
    b = np.zeros_like(a)
    sdfg(a=a, b=b, N=16)
    assert np.allclose(b, HALF * a + a * SCALE)


if __name__ == "__main__":
    test_float32_constants_stay_single_precision()
