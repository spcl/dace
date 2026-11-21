# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""An integer count histogram ``o[b] = o[b] + 1`` is a parallel array reduction, like the float one.

Its load ``o[b]`` is an integer, so the frontend makes it a symbol, and demotion put the load in a state of its own,
apart from the store, behind a scalar copy: no ``AugAssignToWCR`` shape matched and the loop stayed sequential
(azimint_hist). The histogram buffer then also has to stay a local array, not a state-struct member, or the
``reduction(+:o[0:npt])`` clause is lost to per-element atomics.
"""

import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.canonicalize.pipeline import canonicalize

N = dace.symbol("N", dace.int64, positive=True)
npt = dace.symbol("npt", dace.int64, positive=True)


def histogram(dtype):
    @dace.program
    def counts(r: dace.float64[N], out: dace.float64[npt]):
        o = np.zeros(npt, dtype)
        for i in range(N):
            b = int(r[i] * npt)
            if b > npt - 1:
                b = npt - 1
            o[b] = o[b] + 1
        out[:] = o

    return counts


@pytest.mark.parametrize("dtype", [dace.int64, dace.float64])
def test_a_count_histogram_is_a_parallel_array_reduction(dtype):
    sdfg = histogram(dtype).to_sdfg(simplify=True)
    canonicalize(sdfg, target="cpu")
    assert not any(isinstance(r, LoopRegion) for r in sdfg.all_control_flow_regions()), "the counting loop must map"
    finalize_for_target(sdfg, "cpu")
    code = sdfg.generate_code()[0].clean_code
    assert "reduction(+:" in code and "reduce_atomic" not in code, "the bins must be reduced, not atomically added"

    r = np.random.default_rng(0).random(5000)
    out = np.zeros(17)
    sdfg(r=r, out=out, N=5000, npt=17)
    assert np.array_equal(out, np.bincount(np.minimum((r * 17).astype(np.int64), 16), minlength=17))


if __name__ == "__main__":
    test_a_count_histogram_is_a_parallel_array_reduction(dace.int64)
    test_a_count_histogram_is_a_parallel_array_reduction(dace.float64)
