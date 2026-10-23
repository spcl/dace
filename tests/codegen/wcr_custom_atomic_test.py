# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A WCR with a lambda no reduction type names updates through a compare-and-swap loop on the element: every
conflicting update lands, under a parallel map, for ``float`` and ``double``."""

import numpy as np
import pytest

import dace
from dace import dtypes

N = dace.symbol("N")


def custom_scatter(dtype) -> dace.SDFG:
    """``out[idx[i]] = out[idx[i]] - val[i]`` over a parallel map, as the WCR lambda ``a - b``."""
    sdfg = dace.SDFG(f"wcr_custom_atomic_{dtype.to_string()}")
    sdfg.add_array("idx", [N], dace.int64)
    sdfg.add_array("val", [N], dtype)
    sdfg.add_array("out", [4], dtype)
    state = sdfg.add_state()
    state.add_mapped_tasklet(
        "scatter",
        {"i": "0:N"},
        {"k": dace.Memlet("idx[i]"), "v": dace.Memlet("val[i]")},
        "o[k] = v",
        {"o": dace.Memlet("out[0:4]", wcr="lambda a, b: a - b")},
        schedule=dtypes.ScheduleType.CPU_Multicore,
        external_edges=True,
    )
    return sdfg


@pytest.mark.parametrize("dtype", [dace.float32, dace.float64])
def test_conflicting_custom_wcr_updates_all_land(dtype):
    sdfg = custom_scatter(dtype)
    code = sdfg.generate_code()[0].clean_code
    assert "reduce_atomic" in code
    n = 100000
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 4, n).astype(np.int64)
    val = np.ones(n, dtype=dtype.type)
    out = np.zeros(4, dtype=dtype.type)
    sdfg(idx=idx, val=val, out=out, N=n)
    assert np.array_equal(out, -np.bincount(idx, minlength=4).astype(dtype.type))


if __name__ == "__main__":
    test_conflicting_custom_wcr_updates_all_land(dace.float32)
    test_conflicting_custom_wcr_updates_all_land(dace.float64)
