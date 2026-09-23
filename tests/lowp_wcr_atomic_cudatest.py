# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Device atomic WCR on the 16-bit floats (``float16``, ``bfloat16``) under contention.

Every thread of one GPU map updates ``out[i % 3]`` through ``reduce_atomic``, so thousands of
threads hit the same element at once, and slots 0/1 -- and slot 2 with the never-written sentinel
slot 3 -- share one 32-bit word. HIP has no 16-bit atomic, so the update CASes that word; a splice
that dropped or clobbered the neighbouring half shows up as a wrong slot or a changed sentinel.

Inputs keep every partial result exactly representable (sum), or make the result independent of
order anyway (max, min, product of +-1), so the expected value is bit-exact whatever the order.
"""
import ml_dtypes
import numpy as np
import pytest

import dace

SLOTS = 3
SENTINEL = 3.140625  # exact in both fp16 and bf16
WCR = {
    "sum": "lambda a, b: a + b",
    "prod": "lambda a, b: a * b",
    "max": "lambda a, b: max(a, b)",
    "min": "lambda a, b: min(a, b)",
}
# (dace type, numpy type, updates per slot): a +1 / -2 / +0.5 walk that many steps stays exact.
DTYPES = {
    "float16": (dace.float16, np.float16, 2048),
    "bfloat16": (dace.bfloat16, ml_dtypes.bfloat16, 256),
}


def make_inputs(op: str, nptype, per_slot: int):
    """(A, out initial value): thread ``i`` adds ``A[i]`` into slot ``i % SLOTS``."""
    n = SLOTS * per_slot
    rng = np.random.default_rng(n)
    if op == "sum":
        a = np.tile(np.array([1.0, -2.0, 0.5]), per_slot)
        init = np.zeros(SLOTS + 1)
    elif op == "prod":
        a = rng.choice([-1.0, 1.0], size=n)
        init = np.ones(SLOTS + 1)
    else:
        # Integers up to 100 are exact in bf16 (8-bit significand) and fp16 alike.
        a = rng.integers(-100, 101, size=n).astype(np.float64)
        init = np.full(SLOTS + 1, -1000.0 if op == "max" else 1000.0)
    init[SLOTS] = SENTINEL
    return a.astype(nptype), init.astype(nptype)


def expected_values(op: str, a, init):
    """Slot-wise reduction of ``init`` and its residue class of ``a``, computed in float64."""
    fold = {"sum": np.sum, "prod": np.prod, "max": np.max, "min": np.min}[op]
    exp = init.astype(np.float64)
    for k in range(SLOTS):
        exp[k] = fold(np.append(a[k::SLOTS].astype(np.float64), exp[k]))
    return exp.astype(init.dtype)


def build_sdfg(op: str, dtype, n: int) -> dace.SDFG:
    sdfg = dace.SDFG(f"lowp_wcr_atomic_{op}_{dtype.to_string()}")
    sdfg.add_array("A", [n], dtype)
    sdfg.add_array("out", [SLOTS + 1], dtype)
    state = sdfg.add_state()
    state.add_mapped_tasklet("update",
                             dict(i=f"0:{n}"),
                             dict(a=dace.Memlet("A[i]")),
                             "o = a",
                             dict(o=dace.Memlet(f"out[i % {SLOTS}]", wcr=WCR[op])),
                             external_edges=True)
    sdfg.apply_gpu_transformations()
    return sdfg


@pytest.mark.gpu
@pytest.mark.parametrize("dtype_name", list(DTYPES))
@pytest.mark.parametrize("op", list(WCR))
def test_contended_atomic_is_exact_and_spares_neighbour(op, dtype_name):
    dtype, nptype, per_slot = DTYPES[dtype_name]
    a, init = make_inputs(op, nptype, per_slot)
    sdfg = build_sdfg(op, dtype, a.size)
    device_code = "\n".join(c.clean_code for c in sdfg.generate_code() if c.title == "CUDA")
    assert "reduce_atomic" in device_code, "the WCR did not lower to a device atomic"
    out = init.copy()
    sdfg(A=a, out=out)
    exp = expected_values(op, a, init)
    assert np.array_equal(out.view(np.uint16), exp.view(np.uint16)), \
        f"{op}/{dtype_name}: got {out.astype(np.float64)}, expected {exp.astype(np.float64)}"


if __name__ == "__main__":
    for op in WCR:
        for dtype_name in DTYPES:
            test_contended_atomic_is_exact_and_spares_neighbour(op, dtype_name)
