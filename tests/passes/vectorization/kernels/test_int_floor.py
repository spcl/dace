# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Integer floor division (Python ``//``) through the vectorization backend.

``//`` lowers to ``int_floor`` (``dace::math::int_floor``), which the backend
must keep as a function call — both the classifier (``_BINOP_SYMBOLS``) and
the tasklet splitter (``ASTSplitter`` / ``_binary_expr``) recognise it, and
the templates dict carries ``vector_int_floor`` intrinsics (defined in
``cpu_vectorizable_math_common.h``) for the packed-vector form. This pins the
end-to-end numerical equivalence of a vectorized ``a[i] // k`` against the
unvectorized reference, and asserts the emitted body keeps ``int_floor`` as a
call (never the broken infix ``a int_floor k`` or a dropped divisor). The
loop-invariant scalar form is covered end-to-end by the TSVC s276 lane
condition.
"""
import copy

import numpy as np
import pytest

import dace
from dace.transformation.interstate import LoopToMap
from dace.transformation.passes.vectorization.vectorize_cpu import VectorizeCPU

N = dace.symbol("N")


@dace.program
def _floor_div_scalar(a: dace.int64[N], out: dace.int64[N]):
    for i in range(N):
        out[i] = a[i] // 3


@pytest.mark.parametrize("kernel,build", [
    (_floor_div_scalar, lambda rng, n: {
        "a": rng.integers(1, 1000, size=n, dtype=np.int64),
        "out": np.zeros(n, dtype=np.int64),
    }),
])
@pytest.mark.parametrize("n", [64, 65])
def test_vector_int_floor(kernel, build, n):
    rng = np.random.default_rng(seed=n)
    args_ref = build(rng, n)
    args_vec = {k: v.copy() for k, v in args_ref.items()}

    sdfg = copy.deepcopy(kernel.to_sdfg(simplify=False))
    sdfg.name = f"{kernel.name}_{n}_ref"
    sdfg.simplify(validate=True, validate_all=True)
    sdfg.apply_transformations_repeated(LoopToMap())
    sdfg.simplify()

    vsdfg = copy.deepcopy(sdfg)
    vsdfg.name = f"{kernel.name}_{n}_vec"
    VectorizeCPU(vector_width=8, fail_on_unvectorizable=True,
                 use_fp_factor=False, branch_normalization=True,
                 remainder_strategy="scalar").apply_pass(vsdfg, {})

    # ``//`` stays an ``int_floor(...)`` call in the vectorized body — never
    # an infix ``int_floor`` operator and never with the divisor dropped.
    bodies = [n_.code.as_string for n_, _ in vsdfg.all_nodes_recursive() if isinstance(n_, dace.nodes.Tasklet)]
    assert any("int_floor(" in b for b in bodies), bodies

    sdfg.compile()(**args_ref, N=n)
    vsdfg.compile()(**args_vec, N=n)
    np.testing.assert_array_equal(args_ref["out"], args_vec["out"])
