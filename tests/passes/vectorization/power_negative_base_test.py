# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``x ** e`` with a negative base must not be lowered through ``exp(e * log(x))``.

``PowerOperatorExpansion._expand_pow`` used to rewrite a non-literal exponent
(the numpy ``power``-ufunc form ``__in1 ** __in2``, where the exponent arrives as
a connector rather than a bare ``ast.Constant``) to ``exp(__in2 * log(__in1))``.
That identity holds only for a POSITIVE base: ``log`` of a negative number is NaN,
so ``np.sin(x) ** 2`` produced NaN on every lane where ``sin(x) < 0`` (npbench
arc_distance). It now keeps ``**`` so the tile binop lowers it to ``std::pow``,
which computes a negative base with an integer exponent correctly and matches
numpy. A LITERAL integer exponent still takes the faster unrolled-multiply path.
"""
import os

os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import numpy as np
import pytest

import dace
from dace.transformation.passes.vectorization.tasklet_preprocessing_passes import _expand_pow
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim

N = dace.symbol('N')


def test_expand_pow_non_literal_exponent_keeps_pow_not_exp_log():
    """A connector/call exponent stays ``**`` (-> ``std::pow``); no ``exp``/``log``."""
    for src in ("__out = __in1 ** __in2", "__out = pow(__in1, __in2)"):
        out = _expand_pow(src)
        assert "log" not in out and "exp" not in out, f"{src!r} -> {out!r} still uses exp/log"
        assert "**" in out, f"{src!r} -> {out!r} dropped the power op"


def test_expand_pow_literal_integer_exponent_still_unrolls():
    """A literal integer exponent keeps the fast unrolled-product path (``x*x``)."""
    assert _expand_pow("__out = __in1 ** 2") == "__out = __in1 * __in1"
    # A non-integer literal has no unrolled form: it stays ``**`` -> ``std::pow`` (= sqrt).
    out = _expand_pow("__out = __in1 ** 0.5")
    assert "log" not in out and "exp" not in out and "**" in out, out


@dace.program
def sin_squared(x: dace.float64[N], y: dace.float64[N]):
    y[:] = np.sin(x)**2


@pytest.mark.parametrize("isa", ["SCALAR", "AVX512"])
def test_sin_squared_negative_base_vectorizes_without_nan(isa):
    """``np.sin(x) ** 2`` over a range where ``sin(x) < 0`` must vectorize to the
    numpy result, not NaN (the ``exp(2 * log(sin))`` regression)."""
    n = 64  # divisible by 8: no remainder lanes
    x = np.linspace(-3.0, 3.0, n)  # sin(x) < 0 for x < 0
    assert (np.sin(x) < 0).any(), "test must exercise a negative base"
    ref = np.sin(x)**2

    sdfg = sin_squared.to_sdfg(simplify=True)
    VectorizeCPUMultiDim(widths=(8, ), target_isa=isa, remainder_strategy='scalar_postamble').apply_pass(sdfg, {})
    sdfg.validate()
    y = np.zeros(n)
    sdfg(x=x.copy(), y=y, N=n)
    assert not np.isnan(y).any(), f"{isa}: NaN in output (exp/log negative-base regression)"
    assert np.allclose(y, ref, rtol=1e-9, atol=1e-12), f"{isa}: {y[:5]} != {ref[:5]}"


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
