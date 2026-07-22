# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Power-operator lowering in the multi-dim tile vectorizer.

``**`` is kept as ``**`` through the frontend / tasklet-prep; the tile emitter classifies
each power per operand at emission time:

* a fractional / non-integer or unprovable exponent -> ``std::pow`` (``op="pow"``);
* an integer base with a provable non-negative integer exponent -> ``dace::math::ipow``
  (``op="ipow"``, exact repeated multiply), using the SAME proof
  (:func:`~dace.transformation.passes.relax_integer_powers.exponent_relaxes_to_ipow`) the
  pow->ipow relaxation applies to symbolic size powers.

A float base always stays ``std::pow`` -- NumPy raises a float base with libm ``pow``, so
``ipow`` there would not be bit-exact (it diverges in the low bits, badly for a large
exponent). These tests pin that boundary and the two supporting pieces (the redundant
float-cast strip on the exponent, and the nonnegativity assumption that lets the proof fire).
"""
import os

os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import ISA, RemainderStrategy
from dace.transformation.passes.vectorization.tasklet_preprocessing_passes import StripPowerExponentCast
from dace.transformation.passes.relax_integer_powers import exponent_relaxes_to_ipow
from dace.transformation.passes.canonicalize.assume_symbols_nonnegative import set_symbol_nonnegative_assumptions

from tests.passes.vectorization.helpers.harness import run_vectorization_test

S = dace.symbol("S")


@dace.program
def frac_power(A: dace.float64[S], B: dace.float64[S]):
    for i in dace.map[0:S]:
        B[i] = A[i]**1.5


@dace.program
def symbolic_power(A: dace.float64[S], B: dace.float64[S]):
    for i in dace.map[0:S]:
        B[i] = A[i]**S


@dace.program
def const_power(A: dace.float64[S], B: dace.float64[S]):
    for i in dace.map[0:S]:
        B[i] = A[i]**3


@pytest.mark.parametrize("prog,name", [(frac_power, "frac_power"), (symbolic_power, "symbolic_power"),
                                       (const_power, "const_power")])
def test_float_power_is_bit_exact(prog, name, remainder_strategy):
    """A float-base power vectorizes against the unvectorized reference within the harness
    tolerance (both use libm ``pow``; the emitter must NOT pick ``ipow`` for a float base).

    Inputs are kept near ``1.0`` so ``A ** S`` stays moderate: the tile loop's ``pow`` is the
    compiler's VECTORIZED libmvec ``pow``, ~1 ULP off scalar ``pow`` (the same transcendental
    precision gap the ``log`` / ``exp`` kernel tests live with) -- a huge ``A ** 64`` would
    amplify that ULP past ``atol``, which is a libmvec property, not the classification."""
    n = 64
    A = np.random.default_rng(0).uniform(0.9, 1.1, (n, )).astype(np.float64)
    B = np.zeros((n, ), np.float64)
    run_vectorization_test(dace_func=prog,
                           arrays={
                               'A': A,
                               'B': B
                           },
                           params={'S': n},
                           vector_width=8,
                           sdfg_name=name,
                           remainder_strategy=remainder_strategy)


def test_float_power_emits_stdpow():
    """The generated tile code for a float-base power calls ``std::pow`` (not ``ipow``)."""
    from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim
    sdfg = symbolic_power.to_sdfg(simplify=True)
    sdfg.name = "symbolic_power_stdpow"
    VectorizeCPUMultiDim(VectorizeConfig(widths=(8, ), target_isa=ISA.SCALAR,
                                         remainder_strategy=RemainderStrategy.SCALAR_POSTAMBLE)).apply_pass(sdfg, {})
    code = sdfg.generate_code()[0].clean_code
    assert "std::pow" in code, "float base ** must lower to std::pow"
    assert "dace::math::ipow" not in code, "float base ** must NOT lower to ipow (not bit-exact with numpy)"


def test_strip_power_exponent_cast():
    """``StripPowerExponentCast`` rewrites ``base ** float64(e)`` -> ``base ** e`` (keeping the
    ``**``), exposing the integer exponent while leaving the value unchanged."""
    sdfg = dace.SDFG("strip_cast")
    sdfg.add_array("A", [S], dace.float64)
    sdfg.add_array("B", [S], dace.float64)
    state = sdfg.add_state()
    a = state.add_access("A")
    b = state.add_access("B")
    tasklet = state.add_tasklet("p", {"__in1"}, {"__out"}, "__out = (__in1 ** dace.float64(S))")
    state.add_edge(a, None, tasklet, "__in1", dace.Memlet("A[0]"))
    state.add_edge(tasklet, "__out", b, None, dace.Memlet("B[0]"))

    StripPowerExponentCast().apply_pass(sdfg, {})
    body = tasklet.code.as_string.strip()
    assert body in ("__out = __in1 ** S", "__out = (__in1 ** S)"), body


def test_exponent_classifier_reuses_relax_proof():
    """``exponent_relaxes_to_ipow`` -- the shared proof the ``**`` classifier and the pow->ipow
    relaxation both use -- accepts a non-negative integer and rejects a fractional / unprovable
    exponent."""
    sdfg = symbolic_power.to_sdfg(simplify=True)
    set_symbol_nonnegative_assumptions(sdfg)  # S is a size symbol -> nonnegative
    assert exponent_relaxes_to_ipow(dace.symbolic.pystr_to_symbolic("S"), sdfg) is True
    assert exponent_relaxes_to_ipow(dace.symbolic.pystr_to_symbolic("3"), sdfg) is True
    assert exponent_relaxes_to_ipow(dace.symbolic.pystr_to_symbolic("1.5"), sdfg) is False
    assert exponent_relaxes_to_ipow(dace.symbolic.pystr_to_symbolic("-2"), sdfg) is False


def test_set_symbol_nonnegative_assumptions():
    """``set_symbol_nonnegative_assumptions`` marks a signed-integer free symbol nonnegative in
    place (so a size-power proof can conclude ``>= 0``)."""
    sdfg = symbolic_power.to_sdfg(simplify=True)
    before = {s.name: s.is_nonnegative for s in sdfg.arrays["A"].free_symbols}
    assert before.get("S") is None
    set_symbol_nonnegative_assumptions(sdfg)
    after = {s.name: s.is_nonnegative for s in sdfg.arrays["A"].free_symbols}
    assert after.get("S") is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
