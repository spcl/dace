# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Readable-codegen correctness for OpenMP scalar reductions (ExpandReduceOpenMP).

The OpenMP reduce expansion emits a native tasklet whose body is a
``#pragma omp parallel for reduction(op:_out[0])`` loop. The readable connector-inlining must NOT
rewrite such a body: the pygments C++ lexer folds the ``#pragma`` line into one preprocessor token
(so a connector named in the clause survives as an undeclared name), and a scalar pointer-base inline
``&x`` pasted into ``_out[0]`` mis-parses as ``&(x[0])``. The generator therefore keeps the classic
connector copy for a preprocessor-directive body -- byte-identical to legacy for that tasklet.
"""
import numpy as np
import pytest

import dace

from conftest import (LEGACY, EXPERIMENTAL, use_implementation, generated_code, run_isolated, assert_outputs_equivalent,
                      experimental_available)

N = dace.symbol("N")


def _reduce_sdfg(op, masked=False):
    """out[0] = reduce_op(a[...]) over float64[N], via an OpenMP-pinned Reduce node."""
    wcr = {"sum": "lambda x, y: x + y", "max": "lambda x, y: max(x, y)"}[op]
    identity = {"sum": 0.0, "max": float(np.finfo(np.float64).min)}[op]
    sdfg = dace.SDFG(f"reduce_{op}{'_masked' if masked else ''}")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("out", [1], dace.float64)
    state = sdfg.add_state("main")
    rd = state.add_read("a")
    wr = state.add_write("out")
    red = state.add_reduce(wcr, None, identity)
    red.implementation = "OpenMP"
    subset = "a[0:N:2]" if masked else "a[0:N]"
    state.add_edge(rd, None, red, None, dace.Memlet(subset))
    state.add_edge(red, None, wr, None, dace.Memlet("out[0]"))
    sdfg.validate()
    return sdfg


def _run(sdfg, impl):

    def build_and_run():
        with use_implementation(impl):
            csdfg = sdfg.compile()
        rng = np.random.default_rng(0)
        a = rng.random(64)
        out = np.zeros(1)
        csdfg(a=a, out=out, N=64)
        return {"out": out}

    return run_isolated(build_and_run)


@pytest.mark.parametrize("op", ["sum", "max"])
@pytest.mark.parametrize("masked", [False, True])
def test_openmp_reduction_generates_clean(op, masked):
    """The readable generator must keep the reduction clause + connector copy, never the mangled
    ``&_out[0]`` / undeclared ``_out`` forms."""
    if not experimental_available():
        pytest.skip("experimental readable codegen not ready")
    sdfg = _reduce_sdfg(op, masked)
    with use_implementation(EXPERIMENTAL):
        code = generated_code(sdfg)
    clause_op = {"sum": "+", "max": "max"}[op]  # OpenMP clause names the operator, not the python op
    assert f"reduction({clause_op}" in code, "reduction clause dropped"
    # the mangled forms the bug produced must not appear
    assert "&_out[0]" not in code and "&rmax[0]" not in code, "scalar reduction sink mis-inlined as &x[0]"


@pytest.mark.parametrize("op", ["sum", "max"])
@pytest.mark.parametrize("masked", [False, True])
def test_openmp_reduction_bit_exact(op, masked):
    """Experimental output is bit-exact vs legacy (single OpenMP thread -> deterministic order)."""
    if not experimental_available():
        pytest.skip("experimental readable codegen not ready")
    sdfg = _reduce_sdfg(op, masked)
    legacy = _run(sdfg, LEGACY)
    experimental = _run(sdfg, EXPERIMENTAL)
    assert_outputs_equivalent(legacy, experimental, "cpu", label=f"reduce_{op}_{masked}")


if __name__ == "__main__":
    for op in ("sum", "max"):
        for masked in (False, True):
            test_openmp_reduction_generates_clean(op, masked)
            test_openmp_reduction_bit_exact(op, masked)
    print("ok")
