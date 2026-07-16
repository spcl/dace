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


import importlib.util
import os

from dace.libraries.standard.nodes.reduce import Reduce
from dace.transformation.dataflow import MapFusion
from dace.transformation.interstate import LoopToMap

# Real corpus kernels whose OpenMP-expanded SCALAR reductions previously mis-compiled under the
# readable generator (``&x[0]`` on a scalar sink + an undeclared ``_out`` in the reduction clause).
_SCALAR_REDUCTION_KERNELS = {
    "azimint_hist": "map_reduce/azimint_hist.py",
    "azimint_naive": "map_reduce/azimint_naive.py",
    "channel_flow": "structured_grids/channel_flow.py",
    "nbody": "n_body_methods/nbody.py",
}


def _load_corpus(relpath):
    path = os.path.join(os.path.dirname(__file__), "..", "..", "corpus", "npbench", relpath)
    spec = importlib.util.spec_from_file_location("k_" + os.path.basename(relpath), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CORPUS["program"]


@pytest.mark.parametrize("kernel", list(_SCALAR_REDUCTION_KERNELS))
def test_scalar_reduction_kernel_compiles(kernel):
    """Each kernel, pinned to the OpenMP reduce expansion, must generate + compile under the readable
    generator with no mangled scalar-sink subscript."""
    if not experimental_available():
        pytest.skip("experimental readable codegen not ready")

    def build_and_run():
        with use_implementation(EXPERIMENTAL):
            sdfg = _load_corpus(_SCALAR_REDUCTION_KERNELS[kernel]).to_sdfg(simplify=True)
            sdfg.apply_transformations_repeated(LoopToMap)
            sdfg.apply_transformations_repeated(MapFusion)
            sdfg.simplify()
            for n, _ in sdfg.all_nodes_recursive():
                if isinstance(n, Reduce):
                    n.implementation = "OpenMP"
            code = generated_code(sdfg)
            assert "&_out[0]" not in code, "scalar reduction sink mis-inlined as &_out[0]"
            sdfg.compile()  # raises CompilationError on the old `&rmax[0]` / undeclared `_out` bug
        return {}

    run_isolated(build_and_run)


if __name__ == "__main__":
    for op in ("sum", "max"):
        for masked in (False, True):
            test_openmp_reduction_generates_clean(op, masked)
            test_openmp_reduction_bit_exact(op, masked)
    for k in _SCALAR_REDUCTION_KERNELS:
        test_scalar_reduction_kernel_compiles(k)
    print("ok")
