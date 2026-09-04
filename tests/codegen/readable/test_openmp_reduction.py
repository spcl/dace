# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Readable-codegen correctness for OpenMP scalar reductions (ExpandReduceOpenMP).

``ExpandReduceOpenMP`` lowers onto the ``dace::reduce`` runtime facility, so the ``reduction``
clause lives in ``dace/reduction.h`` and the generated file carries only the call: one
``::dace::reduce::<op>(base, count, stride, seed)`` for a full reduction, and
``#pragma omp parallel for`` over the kept axes around a ``::dace::reduce::seq::<op>`` call for a
partial one. Never a per-element ``reduce_atomic``.

What the readable generator must not do to that tasklet is unchanged: a scalar pointer-base inline
``&x`` pasted into ``_out[0]`` mis-parses as ``&(x[0])``, and a connector named inside a ``#pragma``
line survives as an undeclared name (the pygments C++ lexer folds the line into one preprocessor
token). The full-reduction body now has no ``#pragma`` at all, so its connectors ARE inlined -- and
must inline to the array itself (``out[0] = ::dace::reduce::sum(a, ...)``), not to ``&out[0]``.
"""
import functools
import importlib.util
import os

import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes.reduce import Reduce
from dace.transformation.dataflow import MapFusion
from dace.transformation.interstate import LoopToMap

from tests.codegen.readable.conftest import (LEGACY, EXPERIMENTAL, use_implementation, generated_code, run_isolated,
                                             assert_outputs_equivalent, experimental_available)

N = dace.symbol("N")


@functools.lru_cache(maxsize=1)
def openmp_reduce_available():
    """True iff the OpenMP reduce expansion generates AND compiles in this build (an extended-only
    feature; the CPU-only PR branch off main lacks the full lowering, so these reduction cases skip
    there). Probes with a compile in a forked child so an off-main failure cannot crash pytest."""

    def probe():
        with use_implementation(EXPERIMENTAL):
            s = dace.SDFG("omp_probe")
            s.add_array("a", [8], dace.float64)
            s.add_array("o", [1], dace.float64)
            st = s.add_state("m")
            red = st.add_reduce("lambda x, y: x + y", None, 0.0)
            red.implementation = "OpenMP"
            st.add_edge(st.add_read("a"), None, red, '_in', dace.Memlet("a[0:8]"))
            st.add_edge(red, '_out', st.add_write("o"), None, dace.Memlet("o[0]"))
            s.validate()
            s.compile()
        return {}

    try:
        run_isolated(probe, timeout=120)
        return True
    except Exception:  # noqa: BLE001 - expansion absent / unsupported off main
        return False


def _require_omp_reduce():
    # Only the Reduce expansion is build-dependent; the generator itself is required.
    assert experimental_available(), "the readable CPU generator is not wired up"
    if not openmp_reduce_available():
        pytest.skip("OpenMP reduce expansion not available in this build")


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
    state.add_edge(rd, None, red, '_in', dace.Memlet(subset))
    state.add_edge(red, '_out', wr, None, dace.Memlet("out[0]"))
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
    """A full reduction lowers to ONE ``dace::reduce`` call, correctly inlined and never atomic."""
    _require_omp_reduce()
    sdfg = _reduce_sdfg(op, masked)
    with use_implementation(EXPERIMENTAL):
        code = generated_code(sdfg)
    # The call, with the operands the shape demands: stride 2 for the masked ``a[0:N:2]`` subset.
    stride = "2" if masked else "1"
    call = f"::dace::reduce::{op}(a, (long)("
    assert call in code, f"{call} missing -- reduce did not lower through the runtime facility"
    # Pins the stride AND the seed operand: the fold starts from the output element it writes back.
    assert f"(long)({stride}), out[0])" in code, f"wrong stride/seed: expected {stride} for masked={masked}"
    assert "out[0] = ::dace::reduce::" in code, "reduction sink is not the output array element"
    # The clause moved into the header, so the generated file must carry NO reduction pragma and no
    # per-element atomic for this shape. Asserted at its new home so "clause dropped" stays covered.
    assert "reduction(" not in code, "a full reduction must not emit its own clause any more"
    assert "reduce_atomic" not in code, "reduce lowered to a per-element atomic"
    header = os.path.join(os.path.dirname(os.path.abspath(dace.__file__)), "runtime", "include", "dace", "reduction.h")
    clause_op = {"sum": "+", "max": "max"}[op]
    assert f"reduction({clause_op} : acc)" in open(header).read(), "reduction clause dropped from the header"
    # the mangled forms the bug produced must not appear
    assert "&_out[0]" not in code and "&rmax[0]" not in code, "scalar reduction sink mis-inlined as &x[0]"
    assert "&out[0]" not in code, "scalar reduction sink mis-inlined as &out[0]"


def _partial_reduce_sdfg():
    """``out[i] = sum(a[i, :])`` over float64[8, 16] -- the kept-axis shape of the expansion."""
    sdfg = dace.SDFG("reduce_partial")
    sdfg.add_array("a", [8, 16], dace.float64)
    sdfg.add_array("out", [8], dace.float64)
    state = sdfg.add_state("main")
    red = state.add_reduce("lambda x, y: x + y", [1], 0.0)
    red.implementation = "OpenMP"
    state.add_edge(state.add_read("a"), None, red, '_in', dace.Memlet("a[0:8, 0:16]"))
    state.add_edge(red, '_out', state.add_write("out"), None, dace.Memlet("out[0:8]"))
    sdfg.validate()
    return sdfg


def test_openmp_partial_reduction_generates_clean():
    """A partial reduction keeps its own loop over the kept axis and calls the SEQUENTIAL entry."""
    _require_omp_reduce()
    sdfg = _partial_reduce_sdfg()
    with use_implementation(EXPERIMENTAL):
        code = generated_code(sdfg)
    assert "#pragma omp parallel for" in code, "kept-axis loop lost its parallel pragma"
    assert "::dace::reduce::seq::sum(" in code, "call under a parallel loop must be the sequential entry"
    assert "::dace::reduce::sum(" not in code, "parallel entry emitted under an enclosing parallel loop"
    assert "reduce_atomic" not in code, "reduce lowered to a per-element atomic"
    assert "&_out[0]" not in code and "&out[0]" not in code, "scalar reduction sink mis-inlined as &x[0]"

    def build_and_run():
        with use_implementation(EXPERIMENTAL):
            csdfg = sdfg.compile()
        a = np.random.default_rng(0).random((8, 16))
        out = np.zeros(8)
        csdfg(a=a, out=out)
        return {"out": out, "want": a.sum(axis=1)}

    res = run_isolated(build_and_run)
    assert np.allclose(res["out"], res["want"], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("op", ["sum", "max"])
@pytest.mark.parametrize("masked", [False, True])
def test_openmp_reduction_bit_exact(op, masked):
    """Experimental output is bit-exact vs legacy (single OpenMP thread -> deterministic order)."""
    _require_omp_reduce()
    sdfg = _reduce_sdfg(op, masked)
    legacy = _run(sdfg, LEGACY)
    experimental = _run(sdfg, EXPERIMENTAL)
    assert_outputs_equivalent(legacy, experimental, "cpu", label=f"reduce_{op}_{masked}")
    # An absolute reference too, so the pair cannot agree by being equally wrong.
    a = np.random.default_rng(0).random(64)
    reduced = a[0:64:2] if masked else a
    want = reduced.sum() if op == "sum" else reduced.max()
    assert np.allclose(experimental["out"], want, rtol=1e-12, atol=1e-12), (experimental["out"], want)


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
    _require_omp_reduce()

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
    test_openmp_partial_reduction_generates_clean()
    for k in _SCALAR_REDUCTION_KERNELS:
        test_scalar_reduction_kernel_compiles(k)
    print("ok")
