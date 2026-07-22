# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""CPU OpenMP reduction for the tile-op vectorizer's map-exit WCR.

Scalar reduction (``acc += A[i]``, ``max``, ``min``, ``acc = acc * A[i]``) on CPU:
tile folds with ``dace::tileops::tile_reduce`` (within-tile horizontal fold); map-exit
WCR lifts to OpenMP ``reduction(op:var)`` on ``#pragma omp parallel for`` (per-thread
privatized accumulator + end tree-reduce), not a contended per-iteration atomic.
Partial ``_nmr_out`` = single-element register; accumulator ``acc`` stays a true
``Scalar`` (clause needs a scalar, not a pointer-passed array slot).

``min``/``max``/``*`` reach this path only via ``AugAssignToWCR``, which converts
their loop-carried ``acc = f(acc, A[i])`` (frontend emits combine-then-copyback
subgraph ``acc -> combine -> slice -> copyback -> acc``) into a WCR write so
``LoopToMap`` can parallelize. Asserts clause + tile fold + numeric exactness.
"""
import os

os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import shutil

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.transformation.interstate import LoopToMap
from dace.transformation.dataflow.wcr_conversion import AugAssignToWCR
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim

N = dace.symbol("N")


@dace.program
def _vsum32(A: dace.float32[N], out: dace.float32[1]):
    acc = dace.float32(0.0)
    for i in dace.map[0:N]:
        acc += A[i]
    out[0] = acc


@dace.program
def _vmax32(A: dace.float32[N], out: dace.float32[1]):
    acc = dace.float32(-1.0e30)
    for i in range(N):
        acc = max(acc, A[i])
    out[0] = acc


@dace.program
def _vmin32(A: dace.float32[N], out: dace.float32[1]):
    acc = dace.float32(1.0e30)
    for i in range(N):
        acc = min(acc, A[i])
    out[0] = acc


@dace.program
def _vprod32(A: dace.float32[N], out: dace.float32[1]):
    acc = dace.float32(1.0)
    for i in range(N):
        acc = acc * A[i]
    out[0] = acc


# (program, reduction operator string as it appears in the OMP clause)
_PROGRAMS = {"sum": (_vsum32, "+"), "max": (_vmax32, "max"), "min": (_vmin32, "min"), "prod": (_vprod32, "*")}


def _vectorized(prog):
    sdfg = prog.to_sdfg(simplify=True)
    # min/max/prod loop-carried -> WCR writes here; sum already map+WCR from frontend, unaffected.
    sdfg.apply_transformations_repeated(AugAssignToWCR)
    sdfg.apply_transformations_repeated(LoopToMap)
    sdfg.simplify()
    VectorizeCPUMultiDim(VectorizeConfig(widths=(8, ))).apply_pass(sdfg, {})
    return sdfg


def _cpu_code(sdfg):
    # The OMP pragma + tile fold land in the frame TU; concatenate every C++ object.
    return "\n".join(c.clean_code for c in sdfg.generate_code() if c.language == "cpp")


def _inputs(kind, nval):
    """Inputs whose reduction is bit-exact regardless of the (parallel, tiled)
    combine order -- so a value mismatch is a real miscompile, not reassociation."""
    if kind == "sum":
        # Small integers; the exact sum (< 2**24) is order-independent in float32.
        return (np.arange(nval) % 7).astype(np.float32), lambda a: a.sum()
    if kind in ("max", "min"):
        # max/min are exactly associative + commutative -> always order-independent.
        rng = np.random.default_rng(nval)
        a = (rng.permutation(nval) % 101 - 50).astype(np.float32)
        return a, (np.max if kind == "max" else np.min)
    # prod: powers of two -> every partial product is exact; balance the exponents
    # (k twos, k halves, rest ones) so the running magnitude never over/underflows.
    rng = np.random.default_rng(nval)
    k = nval // 4
    a = np.ones(nval, np.float32)
    a[:k] = 2.0
    a[k:2 * k] = 0.5
    return rng.permutation(a).astype(np.float32), np.prod


@pytest.mark.parametrize("kind", list(_PROGRAMS))
def test_emits_omp_reduction_clause(kind):
    """Parallel map carries ``reduction(op:acc)`` clause; body tiles the fold with
    ``tile_reduce``, not a per-iteration atomic."""
    prog, op = _PROGRAMS[kind]
    code = _cpu_code(_vectorized(prog))
    assert f"reduction({op}:" in code, f"expected an OpenMP reduction({op}:...) clause on the parallel map"
    assert "tile_reduce" in code, "expected a tile_reduce within-tile fold"


@pytest.mark.parametrize("kind", list(_PROGRAMS))
def test_partial_folds_to_single_element(kind):
    """The interposed reduction partial (``NormalizeWCRSource``'s ``_wcr_priv_*_acc`` on the
    ``NSDFG -> AccessNode -[wcr]-> MapExit`` boundary) folds onto a single element -- a scalar,
    not a widened tile buffer -- and the accumulator ``acc`` stays a true Scalar (required by
    the OMP ``reduction`` clause)."""
    sdfg = _vectorized(_PROGRAMS[kind][0])
    parts = [(k, d) for s in sdfg.all_sdfgs_recursive() for k, d in s.arrays.items()
             if k.startswith("_wcr_priv") and k.endswith("_acc")]
    assert parts, "expected an interposed _wcr_priv reduction partial"
    for k, d in parts:
        assert d.total_size == 1, f"{k} reduction partial must fold onto a single element, got {d.total_size}"
    accs = [d for s in sdfg.all_sdfgs_recursive() for k, d in s.arrays.items() if k == "acc"]
    assert accs and all(isinstance(d, dace.data.Scalar) for d in accs), "accumulator acc must stay a Scalar"


@pytest.mark.parametrize("kind", list(_PROGRAMS))
def test_numeric_exact(kind):
    """Bit-exact result under the (parallel, per-thread-privatized) reduction order."""
    sdfg = _vectorized(_PROGRAMS[kind][0])
    sdfg.name = f"cpu_reduction_{kind}"
    shutil.rmtree(os.path.join(".dacecache", sdfg.name), ignore_errors=True)
    csdfg = sdfg.compile()
    for nval in (64, 130, 257, 1000):
        a, ref = _inputs(kind, nval)
        out = np.zeros(1, dtype=np.float32)
        csdfg(A=a.copy(), out=out, N=nval)
        exp = ref(a)
        assert np.array_equal(out[0], exp), f"{kind} N={nval}: {float(out[0])!r} != {float(exp)!r}"


if __name__ == "__main__":
    for _kind in _PROGRAMS:
        test_emits_omp_reduction_clause(_kind)
        test_partial_folds_to_single_element(_kind)
    print("codegen ok")
