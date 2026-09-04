# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""CPU OpenMP reduction for the tile-op vectorizer's map-exit WCR.

Scalar reduction (``acc += A[i]``, ``max``, ``min``, ``acc = acc * A[i]``) on CPU:
tile folds with ``dace::tileops::tile_reduce`` (within-tile horizontal fold); map-exit
WCR lifts to OpenMP ``reduction(op:var)`` on ``#pragma omp parallel for`` (per-thread
privatized accumulator + end tree-reduce), not a contended per-iteration atomic.
Partial ``_nmr_out`` = single-element register; the accumulator the clause names
(``ReductionScalarLocalPrep`` privatizes ``acc`` into ``_priv_acc``) stays a true
``Scalar`` -- the clause needs a scalar, not a pointer-passed array slot.

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

import re
import shutil

import numpy as np
import pytest

import dace
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
    not a widened tile buffer -- and the accumulator the clause names stays a true Scalar
    (the clause needs a scalar, not a pointer-passed array slot).

    The accumulator is read out of the emitted ``reduction(op:VAR)`` clause rather than looked up
    under the program's own ``acc``: ``ReductionScalarLocalPrep`` privatizes the accumulator slot
    into a fresh ``_priv_acc`` scalar, so the name in the clause is the one that has to be a
    Scalar, and pinning the source name would test a descriptor the pragma never mentions."""
    sdfg = _vectorized(_PROGRAMS[kind][0])
    parts = [(k, d) for s in sdfg.all_sdfgs_recursive() for k, d in s.arrays.items()
             if k.startswith("_wcr_priv") and k.endswith("_acc")]
    assert parts, "expected an interposed _wcr_priv reduction partial"
    for k, d in parts:
        assert d.total_size == 1, f"{k} reduction partial must fold onto a single element, got {d.total_size}"

    clause = set(re.findall(r"reduction\([^:)]+:([^)]+)\)", _cpu_code(sdfg)))
    assert clause, "expected an OpenMP reduction clause to name the accumulator"
    for name in clause:
        descs = [d for s in sdfg.all_sdfgs_recursive() for k, d in s.arrays.items() if k == name]
        assert descs, f"reduction clause names {name}, which is not a descriptor of the SDFG"
        assert all(isinstance(d, dace.data.Scalar) for d in descs), \
            f"accumulator {name} must stay a Scalar, got {[type(d).__name__ for d in descs]}"


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


def test_happens_before_edge_does_not_cost_the_reduction_clause():
    """An EMPTY ordering edge in front of the reducing map must not read as a read of ``acc``.

    The clause is refused when the map body also READS the accumulator, because a privatized copy
    starts at the operator identity. An empty memlet reads nothing -- it is the happens-before
    edge ``StateFusionExtended`` leaves to hold a seed write (``acc = 0``) in front of the map
    that accumulates into it, re-anchored onto the map entry by ``MapFusionVertical``. Counting
    one as a read drops the map onto the per-element atomic path: measured at 2662 ms against
    40 ms on TSVC ``s319`` at the XL preset.
    """
    sdfg = dace.SDFG("seeded_reduction")
    sdfg.add_array("A", [N], dace.float64)
    sdfg.add_array("acc", [1], dace.float64, transient=True)
    sdfg.add_array("out", [1], dace.float64)
    state = sdfg.add_state("reduce", is_start_block=True)

    seed = state.add_write("acc")
    state.add_edge(state.add_tasklet("init", {}, {"o"}, "o = 0.0"), "o", seed, None, dace.Memlet("acc[0]"))

    entry, exit_ = state.add_map("red", {"i": "0:N"})
    tasklet = state.add_tasklet("add", {"_in"}, {"_out"}, "_out = _in")
    state.add_edge(state.add_read("A"), None, entry, "IN_A", dace.Memlet("A[0:N]"))
    state.add_edge(entry, "OUT_A", tasklet, "_in", dace.Memlet("A[i]"))
    wcr = "lambda x, y: x + y"
    state.add_edge(tasklet, "_out", exit_, "IN_acc", dace.Memlet("acc[0]", wcr=wcr))
    accumulated = state.add_access("acc")
    state.add_edge(exit_, "OUT_acc", accumulated, None, dace.Memlet("acc[0]", wcr=wcr))
    entry.add_in_connector("IN_A")
    entry.add_out_connector("OUT_A")
    exit_.add_in_connector("IN_acc")
    exit_.add_out_connector("OUT_acc")
    # The ordering edge, exactly as the fusion leaves it: seed -> the accumulating map's entry.
    state.add_nedge(seed, entry, dace.Memlet())
    state.add_edge(accumulated, None, state.add_write("out"), None, dace.Memlet("acc[0] -> [0]"))
    sdfg.validate()

    code = _cpu_code(sdfg)
    assert "reduction(+:" in code, "the happens-before edge cost the OpenMP reduction clause"
    assert "reduce_atomic" not in code, "the reduction fell through to the per-element atomic path"

    shutil.rmtree(os.path.join(".dacecache", sdfg.name), ignore_errors=True)
    csdfg = sdfg.compile()
    for nval in (64, 1000):
        a = (np.arange(nval) % 7).astype(np.float64)
        out = np.zeros(1, dtype=np.float64)
        csdfg(A=a.copy(), out=out, N=nval)
        assert out[0] == a.sum(), f"N={nval}: {float(out[0])!r} != {float(a.sum())!r}"


if __name__ == "__main__":
    for _kind in _PROGRAMS:
        test_emits_omp_reduction_clause(_kind)
        test_partial_folds_to_single_element(_kind)
    test_happens_before_edge_does_not_cost_the_reduction_clause()
    print("codegen ok")
