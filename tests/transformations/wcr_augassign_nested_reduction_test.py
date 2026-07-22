# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``WCRToAugAssign`` must not revert a genuine cross-iteration reduction that lives
inside a NestedSDFG under an enclosing Map/loop.

Inside a NestedSDFG the enclosing map/loop iterators are invisible to the local
scope check, so a WCR write that is injective over its *inner* map but *invariant*
over an outer iterator (``C[k] +=`` accumulated over the outer map's ``i``) looks
conflict-free locally and used to be reverted to a plain ``C = C + x`` RMW. That
drops the reduction and forces a full-range (clobbering) boundary memlet. The gate
now walks the real enclosing scopes across the nested-SDFG boundary and keeps the
WCR -- the "map of reductions" shape (like a gemm k-reduction) surviving the
round-trip. See ``dace/transformation/dataflow/wcr_conversion.py``.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.dataflow.wcr_conversion import WCRToAugAssign

N = dace.symbol("N")


def _nested_reduction_sdfg(reduce_over_i: bool) -> dace.SDFG:
    """Outer ``map[i]`` over a NestedSDFG holding an inner ``map[k]`` that WCR-writes ``C``.

    ``reduce_over_i=True``  -> inner writes ``C[k]``    (invariant over outer ``i`` = reduction over ``i``).
    ``reduce_over_i=False`` -> inner writes ``C[i, k]`` (every outer/inner param in the index = injective).
    """
    cshape = [N] if reduce_over_i else [N, N]
    cidx = "c[k]" if reduce_over_i else "c[i, k]"
    inner = dace.SDFG("inner")
    inner.add_symbol("i", dace.int64)
    inner.add_array("arow", [N], dace.float64)
    inner.add_array("c", cshape, dace.float64)
    ist = inner.add_state("compute")
    ime, imx = ist.add_map("kmap", {"k": "0:N"})
    t = ist.add_tasklet("acc", {"__in"}, {"__out"}, "__out = __in")
    ist.add_memlet_path(ist.add_read("arow"), ime, t, dst_conn="__in", memlet=dace.Memlet("arow[k]"))
    ist.add_memlet_path(t, imx, ist.add_write("c"), src_conn="__out",
                        memlet=dace.Memlet(cidx, wcr="lambda x, y: x + y"))

    sdfg = dace.SDFG("outer_nested_reduction")
    sdfg.add_array("A", [N, N], dace.float64)
    sdfg.add_array("C", cshape, dace.float64)
    state = sdfg.add_state()
    me, mx = state.add_map("outer", {"i": "0:N"})
    nsdfg = state.add_nested_sdfg(inner, {"arow"}, {"c"}, {"N": "N", "i": "i"})
    state.add_memlet_path(state.add_read("A"), me, nsdfg, dst_conn="arow", memlet=dace.Memlet("A[i, 0:N]"))
    cbound = "C[0:N]" if reduce_over_i else "C[i, 0:N]"
    state.add_memlet_path(nsdfg, mx, state.add_write("C"), src_conn="c",
                          memlet=dace.Memlet(cbound, wcr="lambda x, y: x + y"))
    sdfg.validate()
    return sdfg


def _inner_wcr_edges(sdfg: dace.SDFG) -> int:
    """Count WCR edges written by a Tasklet inside any NestedSDFG."""
    n = 0
    for sd in sdfg.all_sdfgs_recursive():
        if sd.parent is None:
            continue
        for st in sd.all_states():
            for e in st.edges():
                if e.data is not None and e.data.wcr is not None and isinstance(e.src, nodes.Tasklet):
                    n += 1
    return n


def test_nested_cross_iteration_reduction_keeps_wcr():
    """An inner ``map[k]`` WCR write ``C[k]`` invariant over the outer map's ``i`` is a
    reduction over ``i``; WCRToAugAssign must keep it (not revert), and the result must
    be the correct column reduction ``C[k] += sum_i A[i, k]``."""
    sdfg = _nested_reduction_sdfg(reduce_over_i=True)
    assert _inner_wcr_edges(sdfg) == 1
    sdfg.apply_transformations_repeated(WCRToAugAssign)
    assert _inner_wcr_edges(sdfg) == 1, "the cross-iteration reduction WCR must be preserved"

    n = 8
    rng = np.random.default_rng(0)
    A = rng.random((n, n))
    C = rng.random(n)
    ref = C + A.sum(axis=0)
    Cw = C.copy()
    sdfg(A=A.copy(), C=Cw, N=n)
    assert np.allclose(Cw, ref), f"maxdiff {np.max(np.abs(Cw - ref))}"


def _toplevel_injective_sdfg() -> dace.SDFG:
    """Top-level ``map[i] { C[i] += A[i] }`` -- injective, not nested. The guard must
    leave this exactly as before (revert to RMW)."""
    sdfg = dace.SDFG("toplevel_injective")
    sdfg.add_array("A", [N], dace.float64)
    sdfg.add_array("C", [N], dace.float64)
    state = sdfg.add_state()
    me, mx = state.add_map("m", {"i": "0:N"})
    t = state.add_tasklet("acc", {"__in"}, {"__out"}, "__out = __in")
    state.add_memlet_path(state.add_read("A"), me, t, dst_conn="__in", memlet=dace.Memlet("A[i]"))
    state.add_memlet_path(t, mx, state.add_write("C"), src_conn="__out",
                          memlet=dace.Memlet("C[i]", wcr="lambda x, y: x + y"))
    sdfg.validate()
    return sdfg


def _toplevel_wcr_edges(sdfg: dace.SDFG) -> int:
    return sum(1 for st in sdfg.all_states() for e in st.edges()
              if e.data is not None and e.data.wcr is not None)


def test_toplevel_injective_write_still_reverts():
    """The nested-SDFG guard must not disturb the ordinary top-level path: an injective
    ``C[i] += A[i]`` map (``sdfg.parent is None``) still reverts to a plain RMW."""
    sdfg = _toplevel_injective_sdfg()
    assert _toplevel_wcr_edges(sdfg) > 0
    sdfg.apply_transformations_repeated(WCRToAugAssign)
    assert _toplevel_wcr_edges(sdfg) == 0, "an injective top-level write should revert to RMW"

    n = 8
    rng = np.random.default_rng(1)
    A = rng.random(n)
    C = rng.random(n)
    ref = C + A
    Cw = C.copy()
    sdfg(A=A.copy(), C=Cw, N=n)
    assert np.allclose(Cw, ref), f"maxdiff {np.max(np.abs(Cw - ref))}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
