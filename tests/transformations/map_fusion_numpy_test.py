# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""MapFusion unit tests written as numpy-style ``@dace.program`` kernels.

Numpy elementwise / reduction expressions lower to Maps; consecutive producer ->
consumer maps fuse VERTICALLY, and independent maps over the same iteration space
sharing inputs fuse HORIZONTALLY. Every case checks value-preservation against a
direct numpy reference; structural assertions pin the map-count reduction.
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
from dace.transformation.dataflow import MapFusion, MapFusionVertical, MapFusionHorizontal

N = dace.symbol("N")
M = dace.symbol("M")


def _nmaps(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))


def _fuse(sdfg, horizontal=True):
    xforms = [MapFusionVertical]
    if horizontal:
        xforms.append(MapFusionHorizontal)
    return sdfg.apply_transformations_repeated(xforms, validate=True, validate_all=True)


def _check(prog, inputs, expected, horizontal=True, syms=None):
    """Build, fuse, run; return (maps_before, maps_after). Assert value-preserving."""
    syms = syms or {}
    sdfg = prog.to_sdfg(simplify=True)
    before = _nmaps(sdfg)
    _fuse(sdfg, horizontal=horizontal)
    after = _nmaps(sdfg)
    work = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in inputs.items()}
    sdfg(**work, **syms)
    for name, exp in expected.items():
        assert np.allclose(work[name], exp, rtol=1e-10, atol=1e-12), \
            f"{prog.name}/{name}: fused output diverged (max {np.max(np.abs(work[name]-exp)):.2e})"
    return before, after


# ---------------------------------------------------------------------------
# Vertical fusion: producer -> consumer chains.
# ---------------------------------------------------------------------------


def test_vertical_two_elementwise_chain():
    """tmp = a+1 ; out = tmp*2 -> the two maps fuse to one."""

    @dace.program
    def k(a: dace.float64[N], out: dace.float64[N]):
        tmp = a + 1.0
        out[:] = tmp * 2.0

    a = np.random.default_rng(0).random(64)
    before, after = _check(k, {"a": a, "out": np.zeros(64)}, {"out": (a + 1.0) * 2.0}, syms={"N": 64})
    assert before == 2 and after == 1


def test_vertical_three_elementwise_chain():
    """A chain of three elementwise maps collapses to one."""

    @dace.program
    def k(a: dace.float64[N], out: dace.float64[N]):
        t1 = a + 1.0
        t2 = t1 * 3.0
        out[:] = t2 - 2.0

    a = np.random.default_rng(1).random(64)
    before, after = _check(k, {"a": a, "out": np.zeros(64)}, {"out": (a + 1.0) * 3.0 - 2.0}, syms={"N": 64})
    assert before == 3 and after == 1


def test_vertical_2d_elementwise_chain():
    """2-D elementwise producer->consumer fuses (collapsed maps)."""

    @dace.program
    def k(a: dace.float64[N, M], out: dace.float64[N, M]):
        tmp = a * a
        out[:] = tmp + 1.0

    a = np.random.default_rng(2).random((12, 20))
    before, after = _check(k, {"a": a, "out": np.zeros((12, 20))}, {"out": a * a + 1.0}, syms={"N": 12, "M": 20})
    assert before >= 2 and after < before


def test_vertical_binary_then_unary():
    """out = (a + b) computed then squared -> fuse producer into consumer."""

    @dace.program
    def k(a: dace.float64[N], b: dace.float64[N], out: dace.float64[N]):
        s = a + b
        out[:] = s * s

    rng = np.random.default_rng(3)
    a, b = rng.random(64), rng.random(64)
    before, after = _check(k, {"a": a, "b": b, "out": np.zeros(64)}, {"out": (a + b)**2}, syms={"N": 64})
    assert before == 2 and after == 1


def test_vertical_fma_chain():
    """out = a*b + c as two maps (mul then add) -> fuse."""

    @dace.program
    def k(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], out: dace.float64[N]):
        p = a * b
        out[:] = p + c

    rng = np.random.default_rng(4)
    a, b, c = rng.random(64), rng.random(64), rng.random(64)
    before, after = _check(k, {"a": a, "b": b, "c": c, "out": np.zeros(64)}, {"out": a * b + c}, syms={"N": 64})
    assert before == 2 and after == 1


def test_vertical_long_chain_five():
    """Five chained elementwise maps fuse down to one."""

    @dace.program
    def k(a: dace.float64[N], out: dace.float64[N]):
        t1 = a + 1.0
        t2 = t1 * 2.0
        t3 = t2 - 3.0
        t4 = t3 * 0.5
        out[:] = t4 + 10.0

    a = np.random.default_rng(5).random(64)
    exp = ((a + 1.0) * 2.0 - 3.0) * 0.5 + 10.0
    before, after = _check(k, {"a": a, "out": np.zeros(64)}, {"out": exp}, syms={"N": 64})
    assert before == 5 and after == 1


# ---------------------------------------------------------------------------
# Horizontal fusion: independent maps sharing inputs / iteration space.
# ---------------------------------------------------------------------------


def test_horizontal_shared_inputs():
    """c = a+b ; d = a*b -- same range, shared inputs -> horizontal fuse."""

    @dace.program
    def k(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
        c[:] = a + b
        d[:] = a * b

    rng = np.random.default_rng(6)
    a, b = rng.random(64), rng.random(64)
    before, after = _check(k, {"a": a, "b": b, "c": np.zeros(64), "d": np.zeros(64)},
                           {"c": a + b, "d": a * b}, syms={"N": 64})
    assert before == 2 and after == 1


def test_horizontal_three_way():
    """Three independent maps over the same array fuse together."""

    @dace.program
    def k(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
        b[:] = a + 1.0
        c[:] = a * 2.0
        d[:] = a - 3.0

    a = np.random.default_rng(7).random(64)
    before, after = _check(k, {"a": a, "b": np.zeros(64), "c": np.zeros(64), "d": np.zeros(64)},
                           {"b": a + 1.0, "c": a * 2.0, "d": a - 3.0}, syms={"N": 64})
    assert before == 3 and after < before


# ---------------------------------------------------------------------------
# Reductions + mixed.
# ---------------------------------------------------------------------------


def test_elementwise_then_reduction_value_preserving():
    """out = sum((a+1)*(a+1)) -- elementwise producer feeds a reduction."""

    @dace.program
    def k(a: dace.float64[N], out: dace.float64[1]):
        t = (a + 1.0) * (a + 1.0)
        out[0] = np.sum(t)

    a = np.random.default_rng(8).random(64)
    before, after = _check(k, {"a": a, "out": np.zeros(1)}, {"out": np.array([np.sum((a + 1.0)**2)])},
                           syms={"N": 64})
    assert after <= before


def test_dot_product_chain_value_preserving():
    """out = sum(a*b) -- mul map feeds the reduction; value-preserving."""

    @dace.program
    def k(a: dace.float64[N], b: dace.float64[N], out: dace.float64[1]):
        p = a * b
        out[0] = np.sum(p)

    rng = np.random.default_rng(9)
    a, b = rng.random(128), rng.random(128)
    before, after = _check(k, {"a": a, "b": b, "out": np.zeros(1)}, {"out": np.array([np.sum(a * b)])},
                           syms={"N": 128})
    assert after <= before


def test_mixed_vertical_and_horizontal():
    """A vertical chain and a horizontal sibling in one kernel."""

    @dace.program
    def k(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N], d: dace.float64[N]):
        t = a + b            # producer
        c[:] = t * 2.0       # vertical consumer of t
        d[:] = a - b         # horizontal sibling (same range, shares a,b)

    rng = np.random.default_rng(10)
    a, b = rng.random(64), rng.random(64)
    before, after = _check(k, {"a": a, "b": b, "c": np.zeros(64), "d": np.zeros(64)},
                           {"c": (a + b) * 2.0, "d": a - b}, syms={"N": 64})
    assert after < before


# ---------------------------------------------------------------------------
# Refusal / no-op: nothing to fuse, or shapes incompatible.
# ---------------------------------------------------------------------------


def test_single_map_no_fuse():
    """One elementwise map -> nothing to fuse (no-op), value-preserving."""

    @dace.program
    def k(a: dace.float64[N], out: dace.float64[N]):
        out[:] = a + 1.0

    a = np.random.default_rng(11).random(64)
    before, after = _check(k, {"a": a, "out": np.zeros(64)}, {"out": a + 1.0}, syms={"N": 64})
    assert before == 1 and after == 1


def test_incompatible_shapes_not_vertically_fused():
    """A length-N map and a length-M map over different sizes do not fuse; the
    matmul-free 1-D outer + inner elementwise still computes correctly."""

    @dace.program
    def k(a: dace.float64[N], b: dace.float64[M], oa: dace.float64[N], ob: dace.float64[M]):
        oa[:] = a + 1.0
        ob[:] = b * 2.0

    rng = np.random.default_rng(12)
    a, b = rng.random(32), rng.random(48)
    before, after = _check(k, {"a": a, "b": b, "oa": np.zeros(32), "ob": np.zeros(48)},
                           {"oa": a + 1.0, "ob": b * 2.0}, syms={"N": 32, "M": 48})
    assert after == before  # different iteration spaces: not fused


@pytest.mark.parametrize("n", [1, 3, 33, 128])
def test_vertical_chain_value_preserving_across_sizes(n):
    """Vertical fusion stays value-preserving across a spread of sizes."""

    @dace.program
    def k(a: dace.float64[N], out: dace.float64[N]):
        t = a * a
        out[:] = t + a

    a = np.random.default_rng(n).random(n)
    _check(k, {"a": a, "out": np.zeros(n)}, {"out": a * a + a}, syms={"N": n})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
