# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Zeros stored just before a lifted contraction (``zero_prior``): the einsum overwrites (``beta=0``) and the store,
once nothing else reads it and the einsum covers it, goes.

Counted as a prior value, the zeros kept ``beta=1`` and the store swept the whole accumulator once more per call:
``MapReduceFusion``'s init for ``np.sum(a[..., None] * b[:, None], axis=2)`` (cp2k_density_matrix_trs4's block
products), the same init fused into an unrelated map (gemver), and ``C[:] = 0`` ahead of an ``ijk`` accumulate loop.
"""

import numpy as np

import dace
from dace.libraries.blas import Einsum
from dace.libraries.standard.nodes.fill.node import FillLibraryNode
from dace.transformation.dataflow.lift_einsum import LiftEinsum
from dace.transformation.passes.canonicalize.loop_to_einsum import LoopToEinsum
from dace.transformation.passes.canonicalize.pipeline import canonicalize

N, K = (dace.symbol(s) for s in "NK")


def block_products(fill: str = "0", peek: bool = False, rows: str = "0:N") -> dace.SDFG:
    """``t[rows, i, j] = fill; t[n, i, j] += a[n, i, k] * b[n, k, j]; c = t``. ``peek`` copies ``t`` out between the
    fill and the sum, so the fill's zeros are observed."""
    sdfg = dace.SDFG(f"lift_einsum_zero_fill_{fill.replace('.', '_')}_{peek}")
    for name in "abc":
        sdfg.add_array(name, [N, K, K], dace.float64)
    sdfg.add_array("d", [N, K, K], dace.float64)
    sdfg.add_transient("t", [N, K, K], dace.float64)
    state = sdfg.add_state()
    filled = state.add_access("t")
    state.add_mapped_tasklet(
        "zero",
        {"n": rows, "i": "0:K", "j": "0:K"},
        {},
        f"o = {fill}",
        {"o": dace.Memlet("t[n, i, j]")},
        external_edges=True,
        output_nodes={"t": filled},
    )
    summed = state.add_access("t")
    _, entry, _ = state.add_mapped_tasklet(
        "mul",
        {"n": "0:N", "i": "0:K", "j": "0:K", "k": "0:K"},
        {"x": dace.Memlet("a[n, i, k]"), "y": dace.Memlet("b[n, k, j]")},
        "o = x * y",
        {"o": dace.Memlet("t[n, i, j]", wcr="lambda p, q: p + q")},
        external_edges=True,
        output_nodes={"t": summed},
    )
    if peek:
        state.add_nedge(filled, state.add_access("d"), dace.Memlet("t[0:N, 0:K, 0:K]"))
    state.add_nedge(filled, entry, dace.Memlet())
    state.add_nedge(summed, state.add_write("c"), dace.Memlet("t[0:N, 0:K, 0:K]"))
    sdfg.validate()
    return sdfg


def run(sdfg: dace.SDFG) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n, k = 3, 4
    rng = np.random.default_rng(0)
    a, b = rng.random((n, k, k)), rng.random((n, k, k))
    c, d = np.full((n, k, k), np.nan), np.full((n, k, k), np.nan)
    sdfg(a=a, b=b, c=c, d=d, N=n, K=k)
    return a, b, c, d


def fill_maps(sdfg: dace.SDFG) -> list[str]:
    """Every map and fill node left: the zero stores, as nothing else here is either."""
    return [
        n.map.label if isinstance(n, dace.nodes.MapEntry) else n.label
        for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, (dace.nodes.MapEntry, FillLibraryNode))
    ]


def einsum_of(sdfg: dace.SDFG) -> Einsum:
    (einsum,) = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Einsum)]
    return einsum


def test_zero_fill_ahead_of_the_sum_becomes_an_overwrite():
    sdfg = block_products()
    assert sdfg.apply_transformations(LiftEinsum) == 1
    sdfg.validate()
    assert dace.symbolic.equal_valued(0, einsum_of(sdfg).beta)
    assert fill_maps(sdfg) == []
    a, b, c, _ = run(sdfg)
    assert np.allclose(c, a @ b)


def test_observed_zeros_stay_and_the_sum_still_overwrites():
    """The copy into ``d`` reads the zeros before the sum: the store stays, ordered before the einsum, which still
    overwrites them."""
    sdfg = block_products(peek=True)
    assert sdfg.apply_transformations(LiftEinsum) == 1
    sdfg.validate()
    state = sdfg.start_state
    einsum = einsum_of(sdfg)
    assert dace.symbolic.equal_valued(0, einsum.beta)
    assert fill_maps(sdfg) == ["zero_map"]
    (zeroed,) = [
        n
        for n in state.data_nodes()
        if n.data == "t"
        and state.out_degree(n)
        and state.in_degree(n)
        and isinstance(state.in_edges(n)[0].src, dace.nodes.MapExit)
    ]
    assert einsum in set(state.bfs_nodes(zeroed))
    a, b, c, d = run(sdfg)
    assert np.allclose(c, a @ b) and np.all(d == 0)


def test_partial_zeros_are_folded_onto():
    """Only row 0 is zeroed: the other rows' prior value is uninitialized memory the sum must not be told is zero."""
    sdfg = block_products(rows="0:1")
    assert sdfg.apply_transformations(LiftEinsum) == 1
    assert dace.symbolic.equal_valued(1, einsum_of(sdfg).beta)
    assert fill_maps(sdfg) == ["zero_map"]


def test_nonzero_fill_is_folded_onto():
    sdfg = block_products(fill="1.0")
    assert sdfg.apply_transformations(LiftEinsum) == 1
    sdfg.validate()
    assert dace.symbolic.equal_valued(1, einsum_of(sdfg).beta)
    assert fill_maps(sdfg) == ["zero_map"]
    a, b, c, _ = run(sdfg)
    assert np.allclose(c, a @ b + 1.0)


@dace.program
def gemm_accum(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N]):
    C[:] = 0
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i, j] += A[i, k] * B[k, j]


def test_zeros_ahead_of_an_accumulate_loop_become_an_overwrite():
    """``C[:] = 0`` in the state before the ``ijk`` nest: the lifted einsum overwrites ``C`` and the zeroing goes.
    ``C`` starts at NaN, so a fold onto the caller's ``C`` would show."""
    sdfg = gemm_accum.to_sdfg(simplify=True)
    LoopToEinsum().apply_pass(sdfg, {})
    sdfg.validate()
    assert dace.symbolic.equal_valued(0, einsum_of(sdfg).beta)
    assert fill_maps(sdfg) == []
    n = 9
    rng = np.random.default_rng(1)
    a, b, c = rng.random((n, n)), rng.random((n, n)), np.full((n, n), np.nan)
    sdfg(A=a, B=b, C=c, N=n)
    assert np.allclose(c, a @ b)


@dace.program
def gemver(
    alpha: dace.float64,
    beta: dace.float64,
    A: dace.float64[N, N],
    u1: dace.float64[N],
    v1: dace.float64[N],
    u2: dace.float64[N],
    v2: dace.float64[N],
    w: dace.float64[N],
    x: dace.float64[N],
    y: dace.float64[N],
    z: dace.float64[N],
):
    A += np.multiply.outer(u1, v1) + np.multiply.outer(u2, v2)
    x += beta * y @ A + z
    w += alpha * A @ x


def test_gemver_reduction_inits_fused_into_other_maps_become_overwrites():
    """Fusion joins each matvec's zero init into a map computing something else (``beta * y``): only that store is
    the identity, so the einsums overwrite and the stores go while the rest of the map stays."""
    sdfg = gemver.to_sdfg(simplify=True)
    canonicalize(sdfg, target="cpu", validate=True)
    betas = [n.beta for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Einsum)]
    assert betas and all(dace.symbolic.equal_valued(0, b) for b in betas), betas
    n = 11
    rng = np.random.default_rng(2)
    args = dict(
        alpha=1.5,
        beta=1.2,
        A=rng.random((n, n)),
        **{k: rng.random(n) for k in ("u1", "v1", "u2", "v2", "w", "x", "y", "z")},
    )
    ref = {k: np.copy(v) for k, v in args.items()}
    ref["A"] += np.outer(ref["u1"], ref["v1"]) + np.outer(ref["u2"], ref["v2"])
    ref["x"] += ref["beta"] * ref["y"] @ ref["A"] + ref["z"]
    ref["w"] += ref["alpha"] * ref["A"] @ ref["x"]
    sdfg(**args, N=n)
    for k in ("A", "x", "w"):
        assert np.allclose(args[k], ref[k]), k


if __name__ == "__main__":
    test_zero_fill_ahead_of_the_sum_becomes_an_overwrite()
    test_observed_zeros_stay_and_the_sum_still_overwrites()
    test_partial_zeros_are_folded_onto()
    test_nonzero_fill_is_folded_onto()
    test_zeros_ahead_of_an_accumulate_loop_become_an_overwrite()
    test_gemver_reduction_inits_fused_into_other_maps_become_overwrites()
