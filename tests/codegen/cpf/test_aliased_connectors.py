# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A CPF helper takes each array once: a nested SDFG binding one container through two connectors (icon_gather's
``f0(&A[0], &A[0], ...)``) is merged into one connector before rendering (``FuseIdenticalConnectors``)."""

import os
import re

import numpy as np

import dace
from dace.codegen.cpf import Rendering, render
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from tests.codegen.cpf.conftest import assert_matches, build_standalone, call_standalone

nproma = dace.symbol("nproma", dtype=dace.int64, positive=True)
nlev = dace.symbol("nlev", dtype=dace.int64, positive=True)
nblks = dace.symbol("nblks", dtype=dace.int64, positive=True)
nnbr = dace.symbol("nnbr", dtype=dace.int64, positive=True)


@dace.program
def gather_twice(
    A: dace.float64[nblks, nlev, nproma],
    nbr_idx: dace.int64[nnbr, nblks, nproma],
    nbr_blk: dace.int64[nnbr, nblks, nproma],
    coef: dace.float64[nblks, nnbr, nproma],
    out: dace.float64[nblks, nlev, nproma],
    out_semi: dace.float64[nblks, nlev, nproma],
):
    for jk in range(nlev):
        acc = np.zeros((nblks, nproma), A.dtype)
        acc_semi = np.zeros((nblks, nproma), A.dtype)
        for n in range(nnbr):
            blk = nbr_blk[n, :, :]
            idx = nbr_idx[n, :, :]
            gathered = np.empty(blk.shape, A.dtype)
            for b in range(nblks):
                for p in range(nproma):
                    gathered[b, p] = A[blk[b, p], jk, idx[b, p]]
            acc[:] = acc + coef[:, n, :] * gathered
            semi_idx = nbr_idx[n, :, :]
            semi = np.empty(semi_idx.shape, A.dtype)
            for b in range(nblks):
                for p in range(nproma):
                    semi[b, p] = A[0, jk, semi_idx[b, p]]
            acc_semi[:] = acc_semi + coef[:, n, :] * semi
        out[:, jk, :] = acc
        out_semi[:, jk, :] = acc_semi


N = dace.symbol("N", dtype=dace.int64, positive=True)
M = dace.symbol("M", dtype=dace.int64, positive=True)
L = dace.symbol("L", dtype=dace.int64, positive=True)


@dace.program
def scatter_twice(val: dace.float64[N, M], idx: dace.int64[L, N], out: dace.float64[M, N], out2: dace.float64[M, N]):
    for jk in range(M):
        for n in range(L):
            v0 = val[:, jk].copy()
            for i in range(N):
                out[jk, idx[n, i]] = out[jk, idx[n, i]] + v0[i]
            v1 = val[:, jk].copy()
            for i in range(N):
                out2[jk, idx[n, i]] = out2[jk, idx[n, i]] + 2.0 * v1[i]


@dace.program
def gather_columns(A: dace.float64[N, M + 1], idx: dace.int64[N], B: dace.float64[N, M]):
    for t in range(2):
        for j in dace.map[0:M]:
            c0 = A[:, j].copy()
            c1 = A[:, j + 1].copy()
            for i in range(N):
                B[i, j] = B[i, j] + c0[idx[i]] * c1[idx[i]]


def containers_bound_twice(sdfg: dace.SDFG) -> list[str]:
    """``'<nested label>: <container> [<connectors>]'`` for each container a nested SDFG binds more than once."""
    found = []
    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.NestedSDFG):
            bindings = [(e.data.data, e.dst_conn) for e in state.in_edges(node)]
            bindings += [(e.data.data, e.src_conn) for e in state.out_edges(node)]
            by_container: dict[str, dict[str, None]] = {}
            for name, conn in bindings:
                if name is not None:
                    by_container.setdefault(name, {})[conn] = None
            found += [f"{node.label}: {name} {list(c)}" for name, c in by_container.items() if len(c) > 1]
    return found


def rendered_once(program: dace.frontend.python.parser.DaceProgram, name: str) -> tuple[list[str], Rendering]:
    """Canonicalize ``program`` and render it in C; the canonical form's aliased bindings and the rendering."""
    sdfg = program.to_sdfg(simplify=True)
    sdfg.name = name
    canonicalize(sdfg, target="cpu")
    finalize_for_target(sdfg, "cpu")
    aliased = containers_bound_twice(sdfg)
    result = render(sdfg, language="c")
    result.sdfg.validate()
    assert containers_bound_twice(result.sdfg) == []
    assert re.search(r"[(,] *&(\w+)\[0\][^;]*, *&\1\[0\]", result.code) is None, result.code
    return aliased, result


def test_a_helper_reading_one_array_at_two_places_takes_it_once():
    aliased, result = rendered_once(gather_twice, "cpf_gather_twice")
    # A loop nested by canonicalization binds A twice; rendering takes it once
    assert aliased == ["loop_body: A ['A', 'A_0']"]
    library = build_standalone(result.code, result.sdfg.name, language="c")
    rng = np.random.default_rng(7)
    sizes = {"nproma": 16, "nlev": 5, "nblks": 4, "nnbr": 3}
    a = rng.random((sizes["nblks"], sizes["nlev"], sizes["nproma"]))
    idx = rng.integers(0, sizes["nproma"], (sizes["nnbr"], sizes["nblks"], sizes["nproma"]), dtype=np.int64)
    blk = rng.integers(0, sizes["nblks"], (sizes["nnbr"], sizes["nblks"], sizes["nproma"]), dtype=np.int64)
    coef = rng.random((sizes["nblks"], sizes["nnbr"], sizes["nproma"]))
    expected = np.zeros_like(a)
    expected_semi = np.zeros_like(a)
    for jk in range(sizes["nlev"]):
        for n in range(sizes["nnbr"]):
            expected[:, jk, :] += coef[:, n, :] * a[blk[n], jk, idx[n]]
            expected_semi[:, jk, :] += coef[:, n, :] * a[0, jk, idx[n]]
    got = {"out": np.zeros_like(a), "out_semi": np.zeros_like(a)}
    call_standalone(library, result.sdfg, {"A": a, "nbr_idx": idx, "nbr_blk": blk, "coef": coef, **got, **sizes})
    assert_matches({"out": expected, "out_semi": expected_semi}, got)


def test_two_copies_of_one_column_that_canonicalization_folds_are_passed_once():
    aliased, result = rendered_once(scatter_twice, "cpf_scatter_twice")
    # The copies fold into reads of ``val`` after the last canonical cleanup: only the render's run merges them
    assert aliased == ["loop_body: val ['v0', 'v1']"]
    library = build_standalone(result.code, result.sdfg.name, language="c")
    rng = np.random.default_rng(8)
    sizes = {"N": 32, "M": 4, "L": 3}
    val = rng.random((sizes["N"], sizes["M"]))
    idx = rng.integers(0, sizes["N"], (sizes["L"], sizes["N"]), dtype=np.int64)
    expected = np.zeros((sizes["M"], sizes["N"]))
    expected2 = np.zeros((sizes["M"], sizes["N"]))
    for jk in range(sizes["M"]):
        for n in range(sizes["L"]):
            np.add.at(expected[jk], idx[n], val[:, jk])
            np.add.at(expected2[jk], idx[n], 2.0 * val[:, jk])
    got = {"out": np.zeros_like(expected), "out2": np.zeros_like(expected2)}
    call_standalone(library, result.sdfg, {"val": val, "idx": idx, **got, **sizes})
    assert_matches({"out": expected, "out2": expected2}, got)


def test_neighbouring_columns_read_inside_a_map_are_passed_once():
    aliased, result = rendered_once(gather_columns, "cpf_gather_columns")
    # ``A[:, j]`` and ``A[:, j + 1]`` overlap no element but share the array: one pointer, the hull of both subsets
    assert aliased == ["loop_body: A ['c0', 'c1']"]
    library = build_standalone(result.code, result.sdfg.name, language="c")
    rng = np.random.default_rng(9)
    sizes = {"N": 32, "M": 6}
    a = rng.random((sizes["N"], sizes["M"] + 1))
    idx = rng.integers(0, sizes["N"], sizes["N"], dtype=np.int64)
    expected = np.zeros((sizes["N"], sizes["M"]))
    for j in range(sizes["M"]):
        expected[:, j] = 2 * a[idx, j] * a[idx, j + 1]
    got = {"B": np.zeros_like(expected)}
    call_standalone(library, result.sdfg, {"A": a, "idx": idx, **got, **sizes})
    assert_matches({"B": expected}, got)


def test_a_self_copy_lowered_at_codegen_passes_its_array_once():
    # fv3_dycore's ``q[5, :] -> q[0, :]``: the copy becomes a nest only inside code generation, binding q twice
    sdfg = dace.SDFG("cpf_self_copy")
    sdfg.add_array("q", [6, N], dace.float64)
    state = sdfg.add_state()
    state.add_edge(
        state.add_read("q"),
        None,
        state.add_write("q"),
        None,
        dace.Memlet(data="q", subset="5, 0:N", other_subset="0, 0:N"),
    )
    result = render(sdfg, language="c")
    assert re.search(r"[(,] *&(\w+)\[0\][^;]*, *&\1\[0\]", result.code) is None, result.code
    library = build_standalone(result.code, result.sdfg.name, language="c")
    q = np.random.default_rng(10).random((6, 16))
    expected = q.copy()
    expected[0] = expected[5]
    call_standalone(library, result.sdfg, {"q": q, "N": 16})
    assert_matches({"q": expected}, {"q": q})


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    test_a_helper_reading_one_array_at_two_places_takes_it_once()
    test_two_copies_of_one_column_that_canonicalization_folds_are_passed_once()
    test_neighbouring_columns_read_inside_a_map_are_passed_once()
    test_a_self_copy_lowered_at_codegen_passes_its_array_once()
