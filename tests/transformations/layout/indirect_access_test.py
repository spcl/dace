# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Basic indirect-access layout support: detect a data-dependent gather ``x[col[i]]`` and lay out the
gathered array ``x`` by Shuffle -- transparently, through the indirection.

``indirect_accesses`` recognises the symbol-promotion form an indirect access takes in a sequential
loop (``sym := col[i]`` on the loop's interstate edge, then ``x[sym]``) and names the
``(index=col, data=x)`` pair. ``indirection_candidates`` then reorders the DATA array ``x`` by each
registered shuffle ``sigma``, composing ``sigma^-1`` onto the runtime index so
``x'[sigma^-1(col[i])] == x[col[i]]`` for any ``col``. Every candidate reproduces the numpy oracle;
the sweep only picks the physical layout of ``x``."""
import numpy
import pytest
import dace

from dace.libraries.layout.shuffle import register_shuffle
from dace.transformation.layout.brute_force import sweep, best, indirection_candidates
from dace.transformation.layout.indirect_access import IndirectAccess, indirect_accesses

N = dace.symbol("N")


@dace.program
def gather(val: dace.float64[N], col: dace.int64[N], x: dace.float64[N], y: dace.float64[N]):
    """A pure data-dependent gather ``y[i] = val[i] * x[col[i]]`` written as a sequential loop, so the
    indirection lowers to the promoted-symbol form (``col_index := col[i]``; ``x[col_index]``)."""
    for i in range(N):
        y[i] = val[i] * x[col[i]]


# Closed-form self-inverse-checked cell bijections on [0, N), valid for any N (registered at import
# so the sympy / C lowerings exist before the sweep builds the shuffle candidates).
register_shuffle("gather_cyc", "(i + 1) % N", "(i + N - 1) % N")
register_shuffle("gather_rev", "N - 1 - i", "N - 1 - i")


def make_inputs(n, seed=0):
    rng = numpy.random.default_rng(seed)
    return {
        "val": rng.random(n),
        "col": rng.permutation(n).astype(numpy.int64),  # any col works; a permutation keeps y a clean gather
        "x": rng.random(n),
    }


def oracle(val, col, x):
    return {"y": val * x[col]}


def test_indirect_accesses_detects_gather_pair():
    """The detector names the ``(index=col, data=x)`` gather pair from the promoted-symbol form."""
    sdfg = gather.to_sdfg(simplify=True)
    accesses = indirect_accesses(sdfg)
    assert IndirectAccess("col", "x", "gather") in accesses, accesses


def test_indirection_shuffle_candidates_all_verify():
    """Shuffle-on-data candidates for the detected gather all reproduce the oracle (transparent)."""
    n = 16
    inp = make_inputs(n)
    ref = oracle(inp["val"], inp["col"], inp["x"])

    cands = {}
    for name, apply in indirection_candidates("col", "x", 0, 1, ["gather_cyc", "gather_rev"]):

        def make(apply=apply):
            sdfg = gather.to_sdfg(simplify=True)
            apply(sdfg)
            return sdfg

        cands[name] = make

    def run(sdfg):
        y = numpy.zeros(n)
        sdfg(val=inp["val"].copy(), col=inp["col"].copy(), x=inp["x"].copy(), y=y, N=n)
        return {"y": y}

    results = sweep(cands, run, ref, do_time=False)
    assert all(r.correct for r in results), [(r.name, r.error) for r in results]
    assert best(results) is not None
    # the identity baseline plus the two shuffles (1-D x -> no permute candidates)
    assert set(cands) == {"noindir_x", "indir_shuffle_x_gather_cyc", "indir_shuffle_x_gather_rev"}


if __name__ == "__main__":
    test_indirect_accesses_detects_gather_pair()
    test_indirection_shuffle_candidates_all_verify()
    print("indirect access tests PASS")
