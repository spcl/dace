# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""An index-set split point must hold one value for the whole loop it splits.

cegterg's gather ``for j: idx = unconv[j]; vc[:, j] = vc[:, idx]`` solves its broadcast conflict to
``x = idx``. The segment bounds re-read ``x`` every iteration while the body reassigns it, so the
segments skip or repeat iterations whenever ``unconv`` is not increasing.
"""
from typing import Tuple

import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate.loop_to_map import loop_varying_symbols
from dace.transformation.passes.canonicalize.pipeline import canonicalize

K = dace.symbol('K')
M = dace.symbol('M')
TRIALS = 40


@dace.program
def gather(e: dace.float64[K], ew: dace.float64[M], vc: dace.float64[M, M], unconv: dace.int64[K], count: dace.int64,
           nbase: dace.int64, width: dace.int64):
    for j in range(count):
        idx = int(unconv[j])
        ew[nbase + j] = e[idx]
        for ii in range(width):
            vc[ii, j] = vc[ii, idx]


def reference(e: np.ndarray, ew: np.ndarray, vc: np.ndarray, unconv: np.ndarray, count: int, nbase: int) -> None:
    for j in range(count):
        idx = int(unconv[j])
        ew[nbase + j] = e[idx]
        vc[:, j] = vc[:, idx]


@pytest.fixture(scope='module')
def canonical() -> dace.SDFG:
    sdfg = gather.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True, validate_all=False, target='cpu')
    return sdfg


def test_no_loop_bound_reads_a_symbol_its_body_assigns(canonical: dace.SDFG):
    offending = []
    for node, _ in canonical.all_nodes_recursive():
        if not isinstance(node, LoopRegion):
            continue
        header = {
            str(s)
            for c in (node.init_statement, node.loop_condition) if c is not None for s in c.get_free_symbols()
        }
        varying = loop_varying_symbols(node) & header
        if varying:
            offending.append((node.label, node.loop_condition.as_string, sorted(varying)))
    assert not offending, offending


def trial(rng: np.random.Generator) -> Tuple[np.ndarray, int, int]:
    """Indices in any order, repeats allowed: nothing but ``idx < K`` holds for them."""
    count = int(rng.integers(1, 9))
    unconv = np.zeros(8, dtype=np.int64)
    unconv[:count] = rng.integers(0, 8, size=count)
    return unconv, count, int(rng.integers(0, 9))


def test_split_gather_matches_numpy_for_unordered_indices(canonical: dace.SDFG):
    rng = np.random.default_rng(0)
    mismatches = []
    for number in range(TRIALS):
        unconv, count, nbase = trial(rng)
        e, ew, vc = rng.random(8), rng.random(16), rng.random((16, 16))
        want_ew, want_vc = ew.copy(), vc.copy()
        reference(e, want_ew, want_vc, unconv, count, nbase)
        canonical(e=e, ew=ew, vc=vc, unconv=unconv, count=count, nbase=nbase, width=16, K=8, M=16)
        if not (np.array_equal(ew, want_ew) and np.array_equal(vc, want_vc)):
            mismatches.append((number, unconv[:count].tolist()))
    assert not mismatches, mismatches


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__]))
