# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The Thomas tridiagonal solve: its back substitution is a reverse affine recurrence, lifted to a scan."""

import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes.scan import Scan, ScanOp
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import canonicalize
from tests.corpus.tsvc_2_5 import tsvc_2_5
from tests.corpus.tsvc_2_5.tsvc_2_5_numpy import ref_thomas_solve

N = dace.symbol("N", dtype=dace.int64)


def scan_ops(sdfg: dace.SDFG) -> list[ScanOp]:
    return [node.op for node, parent in sdfg.all_nodes_recursive() if isinstance(node, Scan)]


def loops(sdfg: dace.SDFG) -> list[str]:
    return [r.label for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]


@dace.program
def back_substitution(c: dace.float64[N], d: dace.float64[N], x: dace.float64[N]):
    x[N - 1] = d[N - 1]
    for i in range(N - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]


def tridiagonal_system(n: int) -> dict[str, np.ndarray]:
    """A diagonally dominant system, so the sequential reference is well conditioned."""
    rng = np.random.default_rng(n)
    a, c = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
    return {
        "a": a,
        "b": 2.5 + np.abs(a) + np.abs(c) + rng.random(n),
        "c": c,
        "d": rng.standard_normal(n),
        "x": np.zeros(n),
    }


# Reproducer: thomas_solve back substitution stayed sequential: a reverse affine recurrence (LLR thomas_solve)
@pytest.mark.parametrize("target", ["cpu", "gpu"])
@pytest.mark.parametrize("n", [2, 3, 1000])
def test_a_reverse_affine_recurrence_lifts_to_an_affine_scan(target: str, n: int):
    sdfg = back_substitution.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True, target=target)
    assert scan_ops(sdfg) == [ScanOp.AFFINE]
    assert not loops(sdfg), f"the recurrence loop survived: {loops(sdfg)}"
    rng = np.random.default_rng(n)
    c, d = rng.uniform(-0.9, 0.9, n), rng.standard_normal(n)
    want = np.empty(n)
    want[n - 1] = d[n - 1]
    for i in range(n - 2, -1, -1):
        want[i] = d[i] - c[i] * want[i + 1]
    x = np.zeros(n)
    sdfg(c=c, d=d, x=x, N=n)
    assert np.allclose(x, want, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("n", [2, 1000, 100000])
def test_the_thomas_back_substitution_is_a_scan_and_the_solve_matches_numpy(n: int):
    sdfg = tsvc_2_5.thomas_solve.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    assert scan_ops(sdfg) == [ScanOp.AFFINE]
    system = tridiagonal_system(n)
    want = {name: value.copy() for name, value in system.items()}
    ref_thomas_solve(**want)
    sdfg(**system, LEN_1D=n)
    assert np.allclose(system["x"], want["x"], rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    for target in ("cpu", "gpu"):
        for n in (2, 3, 1000):
            test_a_reverse_affine_recurrence_lifts_to_an_affine_scan(target, n)
    for n in (2, 1000, 100000):
        test_the_thomas_back_substitution_is_a_scan_and_the_solve_matches_numpy(n)
