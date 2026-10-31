# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A scan over ``range(M, N)`` with ``M > N`` runs nothing, like the loop it replaces.

The scan form sizes its buffers by the trip count ``N - M`` and reads its seed ``y[M - 1]``; for ``M > N`` the
buffer size is negative (``std::bad_array_new_length`` at initialization). ``LoopToScan`` counts ``Max(0, N - M)``
when the facts do not prove the loop runs, and clamps the seed read into ``y``.
"""

import numpy as np

import dace
from dace.libraries.standard.nodes.scan import Scan
from dace.transformation.passes.lift_preprocess import LiftPreprocess
from dace.transformation.passes.loop_to_scan import LoopToScan
from tests.helpers.isolation import MISMATCH, exit_code

N, M = dace.symbol("N"), dace.symbol("M")


@dace.program
def scan_from_m(x: dace.float64[N], y: dace.float64[N]):
    for i in range(M, N):
        y[i] = y[i - 1] + x[i]


def lifted(name: str) -> dace.SDFG:
    sdfg = scan_from_m.to_sdfg(simplify=True)
    sdfg.name = name
    LiftPreprocess().apply_pass(sdfg, {})
    assert LoopToScan().apply_pass(sdfg, {}) == 1
    assert any(isinstance(node, Scan) for node, _ in sdfg.all_nodes_recursive())
    return sdfg


def test_empty_scan_runs_nothing():
    """Run in a spawned child: the unguarded scan aborts the process."""
    y = np.ones(8)
    code = exit_code(lifted("scan_empty_trip"), {"x": np.ones(8), "y": y, "N": 8, "M": 20}, [(y, np.ones(8))])
    assert code >= 0, f"killed by signal {-code}"
    assert code != MISMATCH, "an empty scan wrote to y"
    assert code == 0, code


def test_guarded_scan_still_scans():
    x, y = np.arange(8.0), np.ones(8)
    expected = y.copy()
    for i in range(3, 8):
        expected[i] = expected[i - 1] + x[i]
    lifted("scan_guarded_trip")(x=x, y=y, N=8, M=3)
    assert np.allclose(y, expected)


if __name__ == "__main__":
    test_empty_scan_runs_nothing()
    test_guarded_scan_still_scans()
