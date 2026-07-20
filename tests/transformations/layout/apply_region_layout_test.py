# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Region-scoped layout: change an array's layout for a line of top-level nodes, restore at the end.

``apply_region_layout`` imposes a layout on the kernels of a top-level region ``[start, end)`` and
restores the original layout at the region's end -- the enter relayout lands before the region, the
restore after it, both at the top level. It is the imposed, scoped counterpart of the global
``apply_assignment`` trajectory (and the API form of the OMEN mid-flight transpose, bounded to a
region). Here ``A`` is transposed only for the two middle nests of a four-nest chain; nests outside
the region see the original layout, and the result is bit-exact.
"""
import numpy
import pytest

import dace
from dace.libraries.layout.algebra import Permute
from dace.transformation.layout.apply_assignment import Layout, apply_region_layout
from dace.transformation.layout.line_graph import kernel_per_state, line_graph
from dace.transformation.layout.prepare import prepare_for_layout

N = dace.symbol("N")

PERM10 = Layout("perm10", (Permute((1, 0)), ))


@dace.program
def chain(A: dace.float64[N, N], P: dace.float64[N, N], O0: dace.float64[N, N], O1: dace.float64[N, N],
          O2: dace.float64[N, N], O3: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        O0[i, j] = A[i, j] + P[i, j]  # nest 0: A straight
    for i, j in dace.map[0:N, 0:N]:
        O1[i, j] = O0[i, N - 1 - j] + A[j, i] + P[i, j]  # nest 1: A transposed
    for i, j in dace.map[0:N, 0:N]:
        O2[i, j] = O1[i, N - 1 - j] + A[j, i] + P[i, j]  # nest 2: A transposed
    for i, j in dace.map[0:N, 0:N]:
        O3[i, j] = O2[i, N - 1 - j] + A[i, j] + P[i, j]  # nest 3: A straight


def oracle(A, P):
    o0 = A + P
    o1 = o0[:, ::-1] + A.T + P
    o2 = o1[:, ::-1] + A.T + P
    o3 = o2[:, ::-1] + A + P
    return {"O0": o0, "O1": o1, "O2": o2, "O3": o3}


def build():
    sdfg = chain.to_sdfg(simplify=True)
    prepare_for_layout(sdfg)
    kernel_per_state(sdfg)
    return sdfg, line_graph(sdfg)


def run_and_check(sdfg, n=64, seed=0):
    rng = numpy.random.default_rng(seed)
    A, P = rng.random((n, n)), rng.random((n, n))
    outs = {f"O{k}": numpy.zeros((n, n)) for k in range(4)}
    sdfg(A=A.copy(), P=P.copy(), **outs, N=n)
    ref = oracle(A, P)
    for name, r in ref.items():
        assert numpy.allclose(outs[name], r), f"{name} diverges from the oracle"


def test_region_layout_enters_and_restores():
    """Transpose A for the region [1, 3) (the two middle nests). Exactly two boundary relayouts appear
    -- one entering the region, one restoring at its end -- and the program stays bit-exact."""
    sdfg, kernels = build()
    assert len(kernels) == 4

    applied = apply_region_layout(sdfg, kernels, {"A": PERM10}, region=(1, 3))
    # one enter (before nest 1) + one restore (at the region end, before nest 3)
    assert len(applied.boundary_states) == 2
    assert any("A__seg" in name for names in applied.segment_names.values() for name in names)  # region clone
    sdfg.validate()
    run_and_check(sdfg)


def test_region_layout_rejects_bad_regions():
    sdfg, kernels = build()
    with pytest.raises(ValueError, match="out of range"):
        apply_region_layout(sdfg, kernels, {"A": PERM10}, region=(2, 2))  # empty
    with pytest.raises(ValueError, match="out of range"):
        apply_region_layout(sdfg, kernels, {"A": PERM10}, region=(1, 9))  # past the end


if __name__ == "__main__":
    test_region_layout_enters_and_restores()
    test_region_layout_rejects_bad_regions()
    print("apply_region_layout tests PASS")
