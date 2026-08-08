# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Mimic the OMEN data-layout optimization with the layout API DIRECTLY -- no brute-force sweep, no
cost-model DP. The layout is IMPOSED, not searched.

The OMEN SCBA reformulation transposes the Green's-function tensor ``G`` between its two phases: the
RGF-like producer streams it one way, the SSE window consumer the other (arXiv:1912.10024). On the
local k10 witness (``kernels/k10_omen_windowed_contraction.py``) that decision is the order of ``G``'s
two batch axes ``(NA, NE)``. Here we express it by HAND, two ways:

1. Global -- one ``PermuteDimensions`` call stores ``G`` producer-order for the whole program (the k10
   candidate, applied directly instead of enumerated).
2. Mid-flight -- ``apply_assignment`` with a hand-built ``{G: [identity, batch-transpose]}`` inserts the
   explicit transpose BETWEEN the producer and consumer nests: the single-process analog of the
   distributed ``MPI_Alltoall`` transpose in ``tests/library/mpi/mpi_omen_transpose_test.py``.

Both are bit-exact against the k10 oracle. Note the API accepts the rank-4 batch permute that the
auto-search's ``permutation_layouts`` refuses (``MAX_PERMUTE_NDIM = 3``): imposing a layout is not bound
by the enumeration cap.
"""
import numpy

from dace.libraries.layout.algebra import Permute
from dace.transformation.layout.apply_assignment import IDENTITY_LAYOUT, Layout, apply_assignment
from dace.transformation.layout.line_graph import kernel_per_state, line_graph
from dace.transformation.layout.permute_dimensions import PermuteDimensions
from dace.transformation.layout.prepare import prepare_for_layout

from tests.transformations.layout.kernels import k10_omen_windowed_contraction as k10

BATCH_TRANSPOSE = Layout("perm1023", (Permute((1, 0, 2, 3)), ))  # swap G's (NA, NE) batch axes


def test_omen_layout_global_permute_via_api():
    """The OMEN layout choice as ONE direct API call: store G producer-order (NE, NA) for the whole
    program. No sweep -- the candidate is imposed."""
    na, ne = 3, 8
    inp = k10.make_inputs(na, ne)
    ref = k10.oracle(inp["H"], inp["X"], inp["D"])

    sdfg = k10.omen.to_sdfg(simplify=True)
    PermuteDimensions(permute_map={"G": [1, 0, 2, 3]}, add_permute_maps=True).apply_pass(sdfg, {})
    sdfg.validate()

    out = k10.run_closure(inp, na, ne)(sdfg)
    assert numpy.allclose(out["Sigma"], ref["Sigma"])


def test_omen_layout_midflight_transpose_via_api():
    """The OMEN transpose as an explicit MID-FLIGHT relayout, placed by hand: G is identity in the
    producer nest and batch-transposed for the consumer nest, so apply_assignment materializes one
    transpose between the phases -- the single-process form of the distributed Alltoall."""
    na, ne = 3, 8
    inp = k10.make_inputs(na, ne)
    ref = k10.oracle(inp["H"], inp["X"], inp["D"])

    sdfg = k10.omen.to_sdfg(simplify=True)
    prepare_for_layout(sdfg)
    kernel_per_state(sdfg)
    kernels = line_graph(sdfg)
    assert len(kernels) == 2  # RGF producer nest, SSE consumer nest

    # Impose the OMEN optimization directly: producer keeps identity, consumer reads the transpose.
    assignment = {"G": [IDENTITY_LAYOUT, BATCH_TRANSPOSE]}
    applied = apply_assignment(sdfg, kernels, assignment)
    assert len(applied.boundary_states) >= 1  # the transpose materialized between the two phases
    sdfg.validate()

    out = k10.run_closure(inp, na, ne)(sdfg)
    assert numpy.allclose(out["Sigma"], ref["Sigma"])


if __name__ == "__main__":
    test_omen_layout_global_permute_via_api()
    test_omen_layout_midflight_transpose_via_api()
    print("omen layout API tests PASS")
