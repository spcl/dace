# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A contraction whose output is read only in part carries a hint saying so; one read whole carries none."""

import numpy as np

import dace
from dace.libraries.linalg.nodes.tensordot import TensorDot
from dace.transformation.passes.canonicalize.annotate_loop_kinds import AnnotateLoopKinds


@dace.program
def gathered_contraction(A: dace.float64[4, 5, 6], B: dace.float64[6, 3], idx: dace.int64[2], out: dace.float64[2]):
    C = np.tensordot(A, B, axes=([2], [0]))
    for i in dace.map[0:2]:
        out[i] = C[idx[i], 0, 0]


@dace.program
def whole_contraction(A: dace.float64[4, 5, 6], B: dace.float64[6, 3], out: dace.float64[4, 5, 3]):
    C = np.tensordot(A, B, axes=([2], [0]))
    out[:] = 2.0 * C


def contraction_hint(program) -> str | None:
    sdfg = program.to_sdfg(simplify=True)
    AnnotateLoopKinds().apply_pass(sdfg, {})
    found = [node for node, _ in sdfg.all_nodes_recursive() if isinstance(node, TensorDot)]
    assert len(found) == 1, "the frontend lowers np.tensordot to one TensorDot"
    return found[0].specialization_hint


def test_a_contraction_read_at_a_few_gathered_points_says_so():
    hint = contraction_hint(gathered_contraction)
    assert hint is not None and hint.startswith("full contraction -- downstream reads at most"), hint
    assert "of the 60 elements" in hint, hint


def test_a_contraction_read_whole_carries_no_hint():
    assert contraction_hint(whole_contraction) is None


if __name__ == "__main__":
    test_a_contraction_read_at_a_few_gathered_points_says_so()
    test_a_contraction_read_whole_carries_no_hint()
