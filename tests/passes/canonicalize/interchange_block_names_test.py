# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Block names must stay unique through the canonicalization pipeline.

``LoopToMap`` names every state it creates for a nested loop body ``single_state_body``, and
``MoveLoopIntoMap`` (the ``interchange`` stage) hoists such a state out of its LoopRegion into
the parent region with a raw ``add_node`` -- which, unlike ``add_state``, does not run the
block-name uniquifier. The second loop in a region to go through both therefore produced two
blocks with the same name, and ``validate_sdfg`` rejects that SDFG.

Nothing downstream of ``interchange`` looked at block names, so the corruption only surfaced
once per-stage validation was cheap enough to leave on -- which is the point of the stage
scoping in ``CanonicalizationPipeline``.
"""

import collections

import numpy as np

import dace
from dace.transformation.passes.canonicalize.pipeline import _build_stages, canonicalize

N = dace.symbol('N')


@dace.program
def stencil_reduce_recurrence(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N]):
    """Three nests: a parallel stencil, a reduction, and a loop-carried recurrence.

    The recurrence is what survives ``parallelize`` as ``for i { map[j] }`` and reaches the
    ``interchange`` stage, and the earlier nests supply the sibling that makes the hoisted
    block names collide.
    """
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            B[i, j] = 0.25 * (A[i - 1, j] + A[i + 1, j] + A[i, j - 1] + A[i, j + 1])
    for i in range(N):
        s = 0.0
        for j in range(N):
            s += B[i, j] * B[i, j]
        C[i, 0] = s
    for i in range(1, N):
        for j in range(N):
            C[i, j] = C[i - 1, j] + B[i, j]


def _duplicate_block_names(sdfg: dace.SDFG):
    """``{cfg label: {block label: count}}`` for every region holding a repeated name."""
    duplicates = {}
    for cfg in sdfg.all_control_flow_regions(recursive=True):
        counts = collections.Counter(block.label for block in cfg.nodes())
        repeated = {label: count for label, count in counts.items() if count > 1}
        if repeated:
            duplicates[cfg.label] = repeated
    return duplicates


def test_no_stage_introduces_duplicate_block_names():
    """Walk the pipeline stage by stage and name the stage that breaks the invariant.

    Asserting on the final SDFG alone would say "invalid" without saying which of the ~160
    stages did it, so the loop reports the culprit."""
    sdfg = stencil_reduce_recurrence.to_sdfg(simplify=False)
    for label, unit in _build_stages():
        unit.apply_pass(sdfg, {})
        duplicates = _duplicate_block_names(sdfg)
        assert not duplicates, f'stage {label!r} introduced duplicate block names: {duplicates}'


def test_canonicalize_result_is_valid_and_correct():
    """End to end: the canonicalized SDFG validates and computes what the Python does."""
    sdfg = stencil_reduce_recurrence.to_sdfg(simplify=False)
    canonicalize(sdfg, validate=True)
    sdfg.validate()

    size = 12
    rng = np.random.default_rng(0)
    a = rng.random((size, size))

    # Reference written out rather than calling ``.f``: the program's ranges are in terms of
    #  the symbol ``N``, which plain Python cannot iterate.
    b_ref, c_ref = np.zeros((size, size)), np.zeros((size, size))
    for i in range(1, size - 1):
        for j in range(1, size - 1):
            b_ref[i, j] = 0.25 * (a[i - 1, j] + a[i + 1, j] + a[i, j - 1] + a[i, j + 1])
    for i in range(size):
        c_ref[i, 0] = sum(b_ref[i, j] * b_ref[i, j] for j in range(size))
    for i in range(1, size):
        for j in range(size):
            c_ref[i, j] = c_ref[i - 1, j] + b_ref[i, j]

    b_got, c_got = np.zeros((size, size)), np.zeros((size, size))
    sdfg(A=a.copy(), B=b_got, C=c_got, N=size)
    assert np.allclose(b_got, b_ref, rtol=1e-13, atol=1e-13), 'stencil output diverges'
    assert np.allclose(c_got, c_ref, rtol=1e-13, atol=1e-13), 'reduction/recurrence output diverges'


if __name__ == '__main__':
    test_no_stage_introduces_duplicate_block_names()
    test_canonicalize_result_is_valid_and_correct()
