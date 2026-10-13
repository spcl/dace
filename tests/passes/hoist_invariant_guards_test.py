# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The guard step of specialization: canonicalization pushes invariant guards inward, ``hoist_invariant_guards``
takes them back up, and what it leaves behind is fused.

Two contracts: maps that end up under equal hoisted guards are fused, and the versioning guard derive-parallelism
emits (``if <side condition>: map else: sequential loop``) is a different kind of guard that neither the push-in
passes nor the hoist may take apart.
"""

import numpy as np

import dace
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation.interstate.move_map_invariant_if_up import hoist_invariant_guards
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.move_if_into_loop import MoveIfIntoLoop

N = dace.symbol("N")
C = dace.symbol("C")
INC = dace.symbol("INC")


def maps(sdfg: dace.SDFG) -> list[nodes.MapEntry]:
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)]


def conditionals(sdfg: dace.SDFG) -> list[ConditionalBlock]:
    return [b for b in sdfg.all_control_flow_blocks(recursive=True) if isinstance(b, ConditionalBlock)]


@dace.program
def two_maps_under_one_guard(x: dace.float64[N], y: dace.float64[N], z: dace.float64[N]):
    for i in dace.map[0:N]:
        if C > 0:
            y[i] = 2.0 * x[i]
    for i in dace.map[0:N]:
        if C > 0:
            z[i] = y[i] + 1.0


def test_maps_under_equal_hoisted_guards_are_fused():
    """Canonicalization keeps the map-invariant ``C > 0`` inside both map bodies. Hoisted, each map becomes
    ``if C > 0: map``; the step merges the two guards and fuses the two maps under the one that is left."""
    sdfg = two_maps_under_one_guard.to_sdfg(simplify=True)
    canonicalize(sdfg)
    # The canonical form keeps both guards innermost, one per map body.
    assert len(maps(sdfg)) == 2 and len(conditionals(sdfg)) == 2
    hoist_invariant_guards(sdfg, require_full_hoist=False)

    guards = [b for b in sdfg.nodes() if isinstance(b, ConditionalBlock)]
    assert len(guards) == 1 and len(conditionals(sdfg)) == 1, conditionals(sdfg)
    assert len(maps(sdfg)) == 1, "the two maps under the merged guard were not fused"
    views = [
        name for sd in sdfg.all_sdfgs_recursive() for name, d in sd.arrays.items() if isinstance(d, dace.data.View)
    ]
    assert not views, f"the hoist minted views (No-View nested SDFGs bind whole containers): {views}"
    for c_value in (1, 0):
        x = np.random.rand(16)
        y = np.full(16, 7.0)
        z = np.full(16, 9.0)
        sdfg(x=x, y=y, z=z, N=16, C=c_value)
        expected_y = 2.0 * x if c_value > 0 else np.full(16, 7.0)
        assert np.allclose(y, expected_y) and np.allclose(z, expected_y + 1.0 if c_value > 0 else np.full(16, 9.0))


@dace.program
def strided_update(a: dace.float64[N * 4], b: dace.float64[N]):
    for i in range(N):
        a[i * INC] = a[i * INC] + b[i]


def versioning_guard(sdfg: dace.SDFG) -> ConditionalBlock:
    """The guard ``ParallelizeUnderConstraint`` emits: one branch a Map, the other a sequential LoopRegion."""
    for block in conditionals(sdfg):
        branches = [region for _, region in block.branches]
        has_map = [any(isinstance(n, nodes.MapEntry) for n, _ in region.all_nodes_recursive()) for region in branches]
        has_loop = [any(isinstance(r, LoopRegion) for r in region.all_control_flow_regions()) for region in branches]
        if len(branches) == 2 and any(has_map) and any(has_loop) and has_map != has_loop:
            return block
    raise AssertionError("no map-or-loop versioning guard in the canonical form")


def test_the_versioning_guard_survives_push_in_and_hoist():
    """``a[i * INC]`` is injective only when ``INC != 0``, so derive-parallelism emits ``if INC != 0: map else: loop``.
    The guard selects between two schedules of one computation; the push-in pass and the hoist must leave it as is."""
    sdfg = strided_update.to_sdfg(simplify=True)
    canonicalize(sdfg)
    guard = versioning_guard(sdfg)
    condition = guard.branches[0][0].as_string

    MoveIfIntoLoop().apply_pass(sdfg, {})
    hoist_invariant_guards(sdfg, require_full_hoist=False)
    hoist_invariant_guards(sdfg, require_full_hoist=True)

    after = versioning_guard(sdfg)
    assert after.branches[0][0].as_string == condition
    for inc in (1, 0):
        a = np.random.rand(32)
        b = np.random.rand(8)
        expected = a.copy()
        for i in range(8):
            expected[i * inc] += b[i]
        sdfg(a=a, b=b, N=8, INC=inc)
        assert np.allclose(a, expected)


if __name__ == "__main__":
    test_maps_under_equal_hoisted_guards_are_fused()
    test_the_versioning_guard_survives_push_in_and_hoist()
