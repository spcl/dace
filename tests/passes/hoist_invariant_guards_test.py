# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The guard step of specialization: canonicalization pushes invariant guards inward, ``hoist_invariant_guards``
takes them back up, and what it leaves behind is fused.

Two contracts: maps that end up under equal hoisted guards are fused, and the versioning guard derive-parallelism
emits (``if <side condition>: map else: sequential loop``) is a different kind of guard that neither the push-in
passes nor the hoist may take apart.
"""

import numpy as np

import dace
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
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


def guard_on_a_device_flag() -> dace.SDFG:
    """A GPU kernel whose body branches on ``flag[0]``, read in the condition through a connector."""
    gpu = dace.StorageType.GPU_Global
    sdfg = dace.SDFG("guard_on_device_flag")
    for name in ("a", "b"):
        sdfg.add_array(name, [N], dace.float64, storage=gpu)
    sdfg.add_array("flag", [1], dace.int32, storage=gpu)
    inner = dace.SDFG("guard_on_device_flag_body")
    inner.add_symbol("i", dace.int64)
    inner.add_array("a", [N], dace.float64, storage=gpu)
    inner.add_array("b", [N], dace.float64, storage=gpu)
    inner.add_array("flag", [1], dace.int32, storage=gpu)
    guard = ConditionalBlock("if_flag", sdfg=inner)
    inner.add_node(guard, is_start_block=True)
    branch = ControlFlowRegion("if_flag_body", sdfg=inner)
    guard.add_branch(CodeBlock("flag[0] > 0"), branch)
    body = branch.add_state("scale", is_start_block=True)
    scale = body.add_tasklet("scale", {"x"}, {"y"}, "y = 2.0 * x")
    body.add_edge(body.add_read("a"), None, scale, "x", dace.Memlet("a[i]"))
    body.add_edge(scale, "y", body.add_write("b"), None, dace.Memlet("b[i]"))
    state = sdfg.add_state("main", is_start_block=True)
    entry, exit_ = state.add_map("kernel", dict(i="0:N"), schedule=dace.ScheduleType.GPU_Device)
    nest = state.add_nested_sdfg(inner, {"a": None, "flag": None}, {"b": None}, symbol_mapping={"N": N, "i": "i"})
    state.add_memlet_path(state.add_read("a"), entry, nest, dst_conn="a", memlet=dace.Memlet("a[0:N]"))
    state.add_memlet_path(state.add_read("flag"), entry, nest, dst_conn="flag", memlet=dace.Memlet("flag[0]"))
    state.add_memlet_path(nest, exit_, state.add_write("b"), src_conn="b", memlet=dace.Memlet("b[0:N]"))
    sdfg.validate()
    return sdfg


def test_a_guard_reading_device_memory_stays_in_the_kernel():
    """A kernel's guard on ``flag[0]`` is map-invariant, but ``flag`` lives in device memory: hoisted out of the
    kernel, the guard became a host branch dereferencing a device pointer (cloudsc_subsidence_forcing's species
    guard on ``iphase``, a SIGSEGV on the GPU canon column). It stays inside the kernel."""
    sdfg = guard_on_a_device_flag()
    hoist_invariant_guards(sdfg, require_full_hoist=True)
    sdfg.validate()
    assert not [b for b in sdfg.nodes() if isinstance(b, ConditionalBlock)], "the guard left the kernel"


if __name__ == "__main__":
    test_maps_under_equal_hoisted_guards_are_fused()
    test_the_versioning_guard_survives_push_in_and_hoist()
    test_a_guard_reading_device_memory_stays_in_the_kernel()
