# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``SelectBranchMaps``: an ``if/else`` on a symbol whose branches each map the same range into the same array
becomes one map selecting per element, which then fuses with its producer."""

import numpy as np

import dace
from dace.sdfg import nodes
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.canonicalize.select_branch_maps import SelectBranchMaps

N, M = dace.symbol("N"), dace.symbol("M")


@dace.program
def limiter(a: dace.float64[N], b: dace.float64[N], out: dace.float64[N]):
    bl = a - b
    br = a + b
    if M == 5:
        for i in dace.map[0:N]:
            out[i] = bl[i] * br[i]
    else:
        for i in dace.map[0:N]:
            out[i] = 3.0 * abs(bl[i]) - br[i]


@dace.program
def limiter_on_a_scalar(a: dace.float64[N], b: dace.float64[N], out: dace.float64[N], mode: dace.int64):
    bl = a - b
    br = a + b
    ord_ = mode
    if abs(ord_) == 5:
        for i in dace.map[0:N]:
            out[i] = bl[i] * br[i]
    else:
        for i in dace.map[0:N]:
            out[i] = 3.0 * abs(bl[i]) - br[i]


def expected(a, b, m):
    bl, br = a - b, a + b
    return bl * br if m == 5 else 3.0 * np.abs(bl) - br


@dace.program
def lone_branches(a: dace.float64[N], out: dace.float64[N]):
    if M == 5:
        for i in dace.map[0:N]:
            out[i] = a[i] + 1.0
    else:
        for i in dace.map[0:N]:
            out[i] = a[i] - 1.0


@dace.program
def divisor_guarded_by_the_branch(a: dace.int64[N], b: dace.int64[N], out: dace.int64[N]):
    bl = a - b
    br = a + b
    if M == 5:
        for i in dace.map[0:N]:
            out[i] = bl[i] // br[i]
    else:
        for i in dace.map[0:N]:
            out[i] = bl[i] - br[i]


@dace.program
def read_kept_in_range_by_the_branch(a: dace.float64[N], c: dace.float64[N], out: dace.float64[N]):
    bl = a + 1.0
    if M == 5:
        for i in dace.map[0:N]:
            out[i] = bl[i] * c[i + M]
    else:
        for i in dace.map[0:N]:
            out[i] = bl[i] - 1.0


def test_a_branch_the_guard_protects_stays_a_branch():
    """Merged, every element would compute the guarded arm: a division by zero, a read past the end of ``c``."""
    for program in (divisor_guarded_by_the_branch, read_kept_in_range_by_the_branch):
        sdfg = program.to_sdfg(simplify=True)
        assert SelectBranchMaps().apply_pass(sdfg, {}) is None


def test_branch_maps_without_a_fusion_partner_stay_apart():
    """Merged, each element would compute both arms with nothing to fuse with: the branch stays."""
    sdfg = lone_branches.to_sdfg(simplify=True)
    assert SelectBranchMaps().apply_pass(sdfg, {}) is None
    assert any(isinstance(b, ConditionalBlock) for b in sdfg.all_control_flow_blocks())


def test_the_two_branch_maps_become_one_selecting_map():
    sdfg = limiter.to_sdfg(simplify=True)
    assert any(isinstance(b, ConditionalBlock) for b in sdfg.all_control_flow_blocks())
    assert SelectBranchMaps().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    rng = np.random.default_rng(0)
    a, b = rng.random(16) - 0.5, rng.random(16) - 0.5
    for m in (5, 3):
        out = np.zeros(16)
        sdfg(a=a, b=b, out=out, N=16, M=m)
        assert np.allclose(out, expected(a, b, m))


def test_a_branch_copying_beside_its_map_stays_a_branch():
    """cloudsc_monolith's else arm copies ``zvqx[k]`` into a scalar beside its map: merged, the copy is lost."""
    sdfg = limiter.to_sdfg(simplify=True)
    sdfg.add_array("c", [1], dace.float64)
    block = next(b for b in sdfg.all_control_flow_blocks() if isinstance(b, ConditionalBlock))
    state = next(st for _, region in block.branches for st in region.nodes() if st.number_of_nodes() > 0)
    state.add_nedge(state.add_read("a"), state.add_write("c"), dace.Memlet("a[0] -> [0]"))
    sdfg.validate()

    assert SelectBranchMaps().apply_pass(sdfg, {}) is None
    assert any(isinstance(b, ConditionalBlock) for b in sdfg.all_control_flow_blocks())


def test_canonicalization_fuses_the_select_with_its_producers():
    sdfg = limiter.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    assert not any(isinstance(b, ConditionalBlock) for b in sdfg.all_control_flow_blocks())
    top = [n for st in sdfg.states() for n in st.nodes() if isinstance(n, nodes.MapEntry) and st.entry_node(n) is None]
    assert len(top) == 1, f"expected one fused map, got {[m.map.label for m in top]}"
    rng = np.random.default_rng(1)
    a, b = rng.random(16) - 0.5, rng.random(16) - 0.5
    for m in (5, 2):
        out = np.zeros(16)
        sdfg(a=a, b=b, out=out, N=16, M=m)
        assert np.allclose(out, expected(a, b, m))


def test_a_condition_on_a_scalar_is_handed_to_the_select():
    """fv3_dycore's limiter choice reads a scalar copied from an argument; the select reads it per element."""
    sdfg = limiter_on_a_scalar.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    assert not any(isinstance(b, ConditionalBlock) for b in sdfg.all_control_flow_blocks())
    rng = np.random.default_rng(2)
    a, b = rng.random(16) - 0.5, rng.random(16) - 0.5
    for m in (5, -5, 1):
        out = np.zeros(16)
        sdfg(a=a, b=b, out=out, mode=m, N=16)
        assert np.allclose(out, expected(a, b, abs(m)))


def guarded_arm_sdfg(name: str, cond: str, flag: bool) -> dace.SDFG:
    """``if <cond>: y = a * 3 else: y = a * 2`` then ``z = y + 1``; ``flag`` guards the else arm on ``d > 0``."""
    sdfg = dace.SDFG(name)
    for arr, size in (("a", 4), ("y", 4), ("z", 4), ("c", 1)):
        sdfg.add_array(arr, [size], dace.float64)
    sdfg.add_symbol("d", dace.int64)
    block = ConditionalBlock("branch", sdfg=sdfg, parent=sdfg)
    sdfg.add_node(block, is_start_block=True)
    after = sdfg.add_state_after(block, "after")
    after.add_mapped_tasklet("use", {"i": "0:4"}, {"q": dace.Memlet("y[i]")}, "o = q + 1", {"o": dace.Memlet("z[i]")},
                             external_edges=True)
    for code, tag, arm in (("o = q * 3", "t", cond), ("o = q * 2", "f", None)):
        region = ControlFlowRegion(tag, sdfg=sdfg, parent=block)
        block.add_branch(CodeBlock(arm) if arm else None, region)
        first = region.add_state(f"first_{tag}", is_start_block=True)
        state = first
        if flag and tag == "f":
            state = region.add_state("guarded")
            region.add_edge(first, state, dace.InterstateEdge("d > 0"))
        state.add_mapped_tasklet(tag, {"i": "0:4"}, {"q": dace.Memlet("a[i]")}, code, {"o": dace.Memlet("y[i]")},
                                 external_edges=True)
    sdfg.reset_cfg_list()
    return sdfg


def test_an_arm_behind_an_interstate_condition_stays_a_branch():
    sdfg = guarded_arm_sdfg("select_guarded_arm", "d > 5", True)
    assert not SelectBranchMaps().apply_pass(sdfg, {})


if __name__ == "__main__":
    test_a_branch_the_guard_protects_stays_a_branch()
    test_branch_maps_without_a_fusion_partner_stay_apart()
    test_the_two_branch_maps_become_one_selecting_map()
    test_a_branch_copying_beside_its_map_stays_a_branch()
    test_canonicalization_fuses_the_select_with_its_producers()
    test_a_condition_on_a_scalar_is_handed_to_the_select()
    test_an_arm_behind_an_interstate_condition_stays_a_branch()
