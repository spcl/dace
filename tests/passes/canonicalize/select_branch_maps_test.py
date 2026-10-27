# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``SelectBranchMaps``: an ``if/else`` on a symbol whose branches each map the same range into the same array
becomes one map selecting per element, which then fuses with its producer."""

import numpy as np

import dace
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock
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


if __name__ == "__main__":
    test_branch_maps_without_a_fusion_partner_stay_apart()
    test_the_two_branch_maps_become_one_selecting_map()
    test_a_branch_copying_beside_its_map_stays_a_branch()
    test_canonicalization_fuses_the_select_with_its_producers()
    test_a_condition_on_a_scalar_is_handed_to_the_select()
