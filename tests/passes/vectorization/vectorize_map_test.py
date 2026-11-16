# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``VectorizeMap`` tiles the one map it is given and leaves every other map of the program scalar."""

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import ISA
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import TILE_NODE_TYPES
from dace.transformation.passes.vectorization.vectorize_map import VectorizeMap

N = dace.symbol("N", dace.int64)
SIZE = 37  # not a multiple of the width: the remainder tile runs too


@dace.program
def axpy_then_square(x: dace.float64[N], y: dace.float64[N], z: dace.float64[N]):
    for i in dace.map[0:N]:
        y[i] = 2.0 * x[i] + y[i]
    for i in dace.map[0:N]:
        z[i] = x[i] * x[i]


def simplified_program() -> tuple[dace.SDFG, dace.SDFGState, nodes.MapEntry, nodes.MapEntry]:
    """The simplified program, with the map writing ``y`` and the one writing ``z``."""
    sdfg = axpy_then_square.to_sdfg(simplify=True)
    writer = {}
    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.MapEntry) and state.entry_node(node) is None:
            (written,) = {e.data.data for e in state.out_edges(state.exit_node(node))}
            writer[written] = (state, node)
    return sdfg, writer["y"][0], writer["y"][1], writer["z"][1]


def tile_nodes_in(sdfg: dace.SDFG) -> int:
    return sum(isinstance(n, TILE_NODE_TYPES) for n, _ in sdfg.all_nodes_recursive())


@pytest.mark.parametrize("isa", [ISA.SCALAR, ISA.AUTO])
@pytest.mark.parametrize("width", [4, 8])
def test_vectorize_map_tiles_the_named_map_and_leaves_the_other_scalar(width: int, isa: ISA):
    sdfg, state, axpy, square = simplified_program()
    square_state = next(s for n, s in sdfg.all_nodes_recursive() if n is square)
    square_tasklets = [n for n in square_state.scope_subgraph(square).nodes() if isinstance(n, nodes.Tasklet)]
    assert tile_nodes_in(sdfg) == 0

    nest = VectorizeMap(VectorizeConfig(widths=(width,), target_isa=isa), state, axpy).apply_pass(sdfg, {})

    sdfg.validate()
    assert tile_nodes_in(nest.sdfg) > 0, "the named map was not vectorized"
    assert tile_nodes_in(sdfg) == tile_nodes_in(nest.sdfg), "a tile op landed outside the named map"
    assert square in square_state.nodes() and square.map.range.ranges[0][2] == 1, "the other map was tiled"
    assert [n for n in square_state.scope_subgraph(square).nodes() if isinstance(n, nodes.Tasklet)] == square_tasklets
    rng = np.random.default_rng(0)
    x, y, z = rng.random(SIZE), rng.random(SIZE), np.zeros(SIZE)
    expected_y = 2.0 * x + y
    sdfg(x=x, y=y, z=z, N=SIZE)
    np.testing.assert_allclose(y, expected_y, rtol=1e-15)
    np.testing.assert_allclose(z, x * x, rtol=1e-15)


def test_vectorize_map_refuses_a_map_inside_another():
    sdfg = dace.SDFG("vectorize_map_inner")
    sdfg.add_array("a", [N, N], dace.float64)
    state = sdfg.add_state()
    outer_entry, outer_exit = state.add_map("outer", {"i": "0:N"})
    inner_entry, inner_exit = state.add_map("inner", {"j": "0:N"})
    tasklet = state.add_tasklet("one", {}, {"o"}, "o = 1.0")
    state.add_nedge(outer_entry, inner_entry, dace.Memlet())
    state.add_nedge(inner_entry, tasklet, dace.Memlet())
    state.add_memlet_path(
        tasklet, inner_exit, outer_exit, state.add_write("a"), src_conn="o", memlet=dace.Memlet("a[i, j]")
    )

    with pytest.raises(ValueError, match="inside another map"):
        VectorizeMap(VectorizeConfig(widths=(4,)), state, inner_entry).apply_pass(sdfg, {})


@dace.program
def absolute_then_square(x: dace.float64[N], y: dace.float64[N], z: dace.float64[N]):
    for i in dace.map[0:N]:
        if x[i] > 0:
            y[i] = x[i]
        else:
            y[i] = -x[i]
    for i in dace.map[0:N]:
        z[i] = x[i] * x[i]


def test_vectorize_map_tiles_a_map_whose_body_branches():
    # The body's nested SDFG holds a conditional, so the vectorizer fuses states inside the nest VectorizeMap made;
    # what encloses that nest is the program's, not the vectorizer's
    sdfg = absolute_then_square.to_sdfg(simplify=True)
    ((state, branchy),) = [
        (s, n)
        for n, s in sdfg.all_nodes_recursive()
        if isinstance(n, nodes.MapEntry)
        and s.entry_node(n) is None
        and {e.data.data for e in s.out_edges(s.exit_node(n))} == {"y"}
    ]

    nest = VectorizeMap(VectorizeConfig(widths=(4,), target_isa=ISA.SCALAR), state, branchy).apply_pass(sdfg, {})

    sdfg.validate()
    assert tile_nodes_in(nest.sdfg) > 0, "the branching map was not vectorized"
    assert tile_nodes_in(sdfg) == tile_nodes_in(nest.sdfg), "a tile op landed outside the named map"
    x = np.random.default_rng(0).random(SIZE) - 0.5
    y, z = np.zeros(SIZE), np.zeros(SIZE)
    sdfg(x=x, y=y, z=z, N=SIZE)
    np.testing.assert_allclose(y, np.abs(x), rtol=0)
    np.testing.assert_allclose(z, x * x, rtol=0)


if __name__ == "__main__":
    for width in (4, 8):
        for isa in (ISA.SCALAR, ISA.AUTO):
            test_vectorize_map_tiles_the_named_map_and_leaves_the_other_scalar(width, isa)
    test_vectorize_map_refuses_a_map_inside_another()
    test_vectorize_map_tiles_a_map_whose_body_branches()
