# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`MergeLibraryNode`, Fortran ``MERGE(tsource, fsource, mask)`` and NumPy ``where``: every operand
is read by the NumPy broadcasting rule against the result."""

import itertools
import os

import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes import MergeLibraryNode

N = dace.symbol("N", dace.int64, nonnegative=True)
CONNECTORS = {
    "t": MergeLibraryNode.TRUE_CONNECTOR_NAME,
    "f": MergeLibraryNode.FALSE_CONNECTOR_NAME,
    "mask": MergeLibraryNode.MASK_CONNECTOR_NAME,
}

#: One compiled library per built SDFG and process: a reused name would load the previous case's library,
#: and pytest-xdist workers share the build folder.
BUILD_IDS = itertools.count()


def build(shapes, strides=None, dtypes=None, memlets=None, offsets=None):
    """One MergeLibraryNode over arrays ``t``, ``f``, ``mask`` and ``out`` of the given shapes. ``memlets`` maps
    an array to a subset string; the others are read or written whole."""
    dtypes = {"t": dace.float64, "f": dace.float64, "mask": dace.int32, "out": dace.float64, **(dtypes or {})}
    sdfg = dace.SDFG(f"merge_{os.getpid()}_{next(BUILD_IDS)}")
    for name, shape in shapes.items():
        sdfg.add_array(name, shape, dtypes[name], strides=strides, offset=(offsets or {}).get(name))
    state = sdfg.add_state()
    node = MergeLibraryNode("merge")
    state.add_node(node)

    def memlet(name):
        subset = (memlets or {}).get(name)
        return dace.Memlet(f"{name}[{subset}]") if subset else dace.Memlet.from_array(name, sdfg.arrays[name])

    for name, connector in CONNECTORS.items():
        state.add_edge(state.add_read(name), None, node, connector, memlet(name))
    state.add_edge(node, MergeLibraryNode.OUTPUT_CONNECTOR_NAME, state.add_write("out"), None, memlet("out"))
    return sdfg


def operands(shapes, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "t": rng.standard_normal(shapes["t"]),
        "f": rng.standard_normal(shapes["f"]),
        "mask": (rng.random(shapes["mask"]) > 0.5).astype(np.int32),
        "out": np.zeros(shapes["out"]),
    }


BROADCAST_CASES = [
    ((1,), (1,), (1,), (1,)),
    ((16,), (16,), (16,), (16,)),
    ((6, 8), (6, 8), (6, 8), (6, 8)),
    ((1,), (1,), (10,), (10,)),
    ((1,), (12,), (12,), (12,)),
    ((14,), (1,), (14,), (14,)),
    ((5, 1), (5, 4), (5, 4), (5, 4)),
    ((4,), (5, 4), (5, 4), (5, 4)),
    ((5, 4), (5, 4), (1,), (5, 4)),
    ((1, 1), (1,), (1,), (1,)),
]


@pytest.mark.parametrize("t, f, mask, out", BROADCAST_CASES, ids=lambda shape: "x".join(map(str, shape)))
def test_each_operand_broadcasts_against_the_result(t, f, mask, out):
    shapes = {"t": t, "f": f, "mask": mask, "out": out}
    arrays = operands(shapes)
    build(shapes)(**arrays)
    expected = np.where(arrays["mask"].astype(bool), arrays["t"], arrays["f"])
    np.testing.assert_array_equal(arrays["out"], expected.reshape(out))


def test_fortran_layout_operands_are_addressed_by_their_strides():
    n, m = 6, 8
    shapes = dict.fromkeys(CONNECTORS, (n, m)) | {"out": (n, m)}
    sdfg = build(shapes, strides=(1, n))
    sdfg.expand_library_nodes()
    inner = next(nd.sdfg for nd, _ in sdfg.all_nodes_recursive() if isinstance(nd, dace.nodes.NestedSDFG))
    for connector in (*CONNECTORS.values(), MergeLibraryNode.OUTPUT_CONNECTOR_NAME):
        assert tuple(inner.arrays[connector].strides) == (1, n), connector
    arrays = {name: np.asfortranarray(value) for name, value in operands(shapes, seed=1).items()}
    sdfg(**arrays)
    np.testing.assert_array_equal(arrays["out"], np.where(arrays["mask"].astype(bool), arrays["t"], arrays["f"]))


def test_symbolic_extents_give_a_map_over_that_extent():
    sdfg = build(
        dict.fromkeys(("t", "f", "mask", "out"), (N,)), memlets=dict.fromkeys(("t", "f", "mask", "out"), "0:N")
    )
    sdfg.expand_library_nodes()
    inner = next(nd.sdfg for nd, _ in sdfg.all_nodes_recursive() if isinstance(nd, dace.nodes.NestedSDFG))
    entry = next(nd for state in inner.states() for nd in state.nodes() if isinstance(nd, dace.nodes.MapEntry))
    assert str(entry.map.range[0][1] + 1) == "N"
    shapes = dict.fromkeys(("t", "f", "mask", "out"), (6,))
    arrays = operands(shapes, seed=2)
    sdfg(**arrays, N=6)
    np.testing.assert_array_equal(arrays["out"], np.where(arrays["mask"].astype(bool), arrays["t"], arrays["f"]))


def test_sliced_operands_with_a_lower_bound_offset():
    """The slices select 3 of 6 elements; ``t`` has an offset of -1, which shifts what its memlet reads."""
    shapes = dict.fromkeys(("t", "f", "mask"), (6,)) | {"out": (3,)}
    memlets = {"t": "1:4", "f": "2:5", "mask": "0:3"}
    arrays = operands(shapes, seed=3)
    build(shapes, memlets=memlets, offsets={"t": [-1]})(**arrays)
    expected = np.where(arrays["mask"][0:3].astype(bool), arrays["t"][0:3], arrays["f"][2:5])
    np.testing.assert_array_equal(arrays["out"], expected)


def test_operand_types_convert_to_the_result_type():
    shapes = dict.fromkeys(("t", "f", "mask", "out"), (8,))
    arrays = operands(shapes, seed=4)
    arrays["t"] = np.arange(8, dtype=np.int32)
    build(shapes, dtypes={"t": dace.int32})(**arrays)
    np.testing.assert_array_equal(arrays["out"], np.where(arrays["mask"].astype(bool), arrays["t"], arrays["f"]))


def test_an_operand_that_cannot_broadcast_is_refused_before_expansion():
    sdfg = build({"t": (3,), "f": (4,), "mask": (4,), "out": (4,)})
    (state,) = sdfg.states()
    node = next(n for n in state.nodes() if isinstance(n, MergeLibraryNode))
    with pytest.raises(ValueError, match="_mrg_t"):
        node.validate(sdfg, state)


if __name__ == "__main__":
    for case in BROADCAST_CASES:
        test_each_operand_broadcasts_against_the_result(*case)
    test_fortran_layout_operands_are_addressed_by_their_strides()
    test_symbolic_extents_give_a_map_over_that_extent()
    test_sliced_operands_with_a_lower_bound_offset()
    test_operand_types_convert_to_the_result_type()
    test_an_operand_that_cannot_broadcast_is_refused_before_expansion()
