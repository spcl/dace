# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``memlet_to_map`` must read and write the elements the copy's subsets name, steps and offsets included.

The maps are not compiled: the test evaluates the index expressions of the new tasklet at every point of the map and
applies the copy with NumPy, then compares with the slicing the memlet stands for.
"""

import itertools
import re

import numpy as np
import pytest

import dace
from dace import nodes, subsets, symbolic
from dace.sdfg import memlet_utils as mu

IT, MP, H = (dace.symbol(name, dace.int64) for name in ("it", "mp", "H"))
FH = symbolic.int_floor(H - MP, MP)


def evaluate(expr, values: dict[str, int]) -> int:
    return int(symbolic.evaluate(expr, values))


def region(subset: subsets.Range, values: dict[str, int]) -> tuple[slice, ...]:
    """The NumPy slices a subset stands for, with the symbols given their values."""
    return tuple(
        slice(evaluate(b, values), evaluate(e, values) + 1, evaluate(s, values)) for b, e, s in subset.ndrange()
    )


def copy_through_map(src_shape, dst_shape, src_subset: str, dst_subset: str, values: dict[str, int]):
    """Turn ``dst[dst_subset] = src[src_subset]`` into a map, run the map by hand, return (result, expected)."""
    sdfg = dace.SDFG("copy_through_map")
    sdfg.add_array("src", src_shape, dace.float64)
    sdfg.add_array("dst", dst_shape, dace.float64)
    state = sdfg.add_state()
    state.add_nedge(
        state.add_read("src"),
        state.add_write("dst"),
        dace.Memlet(data="dst", subset=dst_subset, other_subset=src_subset),
    )
    (edge,) = state.edges()
    assert mu.can_memlet_be_turned_into_a_map(edge, state, sdfg)
    mu.memlet_to_map(edge, state, sdfg)
    sdfg.validate()

    src = np.arange(int(np.prod([evaluate(extent, values) for extent in src_shape])), dtype=np.float64)
    src = src.reshape([evaluate(extent, values) for extent in src_shape])
    dst = np.full([evaluate(extent, values) for extent in dst_shape], -1.0)
    expected = dst.copy()
    window = src[region(subsets.Range.from_string(src_subset), values)]
    target = region(subsets.Range.from_string(dst_subset), values)
    expected[target] = window.reshape(expected[target].shape)

    entry = next(node for node in state.nodes() if isinstance(node, nodes.MapEntry))
    tasklet = next(node for node in state.nodes() if isinstance(node, nodes.Tasklet))
    (read,) = state.in_edges(tasklet)
    (write,) = state.out_edges(tasklet)
    bounds = [
        range(evaluate(b, values), evaluate(e, values) + 1, evaluate(s, values))
        for b, e, s in entry.map.range.ndrange()
    ]
    for point in itertools.product(*bounds):
        here = {**values, **dict(zip(entry.map.params, point, strict=True))}
        source = tuple(evaluate(index, here) for index, end, step in read.data.subset.ndrange())
        destination = tuple(evaluate(index, here) for index, end, step in write.data.subset.ndrange())
        dst[destination] = src[source]
    return dst, expected, (read.data.subset, write.data.subset)


def test_a_strided_source_is_read_at_its_step():
    dst, expected, subsets = copy_through_map(
        [2, 3, 15, 15], [2, 3, 4, 4], "0:2, 0:3, it:it+13:4, it:it+13:4", "0:2, 0:3, 0:4, 0:4", {"it": 2}
    )

    np.testing.assert_array_equal(dst, expected)


def test_a_strided_destination_is_written_at_its_step():
    dst, expected, subsets = copy_through_map(
        [2, 3, 4, 4], [2, 3, 8, 8], "0:2, 0:3, 0:4, 0:4", "0:2, 0:3, 1:8:2, 0:8:2", {}
    )

    np.testing.assert_array_equal(dst, expected)


def test_both_sides_strided_with_offsets():
    dst, expected, subsets = copy_through_map([20], [12], "3:19:4", "2:10:2", {})

    np.testing.assert_array_equal(dst, expected)


def test_a_symbolic_step_whose_size_is_only_equal_mathematically():
    """The conv2d max-pool window: ``padded[..., it : it + mp*FH + 1 : mp]`` into ``(FH + 1)`` elements."""
    values = {"H": 20, "mp": 4, "it": 1}
    dst, expected, subsets = copy_through_map(
        [2, 3, H, H],
        [2, 3, FH + 1, FH + 1],
        "0:2, 0:3, it:it+mp*int_floor(H-mp, mp)+1:mp, it:it+mp*int_floor(H-mp, mp)+1:mp",
        "0:2, 0:3, 0:int_floor(H-mp, mp)+1, 0:int_floor(H-mp, mp)+1",
        values,
    )

    np.testing.assert_array_equal(dst, expected)


def test_a_copy_between_different_ranks_keeps_offsets_and_steps():
    dst, expected, subsets = copy_through_map([6, 8], [4, 6], "0:6, 1:8:2", "0:4, 0:6", {})

    np.testing.assert_array_equal(dst, expected)


def test_a_flattening_copy_into_an_offset_range():
    dst, expected, subsets = copy_through_map([2, 3, 4], [40], "0:2, 0:3, 0:4", "5:29", {})

    np.testing.assert_array_equal(dst, expected)


def test_a_unit_stride_copy_keeps_the_plain_offset_form():
    dst, expected, (read, write) = copy_through_map([6, 6], [3, 3], "1:4, 2:5", "0:3, 0:3", {})

    np.testing.assert_array_equal(dst, expected)
    plain = re.compile(r"(__j[0-9])|(__j[0-9]+\s*\+\s*[0-9]+)|([0-9]+)")
    assert all(plain.fullmatch(str(index).strip()) for index, end, step in [*read.ndrange(), *write.ndrange()])


@pytest.mark.parametrize("size_one_axis", [0, 1])
def test_a_size_one_dimension_stays_fixed_while_the_others_step(size_one_axis):
    src_subset = ["2, 0:8:2", "0:8:2, 3"][size_one_axis]
    dst, expected, subsets = copy_through_map([8, 8], [4], src_subset, "0:4", {})

    np.testing.assert_array_equal(dst, expected)


if __name__ == "__main__":
    test_a_strided_source_is_read_at_its_step()
