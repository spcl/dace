# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes import CShift, ShiftDirection


def cshift_sdfg(shape, dim, shift, direction=ShiftDirection.FORTRAN):
    sdfg = dace.SDFG(f"cshift_{len(shape)}d_dim{dim}_shift{shift}_{direction.value}".replace("-", "m"))
    sdfg.add_array("v", list(shape), dace.float64)
    sdfg.add_array("out", list(shape), dace.float64)
    state = sdfg.add_state()
    node = CShift("cshift", dim=dim, shift=shift, direction=direction)
    state.add_edge(state.add_read("v"), None, node, "_x", dace.Memlet.from_array("v", sdfg.arrays["v"]))
    state.add_edge(node, "_out", state.add_write("out"), None, dace.Memlet.from_array("out", sdfg.arrays["out"]))
    return sdfg


@pytest.mark.parametrize("shift", [2, -1])
def test_cshift_rotates_left(shift):
    """``CSHIFT(a, s)`` is ``np.roll(a, -s)``; a negative shift wraps."""
    arr = np.arange(1.0, 6.0)
    out = np.zeros(5)
    cshift_sdfg((5,), 1, shift)(v=arr, out=out)
    np.testing.assert_array_equal(out, np.roll(arr, -shift))


@pytest.mark.parametrize("dim,shift", [(1, 1), (2, -1)])
def test_cshift_rotates_every_section_along_dim(dim, shift):
    arr = np.arange(12.0).reshape((3, 4)).copy()
    out = np.zeros((3, 4))
    cshift_sdfg((3, 4), dim, shift)(v=arr, out=out)
    np.testing.assert_array_equal(out, np.roll(arr, -shift, axis=dim - 1))


@pytest.mark.parametrize("dim,shift", [(1, 2), (2, -3)])
def test_numpy_direction_is_roll(dim, shift):
    arr = np.arange(12.0).reshape((3, 4)).copy()
    out = np.zeros((3, 4))
    cshift_sdfg((3, 4), dim, shift, ShiftDirection.NUMPY)(v=arr, out=out)
    np.testing.assert_array_equal(out, np.roll(arr, shift, axis=dim - 1))


def test_direction_survives_serialization():
    sdfg = cshift_sdfg((5,), 1, 2, ShiftDirection.NUMPY)
    loaded = dace.SDFG.from_json(sdfg.to_json())
    (node,) = [n for n, _ in loaded.all_nodes_recursive() if isinstance(n, CShift)]
    assert node.direction is ShiftDirection.NUMPY
    assert node.shift == 2


def test_dim_out_of_range_is_refused():
    with pytest.raises(dace.sdfg.validation.InvalidSDFGNodeError, match="out of range"):
        cshift_sdfg((5,), 2, 1).validate()


if __name__ == "__main__":
    test_cshift_rotates_left(2)
    test_cshift_rotates_left(-1)
    test_cshift_rotates_every_section_along_dim(1, 1)
    test_cshift_rotates_every_section_along_dim(2, -1)
    test_numpy_direction_is_roll(1, 2)
    test_numpy_direction_is_roll(2, -3)
    test_direction_survives_serialization()
    test_dim_out_of_range_is_refused()
