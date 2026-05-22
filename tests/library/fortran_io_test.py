# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
"""Round-trip tests for the ``dace.libraries.fortran_io`` Read / Write nodes.

Each test writes arrays through a ``Write`` node (the shipped iso_c_binding
Fortran wrappers perform the transfer) and reads them back through a ``Read``
node, asserting the values survive the file round-trip.  Skips when no Fortran
compiler is available, since the wrappers must be compiled + linked.
"""
import shutil

import numpy as np
import pytest

import dace
from dace.libraries.fortran_io.nodes.read import Read
from dace.libraries.fortran_io.nodes.write import Write

pytestmark = pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not available")


def _write_sdfg(path: str, shapes, dtype, tag: str = "") -> dace.SDFG:
    sdfg = dace.SDFG(f"fortran_io_write{tag}")
    state = sdfg.add_state()
    node = Write("write", filename=path, num_items=len(shapes))
    state.add_node(node)
    for i, shape in enumerate(shapes):
        name = f"a{i}"
        sdfg.add_array(name, shape, dtype)
        state.add_edge(state.add_access(name), None, node, f"_in_{i}", dace.Memlet.from_array(name, sdfg.arrays[name]))
    return sdfg


def _read_sdfg(path: str, shapes, dtype, tag: str = "") -> dace.SDFG:
    sdfg = dace.SDFG(f"fortran_io_read{tag}")
    state = sdfg.add_state()
    node = Read("read", filename=path, num_items=len(shapes))
    state.add_node(node)
    for i, shape in enumerate(shapes):
        name = f"a{i}"
        sdfg.add_array(name, shape, dtype)
        state.add_edge(node, f"_out_{i}", state.add_access(name), None, dace.Memlet.from_array(name, sdfg.arrays[name]))
    return sdfg


def test_write_read_roundtrip_f64(tmp_path):
    path = str(tmp_path / "vals_f64.txt")
    shapes = [[4], [3]]
    a0 = np.array([1.5, -2.25, 3.0, 4.125], dtype=np.float64)
    a1 = np.array([10.0, 20.0, 30.0], dtype=np.float64)

    _write_sdfg(path, shapes, dace.float64, "_f64")(a0=a0.copy(), a1=a1.copy())

    r0 = np.zeros(4, dtype=np.float64)
    r1 = np.zeros(3, dtype=np.float64)
    _read_sdfg(path, shapes, dace.float64, "_f64")(a0=r0, a1=r1)

    np.testing.assert_allclose(r0, a0)
    np.testing.assert_allclose(r1, a1)


def test_write_read_roundtrip_i32(tmp_path):
    path = str(tmp_path / "vals_i32.txt")
    shapes = [[5]]
    a0 = np.array([3, 1, 4, 1, 5], dtype=np.int32)

    _write_sdfg(path, shapes, dace.int32, "_i32")(a0=a0.copy())

    r0 = np.zeros(5, dtype=np.int32)
    _read_sdfg(path, shapes, dace.int32, "_i32")(a0=r0)

    np.testing.assert_array_equal(r0, a0)
