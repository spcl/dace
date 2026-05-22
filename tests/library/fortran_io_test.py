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
from dace.libraries.fortran_io.nodes.namelist import NamelistRead
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


def test_write_read_roundtrip_f32(tmp_path):
    path = str(tmp_path / "vals_f32.txt")
    shapes = [[3]]
    a0 = np.array([1.5, -2.25, 3.0], dtype=np.float32)

    _write_sdfg(path, shapes, dace.float32, "_f32")(a0=a0.copy())

    r0 = np.zeros(3, dtype=np.float32)
    _read_sdfg(path, shapes, dace.float32, "_f32")(a0=r0)

    np.testing.assert_allclose(r0, a0, rtol=1e-6)


def test_write_read_roundtrip_i64(tmp_path):
    path = str(tmp_path / "vals_i64.txt")
    shapes = [[4]]
    a0 = np.array([100, -200, 300, 1 << 40], dtype=np.int64)

    _write_sdfg(path, shapes, dace.int64, "_i64")(a0=a0.copy())

    r0 = np.zeros(4, dtype=np.int64)
    _read_sdfg(path, shapes, dace.int64, "_i64")(a0=r0)

    np.testing.assert_array_equal(r0, a0)


def test_read_array_spans_multiple_lines(tmp_path):
    """A single array read consumes a list-directed sequence across records:
    one ``dace_fio_read_f64_arr`` for n values reads as many lines as needed."""
    path = tmp_path / "seq.txt"
    path.write_text("1.0 2.0 3.0\n4.0 5.0\n6.0\n")  # six values over three lines
    r0 = np.zeros(6, dtype=np.float64)
    _read_sdfg(str(path), [[6]], dace.float64, "_seq")(a0=r0)
    np.testing.assert_allclose(r0, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


def _namelist_sdfg(path, group, members, dtypes_, shapes, tag=""):
    sdfg = dace.SDFG(f"fortran_io_namelist{tag}")
    state = sdfg.add_state()
    node = NamelistRead("nml", filename=path, group=group, members=members)
    state.add_node(node)
    for i, (shape, dt) in enumerate(zip(shapes, dtypes_)):
        name = f"m{i}"
        sdfg.add_array(name, shape, dt)
        state.add_edge(node, f"_out_{i}", state.add_access(name), None, dace.Memlet.from_array(name, sdfg.arrays[name]))
    return sdfg


def test_namelist_read(tmp_path):
    """Read scalar + array members of a namelist group into output arrays."""
    nml = tmp_path / "phys.nml"
    nml.write_text("&phys\n  alpha = 1.5,\n  nsteps = 7\n  coef = 10.0, 20.0, 30.0\n/\n")
    sdfg = _namelist_sdfg(str(nml), "phys", ["alpha", "nsteps", "coef"], [dace.float64, dace.int32, dace.float64],
                          [[1], [1], [3]])

    alpha = np.zeros(1, dtype=np.float64)
    nsteps = np.zeros(1, dtype=np.int32)
    coef = np.zeros(3, dtype=np.float64)
    sdfg(m0=alpha, m1=nsteps, m2=coef)

    np.testing.assert_allclose(alpha, [1.5])
    np.testing.assert_array_equal(nsteps, [7])
    np.testing.assert_allclose(coef, [10.0, 20.0, 30.0])


def test_namelist_selects_named_group(tmp_path):
    """The reader skips other groups and reads members from the named one,
    across mixed member types (f32 scalar, i32 array, f64 scalar)."""
    nml = tmp_path / "multi.nml"
    nml.write_text("&other\n  scale = 99.0\n/\n&phys\n  scale = 2.5\n  flags = 1 2 3\n  ratio = 0.5\n/\n")
    sdfg = _namelist_sdfg(str(nml), "phys", ["scale", "flags", "ratio"], [dace.float32, dace.int32, dace.float64],
                          [[1], [3], [1]], "_multi")

    scale = np.zeros(1, dtype=np.float32)
    flags = np.zeros(3, dtype=np.int32)
    ratio = np.zeros(1, dtype=np.float64)
    sdfg(m0=scale, m1=flags, m2=ratio)

    np.testing.assert_allclose(scale, [2.5], rtol=1e-6)
    np.testing.assert_array_equal(flags, [1, 2, 3])
    np.testing.assert_allclose(ratio, [0.5])
