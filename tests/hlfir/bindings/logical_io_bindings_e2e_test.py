"""End-to-end LOGICAL ``intent(out)`` / ``intent(inout)`` binding tests.

``logical_bindings_e2e_test.py`` exercises only the *copy-in* direction
of the ``LOGICAL -> logical(c_bool)`` bridge: every kernel there takes a
``logical, intent(in) :: mask`` and writes an ``integer`` result, so the
generated wrapper only ever has to widen the caller's bool image into
the SDFG's 1-byte ``bool`` ABI on the way *in*.

The bridge also emits a *copy-out* leg
(``<arg> = <arg>_cbool`` followed by ``deallocate``) for every
``intent(out)`` / ``intent(inout)`` LOGICAL dummy, narrowing the SDFG's
1-byte ``bool`` result back into the caller's wider ``LOGICAL(KIND=N)``
image.  That leg was never run end-to-end.  These tests close the gap:

    * rank-1 LOGICAL ``intent(out)``  -- default kind + KIND 1/2/4/8
    * rank-1 LOGICAL ``intent(inout)`` -- both copy-in and copy-out legs
    * scalar LOGICAL ``intent(inout)`` -- the length-1 c_bool buffer
      round-trip

For every case BOTH paths are checked against a gfortran reference of
the same source:

    1. the SDFG invoked directly via the DaCe flat ABI
       (``np.bool_`` in / out), and
    2. the dace-generated F90 binding compiled + linked against the
       SDFG ``.so`` and called via ctypes.

The gfortran + ctypes harness (rather than f2py) is used throughout so
``LOGICAL(KIND=2)`` and ``logical(c_bool)`` -- which f2py's crackfortran
mis-maps -- are exercised on the same footing as the default kind.
"""

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang
from dace.frontend.hlfir.bindings import (
    FlattenPlan,
    OriginalArg,
    OriginalInterface,
    emit_bindings,
)

pytestmark = [
    pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH"),
    pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH"),
]

# flang-portable trio: keep the SDFG-linked binding and the gfortran
# reference on byte-identical arithmetic semantics.
_FFLAGS = ["-O0", "-fno-fast-math", "-ffp-contract=off"]


def _compile_so(out_so: Path, *sources: Path, mod_dir: Path, link_so: Path | None = None):
    """gfortran-compile ``sources`` into the shared object ``out_so``.

    :param out_so: output ``.so`` path.
    :param sources: Fortran sources, compiled in order.
    :param mod_dir: directory for ``.mod`` files and the build cwd.
    :param link_so: optional SDFG library to link against.
    """
    cmd = ["gfortran", "-shared", "-fPIC", *_FFLAGS, f"-J{mod_dir}"]
    cmd.extend(str(s) for s in sources)
    cmd.extend(["-o", str(out_so)])
    if link_so is not None:
        cmd.extend([f"-L{link_so.parent}", f"-Wl,-rpath,{link_so.parent}", f"-l:{link_so.name}"])
    subprocess.check_call(cmd, cwd=mod_dir)


def _build_binding_lib(tmp_path: Path, *, kernel_src: str, name: str, entry: str, iface: OriginalInterface,
                       driver_src: str):
    """Build the SDFG, emit its F90 binding, gfortran-link the binding +
    driver against the SDFG ``.so``.

    :returns: ``(ctypes lib, compiled SDFG)`` -- the SDFG is returned so
        the same build can also be exercised directly via the flat ABI.
    """
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(kernel_src, sdfg_dir, name=name, entry=entry).build()
    sdfg.name = name
    compiled = sdfg.compile()
    so_path = Path(compiled._lib._library_filename)

    bindings_path = tmp_path / f"{name}_bindings.f90"
    emit_bindings(sdfg._frozen_signature, iface, FlattenPlan(entries=()), str(bindings_path))
    driver_path = tmp_path / f"{name}_driver.f90"
    driver_path.write_text(driver_src)

    build_dir = tmp_path / "bind_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    drv_so = build_dir / f"{name}_drv.so"
    _compile_so(drv_so, bindings_path, driver_path, mod_dir=build_dir, link_so=so_path)
    return ctypes.CDLL(str(drv_so)), sdfg


def _build_ref_lib(tmp_path: Path, *, kernel_src: str, ref_driver_src: str, name: str):
    """gfortran-compile the plain kernel + a ``bind(c)`` reference driver
    (same raw-pointer ABI as the SDFG driver) into a ``.so``."""
    ref_dir = tmp_path / "ref_build"
    ref_dir.mkdir(parents=True, exist_ok=True)
    k = ref_dir / f"{name}_k.f90"
    k.write_text(kernel_src)
    d = ref_dir / f"{name}_d.f90"
    d.write_text(ref_driver_src)
    so = ref_dir / f"{name}_ref.so"
    _compile_so(so, k, d, mod_dir=ref_dir)
    return ctypes.CDLL(str(so))


# ---------------------------------------------------------------------------
# rank-1 LOGICAL intent(out): copy-out leg, every ABI-relevant kind
# ---------------------------------------------------------------------------


def _out_kernel(kind_spec: str, suffix: str) -> str:
    return f"""
SUBROUTINE inv_out{suffix}(a, b, n)
integer, intent(in) :: n
logical{kind_spec}, intent(in)  :: a(n)
logical{kind_spec}, intent(out) :: b(n)
integer :: i
DO i = 1, n
  b(i) = .NOT. a(i)
ENDDO
END SUBROUTINE inv_out{suffix}
"""


def _out_sdfg_driver(kind_spec: str, suffix: str) -> str:
    return f"""
subroutine run_inv_out{suffix}(a, b, n) bind(c, name='run_inv_out{suffix}')
  use iso_c_binding
  use inv_out{suffix}_dace_bindings
  implicit none
  integer(c_int), value :: n
  logical{kind_spec}, intent(in)  :: a(n)
  logical{kind_spec}, intent(out) :: b(n)
  call inv_out{suffix}_dace(a, b, n)
  call inv_out{suffix}_dace_finalize()
end subroutine run_inv_out{suffix}
"""


def _out_ref_driver(kind_spec: str, suffix: str) -> str:
    return f"""
subroutine run_inv_out{suffix}_ref(a, b, n) bind(c, name='run_inv_out{suffix}_ref')
  use iso_c_binding
  implicit none
  integer(c_int), value :: n
  logical{kind_spec}, intent(in)  :: a(n)
  logical{kind_spec}, intent(out) :: b(n)
  external :: inv_out{suffix}
  call inv_out{suffix}(a, b, n)
end subroutine run_inv_out{suffix}_ref
"""


# (kind_spec on the Fortran declaration, np int width of that LOGICAL
# image as seen across the raw-pointer ABI, ctypes element type).
_KIND_MATRIX = [
    pytest.param("", 4, ctypes.c_int32, id="default"),
    pytest.param("(kind=1)", 1, ctypes.c_int8, id="kind1"),
    pytest.param("(kind=2)", 2, ctypes.c_int16, id="kind2"),
    pytest.param("(kind=4)", 4, ctypes.c_int32, id="kind4"),
    pytest.param("(kind=8)", 8, ctypes.c_int64, id="kind8"),
]


@pytest.mark.parametrize("kind_spec, width, cty", _KIND_MATRIX)
def test_e2e_logical_intent_out(tmp_path: Path, kind_spec: str, width: int, cty):
    """``logical(kind=N), intent(out) :: b`` -- the copy-OUT leg of the
    c_bool bridge, narrowing the SDFG's 1-byte bool result back into the
    caller's KIND=N image.  Asserts BOTH the SDFG-direct call and the
    F90 binding match the gfortran reference exactly (LOGICAL: exact)."""
    suffix = "" if kind_spec == "" else f"_{width}"
    name = f"inv_out{suffix}"
    kernel = _out_kernel(kind_spec, suffix)
    iface = OriginalInterface(
        entry=name,
        args=(
            OriginalArg(name="a", fortran_type=f"logical{kind_spec}", rank=1, shape=("n", ), intent="in"),
            OriginalArg(name="b", fortran_type=f"logical{kind_spec}", rank=1, shape=("n", ), intent="out"),
            OriginalArg(name="n", fortran_type="integer(c_int)", rank=0, intent="in"),
        ),
    )
    lib, sdfg = _build_binding_lib(tmp_path,
                                   kernel_src=kernel,
                                   name=name,
                                   entry=f"_QP{name}",
                                   iface=iface,
                                   driver_src=_out_sdfg_driver(kind_spec, suffix))
    ref = _build_ref_lib(tmp_path, kernel_src=kernel, ref_driver_src=_out_ref_driver(kind_spec, suffix), name=name)

    n = 9
    rng = np.random.default_rng(101 + width)
    a_bits = rng.integers(0, 2, n).astype(np.bool_)

    # gfortran reference (KIND=N image: -1 / 0 for .true./.false., but
    # the kernel only ever assigns logical results so the *truthiness*
    # round-trip is what we compare, normalised to {0, 1}).
    a_w = a_bits.astype(f"int{width * 8}")
    b_ref = np.zeros(n, dtype=f"int{width * 8}")
    fref = getattr(ref, f"run_{name}_ref")
    fref.restype = None
    fref.argtypes = [ctypes.POINTER(cty), ctypes.POINTER(cty), ctypes.c_int]
    fref(a_w.ctypes.data_as(ctypes.POINTER(cty)), b_ref.ctypes.data_as(ctypes.POINTER(cty)), n)
    expected = (b_ref != 0).astype(np.int8)

    # (1) F90 binding path.
    a_w2 = a_bits.astype(f"int{width * 8}")
    b_bind = np.zeros(n, dtype=f"int{width * 8}")
    fbind = getattr(lib, f"run_{name}")
    fbind.restype = None
    fbind.argtypes = [ctypes.POINTER(cty), ctypes.POINTER(cty), ctypes.c_int]
    fbind(a_w2.ctypes.data_as(ctypes.POINTER(cty)), b_bind.ctypes.data_as(ctypes.POINTER(cty)), n)
    np.testing.assert_array_equal((b_bind != 0).astype(np.int8), expected)

    # (2) SDFG-direct path (DaCe flat ABI: np.bool_).
    a_d = a_bits.copy()
    b_d = np.zeros(n, dtype=np.bool_)
    sdfg(a=a_d, b=b_d, n=n)
    np.testing.assert_array_equal(b_d.astype(np.int8), expected)


# ---------------------------------------------------------------------------
# rank-1 LOGICAL intent(inout): both legs on the same dummy
# ---------------------------------------------------------------------------

_INOUT_KERNEL = """
SUBROUTINE toggle_io(mask, n)
integer, intent(in) :: n
logical, intent(inout) :: mask(n)
integer :: i
DO i = 1, n
  mask(i) = .NOT. mask(i)
ENDDO
END SUBROUTINE toggle_io
"""

_INOUT_SDFG_DRIVER = """
subroutine run_toggle_io(mask, n) bind(c, name='run_toggle_io')
  use iso_c_binding
  use toggle_io_dace_bindings
  implicit none
  integer(c_int), value :: n
  logical, intent(inout) :: mask(n)
  call toggle_io_dace(mask, n)
  call toggle_io_dace_finalize()
end subroutine run_toggle_io
"""

_INOUT_REF_DRIVER = """
subroutine run_toggle_io_ref(mask, n) bind(c, name='run_toggle_io_ref')
  use iso_c_binding
  implicit none
  integer(c_int), value :: n
  logical, intent(inout) :: mask(n)
  external :: toggle_io
  call toggle_io(mask, n)
end subroutine run_toggle_io_ref
"""


def test_e2e_logical_intent_inout(tmp_path: Path):
    """``logical, intent(inout) :: mask`` -- the caller's buffer is read
    (copy-in leg) AND written back (copy-out leg) through the same
    c_bool scratch.  Both the SDFG-direct and the F90-binding result
    must match the gfortran reference's in-place toggle exactly."""
    iface = OriginalInterface(
        entry="toggle_io",
        args=(
            OriginalArg(name="mask", fortran_type="logical", rank=1, shape=("n", ), intent="inout"),
            OriginalArg(name="n", fortran_type="integer(c_int)", rank=0, intent="in"),
        ),
    )
    lib, sdfg = _build_binding_lib(tmp_path,
                                   kernel_src=_INOUT_KERNEL,
                                   name="toggle_io",
                                   entry="_QPtoggle_io",
                                   iface=iface,
                                   driver_src=_INOUT_SDFG_DRIVER)
    ref = _build_ref_lib(tmp_path, kernel_src=_INOUT_KERNEL, ref_driver_src=_INOUT_REF_DRIVER, name="toggle_io")

    n = 7
    rng = np.random.default_rng(202)
    init = rng.integers(0, 2, n).astype(np.bool_)

    m_ref = init.astype(np.int32)
    fref = ref.run_toggle_io_ref
    fref.restype = None
    fref.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
    fref(m_ref.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), n)
    expected = (m_ref != 0).astype(np.int8)

    m_bind = init.astype(np.int32)
    fbind = lib.run_toggle_io
    fbind.restype = None
    fbind.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
    fbind(m_bind.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), n)
    np.testing.assert_array_equal((m_bind != 0).astype(np.int8), expected)

    m_direct = init.copy()
    sdfg(mask=m_direct, n=n)
    np.testing.assert_array_equal(m_direct.astype(np.int8), expected)
    # Symmetry: a second toggle restores the original pattern.
    sdfg(mask=m_direct, n=n)
    np.testing.assert_array_equal(m_direct, init)


# ---------------------------------------------------------------------------
# scalar LOGICAL intent(inout): the length-1 c_bool buffer round-trip
# ---------------------------------------------------------------------------

_SCALAR_INOUT_KERNEL = """
SUBROUTINE flip_flag(flag, hits, n)
integer, intent(in) :: n
logical, intent(inout) :: flag
integer, intent(out) :: hits(n)
integer :: i
DO i = 1, n
  IF (flag) THEN
    hits(i) = i
  ELSE
    hits(i) = -i
  ENDIF
ENDDO
flag = .NOT. flag
END SUBROUTINE flip_flag
"""

_SCALAR_INOUT_SDFG_DRIVER = """
subroutine run_flip_flag(flag, hits, n) bind(c, name='run_flip_flag')
  use iso_c_binding
  use flip_flag_dace_bindings
  implicit none
  integer(c_int), value :: n
  logical, intent(inout) :: flag
  integer(c_int), intent(out) :: hits(n)
  call flip_flag_dace(flag, hits, n)
  call flip_flag_dace_finalize()
end subroutine run_flip_flag
"""

_SCALAR_INOUT_REF_DRIVER = """
subroutine run_flip_flag_ref(flag, hits, n) bind(c, name='run_flip_flag_ref')
  use iso_c_binding
  implicit none
  integer(c_int), value :: n
  logical, intent(inout) :: flag
  integer(c_int), intent(out) :: hits(n)
  external :: flip_flag
  call flip_flag(flag, hits, n)
end subroutine run_flip_flag_ref
"""


def test_e2e_scalar_logical_intent_inout(tmp_path: Path):
    """Scalar ``logical, intent(inout) :: flag`` -- the kernel branches
    on ``flag`` then flips it.  Exercises the scalar-LOGICAL length-1
    c_bool buffer in BOTH directions (read for the branch, written for
    the flip).  Both paths vs the gfortran reference, for an initially
    true and an initially false flag."""
    iface = OriginalInterface(
        entry="flip_flag",
        args=(
            OriginalArg(name="flag", fortran_type="logical", rank=0, intent="inout"),
            OriginalArg(name="hits", fortran_type="integer(c_int)", rank=1, shape=("n", ), intent="out"),
            OriginalArg(name="n", fortran_type="integer(c_int)", rank=0, intent="in"),
        ),
    )
    lib, sdfg = _build_binding_lib(tmp_path,
                                   kernel_src=_SCALAR_INOUT_KERNEL,
                                   name="flip_flag",
                                   entry="_QPflip_flag",
                                   iface=iface,
                                   driver_src=_SCALAR_INOUT_SDFG_DRIVER)
    ref = _build_ref_lib(tmp_path,
                         kernel_src=_SCALAR_INOUT_KERNEL,
                         ref_driver_src=_SCALAR_INOUT_REF_DRIVER,
                         name="flip_flag")

    n = 5
    for start in (True, False):
        flag_ref = ctypes.c_int32(1 if start else 0)
        hits_ref = np.zeros(n, dtype=np.int32)
        fref = ref.run_flip_flag_ref
        fref.restype = None
        fref.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
        fref(ctypes.byref(flag_ref), hits_ref.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), n)
        flag_out_ref = flag_ref.value != 0

        flag_bind = ctypes.c_int32(1 if start else 0)
        hits_bind = np.zeros(n, dtype=np.int32)
        fbind = lib.run_flip_flag
        fbind.restype = None
        fbind.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
        fbind(ctypes.byref(flag_bind), hits_bind.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), n)
        np.testing.assert_array_equal(hits_bind, hits_ref)
        assert (flag_bind.value != 0) == flag_out_ref

        flag_d = np.array([start], dtype=np.bool_)
        hits_d = np.zeros(n, dtype=np.int32)
        sdfg(flag=flag_d, hits=hits_d, n=n)
        np.testing.assert_array_equal(hits_d, hits_ref)
        assert bool(flag_d[0]) == flag_out_ref
