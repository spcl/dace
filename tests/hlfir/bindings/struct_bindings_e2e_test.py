"""End-to-end struct-dummy emit-bindings tests.

``tests/hlfir/bindings/emit_bindings_test.py`` covers three struct-flatten
fixtures via string-match assertions only:
    * ``_two_real_array_struct``  --  ``type(t_fields)`` with two plain
      ``real(c_double)`` members; everything aliases.
    * ``_complex_split_struct``   --  ``complex(c_double)`` member split
      into re/im scratch + copy loops, plus a plain real member that
      still aliases.
    * ``_nested_struct``          --  two-level nested struct (``st%a%v``
      / ``st%b%v``) where the ``%``-path is preserved through ``c_loc``.

Per ``feedback_e2e_valid_fortran``, every bindings test whose input is
a valid Fortran program must compile-and-run, not only string-match.
This module is the e2e companion: each test builds the SDFG, lets the
bridge stamp the real ``hlfir.flatten_plan`` attribute, lifts that
into a Python ``FlattenPlan``, runs the emitter, gfortran-compiles
the wrapper + a Fortran driver, links to the SDFG ``.so``, and
asserts numerical equality against a gfortran-compiled reference of
the same source.

Both the SDFG-via-bindings library and the gfortran reference library
are compiled by gfortran into ``.so`` files and loaded via ctypes.
f2py's crackfortran can't parse the bindings wrapper -- its dummy is
``type(t_fields)``, which maps to ``'void'`` in f2py's C-type table
and crashes lookup -- and the same struct-typed kernel dummy would
crash the reference build too.  Skipping f2py for both paths keeps
the test surface uniform and dodges that parser limitation entirely.
"""
from __future__ import annotations

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


def _compile_so(out_so: Path, *sources: Path, mod_dir: Path, link_so: Path | None = None):
    """gfortran-compile ``sources`` into ``out_so``, writing ``.mod``
    files to ``mod_dir`` and (optionally) linking against ``link_so``.

    We always set ``cwd=mod_dir`` and ``-J<mod_dir>`` so gfortran
    doesn't search the repository root for ``.mod`` files -- earlier
    f2py probes leak stale modules there and the format isn't
    cross-compiler compatible.
    """
    cmd = ["gfortran", "-shared", "-fPIC", f"-J{mod_dir}"]
    cmd.extend(str(s) for s in sources)
    cmd.extend(["-o", str(out_so)])
    if link_so is not None:
        cmd.extend([
            f"-L{link_so.parent}",
            f"-Wl,-rpath,{link_so.parent}",
            f"-l:{link_so.name}",
        ])
    subprocess.check_call(cmd, cwd=mod_dir)


def _build_sdfg_lib(
    tmp_path: Path,
    *,
    kernel_src: str,
    types_src: str,
    name: str,
    entry: str,
    iface: OriginalInterface,
    driver_src: str,
):
    """SDFG-via-bindings path: build SDFG, emit bindings, gfortran-link
    the types + bindings + driver into one ``.so`` against the SDFG
    library, return the loaded ctypes lib.

    ``FlattenPlan`` is read off the bridge module after the pass
    pipeline runs, so the emitter sees the same recipe
    ``hlfir-flatten-structs`` actually recorded.
    """
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    builder = build_sdfg(kernel_src, sdfg_dir, name=name, entry=entry)
    plan = FlattenPlan.from_dict(builder.module.get_flatten_plan())
    sdfg = builder.build()
    sdfg.name = name
    compiled = sdfg.compile()
    so_path = Path(compiled._lib._library_filename)
    fs = sdfg._frozen_signature

    bindings_path = tmp_path / f"{name}_bindings.f90"
    emit_bindings(fs, iface, plan, str(bindings_path))
    types_path = tmp_path / f"{name}_types.f90"
    types_path.write_text(types_src)
    driver_path = tmp_path / f"{name}_driver.f90"
    driver_path.write_text(driver_src)

    build_dir = tmp_path / "sdfg_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    driver_so = build_dir / f"{name}_driver.so"
    _compile_so(driver_so, types_path, bindings_path, driver_path, mod_dir=build_dir, link_so=so_path)
    return ctypes.CDLL(str(driver_so))


def _build_reference_lib(
    tmp_path: Path,
    *,
    types_src: str,
    kernel_src: str,
    ref_driver_src: str,
    name: str,
):
    """gfortran reference path: compile types + plain Fortran kernel +
    reference driver into one ``.so``.  The reference driver uses the
    same ``bind(c)`` raw-pointer entry-point convention as the SDFG
    driver so we can swap them with ctypes."""
    types_path = tmp_path / f"{name}_ref_types.f90"
    types_path.write_text(types_src)
    kernel_path = tmp_path / f"{name}_ref_kernel.f90"
    kernel_path.write_text(kernel_src)
    driver_path = tmp_path / f"{name}_ref_driver.f90"
    driver_path.write_text(ref_driver_src)

    build_dir = tmp_path / "ref_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    ref_so = build_dir / f"{name}_ref.so"
    _compile_so(ref_so, types_path, kernel_path, driver_path, mod_dir=build_dir)
    return ctypes.CDLL(str(ref_so))


# ---------------------------------------------------------------------------
# Two-real-array struct  --  zero-copy alias path
# ---------------------------------------------------------------------------

_TWO_REAL_TYPES_SRC = """
module mo_fields
  use iso_c_binding
  implicit none
  integer, parameter :: NX = 4, NY = 5
  type :: t_fields
     real(c_double) :: a(NX, NY)
     real(c_double) :: b(NX, NY)
  end type t_fields
end module mo_fields
"""

_TWO_REAL_KERNEL_SRC = """
subroutine kernel_two_real(fld)
  use mo_fields
  use iso_c_binding
  implicit none
  type(t_fields), intent(inout) :: fld
  integer :: i, j
  do j = 1, NY
     do i = 1, NX
        fld%a(i, j) = fld%a(i, j) + fld%b(i, j)
     end do
  end do
end subroutine kernel_two_real
"""

_TWO_REAL_REF_DRIVER_SRC = """
! Reference C-callable driver: same Fortran kernel ``kernel_two_real``,
! same raw-pointer ABI as the SDFG driver, so the test calls one or
! the other via ctypes and they're directly comparable.
subroutine run_two_real_ref(a_ptr, b_ptr) bind(c, name='run_two_real_ref')
  use iso_c_binding
  use mo_fields, only: t_fields, NX, NY
  implicit none
  real(c_double), intent(inout) :: a_ptr(NX, NY), b_ptr(NX, NY)
  type(t_fields) :: fld
  external :: kernel_two_real
  fld%a = a_ptr
  fld%b = b_ptr
  call kernel_two_real(fld)
  a_ptr = fld%a
  b_ptr = fld%b
end subroutine run_two_real_ref
"""

# Full source the bridge consumes: types + kernel.
_TWO_REAL_SRC = _TWO_REAL_TYPES_SRC + _TWO_REAL_KERNEL_SRC

_TWO_REAL_DRIVER = """
! C-callable driver that loads ``a``, ``b`` into a ``type(t_fields)``,
! calls the bindings wrapper, copies the post-call values back out.
! Linking is via ctypes (the bindings module's ``type(t_fields)`` arg
! defeats f2py's crackfortran), so the entry point is bind(c) with
! raw c_double pointers.
subroutine run_two_real(a_ptr, b_ptr) bind(c, name='run_two_real')
  use iso_c_binding
  use mo_fields, only: t_fields, NX, NY
  use kernel_two_real_dace_bindings
  implicit none
  real(c_double), intent(inout) :: a_ptr(NX, NY), b_ptr(NX, NY)
  type(t_fields), target :: fld
  fld%a = a_ptr
  fld%b = b_ptr
  call kernel_two_real_dace(fld)
  a_ptr = fld%a
  b_ptr = fld%b
  call kernel_two_real_dace_finalize()
end subroutine run_two_real
"""


def test_e2e_two_real_array_struct(tmp_path: Path):
    """``type(t_fields)`` with two static ``real(c_double)`` members.
    Both members alias zero-copy through ``c_loc``.  ``kernel`` does
    ``fld%a = fld%a + fld%b``; reference and SDFG paths must produce
    identical ``fld%a`` post-call."""
    iface = OriginalInterface(
        entry="kernel_two_real",
        args=(OriginalArg(name="fld", fortran_type="type(t_fields)", rank=0, intent="inout", struct_type="t_fields"), ),
        used_modules={"mo_fields": ("t_fields", )},
    )
    sdfg_lib = _build_sdfg_lib(
        tmp_path,
        kernel_src=_TWO_REAL_SRC,
        types_src=_TWO_REAL_TYPES_SRC,
        name="kernel_two_real",
        entry="_QPkernel_two_real",
        iface=iface,
        driver_src=_TWO_REAL_DRIVER,
    )
    sdfg_lib.run_two_real.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    sdfg_lib.run_two_real.restype = None

    ref_lib = _build_reference_lib(
        tmp_path,
        types_src=_TWO_REAL_TYPES_SRC,
        kernel_src=_TWO_REAL_KERNEL_SRC,
        ref_driver_src=_TWO_REAL_REF_DRIVER_SRC,
        name="kernel_two_real",
    )
    ref_lib.run_two_real_ref.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    ref_lib.run_two_real_ref.restype = None

    rng = np.random.default_rng(17)
    nx, ny = 4, 5
    a_init = np.asfortranarray(rng.standard_normal((nx, ny)))
    b_init = np.asfortranarray(rng.standard_normal((nx, ny)))

    a_ref = a_init.copy(order="F")
    b_ref = b_init.copy(order="F")
    ref_lib.run_two_real_ref(
        a_ref.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        b_ref.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )

    a_sdfg = a_init.copy(order="F")
    b_sdfg = b_init.copy(order="F")
    sdfg_lib.run_two_real(
        a_sdfg.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        b_sdfg.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )

    np.testing.assert_array_equal(a_sdfg, a_ref)
    np.testing.assert_array_equal(b_sdfg, b_ref)


# ---------------------------------------------------------------------------
# Nested struct ``st%a%v`` / ``st%b%v``  --  bridge gap (xfail)
# ---------------------------------------------------------------------------
#
# ``passes/FlattenStructs.cpp`` calls ``replaceStructArgNested`` for the
# nested case and intentionally does NOT call ``recordStructArgEntry`` --
# the comment in that block reads:
#
#     "Bindings-side FlattenEntry emission for outer_kind='dummy_nested'
#      is a separate follow-up -- Python-side callers can pass the flat
#      companions directly via kwargs today; the Fortran caller wrapper
#      needs the recipe to pack the nested struct's path-form members
#      on its end."
#
# Consequence: the bridge produces ``st_a_v``/``st_b_v`` flat dummies on
# the SDFG but ``get_flatten_plan()`` returns ``{'entries': []}``.  The
# emitter therefore emits no ``c_f_pointer`` aliases and the wrapper
# call site passes uninitialised ``st_a_v``/``st_b_v`` pointers.
#
# Pinning the e2e shape here so the test flips green when nested-plan
# emission lands in FlattenStructs.cpp.

_NESTED_TYPES_SRC = """
module mo_nested
  use iso_c_binding
  implicit none
  integer, parameter :: NX = 4, NY = 5
  type :: t_inner
     real(c_double) :: v(NX, NY)
  end type t_inner
  type :: t_outer
     type(t_inner) :: a
     type(t_inner) :: b
  end type t_outer
end module mo_nested
"""

_NESTED_KERNEL_SRC = """
subroutine kernel_nested(st)
  use mo_nested
  use iso_c_binding
  implicit none
  type(t_outer), intent(inout) :: st
  integer :: i, j
  do j = 1, NY
     do i = 1, NX
        st%a%v(i, j) = st%a%v(i, j) + st%b%v(i, j)
     end do
  end do
end subroutine kernel_nested
"""

_NESTED_REF_DRIVER_SRC = """
subroutine run_nested_ref(a_ptr, b_ptr) bind(c, name='run_nested_ref')
  use iso_c_binding
  use mo_nested, only: t_outer, NX, NY
  implicit none
  real(c_double), intent(inout) :: a_ptr(NX, NY), b_ptr(NX, NY)
  type(t_outer) :: st
  external :: kernel_nested
  st%a%v = a_ptr
  st%b%v = b_ptr
  call kernel_nested(st)
  a_ptr = st%a%v
  b_ptr = st%b%v
end subroutine run_nested_ref
"""

_NESTED_SRC = _NESTED_TYPES_SRC + _NESTED_KERNEL_SRC

_NESTED_DRIVER = """
subroutine run_nested(a_ptr, b_ptr) bind(c, name='run_nested')
  use iso_c_binding
  use mo_nested, only: t_outer, NX, NY
  use kernel_nested_dace_bindings
  implicit none
  real(c_double), intent(inout) :: a_ptr(NX, NY), b_ptr(NX, NY)
  type(t_outer), target :: st
  st%a%v = a_ptr
  st%b%v = b_ptr
  call kernel_nested_dace(st)
  a_ptr = st%a%v
  b_ptr = st%b%v
  call kernel_nested_dace_finalize()
end subroutine run_nested
"""


def test_e2e_nested_struct(tmp_path: Path):
    """``type(t_outer)`` containing two ``type(t_inner)`` members, each
    with a static ``real(c_double)`` array.  The kernel does
    ``st%a%v = st%a%v + st%b%v``.  ``recordNestedStructArgEntry`` in
    ``FlattenStructs.cpp`` emits one FlattenEntry whose recipe carries
    a flat name + a dotted read_expr per leaf, so the bindings emitter
    aliases each leaf via ``c_f_pointer(c_loc(st%a%v), st_a_v, [...])``."""
    iface = OriginalInterface(
        entry="kernel_nested",
        args=(OriginalArg(name="st", fortran_type="type(t_outer)", rank=0, intent="inout", struct_type="t_outer"), ),
        used_modules={"mo_nested": ("t_outer", )},
    )
    sdfg_lib = _build_sdfg_lib(
        tmp_path,
        kernel_src=_NESTED_SRC,
        types_src=_NESTED_TYPES_SRC,
        name="kernel_nested",
        entry="_QPkernel_nested",
        iface=iface,
        driver_src=_NESTED_DRIVER,
    )
    sdfg_lib.run_nested.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    sdfg_lib.run_nested.restype = None

    ref_lib = _build_reference_lib(
        tmp_path,
        types_src=_NESTED_TYPES_SRC,
        kernel_src=_NESTED_KERNEL_SRC,
        ref_driver_src=_NESTED_REF_DRIVER_SRC,
        name="kernel_nested",
    )
    ref_lib.run_nested_ref.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    ref_lib.run_nested_ref.restype = None

    rng = np.random.default_rng(23)
    nx, ny = 4, 5
    a_init = np.asfortranarray(rng.standard_normal((nx, ny)))
    b_init = np.asfortranarray(rng.standard_normal((nx, ny)))

    a_ref = a_init.copy(order="F")
    b_ref = b_init.copy(order="F")
    ref_lib.run_nested_ref(
        a_ref.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        b_ref.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )

    a_sdfg = a_init.copy(order="F")
    b_sdfg = b_init.copy(order="F")
    sdfg_lib.run_nested(
        a_sdfg.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        b_sdfg.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )

    np.testing.assert_array_equal(a_sdfg, a_ref)
    np.testing.assert_array_equal(b_sdfg, b_ref)


# ---------------------------------------------------------------------------
# Complex member struct  --  complex stays as a native complex128 SDFG dtype
# ---------------------------------------------------------------------------
#
# ``complex(c_double)`` struct members flatten to a single
# ``complex128`` companion array, NOT into a re/im pair.  Per the
# user's policy "complex types are supported in DaCe; complex arrays
# should be flattened using the complex dtype": DaCe handles complex
# arithmetic on a single ``std::complex<T>`` array natively.
#
# Bridge plumbing:
#   * ``FlattenStructs.cpp::isSimpleScalar`` accepts ``ComplexType``.
#   * ``dtypeName`` maps to ``complex64`` / ``complex128``.
#   * ``recordStructArgEntry`` emits one FlattenEntry per member so
#     mixed-dtype structs (complex + real) carry the right
#     ``scratch_dtype`` per entry.
#   * AST extractor ``ast/expressions.cpp`` recognises standalone
#     ``fir.extract_value`` from a complex value and emits
#     ``<z>.real()`` / ``<z>.imag()``; cppunparse renders these as
#     ``std::complex<T>::real()`` / ``::imag()`` method calls in C++.

_COMPLEX_TYPES_SRC = """
module mo_state
  use iso_c_binding
  implicit none
  integer, parameter :: NX = 4, NY = 5
  type :: t_state
     complex(c_double) :: z(NX, NY)
     real(c_double)    :: u(NX, NY)
  end type t_state
end module mo_state
"""

_COMPLEX_KERNEL_SRC = """
subroutine kernel_complex(st)
  use mo_state
  use iso_c_binding
  implicit none
  type(t_state), intent(inout) :: st
  integer :: i, j
  do j = 1, NY
     do i = 1, NX
        st%u(i, j) = real(st%z(i, j), kind=c_double) + aimag(st%z(i, j))
     end do
  end do
end subroutine kernel_complex
"""

_COMPLEX_REF_DRIVER_SRC = """
subroutine run_complex_ref(z_re_ptr, z_im_ptr, u_ptr) bind(c, name='run_complex_ref')
  use iso_c_binding
  use mo_state, only: t_state, NX, NY
  implicit none
  real(c_double), intent(in)    :: z_re_ptr(NX, NY)
  real(c_double), intent(in)    :: z_im_ptr(NX, NY)
  real(c_double), intent(inout) :: u_ptr(NX, NY)
  type(t_state) :: st
  external :: kernel_complex
  st%z = cmplx(z_re_ptr, z_im_ptr, kind=c_double)
  st%u = u_ptr
  call kernel_complex(st)
  u_ptr = st%u
end subroutine run_complex_ref
"""

_COMPLEX_SRC = _COMPLEX_TYPES_SRC + _COMPLEX_KERNEL_SRC

_COMPLEX_DRIVER = """
subroutine run_complex(z_re_ptr, z_im_ptr, u_ptr) bind(c, name='run_complex')
  use iso_c_binding
  use mo_state, only: t_state, NX, NY
  use kernel_complex_dace_bindings
  implicit none
  real(c_double), intent(in)    :: z_re_ptr(NX, NY)
  real(c_double), intent(in)    :: z_im_ptr(NX, NY)
  real(c_double), intent(inout) :: u_ptr(NX, NY)
  type(t_state), target :: st
  st%z = cmplx(z_re_ptr, z_im_ptr, kind=c_double)
  st%u = u_ptr
  call kernel_complex_dace(st)
  u_ptr = st%u
  call kernel_complex_dace_finalize()
end subroutine run_complex
"""


def test_e2e_complex_member_struct(tmp_path: Path):
    """``type(t_state)`` with ``complex(c_double)`` and ``real(c_double)``
    array members.  Kernel does ``st%u = real(st%z) + aimag(st%z)``.
    The complex member flattens to a single ``complex128`` companion
    (NOT split into re/im); the bindings emitter aliases it via
    ``c_f_pointer(c_loc(st%z), st_z, [...])`` and DaCe's tasklet
    codegen handles the ``.real()`` / ``.imag()`` method calls on
    ``std::complex<double>``."""
    iface = OriginalInterface(
        entry="kernel_complex",
        args=(OriginalArg(name="st", fortran_type="type(t_state)", rank=0, intent="inout", struct_type="t_state"), ),
        used_modules={"mo_state": ("t_state", )},
    )
    sdfg_lib = _build_sdfg_lib(
        tmp_path,
        kernel_src=_COMPLEX_SRC,
        types_src=_COMPLEX_TYPES_SRC,
        name="kernel_complex",
        entry="_QPkernel_complex",
        iface=iface,
        driver_src=_COMPLEX_DRIVER,
    )
    sdfg_lib.run_complex.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    sdfg_lib.run_complex.restype = None

    ref_lib = _build_reference_lib(
        tmp_path,
        types_src=_COMPLEX_TYPES_SRC,
        kernel_src=_COMPLEX_KERNEL_SRC,
        ref_driver_src=_COMPLEX_REF_DRIVER_SRC,
        name="kernel_complex",
    )
    ref_lib.run_complex_ref.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    ref_lib.run_complex_ref.restype = None

    rng = np.random.default_rng(29)
    nx, ny = 4, 5
    z_re_init = np.asfortranarray(rng.standard_normal((nx, ny)))
    z_im_init = np.asfortranarray(rng.standard_normal((nx, ny)))
    u_init = np.asfortranarray(rng.standard_normal((nx, ny)))

    u_ref = u_init.copy(order="F")
    ref_lib.run_complex_ref(
        z_re_init.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        z_im_init.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        u_ref.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )

    u_sdfg = u_init.copy(order="F")
    sdfg_lib.run_complex(
        z_re_init.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        z_im_init.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        u_sdfg.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )

    np.testing.assert_allclose(u_sdfg, u_ref, rtol=1e-12, atol=1e-12)
