"""End-to-end F90-binding coverage for flat numeric kernels.

The numeric baseline suites (``baseline_arithmetic_test``,
``baseline_integer_kinds_test``, ``baseline_intrinsic_min_test``,
``baseline_loops_test``) exercise the SDFG through DaCe's flat Python
ABI only -- they never compile-and-run the dace-generated ``.f90``
binding.  Per ``feedback_e2e_valid_fortran`` every valid-Fortran
bridge test should drive the full pipeline; per ``feedback_e2e_numerical``
the reference must be a non-transformed gfortran build.

This module promotes a representative slice to a *dual-path* check:
for each kernel it asserts that BOTH

    1. the SDFG invoked directly (DaCe flat ABI), and
    2. the dace-generated F90 binding (gfortran-compiled + linked
       against the SDFG ``.so``, called via ctypes)

match a gfortran reference of the same source, bit-for-bit
(``rtol = atol = 1e-12`` for floats; exact equality for integers).

The matrix deliberately spans the ABI-relevant numeric surface that
the LOGICAL suites don't touch:
    * ``real(8)`` arithmetic chains with ``intent(inout)``
    * a scalar ``real(8) intent(out)`` reduction -- the length-1-array
      scalar-output convention through the binding
    * ``integer(kind=1/2/4/8)`` element-wise arithmetic (every signed
      integer width that crosses the C ABI)
    * the ``min`` / ``max`` intrinsic lowered through the binding
"""

import ctypes
import shutil
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, gfortran_compile_so, have_flang
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


def _build_binding_lib(tmp_path: Path, *, kernel_src: str, name: str, entry: str, iface: OriginalInterface,
                       driver_src: str):
    """Build the SDFG, emit its binding, gfortran-link binding + driver
    against the SDFG ``.so``.

    :returns: ``(ctypes lib, compiled SDFG)`` -- the SDFG is also used
        for the direct flat-ABI leg of the same test.
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
    gfortran_compile_so(drv_so, bindings_path, driver_path, mod_dir=build_dir, link_so=so_path)
    return ctypes.CDLL(str(drv_so)), sdfg


def _build_ref_lib(tmp_path: Path, *, kernel_src: str, ref_driver_src: str, name: str):
    """gfortran reference: plain kernel + a ``bind(c)`` driver sharing
    the SDFG driver's raw-pointer ABI, so ctypes can swap them."""
    ref_dir = tmp_path / "ref_build"
    ref_dir.mkdir(parents=True, exist_ok=True)
    k = ref_dir / f"{name}_k.f90"
    k.write_text(kernel_src)
    d = ref_dir / f"{name}_d.f90"
    d.write_text(ref_driver_src)
    so = ref_dir / f"{name}_ref.so"
    gfortran_compile_so(so, k, d, mod_dir=ref_dir)
    return ctypes.CDLL(str(so))


# ---------------------------------------------------------------------------
# real(8) arithmetic chain, intent(inout)
# ---------------------------------------------------------------------------

_DAXPY_KERNEL = """
subroutine daxpy_lite(x, y, z, n)
  implicit none
  integer, intent(in)    :: n
  real(8), intent(in)    :: x(n), y(n)
  real(8), intent(inout) :: z(n)
  integer :: i
  do i = 1, n
    z(i) = 2.0d0 * x(i) + y(i) - 0.5d0 * x(i)
  end do
end subroutine daxpy_lite
"""

_DAXPY_SDFG_DRIVER = """
subroutine run_daxpy(x, y, z, n) bind(c, name='run_daxpy')
  use iso_c_binding
  use daxpy_lite_dace_bindings
  implicit none
  integer(c_int), value :: n
  real(c_double), intent(in)    :: x(n), y(n)
  real(c_double), intent(inout) :: z(n)
  call daxpy_lite_dace(x, y, z, n)
  call daxpy_lite_dace_finalize()
end subroutine run_daxpy
"""

_DAXPY_REF_DRIVER = """
subroutine run_daxpy_ref(x, y, z, n) bind(c, name='run_daxpy_ref')
  use iso_c_binding
  implicit none
  integer(c_int), value :: n
  real(c_double), intent(in)    :: x(n), y(n)
  real(c_double), intent(inout) :: z(n)
  external :: daxpy_lite
  call daxpy_lite(x, y, z, n)
end subroutine run_daxpy_ref
"""


def test_e2e_real8_arith_inout(tmp_path: Path):
    """``real(8)`` arithmetic chain with an ``intent(inout)`` result.
    Binding AND direct SDFG vs gfortran (``rtol=1e-12``)."""
    iface = OriginalInterface(
        entry="daxpy_lite",
        args=(
            OriginalArg(name="x", fortran_type="real(8)", rank=1, shape=("n", ), intent="in"),
            OriginalArg(name="y", fortran_type="real(8)", rank=1, shape=("n", ), intent="in"),
            OriginalArg(name="z", fortran_type="real(8)", rank=1, shape=("n", ), intent="inout"),
            OriginalArg(name="n", fortran_type="integer(c_int)", rank=0, intent="in"),
        ),
    )
    lib, sdfg = _build_binding_lib(tmp_path,
                                   kernel_src=_DAXPY_KERNEL,
                                   name="daxpy_lite",
                                   entry="_QPdaxpy_lite",
                                   iface=iface,
                                   driver_src=_DAXPY_SDFG_DRIVER)
    ref = _build_ref_lib(tmp_path, kernel_src=_DAXPY_KERNEL, ref_driver_src=_DAXPY_REF_DRIVER, name="daxpy_lite")

    n = 16
    rng = np.random.default_rng(1)
    x = np.asfortranarray(rng.standard_normal(n))
    y = np.asfortranarray(rng.standard_normal(n))

    z_ref = np.asfortranarray(np.ones(n))
    fref = ref.run_daxpy_ref
    fref.restype = None
    dp = ctypes.POINTER(ctypes.c_double)
    fref.argtypes = [dp, dp, dp, ctypes.c_int]
    fref(x.ctypes.data_as(dp), y.ctypes.data_as(dp), z_ref.ctypes.data_as(dp), n)

    z_bind = np.asfortranarray(np.ones(n))
    fbind = lib.run_daxpy
    fbind.restype = None
    fbind.argtypes = [dp, dp, dp, ctypes.c_int]
    fbind(x.ctypes.data_as(dp), y.ctypes.data_as(dp), z_bind.ctypes.data_as(dp), n)
    np.testing.assert_allclose(z_bind, z_ref, rtol=1e-12, atol=1e-12)

    z_d = np.ones(n, dtype=np.float64)
    sdfg(x=np.ascontiguousarray(x), y=np.ascontiguousarray(y), z=z_d, n=n)
    np.testing.assert_allclose(z_d, z_ref, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# scalar real(8) intent(out): the length-1-array scalar-output convention
# ---------------------------------------------------------------------------

_SUM_KERNEL = """
subroutine sum_reduce(a, s, n)
  implicit none
  integer, intent(in)  :: n
  real(8), intent(in)  :: a(n)
  real(8), intent(out) :: s
  integer :: i
  s = 0.0d0
  do i = 1, n
    s = s + a(i) * a(i)
  end do
end subroutine sum_reduce
"""

_SUM_SDFG_DRIVER = """
subroutine run_sum(a, s, n) bind(c, name='run_sum')
  use iso_c_binding
  use sum_reduce_dace_bindings
  implicit none
  integer(c_int), value :: n
  real(c_double), intent(in)  :: a(n)
  real(c_double), intent(out) :: s
  call sum_reduce_dace(a, s, n)
  call sum_reduce_dace_finalize()
end subroutine run_sum
"""

_SUM_REF_DRIVER = """
subroutine run_sum_ref(a, s, n) bind(c, name='run_sum_ref')
  use iso_c_binding
  implicit none
  integer(c_int), value :: n
  real(c_double), intent(in)  :: a(n)
  real(c_double), intent(out) :: s
  external :: sum_reduce
  call sum_reduce(a, s, n)
end subroutine run_sum_ref
"""


def test_e2e_scalar_real_intent_out(tmp_path: Path):
    """Scalar ``real(8), intent(out) :: s`` -- the scalar-output
    convention (caller scalar, length-1 ``Array`` on the SDFG).  The
    binding must allocate a length-1 buffer and write the scalar back.
    Binding AND direct SDFG vs gfortran."""
    iface = OriginalInterface(
        entry="sum_reduce",
        args=(
            OriginalArg(name="a", fortran_type="real(8)", rank=1, shape=("n", ), intent="in"),
            OriginalArg(name="s", fortran_type="real(8)", rank=0, intent="out"),
            OriginalArg(name="n", fortran_type="integer(c_int)", rank=0, intent="in"),
        ),
    )
    lib, sdfg = _build_binding_lib(tmp_path,
                                   kernel_src=_SUM_KERNEL,
                                   name="sum_reduce",
                                   entry="_QPsum_reduce",
                                   iface=iface,
                                   driver_src=_SUM_SDFG_DRIVER)
    ref = _build_ref_lib(tmp_path, kernel_src=_SUM_KERNEL, ref_driver_src=_SUM_REF_DRIVER, name="sum_reduce")

    n = 20
    rng = np.random.default_rng(5)
    a = np.asfortranarray(rng.standard_normal(n))
    dp = ctypes.POINTER(ctypes.c_double)

    s_ref = ctypes.c_double(0.0)
    fref = ref.run_sum_ref
    fref.restype = None
    fref.argtypes = [dp, dp, ctypes.c_int]
    fref(a.ctypes.data_as(dp), ctypes.byref(s_ref), n)

    s_bind = ctypes.c_double(0.0)
    fbind = lib.run_sum
    fbind.restype = None
    fbind.argtypes = [dp, dp, ctypes.c_int]
    fbind(a.ctypes.data_as(dp), ctypes.byref(s_bind), n)
    np.testing.assert_allclose(s_bind.value, s_ref.value, rtol=1e-12, atol=1e-12)

    s_d = np.zeros(1, dtype=np.float64)
    sdfg(a=np.ascontiguousarray(a), s=s_d, n=n)
    np.testing.assert_allclose(s_d[0], s_ref.value, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# integer(kind=1/2/4/8) element-wise arithmetic
# ---------------------------------------------------------------------------


def _intk_kernel(kind: int) -> str:
    return f"""
subroutine madd_k{kind}(a, b, c, n)
  implicit none
  integer, intent(in) :: n
  integer(kind={kind}), intent(in)  :: a(n), b(n)
  integer(kind={kind}), intent(out) :: c(n)
  integer :: i
  do i = 1, n
    c(i) = a(i) * 2_{kind} + b(i)
  end do
end subroutine madd_k{kind}
"""


def _intk_sdfg_driver(kind: int, cty_name: str) -> str:
    return f"""
subroutine run_madd_k{kind}(a, b, c, n) bind(c, name='run_madd_k{kind}')
  use iso_c_binding
  use madd_k{kind}_dace_bindings
  implicit none
  integer(c_int), value :: n
  integer(kind={kind}), intent(in)  :: a(n), b(n)
  integer(kind={kind}), intent(out) :: c(n)
  call madd_k{kind}_dace(a, b, c, n)
  call madd_k{kind}_dace_finalize()
end subroutine run_madd_k{kind}
"""


def _intk_ref_driver(kind: int) -> str:
    return f"""
subroutine run_madd_k{kind}_ref(a, b, c, n) bind(c, name='run_madd_k{kind}_ref')
  use iso_c_binding
  implicit none
  integer(c_int), value :: n
  integer(kind={kind}), intent(in)  :: a(n), b(n)
  integer(kind={kind}), intent(out) :: c(n)
  external :: madd_k{kind}
  call madd_k{kind}(a, b, c, n)
end subroutine run_madd_k{kind}_ref
"""


_INT_MATRIX = [
    pytest.param(1, np.int8, ctypes.c_int8, id="int8"),
    pytest.param(2, np.int16, ctypes.c_int16, id="int16"),
    pytest.param(4, np.int32, ctypes.c_int32, id="int32"),
    pytest.param(8, np.int64, ctypes.c_int64, id="int64"),
]


@pytest.mark.parametrize("kind, npty, cty", _INT_MATRIX)
def test_e2e_integer_kind_arith(tmp_path: Path, kind: int, npty, cty):
    """``c = a * 2 + b`` over ``integer(kind=N)`` arrays for every
    signed-integer width that crosses the C ABI.  Binding AND direct
    SDFG vs gfortran, exact integer equality."""
    name = f"madd_k{kind}"
    kernel = _intk_kernel(kind)
    iface = OriginalInterface(
        entry=name,
        args=(
            OriginalArg(name="a", fortran_type=f"integer(kind={kind})", rank=1, shape=("n", ), intent="in"),
            OriginalArg(name="b", fortran_type=f"integer(kind={kind})", rank=1, shape=("n", ), intent="in"),
            OriginalArg(name="c", fortran_type=f"integer(kind={kind})", rank=1, shape=("n", ), intent="out"),
            OriginalArg(name="n", fortran_type="integer(c_int)", rank=0, intent="in"),
        ),
    )
    lib, sdfg = _build_binding_lib(tmp_path,
                                   kernel_src=kernel,
                                   name=name,
                                   entry=f"_QP{name}",
                                   iface=iface,
                                   driver_src=_intk_sdfg_driver(kind, cty.__name__))
    ref = _build_ref_lib(tmp_path, kernel_src=kernel, ref_driver_src=_intk_ref_driver(kind), name=name)

    n = 6
    rng = np.random.default_rng(31 + kind)
    lo, hi = (-5, 6) if kind == 1 else (-1000, 1000)
    a = np.asfortranarray(rng.integers(lo, hi, n).astype(npty))
    b = np.asfortranarray(rng.integers(lo, hi, n).astype(npty))
    expected = (a * npty(2) + b).astype(npty)
    cp = ctypes.POINTER(cty)

    c_ref = np.zeros(n, dtype=npty)
    fref = getattr(ref, f"run_{name}_ref")
    fref.restype = None
    fref.argtypes = [cp, cp, cp, ctypes.c_int]
    fref(a.ctypes.data_as(cp), b.ctypes.data_as(cp), c_ref.ctypes.data_as(cp), n)
    np.testing.assert_array_equal(c_ref, expected)

    c_bind = np.zeros(n, dtype=npty)
    fbind = getattr(lib, f"run_{name}")
    fbind.restype = None
    fbind.argtypes = [cp, cp, cp, ctypes.c_int]
    fbind(a.ctypes.data_as(cp), b.ctypes.data_as(cp), c_bind.ctypes.data_as(cp), n)
    np.testing.assert_array_equal(c_bind, expected)

    c_d = np.zeros(n, dtype=npty)
    sdfg(a=np.ascontiguousarray(a), b=np.ascontiguousarray(b), c=c_d, n=n)
    np.testing.assert_array_equal(c_d, expected)


# ---------------------------------------------------------------------------
# min / max intrinsic through the binding
# ---------------------------------------------------------------------------

_MINMAX_KERNEL = """
subroutine clamp_kernel(a, lo, hi, out, n)
  implicit none
  integer, intent(in)  :: n
  real(8), intent(in)  :: a(n)
  real(8), intent(in)  :: lo, hi
  real(8), intent(out) :: out(n)
  integer :: i
  do i = 1, n
    out(i) = max(lo, min(hi, a(i)))
  end do
end subroutine clamp_kernel
"""

_MINMAX_SDFG_DRIVER = """
subroutine run_clamp(a, lo, hi, out, n) bind(c, name='run_clamp')
  use iso_c_binding
  use clamp_kernel_dace_bindings
  implicit none
  integer(c_int), value :: n
  real(c_double), intent(in)  :: a(n)
  real(c_double), value :: lo, hi
  real(c_double), intent(out) :: out(n)
  call clamp_kernel_dace(a, lo, hi, out, n)
  call clamp_kernel_dace_finalize()
end subroutine run_clamp
"""

_MINMAX_REF_DRIVER = """
subroutine run_clamp_ref(a, lo, hi, out, n) bind(c, name='run_clamp_ref')
  use iso_c_binding
  implicit none
  integer(c_int), value :: n
  real(c_double), intent(in)  :: a(n)
  real(c_double), value :: lo, hi
  real(c_double), intent(out) :: out(n)
  external :: clamp_kernel
  call clamp_kernel(a, lo, hi, out, n)
end subroutine run_clamp_ref
"""


def test_e2e_minmax_intrinsic(tmp_path: Path):
    """``out = max(lo, min(hi, a))`` -- the ``min`` / ``max`` intrinsic
    plus scalar ``intent(in)`` (pass-by-value) args through the binding.
    Binding AND direct SDFG vs gfortran, bit-exact."""
    iface = OriginalInterface(
        entry="clamp_kernel",
        args=(
            OriginalArg(name="a", fortran_type="real(8)", rank=1, shape=("n", ), intent="in"),
            OriginalArg(name="lo", fortran_type="real(8)", rank=0, intent="in"),
            OriginalArg(name="hi", fortran_type="real(8)", rank=0, intent="in"),
            OriginalArg(name="out", fortran_type="real(8)", rank=1, shape=("n", ), intent="out"),
            OriginalArg(name="n", fortran_type="integer(c_int)", rank=0, intent="in"),
        ),
    )
    lib, sdfg = _build_binding_lib(tmp_path,
                                   kernel_src=_MINMAX_KERNEL,
                                   name="clamp_kernel",
                                   entry="_QPclamp_kernel",
                                   iface=iface,
                                   driver_src=_MINMAX_SDFG_DRIVER)
    ref = _build_ref_lib(tmp_path, kernel_src=_MINMAX_KERNEL, ref_driver_src=_MINMAX_REF_DRIVER, name="clamp_kernel")

    n = 12
    rng = np.random.default_rng(9)
    a = np.asfortranarray(rng.standard_normal(n) * 3.0)
    lo, hi = -1.25, 1.5
    dp = ctypes.POINTER(ctypes.c_double)

    out_ref = np.zeros(n, dtype=np.float64, order="F")
    fref = ref.run_clamp_ref
    fref.restype = None
    fref.argtypes = [dp, ctypes.c_double, ctypes.c_double, dp, ctypes.c_int]
    fref(a.ctypes.data_as(dp), ctypes.c_double(lo), ctypes.c_double(hi), out_ref.ctypes.data_as(dp), n)

    out_bind = np.zeros(n, dtype=np.float64, order="F")
    fbind = lib.run_clamp
    fbind.restype = None
    fbind.argtypes = [dp, ctypes.c_double, ctypes.c_double, dp, ctypes.c_int]
    fbind(a.ctypes.data_as(dp), ctypes.c_double(lo), ctypes.c_double(hi), out_bind.ctypes.data_as(dp), n)
    np.testing.assert_allclose(out_bind, out_ref, rtol=1e-12, atol=1e-12)

    out_d = np.zeros(n, dtype=np.float64)
    sdfg(a=np.ascontiguousarray(a), lo=lo, hi=hi, out=out_d, n=n)
    np.testing.assert_allclose(out_d, out_ref, rtol=1e-12, atol=1e-12)
