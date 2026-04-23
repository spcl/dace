"""HLFIR ports of the non-legacy tests from ``origin/f2dace-windmill``.

The source tests under ``f2dace-windmill:tests/fortran/`` exercise the
legacy Python Fortran frontend (``create_sdfg_from_string`` /
``create_singular_sdfg_from_string``) and rely on ``SourceCodeBuilder``
plus ``check_with_gfortran`` helpers.  Here we rewrite each one to go
through the HLFIR frontend (flang → HLFIR → SDFG), compile the same
Fortran source with gfortran via ``numpy.f2py``, and compare outputs
numerically on random (or fixed) inputs.

Tests where the HLFIR frontend does not yet lower the feature are marked
``pytest.mark.xfail(strict=True, …)`` so implementing the corresponding
lowering flips them green — they are a live TODO list.

Feature coverage as of this commit:

  * DO loops with static bounds, straight-line assignments, simple
    arithmetic, array reads/writes — supported.
  * Indirect index access — supported.
  * Derived-type dummy arguments / locals (uniform + jagged) — supported
    via ``hlfir-flatten-structs``.
  * IF/ELSE, SELECT CASE, CYCLE/EXIT, DO WHILE — NOT YET.
  * Array sections (``res(i:j)``), reshapes, views — NOT YET.
  * OPTIONAL / POINTER / ALLOCATABLE dummies — NOT YET.
  * Intrinsics (MIN, MAX, SUM, …), user subroutine calls — NOT YET.
"""
from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

try:
    ctypes.CDLL("libgomp.so.1", ctypes.RTLD_GLOBAL)
except OSError:
    pass

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not available")


def _f2py(src_text: str, out_dir: Path, mod_name: str):
    if shutil.which("gfortran") is None:
        pytest.skip("gfortran not available")
    if shutil.which("meson") is None:
        pytest.skip("meson not available (f2py backend on Python>=3.12)")
    out_dir.mkdir(parents=True, exist_ok=True)
    src_file = out_dir / f"{mod_name}.f90"
    src_file.write_text(src_text)
    subprocess.check_call([sys.executable, "-m", "numpy.f2py", "-c",
                           str(src_file), "-m", mod_name, "--quiet"],
                          cwd=out_dir)
    if str(out_dir) not in sys.path:
        sys.path.insert(0, str(out_dir))
    __import__(mod_name)
    return sys.modules[mod_name]


def _build(src: str, tmp: Path, name: str):
    tmp.mkdir(parents=True, exist_ok=True)
    return build_sdfg(src, tmp, name=name, pipeline="hlfir-propagate-shapes").build()


# ---------------------------------------------------------------------------
# Currently supported features — should pass today.
# ---------------------------------------------------------------------------


def test_ported_elementwise_scalar_arithmetic(tmp_path):
    src = """
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
    mod = _f2py(src, tmp_path / "ref", "daxpy_lite")
    sdfg = _build(src, tmp_path / "sdfg", name="daxpy_lite")

    rng = np.random.default_rng(1)
    n = 16
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)

    z_ref = np.zeros(n, order="F")
    mod.daxpy_lite(np.asfortranarray(x), np.asfortranarray(y), z_ref)

    z_sdfg = np.zeros(n, dtype=np.float64)
    sdfg(x=np.ascontiguousarray(x), y=np.ascontiguousarray(y), z=z_sdfg, n=n)
    np.testing.assert_allclose(z_sdfg, z_ref, rtol=1e-12, atol=1e-12)


def test_ported_two_tasklets_raw(tmp_path):
    """Two statements in the same body, the second consuming the first —
    exercises the single-access-node-per-state invariant numerically."""
    src = """
subroutine raw_chain(a, out, n)
  implicit none
  integer, intent(in)    :: n
  real(8), intent(in)    :: a(n)
  real(8), intent(inout) :: out(n)
  real(8) :: tmp(n)
  integer :: i
  do i = 1, n
    tmp(i) = a(i) * 2.0d0
    out(i) = tmp(i) + 1.0d0
  end do
end subroutine raw_chain
"""
    mod = _f2py(src, tmp_path / "ref", "raw_chain")
    sdfg = _build(src, tmp_path / "sdfg", name="raw_chain")

    rng = np.random.default_rng(2)
    n = 8
    a = rng.standard_normal(n)
    out_ref = np.zeros(n, order="F")
    mod.raw_chain(np.asfortranarray(a), out_ref)
    out_sdfg = np.zeros(n, dtype=np.float64)
    sdfg(a=np.ascontiguousarray(a), out=out_sdfg, n=n)
    np.testing.assert_allclose(out_sdfg, out_ref, rtol=1e-12, atol=1e-12)


def test_ported_nested_array_indirect(tmp_path):
    """Port of ``nested_array_test``'s shape — an index array feeds into
    another array read, i.e. classic indirect access."""
    src = """
subroutine nested_idx(out, idx, src, n)
  implicit none
  integer, intent(in)    :: n
  integer, intent(in)    :: idx(n)
  real(8), intent(in)    :: src(n)
  real(8), intent(inout) :: out(n)
  integer :: i
  do i = 1, n
    out(i) = src(idx(i))
  end do
end subroutine nested_idx
"""
    mod = _f2py(src, tmp_path / "ref", "nested_idx")
    sdfg = _build(src, tmp_path / "sdfg", name="nested_idx")

    rng = np.random.default_rng(3)
    n = 7
    src_data = rng.standard_normal(n)
    idx = rng.integers(1, n + 1, size=n, dtype=np.int32)

    out_ref = np.zeros(n, order="F")
    mod.nested_idx(out_ref, np.asfortranarray(idx), np.asfortranarray(src_data))

    out_sdfg = np.zeros(n, dtype=np.float64)
    sdfg(idx=np.ascontiguousarray(idx), src=np.ascontiguousarray(src_data), out=out_sdfg, n=n)
    np.testing.assert_allclose(out_sdfg, out_ref, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# xfail: features the HLFIR frontend does not yet lower.  Each test runs
# both gfortran and SDFG and asserts numerical agreement — the HLFIR
# frontend silently drops the unsupported construct, so the SDFG result
# diverges and the assertion fails (as xfail expects).  When the feature
# lands, the test will xpass and ``strict=True`` will flip the suite red.
# ---------------------------------------------------------------------------


def _xfail(reason: str):
    return pytest.mark.xfail(strict=True, reason=reason)


def test_ported_empty_if_else(tmp_path):
    """Port of ``empty_test``."""
    src = """
subroutine pick(d, flag)
  implicit none
  logical, intent(in)    :: flag
  real(8), intent(inout) :: d(2)
  if (flag) then
    d(1) = 7.0d0
  else
    d(1) = -7.0d0
  end if
end subroutine pick
"""
    mod = _f2py(src, tmp_path / "ref", "pick")
    sdfg = _build(src, tmp_path / "sdfg", name="pick")

    d_ref = np.zeros(2, order="F")
    mod.pick(d_ref, True)
    d_sdfg = np.zeros(2, dtype=np.float64)
    # ``flag`` reads into a branch condition, so the classifier promotes
    # it to an SDFG symbol — pass a Python bool (DaCe casts it to int).
    sdfg(d=d_sdfg, flag=True)
    np.testing.assert_allclose(d_sdfg, d_ref)


def test_ported_cond_array(tmp_path):
    """Port of ``cond_array_test``."""
    src = """
subroutine cond_arr(d)
  implicit none
  real(4), intent(inout) :: d(5, 5)
  real(4) :: s
  s = d(2, 1) + 1.0
  if (s + 5.5 > 5.0) then
    d(2, 1) = 11.0
  else
    d(2, 1) = 12.0
  end if
end subroutine cond_arr
"""
    mod = _f2py(src, tmp_path / "ref", "cond_arr")
    sdfg = _build(src, tmp_path / "sdfg", name="cond_arr")

    d_ref = np.full((5, 5), 42.0, order="F", dtype=np.float32)
    mod.cond_arr(d_ref)
    d_sdfg = np.full((5, 5), 42.0, dtype=np.float32)
    sdfg(d=d_sdfg)
    np.testing.assert_allclose(d_sdfg, d_ref)


def test_ported_if_cycle(tmp_path):
    """Port of ``ifcycle_test``."""
    src = """
subroutine ifcyc(d)
  implicit none
  real(8), intent(inout) :: d(4)
  integer :: i
  do i = 1, 4
    if (i == 2) cycle
    d(i) = 5.5d0
  end do
end subroutine ifcyc
"""
    mod = _f2py(src, tmp_path / "ref", "ifcyc")
    sdfg = _build(src, tmp_path / "sdfg", name="ifcyc")

    d_ref = np.full(4, 42.0, order="F")
    mod.ifcyc(d_ref)
    d_sdfg = np.full(4, 42.0, dtype=np.float64)
    sdfg(d=d_sdfg)
    np.testing.assert_allclose(d_sdfg, d_ref)


def test_ported_do_while(tmp_path):
    """Port of ``while_test``."""
    src = """
subroutine while_count(res)
  implicit none
  real(4), intent(inout) :: res(2)
  integer :: i
  i = 0
  res(1) = 0.0
  do while (i < 10)
    res(1) = res(1) + 1.0
    i = i + 1
  end do
end subroutine while_count
"""
    mod = _f2py(src, tmp_path / "ref", "while_count")
    # Flang drops ``DO WHILE`` to raw cf.br + cf.cond_br; the ``lift-cf-to-scf``
    # pass is what turns it into the scf.while our bridge walks, so this
    # test needs a wider pipeline than the windmill ``_build`` helper's
    # minimal hlfir-propagate-shapes.
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name="while_count").build()

    r_ref = np.zeros(2, order="F", dtype=np.float32)
    mod.while_count(r_ref)
    r_sdfg = np.zeros(2, dtype=np.float32)
    sdfg(res=r_sdfg)
    np.testing.assert_allclose(r_sdfg, r_ref)


def test_ported_select_case(tmp_path):
    """Port of ``case_test`` (simple scalar case)."""
    src = """
subroutine case_pick(v, out)
  implicit none
  integer, intent(in)    :: v
  integer, intent(inout) :: out
  select case (v)
  case (1)
    out = 10
  case (2)
    out = 20
  case default
    out = -1
  end select
end subroutine case_pick
"""
    mod = _f2py(src, tmp_path / "ref", "case_pick")
    sdfg = _build(src, tmp_path / "sdfg", name="case_pick")

    o_ref = np.zeros(1, order="F", dtype=np.int32)
    mod.case_pick(2, o_ref)
    # Scalar intent(in) / intent(inout) args land as size-1 Arrays on the
    # SDFG signature, so the caller boxes the Python ints.
    v_sdfg = np.array([2], dtype=np.int32)
    o_sdfg = np.zeros(1, dtype=np.int32)
    sdfg(v=v_sdfg, out=o_sdfg)
    assert int(o_sdfg[0]) == int(o_ref[0])


@_xfail("HLFIR frontend: array-section assignment not lowered")
def test_ported_array_section_assign(tmp_path):
    """Port of ``struct_test``'s array-section assignment: ``res(a:b) = 42``."""
    src = """
subroutine fill_range(res, a, b)
  implicit none
  integer, intent(in)    :: a, b
  integer, intent(inout) :: res(6)
  res(a:b) = 42
end subroutine fill_range
"""
    mod = _f2py(src, tmp_path / "ref", "fill_range")
    sdfg = _build(src, tmp_path / "sdfg", name="fill_range")

    r_ref = np.zeros(6, order="F", dtype=np.int32)
    mod.fill_range(r_ref, 2, 5)
    r_sdfg = np.zeros(6, dtype=np.int32)
    sdfg(res=r_sdfg, a=2, b=5)
    np.testing.assert_array_equal(r_sdfg, r_ref)


def test_ported_min_intrinsic(tmp_path):
    """Port of ``tasklet_test``'s ``MIN`` intrinsic use."""
    src = """
subroutine min_res(d, res)
  implicit none
  real(4), intent(inout) :: d(2)
  real(4), intent(inout) :: res(2)
  real(4) :: temp
  temp = 88.0
  d(1) = d(1) * 2.0
  temp = min(d(1), temp)
  res(1) = temp + 10.0
end subroutine min_res
"""
    mod = _f2py(src, tmp_path / "ref", "min_res")
    sdfg = _build(src, tmp_path / "sdfg", name="min_res")

    d_ref = np.full(2, 42.0, order="F", dtype=np.float32)
    r_ref = np.full(2, 42.0, order="F", dtype=np.float32)
    mod.min_res(d_ref, r_ref)

    d_sdfg = np.full(2, 42.0, dtype=np.float32)
    r_sdfg = np.full(2, 42.0, dtype=np.float32)
    sdfg(d=d_sdfg, res=r_sdfg)
    np.testing.assert_allclose(r_sdfg, r_ref)


def test_ported_intersub_call(tmp_path):
    """Port of ``multisdfg_construction_test.test_minimal``: two
    subroutines in one file, outer calls inner."""
    src = """
subroutine inner(d)
  implicit none
  real(8), intent(inout) :: d(4)
  d(2) = 4.2d0
end subroutine inner

subroutine outer(d)
  implicit none
  real(8), intent(inout) :: d(4)
  d(2) = 5.5d0
  call inner(d)
end subroutine outer
"""
    mod = _f2py(src, tmp_path / "ref", "outer_mod")
    # f2py wraps every subroutine in the file — we pick `outer`.
    sdfg = _build(src, tmp_path / "sdfg", name="outer")

    d_ref = np.zeros(4, order="F")
    mod.outer(d_ref)
    d_sdfg = np.zeros(4, dtype=np.float64)
    sdfg(d=d_sdfg)
    np.testing.assert_allclose(d_sdfg, d_ref)


@_xfail("HLFIR frontend: OPTIONAL dummy arguments not lowered")
def test_ported_optional_arg(tmp_path):
    """Port of ``optional_args_test`` (scalar optional, present branch)."""
    src = """
subroutine opt_sum(res, a)
  implicit none
  integer, intent(inout) :: res(2)
  integer, optional      :: a
  if (present(a)) then
    res(1) = a
  else
    res(1) = 0
  end if
end subroutine opt_sum
"""
    mod = _f2py(src, tmp_path / "ref", "opt_sum")
    sdfg = _build(src, tmp_path / "sdfg", name="opt_sum")

    r_ref = np.zeros(2, order="F", dtype=np.int32)
    mod.opt_sum(r_ref, 5)
    r_sdfg = np.zeros(2, dtype=np.int32)
    sdfg(res=r_sdfg, a=5)
    np.testing.assert_array_equal(r_sdfg, r_ref)


@_xfail("HLFIR frontend: ELEMENTAL procedures not lowered")
def test_ported_elemental(tmp_path):
    """Port of ``elemental_test`` — elemental subroutine called on arrays."""
    src = """
module elemod
  implicit none
contains
  elemental subroutine delta(od, scat_od, g)
    real(8), intent(inout) :: od, scat_od, g
    real(8) :: f
    f = g * g
    od = od - scat_od * f
    scat_od = scat_od * (1.0d0 - f)
    g = g / (1.0d0 + g)
  end subroutine delta
end module elemod

subroutine apply_delta(od, scat_od, g)
  use elemod
  implicit none
  real(8), intent(inout) :: od(14), scat_od(14), g(14)
  call delta(od, scat_od, g)
end subroutine apply_delta
"""
    mod = _f2py(src, tmp_path / "ref", "apply_delta")
    sdfg = _build(src, tmp_path / "sdfg", name="apply_delta")

    rng = np.random.default_rng(7)
    od = rng.standard_normal(14)
    s = rng.standard_normal(14)
    g = rng.standard_normal(14)

    od_ref, s_ref, g_ref = (np.asfortranarray(od.copy()), np.asfortranarray(s.copy()), np.asfortranarray(g.copy()))
    mod.apply_delta(od_ref, s_ref, g_ref)

    od_sdfg, s_sdfg, g_sdfg = (np.ascontiguousarray(od.copy()), np.ascontiguousarray(s.copy()),
                               np.ascontiguousarray(g.copy()))
    sdfg(od=od_sdfg, scat_od=s_sdfg, g=g_sdfg)
    np.testing.assert_allclose(od_sdfg, od_ref, rtol=1e-12, atol=1e-12)
