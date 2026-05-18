"""Tests for ``merge_used_modules`` (the fparser-free single-TU pass).

Modelled on f2dace-windmill's ``recursive_ast_improver_test`` multi-
module fixtures (``lib`` <- ``lib_indirect`` <- entry).  The contract:

* compiling the project as **separate files together** and compiling
  the **single merged file** must be numerically equivalent (the merge
  preserves semantics), and
* the HLFIR bridge must build a correct SDFG from the merged TU
  (e2e vs the gfortran reference, per ``feedback_e2e_numerical``),

plus the structural guards: pass-through on a self-contained input,
idempotence, intrinsic ``USE`` left untouched, dependency order.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang
from dace.frontend.hlfir.preprocess import merge_used_modules

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

# Transitive project: physmod  <-  drivermod (use physmod)  <-  drv (use drivermod)
_PHYSMOD = """\
module physmod
  implicit none
contains
  pure real(8) function scale_one(x) result(r)
    real(8), intent(in) :: x
    r = 2.5d0 * x + 1.0d0
  end function scale_one
end module physmod
"""

_DRIVERMOD = """\
module drivermod
  use physmod, only: scale_one
  implicit none
contains
  subroutine apply(a, n)
    integer, intent(in) :: n
    real(8), intent(inout) :: a(n)
    integer :: i
    do i = 1, n
      a(i) = scale_one(a(i))
    end do
  end subroutine apply
end module drivermod
"""

_DRV = """\
subroutine drv(a, n)
  use drivermod, only: apply
  implicit none
  integer, intent(in) :: n
  real(8), intent(inout) :: a(n)
  call drv_inner(a, n)
contains
  subroutine drv_inner(b, m)
    integer, intent(in) :: m
    real(8), intent(inout) :: b(m)
    call apply(b, m)
  end subroutine drv_inner
end subroutine drv
"""


def _f2py(out_dir: Path, mod: str, *src_files: Path):
    """f2py-compile ``src_files`` together into module ``mod``."""
    if shutil.which("gfortran") is None:
        pytest.skip("gfortran not available")
    if shutil.which("meson") is None:
        pytest.skip("meson not available (f2py backend on Python>=3.12)")
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [
            sys.executable, "-m", "numpy.f2py", "-c", *[str(s) for s in src_files], "-m", mod, "--quiet",
            "--f90flags=-O0 -fno-fast-math -ffp-contract=off"
        ],
        cwd=out_dir,
    )
    if str(out_dir) not in sys.path:
        sys.path.insert(0, str(out_dir))
    __import__(mod)
    return sys.modules[mod]


def _proj(d: Path, **files: str) -> Path:
    """Write ``name_f90=<src>`` kwargs out as ``name.f90`` files."""
    d.mkdir(parents=True, exist_ok=True)
    for key, txt in files.items():
        assert key.endswith("_f90"), key
        (d / (key[:-4] + ".f90")).write_text(txt)
    return d


def test_merge_then_single_file_matches_all_files_together(tmp_path: Path):
    """Reference built from the three separate files compiled *together*
    must equal the reference built from the *single merged* file, and
    the SDFG the bridge builds from the merged TU must match both."""
    proj = _proj(tmp_path / "proj", physmod_f90=_PHYSMOD, drivermod_f90=_DRIVERMOD, drv_f90=_DRV)

    # Variant A: compile all files together (dependency order for f2py).
    ref_all = _f2py(tmp_path / "a", "mm_all", proj / "physmod.f90", proj / "drivermod.f90", proj / "drv.f90")

    # Variant B: merge -> one self-contained file, compile that alone.
    merged = merge_used_modules(_DRV, search_dirs=[proj])
    assert merged != _DRV, "multi-module entry must actually be merged"
    lo = merged.lower()
    assert lo.index("module physmod") < lo.index("module drivermod") < lo.index("subroutine drv"), \
        "merged TU must place dependencies before dependents"
    md = tmp_path / "b"
    md.mkdir(parents=True, exist_ok=True)
    (md / "merged.f90").write_text(merged)
    ref_merged = _f2py(tmp_path / "bw", "mm_merged", md / "merged.f90")

    # SDFG via the bridge from the merged single TU.
    sdfg = build_sdfg(merged, tmp_path / "sdfg", name="drv", entry="_QPdrv").build()

    rng = np.random.default_rng(0)
    base = np.asfortranarray(rng.standard_normal(7))

    aa = base.copy(order="F")
    ref_all.drv(aa, aa.size)
    bb = base.copy(order="F")
    ref_merged.drv(bb, bb.size)
    cc = base.copy(order="F")
    from dace.data import Scalar
    al = sdfg.arglist()
    nkw = {"n": 7 if isinstance(al.get("n"), Scalar) else np.array([7], np.int32)}
    sdfg(a=cc, **nkw)

    np.testing.assert_array_equal(aa, bb)  # merge == all-files-together (bit-exact)
    np.testing.assert_allclose(cc, aa, rtol=1e-13, atol=1e-13)  # bridge == reference


def test_merge_noop_on_self_contained_and_idempotent(tmp_path: Path):
    """A single self-contained file (its only ``USE`` resolves to a
    module defined in the same file) is returned unchanged, and the
    pass is idempotent."""
    src = _PHYSMOD + "\n" + _DRIVERMOD + "\n" + _DRV
    d = _proj(tmp_path / "self", only_f90=src)
    once = merge_used_modules(src, search_dirs=[d])
    assert once == src, "self-contained source must pass through unchanged"
    assert merge_used_modules(once, search_dirs=[d]) == once, "must be idempotent"
    assert merge_used_modules(src, search_dirs=[]) == src, "no search dirs -> no-op"


def test_merge_leaves_intrinsic_use_untouched(tmp_path: Path):
    """An ``intrinsic`` / compiler-provided module is never merged."""
    src = ("subroutine k(x)\n"
           "  use iso_c_binding, only: c_double\n"
           "  use iso_fortran_env, only: real64\n"
           "  implicit none\n"
           "  real(c_double), intent(inout) :: x\n"
           "  x = x + 1.0_real64\n"
           "end subroutine k\n")
    d = _proj(tmp_path / "intr", k_f90=src)
    assert merge_used_modules(src, search_dirs=[d]) == src
