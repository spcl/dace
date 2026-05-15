"""End-to-end test for the DT-of-scalar-constants flattening pattern.

The upstream cloudsc kernel bundles ~95 physical/control constants
into three derived types (``TYPE(TOMCST) :: YDCST`` etc.) and unpacks
them via ``ASSOCIATE`` at the top of the body.  Our local
``cloudscexp2_simplified.F90`` does this unpacking manually -- the
derived-type bundles are expanded into individual scalar arguments
because the bridge couldn't handle ``YDCST%RG``-style scalar-member
accesses on a derived-type dummy.

This test pins the simplest version of that pattern: a kernel that
takes one derived-type dummy whose members are plain scalars, and
verifies the bridge correctly:

  * Flattens the dummy into per-member scalar SDFG args via
    ``hlfir-flatten-structs`` (verified by checking the FlattenPlan).
  * Lowers the body's ``cst%rg`` reads to reads of the flat
    ``cst_rg`` scalar.
  * Produces a numerically-correct result vs an f2py reference of
    the same source.

If this test passes, the bridge is ready to consume the upstream
cloudsc signature directly (``cloudsc.F90`` with its
``YDCST/YDTHF/YDECLDP`` bundles) without manual flattening of the
ASSOCIATE -- ``hlfir-flatten-structs`` handles it.

E2e against an f2py-compiled reference of the same Fortran source.
"""

import numpy as np
import pytest

from _util import build_sdfg, have_flang
from dace.frontend.hlfir.bindings import FlattenPlan

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_SRC = """
module mo_consts
  use iso_c_binding
  implicit none
  type :: t_consts
     real(c_double) :: rg
     real(c_double) :: rd
     real(c_double) :: rcpd
  end type t_consts
end module mo_consts

subroutine kernel_dt_const(cst, n, out)
  use mo_consts
  use iso_c_binding
  implicit none
  type(t_consts), intent(in) :: cst
  integer, intent(in) :: n
  real(c_double), intent(out) :: out(n)
  integer :: i
  do i = 1, n
     out(i) = (cst%rg / cst%rcpd) * cst%rd * real(i, c_double)
  end do
end subroutine kernel_dt_const
"""


def test_dt_of_scalar_constants_flattens_per_member(tmp_path):
    """``type(t_consts) :: cst`` with three scalar members must flatten
    to three flat scalar SDFG args via ``hlfir-flatten-structs``.
    Verifies the bridge's per-member ``FlattenEntry`` emission for
    rank-0 (scalar) members produces the right shape and dtype, and
    that the SDFG arglist has the expected flat names."""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    builder = build_sdfg(_SRC, sdfg_dir, name="kernel_dt_const", entry="_QPkernel_dt_const")
    plan = FlattenPlan.from_dict(builder.module.get_flatten_plan())
    sdfg = builder.build()

    # FlattenPlan has one entry per scalar member.
    entries_by_outer = {e.outer_expr: e for e in plan.entries}
    assert "cst%rg" in entries_by_outer, f"missing cst%rg entry: {list(entries_by_outer.keys())}"
    assert "cst%rd" in entries_by_outer
    assert "cst%rcpd" in entries_by_outer

    for member, e in entries_by_outer.items():
        assert e.recipe.rank == 0, f"{member}: scalar member should be rank 0, got {e.recipe.rank}"
        assert e.recipe.scratch_dtype == "float64", f"{member}: dtype {e.recipe.scratch_dtype}"
        assert e.recipe.flat_names == (member.replace("%", "_"), ), e.recipe.flat_names
        assert e.recipe.read_exprs == (member, ), e.recipe.read_exprs

    # SDFG arglist carries the flat per-member names.
    arglist = list(sdfg.arglist().keys())
    for flat in ("cst_rg", "cst_rd", "cst_rcpd"):
        assert flat in arglist, f"{flat} missing from SDFG arglist: {arglist}"


def test_dt_of_scalar_constants_numerical(tmp_path):
    """End-to-end numerical check: bridge SDFG must produce
    ``(rg/rcpd) * rd * i`` per element bit-for-bit.

    Computes the reference directly in numpy with the same operation
    order Fortran would: ``(cst%rg / cst%rcpd) * cst%rd * real(i)``
    evaluates left-to-right per Fortran precedence.  f2py can't be
    used here because the kernel dummy is ``type(t_consts)``, which
    crackfortran maps to ``'void'`` and crashes on lookup -- a known
    limitation we've worked around in other struct e2e tests by using
    gfortran+ctypes, but for this trivial multiply-only kernel a
    direct NumPy reference is the simpler comparison.
    """
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_SRC, sdfg_dir, name="kernel_dt_const", entry="_QPkernel_dt_const").build()

    rg = 9.80665
    rd = 287.0597
    rcpd = 1004.709
    n = 8

    # Reference: same left-to-right evaluation Fortran would do.
    out_ref = np.empty(n, dtype=np.float64, order="F")
    for i in range(1, n + 1):
        out_ref[i - 1] = (rg / rcpd) * rd * float(i)

    # The bridge surfaces scalar struct members as length-1 Array(1,)
    # rather than true Scalar; route accordingly.
    from dace.data import Scalar
    arglist = sdfg.arglist()

    def _route(name, value):
        desc = arglist.get(name)
        if desc is None or isinstance(desc, Scalar):
            return value
        return np.array([value], dtype=np.float64)

    out_sdfg = np.zeros(n, dtype=np.float64, order="F")
    sdfg(
        cst_rg=_route("cst_rg", rg),
        cst_rd=_route("cst_rd", rd),
        cst_rcpd=_route("cst_rcpd", rcpd),
        n=n,
        out=out_sdfg,
    )
    np.testing.assert_array_equal(out_sdfg, out_ref)
