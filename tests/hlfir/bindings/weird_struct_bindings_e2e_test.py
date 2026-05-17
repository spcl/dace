"""End-to-end F90-binding coverage for a "weird" derived-type dummy:
a 3-D ``real(c_double)`` array member alongside a *scalar* member.

``struct_bindings_e2e_test`` already covers the two-2D-real-array, the
nested, and the complex-member shapes.  The shape NOT yet exercised
e2e is a struct whose members have *different ranks* -- specifically a
rank-3 array member plus a rank-0 scalar member.  The scalar member is
lifted by the bridge to a length-1 ``Array`` on the SDFG surface (the
scalar-output convention), so this pins the binding's struct
reconstruction across a member-rank mismatch -- the same class of
defect that ``0318f9efe`` fixed for the rank-0 scalar struct member.

Both paths are checked against a gfortran reference of the same
source:

    1. the dace-generated F90 binding (struct reconstructed via
       ``c_f_pointer`` aliases), and
    2. the SDFG invoked directly through the DaCe flat ABI with the
       struct members passed as separate flat companions.

The FlattenPlan is asserted non-empty so the test fails loudly if the
struct silently stops being unpacked.
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

_FFLAGS = ["-O0", "-fno-fast-math", "-ffp-contract=off"]

_NX, _NY, _NZ = 4, 3, 2

_TYPES_SRC = f"""
module mo_w
  use iso_c_binding
  implicit none
  integer, parameter :: NX = {_NX}, NY = {_NY}, NZ = {_NZ}
  type :: t_w
     real(c_double) :: vol(NX, NY, NZ)
     real(c_double) :: coef
  end type t_w
end module mo_w
"""

_KERNEL_SRC = """
subroutine scale3d(w)
  use mo_w
  implicit none
  type(t_w), intent(inout) :: w
  integer :: i, j, k
  do k = 1, NZ
    do j = 1, NY
      do i = 1, NX
        w%vol(i, j, k) = w%vol(i, j, k) * w%coef + real(i + j + k, c_double)
      end do
    end do
  end do
end subroutine scale3d
"""

_SRC = _TYPES_SRC + _KERNEL_SRC

_SDFG_DRIVER = """
subroutine run_scale3d(vol, coef) bind(c, name='run_scale3d')
  use iso_c_binding
  use mo_w, only: t_w, NX, NY, NZ
  use scale3d_dace_bindings
  implicit none
  real(c_double), intent(inout) :: vol(NX, NY, NZ)
  real(c_double), value :: coef
  type(t_w), target :: w
  w%vol = vol
  w%coef = coef
  call scale3d_dace(w)
  call scale3d_dace_finalize()
  vol = w%vol
end subroutine run_scale3d
"""

_REF_DRIVER = """
subroutine run_scale3d_ref(vol, coef) bind(c, name='run_scale3d_ref')
  use iso_c_binding
  use mo_w, only: t_w, NX, NY, NZ
  implicit none
  real(c_double), intent(inout) :: vol(NX, NY, NZ)
  real(c_double), value :: coef
  type(t_w) :: w
  external :: scale3d
  w%vol = vol
  w%coef = coef
  call scale3d(w)
  vol = w%vol
end subroutine run_scale3d_ref
"""


def _compile_so(out_so: Path, *sources: Path, mod_dir: Path, link_so: Path | None = None):
    """gfortran-compile ``sources`` into ``out_so`` with the
    flang-portable flag trio, optionally linking ``link_so``."""
    cmd = ["gfortran", "-shared", "-fPIC", *_FFLAGS, f"-J{mod_dir}"]
    cmd.extend(str(s) for s in sources)
    cmd.extend(["-o", str(out_so)])
    if link_so is not None:
        cmd.extend([f"-L{link_so.parent}", f"-Wl,-rpath,{link_so.parent}", f"-l:{link_so.name}"])
    subprocess.check_call(cmd, cwd=mod_dir)


def test_e2e_mixed_rank_struct(tmp_path: Path):
    """``type(t_w)`` with a rank-3 array member ``vol`` and a rank-0
    scalar member ``coef``.  The kernel does
    ``w%vol = w%vol * w%coef + (i+j+k)``.  Binding AND direct SDFG vs a
    gfortran reference, bit-exact; the FlattenPlan must be non-empty."""
    iface = OriginalInterface(
        entry="scale3d",
        args=(OriginalArg(name="w", fortran_type="type(t_w)", rank=0, intent="inout", struct_type="t_w"), ),
        used_modules={"mo_w": ("t_w", )},
    )

    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    builder = build_sdfg(_SRC, sdfg_dir, name="scale3d", entry="_QPscale3d")
    plan = FlattenPlan.from_dict(builder.module.get_flatten_plan())
    sdfg = builder.build()
    sdfg.name = "scale3d"
    compiled = sdfg.compile()
    so_path = Path(compiled._lib._library_filename)

    flat_targets = {fn for e in plan.entries for fn in e.recipe.flat_names}
    assert plan.entries, "mixed-rank struct dummy must produce a non-empty FlattenPlan"
    assert any("vol" in t for t in flat_targets), flat_targets
    assert any("coef" in t for t in flat_targets), flat_targets

    bindings_path = tmp_path / "scale3d_bindings.f90"
    emit_bindings(sdfg._frozen_signature, iface, plan, str(bindings_path))
    types_path = tmp_path / "scale3d_types.f90"
    types_path.write_text(_TYPES_SRC)
    drv_path = tmp_path / "scale3d_driver.f90"
    drv_path.write_text(_SDFG_DRIVER)

    build_dir = tmp_path / "bind_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    drv_so = build_dir / "scale3d_drv.so"
    _compile_so(drv_so, types_path, bindings_path, drv_path, mod_dir=build_dir, link_so=so_path)
    lib = ctypes.CDLL(str(drv_so))

    ref_dir = tmp_path / "ref_build"
    ref_dir.mkdir(parents=True, exist_ok=True)
    rt = ref_dir / "scale3d_types.f90"
    rt.write_text(_TYPES_SRC)
    rk = ref_dir / "scale3d_k.f90"
    rk.write_text(_KERNEL_SRC)
    rd = ref_dir / "scale3d_d.f90"
    rd.write_text(_REF_DRIVER)
    ref_so = ref_dir / "scale3d_ref.so"
    _compile_so(ref_so, rt, rk, rd, mod_dir=ref_dir)
    ref = ctypes.CDLL(str(ref_so))

    rng = np.random.default_rng(43)
    vol0 = np.asfortranarray(rng.standard_normal((_NX, _NY, _NZ)))
    coef = 0.625
    dp = ctypes.POINTER(ctypes.c_double)

    vol_ref = vol0.copy(order="F")
    fref = ref.run_scale3d_ref
    fref.restype = None
    fref.argtypes = [dp, ctypes.c_double]
    fref(vol_ref.ctypes.data_as(dp), ctypes.c_double(coef))

    vol_bind = vol0.copy(order="F")
    fbind = lib.run_scale3d
    fbind.restype = None
    fbind.argtypes = [dp, ctypes.c_double]
    fbind(vol_bind.ctypes.data_as(dp), ctypes.c_double(coef))
    np.testing.assert_allclose(vol_bind, vol_ref, rtol=1e-12, atol=1e-12)

    # Direct SDFG path: struct members as separate flat companions.
    # HLFIR-bridged arrays keep Fortran (column-major) layout on the
    # flat ABI, so the companion must be F-ordered (matching how the
    # binding driver's ``w%vol = vol`` assignment lays it out).
    vol_d = vol0.copy(order="F")
    coef_d = np.array([coef], dtype=np.float64)
    sdfg(w_vol=vol_d, w_coef=coef_d)
    np.testing.assert_allclose(vol_d, vol_ref, rtol=1e-12, atol=1e-12)
