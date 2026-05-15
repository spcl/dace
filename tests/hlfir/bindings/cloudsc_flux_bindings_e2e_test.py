"""End-to-end test of the GENERATED ``<entry>_bindings.f90`` module on
a CLOUDSC-representative flat kernel.

The numerical e2e tests (``cloudsc_full`` / ``velocity_full``) call
the SDFG through DaCe's flat Python ABI and never exercise the
emitted Fortran binding module.  This test closes that gap for the
flat-argument case: it runs ``emit_bindings`` for the
loop-carried flux kernel, gfortran-compiles the generated
``cloudscouter_dace`` wrapper, links it against the SDFG ``.so``,
calls it through its Fortran interface, and asserts the result
equals a plain-gfortran reference of the same kernel.

Flat scalars + explicit-shape arrays only (no derived types), so the
``OriginalInterface`` is the kernel's literal dummy list -- the
binding emitter's symbol-population path (``size`` / ``lbound`` into
the SDFG free symbols) is what's under test here.
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

# Same loop-carried flux recurrence as cloudsc_flux_recurrence_repro,
# the cloudsc.F90 Section-8 shape: PFSQLF(JK+1)=PFSQLF(JK) then
# accumulate, cross-array PFSQRF<-PFSQLF(JK), inlined-callee
# INTENT(OUT) section slice through a block wrapper.
_KERNEL_SRC = """
subroutine cloudsc(klon, klev, paph, za, zb, pfsqlf, pfsqrf)
  implicit none
  integer, intent(in) :: klon, klev
  real(8), intent(in)  :: paph(klon, klev+1)
  real(8), intent(in)  :: za(klon, klev), zb(klon, klev)
  real(8), intent(out) :: pfsqlf(klon, klev+1)
  real(8), intent(out) :: pfsqrf(klon, klev+1)
  integer :: jk, jl
  real(8) :: zgdph_r
  do jl = 1, klon
    pfsqlf(jl, 1) = 0.0d0
    pfsqrf(jl, 1) = 0.0d0
  end do
  do jk = 1, klev
    do jl = 1, klon
      zgdph_r = -(paph(jl, jk+1) - paph(jl, jk))
      pfsqlf(jl, jk+1) = pfsqlf(jl, jk)
      pfsqrf(jl, jk+1) = pfsqlf(jl, jk)
      pfsqlf(jl, jk+1) = pfsqlf(jl, jk+1) + za(jl, jk) * zgdph_r
      pfsqrf(jl, jk+1) = pfsqrf(jl, jk+1) + zb(jl, jk) * zgdph_r
    end do
  end do
end subroutine cloudsc

subroutine cloudscouter(klon, klev, nblocks, paph, za, zb, pfsqlf, pfsqrf)
  implicit none
  integer, intent(in) :: klon, klev, nblocks
  real(8), intent(in)  :: paph(klon, klev+1, nblocks)
  real(8), intent(in)  :: za(klon, klev, nblocks), zb(klon, klev, nblocks)
  real(8), intent(out) :: pfsqlf(klon, klev+1, nblocks)
  real(8), intent(out) :: pfsqrf(klon, klev+1, nblocks)
  integer :: ibl
  do ibl = 1, nblocks
    call cloudsc(klon, klev, paph(:, :, ibl), za(:, :, ibl), &
                 zb(:, :, ibl), pfsqlf(:, :, ibl), pfsqrf(:, :, ibl))
  end do
end subroutine cloudscouter
"""

# C-callable driver that calls the GENERATED binding wrapper
# ``cloudscouter_dace`` (defeats f2py via the SDFG handle state, so
# bind(c) + ctypes).  klon/klev/nblocks are passed by value.
_SDFG_DRIVER = """
subroutine run_flux(klon, klev, nblocks, paph, za, zb, lf, rf) &
    bind(c, name='run_flux')
  use iso_c_binding
  use cloudscouter_dace_bindings
  implicit none
  integer(c_int), value :: klon, klev, nblocks
  real(c_double), intent(in)    :: paph(klon, klev+1, nblocks)
  real(c_double), intent(in)    :: za(klon, klev, nblocks)
  real(c_double), intent(in)    :: zb(klon, klev, nblocks)
  real(c_double), intent(inout) :: lf(klon, klev+1, nblocks)
  real(c_double), intent(inout) :: rf(klon, klev+1, nblocks)
  call cloudscouter_dace(klon, klev, nblocks, paph, za, zb, lf, rf)
  call cloudscouter_dace_finalize()
end subroutine run_flux
"""

_REF_DRIVER = """
subroutine run_flux_ref(klon, klev, nblocks, paph, za, zb, lf, rf) &
    bind(c, name='run_flux_ref')
  use iso_c_binding
  implicit none
  integer(c_int), value :: klon, klev, nblocks
  real(c_double), intent(in)    :: paph(klon, klev+1, nblocks)
  real(c_double), intent(in)    :: za(klon, klev, nblocks)
  real(c_double), intent(in)    :: zb(klon, klev, nblocks)
  real(c_double), intent(inout) :: lf(klon, klev+1, nblocks)
  real(c_double), intent(inout) :: rf(klon, klev+1, nblocks)
  external :: cloudscouter
  call cloudscouter(klon, klev, nblocks, paph, za, zb, lf, rf)
end subroutine run_flux_ref
"""


def _compile_so(out_so: Path, *sources: Path, mod_dir: Path, link_so: Path | None = None):
    cmd = ["gfortran", "-shared", "-fPIC", "-O0", "-fno-fast-math", "-ffp-contract=off", f"-J{mod_dir}"]
    cmd.extend(str(s) for s in sources)
    cmd.extend(["-o", str(out_so)])
    if link_so is not None:
        cmd.extend([f"-L{link_so.parent}", f"-Wl,-rpath,{link_so.parent}", f"-l:{link_so.name}"])
    subprocess.check_call(cmd, cwd=mod_dir)


_IFACE = OriginalInterface(
    entry="cloudscouter",
    args=(
        OriginalArg(name="klon", fortran_type="integer", rank=0, shape=(), intent="in", struct_type=None),
        OriginalArg(name="klev", fortran_type="integer", rank=0, shape=(), intent="in", struct_type=None),
        OriginalArg(name="nblocks", fortran_type="integer", rank=0, shape=(), intent="in", struct_type=None),
        OriginalArg(name="paph",
                    fortran_type="real(8)",
                    rank=3,
                    shape=("klon", "klev+1", "nblocks"),
                    intent="in",
                    struct_type=None),
        OriginalArg(name="za",
                    fortran_type="real(8)",
                    rank=3,
                    shape=("klon", "klev", "nblocks"),
                    intent="in",
                    struct_type=None),
        OriginalArg(name="zb",
                    fortran_type="real(8)",
                    rank=3,
                    shape=("klon", "klev", "nblocks"),
                    intent="in",
                    struct_type=None),
        OriginalArg(name="pfsqlf",
                    fortran_type="real(8)",
                    rank=3,
                    shape=("klon", "klev+1", "nblocks"),
                    intent="out",
                    struct_type=None),
        OriginalArg(name="pfsqrf",
                    fortran_type="real(8)",
                    rank=3,
                    shape=("klon", "klev+1", "nblocks"),
                    intent="out",
                    struct_type=None),
    ),
    struct_types={},
    used_modules={},
)


def test_cloudsc_flux_f90_bindings_e2e(tmp_path: Path):
    """The generated ``cloudscouter_dace`` Fortran binding, linked
    against the SDFG ``.so``, must produce the same flux arrays as a
    plain-gfortran reference of the same kernel."""
    # --- SDFG-via-binding library ---
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    builder = build_sdfg(_KERNEL_SRC, sdfg_dir, name="cloudsc", entry="_QPcloudscouter")
    plan = FlattenPlan.from_dict(builder.module.get_flatten_plan())
    sdfg = builder.build()
    # The generated binding references ``__dace_{init,exit}_<iface.entry>``;
    # rename the SDFG so its exported handle symbols match (the test
    # builder otherwise hash-suffixes the name for xdist isolation).
    sdfg.name = "cloudscouter"
    compiled = sdfg.compile()
    so_path = Path(compiled._lib._library_filename)
    fs = sdfg._frozen_signature

    bindings_path = tmp_path / "cloudscouter_bindings.f90"
    emit_bindings(fs, _IFACE, plan, str(bindings_path))
    driver_path = tmp_path / "sdfg_driver.f90"
    driver_path.write_text(_SDFG_DRIVER)

    build_dir = tmp_path / "sdfg_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    sdfg_drv_so = build_dir / "flux_sdfg_driver.so"
    _compile_so(sdfg_drv_so, bindings_path, driver_path, mod_dir=build_dir, link_so=so_path)
    sdfg_lib = ctypes.CDLL(str(sdfg_drv_so))

    # --- gfortran reference library ---
    ref_dir = tmp_path / "ref_build"
    ref_dir.mkdir(parents=True, exist_ok=True)
    kern_path = ref_dir / "kernel.f90"
    kern_path.write_text(_KERNEL_SRC)
    ref_drv_path = ref_dir / "ref_driver.f90"
    ref_drv_path.write_text(_REF_DRIVER)
    ref_so = ref_dir / "flux_ref.so"
    _compile_so(ref_so, kern_path, ref_drv_path, mod_dir=ref_dir)
    ref_lib = ctypes.CDLL(str(ref_so))

    klon, klev, nblocks = 4, 6, 3
    rng = np.random.default_rng(7)

    def _f(shape):
        return np.asfortranarray(rng.standard_normal(shape, dtype=np.float64))

    paph = _f((klon, klev + 1, nblocks))
    za = _f((klon, klev, nblocks))
    zb = _f((klon, klev, nblocks))

    def _call(lib, fn):
        f = getattr(lib, fn)
        f.restype = None
        f.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int] + \
            [ctypes.POINTER(ctypes.c_double)] * 5
        lf = np.zeros((klon, klev + 1, nblocks), dtype=np.float64, order="F")
        rf = np.zeros_like(lf)
        f(klon, klev, nblocks, paph.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
          za.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), zb.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
          lf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), rf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        return lf, rf

    lf_sdfg, rf_sdfg = _call(sdfg_lib, "run_flux")
    lf_ref, rf_ref = _call(ref_lib, "run_flux_ref")

    np.testing.assert_allclose(lf_sdfg, lf_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(rf_sdfg, rf_ref, rtol=1e-12, atol=1e-12)
