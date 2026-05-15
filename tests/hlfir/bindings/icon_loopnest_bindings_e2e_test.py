"""f90-binding e2e for an ICON velocity-advection loopnest.

Exercises the generated ``<entry>_bindings.f90`` on the
``kernel_flat`` of ``icon_loopnest_2`` (the direct
``z_w_concorr_me = vn*ddxn + vt*ddxt`` stencil over a partial
vertical range).  This is the ICON-shaped counterpart of
``cloudsc_flux_bindings_e2e_test``: flat explicit-shape arrays +
integer dimension/loop-bound scalars, no derived types, so the
``OriginalInterface`` is the kernel's literal dummy list and the
binding emitter's symbol-population path is what's under test.

Full-ICON ``velocity_tendencies`` (5 derived-type dummies) needs a
hand-authored derived-type ``OriginalInterface`` and is tracked
separately; the loopnest carve-outs are the tractable flat surface.
"""

import ctypes
import re
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

_LOOPNEST = Path(__file__).resolve().parent.parent / "icon_loopnests" / "icon_loopnest_2.f90"


def _extract_flat_kernel(bundle: Path) -> str:
    """Slice the bare ``subroutine kernel_flat(...) ... end subroutine``
    out of the loopnest bundle (drops the enclosing module + bench
    program so the bridge sees a free-standing subroutine)."""
    txt = bundle.read_text()
    m = re.search(r"(?is)(subroutine\s+kernel_flat\s*\(.*?\bend\s+subroutine)", txt)
    if not m:
        raise RuntimeError("kernel_flat not found")
    return m.group(1) + "\n"


def _compile_so(out_so: Path, *sources: Path, mod_dir: Path, link_so: Path | None = None):
    cmd = ["gfortran", "-shared", "-fPIC", "-O0", "-fno-fast-math", "-ffp-contract=off", f"-J{mod_dir}"]
    cmd.extend(str(s) for s in sources)
    cmd.extend(["-o", str(out_so)])
    if link_so is not None:
        cmd.extend([f"-L{link_so.parent}", f"-Wl,-rpath,{link_so.parent}", f"-l:{link_so.name}"])
    subprocess.check_call(cmd, cwd=mod_dir)


_3D = ("nproma", "nlev", "nblks_e")


def _arr(name, intent):
    return OriginalArg(name=name, fortran_type="real(8)", rank=3, shape=_3D, intent=intent, struct_type=None)


def _int(name):
    return OriginalArg(name=name, fortran_type="integer", rank=0, shape=(), intent="in", struct_type=None)


_IFACE = OriginalInterface(
    entry="kernel_flat",
    args=(
        _arr("vn", "in"),
        _arr("vt", "in"),
        _arr("ddxn", "in"),
        _arr("ddxt", "in"),
        _arr("z_w_concorr_me", "inout"),
        _int("nproma"),
        _int("nlev"),
        _int("nblks_e"),
        _int("nflatlev"),
        _int("i_startblk"),
        _int("i_endblk"),
        _int("i_startidx"),
        _int("i_endidx"),
    ),
    struct_types={},
    used_modules={},
)

_SDFG_DRIVER = """
subroutine run_ln2(vn, vt, ddxn, ddxt, z, &
    nproma, nlev, nblks_e, nflatlev, isb, ieb, isi, iei) &
    bind(c, name='run_ln2')
  use iso_c_binding
  use kernel_flat_dace_bindings
  implicit none
  integer(c_int), value :: nproma, nlev, nblks_e, nflatlev
  integer(c_int), value :: isb, ieb, isi, iei
  real(c_double), intent(in)    :: vn(nproma,nlev,nblks_e), vt(nproma,nlev,nblks_e)
  real(c_double), intent(in)    :: ddxn(nproma,nlev,nblks_e), ddxt(nproma,nlev,nblks_e)
  real(c_double), intent(inout) :: z(nproma,nlev,nblks_e)
  call kernel_flat_dace(vn, vt, ddxn, ddxt, z, nproma, nlev, nblks_e, &
                        nflatlev, isb, ieb, isi, iei)
  call kernel_flat_dace_finalize()
end subroutine run_ln2
"""

_REF_DRIVER = """
subroutine run_ln2_ref(vn, vt, ddxn, ddxt, z, &
    nproma, nlev, nblks_e, nflatlev, isb, ieb, isi, iei) &
    bind(c, name='run_ln2_ref')
  use iso_c_binding
  implicit none
  integer(c_int), value :: nproma, nlev, nblks_e, nflatlev
  integer(c_int), value :: isb, ieb, isi, iei
  real(c_double), intent(in)    :: vn(nproma,nlev,nblks_e), vt(nproma,nlev,nblks_e)
  real(c_double), intent(in)    :: ddxn(nproma,nlev,nblks_e), ddxt(nproma,nlev,nblks_e)
  real(c_double), intent(inout) :: z(nproma,nlev,nblks_e)
  external :: kernel_flat
  call kernel_flat(vn, vt, ddxn, ddxt, z, nproma, nlev, nblks_e, &
                   nflatlev, isb, ieb, isi, iei)
end subroutine run_ln2_ref
"""


def test_icon_loopnest2_f90_bindings_e2e(tmp_path: Path):
    """The generated ``kernel_flat_dace`` binding for ICON loopnest 2,
    linked against the SDFG ``.so``, must equal a plain-gfortran
    reference of the same flat kernel."""
    flat_src = _extract_flat_kernel(_LOOPNEST)

    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    builder = build_sdfg(flat_src, sdfg_dir, name="kernel_flat", entry="_QPkernel_flat")
    plan = FlattenPlan.from_dict(builder.module.get_flatten_plan())
    sdfg = builder.build()
    sdfg.name = "kernel_flat"
    compiled = sdfg.compile()
    so_path = Path(compiled._lib._library_filename)

    assert not plan.entries, "flat ICON kernel must not flatten"

    bindings_path = tmp_path / "kernel_flat_bindings.f90"
    emit_bindings(sdfg._frozen_signature, _IFACE, plan, str(bindings_path))
    drv_path = tmp_path / "ln2_driver.f90"
    drv_path.write_text(_SDFG_DRIVER)
    build_dir = tmp_path / "sdfg_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    sdfg_drv_so = build_dir / "ln2_sdfg.so"
    _compile_so(sdfg_drv_so, bindings_path, drv_path, mod_dir=build_dir, link_so=so_path)
    sdfg_lib = ctypes.CDLL(str(sdfg_drv_so))

    ref_dir = tmp_path / "ref_build"
    ref_dir.mkdir(parents=True, exist_ok=True)
    kp = ref_dir / "kernel.f90"
    kp.write_text(flat_src)
    rd = ref_dir / "ref_driver.f90"
    rd.write_text(_REF_DRIVER)
    ref_so = ref_dir / "ln2_ref.so"
    _compile_so(ref_so, kp, rd, mod_dir=ref_dir)
    ref_lib = ctypes.CDLL(str(ref_so))

    nproma, nlev, nblks_e, nflatlev = 32, 16, 8, 4
    isb, ieb, isi, iei = 1, nblks_e, 1, nproma
    rng = np.random.default_rng(2)

    def _f():
        return np.asfortranarray(rng.random((nproma, nlev, nblks_e), dtype=np.float64))

    vn, vt, ddxn, ddxt = _f(), _f(), _f(), _f()

    def _call(lib, fn):
        f = getattr(lib, fn)
        f.restype = None
        f.argtypes = [ctypes.POINTER(ctypes.c_double)] * 5 + [ctypes.c_int] * 8
        z = np.zeros((nproma, nlev, nblks_e), dtype=np.float64, order="F")
        dp = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        f(dp(vn), dp(vt), dp(ddxn), dp(ddxt), dp(z), nproma, nlev, nblks_e, nflatlev, isb, ieb, isi, iei)
        return z

    z_sdfg = _call(sdfg_lib, "run_ln2")
    z_ref = _call(ref_lib, "run_ln2_ref")
    assert z_ref.any(), "reference produced all zeros"
    np.testing.assert_allclose(z_sdfg, z_ref, rtol=1e-12, atol=1e-12)
