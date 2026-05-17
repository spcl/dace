"""f90-binding e2e for the ICON velocity-advection loopnests.

Exercises the generated ``<entry>_bindings.f90`` on the
``kernel_flat`` of every ICON velocity-tendencies representative
loopnest (1, 2, 3, 4, 5, 6).  Each is the ICON-shaped counterpart of
``cloudsc_flux_bindings_e2e_test``: flat explicit-shape arrays +
integer dimension/loop-bound scalars, no derived types, so the
``OriginalInterface`` is the kernel's literal dummy list and the
binding emitter's symbol-population path is what's under test.

The covered loopnests are:

* ``2`` -- direct ``z_w_concorr_me = vn*ddxn + vt*ddxt`` stencil.
* ``1`` -- two-way cell + vertex indirect stencil (3D index tables).
* ``3`` -- direct stencil with per-level deepatmo profiles.
* ``4`` -- indirect stencil + ``vn_ie(jk)-vn_ie(jk+1)`` term writing
  a 4-D output indexed on its last dim by the ``ntnd`` scalar
  (``vn_ie`` has an ``nlev+1`` extent).
* ``5`` -- horizontal boundary writes with literal level indices and
  ``nlev-1``/``nlev-2`` loop-bound arithmetic.
* ``6`` -- ``levelmask(jk) = ANY(levmask(:, jk))`` LOGICAL reduction
  exercising the binding's ``logical(c_bool)`` bridge.

Unlike ``test_sdfg_equivalence.py`` (which drives the SDFG through
DaCe's flat Python ABI and an f2py reference), these tests compile
and link the *generated* ``.f90`` binding against the SDFG ``.so``,
call it through its Fortran/ctypes interface, and compare against a
plain-gfortran reference of the same flat kernel.

Full-ICON ``velocity_tendencies`` (5 derived-type dummies) needs a
hand-authored derived-type ``OriginalInterface`` and is tracked
separately; the loopnest carve-outs are the tractable flat surface.
"""

import ctypes
import re
import shutil
import subprocess
from dataclasses import replace
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

_LOOPNESTS_DIR = Path(__file__).resolve().parent.parent / "icon_loopnests"
_LOOPNEST = _LOOPNESTS_DIR / "icon_loopnest_2.f90"


def _extract_flat_kernel(bundle: Path) -> str:
    """Slice the bare ``subroutine kernel_flat(...) ... end subroutine``
    out of the loopnest bundle (drops the enclosing module + bench
    program so the bridge sees a free-standing subroutine).

    :param bundle: path to an ``icon_loopnest_N.f90`` source bundle.
    :returns: the standalone ``kernel_flat`` subroutine text.
    :raises RuntimeError: if no ``kernel_flat`` subroutine is found.
    """
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


def _arg_real(name: str, rank: int, shape: tuple, intent: str) -> "OriginalArg":
    """Build a ``real(8)`` flat ``OriginalArg`` (no derived type)."""
    return OriginalArg(name=name, fortran_type="real(8)", rank=rank, shape=shape, intent=intent, struct_type=None)


def _arg_int(name: str, rank: int = 0, shape: tuple = (), intent: str = "in") -> "OriginalArg":
    """Build an ``integer`` flat ``OriginalArg`` (scalar by default)."""
    return OriginalArg(name=name, fortran_type="integer", rank=rank, shape=shape, intent=intent, struct_type=None)


def _arg_logical(name: str, rank: int, shape: tuple, intent: str) -> "OriginalArg":
    """Build a default-``logical`` flat ``OriginalArg`` (no derived type)."""
    return OriginalArg(name=name, fortran_type="logical", rank=rank, shape=shape, intent=intent, struct_type=None)


def _build_binding_and_ref(
    loopnest_no: int,
    iface: "OriginalInterface",
    sdfg_driver: str,
    ref_driver: str,
    tmp_path: Path,
):
    """Build the SDFG-binding and plain-gfortran reference shared libraries
    for one ICON loopnest and return ``(sdfg_lib, ref_lib)`` ctypes handles.

    The flat ``kernel_flat`` is sliced from ``icon_loopnest_<n>.f90``,
    lowered through the default HLFIR pipeline, compiled to an SDFG
    ``.so``; the generated ``kernel_flat_bindings.f90`` is gfortran-linked
    against it and driven by ``sdfg_driver``.  The reference path
    gfortran-compiles the same flat kernel behind ``ref_driver``.

    :param loopnest_no: ICON loopnest number (1, 3, 4, 5 or 6).
    :param iface: literal flat ``OriginalInterface`` for ``kernel_flat``;
                  its ``entry`` is rewritten to a per-loopnest unique
                  name so concurrently-loaded SDFGs don't collide.
    :param sdfg_driver: ``bind(c)`` Fortran driver calling the binding;
                  the ``@ENTRY@`` placeholder is substituted with the
                  per-loopnest entry name.
    :param ref_driver: ``bind(c)`` Fortran driver calling the raw kernel.
    :param tmp_path: pytest scratch directory.
    :returns: ``(sdfg_lib, ref_lib)`` ``ctypes.CDLL`` handles.
    :raises AssertionError: if the flat kernel unexpectedly flattens.
    """
    flat_src = _extract_flat_kernel(_LOOPNESTS_DIR / f"icon_loopnest_{loopnest_no}.f90")

    # Every loopnest's flat kernel is the same subroutine ``kernel_flat``;
    # DaCe keys its in-process library cache by SDFG name, so reusing
    # ``kernel_flat`` across tests recompiles under a hash-suffixed name
    # and the binding's ``__dace_{init,exit}_kernel_flat`` symbols no
    # longer match.  Give each loopnest a unique entry/SDFG name and
    # thread it through the binding + driver.
    entry = f"kernel_flat_ln{loopnest_no}"
    iface = replace(iface, entry=entry)

    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    builder = build_sdfg(flat_src, sdfg_dir, name="kernel_flat", entry="_QPkernel_flat")
    plan = FlattenPlan.from_dict(builder.module.get_flatten_plan())
    sdfg = builder.build()
    sdfg.name = entry
    compiled = sdfg.compile()
    so_path = Path(compiled._lib._library_filename)

    assert not plan.entries, "flat ICON kernel must not flatten"

    bindings_path = tmp_path / f"{entry}_bindings.f90"
    emit_bindings(sdfg._frozen_signature, iface, plan, str(bindings_path))
    drv_path = tmp_path / "sdfg_driver.f90"
    drv_path.write_text(sdfg_driver.replace("@ENTRY@", entry))
    build_dir = tmp_path / "sdfg_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    sdfg_drv_so = build_dir / "ln_sdfg.so"
    _compile_so(sdfg_drv_so, bindings_path, drv_path, mod_dir=build_dir, link_so=so_path)
    sdfg_lib = ctypes.CDLL(str(sdfg_drv_so))

    ref_dir = tmp_path / "ref_build"
    ref_dir.mkdir(parents=True, exist_ok=True)
    kp = ref_dir / "kernel.f90"
    kp.write_text(flat_src)
    rd = ref_dir / "ref_driver.f90"
    rd.write_text(ref_driver)
    ref_so = ref_dir / "ln_ref.so"
    _compile_so(ref_so, kp, rd, mod_dir=ref_dir)
    ref_lib = ctypes.CDLL(str(ref_so))

    return sdfg_lib, ref_lib


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


# ---------------------------------------------------------------------------
# Loopnest 1  --  two-way cell + vertex indirect stencil
# ---------------------------------------------------------------------------

_LN1_IFACE = OriginalInterface(
    entry="kernel_flat",
    args=(
        _arg_real("vn_ie", 3, ("nproma", "nlev", "nblks_e"), "in"),
        _arg_real("inv_dual", 2, ("nproma", "nblks_e"), "in"),
        _arg_real("inv_primal", 2, ("nproma", "nblks_e"), "in"),
        _arg_real("tangent", 2, ("nproma", "nblks_e"), "in"),
        _arg_real("w", 3, ("nproma", "nlev", "nblks_c"), "in"),
        _arg_real("z_vt_ie", 3, ("nproma", "nlev", "nblks_e"), "in"),
        _arg_real("z_w_v", 3, ("nproma", "nlev", "nblks_v"), "in"),
        _arg_int("icidx", 3, ("nproma", "nblks_e", "2")),
        _arg_int("icblk", 3, ("nproma", "nblks_e", "2")),
        _arg_int("ividx", 3, ("nproma", "nblks_e", "2")),
        _arg_int("ivblk", 3, ("nproma", "nblks_e", "2")),
        _arg_real("z_v_grad_w", 3, ("nproma", "nlev", "nblks_e"), "inout"),
        _arg_int("nproma"),
        _arg_int("nlev"),
        _arg_int("nblks_e"),
        _arg_int("nblks_c"),
        _arg_int("nblks_v"),
        _arg_int("i_startblk"),
        _arg_int("i_endblk"),
        _arg_int("i_startidx"),
        _arg_int("i_endidx"),
    ),
    struct_types={},
    used_modules={},
)

_LN1_SDFG_DRIVER = """
subroutine run_ln1(vn_ie, inv_dual, inv_primal, tangent, w, z_vt_ie, z_w_v, &
    icidx, icblk, ividx, ivblk, z, &
    nproma, nlev, nblks_e, nblks_c, nblks_v, isb, ieb, isi, iei) &
    bind(c, name='run_ln1')
  use iso_c_binding
  use @ENTRY@_dace_bindings
  implicit none
  integer(c_int), value :: nproma, nlev, nblks_e, nblks_c, nblks_v
  integer(c_int), value :: isb, ieb, isi, iei
  real(c_double), intent(in)    :: vn_ie(nproma,nlev,nblks_e)
  real(c_double), intent(in)    :: inv_dual(nproma,nblks_e), inv_primal(nproma,nblks_e)
  real(c_double), intent(in)    :: tangent(nproma,nblks_e)
  real(c_double), intent(in)    :: w(nproma,nlev,nblks_c)
  real(c_double), intent(in)    :: z_vt_ie(nproma,nlev,nblks_e)
  real(c_double), intent(in)    :: z_w_v(nproma,nlev,nblks_v)
  integer(c_int), intent(in)    :: icidx(nproma,nblks_e,2), icblk(nproma,nblks_e,2)
  integer(c_int), intent(in)    :: ividx(nproma,nblks_e,2), ivblk(nproma,nblks_e,2)
  real(c_double), intent(inout) :: z(nproma,nlev,nblks_e)
  call @ENTRY@_dace(vn_ie, inv_dual, inv_primal, tangent, w, z_vt_ie, z_w_v, &
       icidx, icblk, ividx, ivblk, z, &
       nproma, nlev, nblks_e, nblks_c, nblks_v, isb, ieb, isi, iei)
  call @ENTRY@_dace_finalize()
end subroutine run_ln1
"""

_LN1_REF_DRIVER = """
subroutine run_ln1_ref(vn_ie, inv_dual, inv_primal, tangent, w, z_vt_ie, z_w_v, &
    icidx, icblk, ividx, ivblk, z, &
    nproma, nlev, nblks_e, nblks_c, nblks_v, isb, ieb, isi, iei) &
    bind(c, name='run_ln1_ref')
  use iso_c_binding
  implicit none
  integer(c_int), value :: nproma, nlev, nblks_e, nblks_c, nblks_v
  integer(c_int), value :: isb, ieb, isi, iei
  real(c_double), intent(in)    :: vn_ie(nproma,nlev,nblks_e)
  real(c_double), intent(in)    :: inv_dual(nproma,nblks_e), inv_primal(nproma,nblks_e)
  real(c_double), intent(in)    :: tangent(nproma,nblks_e)
  real(c_double), intent(in)    :: w(nproma,nlev,nblks_c)
  real(c_double), intent(in)    :: z_vt_ie(nproma,nlev,nblks_e)
  real(c_double), intent(in)    :: z_w_v(nproma,nlev,nblks_v)
  integer(c_int), intent(in)    :: icidx(nproma,nblks_e,2), icblk(nproma,nblks_e,2)
  integer(c_int), intent(in)    :: ividx(nproma,nblks_e,2), ivblk(nproma,nblks_e,2)
  real(c_double), intent(inout) :: z(nproma,nlev,nblks_e)
  external :: kernel_flat
  call kernel_flat(vn_ie, inv_dual, inv_primal, tangent, w, z_vt_ie, z_w_v, &
       icidx, icblk, ividx, ivblk, z, &
       nproma, nlev, nblks_e, nblks_c, nblks_v, isb, ieb, isi, iei)
end subroutine run_ln1_ref
"""


def test_icon_loopnest1_f90_bindings_e2e(tmp_path: Path):
    """The generated ``kernel_flat_dace`` binding for ICON loopnest 1
    (two-way cell + vertex indirect stencil), linked against the SDFG
    ``.so``, must equal a plain-gfortran reference of the same kernel."""
    sdfg_lib, ref_lib = _build_binding_and_ref(1, _LN1_IFACE, _LN1_SDFG_DRIVER, _LN1_REF_DRIVER, tmp_path)

    nproma, nlev, nblks_e, nblks_c, nblks_v = 32, 16, 8, 8, 8
    isb, ieb, isi, iei = 1, nblks_e, 1, nproma
    rng = np.random.default_rng(1)

    def _f(shape):
        return np.asfortranarray(rng.random(shape, dtype=np.float64))

    def _idx(shape, hi):
        return np.asfortranarray(rng.integers(1, hi + 1, size=shape, dtype=np.int32))

    vn_ie = _f((nproma, nlev, nblks_e))
    inv_dual = _f((nproma, nblks_e))
    inv_primal = _f((nproma, nblks_e))
    tangent = _f((nproma, nblks_e))
    w = _f((nproma, nlev, nblks_c))
    z_vt_ie = _f((nproma, nlev, nblks_e))
    z_w_v = _f((nproma, nlev, nblks_v))
    icidx = _idx((nproma, nblks_e, 2), nproma)
    icblk = _idx((nproma, nblks_e, 2), nblks_c)
    ividx = _idx((nproma, nblks_e, 2), nproma)
    ivblk = _idx((nproma, nblks_e, 2), nblks_v)

    dp = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    ip = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    def _call(lib, fn):
        f = getattr(lib, fn)
        f.restype = None
        f.argtypes = ([ctypes.POINTER(ctypes.c_double)] * 7 + [ctypes.POINTER(ctypes.c_int)] * 4 +
                      [ctypes.POINTER(ctypes.c_double)] + [ctypes.c_int] * 9)
        z = np.zeros((nproma, nlev, nblks_e), dtype=np.float64, order="F")
        f(dp(vn_ie), dp(inv_dual), dp(inv_primal), dp(tangent), dp(w), dp(z_vt_ie), dp(z_w_v), ip(icidx), ip(icblk),
          ip(ividx), ip(ivblk), dp(z), nproma, nlev, nblks_e, nblks_c, nblks_v, isb, ieb, isi, iei)
        return z

    z_sdfg = _call(sdfg_lib, "run_ln1")
    z_ref = _call(ref_lib, "run_ln1_ref")
    assert z_ref.any(), "reference produced all zeros"
    np.testing.assert_allclose(z_sdfg, z_ref, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Loopnest 3  --  direct stencil with per-level deepatmo profiles
# ---------------------------------------------------------------------------

_LN3_IFACE = OriginalInterface(
    entry="kernel_flat",
    args=(
        _arg_real("vn_ie", 3, ("nproma", "nlev", "nblks_e"), "in"),
        _arg_real("z_vt_ie", 3, ("nproma", "nlev", "nblks_e"), "in"),
        _arg_real("ft_e", 2, ("nproma", "nblks_e"), "in"),
        _arg_real("fn_e", 2, ("nproma", "nblks_e"), "in"),
        _arg_real("gradh", 1, ("nlev", ), "in"),
        _arg_real("invr", 1, ("nlev", ), "in"),
        _arg_real("z_v_grad_w", 3, ("nproma", "nlev", "nblks_e"), "inout"),
        _arg_int("nproma"),
        _arg_int("nlev"),
        _arg_int("nblks_e"),
        _arg_int("i_startblk"),
        _arg_int("i_endblk"),
        _arg_int("i_startidx"),
        _arg_int("i_endidx"),
    ),
    struct_types={},
    used_modules={},
)

_LN3_SDFG_DRIVER = """
subroutine run_ln3(vn_ie, z_vt_ie, ft_e, fn_e, gradh, invr, z, &
    nproma, nlev, nblks_e, isb, ieb, isi, iei) &
    bind(c, name='run_ln3')
  use iso_c_binding
  use @ENTRY@_dace_bindings
  implicit none
  integer(c_int), value :: nproma, nlev, nblks_e, isb, ieb, isi, iei
  real(c_double), intent(in)    :: vn_ie(nproma,nlev,nblks_e), z_vt_ie(nproma,nlev,nblks_e)
  real(c_double), intent(in)    :: ft_e(nproma,nblks_e), fn_e(nproma,nblks_e)
  real(c_double), intent(in)    :: gradh(nlev), invr(nlev)
  real(c_double), intent(inout) :: z(nproma,nlev,nblks_e)
  call @ENTRY@_dace(vn_ie, z_vt_ie, ft_e, fn_e, gradh, invr, z, &
       nproma, nlev, nblks_e, isb, ieb, isi, iei)
  call @ENTRY@_dace_finalize()
end subroutine run_ln3
"""

_LN3_REF_DRIVER = """
subroutine run_ln3_ref(vn_ie, z_vt_ie, ft_e, fn_e, gradh, invr, z, &
    nproma, nlev, nblks_e, isb, ieb, isi, iei) &
    bind(c, name='run_ln3_ref')
  use iso_c_binding
  implicit none
  integer(c_int), value :: nproma, nlev, nblks_e, isb, ieb, isi, iei
  real(c_double), intent(in)    :: vn_ie(nproma,nlev,nblks_e), z_vt_ie(nproma,nlev,nblks_e)
  real(c_double), intent(in)    :: ft_e(nproma,nblks_e), fn_e(nproma,nblks_e)
  real(c_double), intent(in)    :: gradh(nlev), invr(nlev)
  real(c_double), intent(inout) :: z(nproma,nlev,nblks_e)
  external :: kernel_flat
  call kernel_flat(vn_ie, z_vt_ie, ft_e, fn_e, gradh, invr, z, &
       nproma, nlev, nblks_e, isb, ieb, isi, iei)
end subroutine run_ln3_ref
"""


def test_icon_loopnest3_f90_bindings_e2e(tmp_path: Path):
    """The generated ``kernel_flat_dace`` binding for ICON loopnest 3
    (direct stencil with per-level deepatmo profiles), linked against
    the SDFG ``.so``, must equal a plain-gfortran reference."""
    sdfg_lib, ref_lib = _build_binding_and_ref(3, _LN3_IFACE, _LN3_SDFG_DRIVER, _LN3_REF_DRIVER, tmp_path)

    nproma, nlev, nblks_e = 32, 16, 8
    isb, ieb, isi, iei = 1, nblks_e, 1, nproma
    rng = np.random.default_rng(3)

    def _f(shape):
        return np.asfortranarray(rng.random(shape, dtype=np.float64))

    vn_ie = _f((nproma, nlev, nblks_e))
    z_vt_ie = _f((nproma, nlev, nblks_e))
    ft_e = _f((nproma, nblks_e))
    fn_e = _f((nproma, nblks_e))
    gradh = _f((nlev, ))
    invr = _f((nlev, ))
    z_init = _f((nproma, nlev, nblks_e))

    dp = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    def _call(lib, fn):
        f = getattr(lib, fn)
        f.restype = None
        f.argtypes = [ctypes.POINTER(ctypes.c_double)] * 7 + [ctypes.c_int] * 7
        z = np.array(z_init, order="F")
        f(dp(vn_ie), dp(z_vt_ie), dp(ft_e), dp(fn_e), dp(gradh), dp(invr), dp(z), nproma, nlev, nblks_e, isb, ieb, isi,
          iei)
        return z

    z_sdfg = _call(sdfg_lib, "run_ln3")
    z_ref = _call(ref_lib, "run_ln3_ref")
    assert not np.array_equal(z_ref, z_init), "reference left z unchanged"
    np.testing.assert_allclose(z_sdfg, z_ref, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Loopnest 4  --  indirect stencil + vn_ie(jk)-vn_ie(jk+1), 4-D output
# ---------------------------------------------------------------------------

_LN4_IFACE = OriginalInterface(
    entry="kernel_flat",
    args=(
        _arg_real("vt", 3, ("nproma", "nlev", "nblks_e"), "in"),
        _arg_real("vn_ie", 3, ("nproma", "nlev+1", "nblks_e"), "in"),
        _arg_real("f_e", 2, ("nproma", "nblks_e"), "in"),
        _arg_real("coeff_gradekin", 3, ("nproma", "2", "nblks_e"), "in"),
        _arg_real("c_lin_e", 3, ("nproma", "2", "nblks_e"), "in"),
        _arg_real("ddqz", 3, ("nproma", "nlev", "nblks_e"), "in"),
        _arg_real("z_kin_hor_e", 3, ("nproma", "nlev", "nblks_e"), "in"),
        _arg_real("z_ekinh", 3, ("nproma", "nlev", "nblks_c"), "in"),
        _arg_real("zeta", 3, ("nproma", "nlev", "nblks_v"), "in"),
        _arg_real("z_w_con_c_full", 3, ("nproma", "nlev", "nblks_c"), "in"),
        _arg_int("icidx", 3, ("nproma", "nblks_e", "2")),
        _arg_int("icblk", 3, ("nproma", "nblks_e", "2")),
        _arg_int("ividx", 3, ("nproma", "nblks_e", "2")),
        _arg_int("ivblk", 3, ("nproma", "nblks_e", "2")),
        _arg_real("ddt_vn_apc_pc", 4, ("nproma", "nlev", "nblks_e", "nproma_tnd"), "inout"),
        _arg_int("ntnd"),
        _arg_int("nproma"),
        _arg_int("nlev"),
        _arg_int("nblks_e"),
        _arg_int("nblks_c"),
        _arg_int("nblks_v"),
        _arg_int("nproma_tnd"),
        _arg_int("i_startblk"),
        _arg_int("i_endblk"),
        _arg_int("i_startidx"),
        _arg_int("i_endidx"),
    ),
    struct_types={},
    used_modules={},
)

_LN4_SDFG_DRIVER = """
subroutine run_ln4(vt, vn_ie, f_e, coeff_gradekin, c_lin_e, ddqz, &
    z_kin_hor_e, z_ekinh, zeta, z_w_con_c_full, &
    icidx, icblk, ividx, ivblk, ddt, ntnd, &
    nproma, nlev, nblks_e, nblks_c, nblks_v, nproma_tnd, &
    isb, ieb, isi, iei) bind(c, name='run_ln4')
  use iso_c_binding
  use @ENTRY@_dace_bindings
  implicit none
  integer(c_int), value :: ntnd, nproma, nlev, nblks_e, nblks_c, nblks_v, nproma_tnd
  integer(c_int), value :: isb, ieb, isi, iei
  real(c_double), intent(in)    :: vt(nproma,nlev,nblks_e), vn_ie(nproma,nlev+1,nblks_e)
  real(c_double), intent(in)    :: f_e(nproma,nblks_e)
  real(c_double), intent(in)    :: coeff_gradekin(nproma,2,nblks_e), c_lin_e(nproma,2,nblks_e)
  real(c_double), intent(in)    :: ddqz(nproma,nlev,nblks_e)
  real(c_double), intent(in)    :: z_kin_hor_e(nproma,nlev,nblks_e)
  real(c_double), intent(in)    :: z_ekinh(nproma,nlev,nblks_c)
  real(c_double), intent(in)    :: zeta(nproma,nlev,nblks_v)
  real(c_double), intent(in)    :: z_w_con_c_full(nproma,nlev,nblks_c)
  integer(c_int), intent(in)    :: icidx(nproma,nblks_e,2), icblk(nproma,nblks_e,2)
  integer(c_int), intent(in)    :: ividx(nproma,nblks_e,2), ivblk(nproma,nblks_e,2)
  real(c_double), intent(inout) :: ddt(nproma,nlev,nblks_e,nproma_tnd)
  call @ENTRY@_dace(vt, vn_ie, f_e, coeff_gradekin, c_lin_e, ddqz, &
       z_kin_hor_e, z_ekinh, zeta, z_w_con_c_full, &
       icidx, icblk, ividx, ivblk, ddt, ntnd, &
       nproma, nlev, nblks_e, nblks_c, nblks_v, nproma_tnd, isb, ieb, isi, iei)
  call @ENTRY@_dace_finalize()
end subroutine run_ln4
"""

_LN4_REF_DRIVER = """
subroutine run_ln4_ref(vt, vn_ie, f_e, coeff_gradekin, c_lin_e, ddqz, &
    z_kin_hor_e, z_ekinh, zeta, z_w_con_c_full, &
    icidx, icblk, ividx, ivblk, ddt, ntnd, &
    nproma, nlev, nblks_e, nblks_c, nblks_v, nproma_tnd, &
    isb, ieb, isi, iei) bind(c, name='run_ln4_ref')
  use iso_c_binding
  implicit none
  integer(c_int), value :: ntnd, nproma, nlev, nblks_e, nblks_c, nblks_v, nproma_tnd
  integer(c_int), value :: isb, ieb, isi, iei
  real(c_double), intent(in)    :: vt(nproma,nlev,nblks_e), vn_ie(nproma,nlev+1,nblks_e)
  real(c_double), intent(in)    :: f_e(nproma,nblks_e)
  real(c_double), intent(in)    :: coeff_gradekin(nproma,2,nblks_e), c_lin_e(nproma,2,nblks_e)
  real(c_double), intent(in)    :: ddqz(nproma,nlev,nblks_e)
  real(c_double), intent(in)    :: z_kin_hor_e(nproma,nlev,nblks_e)
  real(c_double), intent(in)    :: z_ekinh(nproma,nlev,nblks_c)
  real(c_double), intent(in)    :: zeta(nproma,nlev,nblks_v)
  real(c_double), intent(in)    :: z_w_con_c_full(nproma,nlev,nblks_c)
  integer(c_int), intent(in)    :: icidx(nproma,nblks_e,2), icblk(nproma,nblks_e,2)
  integer(c_int), intent(in)    :: ividx(nproma,nblks_e,2), ivblk(nproma,nblks_e,2)
  real(c_double), intent(inout) :: ddt(nproma,nlev,nblks_e,nproma_tnd)
  external :: kernel_flat
  call kernel_flat(vt, vn_ie, f_e, coeff_gradekin, c_lin_e, ddqz, &
       z_kin_hor_e, z_ekinh, zeta, z_w_con_c_full, &
       icidx, icblk, ividx, ivblk, ddt, ntnd, &
       nproma, nlev, nblks_e, nblks_c, nblks_v, nproma_tnd, isb, ieb, isi, iei)
end subroutine run_ln4_ref
"""


def test_icon_loopnest4_f90_bindings_e2e(tmp_path: Path):
    """The generated ``kernel_flat_dace`` binding for ICON loopnest 4
    (indirect stencil + ``vn_ie(jk)-vn_ie(jk+1)``, 4-D output indexed
    by ``ntnd``), linked against the SDFG ``.so``, must equal a
    plain-gfortran reference."""
    sdfg_lib, ref_lib = _build_binding_and_ref(4, _LN4_IFACE, _LN4_SDFG_DRIVER, _LN4_REF_DRIVER, tmp_path)

    nproma, nlev, nblks_e, nblks_c, nblks_v, nproma_tnd = 32, 16, 8, 8, 8, 3
    ntnd = 2
    isb, ieb, isi, iei = 1, nblks_e, 1, nproma
    rng = np.random.default_rng(4)

    def _f(shape):
        return np.asfortranarray(rng.random(shape, dtype=np.float64))

    def _idx(shape, hi):
        return np.asfortranarray(rng.integers(1, hi + 1, size=shape, dtype=np.int32))

    vt = _f((nproma, nlev, nblks_e))
    vn_ie = _f((nproma, nlev + 1, nblks_e))
    f_e = _f((nproma, nblks_e))
    coeff_gradekin = _f((nproma, 2, nblks_e))
    c_lin_e = _f((nproma, 2, nblks_e))
    ddqz = _f((nproma, nlev, nblks_e))
    z_kin_hor_e = _f((nproma, nlev, nblks_e))
    z_ekinh = _f((nproma, nlev, nblks_c))
    zeta = _f((nproma, nlev, nblks_v))
    z_w_con_c_full = _f((nproma, nlev, nblks_c))
    icidx = _idx((nproma, nblks_e, 2), nproma)
    icblk = _idx((nproma, nblks_e, 2), nblks_c)
    ividx = _idx((nproma, nblks_e, 2), nproma)
    ivblk = _idx((nproma, nblks_e, 2), nblks_v)

    dp = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    ip = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    def _call(lib, fn):
        f = getattr(lib, fn)
        f.restype = None
        f.argtypes = ([ctypes.POINTER(ctypes.c_double)] * 10 + [ctypes.POINTER(ctypes.c_int)] * 4 +
                      [ctypes.POINTER(ctypes.c_double)] + [ctypes.c_int] * 11)
        ddt = np.zeros((nproma, nlev, nblks_e, nproma_tnd), dtype=np.float64, order="F")
        f(dp(vt), dp(vn_ie), dp(f_e), dp(coeff_gradekin), dp(c_lin_e), dp(ddqz), dp(z_kin_hor_e), dp(z_ekinh), dp(zeta),
          dp(z_w_con_c_full), ip(icidx), ip(icblk), ip(ividx), ip(ivblk), dp(ddt), ntnd, nproma, nlev, nblks_e, nblks_c,
          nblks_v, nproma_tnd, isb, ieb, isi, iei)
        return ddt

    ddt_sdfg = _call(sdfg_lib, "run_ln4")
    ddt_ref = _call(ref_lib, "run_ln4_ref")
    assert ddt_ref.any(), "reference produced all zeros"
    np.testing.assert_allclose(ddt_sdfg, ddt_ref, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Loopnest 5  --  horizontal boundary writes, literal level indices
# ---------------------------------------------------------------------------

_LN5_IFACE = OriginalInterface(
    entry="kernel_flat",
    args=(
        _arg_real("vn", 3, ("nproma", "nlev", "nblks_e"), "in"),
        _arg_real("vt", 3, ("nproma", "nlev", "nblks_e"), "in"),
        _arg_real("wgtfacq_e", 3, ("nproma", "3", "nblks_e"), "in"),
        _arg_real("vn_ie", 3, ("nproma", "nlevp1", "nblks_e"), "inout"),
        _arg_real("z_vt_ie", 3, ("nproma", "nlevp1", "nblks_e"), "inout"),
        _arg_real("z_kin_hor_e", 3, ("nproma", "nlevp1", "nblks_e"), "inout"),
        _arg_int("nproma"),
        _arg_int("nlev"),
        _arg_int("nlevp1"),
        _arg_int("nblks_e"),
        _arg_int("i_startblk"),
        _arg_int("i_endblk"),
        _arg_int("i_startidx"),
        _arg_int("i_endidx"),
    ),
    struct_types={},
    used_modules={},
)

_LN5_SDFG_DRIVER = """
subroutine run_ln5(vn, vt, wgtfacq_e, vn_ie, z_vt_ie, z_k, &
    nproma, nlev, nlevp1, nblks_e, isb, ieb, isi, iei) &
    bind(c, name='run_ln5')
  use iso_c_binding
  use @ENTRY@_dace_bindings
  implicit none
  integer(c_int), value :: nproma, nlev, nlevp1, nblks_e, isb, ieb, isi, iei
  real(c_double), intent(in)    :: vn(nproma,nlev,nblks_e), vt(nproma,nlev,nblks_e)
  real(c_double), intent(in)    :: wgtfacq_e(nproma,3,nblks_e)
  real(c_double), intent(inout) :: vn_ie(nproma,nlevp1,nblks_e)
  real(c_double), intent(inout) :: z_vt_ie(nproma,nlevp1,nblks_e), z_k(nproma,nlevp1,nblks_e)
  call @ENTRY@_dace(vn, vt, wgtfacq_e, vn_ie, z_vt_ie, z_k, &
       nproma, nlev, nlevp1, nblks_e, isb, ieb, isi, iei)
  call @ENTRY@_dace_finalize()
end subroutine run_ln5
"""

_LN5_REF_DRIVER = """
subroutine run_ln5_ref(vn, vt, wgtfacq_e, vn_ie, z_vt_ie, z_k, &
    nproma, nlev, nlevp1, nblks_e, isb, ieb, isi, iei) &
    bind(c, name='run_ln5_ref')
  use iso_c_binding
  implicit none
  integer(c_int), value :: nproma, nlev, nlevp1, nblks_e, isb, ieb, isi, iei
  real(c_double), intent(in)    :: vn(nproma,nlev,nblks_e), vt(nproma,nlev,nblks_e)
  real(c_double), intent(in)    :: wgtfacq_e(nproma,3,nblks_e)
  real(c_double), intent(inout) :: vn_ie(nproma,nlevp1,nblks_e)
  real(c_double), intent(inout) :: z_vt_ie(nproma,nlevp1,nblks_e), z_k(nproma,nlevp1,nblks_e)
  external :: kernel_flat
  call kernel_flat(vn, vt, wgtfacq_e, vn_ie, z_vt_ie, z_k, &
       nproma, nlev, nlevp1, nblks_e, isb, ieb, isi, iei)
end subroutine run_ln5_ref
"""


def test_icon_loopnest5_f90_bindings_e2e(tmp_path: Path):
    """The generated ``kernel_flat_dace`` binding for ICON loopnest 5
    (horizontal boundary writes with literal level indices and
    ``nlev-1``/``nlev-2`` loop-bound arithmetic), linked against the
    SDFG ``.so``, must equal a plain-gfortran reference for all three
    output arrays."""
    sdfg_lib, ref_lib = _build_binding_and_ref(5, _LN5_IFACE, _LN5_SDFG_DRIVER, _LN5_REF_DRIVER, tmp_path)

    nproma, nlev, nblks_e = 32, 16, 8
    nlevp1 = nlev + 1
    isb, ieb, isi, iei = 1, nblks_e, 1, nproma
    rng = np.random.default_rng(5)

    vn = np.asfortranarray(rng.random((nproma, nlev, nblks_e)))
    vt = np.asfortranarray(rng.random((nproma, nlev, nblks_e)))
    wgtfacqe = np.asfortranarray(rng.random((nproma, 3, nblks_e)))

    dp = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    def _call(lib, fn):
        f = getattr(lib, fn)
        f.restype = None
        f.argtypes = [ctypes.POINTER(ctypes.c_double)] * 6 + [ctypes.c_int] * 8
        vn_ie = np.zeros((nproma, nlevp1, nblks_e), dtype=np.float64, order="F")
        z_vt = np.zeros((nproma, nlevp1, nblks_e), dtype=np.float64, order="F")
        z_k = np.zeros((nproma, nlevp1, nblks_e), dtype=np.float64, order="F")
        f(dp(vn), dp(vt), dp(wgtfacqe), dp(vn_ie), dp(z_vt), dp(z_k), nproma, nlev, nlevp1, nblks_e, isb, ieb, isi, iei)
        return vn_ie, z_vt, z_k

    vn_ie_s, z_vt_s, z_k_s = _call(sdfg_lib, "run_ln5")
    vn_ie_r, z_vt_r, z_k_r = _call(ref_lib, "run_ln5_ref")
    assert vn_ie_r.any(), "reference produced all zeros"
    np.testing.assert_allclose(vn_ie_s, vn_ie_r, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(z_vt_s, z_vt_r, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(z_k_s, z_k_r, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Loopnest 6  --  LOGICAL ANY reduction (logical(c_bool) binding bridge)
# ---------------------------------------------------------------------------

_LN6_IFACE = OriginalInterface(
    entry="kernel_flat",
    args=(
        _arg_logical("levmask", 2, ("nblks_c", "nlev"), "in"),
        _arg_logical("levelmask", 1, ("nlev", ), "inout"),
        _arg_int("nlev"),
        _arg_int("nblks_c"),
        _arg_int("jk_start"),
        _arg_int("jk_end"),
        _arg_int("i_startblk"),
        _arg_int("i_endblk"),
    ),
    struct_types={},
    used_modules={},
)

# Default Fortran ``logical`` is 4-byte for gfortran; both driver paths
# stay in default logical so the c_int buffers compare bit-for-bit.
_LN6_SDFG_DRIVER = """
subroutine run_ln6(levmask, levelmask, nlev, nblks_c, jks, jke, isb, ieb) &
    bind(c, name='run_ln6')
  use iso_c_binding
  use @ENTRY@_dace_bindings
  implicit none
  integer(c_int), value :: nlev, nblks_c, jks, jke, isb, ieb
  logical, intent(in)    :: levmask(nblks_c, nlev)
  logical, intent(inout) :: levelmask(nlev)
  call @ENTRY@_dace(levmask, levelmask, nlev, nblks_c, jks, jke, isb, ieb)
  call @ENTRY@_dace_finalize()
end subroutine run_ln6
"""

_LN6_REF_DRIVER = """
subroutine run_ln6_ref(levmask, levelmask, nlev, nblks_c, jks, jke, isb, ieb) &
    bind(c, name='run_ln6_ref')
  use iso_c_binding
  implicit none
  integer(c_int), value :: nlev, nblks_c, jks, jke, isb, ieb
  logical, intent(in)    :: levmask(nblks_c, nlev)
  logical, intent(inout) :: levelmask(nlev)
  external :: kernel_flat
  call kernel_flat(levmask, levelmask, nlev, nblks_c, jks, jke, isb, ieb)
end subroutine run_ln6_ref
"""


def test_icon_loopnest6_f90_bindings_e2e(tmp_path: Path):
    """The generated ``kernel_flat_dace`` binding for ICON loopnest 6
    (``levelmask(jk) = ANY(levmask(:, jk))``), linked against the SDFG
    ``.so``, must equal a plain-gfortran reference.  Exercises the
    binding's ``logical(c_bool)`` scratch bridge end-to-end."""
    sdfg_lib, ref_lib = _build_binding_and_ref(6, _LN6_IFACE, _LN6_SDFG_DRIVER, _LN6_REF_DRIVER, tmp_path)

    nlev, nblks_c = 64, 12
    jks, jke = 3, nlev - 3
    isb, ieb = 2, 10
    rng = np.random.default_rng(6)

    # gfortran default LOGICAL is 4-byte; .true. == 1.  Build the mask
    # as int32 so the same truth values feed both call paths and the
    # output buffers compare exactly.
    levmask = np.asfortranarray((rng.random((nblks_c, nlev)) > 0.7).astype(np.int32))

    ip = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    def _call(lib, fn):
        f = getattr(lib, fn)
        f.restype = None
        f.argtypes = [ctypes.POINTER(ctypes.c_int)] * 2 + [ctypes.c_int] * 6
        levelmask = np.zeros(nlev, dtype=np.int32)
        f(ip(levmask), ip(levelmask), nlev, nblks_c, jks, jke, isb, ieb)
        return levelmask

    lm_sdfg = _call(sdfg_lib, "run_ln6")
    lm_ref = _call(ref_lib, "run_ln6_ref")
    assert lm_ref.any(), "reference produced an all-false mask"
    np.testing.assert_array_equal(lm_sdfg.astype(bool), lm_ref.astype(bool))
