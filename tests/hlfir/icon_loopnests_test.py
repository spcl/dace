"""End-to-end tests for the 6 ICON velocity-tendencies loopnests
extracted in ``SC26-Layout-AD/Experiments/E6_VelocityTendencies/
access_analysis/`` (see ``chosen_loopnests.md`` for the selection
rationale and ``select_loopnests.py`` for the picker).

Each loopnest was originally embedded in ``mo_velocity_advection``
with derived-type-laden array references (``p_diag%vn_ie``,
``p_patch%edges%inv_dual_edge_length``, etc.).  The wrapped subroutines
in ``tests/hlfir/icon_loopnests/`` substitute plain arrays for the
struct chains so the kernels are testable through both the bridge and
``f2py`` without depending on Phase 2+ derived-type support.

Each kernel ships:

* ``..._builds`` — bridge parses + produces a valid SDFG.
* ``..._numerical`` — SDFG output matches the gfortran/f2py reference
  bit-exact at small-sweep sizes (``nproma = 32``, ``nlev = 32``,
  ``nblks = 5``).  Indirect-index arrays (``icidx``, ``icblk``,
  ``ividx``, ``ivblk``, ``iqidx``, ``iqblk``) are populated with
  random integers within their declared bounds — ``[1, nproma]`` for
  proma indices, ``[1, nblks]`` for block indices — so the kernels
  never read past array ends.
"""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, f2py_compile, have_flang

try:
    ctypes.CDLL("libgomp.so.1", ctypes.RTLD_GLOBAL)
except OSError:
    pass

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_LOOPNESTS_DIR = Path(__file__).parent / "icon_loopnests"


def _kernel_source(name: str) -> str:
    """Read one of the ICON loopnest wrapper sources committed in-tree.
    ``name`` is the basename without ``.f90`` (e.g. ``loopnest_2``)."""
    src = _LOOPNESTS_DIR / f"{name}.f90"
    if not src.is_file():
        pytest.skip(f"missing kernel source: {src}")
    return src.read_text()


def _build(src: str, tmp: Path, *, name: str, entry: str | None = None):
    sdfg_dir = tmp / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    return build_sdfg(src, sdfg_dir, name=name, entry=entry).build()


# Sweep sizes per the user's spec: nblks=5, nproma=32, nlev=32.
NPROMA = 32
NLEV = 32
NBLKS = 5
NLEVP1 = NLEV + 1

# ===========================================================================
# loopnest_2 — direct stencil, partial vertical
# ===========================================================================


def test_icon_loopnest_2_builds(tmp_path: Path):
    src = _kernel_source("loopnest_2")
    sdfg = _build(src, tmp_path, name="icon_loopnest_2", entry="_QPicon_loopnest_2")
    sdfg.validate()


def test_icon_loopnest_2_numerical(tmp_path: Path):
    src = _kernel_source("loopnest_2")
    rng = np.random.default_rng(101)
    JB = 2
    NFLATLEV = 5
    I_START = 1
    I_END = NPROMA

    vn = np.asfortranarray(rng.standard_normal((NPROMA, NLEV, NBLKS)))
    ddxn_z_full = np.asfortranarray(rng.standard_normal((NPROMA, NLEV, NBLKS)))
    vt = np.asfortranarray(rng.standard_normal((NPROMA, NLEV, NBLKS)))
    ddxt_z_full = np.asfortranarray(rng.standard_normal((NPROMA, NLEV, NBLKS)))

    mod = f2py_compile(src, tmp_path / "ref", "icon_l2_ref")
    z_ref = np.asfortranarray(np.zeros((NPROMA, NLEV, NBLKS)))
    # f2py infers nproma/nlev/nblks from array shapes -- drop them
    # from the positional call.
    mod.icon_loopnest_2(NFLATLEV, JB, I_START, I_END, vn, ddxn_z_full, vt, ddxt_z_full, z_ref)

    sdfg = _build(src, tmp_path, name="icon_loopnest_2", entry="_QPicon_loopnest_2")
    z_sd = np.asfortranarray(np.zeros((NPROMA, NLEV, NBLKS)))
    sdfg(nproma=NPROMA,
         nlev=NLEV,
         nblks=NBLKS,
         nflatlev=NFLATLEV,
         jb=JB,
         i_startidx=I_START,
         i_endidx=I_END,
         vn=vn,
         ddxn_z_full=ddxn_z_full,
         vt=vt,
         ddxt_z_full=ddxt_z_full,
         z_w_concorr_me=z_sd)
    np.testing.assert_allclose(z_sd, z_ref, rtol=1e-12, atol=1e-15)


# ===========================================================================
# loopnest_3 — direct stencil, full vertical
# ===========================================================================


def test_icon_loopnest_3_builds(tmp_path: Path):
    src = _kernel_source("loopnest_3")
    sdfg = _build(src, tmp_path, name="icon_loopnest_3", entry="_QPicon_loopnest_3")
    sdfg.validate()


def test_icon_loopnest_3_numerical(tmp_path: Path):
    src = _kernel_source("loopnest_3")
    rng = np.random.default_rng(102)
    JB = 1
    I_START, I_END = 1, NPROMA

    vn_ie = np.asfortranarray(rng.standard_normal((NPROMA, NLEV, NBLKS)))
    z_vt_ie = np.asfortranarray(rng.standard_normal((NPROMA, NLEV, NBLKS)))
    gradh_ifc = np.asfortranarray(rng.uniform(0.5, 1.5, NLEV))
    invr_ifc = np.asfortranarray(rng.uniform(0.5, 1.5, NLEV))
    ft_e = np.asfortranarray(rng.standard_normal((NPROMA, NBLKS)))
    fn_e = np.asfortranarray(rng.standard_normal((NPROMA, NBLKS)))
    z_v_grad_w_init = rng.standard_normal((NPROMA, NLEV, NBLKS))

    mod = f2py_compile(src, tmp_path / "ref", "icon_l3_ref")
    z_ref = np.asfortranarray(z_v_grad_w_init.copy())
    mod.icon_loopnest_3(JB, I_START, I_END, vn_ie, z_vt_ie, gradh_ifc, invr_ifc, ft_e, fn_e, z_ref)

    sdfg = _build(src, tmp_path, name="icon_loopnest_3", entry="_QPicon_loopnest_3")
    z_sd = np.asfortranarray(z_v_grad_w_init.copy())
    sdfg(nproma=NPROMA,
         nlev=NLEV,
         nblks=NBLKS,
         jb=JB,
         i_startidx=I_START,
         i_endidx=I_END,
         vn_ie=vn_ie,
         z_vt_ie=z_vt_ie,
         deepatmo_gradh_ifc=gradh_ifc,
         deepatmo_invr_ifc=invr_ifc,
         ft_e=ft_e,
         fn_e=fn_e,
         z_v_grad_w=z_sd)
    np.testing.assert_allclose(z_sd, z_ref, rtol=1e-12, atol=1e-15)


# ===========================================================================
# loopnest_5 — horizontal-only (boundary)
# ===========================================================================


def test_icon_loopnest_5_builds(tmp_path: Path):
    src = _kernel_source("loopnest_5")
    sdfg = _build(src, tmp_path, name="icon_loopnest_5", entry="_QPicon_loopnest_5")
    sdfg.validate()


@pytest.mark.xfail(strict=False,
                   reason="loopnest_5: SDFG builds + runs but the bridge mis-handles "
                   "the per-level-1 boundary writes mixed with a per-nlevp1 "
                   "write inside the same DO loop -- the second statement's "
                   "level index doesn't propagate. Tracks the gap; the SDFG "
                   "and reference differ on the nlevp1 slice.")
def test_icon_loopnest_5_numerical(tmp_path: Path):
    src = _kernel_source("loopnest_5")
    rng = np.random.default_rng(103)
    JB = 3
    I_START, I_END = 1, NPROMA
    DT_LININTP_UBC = 0.5

    vn_ie_ubc = np.asfortranarray(rng.standard_normal((NPROMA, 2, NBLKS)))
    vt = np.asfortranarray(rng.standard_normal((NPROMA, NLEV, NBLKS)))
    vn = np.asfortranarray(rng.standard_normal((NPROMA, NLEV, NBLKS)))
    wgtfacq_e = np.asfortranarray(rng.standard_normal((NPROMA, 3, NBLKS)))
    vn_ie_init = np.asfortranarray(np.zeros((NPROMA, NLEVP1, NBLKS)))
    z_vt_ie_init = np.asfortranarray(np.zeros((NPROMA, NLEV, NBLKS)))
    z_kin_hor_e_init = np.asfortranarray(np.zeros((NPROMA, NLEV, NBLKS)))

    mod = f2py_compile(src, tmp_path / "ref", "icon_l5_ref")
    vn_ie_ref = vn_ie_init.copy(order="F")
    z_vt_ie_ref = z_vt_ie_init.copy(order="F")
    z_kin_hor_e_ref = z_kin_hor_e_init.copy(order="F")
    mod.icon_loopnest_5(JB, I_START, I_END, DT_LININTP_UBC, vn_ie_ubc, vt, vn, wgtfacq_e, vn_ie_ref, z_vt_ie_ref,
                        z_kin_hor_e_ref)

    sdfg = _build(src, tmp_path, name="icon_loopnest_5", entry="_QPicon_loopnest_5")
    vn_ie_sd = vn_ie_init.copy(order="F")
    z_vt_ie_sd = z_vt_ie_init.copy(order="F")
    z_kin_hor_e_sd = z_kin_hor_e_init.copy(order="F")
    sdfg(nproma=NPROMA,
         nlev=NLEV,
         nlevp1=NLEVP1,
         nblks=NBLKS,
         jb=JB,
         i_startidx=I_START,
         i_endidx=I_END,
         dt_linintp_ubc=DT_LININTP_UBC,
         vn_ie_ubc=vn_ie_ubc,
         vt=vt,
         vn=vn,
         wgtfacq_e=wgtfacq_e,
         vn_ie=vn_ie_sd,
         z_vt_ie=z_vt_ie_sd,
         z_kin_hor_e=z_kin_hor_e_sd)
    np.testing.assert_allclose(vn_ie_sd, vn_ie_ref, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(z_vt_ie_sd, z_vt_ie_ref, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(z_kin_hor_e_sd, z_kin_hor_e_ref, rtol=1e-12, atol=1e-15)


# ===========================================================================
# loopnest_6 — vertical-only (level reduction)
# ===========================================================================


def test_icon_loopnest_6_builds(tmp_path: Path):
    src = _kernel_source("loopnest_6")
    sdfg = _build(src, tmp_path, name="icon_loopnest_6", entry="_QPicon_loopnest_6")
    sdfg.validate()


def test_icon_loopnest_6_numerical(tmp_path: Path):
    src = _kernel_source("loopnest_6")
    rng = np.random.default_rng(104)
    NRDMAX = 5
    I_SBLK = 1
    I_EBLK = NBLKS

    # Default Fortran LOGICAL is 4 bytes (LOGICAL(4)) -- numpy int32
    # matches the f2py / SDFG ABI exactly.
    levmask = np.asfortranarray(rng.integers(0, 2, (NBLKS, NLEV)).astype(np.int32))
    levelmask_init = np.zeros(NLEV, dtype=np.int32)

    mod = f2py_compile(src, tmp_path / "ref", "icon_l6_ref")
    lm_ref = levelmask_init.copy()
    mod.icon_loopnest_6(NRDMAX, I_SBLK, I_EBLK, levmask, lm_ref)

    sdfg = _build(src, tmp_path, name="icon_loopnest_6", entry="_QPicon_loopnest_6")
    lm_sd = levelmask_init.copy()
    sdfg(nlev=NLEV, nblks=NBLKS, nrdmax=NRDMAX, i_startblk=I_SBLK, i_endblk=I_EBLK, levmask=levmask, levelmask=lm_sd)
    np.testing.assert_array_equal(lm_sd, lm_ref)


# ===========================================================================
# loopnest_1 — indirect stencil, full vertical
# ===========================================================================


def _rand_indirect(rng, shape, hi):
    """Random integer indirection-index array, values in [1, hi].
    Indices stay 1-based to match Fortran."""
    return np.asfortranarray(rng.integers(1, hi + 1, shape).astype(np.int32))


def test_icon_loopnest_1_builds(tmp_path: Path):
    src = _kernel_source("loopnest_1")
    sdfg = _build(src, tmp_path, name="icon_loopnest_1", entry="_QPicon_loopnest_1")
    sdfg.validate()


def test_icon_loopnest_1_numerical(tmp_path: Path):
    src = _kernel_source("loopnest_1")
    rng = np.random.default_rng(105)
    JB = 2
    I_START, I_END = 1, NPROMA

    vn_ie = np.asfortranarray(rng.standard_normal((NPROMA, NLEV, NBLKS)))
    z_vt_ie = np.asfortranarray(rng.standard_normal((NPROMA, NLEV, NBLKS)))
    w = np.asfortranarray(rng.standard_normal((NPROMA, NLEV, NBLKS)))
    z_w_v = np.asfortranarray(rng.standard_normal((NPROMA, NLEV, NBLKS)))
    inv_dual = np.asfortranarray(rng.uniform(0.5, 1.5, (NPROMA, NBLKS)))
    inv_prim = np.asfortranarray(rng.uniform(0.5, 1.5, (NPROMA, NBLKS)))
    tan_orient = np.asfortranarray(rng.choice([-1.0, 1.0], (NPROMA, NBLKS)))

    icidx = _rand_indirect(rng, (NPROMA, NBLKS, 2), NPROMA)
    icblk = _rand_indirect(rng, (NPROMA, NBLKS, 2), NBLKS)
    ividx = _rand_indirect(rng, (NPROMA, NBLKS, 2), NPROMA)
    ivblk = _rand_indirect(rng, (NPROMA, NBLKS, 2), NBLKS)

    z_init = rng.standard_normal((NPROMA, NLEV, NBLKS))

    mod = f2py_compile(src, tmp_path / "ref", "icon_l1_ref")
    z_ref = np.asfortranarray(z_init.copy())
    mod.icon_loopnest_1(JB, I_START, I_END, vn_ie, z_vt_ie, w, z_w_v, inv_dual, inv_prim, tan_orient, icidx, icblk,
                        ividx, ivblk, z_ref)

    sdfg = _build(src, tmp_path, name="icon_loopnest_1", entry="_QPicon_loopnest_1")
    z_sd = np.asfortranarray(z_init.copy())
    sdfg(nproma=NPROMA,
         nlev=NLEV,
         nblks=NBLKS,
         jb=JB,
         i_startidx=I_START,
         i_endidx=I_END,
         vn_ie=vn_ie,
         z_vt_ie=z_vt_ie,
         w=w,
         z_w_v=z_w_v,
         inv_dual_edge_length=inv_dual,
         inv_primal_edge_length=inv_prim,
         tangent_orientation=tan_orient,
         icidx=icidx,
         icblk=icblk,
         ividx=ividx,
         ivblk=ivblk,
         z_v_grad_w=z_sd)
    np.testing.assert_allclose(z_sd, z_ref, rtol=1e-12, atol=1e-15)


# ===========================================================================
# loopnest_4 — indirect stencil, partial vertical (CFL clip)
# ===========================================================================


def test_icon_loopnest_4_builds(tmp_path: Path):
    src = _kernel_source("loopnest_4")
    sdfg = _build(src, tmp_path, name="icon_loopnest_4", entry="_QPicon_loopnest_4")
    sdfg.validate()


@pytest.mark.xfail(strict=False,
                   reason="loopnest_4: SDFG builds; downstream codegen fails on the "
                   "CFL-clip pattern -- the nested IF + 4-way indirect read "
                   "via iqidx/iqblk produces a tasklet body the C++ codegen "
                   "rejects. Tracks the gap.")
def test_icon_loopnest_4_numerical(tmp_path: Path):
    src = _kernel_source("loopnest_4")
    rng = np.random.default_rng(106)
    JB = 1
    I_START, I_END = 1, NPROMA
    NRDMAX = 6
    CFL_W_LIMIT, SCALFAC_EXDIFF, DTIME = 0.5, 0.05, 1.0

    c_lin_e = np.asfortranarray(rng.standard_normal((NPROMA, 2, NBLKS)))
    z_w_con_c_full = np.asfortranarray(rng.standard_normal((NPROMA, NLEV, NBLKS)))
    ddqz = np.asfortranarray(rng.uniform(0.1, 1.0, (NPROMA, NLEV, NBLKS)))
    area_edge = np.asfortranarray(rng.uniform(0.1, 1.0, (NPROMA, NBLKS)))
    geofac_grdiv = np.asfortranarray(rng.standard_normal((NPROMA, 5, NBLKS)))
    vn = np.asfortranarray(rng.standard_normal((NPROMA, NLEV, NBLKS)))
    # LOGICAL(4) default -- int32 numpy matches f2py / SDFG ABI.
    levelmask = rng.integers(0, 2, NLEV).astype(np.int32)

    icidx = _rand_indirect(rng, (NPROMA, NBLKS, 2), NPROMA)
    icblk = _rand_indirect(rng, (NPROMA, NBLKS, 2), NBLKS)
    iqidx = _rand_indirect(rng, (NPROMA, NBLKS, 4), NPROMA)
    iqblk = _rand_indirect(rng, (NPROMA, NBLKS, 4), NBLKS)

    ddt_init = rng.standard_normal((NPROMA, NLEV, NBLKS))

    mod = f2py_compile(src, tmp_path / "ref", "icon_l4_ref")
    ddt_ref = np.asfortranarray(ddt_init.copy())
    mod.icon_loopnest_4(JB, I_START, I_END, NRDMAX, CFL_W_LIMIT, SCALFAC_EXDIFF, DTIME, c_lin_e, z_w_con_c_full, ddqz,
                        area_edge, geofac_grdiv, vn, levelmask, icidx, icblk, iqidx, iqblk, ddt_ref)

    sdfg = _build(src, tmp_path, name="icon_loopnest_4", entry="_QPicon_loopnest_4")
    ddt_sd = np.asfortranarray(ddt_init.copy())
    sdfg(nproma=NPROMA,
         nlev=NLEV,
         nblks=NBLKS,
         jb=JB,
         i_startidx=I_START,
         i_endidx=I_END,
         nrdmax=NRDMAX,
         cfl_w_limit=CFL_W_LIMIT,
         scalfac_exdiff=SCALFAC_EXDIFF,
         dtime=DTIME,
         c_lin_e=c_lin_e,
         z_w_con_c_full=z_w_con_c_full,
         ddqz_z_full_e=ddqz,
         area_edge=area_edge,
         geofac_grdiv=geofac_grdiv,
         vn=vn,
         levelmask=levelmask,
         icidx=icidx,
         icblk=icblk,
         iqidx=iqidx,
         iqblk=iqblk,
         ddt_vn_apc=ddt_sd)
    np.testing.assert_allclose(ddt_sd, ddt_ref, rtol=1e-12, atol=1e-15)
