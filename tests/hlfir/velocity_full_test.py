"""End-to-end numerical test for the full ICON velocity_tendencies kernel.

Driver: ``velocity_full.f90`` -- 655-line standalone variant of
``mo_velocity_advection.velocity_tendencies``.

Reference path: ``velocity_full_caller.f90`` -- two ``bind(c)`` Fortran
wrapper subs compiled into a shared library via plain ``gfortran
-shared -fPIC`` and loaded through ``ctypes``.  Bypasses f2py entirely
(no crackfortran derived-type binding, no INTENT(OUT)-becomes-return
quirks, no LOGICAL ABI surprises).

  * ``init_inputs_random_c(seed, dims..., *arrays)`` fills every flat
    buffer with deterministic random data (real arrays via
    ``RANDOM_NUMBER`` seeded by ``seed``; index/block tables cyclic
    in-bounds).
  * ``run_velocity_flat_c(dims..., scalars..., *arrays)`` wraps the
    flat buffers back into derived-type dummies (ALLOCATE+memcpy for
    ALLOCATABLE inputs; ``=>``-pointer-assign for POINTER buffers)
    and forwards to ``velocity_tendencies``.

Test flow: build SDFG, compile reference .so, allocate flat buffers,
``init_inputs_random_c`` once, snapshot for SDFG, ``run_velocity_flat_c``
on the reference copy, run SDFG on the snapshot, compare outputs.
"""
from __future__ import annotations

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_HERE = Path(__file__).resolve().parent
_DRIVER_PATH = _HERE / "velocity_full.f90"
_CALLER_PATH = _HERE / "velocity_full_caller.f90"


def _compile_caller_so(out_dir: Path) -> ctypes.CDLL:
    """gfortran -shared -fPIC compile driver + caller into a .so."""
    if shutil.which("gfortran") is None:
        pytest.skip("gfortran not available")
    out_dir.mkdir(parents=True, exist_ok=True)
    so_path = out_dir / "libvelocity_caller.so"
    subprocess.check_call([
        "gfortran",
        "-shared",
        "-fPIC",
        "-O0",
        "-fno-fast-math",
        "-ffp-contract=off",
        "-ffree-line-length-none",
        str(_DRIVER_PATH),
        str(_CALLER_PATH),
        "-o",
        str(so_path),
    ])
    return ctypes.CDLL(str(so_path))


# ----- ctypes shape helpers -------------------------------------------------

_F64 = ctypes.POINTER(ctypes.c_double)
_I32 = ctypes.POINTER(ctypes.c_int)
_I8 = ctypes.POINTER(ctypes.c_int8)


def _ptr(arr: np.ndarray):
    """numpy array -> typed pointer for the array's dtype."""
    if arr.dtype == np.float64:
        return arr.ctypes.data_as(_F64)
    if arr.dtype == np.int32:
        return arr.ctypes.data_as(_I32)
    if arr.dtype == np.int8:
        return arr.ctypes.data_as(_I8)
    raise TypeError(f"unsupported dtype {arr.dtype}")


# ----- buffer layout --------------------------------------------------------


def _allocate(nproma, nlev, nlevp1, nblks_c, nblks_e, nblks_v):
    F = lambda *s: np.zeros(s, dtype=np.float64, order='F')
    I = lambda *s: np.zeros(s, dtype=np.int32, order='F')
    B = lambda *s: np.zeros(s, dtype=np.int8, order='F')
    return dict(
        p_patch_cells_area=F(nproma, nblks_c),
        p_patch_cells_neighbor_idx=I(nproma, nblks_c, 3),
        p_patch_cells_neighbor_blk=I(nproma, nblks_c, 3),
        p_patch_cells_edge_idx=I(nproma, nblks_c, 3),
        p_patch_cells_edge_blk=I(nproma, nblks_c, 3),
        p_patch_cells_start_index=I(33),
        p_patch_cells_end_index=I(33),
        p_patch_cells_start_block=I(33),
        p_patch_cells_end_block=I(33),
        p_patch_cells_decomp_info_owner_mask=B(nproma, nblks_c),
        p_patch_edges_cell_idx=I(nproma, nblks_e, 2),
        p_patch_edges_cell_blk=I(nproma, nblks_e, 2),
        p_patch_edges_vertex_idx=I(nproma, nblks_e, 4),
        p_patch_edges_vertex_blk=I(nproma, nblks_e, 4),
        p_patch_edges_quad_idx=I(nproma, nblks_e, 4),
        p_patch_edges_quad_blk=I(nproma, nblks_e, 4),
        p_patch_edges_tangent_orientation=F(nproma, nblks_e),
        p_patch_edges_inv_primal_edge_length=F(nproma, nblks_e),
        p_patch_edges_inv_dual_edge_length=F(nproma, nblks_e),
        p_patch_edges_area_edge=F(nproma, nblks_e),
        p_patch_edges_f_e=F(nproma, nblks_e),
        p_patch_edges_fn_e=F(nproma, nblks_e),
        p_patch_edges_ft_e=F(nproma, nblks_e),
        p_patch_edges_start_index=I(33),
        p_patch_edges_end_index=I(33),
        p_patch_edges_start_block=I(33),
        p_patch_edges_end_block=I(33),
        p_patch_verts_cell_idx=I(nproma, nblks_v, 6),
        p_patch_verts_cell_blk=I(nproma, nblks_v, 6),
        p_patch_verts_edge_idx=I(nproma, nblks_v, 6),
        p_patch_verts_edge_blk=I(nproma, nblks_v, 6),
        p_patch_verts_start_index=I(33),
        p_patch_verts_end_index=I(33),
        p_patch_verts_start_block=I(33),
        p_patch_verts_end_block=I(33),
        p_int_c_lin_e=F(nproma, 2, nblks_e),
        p_int_e_bln_c_s=F(nproma, 3, nblks_c),
        p_int_cells_aw_verts=F(nproma, 6, nblks_v),
        p_int_rbf_vec_coeff_e=F(4, nproma, nblks_e),
        p_int_geofac_grdiv=F(nproma, 5, nblks_e),
        p_int_geofac_rot=F(nproma, 6, nblks_v),
        p_int_geofac_n2s=F(nproma, 4, nblks_c),
        p_prog_w=F(nproma, nlevp1, nblks_c),
        p_prog_vn=F(nproma, nlev, nblks_e),
        p_diag_vn_ie_ubc=F(nproma, 2, nblks_e),
        p_diag_vt=F(nproma, nlev, nblks_e),
        p_diag_vn_ie=F(nproma, nlevp1, nblks_e),
        p_diag_w_concorr_c=F(nproma, nlev, nblks_c),
        p_diag_ddt_vn_apc_pc=F(nproma, nlev, nblks_e, 3),
        p_diag_ddt_vn_cor_pc=F(nproma, nlev, nblks_e, 3),
        p_diag_ddt_w_adv_pc=F(nproma, nlevp1, nblks_c, 3),
        p_metrics_ddxn_z_full=F(nproma, nlev, nblks_e),
        p_metrics_ddxt_z_full=F(nproma, nlev, nblks_e),
        p_metrics_ddqz_z_full_e=F(nproma, nlev, nblks_e),
        p_metrics_ddqz_z_half=F(nproma, nlevp1, nblks_c),
        p_metrics_wgtfac_c=F(nproma, nlevp1, nblks_c),
        p_metrics_wgtfac_e=F(nproma, nlevp1, nblks_e),
        p_metrics_wgtfacq_e=F(nproma, 3, nblks_e),
        p_metrics_coeff_gradekin=F(nproma, 2, nblks_e),
        p_metrics_coeff1_dwdz=F(nproma, nlev, nblks_c),
        p_metrics_coeff2_dwdz=F(nproma, nlev, nblks_c),
        p_metrics_deepatmo_gradh_mc=F(nlev),
        p_metrics_deepatmo_invr_mc=F(nlev),
        p_metrics_deepatmo_gradh_ifc=F(nlevp1),
        p_metrics_deepatmo_invr_ifc=F(nlevp1),
    )


# Caller subroutine dummy-argument order -- matches velocity_full_caller.f90.
_INIT_ARRAY_ORDER = (
    'p_patch_cells_area',
    'p_patch_cells_neighbor_idx',
    'p_patch_cells_neighbor_blk',
    'p_patch_cells_edge_idx',
    'p_patch_cells_edge_blk',
    'p_patch_cells_start_index',
    'p_patch_cells_end_index',
    'p_patch_cells_start_block',
    'p_patch_cells_end_block',
    'p_patch_cells_decomp_info_owner_mask',
    'p_patch_edges_cell_idx',
    'p_patch_edges_cell_blk',
    'p_patch_edges_vertex_idx',
    'p_patch_edges_vertex_blk',
    'p_patch_edges_quad_idx',
    'p_patch_edges_quad_blk',
    'p_patch_edges_tangent_orientation',
    'p_patch_edges_inv_primal_edge_length',
    'p_patch_edges_inv_dual_edge_length',
    'p_patch_edges_area_edge',
    'p_patch_edges_f_e',
    'p_patch_edges_fn_e',
    'p_patch_edges_ft_e',
    'p_patch_edges_start_index',
    'p_patch_edges_end_index',
    'p_patch_edges_start_block',
    'p_patch_edges_end_block',
    'p_patch_verts_cell_idx',
    'p_patch_verts_cell_blk',
    'p_patch_verts_edge_idx',
    'p_patch_verts_edge_blk',
    'p_patch_verts_start_index',
    'p_patch_verts_end_index',
    'p_patch_verts_start_block',
    'p_patch_verts_end_block',
    'p_int_c_lin_e',
    'p_int_e_bln_c_s',
    'p_int_cells_aw_verts',
    'p_int_rbf_vec_coeff_e',
    'p_int_geofac_grdiv',
    'p_int_geofac_rot',
    'p_int_geofac_n2s',
    'p_prog_w',
    'p_prog_vn',
    'p_diag_vn_ie_ubc',
    'p_diag_vt',
    'p_diag_vn_ie',
    'p_diag_w_concorr_c',
    'p_diag_ddt_vn_apc_pc',
    'p_diag_ddt_vn_cor_pc',
    'p_diag_ddt_w_adv_pc',
    'p_metrics_ddxn_z_full',
    'p_metrics_ddxt_z_full',
    'p_metrics_ddqz_z_full_e',
    'p_metrics_ddqz_z_half',
    'p_metrics_wgtfac_c',
    'p_metrics_wgtfac_e',
    'p_metrics_wgtfacq_e',
    'p_metrics_coeff_gradekin',
    'p_metrics_coeff1_dwdz',
    'p_metrics_coeff2_dwdz',
    'p_metrics_deepatmo_gradh_mc',
    'p_metrics_deepatmo_invr_mc',
    'p_metrics_deepatmo_gradh_ifc',
    'p_metrics_deepatmo_invr_ifc',
)

_OUTPUT_NAMES = (
    'p_prog_w',
    'p_prog_vn',
    'p_diag_vn_ie_ubc',
    'p_diag_vt',
    'p_diag_vn_ie',
    'p_diag_w_concorr_c',
    'p_diag_ddt_vn_apc_pc',
    'p_diag_ddt_vn_cor_pc',
    'p_diag_ddt_w_adv_pc',
    'z_w_concorr_me',
    'z_kin_hor_e',
    'z_vt_ie',
)


@pytest.mark.xfail(strict=False, reason="full velocity_tendencies e2e -- iterating on remaining bridge gaps")
def test_velocity_full_numerical(tmp_path: Path):
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(
        _DRIVER_PATH.read_text(),
        sdfg_dir,
        name="velocity_full",
        entry="_QMmo_velocity_advectionPvelocity_tendencies",
    ).build()
    sdfg.validate()

    lib = _compile_caller_so(tmp_path / "ref")

    nproma, nlev, nblks_c, nblks_e, nblks_v = 8, 6, 4, 4, 4
    nlevp1 = nlev + 1
    seed = 42

    bufs_ref = _allocate(nproma, nlev, nlevp1, nblks_c, nblks_e, nblks_v)

    # init_inputs_random_c: (seed, nproma, nlev, nlevp1, nblks_c, nblks_e, nblks_v, *arrays)
    init = lib.init_inputs_random_c
    init.restype = None
    init.argtypes = [ctypes.c_int] * 7 + [ctypes.c_void_p] * len(_INIT_ARRAY_ORDER)
    init(
        ctypes.c_int(seed),
        ctypes.c_int(nproma),
        ctypes.c_int(nlev),
        ctypes.c_int(nlevp1),
        ctypes.c_int(nblks_c),
        ctypes.c_int(nblks_e),
        ctypes.c_int(nblks_v),
        *[bufs_ref[k].ctypes.data for k in _INIT_ARRAY_ORDER],
    )

    # Snapshot for SDFG.
    bufs_sdfg = {k: v.copy(order='F') for k, v in bufs_ref.items()}

    # Naked array buffers.
    z_w_concorr_me_ref = np.zeros((nproma, nlev, nblks_e), dtype=np.float64, order='F')
    z_kin_hor_e_ref = np.zeros((nproma, nlev, nblks_e), dtype=np.float64, order='F')
    z_vt_ie_ref = np.zeros((nproma, nlevp1, nblks_e), dtype=np.float64, order='F')
    z_w_concorr_me_sdfg = z_w_concorr_me_ref.copy(order='F')
    z_kin_hor_e_sdfg = z_kin_hor_e_ref.copy(order='F')
    z_vt_ie_sdfg = z_vt_ie_ref.copy(order='F')

    nrdmax_in = np.full(10, nlev, dtype=np.int32, order='F')
    nflatlev_in = np.ones(10, dtype=np.int32, order='F')

    # run_velocity_flat_c signature: dims (6 ints), scalars (ntnd, istep,
    # lvn_only, ldeepatmo, dtime, dt_linintp_ubc), module-data (nrdmax_in
    # ptr, nflatlev_in ptr, lvert_nest_in, lextra_diffu_in, timers_level),
    # arrays in declaration order, z_* arrays.
    run = lib.run_velocity_flat_c
    run.restype = None
    run.argtypes = ([ctypes.c_int] * 6  # dims
                    + [ctypes.c_int, ctypes.c_int]  # ntnd, istep
                    + [ctypes.c_int8, ctypes.c_int8]  # lvn_only, ldeepatmo
                    + [ctypes.c_double, ctypes.c_double]  # dtime, dt_linintp_ubc
                    + [ctypes.c_void_p, ctypes.c_void_p]  # nrdmax_in, nflatlev_in
                    + [ctypes.c_int8, ctypes.c_int8, ctypes.c_int]  # lvert_nest_in, lextra_diffu_in, timers_level_in
                    + [ctypes.c_void_p] * (len(_INIT_ARRAY_ORDER) + 3)  # arrays + z_*
                    )
    run(
        nproma,
        nlev,
        nlevp1,
        nblks_c,
        nblks_e,
        nblks_v,
        1,
        1,
        0,
        0,
        60.0,
        0.0,
        nrdmax_in.ctypes.data,
        nflatlev_in.ctypes.data,
        0,
        0,
        0,
        *[bufs_ref[k].ctypes.data for k in _INIT_ARRAY_ORDER],
        z_w_concorr_me_ref.ctypes.data,
        z_kin_hor_e_ref.ctypes.data,
        z_vt_ie_ref.ctypes.data,
    )

    # SDFG side.  Bridge classification mixes Scalar / length-1 Array
    # per-name depending on the kernel's read/write pattern -- use the
    # arglist to pick the right shape per kwarg.
    arglist = sdfg.arglist()

    def scalar_or_arr(name, value, dtype):
        desc = arglist.get(name)
        if desc is not None and type(desc).__name__ == 'Array':
            return np.asfortranarray(np.array([value], dtype=dtype))
        return dtype(value) if isinstance(dtype, type) else np.array(value, dtype=dtype).item()

    sdfg_kw = dict(bufs_sdfg)
    sdfg_kw.update(
        z_w_concorr_me=z_w_concorr_me_sdfg,
        z_kin_hor_e=z_kin_hor_e_sdfg,
        z_vt_ie=z_vt_ie_sdfg,
        ntnd=scalar_or_arr('ntnd', 1, np.int32),
        istep=scalar_or_arr('istep', 1, np.int32),
        lvn_only=scalar_or_arr('lvn_only', False, np.bool_),
        dtime=scalar_or_arr('dtime', 60.0, np.float64),
        dt_linintp_ubc=scalar_or_arr('dt_linintp_ubc', 0.0, np.float64),
        ldeepatmo=scalar_or_arr('ldeepatmo', False, np.bool_),
        nrdmax=nrdmax_in,
        nflatlev=nflatlev_in,
        lvert_nest=scalar_or_arr('lvert_nest', False, np.bool_),
        lextra_diffu=scalar_or_arr('lextra_diffu', False, np.bool_),
        i_am_accel_node=scalar_or_arr('i_am_accel_node', False, np.bool_),
        timers_level=scalar_or_arr('timers_level', 0, np.int32),
        timer_intp=scalar_or_arr('timer_intp', 0, np.int32),
        timer_solve_nh_veltend=scalar_or_arr('timer_solve_nh_veltend', 0, np.int32),
        p_patch_id=scalar_or_arr('p_patch_id', 1, np.int32),
        p_patch_nblks_c=scalar_or_arr('p_patch_nblks_c', nblks_c, np.int32),
        p_patch_nblks_e=scalar_or_arr('p_patch_nblks_e', nblks_e, np.int32),
        p_patch_nblks_v=scalar_or_arr('p_patch_nblks_v', nblks_v, np.int32),
        p_patch_nlev=scalar_or_arr('p_patch_nlev', nlev, np.int32),
        p_patch_nlevp1=scalar_or_arr('p_patch_nlevp1', nlevp1, np.int32),
        p_patch_nshift=scalar_or_arr('p_patch_nshift', 0, np.int32),
        p_diag_ddt_vn_adv_is_associated=scalar_or_arr('p_diag_ddt_vn_adv_is_associated', False, np.bool_),
        p_diag_ddt_vn_cor_is_associated=scalar_or_arr('p_diag_ddt_vn_cor_is_associated', False, np.bool_),
        p_diag_max_vcfl_dyn=scalar_or_arr('p_diag_max_vcfl_dyn', 0.0, np.float64),
        nproma=scalar_or_arr('nproma', nproma, np.int32),
    )
    # Deferred-shape pointer/allocatable companions need ``<arr>_d<i>``
    # extent bound symbols for every dim that the bridge couldn't resolve
    # symbolically.  Use the arglist to know which ones to provide.
    for nm, arr in list(sdfg_kw.items()):
        if not hasattr(arr, 'shape'):
            continue
        for d in range(len(arr.shape)):
            sym = f'{nm}_d{d}'
            if sym in arglist:
                sdfg_kw.setdefault(sym, np.int64(arr.shape[d]))
    # bridge stores LOGICAL arrays as bool8; ctypes/owner_mask is int8 ->
    # convert to a writable bool view so DaCe's wrapper accepts it.
    sdfg_kw['p_patch_cells_decomp_info_owner_mask'] = \
        sdfg_kw['p_patch_cells_decomp_info_owner_mask'].astype(np.bool_, order='F')
    sdfg(**sdfg_kw)

    mismatches = []
    extras = {
        'z_w_concorr_me': (z_w_concorr_me_sdfg, z_w_concorr_me_ref),
        'z_kin_hor_e': (z_kin_hor_e_sdfg, z_kin_hor_e_ref),
        'z_vt_ie': (z_vt_ie_sdfg, z_vt_ie_ref)
    }
    for nm in _OUTPUT_NAMES:
        if nm in extras:
            sd, rf = extras[nm]
        else:
            sd, rf = bufs_sdfg[nm], bufs_ref[nm]
        if not np.allclose(sd, rf, rtol=1e-10, atol=1e-10, equal_nan=True):
            d = np.abs(sd - rf)
            mismatches.append(f"{nm}: max_abs_diff={d.max():.3e} (n_diff={np.count_nonzero(d > 1e-10)})")
    assert not mismatches, "\n".join(mismatches)
