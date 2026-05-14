"""End-to-end integration probe for the full ICON velocity-tendencies
kernel.

Source: ``velocity_full.f90`` -- the 655-line standalone variant of
``mo_velocity_advection.velocity_tendencies`` that defines every
upstream module (mo_decomposition_tools, mo_model_domain,
mo_nonhydro_types, mo_intp_data_strc, mo_lib_loopindices,
mo_icon_interpolation_scalar, mo_math_divrot, mo_loopindices,
mo_run_config, ...) in the same file and imports the struct types
inside ``velocity_tendencies`` via USE statements.

Phase 1 (this commit): xfail probe.  The bridge currently fails at
``KeyError: 'p_prog_w_d1'`` while attaching the frozen signature --
the pointer-array struct member ``t_nh_prog%w`` does not get its
deferred-shape bound symbols minted.  Marked xfail(strict=False) so
the failure surfaces in the sweep without blocking; each subsequent
commit closes one bridge gap.

All indirection arrays (icidx/icblk/ividx/ivblk/ieidx/ieblk plus
quad_idx/quad_blk on edges, edge_idx/edge_blk on cells & verts) are
generated strictly in-bounds via ``rng.integers(1, nproma+1, ...)``
and ``rng.integers(1, nblks+1, ...)`` so every gather stays inside
the allocated extents on either backend.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_HERE = Path(__file__).resolve().parent
_SRC_PATH = _HERE / "velocity_full.f90"


@pytest.mark.xfail(strict=False, reason="full velocity_tendencies integration probe -- bridge gaps surface here")
def test_velocity_full_numerical(tmp_path: Path):
    src = _SRC_PATH.read_text()

    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    # velocity_tendencies lives inside MODULE mo_velocity_advection, so the
    # mangled entry is _QM<module>P<subroutine>.
    sdfg = build_sdfg(
        src,
        sdfg_dir,
        name="velocity_full",
        entry="_QMmo_velocity_advectionPvelocity_tendencies",
    ).build()
    sdfg.validate()

    nproma, nlev, nblks = 32, 6, 32
    nlevp1 = nlev + 1
    rng = np.random.default_rng(0)

    # Indirection tables: every value strictly in [1, nproma] (for indices)
    # or [1, nblks] (for block ids) so every gather is in-bounds on either
    # backend.  rng.integers high is exclusive -> pass nproma+1 / nblks+1.
    def ii(*shape):
        return np.asfortranarray(rng.integers(1, nproma + 1, size=shape, dtype=np.int32))

    def bi(*shape):
        return np.asfortranarray(rng.integers(1, nblks + 1, size=shape, dtype=np.int32))

    def rr(*shape):
        return np.asfortranarray(rng.standard_normal(shape))

    # ---- p_patch flattened components ---------------------------------
    p_patch_id = np.int32(1)
    p_patch_nblks_c = np.int32(nblks)
    p_patch_nblks_e = np.int32(nblks)
    p_patch_nblks_v = np.int32(nblks)
    p_patch_nlev = np.int32(nlev)
    p_patch_nlevp1 = np.int32(nlevp1)
    p_patch_nshift = np.int32(0)
    p_patch_cells_neighbor_idx = ii(nproma, nblks, 3)
    p_patch_cells_neighbor_blk = bi(nproma, nblks, 3)
    p_patch_cells_edge_idx = ii(nproma, nblks, 3)
    p_patch_cells_edge_blk = bi(nproma, nblks, 3)
    p_patch_cells_area = rr(nproma, nblks)
    p_patch_cells_start_index = np.asfortranarray(np.ones(8, dtype=np.int32))
    p_patch_cells_end_index = np.asfortranarray(np.full(8, nproma, dtype=np.int32))
    p_patch_cells_start_block = np.asfortranarray(np.ones(8, dtype=np.int32))
    p_patch_cells_end_block = np.asfortranarray(np.full(8, nblks, dtype=np.int32))
    p_patch_edges_cell_idx = ii(nproma, nblks, 2)
    p_patch_edges_cell_blk = bi(nproma, nblks, 2)
    p_patch_edges_vertex_idx = ii(nproma, nblks, 4)
    p_patch_edges_vertex_blk = bi(nproma, nblks, 4)
    p_patch_edges_quad_idx = ii(nproma, nblks, 4)
    p_patch_edges_quad_blk = bi(nproma, nblks, 4)
    p_patch_edges_tangent_orientation = rr(nproma, nblks)
    p_patch_edges_inv_primal_edge_length = rr(nproma, nblks)
    p_patch_edges_inv_dual_edge_length = rr(nproma, nblks)
    p_patch_edges_area_edge = rr(nproma, nblks)
    p_patch_edges_f_e = rr(nproma, nblks)
    p_patch_edges_fn_e = rr(nproma, nblks)
    p_patch_edges_ft_e = rr(nproma, nblks)
    p_patch_edges_start_index = np.asfortranarray(np.ones(8, dtype=np.int32))
    p_patch_edges_end_index = np.asfortranarray(np.full(8, nproma, dtype=np.int32))
    p_patch_edges_start_block = np.asfortranarray(np.ones(8, dtype=np.int32))
    p_patch_edges_end_block = np.asfortranarray(np.full(8, nblks, dtype=np.int32))
    p_patch_verts_cell_idx = ii(nproma, nblks, 6)
    p_patch_verts_cell_blk = bi(nproma, nblks, 6)
    p_patch_verts_edge_idx = ii(nproma, nblks, 6)
    p_patch_verts_edge_blk = bi(nproma, nblks, 6)

    # ---- p_int (t_int_state) ------------------------------------------
    p_int_c_lin_e = rr(nproma, 2, nblks)
    p_int_e_bln_c_s = rr(nproma, 3, nblks)
    p_int_cells_aw_verts = rr(nproma, 6, nblks)
    p_int_rbf_vec_coeff_e = rr(4, nproma, nblks)
    p_int_geofac_grdiv = rr(nproma, 5, nblks)
    p_int_geofac_rot = rr(nproma, 6, nblks)
    p_int_geofac_n2s = rr(nproma, 4, nblks)

    # ---- p_prog (t_nh_prog) -------------------------------------------
    p_prog_w = rr(nproma, nlevp1, nblks)
    p_prog_vn = rr(nproma, nlev, nblks)

    # ---- p_diag (t_nh_diag) -- INTENT(INOUT) outputs -------------------
    p_diag_vn_ie_ubc = rr(nproma, 2, nblks)
    p_diag_vt = rr(nproma, nlev, nblks)
    p_diag_vn_ie = rr(nproma, nlevp1, nblks)
    p_diag_w_concorr_c = rr(nproma, nlev, nblks)
    p_diag_ddt_vn_apc_pc = rr(nproma, nlev, nblks, 3)
    p_diag_ddt_vn_cor_pc = rr(nproma, nlev, nblks, 3)
    p_diag_ddt_w_adv_pc = rr(nproma, nlevp1, nblks, 3)

    # ---- p_metrics (t_nh_metrics) -------------------------------------
    p_metrics_ddxn_z_full = rr(nproma, nlev, nblks)
    p_metrics_ddxt_z_full = rr(nproma, nlev, nblks)
    p_metrics_ddqz_z_full_e = rr(nproma, nlev, nblks)
    p_metrics_ddqz_z_half = rr(nproma, nlevp1, nblks)
    p_metrics_wgtfac_c = rr(nproma, nlevp1, nblks)
    p_metrics_wgtfac_e = rr(nproma, nlevp1, nblks)
    p_metrics_wgtfacq_e = rr(nproma, 3, nblks)
    p_metrics_coeff_gradekin = rr(2, nproma, nblks)
    p_metrics_coeff1_dwdz = rr(nproma, nlev, nblks)
    p_metrics_coeff2_dwdz = rr(nproma, nlev, nblks)
    p_metrics_deepatmo_gradh_mc = rr(nlev)
    p_metrics_deepatmo_invr_mc = rr(nlev)
    p_metrics_deepatmo_gradh_ifc = rr(nlevp1)
    p_metrics_deepatmo_invr_ifc = rr(nlevp1)

    # ---- naked array args + scalars -----------------------------------
    z_w_concorr_me = rr(nproma, nlev, nblks)
    z_kin_hor_e = rr(nproma, nlev, nblks)
    z_vt_ie = rr(nproma, nlevp1, nblks)

    sdfg(
        # p_patch flat
        p_patch_id=p_patch_id,
        p_patch_nblks_c=p_patch_nblks_c,
        p_patch_nblks_e=p_patch_nblks_e,
        p_patch_nblks_v=p_patch_nblks_v,
        p_patch_nlev=p_patch_nlev,
        p_patch_nlevp1=p_patch_nlevp1,
        p_patch_nshift=p_patch_nshift,
        p_patch_cells_neighbor_idx=p_patch_cells_neighbor_idx,
        p_patch_cells_neighbor_blk=p_patch_cells_neighbor_blk,
        p_patch_cells_edge_idx=p_patch_cells_edge_idx,
        p_patch_cells_edge_blk=p_patch_cells_edge_blk,
        p_patch_cells_area=p_patch_cells_area,
        p_patch_cells_start_index=p_patch_cells_start_index,
        p_patch_cells_end_index=p_patch_cells_end_index,
        p_patch_cells_start_block=p_patch_cells_start_block,
        p_patch_cells_end_block=p_patch_cells_end_block,
        p_patch_edges_cell_idx=p_patch_edges_cell_idx,
        p_patch_edges_cell_blk=p_patch_edges_cell_blk,
        p_patch_edges_vertex_idx=p_patch_edges_vertex_idx,
        p_patch_edges_vertex_blk=p_patch_edges_vertex_blk,
        p_patch_edges_quad_idx=p_patch_edges_quad_idx,
        p_patch_edges_quad_blk=p_patch_edges_quad_blk,
        p_patch_edges_tangent_orientation=p_patch_edges_tangent_orientation,
        p_patch_edges_inv_primal_edge_length=p_patch_edges_inv_primal_edge_length,
        p_patch_edges_inv_dual_edge_length=p_patch_edges_inv_dual_edge_length,
        p_patch_edges_area_edge=p_patch_edges_area_edge,
        p_patch_edges_f_e=p_patch_edges_f_e,
        p_patch_edges_fn_e=p_patch_edges_fn_e,
        p_patch_edges_ft_e=p_patch_edges_ft_e,
        p_patch_edges_start_index=p_patch_edges_start_index,
        p_patch_edges_end_index=p_patch_edges_end_index,
        p_patch_edges_start_block=p_patch_edges_start_block,
        p_patch_edges_end_block=p_patch_edges_end_block,
        p_patch_verts_cell_idx=p_patch_verts_cell_idx,
        p_patch_verts_cell_blk=p_patch_verts_cell_blk,
        p_patch_verts_edge_idx=p_patch_verts_edge_idx,
        p_patch_verts_edge_blk=p_patch_verts_edge_blk,
        # p_int flat
        p_int_c_lin_e=p_int_c_lin_e,
        p_int_e_bln_c_s=p_int_e_bln_c_s,
        p_int_cells_aw_verts=p_int_cells_aw_verts,
        p_int_rbf_vec_coeff_e=p_int_rbf_vec_coeff_e,
        p_int_geofac_grdiv=p_int_geofac_grdiv,
        p_int_geofac_rot=p_int_geofac_rot,
        p_int_geofac_n2s=p_int_geofac_n2s,
        # p_prog flat
        p_prog_w=p_prog_w,
        p_prog_vn=p_prog_vn,
        # p_diag flat
        p_diag_vn_ie_ubc=p_diag_vn_ie_ubc,
        p_diag_vt=p_diag_vt,
        p_diag_vn_ie=p_diag_vn_ie,
        p_diag_w_concorr_c=p_diag_w_concorr_c,
        p_diag_ddt_vn_apc_pc=p_diag_ddt_vn_apc_pc,
        p_diag_ddt_vn_cor_pc=p_diag_ddt_vn_cor_pc,
        p_diag_ddt_w_adv_pc=p_diag_ddt_w_adv_pc,
        # p_metrics flat
        p_metrics_ddxn_z_full=p_metrics_ddxn_z_full,
        p_metrics_ddxt_z_full=p_metrics_ddxt_z_full,
        p_metrics_ddqz_z_full_e=p_metrics_ddqz_z_full_e,
        p_metrics_ddqz_z_half=p_metrics_ddqz_z_half,
        p_metrics_wgtfac_c=p_metrics_wgtfac_c,
        p_metrics_wgtfac_e=p_metrics_wgtfac_e,
        p_metrics_wgtfacq_e=p_metrics_wgtfacq_e,
        p_metrics_coeff_gradekin=p_metrics_coeff_gradekin,
        p_metrics_coeff1_dwdz=p_metrics_coeff1_dwdz,
        p_metrics_coeff2_dwdz=p_metrics_coeff2_dwdz,
        p_metrics_deepatmo_gradh_mc=p_metrics_deepatmo_gradh_mc,
        p_metrics_deepatmo_invr_mc=p_metrics_deepatmo_invr_mc,
        p_metrics_deepatmo_gradh_ifc=p_metrics_deepatmo_gradh_ifc,
        p_metrics_deepatmo_invr_ifc=p_metrics_deepatmo_invr_ifc,
        # naked arrays + scalars
        z_w_concorr_me=z_w_concorr_me,
        z_kin_hor_e=z_kin_hor_e,
        z_vt_ie=z_vt_ie,
        ntnd=np.int32(1),
        istep=np.int32(1),
        lvn_only=np.bool_(False),
        dtime=np.float64(60.0),
        dt_linintp_ubc=np.float64(0.0),
        ldeepatmo=np.bool_(False),
    )

    # When SDFG build + run starts working, layer a gfortran/f2py reference
    # comparison here.  For now the xfail probe focuses on bridge-build
    # feasibility -- success requires the SDFG to be constructed and run
    # without crashing on this struct-heavy signature.
    assert np.all(np.isfinite(p_prog_w))
