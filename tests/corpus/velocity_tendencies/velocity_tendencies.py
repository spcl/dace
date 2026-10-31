# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""ICON's ``mo_velocity_advection.velocity_tendencies`` (https://gitlab.dkrz.de/icon/icon-model, BSD-3-Clause) as a
single ``dace.program``, ported to NumPy in C order: Fortran ``vn(nproma, nlev, nblks)`` is ``vn[nblks, nlev, nproma]``
and a connectivity table ``idx(nproma, nblks, n)`` is ``idx[n, nblks, nproma]``. Every loop is a statement over a
level or a neighbour slot; the neighbour gathers are explicit loops."""

import numpy as np

import dace as dc

dc_float = dc.float64
istep = dc.symbol("istep", dtype=dc.int64, positive=True)
lvn_only = dc.symbol("lvn_only", dtype=dc.int64, nonnegative=True)
ldeepatmo = dc.symbol("ldeepatmo", dtype=dc.int64, nonnegative=True)
nproma = dc.symbol("nproma", dtype=dc.int64, positive=True)
nlev = dc.symbol("nlev", dtype=dc.int64, positive=True)
nblks_c = dc.symbol("nblks_c", dtype=dc.int64, positive=True)
nblks_e = dc.symbol("nblks_e", dtype=dc.int64, positive=True)
nblks_v = dc.symbol("nblks_v", dtype=dc.int64, positive=True)
nlevp1 = dc.symbol("nlevp1", dtype=dc.int64, nonnegative=True)


@dc.program
def velocity_tendencies(
    p_patch_cells_area: dc_float[nblks_c, nproma],
    p_patch_cells_neighbor_idx: dc.int32[3, nblks_c, nproma],
    p_patch_cells_neighbor_blk: dc.int32[3, nblks_c, nproma],
    p_patch_cells_edge_idx: dc.int32[3, nblks_c, nproma],
    p_patch_cells_edge_blk: dc.int32[3, nblks_c, nproma],
    p_patch_cells_start_index: dc.int32[33],
    p_patch_cells_end_index: dc.int32[33],
    p_patch_cells_start_block: dc.int32[33],
    p_patch_cells_end_block: dc.int32[33],
    p_patch_cells_decomp_info_owner_mask: dc.int8[nblks_c, nproma],
    p_patch_edges_cell_idx: dc.int32[2, nblks_e, nproma],
    p_patch_edges_cell_blk: dc.int32[2, nblks_e, nproma],
    p_patch_edges_vertex_idx: dc.int32[4, nblks_e, nproma],
    p_patch_edges_vertex_blk: dc.int32[4, nblks_e, nproma],
    p_patch_edges_quad_idx: dc.int32[4, nblks_e, nproma],
    p_patch_edges_quad_blk: dc.int32[4, nblks_e, nproma],
    p_patch_edges_tangent_orientation: dc_float[nblks_e, nproma],
    p_patch_edges_inv_primal_edge_length: dc_float[nblks_e, nproma],
    p_patch_edges_inv_dual_edge_length: dc_float[nblks_e, nproma],
    p_patch_edges_area_edge: dc_float[nblks_e, nproma],
    p_patch_edges_f_e: dc_float[nblks_e, nproma],
    p_patch_edges_fn_e: dc_float[nblks_e, nproma],
    p_patch_edges_ft_e: dc_float[nblks_e, nproma],
    p_patch_edges_start_index: dc.int32[33],
    p_patch_edges_end_index: dc.int32[33],
    p_patch_edges_start_block: dc.int32[33],
    p_patch_edges_end_block: dc.int32[33],
    p_patch_verts_cell_idx: dc.int32[6, nblks_v, nproma],
    p_patch_verts_cell_blk: dc.int32[6, nblks_v, nproma],
    p_patch_verts_edge_idx: dc.int32[6, nblks_v, nproma],
    p_patch_verts_edge_blk: dc.int32[6, nblks_v, nproma],
    p_patch_verts_start_index: dc.int32[33],
    p_patch_verts_end_index: dc.int32[33],
    p_patch_verts_start_block: dc.int32[33],
    p_patch_verts_end_block: dc.int32[33],
    p_int_c_lin_e: dc_float[nblks_e, 2, nproma],
    p_int_e_bln_c_s: dc_float[nblks_c, 3, nproma],
    p_int_cells_aw_verts: dc_float[nblks_v, 6, nproma],
    p_int_rbf_vec_coeff_e: dc_float[nblks_e, nproma, 4],
    p_int_geofac_grdiv: dc_float[nblks_e, 5, nproma],
    p_int_geofac_rot: dc_float[nblks_v, 6, nproma],
    p_int_geofac_n2s: dc_float[nblks_c, 4, nproma],
    p_prog_w: dc_float[nblks_c, nlev + 1, nproma],
    p_prog_vn: dc_float[nblks_e, nlev, nproma],
    p_diag_vn_ie_ubc: dc_float[nblks_e, 2, nproma],
    p_diag_vt: dc_float[nblks_e, nlev, nproma],
    p_diag_vn_ie: dc_float[nblks_e, nlev + 1, nproma],
    p_diag_w_concorr_c: dc_float[nblks_c, nlev, nproma],
    p_diag_ddt_vn_apc_pc: dc_float[3, nblks_e, nlev, nproma],
    p_diag_ddt_vn_cor_pc: dc_float[3, nblks_e, nlev, nproma],
    p_diag_ddt_w_adv_pc: dc_float[3, nblks_c, nlev + 1, nproma],
    p_diag_max_vcfl_dyn: dc_float[1],
    p_metrics_ddxn_z_full: dc_float[nblks_e, nlev, nproma],
    p_metrics_ddxt_z_full: dc_float[nblks_e, nlev, nproma],
    p_metrics_ddqz_z_full_e: dc_float[nblks_e, nlev, nproma],
    p_metrics_ddqz_z_half: dc_float[nblks_c, nlev + 1, nproma],
    p_metrics_wgtfac_c: dc_float[nblks_c, nlev + 1, nproma],
    p_metrics_wgtfac_e: dc_float[nblks_e, nlev + 1, nproma],
    p_metrics_wgtfacq_e: dc_float[nblks_e, 3, nproma],
    p_metrics_coeff_gradekin: dc_float[nblks_e, 2, nproma],
    p_metrics_coeff1_dwdz: dc_float[nblks_c, nlev, nproma],
    p_metrics_coeff2_dwdz: dc_float[nblks_c, nlev, nproma],
    p_metrics_deepatmo_gradh_mc: dc_float[nlev],
    p_metrics_deepatmo_invr_mc: dc_float[nlev],
    p_metrics_deepatmo_gradh_ifc: dc_float[nlev + 1],
    p_metrics_deepatmo_invr_ifc: dc_float[nlev + 1],
    z_w_concorr_me: dc_float[nblks_e, nlev, nproma],
    z_kin_hor_e: dc_float[nblks_e, nlev, nproma],
    z_vt_ie: dc_float[nblks_e, nlev + 1, nproma],
    ntnd: dc.int64,
    lextra_diffu: dc.int64,
    l_vert_nested: dc.int64,
    ddt_vn_cor_associated: dc.int64,
    dtime: dc_float,
    dt_linintp_ubc: dc_float,
    nrdmax_jg: dc.int64,
    nflatlev_jg: dc.int64,
):
    t = ntnd - 1
    if lextra_diffu:
        cfl_w_limit = 0.65 / dtime
        scalfac_exdiff = 0.05 / (dtime * (0.85 - cfl_w_limit * dtime))
    else:
        cfl_w_limit = 0.85 / dtime
        scalfac_exdiff = 0.0
    z_w_v = np.zeros((nblks_v, nlevp1, nproma), dtype=p_prog_w.dtype)
    if not lvn_only:
        for jk in range(nlevp1):
            acc_zwv = np.zeros((nblks_v, nproma), dtype=p_prog_w.dtype)
            for n in range(6):
                __gather0_x0 = p_patch_verts_cell_blk[n, :, :]
                __gather0_x2 = p_patch_verts_cell_idx[n, :, :]
                __gather0_o = np.empty(__gather0_x0.shape, p_prog_w.dtype)
                for __gather0_i0 in range(nblks_v):
                    for __gather0_i1 in range(nproma):
                        __gather0_o[__gather0_i0, __gather0_i1] = p_prog_w[
                            __gather0_x0[__gather0_i0, __gather0_i1], jk, __gather0_x2[__gather0_i0, __gather0_i1]
                        ]
                acc_zwv[:] = acc_zwv + p_int_cells_aw_verts[:, n, :] * __gather0_o
            z_w_v[:, jk, :] = acc_zwv
    zeta = np.zeros((nblks_v, nlev, nproma), dtype=p_prog_vn.dtype)
    for jk in range(nlev):
        acc_zeta = np.zeros((nblks_v, nproma), dtype=p_prog_vn.dtype)
        for n in range(6):
            __gather1_x0 = p_patch_verts_edge_blk[n, :, :]
            __gather1_x2 = p_patch_verts_edge_idx[n, :, :]
            __gather1_o = np.empty(__gather1_x0.shape, p_prog_vn.dtype)
            for __gather1_i0 in range(nblks_v):
                for __gather1_i1 in range(nproma):
                    __gather1_o[__gather1_i0, __gather1_i1] = p_prog_vn[
                        __gather1_x0[__gather1_i0, __gather1_i1], jk, __gather1_x2[__gather1_i0, __gather1_i1]
                    ]
            acc_zeta[:] = acc_zeta + __gather1_o * p_int_geofac_rot[:, n, :]
        zeta[:, jk, :] = acc_zeta
    if istep == 1:
        for jk in range(nlev):
            acc_vt = np.zeros((nblks_e, nproma), dtype=p_prog_vn.dtype)
            for n in range(4):
                __gather2_x0 = p_patch_edges_quad_blk[n, :, :]
                __gather2_x2 = p_patch_edges_quad_idx[n, :, :]
                __gather2_o = np.empty(__gather2_x0.shape, p_prog_vn.dtype)
                for __gather2_i0 in range(nblks_e):
                    for __gather2_i1 in range(nproma):
                        __gather2_o[__gather2_i0, __gather2_i1] = p_prog_vn[
                            __gather2_x0[__gather2_i0, __gather2_i1], jk, __gather2_x2[__gather2_i0, __gather2_i1]
                        ]
                acc_vt[:] = acc_vt + p_int_rbf_vec_coeff_e[:, :, n] * __gather2_o
            p_diag_vt[:, jk, :] = acc_vt
        for jk in range(1, nlev):
            we = p_metrics_wgtfac_e[:, jk, :]
            p_diag_vn_ie[:, jk, :] = we * p_prog_vn[:, jk, :] + (1.0 - we) * p_prog_vn[:, jk - 1, :]
            z_kin_hor_e[:, jk, :] = 0.5 * (
                p_prog_vn[:, jk, :] * p_prog_vn[:, jk, :] + p_diag_vt[:, jk, :] * p_diag_vt[:, jk, :]
            )
        if not lvn_only:
            for jk in range(1, nlev):
                we__v2 = p_metrics_wgtfac_e[:, jk, :]
                z_vt_ie[:, jk, :] = we__v2 * p_diag_vt[:, jk, :] + (1.0 - we__v2) * p_diag_vt[:, jk - 1, :]
        for jk in range(nflatlev_jg - 1, nlev):
            z_w_concorr_me[:, jk, :] = (
                p_prog_vn[:, jk, :] * p_metrics_ddxn_z_full[:, jk, :]
                + p_diag_vt[:, jk, :] * p_metrics_ddxt_z_full[:, jk, :]
            )
        if not l_vert_nested:
            p_diag_vn_ie[:, 0, :] = p_prog_vn[:, 0, :]
        else:
            p_diag_vn_ie[:, 0, :] = p_diag_vn_ie_ubc[:, 0, :] + dt_linintp_ubc * p_diag_vn_ie_ubc[:, 1, :]
        z_vt_ie[:, 0, :] = p_diag_vt[:, 0, :]
        z_kin_hor_e[:, 0, :] = 0.5 * (p_prog_vn[:, 0, :] * p_prog_vn[:, 0, :] + p_diag_vt[:, 0, :] * p_diag_vt[:, 0, :])
        p_diag_vn_ie[:, nlevp1 - 1, :] = (
            p_metrics_wgtfacq_e[:, 0, :] * p_prog_vn[:, nlev - 1, :]
            + p_metrics_wgtfacq_e[:, 1, :] * p_prog_vn[:, nlev - 2, :]
            + p_metrics_wgtfacq_e[:, 2, :] * p_prog_vn[:, nlev - 3, :]
        )
    z_v_grad_w = np.zeros((nblks_e, nlev, nproma), dtype=p_diag_vn_ie.dtype)
    if not lvn_only:
        for jk in range(nlev):
            __gather3_x0 = p_patch_edges_cell_blk[0, :, :]
            __gather3_x2 = p_patch_edges_cell_idx[0, :, :]
            __gather3_o = np.empty(__gather3_x0.shape, p_prog_w.dtype)
            for __gather3_i0 in range(nblks_e):
                for __gather3_i1 in range(nproma):
                    __gather3_o[__gather3_i0, __gather3_i1] = p_prog_w[
                        __gather3_x0[__gather3_i0, __gather3_i1], jk, __gather3_x2[__gather3_i0, __gather3_i1]
                    ]
            __gather4_x0 = p_patch_edges_cell_blk[1, :, :]
            __gather4_x2 = p_patch_edges_cell_idx[1, :, :]
            __gather4_o = np.empty(__gather4_x0.shape, p_prog_w.dtype)
            for __gather4_i0 in range(nblks_e):
                for __gather4_i1 in range(nproma):
                    __gather4_o[__gather4_i0, __gather4_i1] = p_prog_w[
                        __gather4_x0[__gather4_i0, __gather4_i1], jk, __gather4_x2[__gather4_i0, __gather4_i1]
                    ]
            __gather5_x0 = p_patch_edges_vertex_blk[0, :, :]
            __gather5_x2 = p_patch_edges_vertex_idx[0, :, :]
            __gather5_o = np.empty(__gather5_x0.shape, z_w_v.dtype)
            for __gather5_i0 in range(nblks_e):
                for __gather5_i1 in range(nproma):
                    __gather5_o[__gather5_i0, __gather5_i1] = z_w_v[
                        __gather5_x0[__gather5_i0, __gather5_i1], jk, __gather5_x2[__gather5_i0, __gather5_i1]
                    ]
            __gather6_x0 = p_patch_edges_vertex_blk[1, :, :]
            __gather6_x2 = p_patch_edges_vertex_idx[1, :, :]
            __gather6_o = np.empty(__gather6_x0.shape, z_w_v.dtype)
            for __gather6_i0 in range(nblks_e):
                for __gather6_i1 in range(nproma):
                    __gather6_o[__gather6_i0, __gather6_i1] = z_w_v[
                        __gather6_x0[__gather6_i0, __gather6_i1], jk, __gather6_x2[__gather6_i0, __gather6_i1]
                    ]
            z_v_grad_w[:, jk, :] = p_diag_vn_ie[:, jk, :] * p_patch_edges_inv_dual_edge_length * (
                __gather3_o - __gather4_o
            ) + z_vt_ie[:, jk, :] * p_patch_edges_inv_primal_edge_length * p_patch_edges_tangent_orientation * (
                __gather5_o - __gather6_o
            )
        if ldeepatmo:
            for jk in range(nlev):
                z_v_grad_w[:, jk, :] = (
                    z_v_grad_w[:, jk, :] * p_metrics_deepatmo_gradh_ifc[jk]
                    + p_diag_vn_ie[:, jk, :]
                    * (p_diag_vn_ie[:, jk, :] * p_metrics_deepatmo_invr_ifc[jk] - p_patch_edges_ft_e)
                    + z_vt_ie[:, jk, :] * (z_vt_ie[:, jk, :] * p_metrics_deepatmo_invr_ifc[jk] + p_patch_edges_fn_e)
                )
    owner = p_patch_cells_decomp_info_owner_mask != 0
    z_ekinh = np.zeros((nblks_c, nlev, nproma), dtype=z_kin_hor_e.dtype)
    for jk in range(nlev):
        acc_ek = np.zeros((nblks_c, nproma), dtype=z_kin_hor_e.dtype)
        for n in range(3):
            __gather7_x0 = p_patch_cells_edge_blk[n, :, :]
            __gather7_x2 = p_patch_cells_edge_idx[n, :, :]
            __gather7_o = np.empty(__gather7_x0.shape, z_kin_hor_e.dtype)
            for __gather7_i0 in range(nblks_c):
                for __gather7_i1 in range(nproma):
                    __gather7_o[__gather7_i0, __gather7_i1] = z_kin_hor_e[
                        __gather7_x0[__gather7_i0, __gather7_i1], jk, __gather7_x2[__gather7_i0, __gather7_i1]
                    ]
            acc_ek[:] = acc_ek + p_int_e_bln_c_s[:, n, :] * __gather7_o
        z_ekinh[:, jk, :] = acc_ek
    if istep == 1:
        z_w_concorr_mc = np.zeros((nblks_c, nlev, nproma), dtype=z_w_concorr_me.dtype)
        for jk in range(nflatlev_jg - 1, nlev):
            acc_wcm = np.zeros((nblks_c, nproma), dtype=z_w_concorr_me.dtype)
            for n in range(3):
                __gather8_x0 = p_patch_cells_edge_blk[n, :, :]
                __gather8_x2 = p_patch_cells_edge_idx[n, :, :]
                __gather8_o = np.empty(__gather8_x0.shape, z_w_concorr_me.dtype)
                for __gather8_i0 in range(nblks_c):
                    for __gather8_i1 in range(nproma):
                        __gather8_o[__gather8_i0, __gather8_i1] = z_w_concorr_me[
                            __gather8_x0[__gather8_i0, __gather8_i1], jk, __gather8_x2[__gather8_i0, __gather8_i1]
                        ]
                acc_wcm[:] = acc_wcm + p_int_e_bln_c_s[:, n, :] * __gather8_o
            z_w_concorr_mc[:, jk, :] = acc_wcm
        for jk in range(nflatlev_jg, nlev):
            wc = p_metrics_wgtfac_c[:, jk, :]
            p_diag_w_concorr_c[:, jk, :] = wc * z_w_concorr_mc[:, jk, :] + (1.0 - wc) * z_w_concorr_mc[:, jk - 1, :]
    z_w_con_c = np.zeros((nblks_c, nlevp1, nproma), dtype=p_prog_w.dtype)
    z_w_con_c[:, :nlev, :] = p_prog_w[:, :nlev, :]
    z_w_con_c[:, nlevp1 - 1, :] = 0.0
    for jk in range(nflatlev_jg, nlev):
        z_w_con_c[:, jk, :] = z_w_con_c[:, jk, :] - p_diag_w_concorr_c[:, jk, :]
    vcflmax = np.zeros(nblks_c, dtype=z_w_con_c.dtype)
    cfl_clip = np.zeros((nblks_c, nlevp1, nproma), dtype=np.bool_)
    jk0_lo = max(3, nrdmax_jg - 2) - 1
    if nlev - 4 >= jk0_lo:
        h = p_metrics_ddqz_z_half[:, jk0_lo : nlev - 4 + 1, :]
        zc = z_w_con_c[:, jk0_lo : nlev - 4 + 1, :]
        clip = np.abs(zc) > cfl_w_limit * h
        vcfl = zc * dtime / h
        cfl_clip[:, jk0_lo : nlev - 4 + 1, :] = clip
        abs_vcfl_masked = np.where(clip, np.abs(vcfl), -np.inf)
        vcflmax = np.maximum(vcflmax, np.max(abs_vcfl_masked, axis=(1, 2)))
        lo_clip = clip & (vcfl < -0.85)
        hi_clip = clip & (vcfl > 0.85)
        z_w_con_c[:, jk0_lo : nlev - 4 + 1, :] = np.where(
            lo_clip, -0.85 * h / dtime, np.where(hi_clip, 0.85 * h / dtime, zc)
        )
    z_w_con_c_full = np.zeros((nblks_c, nlev, nproma), dtype=z_w_con_c.dtype)
    for jk in range(nlev):
        z_w_con_c_full[:, jk, :] = 0.5 * (z_w_con_c[:, jk, :] + z_w_con_c[:, jk + 1, :])
    p_diag_max_vcfl_dyn[0] = max(float(p_diag_max_vcfl_dyn[0]), float(vcflmax.max()))
    if not lvn_only:
        for jk in range(1, nlev):
            p_diag_ddt_w_adv_pc[t, :, jk, :] = -z_w_con_c[:, jk, :] * (
                p_prog_w[:, jk - 1, :] * p_metrics_coeff1_dwdz[:, jk, :]
                - p_prog_w[:, jk + 1, :] * p_metrics_coeff2_dwdz[:, jk, :]
                + p_prog_w[:, jk, :] * (p_metrics_coeff2_dwdz[:, jk, :] - p_metrics_coeff1_dwdz[:, jk, :])
            )
        for jk in range(1, nlev):
            acc_dwa = np.zeros((nblks_c, nproma), dtype=z_v_grad_w.dtype)
            for n in range(3):
                __gather9_x0 = p_patch_cells_edge_blk[n, :, :]
                __gather9_x2 = p_patch_cells_edge_idx[n, :, :]
                __gather9_o = np.empty(__gather9_x0.shape, z_v_grad_w.dtype)
                for __gather9_i0 in range(nblks_c):
                    for __gather9_i1 in range(nproma):
                        __gather9_o[__gather9_i0, __gather9_i1] = z_v_grad_w[
                            __gather9_x0[__gather9_i0, __gather9_i1], jk, __gather9_x2[__gather9_i0, __gather9_i1]
                        ]
                acc_dwa[:] = acc_dwa + p_int_e_bln_c_s[:, n, :] * __gather9_o
            p_diag_ddt_w_adv_pc[t, :, jk, :] = p_diag_ddt_w_adv_pc[t, :, jk, :] + acc_dwa
        if lextra_diffu:
            for jk1 in range(max(3, nrdmax_jg - 2), nlev - 3 + 1):
                jk0 = jk1 - 1
                mask = cfl_clip[:, jk0, :] & owner
                difcoef_c = scalfac_exdiff * np.minimum(
                    0.85 - cfl_w_limit * dtime,
                    np.abs(z_w_con_c[:, jk0, :]) * dtime / p_metrics_ddqz_z_half[:, jk0, :] - cfl_w_limit * dtime,
                )
                __gather10_x0 = p_patch_cells_neighbor_blk[0, :, :]
                __gather10_x2 = p_patch_cells_neighbor_idx[0, :, :]
                __gather10_o = np.empty(__gather10_x0.shape, p_prog_w.dtype)
                for __gather10_i0 in range(nblks_c):
                    for __gather10_i1 in range(nproma):
                        __gather10_o[__gather10_i0, __gather10_i1] = p_prog_w[
                            __gather10_x0[__gather10_i0, __gather10_i1],
                            jk0,
                            __gather10_x2[__gather10_i0, __gather10_i1],
                        ]
                __gather11_x0 = p_patch_cells_neighbor_blk[1, :, :]
                __gather11_x2 = p_patch_cells_neighbor_idx[1, :, :]
                __gather11_o = np.empty(__gather11_x0.shape, p_prog_w.dtype)
                for __gather11_i0 in range(nblks_c):
                    for __gather11_i1 in range(nproma):
                        __gather11_o[__gather11_i0, __gather11_i1] = p_prog_w[
                            __gather11_x0[__gather11_i0, __gather11_i1],
                            jk0,
                            __gather11_x2[__gather11_i0, __gather11_i1],
                        ]
                __gather12_x0 = p_patch_cells_neighbor_blk[2, :, :]
                __gather12_x2 = p_patch_cells_neighbor_idx[2, :, :]
                __gather12_o = np.empty(__gather12_x0.shape, p_prog_w.dtype)
                for __gather12_i0 in range(nblks_c):
                    for __gather12_i1 in range(nproma):
                        __gather12_o[__gather12_i0, __gather12_i1] = p_prog_w[
                            __gather12_x0[__gather12_i0, __gather12_i1],
                            jk0,
                            __gather12_x2[__gather12_i0, __gather12_i1],
                        ]
                lap = (
                    p_prog_w[:, jk0, :] * p_int_geofac_n2s[:, 0, :]
                    + __gather10_o * p_int_geofac_n2s[:, 1, :]
                    + __gather11_o * p_int_geofac_n2s[:, 2, :]
                    + __gather12_o * p_int_geofac_n2s[:, 3, :]
                )
                p_diag_ddt_w_adv_pc[t, :, jk0, :] = p_diag_ddt_w_adv_pc[t, :, jk0, :] + np.where(
                    mask, difcoef_c * p_patch_cells_area * lap, 0.0
                )
    __rsrc0 = cfl_clip[:, :nlev, :]
    __rdo0 = np.empty((nlev,), np.bool_)
    for __rd0_k1 in range(nlev):
        __rdo0[__rd0_k1] = False
        for __rd0_j0 in range(nblks_c):
            for __rd0_j2 in range(nproma):
                if __rsrc0[__rd0_j0, __rd0_k1, __rd0_j2]:
                    __rdo0[__rd0_k1] = True
    levelmask = __rdo0
    for jk in range(nlev):
        __gather13_x0 = p_patch_edges_cell_blk[0, :, :]
        __gather13_x2 = p_patch_edges_cell_idx[0, :, :]
        __gather13_o = np.empty(__gather13_x0.shape, z_ekinh.dtype)
        for __gather13_i0 in range(nblks_e):
            for __gather13_i1 in range(nproma):
                __gather13_o[__gather13_i0, __gather13_i1] = z_ekinh[
                    __gather13_x0[__gather13_i0, __gather13_i1], jk, __gather13_x2[__gather13_i0, __gather13_i1]
                ]
        ekc1 = __gather13_o
        __gather14_x0 = p_patch_edges_cell_blk[1, :, :]
        __gather14_x2 = p_patch_edges_cell_idx[1, :, :]
        __gather14_o = np.empty(__gather14_x0.shape, z_ekinh.dtype)
        for __gather14_i0 in range(nblks_e):
            for __gather14_i1 in range(nproma):
                __gather14_o[__gather14_i0, __gather14_i1] = z_ekinh[
                    __gather14_x0[__gather14_i0, __gather14_i1], jk, __gather14_x2[__gather14_i0, __gather14_i1]
                ]
        ekc2 = __gather14_o
        __gather15_x0 = p_patch_edges_vertex_blk[0, :, :]
        __gather15_x2 = p_patch_edges_vertex_idx[0, :, :]
        __gather15_o = np.empty(__gather15_x0.shape, zeta.dtype)
        for __gather15_i0 in range(nblks_e):
            for __gather15_i1 in range(nproma):
                __gather15_o[__gather15_i0, __gather15_i1] = zeta[
                    __gather15_x0[__gather15_i0, __gather15_i1], jk, __gather15_x2[__gather15_i0, __gather15_i1]
                ]
        zv1 = __gather15_o
        __gather16_x0 = p_patch_edges_vertex_blk[1, :, :]
        __gather16_x2 = p_patch_edges_vertex_idx[1, :, :]
        __gather16_o = np.empty(__gather16_x0.shape, zeta.dtype)
        for __gather16_i0 in range(nblks_e):
            for __gather16_i1 in range(nproma):
                __gather16_o[__gather16_i0, __gather16_i1] = zeta[
                    __gather16_x0[__gather16_i0, __gather16_i1], jk, __gather16_x2[__gather16_i0, __gather16_i1]
                ]
        zv2 = __gather16_o
        __gather17_x0 = p_patch_edges_cell_blk[0, :, :]
        __gather17_x2 = p_patch_edges_cell_idx[0, :, :]
        __gather17_o = np.empty(__gather17_x0.shape, z_w_con_c_full.dtype)
        for __gather17_i0 in range(nblks_e):
            for __gather17_i1 in range(nproma):
                __gather17_o[__gather17_i0, __gather17_i1] = z_w_con_c_full[
                    __gather17_x0[__gather17_i0, __gather17_i1], jk, __gather17_x2[__gather17_i0, __gather17_i1]
                ]
        wcf1 = __gather17_o
        __gather18_x0 = p_patch_edges_cell_blk[1, :, :]
        __gather18_x2 = p_patch_edges_cell_idx[1, :, :]
        __gather18_o = np.empty(__gather18_x0.shape, z_w_con_c_full.dtype)
        for __gather18_i0 in range(nblks_e):
            for __gather18_i1 in range(nproma):
                __gather18_o[__gather18_i0, __gather18_i1] = z_w_con_c_full[
                    __gather18_x0[__gather18_i0, __gather18_i1], jk, __gather18_x2[__gather18_i0, __gather18_i1]
                ]
        wcf2 = __gather18_o
        clin = p_int_c_lin_e[:, 0, :] * wcf1 + p_int_c_lin_e[:, 1, :] * wcf2
        grad_ekin = (
            z_kin_hor_e[:, jk, :] * (p_metrics_coeff_gradekin[:, 0, :] - p_metrics_coeff_gradekin[:, 1, :])
            + p_metrics_coeff_gradekin[:, 1, :] * ekc2
            - p_metrics_coeff_gradekin[:, 0, :] * ekc1
        )
        if not ldeepatmo:
            p_diag_ddt_vn_apc_pc[t, :, jk, :] = -(
                grad_ekin
                + p_diag_vt[:, jk, :] * (p_patch_edges_f_e + 0.5 * (zv1 + zv2))
                + clin * (p_diag_vn_ie[:, jk, :] - p_diag_vn_ie[:, jk + 1, :]) / p_metrics_ddqz_z_full_e[:, jk, :]
            )
            if ddt_vn_cor_associated:
                p_diag_ddt_vn_cor_pc[t, :, jk, :] = -p_diag_vt[:, jk, :] * p_patch_edges_f_e
        else:
            p_diag_ddt_vn_apc_pc[t, :, jk, :] = -(
                grad_ekin * p_metrics_deepatmo_gradh_mc[jk]
                + p_diag_vt[:, jk, :] * (p_patch_edges_f_e + 0.5 * (zv1 + zv2) * p_metrics_deepatmo_gradh_mc[jk])
                + clin
                * (
                    (p_diag_vn_ie[:, jk, :] - p_diag_vn_ie[:, jk + 1, :]) / p_metrics_ddqz_z_full_e[:, jk, :]
                    + p_prog_vn[:, jk, :] * p_metrics_deepatmo_invr_mc[jk]
                    - p_patch_edges_ft_e
                )
            )
            if ddt_vn_cor_associated:
                p_diag_ddt_vn_cor_pc[t, :, jk, :] = -(
                    p_diag_vt[:, jk, :] * p_patch_edges_f_e + clin * -p_patch_edges_ft_e
                )
    if lextra_diffu:
        for jk1 in range(max(3, nrdmax_jg - 2), nlev - 4 + 1):
            jk0__v2 = jk1 - 1
            if not (levelmask[jk0__v2] or levelmask[jk0__v2 + 1]):
                continue
            __gather19_x0 = p_patch_edges_cell_blk[0, :, :]
            __gather19_x2 = p_patch_edges_cell_idx[0, :, :]
            __gather19_o = np.empty(__gather19_x0.shape, z_w_con_c_full.dtype)
            for __gather19_i0 in range(nblks_e):
                for __gather19_i1 in range(nproma):
                    __gather19_o[__gather19_i0, __gather19_i1] = z_w_con_c_full[
                        __gather19_x0[__gather19_i0, __gather19_i1],
                        jk0__v2,
                        __gather19_x2[__gather19_i0, __gather19_i1],
                    ]
            __gather20_x0 = p_patch_edges_cell_blk[1, :, :]
            __gather20_x2 = p_patch_edges_cell_idx[1, :, :]
            __gather20_o = np.empty(__gather20_x0.shape, z_w_con_c_full.dtype)
            for __gather20_i0 in range(nblks_e):
                for __gather20_i1 in range(nproma):
                    __gather20_o[__gather20_i0, __gather20_i1] = z_w_con_c_full[
                        __gather20_x0[__gather20_i0, __gather20_i1],
                        jk0__v2,
                        __gather20_x2[__gather20_i0, __gather20_i1],
                    ]
            w_con_e = p_int_c_lin_e[:, 0, :] * __gather19_o + p_int_c_lin_e[:, 1, :] * __gather20_o
            clip_e = np.abs(w_con_e) > cfl_w_limit * p_metrics_ddqz_z_full_e[:, jk0__v2, :]
            difcoef_e = scalfac_exdiff * np.minimum(
                0.85 - cfl_w_limit * dtime,
                np.abs(w_con_e) * dtime / p_metrics_ddqz_z_full_e[:, jk0__v2, :] - cfl_w_limit * dtime,
            )
            __gather21_x0 = p_patch_edges_quad_blk[0, :, :]
            __gather21_x2 = p_patch_edges_quad_idx[0, :, :]
            __gather21_o = np.empty(__gather21_x0.shape, p_prog_vn.dtype)
            for __gather21_i0 in range(nblks_e):
                for __gather21_i1 in range(nproma):
                    __gather21_o[__gather21_i0, __gather21_i1] = p_prog_vn[
                        __gather21_x0[__gather21_i0, __gather21_i1],
                        jk0__v2,
                        __gather21_x2[__gather21_i0, __gather21_i1],
                    ]
            __gather22_x0 = p_patch_edges_quad_blk[1, :, :]
            __gather22_x2 = p_patch_edges_quad_idx[1, :, :]
            __gather22_o = np.empty(__gather22_x0.shape, p_prog_vn.dtype)
            for __gather22_i0 in range(nblks_e):
                for __gather22_i1 in range(nproma):
                    __gather22_o[__gather22_i0, __gather22_i1] = p_prog_vn[
                        __gather22_x0[__gather22_i0, __gather22_i1],
                        jk0__v2,
                        __gather22_x2[__gather22_i0, __gather22_i1],
                    ]
            __gather23_x0 = p_patch_edges_quad_blk[2, :, :]
            __gather23_x2 = p_patch_edges_quad_idx[2, :, :]
            __gather23_o = np.empty(__gather23_x0.shape, p_prog_vn.dtype)
            for __gather23_i0 in range(nblks_e):
                for __gather23_i1 in range(nproma):
                    __gather23_o[__gather23_i0, __gather23_i1] = p_prog_vn[
                        __gather23_x0[__gather23_i0, __gather23_i1],
                        jk0__v2,
                        __gather23_x2[__gather23_i0, __gather23_i1],
                    ]
            __gather24_x0 = p_patch_edges_quad_blk[3, :, :]
            __gather24_x2 = p_patch_edges_quad_idx[3, :, :]
            __gather24_o = np.empty(__gather24_x0.shape, p_prog_vn.dtype)
            for __gather24_i0 in range(nblks_e):
                for __gather24_i1 in range(nproma):
                    __gather24_o[__gather24_i0, __gather24_i1] = p_prog_vn[
                        __gather24_x0[__gather24_i0, __gather24_i1],
                        jk0__v2,
                        __gather24_x2[__gather24_i0, __gather24_i1],
                    ]
            __gather25_x0 = p_patch_edges_vertex_blk[1, :, :]
            __gather25_x2 = p_patch_edges_vertex_idx[1, :, :]
            __gather25_o = np.empty(__gather25_x0.shape, zeta.dtype)
            for __gather25_i0 in range(nblks_e):
                for __gather25_i1 in range(nproma):
                    __gather25_o[__gather25_i0, __gather25_i1] = zeta[
                        __gather25_x0[__gather25_i0, __gather25_i1],
                        jk0__v2,
                        __gather25_x2[__gather25_i0, __gather25_i1],
                    ]
            __gather26_x0 = p_patch_edges_vertex_blk[0, :, :]
            __gather26_x2 = p_patch_edges_vertex_idx[0, :, :]
            __gather26_o = np.empty(__gather26_x0.shape, zeta.dtype)
            for __gather26_i0 in range(nblks_e):
                for __gather26_i1 in range(nproma):
                    __gather26_o[__gather26_i0, __gather26_i1] = zeta[
                        __gather26_x0[__gather26_i0, __gather26_i1],
                        jk0__v2,
                        __gather26_x2[__gather26_i0, __gather26_i1],
                    ]
            grad = (
                p_int_geofac_grdiv[:, 0, :] * p_prog_vn[:, jk0__v2, :]
                + p_int_geofac_grdiv[:, 1, :] * __gather21_o
                + p_int_geofac_grdiv[:, 2, :] * __gather22_o
                + p_int_geofac_grdiv[:, 3, :] * __gather23_o
                + p_int_geofac_grdiv[:, 4, :] * __gather24_o
                + p_patch_edges_tangent_orientation
                * p_patch_edges_inv_primal_edge_length
                * (__gather25_o - __gather26_o)
            )
            p_diag_ddt_vn_apc_pc[t, :, jk0__v2, :] = p_diag_ddt_vn_apc_pc[t, :, jk0__v2, :] + np.where(
                clip_e, difcoef_e * p_patch_edges_area_edge * grad, 0.0
            )
