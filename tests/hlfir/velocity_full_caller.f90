! Fortran caller wrappers for the velocity_tendencies e2e test.
!
! Two bind(c) entry points compiled into a shared library via plain
! ``gfortran -shared -fPIC``; the Python test loads the .so via ctypes
! and binds numpy arrays through ``arr.ctypes.data_as(...)``.  This
! bypasses f2py entirely (struct-dummy binding, INTENT(OUT) -> return-
! value tuple, LOGICAL ABI surprises all gone) and gives an in-process
! reference call.
!
!   init_inputs_random_c -- fills every flat buffer with deterministic
!                           random data (RANDOM_NUMBER seeded by `seed`,
!                           index/block tables cyclic in-bounds).
!
!   run_velocity_flat_c  -- wraps the flat buffers back into the
!                           t_patch / t_int_state / t_nh_prog /
!                           t_nh_metrics / t_nh_diag derived-type
!                           dummies (ALLOCATE+memcpy for ALLOCATABLE
!                           inputs, `=>`-pointer-assign for POINTER
!                           buffers) and calls velocity_tendencies.
!
! LOGICAL is exchanged as ``integer(c_int8)`` (1 byte) on the wire to
! match np.bool_'s 1-byte layout exactly; ``logical(c_bool)`` would also
! work but is less portable across compilers.  Arrays are explicit-
! shape using the bound dim scalars (value-passed first), so ctypes
! sees plain base-pointers.

SUBROUTINE init_inputs_random_c(seed, nproma, nlev, nlevp1, &
                                nblks_c, nblks_e, nblks_v, &
                                p_patch_cells_area, &
                                p_patch_cells_neighbor_idx, p_patch_cells_neighbor_blk, &
                                p_patch_cells_edge_idx, p_patch_cells_edge_blk, &
                                p_patch_cells_start_index, p_patch_cells_end_index, &
                                p_patch_cells_start_block, p_patch_cells_end_block, &
                                p_patch_cells_decomp_info_owner_mask, &
                                p_patch_edges_cell_idx, p_patch_edges_cell_blk, &
                                p_patch_edges_vertex_idx, p_patch_edges_vertex_blk, &
                                p_patch_edges_quad_idx, p_patch_edges_quad_blk, &
                                p_patch_edges_tangent_orientation, p_patch_edges_inv_primal_edge_length, &
                                p_patch_edges_inv_dual_edge_length, p_patch_edges_area_edge, &
                                p_patch_edges_f_e, p_patch_edges_fn_e, p_patch_edges_ft_e, &
                                p_patch_edges_start_index, p_patch_edges_end_index, &
                                p_patch_edges_start_block, p_patch_edges_end_block, &
                                p_patch_verts_cell_idx, p_patch_verts_cell_blk, &
                                p_patch_verts_edge_idx, p_patch_verts_edge_blk, &
                                p_patch_verts_start_index, p_patch_verts_end_index, &
                                p_patch_verts_start_block, p_patch_verts_end_block, &
                                p_int_c_lin_e, p_int_e_bln_c_s, p_int_cells_aw_verts, &
                                p_int_rbf_vec_coeff_e, p_int_geofac_grdiv, p_int_geofac_rot, p_int_geofac_n2s, &
                                p_prog_w, p_prog_vn, &
                                p_diag_vn_ie_ubc, p_diag_vt, p_diag_vn_ie, p_diag_w_concorr_c, &
                                p_diag_ddt_vn_apc_pc, p_diag_ddt_vn_cor_pc, p_diag_ddt_w_adv_pc, &
                                p_metrics_ddxn_z_full, p_metrics_ddxt_z_full, &
                                p_metrics_ddqz_z_full_e, p_metrics_ddqz_z_half, &
                                p_metrics_wgtfac_c, p_metrics_wgtfac_e, p_metrics_wgtfacq_e, &
                                p_metrics_coeff_gradekin, p_metrics_coeff1_dwdz, p_metrics_coeff2_dwdz, &
                                p_metrics_deepatmo_gradh_mc, p_metrics_deepatmo_invr_mc, &
                                p_metrics_deepatmo_gradh_ifc, p_metrics_deepatmo_invr_ifc) &
    BIND(C, NAME="init_inputs_random_c")
  USE iso_c_binding
  IMPLICIT NONE
  INTEGER(c_int), VALUE :: seed
  INTEGER(c_int), VALUE :: nproma, nlev, nlevp1, nblks_c, nblks_e, nblks_v
  REAL(c_double),     INTENT(OUT) :: p_patch_cells_area(nproma, nblks_c)
  INTEGER(c_int),     INTENT(OUT) :: p_patch_cells_neighbor_idx(nproma, nblks_c, 3)
  INTEGER(c_int),     INTENT(OUT) :: p_patch_cells_neighbor_blk(nproma, nblks_c, 3)
  INTEGER(c_int),     INTENT(OUT) :: p_patch_cells_edge_idx(nproma, nblks_c, 3)
  INTEGER(c_int),     INTENT(OUT) :: p_patch_cells_edge_blk(nproma, nblks_c, 3)
  INTEGER(c_int),     INTENT(OUT) :: p_patch_cells_start_index(33)
  INTEGER(c_int),     INTENT(OUT) :: p_patch_cells_end_index(33)
  INTEGER(c_int),     INTENT(OUT) :: p_patch_cells_start_block(33)
  INTEGER(c_int),     INTENT(OUT) :: p_patch_cells_end_block(33)
  INTEGER(c_int8_t),  INTENT(OUT) :: p_patch_cells_decomp_info_owner_mask(nproma, nblks_c)
  INTEGER(c_int),     INTENT(OUT) :: p_patch_edges_cell_idx(nproma, nblks_e, 2)
  INTEGER(c_int),     INTENT(OUT) :: p_patch_edges_cell_blk(nproma, nblks_e, 2)
  INTEGER(c_int),     INTENT(OUT) :: p_patch_edges_vertex_idx(nproma, nblks_e, 4)
  INTEGER(c_int),     INTENT(OUT) :: p_patch_edges_vertex_blk(nproma, nblks_e, 4)
  INTEGER(c_int),     INTENT(OUT) :: p_patch_edges_quad_idx(nproma, nblks_e, 4)
  INTEGER(c_int),     INTENT(OUT) :: p_patch_edges_quad_blk(nproma, nblks_e, 4)
  REAL(c_double),     INTENT(OUT) :: p_patch_edges_tangent_orientation(nproma, nblks_e)
  REAL(c_double),     INTENT(OUT) :: p_patch_edges_inv_primal_edge_length(nproma, nblks_e)
  REAL(c_double),     INTENT(OUT) :: p_patch_edges_inv_dual_edge_length(nproma, nblks_e)
  REAL(c_double),     INTENT(OUT) :: p_patch_edges_area_edge(nproma, nblks_e)
  REAL(c_double),     INTENT(OUT) :: p_patch_edges_f_e(nproma, nblks_e)
  REAL(c_double),     INTENT(OUT) :: p_patch_edges_fn_e(nproma, nblks_e)
  REAL(c_double),     INTENT(OUT) :: p_patch_edges_ft_e(nproma, nblks_e)
  INTEGER(c_int),     INTENT(OUT) :: p_patch_edges_start_index(33)
  INTEGER(c_int),     INTENT(OUT) :: p_patch_edges_end_index(33)
  INTEGER(c_int),     INTENT(OUT) :: p_patch_edges_start_block(33)
  INTEGER(c_int),     INTENT(OUT) :: p_patch_edges_end_block(33)
  INTEGER(c_int),     INTENT(OUT) :: p_patch_verts_cell_idx(nproma, nblks_v, 6)
  INTEGER(c_int),     INTENT(OUT) :: p_patch_verts_cell_blk(nproma, nblks_v, 6)
  INTEGER(c_int),     INTENT(OUT) :: p_patch_verts_edge_idx(nproma, nblks_v, 6)
  INTEGER(c_int),     INTENT(OUT) :: p_patch_verts_edge_blk(nproma, nblks_v, 6)
  INTEGER(c_int),     INTENT(OUT) :: p_patch_verts_start_index(33)
  INTEGER(c_int),     INTENT(OUT) :: p_patch_verts_end_index(33)
  INTEGER(c_int),     INTENT(OUT) :: p_patch_verts_start_block(33)
  INTEGER(c_int),     INTENT(OUT) :: p_patch_verts_end_block(33)
  REAL(c_double),     INTENT(OUT) :: p_int_c_lin_e(nproma, 2, nblks_e)
  REAL(c_double),     INTENT(OUT) :: p_int_e_bln_c_s(nproma, 3, nblks_c)
  REAL(c_double),     INTENT(OUT) :: p_int_cells_aw_verts(nproma, 6, nblks_v)
  REAL(c_double),     INTENT(OUT) :: p_int_rbf_vec_coeff_e(4, nproma, nblks_e)
  REAL(c_double),     INTENT(OUT) :: p_int_geofac_grdiv(nproma, 5, nblks_e)
  REAL(c_double),     INTENT(OUT) :: p_int_geofac_rot(nproma, 6, nblks_v)
  REAL(c_double),     INTENT(OUT) :: p_int_geofac_n2s(nproma, 4, nblks_c)
  REAL(c_double),     INTENT(OUT) :: p_prog_w(nproma, nlevp1, nblks_c)
  REAL(c_double),     INTENT(OUT) :: p_prog_vn(nproma, nlev, nblks_e)
  REAL(c_double),     INTENT(OUT) :: p_diag_vn_ie_ubc(nproma, 2, nblks_e)
  REAL(c_double),     INTENT(OUT) :: p_diag_vt(nproma, nlev, nblks_e)
  REAL(c_double),     INTENT(OUT) :: p_diag_vn_ie(nproma, nlevp1, nblks_e)
  REAL(c_double),     INTENT(OUT) :: p_diag_w_concorr_c(nproma, nlev, nblks_c)
  REAL(c_double),     INTENT(OUT) :: p_diag_ddt_vn_apc_pc(nproma, nlev, nblks_e, 3)
  REAL(c_double),     INTENT(OUT) :: p_diag_ddt_vn_cor_pc(nproma, nlev, nblks_e, 3)
  REAL(c_double),     INTENT(OUT) :: p_diag_ddt_w_adv_pc(nproma, nlevp1, nblks_c, 3)
  REAL(c_double),     INTENT(OUT) :: p_metrics_ddxn_z_full(nproma, nlev, nblks_e)
  REAL(c_double),     INTENT(OUT) :: p_metrics_ddxt_z_full(nproma, nlev, nblks_e)
  REAL(c_double),     INTENT(OUT) :: p_metrics_ddqz_z_full_e(nproma, nlev, nblks_e)
  REAL(c_double),     INTENT(OUT) :: p_metrics_ddqz_z_half(nproma, nlevp1, nblks_c)
  REAL(c_double),     INTENT(OUT) :: p_metrics_wgtfac_c(nproma, nlevp1, nblks_c)
  REAL(c_double),     INTENT(OUT) :: p_metrics_wgtfac_e(nproma, nlevp1, nblks_e)
  REAL(c_double),     INTENT(OUT) :: p_metrics_wgtfacq_e(nproma, 3, nblks_e)
  REAL(c_double),     INTENT(OUT) :: p_metrics_coeff_gradekin(nproma, 2, nblks_e)
  REAL(c_double),     INTENT(OUT) :: p_metrics_coeff1_dwdz(nproma, nlev, nblks_c)
  REAL(c_double),     INTENT(OUT) :: p_metrics_coeff2_dwdz(nproma, nlev, nblks_c)
  REAL(c_double),     INTENT(OUT) :: p_metrics_deepatmo_gradh_mc(nlev)
  REAL(c_double),     INTENT(OUT) :: p_metrics_deepatmo_invr_mc(nlev)
  REAL(c_double),     INTENT(OUT) :: p_metrics_deepatmo_gradh_ifc(nlevp1)
  REAL(c_double),     INTENT(OUT) :: p_metrics_deepatmo_invr_ifc(nlevp1)
  INTEGER, ALLOCATABLE :: seed_arr(:)
  INTEGER :: n_seed, i

  CALL RANDOM_SEED(SIZE = n_seed)
  ALLOCATE(seed_arr(n_seed))
  seed_arr = seed + [(i, i = 0, n_seed - 1)]
  CALL RANDOM_SEED(PUT = seed_arr)
  DEALLOCATE(seed_arr)

  CALL RANDOM_NUMBER(p_patch_cells_area);                  p_patch_cells_area = 2.0D0 * p_patch_cells_area - 1.0D0
  CALL RANDOM_NUMBER(p_patch_edges_tangent_orientation);   p_patch_edges_tangent_orientation = 2.0D0 * p_patch_edges_tangent_orientation - 1.0D0
  CALL RANDOM_NUMBER(p_patch_edges_inv_primal_edge_length);p_patch_edges_inv_primal_edge_length = 2.0D0 * p_patch_edges_inv_primal_edge_length - 1.0D0
  CALL RANDOM_NUMBER(p_patch_edges_inv_dual_edge_length);  p_patch_edges_inv_dual_edge_length = 2.0D0 * p_patch_edges_inv_dual_edge_length - 1.0D0
  CALL RANDOM_NUMBER(p_patch_edges_area_edge);             p_patch_edges_area_edge = 2.0D0 * p_patch_edges_area_edge - 1.0D0
  CALL RANDOM_NUMBER(p_patch_edges_f_e);                   p_patch_edges_f_e = 2.0D0 * p_patch_edges_f_e - 1.0D0
  CALL RANDOM_NUMBER(p_patch_edges_fn_e);                  p_patch_edges_fn_e = 2.0D0 * p_patch_edges_fn_e - 1.0D0
  CALL RANDOM_NUMBER(p_patch_edges_ft_e);                  p_patch_edges_ft_e = 2.0D0 * p_patch_edges_ft_e - 1.0D0
  CALL RANDOM_NUMBER(p_int_c_lin_e);                       p_int_c_lin_e = 2.0D0 * p_int_c_lin_e - 1.0D0
  CALL RANDOM_NUMBER(p_int_e_bln_c_s);                     p_int_e_bln_c_s = 2.0D0 * p_int_e_bln_c_s - 1.0D0
  CALL RANDOM_NUMBER(p_int_cells_aw_verts);                p_int_cells_aw_verts = 2.0D0 * p_int_cells_aw_verts - 1.0D0
  CALL RANDOM_NUMBER(p_int_rbf_vec_coeff_e);               p_int_rbf_vec_coeff_e = 2.0D0 * p_int_rbf_vec_coeff_e - 1.0D0
  CALL RANDOM_NUMBER(p_int_geofac_grdiv);                  p_int_geofac_grdiv = 2.0D0 * p_int_geofac_grdiv - 1.0D0
  CALL RANDOM_NUMBER(p_int_geofac_rot);                    p_int_geofac_rot = 2.0D0 * p_int_geofac_rot - 1.0D0
  CALL RANDOM_NUMBER(p_int_geofac_n2s);                    p_int_geofac_n2s = 2.0D0 * p_int_geofac_n2s - 1.0D0
  CALL RANDOM_NUMBER(p_prog_w);                            p_prog_w = 2.0D0 * p_prog_w - 1.0D0
  CALL RANDOM_NUMBER(p_prog_vn);                           p_prog_vn = 2.0D0 * p_prog_vn - 1.0D0
  CALL RANDOM_NUMBER(p_diag_vn_ie_ubc);                    p_diag_vn_ie_ubc = 2.0D0 * p_diag_vn_ie_ubc - 1.0D0
  CALL RANDOM_NUMBER(p_diag_vt);                           p_diag_vt = 2.0D0 * p_diag_vt - 1.0D0
  CALL RANDOM_NUMBER(p_diag_vn_ie);                        p_diag_vn_ie = 2.0D0 * p_diag_vn_ie - 1.0D0
  CALL RANDOM_NUMBER(p_diag_w_concorr_c);                  p_diag_w_concorr_c = 2.0D0 * p_diag_w_concorr_c - 1.0D0
  CALL RANDOM_NUMBER(p_diag_ddt_vn_apc_pc);                p_diag_ddt_vn_apc_pc = 2.0D0 * p_diag_ddt_vn_apc_pc - 1.0D0
  CALL RANDOM_NUMBER(p_diag_ddt_vn_cor_pc);                p_diag_ddt_vn_cor_pc = 2.0D0 * p_diag_ddt_vn_cor_pc - 1.0D0
  CALL RANDOM_NUMBER(p_diag_ddt_w_adv_pc);                 p_diag_ddt_w_adv_pc = 2.0D0 * p_diag_ddt_w_adv_pc - 1.0D0
  CALL RANDOM_NUMBER(p_metrics_ddxn_z_full);               p_metrics_ddxn_z_full = 2.0D0 * p_metrics_ddxn_z_full - 1.0D0
  CALL RANDOM_NUMBER(p_metrics_ddxt_z_full);               p_metrics_ddxt_z_full = 2.0D0 * p_metrics_ddxt_z_full - 1.0D0
  CALL RANDOM_NUMBER(p_metrics_ddqz_z_full_e);             p_metrics_ddqz_z_full_e = 2.0D0 * p_metrics_ddqz_z_full_e - 1.0D0
  CALL RANDOM_NUMBER(p_metrics_ddqz_z_half);               p_metrics_ddqz_z_half = 2.0D0 * p_metrics_ddqz_z_half - 1.0D0
  CALL RANDOM_NUMBER(p_metrics_wgtfac_c);                  p_metrics_wgtfac_c = 2.0D0 * p_metrics_wgtfac_c - 1.0D0
  CALL RANDOM_NUMBER(p_metrics_wgtfac_e);                  p_metrics_wgtfac_e = 2.0D0 * p_metrics_wgtfac_e - 1.0D0
  CALL RANDOM_NUMBER(p_metrics_wgtfacq_e);                 p_metrics_wgtfacq_e = 2.0D0 * p_metrics_wgtfacq_e - 1.0D0
  CALL RANDOM_NUMBER(p_metrics_coeff_gradekin);            p_metrics_coeff_gradekin = 2.0D0 * p_metrics_coeff_gradekin - 1.0D0
  CALL RANDOM_NUMBER(p_metrics_coeff1_dwdz);               p_metrics_coeff1_dwdz = 2.0D0 * p_metrics_coeff1_dwdz - 1.0D0
  CALL RANDOM_NUMBER(p_metrics_coeff2_dwdz);               p_metrics_coeff2_dwdz = 2.0D0 * p_metrics_coeff2_dwdz - 1.0D0
  CALL RANDOM_NUMBER(p_metrics_deepatmo_gradh_mc);         p_metrics_deepatmo_gradh_mc = 2.0D0 * p_metrics_deepatmo_gradh_mc - 1.0D0
  CALL RANDOM_NUMBER(p_metrics_deepatmo_invr_mc);          p_metrics_deepatmo_invr_mc = 2.0D0 * p_metrics_deepatmo_invr_mc - 1.0D0
  CALL RANDOM_NUMBER(p_metrics_deepatmo_gradh_ifc);        p_metrics_deepatmo_gradh_ifc = 2.0D0 * p_metrics_deepatmo_gradh_ifc - 1.0D0
  CALL RANDOM_NUMBER(p_metrics_deepatmo_invr_ifc);         p_metrics_deepatmo_invr_ifc = 2.0D0 * p_metrics_deepatmo_invr_ifc - 1.0D0

  CALL fill_idx_3d(p_patch_cells_neighbor_idx, nproma)
  CALL fill_idx_3d(p_patch_cells_neighbor_blk, nblks_c)
  CALL fill_idx_3d(p_patch_cells_edge_idx,     nproma)
  CALL fill_idx_3d(p_patch_cells_edge_blk,     nblks_e)
  CALL fill_idx_3d(p_patch_edges_cell_idx,     nproma)
  CALL fill_idx_3d(p_patch_edges_cell_blk,     nblks_c)
  CALL fill_idx_3d(p_patch_edges_vertex_idx,   nproma)
  CALL fill_idx_3d(p_patch_edges_vertex_blk,   nblks_v)
  CALL fill_idx_3d(p_patch_edges_quad_idx,     nproma)
  CALL fill_idx_3d(p_patch_edges_quad_blk,     nblks_e)
  CALL fill_idx_3d(p_patch_verts_cell_idx,     nproma)
  CALL fill_idx_3d(p_patch_verts_cell_blk,     nblks_c)
  CALL fill_idx_3d(p_patch_verts_edge_idx,     nproma)
  CALL fill_idx_3d(p_patch_verts_edge_blk,     nblks_e)

  ! 33-element buffers map to allocatable bounds (-16:16); kernel
  ! accesses indices in roughly [-10, 7], all valid -> first block.
  p_patch_cells_start_index = 1;       p_patch_cells_end_index = nproma
  p_patch_cells_start_block = 1;       p_patch_cells_end_block = nblks_c
  p_patch_edges_start_index = 1;       p_patch_edges_end_index = nproma
  p_patch_edges_start_block = 1;       p_patch_edges_end_block = nblks_e
  p_patch_verts_start_index = 1;       p_patch_verts_end_index = nproma
  p_patch_verts_start_block = 1;       p_patch_verts_end_block = nblks_v
  p_patch_cells_decomp_info_owner_mask = 1_c_int8_t

CONTAINS

  SUBROUTINE fill_idx_3d(arr, hi)
    INTEGER(c_int), INTENT(OUT) :: arr(:, :, :)
    INTEGER(c_int), INTENT(IN)  :: hi
    INTEGER :: i, j, k
    DO k = 1, SIZE(arr, 3)
      DO j = 1, SIZE(arr, 2)
        DO i = 1, SIZE(arr, 1)
          arr(i, j, k) = MOD((i - 1) * 7 + j * 3 + k, hi) + 1
        END DO
      END DO
    END DO
  END SUBROUTINE fill_idx_3d

END SUBROUTINE init_inputs_random_c


SUBROUTINE run_velocity_flat_c(nproma, nlev, nlevp1, nblks_c, nblks_e, nblks_v, &
                               ntnd, istep, lvn_only, ldeepatmo, &
                               dtime, dt_linintp_ubc, &
                               nrdmax_in, nflatlev_in, &
                               lvert_nest_in, lextra_diffu_in, timers_level_in, &
                               p_patch_cells_area, &
                               p_patch_cells_neighbor_idx, p_patch_cells_neighbor_blk, &
                               p_patch_cells_edge_idx, p_patch_cells_edge_blk, &
                               p_patch_cells_start_index, p_patch_cells_end_index, &
                               p_patch_cells_start_block, p_patch_cells_end_block, &
                               p_patch_cells_decomp_info_owner_mask, &
                               p_patch_edges_cell_idx, p_patch_edges_cell_blk, &
                               p_patch_edges_vertex_idx, p_patch_edges_vertex_blk, &
                               p_patch_edges_quad_idx, p_patch_edges_quad_blk, &
                               p_patch_edges_tangent_orientation, p_patch_edges_inv_primal_edge_length, &
                               p_patch_edges_inv_dual_edge_length, p_patch_edges_area_edge, &
                               p_patch_edges_f_e, p_patch_edges_fn_e, p_patch_edges_ft_e, &
                               p_patch_edges_start_index, p_patch_edges_end_index, &
                               p_patch_edges_start_block, p_patch_edges_end_block, &
                               p_patch_verts_cell_idx, p_patch_verts_cell_blk, &
                               p_patch_verts_edge_idx, p_patch_verts_edge_blk, &
                               p_patch_verts_start_index, p_patch_verts_end_index, &
                               p_patch_verts_start_block, p_patch_verts_end_block, &
                               p_int_c_lin_e, p_int_e_bln_c_s, p_int_cells_aw_verts, &
                               p_int_rbf_vec_coeff_e, p_int_geofac_grdiv, p_int_geofac_rot, p_int_geofac_n2s, &
                               p_prog_w, p_prog_vn, &
                               p_diag_vn_ie_ubc, p_diag_vt, p_diag_vn_ie, p_diag_w_concorr_c, &
                               p_diag_ddt_vn_apc_pc, p_diag_ddt_vn_cor_pc, p_diag_ddt_w_adv_pc, &
                               p_metrics_ddxn_z_full, p_metrics_ddxt_z_full, &
                               p_metrics_ddqz_z_full_e, p_metrics_ddqz_z_half, &
                               p_metrics_wgtfac_c, p_metrics_wgtfac_e, p_metrics_wgtfacq_e, &
                               p_metrics_coeff_gradekin, p_metrics_coeff1_dwdz, p_metrics_coeff2_dwdz, &
                               p_metrics_deepatmo_gradh_mc, p_metrics_deepatmo_invr_mc, &
                               p_metrics_deepatmo_gradh_ifc, p_metrics_deepatmo_invr_ifc, &
                               z_w_concorr_me, z_kin_hor_e, z_vt_ie) &
    BIND(C, NAME="run_velocity_flat_c")
  USE iso_c_binding
  USE mo_decomposition_tools, ONLY: t_grid_domain_decomp_info
  USE mo_model_domain,        ONLY: t_grid_cells, t_grid_edges, t_grid_vertices, t_patch
  USE mo_nonhydro_types,      ONLY: t_nh_prog, t_nh_diag, t_nh_metrics
  USE mo_intp_data_strc,      ONLY: t_int_state
  USE mo_velocity_advection,  ONLY: velocity_tendencies
  USE mo_vertical_grid,       ONLY: nrdmax
  USE mo_init_vgrid,          ONLY: nflatlev
  USE mo_run_config,          ONLY: lvert_nest, timers_level
  USE mo_nonhydrostatic_config, ONLY: lextra_diffu
  ! Module-global ``nproma`` clashes with the caller's value-passed
  ! arg of the same name -- rename so we can set it explicitly.
  USE mo_parallel_config,     ONLY: g_nproma => nproma
  USE mo_timer,               ONLY: timer_solve_nh_veltend, timer_intp
  IMPLICIT NONE
  INTEGER(c_int),    VALUE :: nproma, nlev, nlevp1, nblks_c, nblks_e, nblks_v
  INTEGER(c_int),    VALUE :: ntnd, istep
  INTEGER(c_int8_t), VALUE :: lvn_only, ldeepatmo
  REAL(c_double),    VALUE :: dtime, dt_linintp_ubc
  INTEGER(c_int),    INTENT(IN)    :: nrdmax_in(10), nflatlev_in(10)
  INTEGER(c_int8_t), VALUE :: lvert_nest_in, lextra_diffu_in
  INTEGER(c_int),    VALUE :: timers_level_in
  REAL(c_double),  TARGET, INTENT(IN)    :: p_patch_cells_area(nproma, nblks_c)
  INTEGER(c_int),          INTENT(IN)    :: p_patch_cells_neighbor_idx(nproma, nblks_c, 3)
  INTEGER(c_int),          INTENT(IN)    :: p_patch_cells_neighbor_blk(nproma, nblks_c, 3)
  INTEGER(c_int),          INTENT(IN)    :: p_patch_cells_edge_idx(nproma, nblks_c, 3)
  INTEGER(c_int),          INTENT(IN)    :: p_patch_cells_edge_blk(nproma, nblks_c, 3)
  INTEGER(c_int),          INTENT(IN)    :: p_patch_cells_start_index(33)
  INTEGER(c_int),          INTENT(IN)    :: p_patch_cells_end_index(33)
  INTEGER(c_int),          INTENT(IN)    :: p_patch_cells_start_block(33)
  INTEGER(c_int),          INTENT(IN)    :: p_patch_cells_end_block(33)
  INTEGER(c_int8_t),       INTENT(IN)    :: p_patch_cells_decomp_info_owner_mask(nproma, nblks_c)
  INTEGER(c_int),          INTENT(IN)    :: p_patch_edges_cell_idx(nproma, nblks_e, 2)
  INTEGER(c_int),          INTENT(IN)    :: p_patch_edges_cell_blk(nproma, nblks_e, 2)
  INTEGER(c_int),          INTENT(IN)    :: p_patch_edges_vertex_idx(nproma, nblks_e, 4)
  INTEGER(c_int),          INTENT(IN)    :: p_patch_edges_vertex_blk(nproma, nblks_e, 4)
  INTEGER(c_int),          INTENT(IN)    :: p_patch_edges_quad_idx(nproma, nblks_e, 4)
  INTEGER(c_int),          INTENT(IN)    :: p_patch_edges_quad_blk(nproma, nblks_e, 4)
  REAL(c_double),          INTENT(IN)    :: p_patch_edges_tangent_orientation(nproma, nblks_e)
  REAL(c_double),          INTENT(IN)    :: p_patch_edges_inv_primal_edge_length(nproma, nblks_e)
  REAL(c_double),          INTENT(IN)    :: p_patch_edges_inv_dual_edge_length(nproma, nblks_e)
  REAL(c_double),          INTENT(IN)    :: p_patch_edges_area_edge(nproma, nblks_e)
  REAL(c_double),          INTENT(IN)    :: p_patch_edges_f_e(nproma, nblks_e)
  REAL(c_double),          INTENT(IN)    :: p_patch_edges_fn_e(nproma, nblks_e)
  REAL(c_double),          INTENT(IN)    :: p_patch_edges_ft_e(nproma, nblks_e)
  INTEGER(c_int),          INTENT(IN)    :: p_patch_edges_start_index(33)
  INTEGER(c_int),          INTENT(IN)    :: p_patch_edges_end_index(33)
  INTEGER(c_int),          INTENT(IN)    :: p_patch_edges_start_block(33)
  INTEGER(c_int),          INTENT(IN)    :: p_patch_edges_end_block(33)
  INTEGER(c_int),          INTENT(IN)    :: p_patch_verts_cell_idx(nproma, nblks_v, 6)
  INTEGER(c_int),          INTENT(IN)    :: p_patch_verts_cell_blk(nproma, nblks_v, 6)
  INTEGER(c_int),          INTENT(IN)    :: p_patch_verts_edge_idx(nproma, nblks_v, 6)
  INTEGER(c_int),          INTENT(IN)    :: p_patch_verts_edge_blk(nproma, nblks_v, 6)
  INTEGER(c_int),          INTENT(IN)    :: p_patch_verts_start_index(33)
  INTEGER(c_int),          INTENT(IN)    :: p_patch_verts_end_index(33)
  INTEGER(c_int),          INTENT(IN)    :: p_patch_verts_start_block(33)
  INTEGER(c_int),          INTENT(IN)    :: p_patch_verts_end_block(33)
  REAL(c_double),          INTENT(IN)    :: p_int_c_lin_e(nproma, 2, nblks_e)
  REAL(c_double),          INTENT(IN)    :: p_int_e_bln_c_s(nproma, 3, nblks_c)
  REAL(c_double),          INTENT(IN)    :: p_int_cells_aw_verts(nproma, 6, nblks_v)
  REAL(c_double),          INTENT(IN)    :: p_int_rbf_vec_coeff_e(4, nproma, nblks_e)
  REAL(c_double),          INTENT(IN)    :: p_int_geofac_grdiv(nproma, 5, nblks_e)
  REAL(c_double),          INTENT(IN)    :: p_int_geofac_rot(nproma, 6, nblks_v)
  REAL(c_double),          INTENT(IN)    :: p_int_geofac_n2s(nproma, 4, nblks_c)
  REAL(c_double),  TARGET, INTENT(INOUT) :: p_prog_w(nproma, nlevp1, nblks_c)
  REAL(c_double),  TARGET, INTENT(INOUT) :: p_prog_vn(nproma, nlev, nblks_e)
  REAL(c_double),  TARGET, INTENT(INOUT) :: p_diag_vn_ie_ubc(nproma, 2, nblks_e)
  REAL(c_double),  TARGET, INTENT(INOUT) :: p_diag_vt(nproma, nlev, nblks_e)
  REAL(c_double),  TARGET, INTENT(INOUT) :: p_diag_vn_ie(nproma, nlevp1, nblks_e)
  REAL(c_double),  TARGET, INTENT(INOUT) :: p_diag_w_concorr_c(nproma, nlev, nblks_c)
  REAL(c_double),  TARGET, INTENT(INOUT) :: p_diag_ddt_vn_apc_pc(nproma, nlev, nblks_e, 3)
  REAL(c_double),  TARGET, INTENT(INOUT) :: p_diag_ddt_vn_cor_pc(nproma, nlev, nblks_e, 3)
  REAL(c_double),  TARGET, INTENT(INOUT) :: p_diag_ddt_w_adv_pc(nproma, nlevp1, nblks_c, 3)
  REAL(c_double),  TARGET, INTENT(IN)    :: p_metrics_ddxn_z_full(nproma, nlev, nblks_e)
  REAL(c_double),  TARGET, INTENT(IN)    :: p_metrics_ddxt_z_full(nproma, nlev, nblks_e)
  REAL(c_double),  TARGET, INTENT(IN)    :: p_metrics_ddqz_z_full_e(nproma, nlev, nblks_e)
  REAL(c_double),  TARGET, INTENT(IN)    :: p_metrics_ddqz_z_half(nproma, nlevp1, nblks_c)
  REAL(c_double),  TARGET, INTENT(IN)    :: p_metrics_wgtfac_c(nproma, nlevp1, nblks_c)
  REAL(c_double),  TARGET, INTENT(IN)    :: p_metrics_wgtfac_e(nproma, nlevp1, nblks_e)
  REAL(c_double),  TARGET, INTENT(IN)    :: p_metrics_wgtfacq_e(nproma, 3, nblks_e)
  REAL(c_double),  TARGET, INTENT(IN)    :: p_metrics_coeff_gradekin(nproma, 2, nblks_e)
  REAL(c_double),  TARGET, INTENT(IN)    :: p_metrics_coeff1_dwdz(nproma, nlev, nblks_c)
  REAL(c_double),  TARGET, INTENT(IN)    :: p_metrics_coeff2_dwdz(nproma, nlev, nblks_c)
  REAL(c_double),  TARGET, INTENT(IN)    :: p_metrics_deepatmo_gradh_mc(nlev)
  REAL(c_double),  TARGET, INTENT(IN)    :: p_metrics_deepatmo_invr_mc(nlev)
  REAL(c_double),  TARGET, INTENT(IN)    :: p_metrics_deepatmo_gradh_ifc(nlevp1)
  REAL(c_double),  TARGET, INTENT(IN)    :: p_metrics_deepatmo_invr_ifc(nlevp1)
  REAL(c_double),          INTENT(INOUT) :: z_w_concorr_me(nproma, nlev, nblks_e)
  REAL(c_double),          INTENT(INOUT) :: z_kin_hor_e(nproma, nlev, nblks_e)
  REAL(c_double),          INTENT(INOUT) :: z_vt_ie(nproma, nlevp1, nblks_e)

  TYPE(t_patch), TARGET :: p_patch
  TYPE(t_int_state)     :: p_int
  TYPE(t_nh_prog)       :: p_prog
  TYPE(t_nh_diag)       :: p_diag
  TYPE(t_nh_metrics)    :: p_metrics

  nrdmax       = nrdmax_in
  nflatlev     = nflatlev_in
  lvert_nest   = (lvert_nest_in   /= 0_c_int8_t)
  lextra_diffu = (lextra_diffu_in /= 0_c_int8_t)
  timers_level = timers_level_in
  g_nproma     = nproma
  timer_solve_nh_veltend = 0
  timer_intp             = 0

  p_patch % id      = 1
  p_patch % nblks_c = nblks_c
  p_patch % nblks_e = nblks_e
  p_patch % nblks_v = nblks_v
  p_patch % nlev    = nlev
  p_patch % nlevp1  = nlevp1
  p_patch % nshift  = 0

  ALLOCATE(p_patch % cells % neighbor_idx(nproma, nblks_c, 3))
  ALLOCATE(p_patch % cells % neighbor_blk(nproma, nblks_c, 3))
  ALLOCATE(p_patch % cells % edge_idx(nproma, nblks_c, 3))
  ALLOCATE(p_patch % cells % edge_blk(nproma, nblks_c, 3))
  ! Refined-cell-tag indexing: velocity_tendencies uses indices in
  ! roughly [-10, 7] (ICON convention -- min_rlcell_int = -10,
  ! grf_bdywidth_c = 4, etc.).  Allocate with bounds (-16:16) so
  ! every access lands in bounds.  Buffer dummies carry 33 elements;
  ! buffer(i) maps to allocatable(i - 17), i.e. allocatable(-16) =
  ! buffer(1), allocatable(0) = buffer(17), allocatable(16) =
  ! buffer(33).
  ALLOCATE(p_patch % cells % start_index(-16:16))
  ALLOCATE(p_patch % cells % end_index(-16:16))
  ALLOCATE(p_patch % cells % start_block(-16:16))
  ALLOCATE(p_patch % cells % end_block(-16:16))
  ALLOCATE(p_patch % cells % decomp_info % owner_mask(nproma, nblks_c))
  p_patch % cells % neighbor_idx = p_patch_cells_neighbor_idx
  p_patch % cells % neighbor_blk = p_patch_cells_neighbor_blk
  p_patch % cells % edge_idx     = p_patch_cells_edge_idx
  p_patch % cells % edge_blk     = p_patch_cells_edge_blk
  p_patch % cells % start_index  = p_patch_cells_start_index
  p_patch % cells % end_index    = p_patch_cells_end_index
  p_patch % cells % start_block  = p_patch_cells_start_block
  p_patch % cells % end_block    = p_patch_cells_end_block
  p_patch % cells % decomp_info % owner_mask = (p_patch_cells_decomp_info_owner_mask /= 0_c_int8_t)
  p_patch % cells % area => p_patch_cells_area

  ALLOCATE(p_patch % edges % cell_idx(nproma, nblks_e, 2))
  ALLOCATE(p_patch % edges % cell_blk(nproma, nblks_e, 2))
  ALLOCATE(p_patch % edges % vertex_idx(nproma, nblks_e, 4))
  ALLOCATE(p_patch % edges % vertex_blk(nproma, nblks_e, 4))
  ALLOCATE(p_patch % edges % quad_idx(nproma, nblks_e, 4))
  ALLOCATE(p_patch % edges % quad_blk(nproma, nblks_e, 4))
  ALLOCATE(p_patch % edges % tangent_orientation(nproma, nblks_e))
  ALLOCATE(p_patch % edges % inv_primal_edge_length(nproma, nblks_e))
  ALLOCATE(p_patch % edges % inv_dual_edge_length(nproma, nblks_e))
  ALLOCATE(p_patch % edges % area_edge(nproma, nblks_e))
  ALLOCATE(p_patch % edges % f_e(nproma, nblks_e))
  ALLOCATE(p_patch % edges % fn_e(nproma, nblks_e))
  ALLOCATE(p_patch % edges % ft_e(nproma, nblks_e))
  ALLOCATE(p_patch % edges % start_index(-16:16))
  ALLOCATE(p_patch % edges % end_index(-16:16))
  ALLOCATE(p_patch % edges % start_block(-16:16))
  ALLOCATE(p_patch % edges % end_block(-16:16))
  p_patch % edges % cell_idx               = p_patch_edges_cell_idx
  p_patch % edges % cell_blk               = p_patch_edges_cell_blk
  p_patch % edges % vertex_idx             = p_patch_edges_vertex_idx
  p_patch % edges % vertex_blk             = p_patch_edges_vertex_blk
  p_patch % edges % quad_idx               = p_patch_edges_quad_idx
  p_patch % edges % quad_blk               = p_patch_edges_quad_blk
  p_patch % edges % tangent_orientation    = p_patch_edges_tangent_orientation
  p_patch % edges % inv_primal_edge_length = p_patch_edges_inv_primal_edge_length
  p_patch % edges % inv_dual_edge_length   = p_patch_edges_inv_dual_edge_length
  p_patch % edges % area_edge              = p_patch_edges_area_edge
  p_patch % edges % f_e                    = p_patch_edges_f_e
  p_patch % edges % fn_e                   = p_patch_edges_fn_e
  p_patch % edges % ft_e                   = p_patch_edges_ft_e
  p_patch % edges % start_index            = p_patch_edges_start_index
  p_patch % edges % end_index              = p_patch_edges_end_index
  p_patch % edges % start_block            = p_patch_edges_start_block
  p_patch % edges % end_block              = p_patch_edges_end_block

  ALLOCATE(p_patch % verts % cell_idx(nproma, nblks_v, 6))
  ALLOCATE(p_patch % verts % cell_blk(nproma, nblks_v, 6))
  ALLOCATE(p_patch % verts % edge_idx(nproma, nblks_v, 6))
  ALLOCATE(p_patch % verts % edge_blk(nproma, nblks_v, 6))
  ALLOCATE(p_patch % verts % start_index(-16:16))
  ALLOCATE(p_patch % verts % end_index(-16:16))
  ALLOCATE(p_patch % verts % start_block(-16:16))
  ALLOCATE(p_patch % verts % end_block(-16:16))
  p_patch % verts % cell_idx     = p_patch_verts_cell_idx
  p_patch % verts % cell_blk     = p_patch_verts_cell_blk
  p_patch % verts % edge_idx     = p_patch_verts_edge_idx
  p_patch % verts % edge_blk     = p_patch_verts_edge_blk
  p_patch % verts % start_index  = p_patch_verts_start_index
  p_patch % verts % end_index    = p_patch_verts_end_index
  p_patch % verts % start_block  = p_patch_verts_start_block
  p_patch % verts % end_block    = p_patch_verts_end_block

  ALLOCATE(p_int % c_lin_e(nproma, 2, nblks_e))
  ALLOCATE(p_int % e_bln_c_s(nproma, 3, nblks_c))
  ALLOCATE(p_int % cells_aw_verts(nproma, 6, nblks_v))
  ALLOCATE(p_int % rbf_vec_coeff_e(4, nproma, nblks_e))
  ALLOCATE(p_int % geofac_grdiv(nproma, 5, nblks_e))
  ALLOCATE(p_int % geofac_rot(nproma, 6, nblks_v))
  ALLOCATE(p_int % geofac_n2s(nproma, 4, nblks_c))
  p_int % c_lin_e          = p_int_c_lin_e
  p_int % e_bln_c_s        = p_int_e_bln_c_s
  p_int % cells_aw_verts   = p_int_cells_aw_verts
  p_int % rbf_vec_coeff_e  = p_int_rbf_vec_coeff_e
  p_int % geofac_grdiv     = p_int_geofac_grdiv
  p_int % geofac_rot       = p_int_geofac_rot
  p_int % geofac_n2s       = p_int_geofac_n2s

  p_prog % w  => p_prog_w
  p_prog % vn => p_prog_vn

  p_diag % vn_ie_ubc       => p_diag_vn_ie_ubc
  p_diag % vt              => p_diag_vt
  p_diag % vn_ie           => p_diag_vn_ie
  p_diag % w_concorr_c     => p_diag_w_concorr_c
  p_diag % ddt_vn_apc_pc   => p_diag_ddt_vn_apc_pc
  p_diag % ddt_vn_cor_pc   => p_diag_ddt_vn_cor_pc
  p_diag % ddt_w_adv_pc    => p_diag_ddt_w_adv_pc

  p_metrics % ddxn_z_full        => p_metrics_ddxn_z_full
  p_metrics % ddxt_z_full        => p_metrics_ddxt_z_full
  p_metrics % ddqz_z_full_e      => p_metrics_ddqz_z_full_e
  p_metrics % ddqz_z_half        => p_metrics_ddqz_z_half
  p_metrics % wgtfac_c           => p_metrics_wgtfac_c
  p_metrics % wgtfac_e           => p_metrics_wgtfac_e
  p_metrics % wgtfacq_e          => p_metrics_wgtfacq_e
  p_metrics % coeff_gradekin     => p_metrics_coeff_gradekin
  p_metrics % coeff1_dwdz        => p_metrics_coeff1_dwdz
  p_metrics % coeff2_dwdz        => p_metrics_coeff2_dwdz
  p_metrics % deepatmo_gradh_mc  => p_metrics_deepatmo_gradh_mc
  p_metrics % deepatmo_invr_mc   => p_metrics_deepatmo_invr_mc
  p_metrics % deepatmo_gradh_ifc => p_metrics_deepatmo_gradh_ifc
  p_metrics % deepatmo_invr_ifc  => p_metrics_deepatmo_invr_ifc

  CALL velocity_tendencies(p_prog, p_patch, p_int, p_metrics, p_diag, &
                           z_w_concorr_me, z_kin_hor_e, z_vt_ie, &
                           ntnd, istep, &
                           (lvn_only /= 0_c_int8_t), &
                           dtime, dt_linintp_ubc, &
                           (ldeepatmo /= 0_c_int8_t))
END SUBROUTINE run_velocity_flat_c
