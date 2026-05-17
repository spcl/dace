MODULE mo_decomposition_tools
  IMPLICIT NONE
  TYPE :: t_grid_domain_decomp_info
    LOGICAL, ALLOCATABLE :: owner_mask(:, :)
  END TYPE
  CONTAINS
END MODULE mo_decomposition_tools
MODULE mo_fortran_tools
  IMPLICIT NONE
  CONTAINS
  PURE SUBROUTINE set_acc_host_or_device(lzacc, lacc)
    LOGICAL, INTENT(OUT) :: lzacc
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    lzacc = .FALSE.
  END SUBROUTINE set_acc_host_or_device
END MODULE mo_fortran_tools
MODULE mo_init_vgrid
  IMPLICIT NONE
  INTEGER :: nflatlev(10)
  CONTAINS
END MODULE mo_init_vgrid
MODULE mo_intp_data_strc
  IMPLICIT NONE
  TYPE :: t_int_state
    REAL(KIND = 8), ALLOCATABLE :: c_lin_e(:, :, :)
    REAL(KIND = 8), ALLOCATABLE :: e_bln_c_s(:, :, :)
    REAL(KIND = 8), ALLOCATABLE :: cells_aw_verts(:, :, :)
    REAL(KIND = 8), ALLOCATABLE :: rbf_vec_coeff_e(:, :, :)
    REAL(KIND = 8), ALLOCATABLE :: geofac_grdiv(:, :, :)
    REAL(KIND = 8), ALLOCATABLE :: geofac_rot(:, :, :)
    REAL(KIND = 8), ALLOCATABLE :: geofac_n2s(:, :, :)
  END TYPE t_int_state
END MODULE mo_intp_data_strc
MODULE mo_lib_loopindices
  IMPLICIT NONE
  CONTAINS
  SUBROUTINE get_indices_c_lib(i_startidx_in, i_endidx_in, nproma, i_blk, i_startblk, i_endblk, i_startidx_out, i_endidx_out)
    INTEGER, INTENT(IN) :: i_startidx_in
    INTEGER, INTENT(IN) :: i_endidx_in
    INTEGER, INTENT(IN) :: nproma
    INTEGER, INTENT(IN) :: i_blk
    INTEGER, INTENT(IN) :: i_startblk
    INTEGER, INTENT(IN) :: i_endblk
    INTEGER, INTENT(OUT) :: i_startidx_out, i_endidx_out
    IF (i_blk == i_startblk) THEN
      i_startidx_out = MAX(1, i_startidx_in)
      i_endidx_out = nproma
      IF (i_blk == i_endblk) i_endidx_out = i_endidx_in
    ELSE IF (i_blk == i_endblk) THEN
      i_startidx_out = 1
      i_endidx_out = i_endidx_in
    ELSE
      i_startidx_out = 1
      i_endidx_out = nproma
    END IF
  END SUBROUTINE get_indices_c_lib
  SUBROUTINE get_indices_e_lib(i_startidx_in, i_endidx_in, nproma, i_blk, i_startblk, i_endblk, i_startidx_out, i_endidx_out)
    INTEGER, INTENT(IN) :: i_startidx_in
    INTEGER, INTENT(IN) :: i_endidx_in
    INTEGER, INTENT(IN) :: nproma
    INTEGER, INTENT(IN) :: i_blk
    INTEGER, INTENT(IN) :: i_startblk
    INTEGER, INTENT(IN) :: i_endblk
    INTEGER, INTENT(OUT) :: i_startidx_out, i_endidx_out
    i_startidx_out = MERGE(1, MAX(1, i_startidx_in), i_blk /= i_startblk)
    i_endidx_out = MERGE(nproma, i_endidx_in, i_blk /= i_endblk)
  END SUBROUTINE get_indices_e_lib
  SUBROUTINE get_indices_v_lib(i_startidx_in, i_endidx_in, nproma, i_blk, i_startblk, i_endblk, i_startidx_out, i_endidx_out)
    INTEGER, INTENT(IN) :: i_startidx_in
    INTEGER, INTENT(IN) :: i_endidx_in
    INTEGER, INTENT(IN) :: nproma
    INTEGER, INTENT(IN) :: i_blk
    INTEGER, INTENT(IN) :: i_startblk
    INTEGER, INTENT(IN) :: i_endblk
    INTEGER, INTENT(OUT) :: i_startidx_out, i_endidx_out
    IF (i_blk == i_startblk) THEN
      i_startidx_out = i_startidx_in
      i_endidx_out = nproma
      IF (i_blk == i_endblk) i_endidx_out = i_endidx_in
    ELSE IF (i_blk == i_endblk) THEN
      i_startidx_out = 1
      i_endidx_out = i_endidx_in
    ELSE
      i_startidx_out = 1
      i_endidx_out = nproma
    END IF
  END SUBROUTINE get_indices_v_lib
END MODULE mo_lib_loopindices
MODULE mo_lib_interpolation_scalar
  IMPLICIT NONE
  CONTAINS
  SUBROUTINE cells2verts_scalar_ri_lib(p_cell_in, vert_cell_idx, vert_cell_blk, c_int, p_vert_out, i_startblk, i_endblk, i_startidx_in, i_endidx_in, slev, elev, nproma, lacc, acc_async)
    USE mo_fortran_tools, ONLY: set_acc_host_or_device
    USE mo_lib_loopindices, ONLY: get_indices_v_lib
    REAL(KIND = 8), INTENT(IN) :: p_cell_in(:, :, :)
    INTEGER, TARGET, INTENT(IN) :: vert_cell_idx(:, :, :)
    INTEGER, TARGET, INTENT(IN) :: vert_cell_blk(:, :, :)
    REAL(KIND = 8), INTENT(IN) :: c_int(:, :, :)
    REAL(KIND = 8), INTENT(INOUT) :: p_vert_out(:, :, :)
    INTEGER, INTENT(IN) :: i_startblk
    INTEGER, INTENT(IN) :: i_endblk
    INTEGER, INTENT(IN) :: i_startidx_in
    INTEGER, INTENT(IN) :: i_endidx_in
    INTEGER, INTENT(IN) :: slev
    INTEGER, INTENT(IN) :: elev
    INTEGER, INTENT(IN) :: nproma
    LOGICAL, INTENT(IN), OPTIONAL :: lacc
    LOGICAL, INTENT(IN), OPTIONAL :: acc_async
    INTEGER :: jv, jk, jb
    INTEGER :: i_startidx, i_endidx
    LOGICAL :: lzacc
    CALL set_acc_host_or_device(lzacc, lacc)
    DO jb = i_startblk, i_endblk
      CALL get_indices_v_lib(i_startidx_in, i_endidx_in, nproma, jb, i_startblk, i_endblk, i_startidx, i_endidx)
      DO jk = 1, elev
        DO jv = i_startidx, i_endidx
          p_vert_out(jv, jk, jb) = c_int(jv, 1, jb) * p_cell_in(vert_cell_idx(jv, jb, 1), jk, vert_cell_blk(jv, jb, 1)) + c_int(jv, 2, jb) * p_cell_in(vert_cell_idx(jv, jb, 2), jk, vert_cell_blk(jv, jb, 2)) + c_int(jv, 3, jb) * p_cell_in(vert_cell_idx(jv, jb, 3), jk, vert_cell_blk(jv, jb, 3)) + c_int(jv, 4, jb) * p_cell_in(vert_cell_idx(jv, jb, 4), jk, vert_cell_blk(jv, jb, 4)) + c_int(jv, 5, jb) * p_cell_in(vert_cell_idx(jv, jb, 5), jk, vert_cell_blk(jv, jb, 5)) + c_int(jv, 6, jb) * p_cell_in(vert_cell_idx(jv, jb, 6), jk, vert_cell_blk(jv, jb, 6))
        END DO
      END DO
    END DO
  END SUBROUTINE cells2verts_scalar_ri_lib
END MODULE mo_lib_interpolation_scalar
MODULE mo_model_domain
  USE mo_decomposition_tools, ONLY: t_grid_domain_decomp_info
  IMPLICIT NONE
  TYPE :: t_grid_cells
    INTEGER, ALLOCATABLE :: neighbor_idx(:, :, :)
    INTEGER, ALLOCATABLE :: neighbor_blk(:, :, :)
    INTEGER, ALLOCATABLE :: edge_idx(:, :, :)
    INTEGER, ALLOCATABLE :: edge_blk(:, :, :)
    REAL(KIND = 8), POINTER :: area(:, :)
    INTEGER, ALLOCATABLE :: start_index(:)
    INTEGER, ALLOCATABLE :: end_index(:)
    INTEGER, ALLOCATABLE :: start_block(:)
    INTEGER, ALLOCATABLE :: end_block(:)
    TYPE(t_grid_domain_decomp_info) :: decomp_info
  END TYPE t_grid_cells
  TYPE :: t_grid_edges
    INTEGER, ALLOCATABLE :: cell_idx(:, :, :)
    INTEGER, ALLOCATABLE :: cell_blk(:, :, :)
    INTEGER, ALLOCATABLE :: vertex_idx(:, :, :)
    INTEGER, ALLOCATABLE :: vertex_blk(:, :, :)
    REAL(KIND = 8), ALLOCATABLE :: tangent_orientation(:, :)
    INTEGER, ALLOCATABLE :: quad_idx(:, :, :)
    INTEGER, ALLOCATABLE :: quad_blk(:, :, :)
    REAL(KIND = 8), ALLOCATABLE :: inv_primal_edge_length(:, :)
    REAL(KIND = 8), ALLOCATABLE :: inv_dual_edge_length(:, :)
    REAL(KIND = 8), ALLOCATABLE :: area_edge(:, :)
    REAL(KIND = 8), ALLOCATABLE :: f_e(:, :)
    REAL(KIND = 8), ALLOCATABLE :: fn_e(:, :)
    REAL(KIND = 8), ALLOCATABLE :: ft_e(:, :)
    INTEGER, ALLOCATABLE :: start_index(:)
    INTEGER, ALLOCATABLE :: end_index(:)
    INTEGER, ALLOCATABLE :: start_block(:)
    INTEGER, ALLOCATABLE :: end_block(:)
  END TYPE t_grid_edges
  TYPE :: t_grid_vertices
    INTEGER, ALLOCATABLE :: cell_idx(:, :, :)
    INTEGER, ALLOCATABLE :: cell_blk(:, :, :)
    INTEGER, ALLOCATABLE :: edge_idx(:, :, :)
    INTEGER, ALLOCATABLE :: edge_blk(:, :, :)
    INTEGER, ALLOCATABLE :: start_index(:)
    INTEGER, ALLOCATABLE :: end_index(:)
    INTEGER, ALLOCATABLE :: start_block(:)
    INTEGER, ALLOCATABLE :: end_block(:)
  END TYPE t_grid_vertices
  TYPE :: t_patch
    INTEGER :: id
    INTEGER :: nblks_c
    INTEGER :: nblks_e
    INTEGER :: nblks_v
    INTEGER :: nlev
    INTEGER :: nlevp1
    INTEGER :: nshift
    TYPE(t_grid_cells) :: cells
    TYPE(t_grid_edges) :: edges
    TYPE(t_grid_vertices) :: verts
  END TYPE t_patch
  CONTAINS
END MODULE mo_model_domain
MODULE mo_mpi
  IMPLICIT NONE
  LOGICAL, PUBLIC :: i_am_accel_node = .FALSE.
  CONTAINS
END MODULE mo_mpi
MODULE mo_nonhydro_types
  IMPLICIT NONE
  TYPE :: t_nh_prog
    REAL(KIND = 8), POINTER, CONTIGUOUS :: w(:, :, :), vn(:, :, :)
  END TYPE t_nh_prog
  TYPE :: t_nh_diag
    REAL(KIND = 8), POINTER, CONTIGUOUS :: vn_ie_ubc(:, :, :)
    REAL(KIND = 8), POINTER, CONTIGUOUS :: vt(:, :, :), vn_ie(:, :, :), w_concorr_c(:, :, :), ddt_vn_apc_pc(:, :, :, :), ddt_vn_cor_pc(:, :, :, :), ddt_w_adv_pc(:, :, :, :)
    LOGICAL :: ddt_vn_adv_is_associated = .FALSE., ddt_vn_cor_is_associated = .FALSE.
    REAL(KIND = 8) :: max_vcfl_dyn = 0.0D0
  END TYPE t_nh_diag
  TYPE :: t_nh_metrics
    REAL(KIND = 8), POINTER, CONTIGUOUS :: ddxn_z_full(:, :, :), ddxt_z_full(:, :, :), ddqz_z_full_e(:, :, :), ddqz_z_half(:, :, :), wgtfac_c(:, :, :), wgtfac_e(:, :, :), wgtfacq_e(:, :, :), coeff_gradekin(:, :, :), coeff1_dwdz(:, :, :), coeff2_dwdz(:, :, :)
    REAL(KIND = 8), POINTER, CONTIGUOUS :: deepatmo_gradh_mc(:), deepatmo_invr_mc(:), deepatmo_gradh_ifc(:), deepatmo_invr_ifc(:)
  END TYPE t_nh_metrics
END MODULE mo_nonhydro_types
MODULE mo_nonhydrostatic_config
  IMPLICIT NONE
  LOGICAL :: lextra_diffu
  CONTAINS
END MODULE mo_nonhydrostatic_config
MODULE mo_parallel_config
  IMPLICIT NONE
  INTEGER :: nproma = 0
  CONTAINS
END MODULE mo_parallel_config
MODULE mo_loopindices
  IMPLICIT NONE
  CONTAINS
  SUBROUTINE get_indices_c(p_patch, i_blk, i_startblk, i_endblk, i_startidx, i_endidx, irl_start, opt_rl_end)
    USE mo_model_domain, ONLY: t_patch
    USE mo_lib_loopindices, ONLY: get_indices_c_lib
    USE mo_parallel_config, ONLY: nproma
    TYPE(t_patch), INTENT(IN) :: p_patch
    INTEGER, INTENT(IN) :: i_blk
    INTEGER, INTENT(IN) :: i_startblk
    INTEGER, INTENT(IN) :: i_endblk
    INTEGER, INTENT(IN) :: irl_start
    INTEGER, OPTIONAL, INTENT(IN) :: opt_rl_end
    INTEGER, INTENT(OUT) :: i_startidx, i_endidx
    INTEGER :: irl_end, i_startidx_in, i_endidx_in
    i_startidx_in = p_patch % cells % start_index(irl_start)
    irl_end = opt_rl_end
    i_endidx_in = p_patch % cells % end_index(irl_end)
    CALL get_indices_c_lib(i_startidx_in, i_endidx_in, nproma, i_blk, i_startblk, i_endblk, i_startidx, i_endidx)
  END SUBROUTINE get_indices_c
  SUBROUTINE get_indices_e(p_patch, i_blk, i_startblk, i_endblk, i_startidx, i_endidx, irl_start, opt_rl_end)
    USE mo_model_domain, ONLY: t_patch
    USE mo_lib_loopindices, ONLY: get_indices_e_lib
    USE mo_parallel_config, ONLY: nproma
    TYPE(t_patch), INTENT(IN) :: p_patch
    INTEGER, INTENT(IN) :: i_blk
    INTEGER, INTENT(IN) :: i_startblk
    INTEGER, INTENT(IN) :: i_endblk
    INTEGER, INTENT(IN) :: irl_start
    INTEGER, OPTIONAL, INTENT(IN) :: opt_rl_end
    INTEGER, INTENT(OUT) :: i_startidx, i_endidx
    INTEGER :: irl_end, i_startidx_in, i_endidx_in
    i_startidx_in = p_patch % edges % start_index(irl_start)
    irl_end = opt_rl_end
    i_endidx_in = p_patch % edges % end_index(irl_end)
    CALL get_indices_e_lib(i_startidx_in, i_endidx_in, nproma, i_blk, i_startblk, i_endblk, i_startidx, i_endidx)
  END SUBROUTINE get_indices_e
  SUBROUTINE get_indices_v(p_patch, i_blk, i_startblk, i_endblk, i_startidx, i_endidx, irl_start, opt_rl_end)
    USE mo_model_domain, ONLY: t_patch
    USE mo_lib_loopindices, ONLY: get_indices_v_lib
    USE mo_parallel_config, ONLY: nproma
    TYPE(t_patch), INTENT(IN) :: p_patch
    INTEGER, INTENT(IN) :: i_blk
    INTEGER, INTENT(IN) :: i_startblk
    INTEGER, INTENT(IN) :: i_endblk
    INTEGER, INTENT(IN) :: irl_start
    INTEGER, OPTIONAL, INTENT(IN) :: opt_rl_end
    INTEGER, INTENT(OUT) :: i_startidx, i_endidx
    INTEGER :: irl_end, i_startidx_in, i_endidx_in
    i_startidx_in = p_patch % verts % start_index(2)
    irl_end = -5
    i_endidx_in = p_patch % verts % end_index(-5)
    CALL get_indices_v_lib(i_startidx_in, i_endidx_in, nproma, i_blk, i_startblk, i_endblk, i_startidx, i_endidx)
  END SUBROUTINE get_indices_v
END MODULE mo_loopindices
MODULE mo_math_divrot
  IMPLICIT NONE
  CONTAINS
  SUBROUTINE rot_vertex_ri(vec_e, ptr_patch, ptr_int, rot_vec, opt_slev, opt_elev, opt_rlend, opt_acc_async)
    USE mo_model_domain, ONLY: t_patch
    USE mo_intp_data_strc, ONLY: t_int_state
    USE mo_loopindices, ONLY: get_indices_v
    TYPE(t_patch), TARGET, INTENT(IN) :: ptr_patch
    TYPE(t_int_state), INTENT(IN) :: ptr_int
    REAL(KIND = 8), INTENT(IN) :: vec_e(:, :, :)
    INTEGER, INTENT(IN), OPTIONAL :: opt_slev
    INTEGER, INTENT(IN), OPTIONAL :: opt_elev
    INTEGER, INTENT(IN), OPTIONAL :: opt_rlend
    LOGICAL, INTENT(IN), OPTIONAL :: opt_acc_async
    REAL(KIND = 8), INTENT(INOUT) :: rot_vec(:, :, :)
    INTEGER :: slev, elev
    INTEGER :: jv, jk, jb
    INTEGER :: rl_start, rl_end
    INTEGER :: i_startblk, i_endblk, i_startidx, i_endidx
    slev = 1
    elev = UBOUND(vec_e, 2)
    rl_start = 2
    rl_end = -5
    i_startblk = ptr_patch % verts % start_block(2)
    i_endblk = ptr_patch % verts % end_block(-5)
    DO jb = i_startblk, i_endblk
      CALL get_indices_v(ptr_patch, jb, i_startblk, i_endblk, i_startidx, i_endidx, 2, -5)
      DO jk = slev, elev
        DO jv = i_startidx, i_endidx
          rot_vec(jv, jk, jb) = vec_e(ptr_patch % verts % edge_idx(jv, jb, 1), jk, ptr_patch % verts % edge_blk(jv, jb, 1)) * ptr_int % geofac_rot(jv, 1, jb) + vec_e(ptr_patch % verts % edge_idx(jv, jb, 2), jk, ptr_patch % verts % edge_blk(jv, jb, 2)) * ptr_int % geofac_rot(jv, 2, jb) + vec_e(ptr_patch % verts % edge_idx(jv, jb, 3), jk, ptr_patch % verts % edge_blk(jv, jb, 3)) * ptr_int % geofac_rot(jv, 3, jb) + vec_e(ptr_patch % verts % edge_idx(jv, jb, 4), jk, ptr_patch % verts % edge_blk(jv, jb, 4)) * ptr_int % geofac_rot(jv, 4, jb) + vec_e(ptr_patch % verts % edge_idx(jv, jb, 5), jk, ptr_patch % verts % edge_blk(jv, jb, 5)) * ptr_int % geofac_rot(jv, 5, jb) + vec_e(ptr_patch % verts % edge_idx(jv, jb, 6), jk, ptr_patch % verts % edge_blk(jv, jb, 6)) * ptr_int % geofac_rot(jv, 6, jb)
        END DO
      END DO
    END DO
  END SUBROUTINE rot_vertex_ri
END MODULE mo_math_divrot
MODULE mo_real_timer
  IMPLICIT NONE
  CONTAINS
  SUBROUTINE timer_start(it)
    INTEGER, INTENT(IN) :: it
  END SUBROUTINE timer_start
  SUBROUTINE timer_stop(it)
    INTEGER, INTENT(IN) :: it
  END SUBROUTINE timer_stop
END MODULE mo_real_timer
MODULE mo_run_config
  IMPLICIT NONE
  LOGICAL :: lvert_nest
  INTEGER :: timers_level
  CONTAINS
END MODULE mo_run_config
MODULE mo_timer
  IMPLICIT NONE
  INTEGER :: timer_solve_nh_veltend
  INTEGER :: timer_intp
  CONTAINS
END MODULE mo_timer
MODULE mo_icon_interpolation_scalar
  IMPLICIT NONE
  CONTAINS
  SUBROUTINE cells2verts_scalar_ri(p_cell_in, ptr_patch, c_int, p_vert_out, opt_slev, opt_elev, opt_rlstart, opt_rlend, opt_acc_async)
    USE mo_model_domain, ONLY: t_patch
    USE mo_run_config, ONLY: timers_level
    USE mo_real_timer, ONLY: timer_start, timer_stop
    USE mo_timer, ONLY: timer_intp
    USE mo_lib_interpolation_scalar, ONLY: cells2verts_scalar_ri_lib
    USE mo_parallel_config, ONLY: nproma
    USE mo_mpi, ONLY: i_am_accel_node
    TYPE(t_patch), TARGET, INTENT(IN) :: ptr_patch
    REAL(KIND = 8), INTENT(IN) :: p_cell_in(:, :, :)
    REAL(KIND = 8), INTENT(IN) :: c_int(:, :, :)
    INTEGER, INTENT(IN), OPTIONAL :: opt_slev
    INTEGER, INTENT(IN), OPTIONAL :: opt_elev
    INTEGER, INTENT(IN), OPTIONAL :: opt_rlstart, opt_rlend
    REAL(KIND = 8), INTENT(INOUT) :: p_vert_out(:, :, :)
    LOGICAL, INTENT(IN), OPTIONAL :: opt_acc_async
    INTEGER :: slev, elev
    INTEGER :: rl_start, rl_end
    INTEGER :: i_startblk, i_endblk, i_startidx_in, i_endidx_in
    slev = 1
    elev = UBOUND(p_cell_in, 2)
    rl_start = 2
    rl_end = -5
    i_startblk = ptr_patch % verts % start_block(2)
    i_endblk = ptr_patch % verts % end_block(-5)
    i_startidx_in = ptr_patch % verts % start_index(2)
    i_endidx_in = ptr_patch % verts % end_index(-5)
    IF (timers_level > 10) CALL timer_start(timer_intp)
    CALL cells2verts_scalar_ri_lib(p_cell_in, ptr_patch % verts % cell_idx, ptr_patch % verts % cell_blk, c_int, p_vert_out, i_startblk, i_endblk, i_startidx_in, i_endidx_in, 1, elev, nproma, lacc = i_am_accel_node, acc_async = .TRUE.)
    IF (timers_level > 10) CALL timer_stop(timer_intp)
  END SUBROUTINE cells2verts_scalar_ri
END MODULE mo_icon_interpolation_scalar
MODULE mo_vertical_grid
  IMPLICIT NONE
  INTEGER :: nrdmax(10)
  CONTAINS
END MODULE mo_vertical_grid
MODULE mo_velocity_advection
  IMPLICIT NONE
  CONTAINS
  SUBROUTINE velocity_tendencies(p_prog, p_patch, p_int, p_metrics, p_diag, z_w_concorr_me, z_kin_hor_e, z_vt_ie, ntnd, istep, lvn_only, dtime, dt_linintp_ubc, ldeepatmo)
    USE mo_model_domain, ONLY: t_patch
    USE mo_intp_data_strc, ONLY: t_int_state
    USE mo_nonhydro_types, ONLY: t_nh_diag, t_nh_metrics, t_nh_prog
    USE mo_parallel_config, ONLY: nproma
    USE mo_run_config, ONLY: lvert_nest, timers_level
    USE mo_real_timer, ONLY: timer_start, timer_stop
    USE mo_timer, ONLY: timer_solve_nh_veltend
    USE mo_vertical_grid, ONLY: nrdmax
    USE mo_init_vgrid, ONLY: nflatlev
    USE mo_nonhydrostatic_config, ONLY: lextra_diffu
    USE mo_icon_interpolation_scalar, ONLY: cells2verts_scalar_ri
    USE mo_math_divrot, ONLY: rot_vertex_ri
    USE mo_loopindices, ONLY: get_indices_c, get_indices_e
    TYPE(t_patch), TARGET, INTENT(IN) :: p_patch
    TYPE(t_int_state), TARGET, INTENT(IN) :: p_int
    TYPE(t_nh_prog), INTENT(INOUT) :: p_prog
    TYPE(t_nh_metrics), INTENT(INOUT) :: p_metrics
    TYPE(t_nh_diag), INTENT(INOUT) :: p_diag
    REAL(KIND = 8), DIMENSION(:, :, :), INTENT(INOUT) :: z_w_concorr_me, z_kin_hor_e, z_vt_ie
    INTEGER, INTENT(IN) :: ntnd
    INTEGER, INTENT(IN) :: istep
    LOGICAL, INTENT(IN) :: lvn_only
    REAL(KIND = 8), INTENT(IN) :: dtime
    REAL(KIND = 8), INTENT(IN) :: dt_linintp_ubc
    LOGICAL, INTENT(IN) :: ldeepatmo
    INTEGER :: jb, jk, jc, je
    INTEGER :: i_startblk, i_endblk, i_startidx, i_endidx
    INTEGER :: i_startblk_2, i_endblk_2, i_startidx_2, i_endidx_2
    INTEGER :: rl_start, rl_end, rl_start_2, rl_end_2
    REAL(KIND = 8) :: z_w_concorr_mc(nproma, p_patch % nlev)
    REAL(KIND = 8) :: z_w_con_c(nproma, p_patch % nlevp1)
    REAL(KIND = 8) :: z_w_con_c_full(nproma, p_patch % nlev, p_patch % nblks_c)
    REAL(KIND = 8) :: z_v_grad_w(nproma, p_patch % nlev, p_patch % nblks_e)
    REAL(KIND = 8) :: z_w_v(nproma, p_patch % nlevp1, p_patch % nblks_v)
    REAL(KIND = 8) :: zeta(nproma, p_patch % nlev, p_patch % nblks_v)
    REAL(KIND = 8) :: z_ekinh(nproma, p_patch % nlev, p_patch % nblks_c)
    INTEGER :: nlev, nlevp1
    LOGICAL :: l_vert_nested
    INTEGER :: jg
    REAL(KIND = 8) :: cfl_w_limit, vcfl, maxvcfl, vcflmax(p_patch % nblks_c)
    REAL(KIND = 8) :: w_con_e, scalfac_exdiff, difcoef, max_vcfl_dyn
    INTEGER :: ie, nrdmax_jg, nflatlev_jg, clip_count
    LOGICAL :: levmask(p_patch % nblks_c, p_patch % nlev), levelmask(p_patch % nlev)
    LOGICAL :: cfl_clipping(nproma, p_patch % nlevp1)
    IF (timers_level > 5) CALL timer_start(timer_solve_nh_veltend)
    IF ((lvert_nest) .AND. (p_patch % nshift > 0)) THEN
      l_vert_nested = .TRUE.
    ELSE
      l_vert_nested = .FALSE.
    END IF
    jg = p_patch % id
    nrdmax_jg = nrdmax(jg)
    nflatlev_jg = nflatlev(jg)
    nlev = p_patch % nlev
    nlevp1 = p_patch % nlevp1
    IF (lextra_diffu) THEN
      cfl_w_limit = 0.65D0 / dtime
      scalfac_exdiff = 0.05D0 / (dtime * (0.85D0 - cfl_w_limit * dtime))
    ELSE
      cfl_w_limit = 0.85D0 / dtime
      scalfac_exdiff = 0.0D0
    END IF
    IF (.NOT. lvn_only) CALL cells2verts_scalar_ri(p_prog % w, p_patch, p_int % cells_aw_verts, z_w_v, opt_rlend = -5, opt_acc_async = .TRUE.)
    CALL rot_vertex_ri(p_prog % vn, p_patch, p_int, zeta, opt_rlend = -5, opt_acc_async = .TRUE.)
    IF (istep == 1) THEN
      rl_start = 5
      rl_end = -10
      i_startblk = p_patch % edges % start_block(5)
      i_endblk = p_patch % edges % end_block(-10)
      DO jb = i_startblk, i_endblk
        CALL get_indices_e(p_patch, jb, i_startblk, i_endblk, i_startidx, i_endidx, 5, -10)
        DO jk = 1, nlev
          DO je = i_startidx, i_endidx
            p_diag % vt(je, jk, jb) = p_int % rbf_vec_coeff_e(1, je, jb) * p_prog % vn(p_patch % edges % quad_idx(je, jb, 1), jk, p_patch % edges % quad_blk(je, jb, 1)) + p_int % rbf_vec_coeff_e(2, je, jb) * p_prog % vn(p_patch % edges % quad_idx(je, jb, 2), jk, p_patch % edges % quad_blk(je, jb, 2)) + p_int % rbf_vec_coeff_e(3, je, jb) * p_prog % vn(p_patch % edges % quad_idx(je, jb, 3), jk, p_patch % edges % quad_blk(je, jb, 3)) + p_int % rbf_vec_coeff_e(4, je, jb) * p_prog % vn(p_patch % edges % quad_idx(je, jb, 4), jk, p_patch % edges % quad_blk(je, jb, 4))
          END DO
        END DO
        DO jk = 2, nlev
          DO je = i_startidx, i_endidx
            p_diag % vn_ie(je, jk, jb) = p_metrics % wgtfac_e(je, jk, jb) * p_prog % vn(je, jk, jb) + (1.0D0 - p_metrics % wgtfac_e(je, jk, jb)) * p_prog % vn(je, jk - 1, jb)
            z_kin_hor_e(je, jk, jb) = 0.5D0 * (p_prog % vn(je, jk, jb) ** 2 + p_diag % vt(je, jk, jb) ** 2)
          END DO
        END DO
        IF (.NOT. lvn_only) THEN
          DO jk = 2, nlev
            DO je = i_startidx, i_endidx
              z_vt_ie(je, jk, jb) = p_metrics % wgtfac_e(je, jk, jb) * p_diag % vt(je, jk, jb) + (1.0D0 - p_metrics % wgtfac_e(je, jk, jb)) * p_diag % vt(je, jk - 1, jb)
            END DO
          END DO
        END IF
        DO jk = nflatlev_jg, nlev
          DO je = i_startidx, i_endidx
            z_w_concorr_me(je, jk, jb) = p_prog % vn(je, jk, jb) * p_metrics % ddxn_z_full(je, jk, jb) + p_diag % vt(je, jk, jb) * p_metrics % ddxt_z_full(je, jk, jb)
          END DO
        END DO
        IF (.NOT. l_vert_nested) THEN
          DO je = i_startidx, i_endidx
            p_diag % vn_ie(je, 1, jb) = p_prog % vn(je, 1, jb)
            z_vt_ie(je, 1, jb) = p_diag % vt(je, 1, jb)
            z_kin_hor_e(je, 1, jb) = 0.5D0 * (p_prog % vn(je, 1, jb) ** 2 + p_diag % vt(je, 1, jb) ** 2)
            p_diag % vn_ie(je, nlevp1, jb) = p_metrics % wgtfacq_e(je, 1, jb) * p_prog % vn(je, nlev, jb) + p_metrics % wgtfacq_e(je, 2, jb) * p_prog % vn(je, nlev - 1, jb) + p_metrics % wgtfacq_e(je, 3, jb) * p_prog % vn(je, nlev - 2, jb)
          END DO
        ELSE
          DO je = i_startidx, i_endidx
            p_diag % vn_ie(je, 1, jb) = p_diag % vn_ie_ubc(je, 1, jb) + dt_linintp_ubc * p_diag % vn_ie_ubc(je, 2, jb)
            z_vt_ie(je, 1, jb) = p_diag % vt(je, 1, jb)
            z_kin_hor_e(je, 1, jb) = 0.5D0 * (p_prog % vn(je, 1, jb) ** 2 + p_diag % vt(je, 1, jb) ** 2)
            p_diag % vn_ie(je, nlevp1, jb) = p_metrics % wgtfacq_e(je, 1, jb) * p_prog % vn(je, nlev, jb) + p_metrics % wgtfacq_e(je, 2, jb) * p_prog % vn(je, nlev - 1, jb) + p_metrics % wgtfacq_e(je, 3, jb) * p_prog % vn(je, nlev - 2, jb)
          END DO
        END IF
      END DO
    END IF
    rl_start = 7
    rl_end = -9
    i_startblk = p_patch % edges % start_block(7)
    i_endblk = p_patch % edges % end_block(-9)
    IF (.NOT. lvn_only) THEN
      DO jb = i_startblk, i_endblk
        CALL get_indices_e(p_patch, jb, i_startblk, i_endblk, i_startidx, i_endidx, 7, -9)
        DO jk = 1, nlev
          DO je = i_startidx, i_endidx
            z_v_grad_w(je, jk, jb) = p_diag % vn_ie(je, jk, jb) * p_patch % edges % inv_dual_edge_length(je, jb) * (p_prog % w(p_patch % edges % cell_idx(je, jb, 1), jk, p_patch % edges % cell_blk(je, jb, 1)) - p_prog % w(p_patch % edges % cell_idx(je, jb, 2), jk, p_patch % edges % cell_blk(je, jb, 2))) + z_vt_ie(je, jk, jb) * p_patch % edges % inv_primal_edge_length(je, jb) * p_patch % edges % tangent_orientation(je, jb) * (z_w_v(p_patch % edges % vertex_idx(je, jb, 1), jk, p_patch % edges % vertex_blk(je, jb, 1)) - z_w_v(p_patch % edges % vertex_idx(je, jb, 2), jk, p_patch % edges % vertex_blk(je, jb, 2)))
          END DO
        END DO
      END DO
    END IF
    IF (.NOT. lvn_only .AND. ldeepatmo) THEN
      DO jb = i_startblk, i_endblk
        CALL get_indices_e(p_patch, jb, i_startblk, i_endblk, i_startidx, i_endidx, 7, -9)
        DO jk = 1, nlev
          DO je = i_startidx, i_endidx
            z_v_grad_w(je, jk, jb) = z_v_grad_w(je, jk, jb) * p_metrics % deepatmo_gradh_ifc(jk) + p_diag % vn_ie(je, jk, jb) * (p_diag % vn_ie(je, jk, jb) * p_metrics % deepatmo_invr_ifc(jk) - p_patch % edges % ft_e(je, jb)) + z_vt_ie(je, jk, jb) * (z_vt_ie(je, jk, jb) * p_metrics % deepatmo_invr_ifc(jk) + p_patch % edges % fn_e(je, jb))
          END DO
        END DO
      END DO
    END IF
    rl_start = 4
    rl_end = -5
    i_startblk = p_patch % cells % start_block(4)
    i_endblk = p_patch % cells % end_block(-5)
    rl_start_2 = 5
    rl_end_2 = -4
    i_startblk_2 = p_patch % cells % start_block(5)
    i_endblk_2 = p_patch % cells % end_block(-4)
    DO jb = i_startblk, i_endblk
      CALL get_indices_c(p_patch, jb, i_startblk, i_endblk, i_startidx, i_endidx, 4, -5)
      DO jk = 1, nlev
        DO jc = i_startidx, i_endidx
          z_ekinh(jc, jk, jb) = p_int % e_bln_c_s(jc, 1, jb) * z_kin_hor_e(p_patch % cells % edge_idx(jc, jb, 1), jk, p_patch % cells % edge_blk(jc, jb, 1)) + p_int % e_bln_c_s(jc, 2, jb) * z_kin_hor_e(p_patch % cells % edge_idx(jc, jb, 2), jk, p_patch % cells % edge_blk(jc, jb, 2)) + p_int % e_bln_c_s(jc, 3, jb) * z_kin_hor_e(p_patch % cells % edge_idx(jc, jb, 3), jk, p_patch % cells % edge_blk(jc, jb, 3))
        END DO
      END DO
      IF (istep == 1) THEN
        DO jk = nflatlev_jg, nlev
          DO jc = i_startidx, i_endidx
            z_w_concorr_mc(jc, jk) = p_int % e_bln_c_s(jc, 1, jb) * z_w_concorr_me(p_patch % cells % edge_idx(jc, jb, 1), jk, p_patch % cells % edge_blk(jc, jb, 1)) + p_int % e_bln_c_s(jc, 2, jb) * z_w_concorr_me(p_patch % cells % edge_idx(jc, jb, 2), jk, p_patch % cells % edge_blk(jc, jb, 2)) + p_int % e_bln_c_s(jc, 3, jb) * z_w_concorr_me(p_patch % cells % edge_idx(jc, jb, 3), jk, p_patch % cells % edge_blk(jc, jb, 3))
          END DO
        END DO
        DO jk = nflatlev_jg + 1, nlev
          DO jc = i_startidx, i_endidx
            p_diag % w_concorr_c(jc, jk, jb) = p_metrics % wgtfac_c(jc, jk, jb) * z_w_concorr_mc(jc, jk) + (1.0D0 - p_metrics % wgtfac_c(jc, jk, jb)) * z_w_concorr_mc(jc, jk - 1)
          END DO
        END DO
      END IF
      DO jk = 1, nlev
        DO jc = i_startidx, i_endidx
          z_w_con_c(jc, jk) = p_prog % w(jc, jk, jb)
        END DO
      END DO
      DO jc = i_startidx, i_endidx
        z_w_con_c(jc, nlevp1) = 0.0D0
      END DO
      DO jk = nlev, nflatlev_jg + 1, - 1
        DO jc = i_startidx, i_endidx
          z_w_con_c(jc, jk) = z_w_con_c(jc, jk) - p_diag % w_concorr_c(jc, jk, jb)
        END DO
      END DO
      DO jk = MAX(3, nrdmax_jg - 2), nlev - 3
        levmask(jb, jk) = .FALSE.
      END DO
      maxvcfl = 0
      DO jk = MAX(3, nrdmax_jg - 2), nlev - 3
        clip_count = 0
        DO jc = i_startidx, i_endidx
          cfl_clipping(jc, jk) = (ABS(z_w_con_c(jc, jk)) > cfl_w_limit * p_metrics % ddqz_z_half(jc, jk, jb))
          IF (cfl_clipping(jc, jk)) clip_count = clip_count + 1
        END DO
        IF (clip_count == 0) CYCLE
        DO jc = i_startidx, i_endidx
          IF (cfl_clipping(jc, jk)) THEN
            levmask(jb, jk) = .TRUE.
            vcfl = z_w_con_c(jc, jk) * dtime / p_metrics % ddqz_z_half(jc, jk, jb)
            maxvcfl = MAX(maxvcfl, ABS(vcfl))
            IF (vcfl < - 0.85D0) THEN
              z_w_con_c(jc, jk) = - 0.85D0 * p_metrics % ddqz_z_half(jc, jk, jb) / dtime
            ELSE IF (vcfl > 0.85D0) THEN
              z_w_con_c(jc, jk) = 0.85D0 * p_metrics % ddqz_z_half(jc, jk, jb) / dtime
            END IF
          END IF
        END DO
      END DO
      DO jk = 1, nlev
        DO jc = i_startidx, i_endidx
          z_w_con_c_full(jc, jk, jb) = 0.5D0 * (z_w_con_c(jc, jk) + z_w_con_c(jc, jk + 1))
        END DO
      END DO
      vcflmax(jb) = maxvcfl
      IF (lvn_only) CYCLE
      IF (jb < i_startblk_2 .OR. jb > i_endblk_2) CYCLE
      CALL get_indices_c(p_patch, jb, i_startblk_2, i_endblk_2, i_startidx_2, i_endidx_2, 5, -4)
      DO jk = 2, nlev
        DO jc = i_startidx_2, i_endidx_2
          p_diag % ddt_w_adv_pc(jc, jk, jb, ntnd) = - z_w_con_c(jc, jk) * (p_prog % w(jc, jk - 1, jb) * p_metrics % coeff1_dwdz(jc, jk, jb) - p_prog % w(jc, jk + 1, jb) * p_metrics % coeff2_dwdz(jc, jk, jb) + p_prog % w(jc, jk, jb) * (p_metrics % coeff2_dwdz(jc, jk, jb) - p_metrics % coeff1_dwdz(jc, jk, jb)))
        END DO
      END DO
      DO jk = 2, nlev
        DO jc = i_startidx_2, i_endidx_2
          p_diag % ddt_w_adv_pc(jc, jk, jb, ntnd) = p_diag % ddt_w_adv_pc(jc, jk, jb, ntnd) + p_int % e_bln_c_s(jc, 1, jb) * z_v_grad_w(p_patch % cells % edge_idx(jc, jb, 1), jk, p_patch % cells % edge_blk(jc, jb, 1)) + p_int % e_bln_c_s(jc, 2, jb) * z_v_grad_w(p_patch % cells % edge_idx(jc, jb, 2), jk, p_patch % cells % edge_blk(jc, jb, 2)) + p_int % e_bln_c_s(jc, 3, jb) * z_v_grad_w(p_patch % cells % edge_idx(jc, jb, 3), jk, p_patch % cells % edge_blk(jc, jb, 3))
        END DO
      END DO
      IF (lextra_diffu) THEN
        DO jk = MAX(3, nrdmax_jg - 2), nlev - 3
          IF (levmask(jb, jk)) THEN
            DO jc = i_startidx_2, i_endidx_2
              IF (cfl_clipping(jc, jk) .AND. p_patch % cells % decomp_info % owner_mask(jc, jb)) THEN
                difcoef = scalfac_exdiff * MIN(0.85D0 - cfl_w_limit * dtime, ABS(z_w_con_c(jc, jk)) * dtime / p_metrics % ddqz_z_half(jc, jk, jb) - cfl_w_limit * dtime)
                p_diag % ddt_w_adv_pc(jc, jk, jb, ntnd) = p_diag % ddt_w_adv_pc(jc, jk, jb, ntnd) + difcoef * p_patch % cells % area(jc, jb) * (p_prog % w(jc, jk, jb) * p_int % geofac_n2s(jc, 1, jb) + p_prog % w(p_patch % cells % neighbor_idx(jc, jb, 1), jk, p_patch % cells % neighbor_blk(jc, jb, 1)) * p_int % geofac_n2s(jc, 2, jb) + p_prog % w(p_patch % cells % neighbor_idx(jc, jb, 2), jk, p_patch % cells % neighbor_blk(jc, jb, 2)) * p_int % geofac_n2s(jc, 3, jb) + p_prog % w(p_patch % cells % neighbor_idx(jc, jb, 3), jk, p_patch % cells % neighbor_blk(jc, jb, 3)) * p_int % geofac_n2s(jc, 4, jb))
              END IF
            END DO
          END IF
        END DO
      END IF
    END DO
    DO jk = MAX(3, nrdmax_jg - 2), nlev - 3
      levelmask(jk) = ANY(levmask(i_startblk : i_endblk, jk))
    END DO
    rl_start = 10
    rl_end = -8
    i_startblk = p_patch % edges % start_block(10)
    i_endblk = p_patch % edges % end_block(-8)
    DO jb = i_startblk, i_endblk
      CALL get_indices_e(p_patch, jb, i_startblk, i_endblk, i_startidx, i_endidx, 10, -8)
      IF (.NOT. ldeepatmo) THEN
        DO jk = 1, nlev
          DO je = i_startidx, i_endidx
            p_diag % ddt_vn_apc_pc(je, jk, jb, ntnd) = - (z_kin_hor_e(je, jk, jb) * (p_metrics % coeff_gradekin(je, 1, jb) - p_metrics % coeff_gradekin(je, 2, jb)) + p_metrics % coeff_gradekin(je, 2, jb) * z_ekinh(p_patch % edges % cell_idx(je, jb, 2), jk, p_patch % edges % cell_blk(je, jb, 2)) - p_metrics % coeff_gradekin(je, 1, jb) * z_ekinh(p_patch % edges % cell_idx(je, jb, 1), jk, p_patch % edges % cell_blk(je, jb, 1)) + p_diag % vt(je, jk, jb) * (p_patch % edges % f_e(je, jb) + 0.5D0 * (zeta(p_patch % edges % vertex_idx(je, jb, 1), jk, p_patch % edges % vertex_blk(je, jb, 1)) + zeta(p_patch % edges % vertex_idx(je, jb, 2), jk, p_patch % edges % vertex_blk(je, jb, 2)))) + (p_int % c_lin_e(je, 1, jb) * z_w_con_c_full(p_patch % edges % cell_idx(je, jb, 1), jk, p_patch % edges % cell_blk(je, jb, 1)) + p_int % c_lin_e(je, 2, jb) * z_w_con_c_full(p_patch % edges % cell_idx(je, jb, 2), jk, p_patch % edges % cell_blk(je, jb, 2))) * (p_diag % vn_ie(je, jk, jb) - p_diag % vn_ie(je, jk + 1, jb)) / p_metrics % ddqz_z_full_e(je, jk, jb))
          END DO
        END DO
        IF (p_diag % ddt_vn_adv_is_associated .OR. p_diag % ddt_vn_cor_is_associated) THEN
          DO jk = 1, nlev
            DO je = i_startidx, i_endidx
              p_diag % ddt_vn_cor_pc(je, jk, jb, ntnd) = - p_diag % vt(je, jk, jb) * p_patch % edges % f_e(je, jb)
            END DO
          END DO
        END IF
      ELSE
        DO jk = 1, nlev
          DO je = i_startidx, i_endidx
            p_diag % ddt_vn_apc_pc(je, jk, jb, ntnd) = - ((z_kin_hor_e(je, jk, jb) * (p_metrics % coeff_gradekin(je, 1, jb) - p_metrics % coeff_gradekin(je, 2, jb)) + p_metrics % coeff_gradekin(je, 2, jb) * z_ekinh(p_patch % edges % cell_idx(je, jb, 2), jk, p_patch % edges % cell_blk(je, jb, 2)) - p_metrics % coeff_gradekin(je, 1, jb) * z_ekinh(p_patch % edges % cell_idx(je, jb, 1), jk, p_patch % edges % cell_blk(je, jb, 1))) * p_metrics % deepatmo_gradh_mc(jk) + p_diag % vt(je, jk, jb) * (p_patch % edges % f_e(je, jb) + 0.5D0 * (zeta(p_patch % edges % vertex_idx(je, jb, 1), jk, p_patch % edges % vertex_blk(je, jb, 1)) + zeta(p_patch % edges % vertex_idx(je, jb, 2), jk, p_patch % edges % vertex_blk(je, jb, 2))) * p_metrics % deepatmo_gradh_mc(jk)) + (p_int % c_lin_e(je, 1, jb) * z_w_con_c_full(p_patch % edges % cell_idx(je, jb, 1), jk, p_patch % edges % cell_blk(je, jb, 1)) + p_int % c_lin_e(je, 2, jb) * z_w_con_c_full(p_patch % edges % cell_idx(je, jb, 2), jk, p_patch % edges % cell_blk(je, jb, 2))) * ((p_diag % vn_ie(je, jk, jb) - p_diag % vn_ie(je, jk + 1, jb)) / p_metrics % ddqz_z_full_e(je, jk, jb) + p_prog % vn(je, jk, jb) * p_metrics % deepatmo_invr_mc(jk) - p_patch % edges % ft_e(je, jb)))
          END DO
        END DO
        IF (p_diag % ddt_vn_adv_is_associated .OR. p_diag % ddt_vn_cor_is_associated) THEN
          DO jk = 1, nlev
            DO je = i_startidx, i_endidx
              p_diag % ddt_vn_cor_pc(je, jk, jb, ntnd) = - (+ p_diag % vt(je, jk, jb) * (p_patch % edges % f_e(je, jb)) + (p_int % c_lin_e(je, 1, jb) * z_w_con_c_full(p_patch % edges % cell_idx(je, jb, 1), jk, p_patch % edges % cell_blk(je, jb, 1)) + p_int % c_lin_e(je, 2, jb) * z_w_con_c_full(p_patch % edges % cell_idx(je, jb, 2), jk, p_patch % edges % cell_blk(je, jb, 2))) * (- p_patch % edges % ft_e(je, jb)))
            END DO
          END DO
        END IF
      END IF
      IF (lextra_diffu) THEN
        ie = 0
        DO jk = MAX(3, nrdmax_jg - 2), nlev - 4
          IF (levelmask(jk) .OR. levelmask(jk + 1)) THEN
            DO je = i_startidx, i_endidx
              w_con_e = p_int % c_lin_e(je, 1, jb) * z_w_con_c_full(p_patch % edges % cell_idx(je, jb, 1), jk, p_patch % edges % cell_blk(je, jb, 1)) + p_int % c_lin_e(je, 2, jb) * z_w_con_c_full(p_patch % edges % cell_idx(je, jb, 2), jk, p_patch % edges % cell_blk(je, jb, 2))
              IF (ABS(w_con_e) > cfl_w_limit * p_metrics % ddqz_z_full_e(je, jk, jb)) THEN
                difcoef = scalfac_exdiff * MIN(0.85D0 - cfl_w_limit * dtime, ABS(w_con_e) * dtime / p_metrics % ddqz_z_full_e(je, jk, jb) - cfl_w_limit * dtime)
                p_diag % ddt_vn_apc_pc(je, jk, jb, ntnd) = p_diag % ddt_vn_apc_pc(je, jk, jb, ntnd) + difcoef * p_patch % edges % area_edge(je, jb) * (p_int % geofac_grdiv(je, 1, jb) * p_prog % vn(je, jk, jb) + p_int % geofac_grdiv(je, 2, jb) * p_prog % vn(p_patch % edges % quad_idx(je, jb, 1), jk, p_patch % edges % quad_blk(je, jb, 1)) + p_int % geofac_grdiv(je, 3, jb) * p_prog % vn(p_patch % edges % quad_idx(je, jb, 2), jk, p_patch % edges % quad_blk(je, jb, 2)) + p_int % geofac_grdiv(je, 4, jb) * p_prog % vn(p_patch % edges % quad_idx(je, jb, 3), jk, p_patch % edges % quad_blk(je, jb, 3)) + p_int % geofac_grdiv(je, 5, jb) * p_prog % vn(p_patch % edges % quad_idx(je, jb, 4), jk, p_patch % edges % quad_blk(je, jb, 4)) + p_patch % edges % tangent_orientation(je, jb) * p_patch % edges % inv_primal_edge_length(je, jb) * (zeta(p_patch % edges % vertex_idx(je, jb, 2), jk, p_patch % edges % vertex_blk(je, jb, 2)) - zeta(p_patch % edges % vertex_idx(je, jb, 1), jk, p_patch % edges % vertex_blk(je, jb, 1))))
              END IF
            END DO
          END IF
        END DO
      END IF
    END DO
    i_startblk = p_patch % cells % start_block(4)
    i_endblk = p_patch % cells % end_block(-4)
    max_vcfl_dyn = MAX(p_diag % max_vcfl_dyn, MAXVAL(vcflmax(i_startblk : i_endblk)))
    p_diag % max_vcfl_dyn = max_vcfl_dyn
    IF (timers_level > 5) CALL timer_stop(timer_solve_nh_veltend)
  END SUBROUTINE velocity_tendencies
END MODULE mo_velocity_advection
