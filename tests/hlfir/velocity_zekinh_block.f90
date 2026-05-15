! Velocity-tendencies ``z_ekinh`` reconstruction block extracted from
! ``velocity_full.f90`` lines 511-528.  Bisection narrowed the
! velocity_full segfault to this region; this isolates the smallest
! kernel that still exercises the same offset-generation pattern:
! a ``DO jb`` loop body that gathers ``z_kin_hor_e`` and
! ``z_w_concorr_me`` via the indirect ``p_patch%cells%edge_idx`` /
! ``edge_blk`` tables.
MODULE mo_decomposition_tools
  IMPLICIT NONE
  TYPE :: t_grid_domain_decomp_info
    LOGICAL, ALLOCATABLE :: owner_mask(:, :)
  END TYPE
  CONTAINS
END MODULE mo_decomposition_tools

MODULE mo_model_domain
  USE mo_decomposition_tools, ONLY: t_grid_domain_decomp_info
  IMPLICIT NONE
  TYPE :: t_grid_cells
    INTEGER, ALLOCATABLE :: edge_idx(:, :, :)
    INTEGER, ALLOCATABLE :: edge_blk(:, :, :)
    INTEGER, ALLOCATABLE :: start_block(:)
    INTEGER, ALLOCATABLE :: end_block(:)
  END TYPE
  TYPE :: t_patch
    INTEGER :: nblks_c
    INTEGER :: nlev
    TYPE(t_grid_cells) :: cells
  END TYPE
  CONTAINS
END MODULE mo_model_domain

MODULE mo_intp_data_strc
  IMPLICIT NONE
  TYPE :: t_int_state
    REAL(KIND = 8), ALLOCATABLE :: e_bln_c_s(:, :, :)
  END TYPE
  CONTAINS
END MODULE mo_intp_data_strc

MODULE mo_velocity_zekinh
  IMPLICIT NONE
  CONTAINS
  SUBROUTINE zekinh_block(p_patch, p_int, z_kin_hor_e, z_ekinh, nproma)
    USE mo_model_domain, ONLY: t_patch
    USE mo_intp_data_strc, ONLY: t_int_state
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: nproma
    TYPE(t_patch), TARGET, INTENT(IN) :: p_patch
    TYPE(t_int_state), TARGET, INTENT(IN) :: p_int
    REAL(KIND = 8), INTENT(IN)    :: z_kin_hor_e(:, :, :)
    REAL(KIND = 8), INTENT(INOUT) :: z_ekinh(:, :, :)
    INTEGER :: jb, jk, jc
    INTEGER :: i_startblk, i_endblk
    i_startblk = p_patch % cells % start_block(4)
    i_endblk   = p_patch % cells % end_block(-5)
    DO jb = i_startblk, i_endblk
      DO jk = 1, p_patch % nlev
        DO jc = 1, nproma
          z_ekinh(jc, jk, jb) = &
              p_int % e_bln_c_s(jc, 1, jb) * z_kin_hor_e(p_patch % cells % edge_idx(jc, jb, 1), jk, p_patch % cells % edge_blk(jc, jb, 1)) + &
              p_int % e_bln_c_s(jc, 2, jb) * z_kin_hor_e(p_patch % cells % edge_idx(jc, jb, 2), jk, p_patch % cells % edge_blk(jc, jb, 2)) + &
              p_int % e_bln_c_s(jc, 3, jb) * z_kin_hor_e(p_patch % cells % edge_idx(jc, jb, 3), jk, p_patch % cells % edge_blk(jc, jb, 3))
        END DO
      END DO
    END DO
  END SUBROUTINE zekinh_block
END MODULE mo_velocity_zekinh
