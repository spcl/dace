! Single loop nest in the shape of ICON velocity_tendencies'
! half-level differentiation (upstream line 444-449, simplified):
! pure subtraction + pure copy with no multiplications, so the
! SDFG path and the numpy reference can be compared bit-exact.
! The kernel keeps the same struct-dummy shape (TARGET, POINTER
! members, USE-imported types) so the bridge still exercises:
!   * Pointer-array struct member flattening (t_nh_prog%vn, etc.)
!   * Module-level USE-imported struct types
! and nothing else -- no indirect gather, no nested call.
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
  TYPE :: t_grid_edges
    INTEGER, ALLOCATABLE :: start_index(:)
    INTEGER, ALLOCATABLE :: end_index(:)
    INTEGER, ALLOCATABLE :: start_block(:)
    INTEGER, ALLOCATABLE :: end_block(:)
  END TYPE
  TYPE :: t_patch
    INTEGER :: nblks_e
    INTEGER :: nlev
    TYPE(t_grid_edges) :: edges
  END TYPE
  CONTAINS
END MODULE mo_model_domain

MODULE mo_nonhydro_types
  IMPLICIT NONE
  TYPE :: t_nh_prog
    REAL(KIND=8), POINTER, CONTIGUOUS :: vn(:, :, :)
  END TYPE
  TYPE :: t_nh_diag
    REAL(KIND=8), POINTER, CONTIGUOUS :: vt(:, :, :), vn_ie(:, :, :)
  END TYPE
  TYPE :: t_nh_metrics
    REAL(KIND=8), POINTER, CONTIGUOUS :: wgtfac_e(:, :, :)
  END TYPE
END MODULE mo_nonhydro_types

MODULE mo_velocity_one
  IMPLICIT NONE
  CONTAINS
  SUBROUTINE one_loop_nest(p_prog, p_patch, p_metrics, p_diag, z_kin_hor_e, nproma)
    USE mo_model_domain, ONLY: t_patch
    USE mo_nonhydro_types, ONLY: t_nh_prog, t_nh_metrics, t_nh_diag
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: nproma
    TYPE(t_patch), TARGET, INTENT(IN) :: p_patch
    TYPE(t_nh_prog), INTENT(IN) :: p_prog
    TYPE(t_nh_metrics), INTENT(IN) :: p_metrics
    TYPE(t_nh_diag), INTENT(INOUT) :: p_diag
    REAL(KIND = 8), INTENT(INOUT) :: z_kin_hor_e(nproma, p_patch % nlev, p_patch % nblks_e)
    INTEGER :: jb, jk, je
    INTEGER :: nlev, nblks_e
    nlev = p_patch % nlev
    nblks_e = p_patch % nblks_e
    DO jb = 1, nblks_e
      DO jk = 2, nlev
        DO je = 1, nproma
          p_diag % vn_ie(je, jk, jb) = p_prog % vn(je, jk, jb) - p_prog % vn(je, jk - 1, jb)
          z_kin_hor_e(je, jk, jb) = p_diag % vt(je, jk, jb) - p_metrics % wgtfac_e(je, jk, jb)
        END DO
      END DO
    END DO
  END SUBROUTINE one_loop_nest
END MODULE mo_velocity_one

! Flat-arg wrapper for the f2py reference path.  f2py's crackfortran
! cannot derive a Python binding for derived-type dummies that carry
! POINTER members; this wrapper materialises the struct args from flat
! arrays + dimension scalars inside the Fortran side, then forwards to
! ``one_loop_nest``.  ``TARGET`` on the locals is required so the
! ``=>`` pointer-assoc captures the flat buffer's address.
SUBROUTINE one_loop_nest_flat(vn, wgtfac_e, vt, vn_ie, z_kin_hor_e, &
                              nproma, nlev, nblks_e)
  USE mo_model_domain, ONLY: t_patch
  USE mo_nonhydro_types, ONLY: t_nh_prog, t_nh_metrics, t_nh_diag
  USE mo_velocity_one, ONLY: one_loop_nest
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: nproma, nlev, nblks_e
  REAL(KIND = 8), TARGET,    INTENT(IN)    :: vn(nproma, nlev, nblks_e)
  REAL(KIND = 8), TARGET,    INTENT(IN)    :: wgtfac_e(nproma, nlev, nblks_e)
  REAL(KIND = 8), TARGET,    INTENT(INOUT) :: vt(nproma, nlev, nblks_e)
  REAL(KIND = 8), TARGET,    INTENT(INOUT) :: vn_ie(nproma, nlev, nblks_e)
  REAL(KIND = 8),            INTENT(INOUT) :: z_kin_hor_e(nproma, nlev, nblks_e)
  TYPE(t_patch),     TARGET :: p_patch
  TYPE(t_nh_prog)           :: p_prog
  TYPE(t_nh_metrics)        :: p_metrics
  TYPE(t_nh_diag)           :: p_diag
  p_patch % nblks_e = nblks_e
  p_patch % nlev    = nlev
  p_prog    % vn       => vn
  p_metrics % wgtfac_e => wgtfac_e
  p_diag    % vt       => vt
  p_diag    % vn_ie    => vn_ie
  CALL one_loop_nest(p_prog, p_patch, p_metrics, p_diag, z_kin_hor_e, nproma)
END SUBROUTINE one_loop_nest_flat
