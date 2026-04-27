! ICON velocity_advection loopnest 2 -- DERIVED-TYPE variant.
!
! Same math as ``loopnest_2.f90`` but the input arrays come through
! ICON-style derived-type chains (``p_prog%vn``, ``p_metrics%ddxn_z_full``,
! ``p_diag%vt``, ``p_metrics%ddxt_z_full``).  The bridge's
! ``hlfir-flatten-structs`` pass decomposes each struct into per-field
! flat arrays (``p_prog_vn``, ``p_metrics_ddxn_z_full``, etc.) before
! AST extraction, so the resulting SDFG has a flat signature
! identical-up-to-naming with the non-struct variant.
!
! Phase 1 of derived-type support handles flat-member structs only;
! the array members here are fixed-shape via the module PARAMETER
! constants below so flatten-structs can lower them without seeing
! allocatable / pointer / nested-struct shapes.
MODULE icon_types_loopnest_2
  IMPLICIT NONE
  INTEGER, PARAMETER :: NPROMA = 32
  INTEGER, PARAMETER :: NLEV = 32
  INTEGER, PARAMETER :: NBLKS = 5
  TYPE :: t_prog
    REAL(KIND=8) :: vn(NPROMA, NLEV, NBLKS)
  END TYPE t_prog
  TYPE :: t_metrics
    REAL(KIND=8) :: ddxn_z_full(NPROMA, NLEV, NBLKS)
    REAL(KIND=8) :: ddxt_z_full(NPROMA, NLEV, NBLKS)
  END TYPE t_metrics
  TYPE :: t_diag
    REAL(KIND=8) :: vt(NPROMA, NLEV, NBLKS)
  END TYPE t_diag
END MODULE icon_types_loopnest_2

SUBROUTINE icon_loopnest_2_struct(nflatlev, jb, i_startidx, i_endidx, &
                                  p_prog, p_metrics, p_diag, z_w_concorr_me)
  USE icon_types_loopnest_2
  IMPLICIT NONE
  INTEGER(KIND=4), VALUE :: nflatlev, jb, i_startidx, i_endidx
  TYPE(t_prog),    INTENT(IN)    :: p_prog
  TYPE(t_metrics), INTENT(IN)    :: p_metrics
  TYPE(t_diag),    INTENT(IN)    :: p_diag
  REAL(KIND=8),    INTENT(INOUT) :: z_w_concorr_me(NPROMA, NLEV, NBLKS)
  INTEGER(KIND=4) :: jk, je
  DO jk = nflatlev, NLEV
    DO je = i_startidx, i_endidx
      z_w_concorr_me(je, jk, jb) = p_prog%vn(je, jk, jb) * p_metrics%ddxn_z_full(je, jk, jb) &
                                 + p_diag%vt(je, jk, jb) * p_metrics%ddxt_z_full(je, jk, jb)
    END DO
  END DO
END SUBROUTINE icon_loopnest_2_struct
