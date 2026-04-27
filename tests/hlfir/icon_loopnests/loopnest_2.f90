! ICON velocity_advection loopnest 2 - direct stencil, partial vertical.
! Pattern key: ['v.h', ('partial_vert', 'full_horiz'), 'compute', 'h:S  v:S'].
! Selected by ``select_loopnests.py`` (chosen_loopnests.md, loopnest_2).
!
! Source snippet:
!   DO jk = nflatlev_jg, nlev
!     DO je = i_startidx, i_endidx
!       z_w_concorr_me(je, jk, jb) = p_prog%vn(je, jk, jb) * p_metrics%ddxn_z_full(je, jk, jb) &
!                                  + p_diag%vt(je, jk, jb) * p_metrics%ddxt_z_full(je, jk, jb)
!     END DO
!   END DO
!
! Wrapped as a standalone subroutine taking plain arrays (the derived-type
! struct membership is irrelevant to the math - Phase 2 derived-type
! work would let us call this through the original ``p_diag%vt`` shape).
SUBROUTINE icon_loopnest_2(nproma, nlev, nblks, nflatlev, jb, i_startidx, i_endidx, &
                           vn, ddxn_z_full, vt, ddxt_z_full, z_w_concorr_me)
  IMPLICIT NONE
  INTEGER(KIND=4), VALUE :: nproma, nlev, nblks, nflatlev, jb, i_startidx, i_endidx
  REAL(KIND=8), INTENT(IN)    :: vn(nproma, nlev, nblks)
  REAL(KIND=8), INTENT(IN)    :: ddxn_z_full(nproma, nlev, nblks)
  REAL(KIND=8), INTENT(IN)    :: vt(nproma, nlev, nblks)
  REAL(KIND=8), INTENT(IN)    :: ddxt_z_full(nproma, nlev, nblks)
  REAL(KIND=8), INTENT(INOUT) :: z_w_concorr_me(nproma, nlev, nblks)
  INTEGER(KIND=4) :: jk, je
  DO jk = nflatlev, nlev
    DO je = i_startidx, i_endidx
      z_w_concorr_me(je, jk, jb) = vn(je, jk, jb) * ddxn_z_full(je, jk, jb) &
                                 + vt(je, jk, jb) * ddxt_z_full(je, jk, jb)
    END DO
  END DO
END SUBROUTINE icon_loopnest_2
