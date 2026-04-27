! ICON velocity_advection loopnest 5 - horizontal-only (boundary).
! Pattern key: ['h', ('full_horiz',), 'compute', 'h:S'].
! Selected by ``select_loopnests.py`` (chosen_loopnests.md, loopnest_5).
!
! Source snippet - 4 statements over (je, 1, jb) for level-1 boundary
! quantities, plus one over (je, nlevp1, jb) for the surface:
!   DO je = i_startidx, i_endidx
!     vn_ie(je, 1, jb)       = vn_ie_ubc(je, 1, jb) + dt_linintp_ubc * vn_ie_ubc(je, 2, jb)
!     z_vt_ie(je, 1, jb)     = vt(je, 1, jb)
!     z_kin_hor_e(je, 1, jb) = 0.5_wp * (vn(je, 1, jb)**2 + vt(je, 1, jb)**2)
!     vn_ie(je, nlevp1, jb)  = wgtfacq_e(je, 1, jb) * vn(je, nlev, jb)         &
!                            + wgtfacq_e(je, 2, jb) * vn(je, nlev - 1, jb)    &
!                            + wgtfacq_e(je, 3, jb) * vn(je, nlev - 2, jb)
!   END DO
SUBROUTINE icon_loopnest_5(nproma, nlev, nlevp1, nblks, jb, i_startidx, i_endidx, &
                           dt_linintp_ubc, vn_ie_ubc, vt, vn, wgtfacq_e, &
                           vn_ie, z_vt_ie, z_kin_hor_e)
  IMPLICIT NONE
  INTEGER(KIND=4), VALUE :: nproma, nlev, nlevp1, nblks, jb, i_startidx, i_endidx
  REAL(KIND=8), VALUE :: dt_linintp_ubc
  REAL(KIND=8), INTENT(IN)    :: vn_ie_ubc(nproma, 2, nblks)
  REAL(KIND=8), INTENT(IN)    :: vt(nproma, nlev, nblks)
  REAL(KIND=8), INTENT(IN)    :: vn(nproma, nlev, nblks)
  REAL(KIND=8), INTENT(IN)    :: wgtfacq_e(nproma, 3, nblks)
  REAL(KIND=8), INTENT(INOUT) :: vn_ie(nproma, nlevp1, nblks)
  REAL(KIND=8), INTENT(INOUT) :: z_vt_ie(nproma, nlev, nblks)
  REAL(KIND=8), INTENT(INOUT) :: z_kin_hor_e(nproma, nlev, nblks)
  INTEGER(KIND=4) :: je
  DO je = i_startidx, i_endidx
    vn_ie(je, 1, jb)       = vn_ie_ubc(je, 1, jb) + dt_linintp_ubc * vn_ie_ubc(je, 2, jb)
    z_vt_ie(je, 1, jb)     = vt(je, 1, jb)
    z_kin_hor_e(je, 1, jb) = 0.5D0 * (vn(je, 1, jb)**2 + vt(je, 1, jb)**2)
    vn_ie(je, nlevp1, jb)  = wgtfacq_e(je, 1, jb) * vn(je, nlev, jb)        &
                           + wgtfacq_e(je, 2, jb) * vn(je, nlev - 1, jb)    &
                           + wgtfacq_e(je, 3, jb) * vn(je, nlev - 2, jb)
  END DO
END SUBROUTINE icon_loopnest_5
