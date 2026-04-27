! ICON velocity_advection loopnest 3 - direct stencil, full vertical.
! Pattern key: ['v.h', ('full_vert', 'full_horiz'), 'compute', 'h:S  v:S'].
! Selected by ``select_loopnests.py`` (chosen_loopnests.md, loopnest_3).
!
! Source snippet (vertical-only metric arrays - deepatmo_gradh_ifc /
! deepatmo_invr_ifc - are 1-D over jk; ft_e / fn_e are 2-D over (je, jb)):
!   DO jk = 1, nlev
!     DO je = i_startidx, i_endidx
!       z_v_grad_w(je, jk, jb) = z_v_grad_w(je, jk, jb) * deepatmo_gradh_ifc(jk) &
!         + vn_ie(je, jk, jb) * (vn_ie(je, jk, jb) * deepatmo_invr_ifc(jk) - ft_e(je, jb)) &
!         + z_vt_ie(je, jk, jb) * (z_vt_ie(je, jk, jb) * deepatmo_invr_ifc(jk) + fn_e(je, jb))
!     END DO
!   END DO
SUBROUTINE icon_loopnest_3(nproma, nlev, nblks, jb, i_startidx, i_endidx, &
                           vn_ie, z_vt_ie, deepatmo_gradh_ifc, deepatmo_invr_ifc, &
                           ft_e, fn_e, z_v_grad_w)
  IMPLICIT NONE
  INTEGER(KIND=4), VALUE :: nproma, nlev, nblks, jb, i_startidx, i_endidx
  REAL(KIND=8), INTENT(IN)    :: vn_ie(nproma, nlev, nblks)
  REAL(KIND=8), INTENT(IN)    :: z_vt_ie(nproma, nlev, nblks)
  REAL(KIND=8), INTENT(IN)    :: deepatmo_gradh_ifc(nlev)
  REAL(KIND=8), INTENT(IN)    :: deepatmo_invr_ifc(nlev)
  REAL(KIND=8), INTENT(IN)    :: ft_e(nproma, nblks)
  REAL(KIND=8), INTENT(IN)    :: fn_e(nproma, nblks)
  REAL(KIND=8), INTENT(INOUT) :: z_v_grad_w(nproma, nlev, nblks)
  INTEGER(KIND=4) :: jk, je
  DO jk = 1, nlev
    DO je = i_startidx, i_endidx
      z_v_grad_w(je, jk, jb) = z_v_grad_w(je, jk, jb) * deepatmo_gradh_ifc(jk) &
        + vn_ie(je, jk, jb) * (vn_ie(je, jk, jb) * deepatmo_invr_ifc(jk) - ft_e(je, jb)) &
        + z_vt_ie(je, jk, jb) * (z_vt_ie(je, jk, jb) * deepatmo_invr_ifc(jk) + fn_e(je, jb))
    END DO
  END DO
END SUBROUTINE icon_loopnest_3
