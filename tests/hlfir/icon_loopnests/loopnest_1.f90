! ICON velocity_advection loopnest 1 - indirect stencil, full vertical.
! Pattern key: ['v.h', ('full_vert', 'full_horiz'), 'compute', 'h:U  v:S'].
! Selected by ``select_loopnests.py`` (chosen_loopnests.md, loopnest_1, pinned).
!
! Source snippet (z_v_grad_w producer with edge-to-cell + edge-to-vertex
! indirect indexing via icidx/icblk and ividx/ivblk respectively):
!   DO jk = 1, nlev
!     DO je = i_startidx, i_endidx
!       z_v_grad_w(je, jk, jb) = vn_ie(je, jk, jb) * inv_dual_edge_length(je, jb) &
!         * (w(icidx(je, jb, 1), jk, icblk(je, jb, 1)) - w(icidx(je, jb, 2), jk, icblk(je, jb, 2))) &
!         + z_vt_ie(je, jk, jb) * inv_primal_edge_length(je, jb) * tangent_orientation(je, jb) &
!         * (z_w_v(ividx(je, jb, 1), jk, ivblk(je, jb, 1)) - z_w_v(ividx(je, jb, 2), jk, ivblk(je, jb, 2)))
!     END DO
!   END DO
!
! ``icidx`` / ``icblk`` / ``ividx`` / ``ivblk`` are 3-D INTEGER arrays
! shaped (nproma, nblks, 2) - for each edge (je, jb), entries 1 and 2
! give the (proma-index, block-index) of the two adjacent cells (or
! vertices).  Test inputs MUST keep these within
! ``[1, nproma]`` and ``[1, nblks]`` so the indirect reads stay
! in-bounds.
SUBROUTINE icon_loopnest_1(nproma, nlev, nblks, jb, i_startidx, i_endidx, &
                           vn_ie, z_vt_ie, w, z_w_v, &
                           inv_dual_edge_length, inv_primal_edge_length, tangent_orientation, &
                           icidx, icblk, ividx, ivblk, z_v_grad_w)
  IMPLICIT NONE
  INTEGER(KIND=4), VALUE :: nproma, nlev, nblks, jb, i_startidx, i_endidx
  REAL(KIND=8), INTENT(IN)    :: vn_ie(nproma, nlev, nblks)
  REAL(KIND=8), INTENT(IN)    :: z_vt_ie(nproma, nlev, nblks)
  REAL(KIND=8), INTENT(IN)    :: w(nproma, nlev, nblks)
  REAL(KIND=8), INTENT(IN)    :: z_w_v(nproma, nlev, nblks)
  REAL(KIND=8), INTENT(IN)    :: inv_dual_edge_length(nproma, nblks)
  REAL(KIND=8), INTENT(IN)    :: inv_primal_edge_length(nproma, nblks)
  REAL(KIND=8), INTENT(IN)    :: tangent_orientation(nproma, nblks)
  INTEGER(KIND=4), INTENT(IN) :: icidx(nproma, nblks, 2)
  INTEGER(KIND=4), INTENT(IN) :: icblk(nproma, nblks, 2)
  INTEGER(KIND=4), INTENT(IN) :: ividx(nproma, nblks, 2)
  INTEGER(KIND=4), INTENT(IN) :: ivblk(nproma, nblks, 2)
  REAL(KIND=8), INTENT(INOUT) :: z_v_grad_w(nproma, nlev, nblks)
  INTEGER(KIND=4) :: jk, je
  DO jk = 1, nlev
    DO je = i_startidx, i_endidx
      z_v_grad_w(je, jk, jb) = vn_ie(je, jk, jb) * inv_dual_edge_length(je, jb) &
        * (w(icidx(je, jb, 1), jk, icblk(je, jb, 1)) - w(icidx(je, jb, 2), jk, icblk(je, jb, 2))) &
        + z_vt_ie(je, jk, jb) * inv_primal_edge_length(je, jb) * tangent_orientation(je, jb) &
        * (z_w_v(ividx(je, jb, 1), jk, ivblk(je, jb, 1)) - z_w_v(ividx(je, jb, 2), jk, ivblk(je, jb, 2)))
    END DO
  END DO
END SUBROUTINE icon_loopnest_1
