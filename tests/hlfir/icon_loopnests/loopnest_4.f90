! ICON velocity_advection loopnest 4 - indirect stencil, partial vertical.
! Pattern key: ['v.h', ('partial_vert', 'full_horiz'), 'compute', 'h:U  v:S'].
! Selected by ``select_loopnests.py`` (chosen_loopnests.md, loopnest_4).
!
! CFL-clip diffusion path: levelmask-gated, with both edge-to-cell
! indirect (icidx/icblk) and edge-to-edge indirect (iqidx/iqblk).
!
! Source snippet (lines reflowed; ``ntnd`` collapsed since it's a
! constant time-tendency channel):
!   DO jk = MAX(3, nrdmax_jg - 2), nlev - 4
!     IF (levelmask(jk) .OR. levelmask(jk + 1)) THEN
!       DO je = i_startidx, i_endidx
!         w_con_e = c_lin_e(je, 1, jb) * z_w_con_c_full(icidx(je, jb, 1), jk, icblk(je, jb, 1)) &
!                 + c_lin_e(je, 2, jb) * z_w_con_c_full(icidx(je, jb, 2), jk, icblk(je, jb, 2))
!         IF (ABS(w_con_e) > cfl_w_limit * ddqz_z_full_e(je, jk, jb)) THEN
!           difcoef = scalfac_exdiff * MIN(0.85D0 - cfl_w_limit * dtime, &
!                                          ABS(w_con_e) * dtime / ddqz_z_full_e(je, jk, jb) - cfl_w_limit * dtime)
!           ddt_vn_apc(je, jk, jb) = ddt_vn_apc(je, jk, jb) + difcoef * area_edge(je, jb) &
!             * (geofac_grdiv(je, 1, jb) * vn(je, jk, jb) &
!              + geofac_grdiv(je, 2, jb) * vn(iqidx(je, jb, 1), jk, iqblk(je, jb, 1)) &
!              + geofac_grdiv(je, 3, jb) * vn(iqidx(je, jb, 2), jk, iqblk(je, jb, 2)) &
!              + geofac_grdiv(je, 4, jb) * vn(iqidx(je, jb, 3), jk, iqblk(je, jb, 3)) &
!              + geofac_grdiv(je, 5, jb) * vn(iqidx(je, jb, 4), jk, iqblk(je, jb, 4)))
!         END IF
!       END DO
!     END IF
!   END DO
!
! Indirect arrays must be initialised with values in [1, nproma] (idx)
! and [1, nblks] (blk).  ``levelmask`` is a 1-D LOGICAL array of length
! ``nlev``.  ``laericeauto``-style INTEGER-as-bool isn't used here.
SUBROUTINE icon_loopnest_4(nproma, nlev, nblks, jb, i_startidx, i_endidx, &
                           nrdmax, cfl_w_limit, scalfac_exdiff, dtime, &
                           c_lin_e, z_w_con_c_full, ddqz_z_full_e, area_edge, &
                           geofac_grdiv, vn, levelmask, &
                           icidx, icblk, iqidx, iqblk, ddt_vn_apc)
  IMPLICIT NONE
  INTEGER(KIND=4), VALUE :: nproma, nlev, nblks, jb, i_startidx, i_endidx, nrdmax
  REAL(KIND=8), VALUE :: cfl_w_limit, scalfac_exdiff, dtime
  REAL(KIND=8), INTENT(IN)    :: c_lin_e(nproma, 2, nblks)
  REAL(KIND=8), INTENT(IN)    :: z_w_con_c_full(nproma, nlev, nblks)
  REAL(KIND=8), INTENT(IN)    :: ddqz_z_full_e(nproma, nlev, nblks)
  REAL(KIND=8), INTENT(IN)    :: area_edge(nproma, nblks)
  REAL(KIND=8), INTENT(IN)    :: geofac_grdiv(nproma, 5, nblks)
  REAL(KIND=8), INTENT(IN)    :: vn(nproma, nlev, nblks)
  LOGICAL, INTENT(IN)         :: levelmask(nlev)
  INTEGER(KIND=4), INTENT(IN) :: icidx(nproma, nblks, 2)
  INTEGER(KIND=4), INTENT(IN) :: icblk(nproma, nblks, 2)
  INTEGER(KIND=4), INTENT(IN) :: iqidx(nproma, nblks, 4)
  INTEGER(KIND=4), INTENT(IN) :: iqblk(nproma, nblks, 4)
  REAL(KIND=8), INTENT(INOUT) :: ddt_vn_apc(nproma, nlev, nblks)
  INTEGER(KIND=4) :: jk, je
  REAL(KIND=8)    :: w_con_e, difcoef
  DO jk = MAX(3, nrdmax - 2), nlev - 4
    IF (levelmask(jk) .OR. levelmask(jk + 1)) THEN
      DO je = i_startidx, i_endidx
        w_con_e = c_lin_e(je, 1, jb) * z_w_con_c_full(icidx(je, jb, 1), jk, icblk(je, jb, 1)) &
                + c_lin_e(je, 2, jb) * z_w_con_c_full(icidx(je, jb, 2), jk, icblk(je, jb, 2))
        IF (ABS(w_con_e) > cfl_w_limit * ddqz_z_full_e(je, jk, jb)) THEN
          difcoef = scalfac_exdiff * MIN(0.85D0 - cfl_w_limit * dtime, &
                                         ABS(w_con_e) * dtime / ddqz_z_full_e(je, jk, jb) - cfl_w_limit * dtime)
          ddt_vn_apc(je, jk, jb) = ddt_vn_apc(je, jk, jb) + difcoef * area_edge(je, jb) &
            * (geofac_grdiv(je, 1, jb) * vn(je, jk, jb) &
             + geofac_grdiv(je, 2, jb) * vn(iqidx(je, jb, 1), jk, iqblk(je, jb, 1)) &
             + geofac_grdiv(je, 3, jb) * vn(iqidx(je, jb, 2), jk, iqblk(je, jb, 2)) &
             + geofac_grdiv(je, 4, jb) * vn(iqidx(je, jb, 3), jk, iqblk(je, jb, 3)) &
             + geofac_grdiv(je, 5, jb) * vn(iqidx(je, jb, 4), jk, iqblk(je, jb, 4)))
        END IF
      END DO
    END IF
  END DO
END SUBROUTINE icon_loopnest_4
