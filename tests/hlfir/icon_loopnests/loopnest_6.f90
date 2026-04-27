! ICON velocity_advection loopnest 6 - vertical-only (level reduction).
! Pattern key: ['v', ('partial_vert',), 'compute', 'v:S'].
! Selected by ``select_loopnests.py`` (chosen_loopnests.md, loopnest_6).
!
! Source snippet:
!   DO jk = MAX(3, nrdmax_jg - 2), nlev - 3
!     levelmask(jk) = ANY(levmask(i_startblk:i_endblk, jk))
!   END DO
!
! ``levelmask`` and ``levmask`` are LOGICAL in ICON.  The bridge handles
! LOGICAL via 1-byte storage (uint8 numpy dtype); the test passes them
! as the equivalent ``LOGICAL(KIND=1)`` arrays.
SUBROUTINE icon_loopnest_6(nlev, nblks, nrdmax, i_startblk, i_endblk, levmask, levelmask)
  IMPLICIT NONE
  INTEGER(KIND=4), VALUE :: nlev, nblks, nrdmax, i_startblk, i_endblk
  LOGICAL, INTENT(IN)    :: levmask(nblks, nlev)
  LOGICAL, INTENT(INOUT) :: levelmask(nlev)
  INTEGER(KIND=4) :: jk
  DO jk = MAX(3, nrdmax - 2), nlev - 3
    levelmask(jk) = ANY(levmask(i_startblk:i_endblk, jk))
  END DO
END SUBROUTINE icon_loopnest_6
