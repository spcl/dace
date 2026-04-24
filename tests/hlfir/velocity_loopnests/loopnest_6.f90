!==============================================================================
! E6 / Loopnest 6 — levelmask vertical reduction
!
! Line 297 of velocity_advection_preprocessed.f90:
!   levelmask(jk) = ANY(levmask(i_startblk:i_endblk, jk))
!
! A per-level logical OR across a horizontal-block slice — canonical
! vertical-only reduction pattern.  There are no derived types on the
! data path here, so the "struct" variant exercises a 1D-derived-type
! wrapper to keep the struct-vs-flat comparison meaningful.
!==============================================================================

module loopnest_6_mod
  implicit none

  type :: mask_wrap_t
    logical, allocatable :: levmask(:,:)        ! (nblks_c, nlev)
    logical, allocatable :: levelmask(:)        ! (nlev)
  end type

contains

  subroutine kernel_struct(m, jk_start, jk_end, i_startblk, i_endblk)
    type(mask_wrap_t), intent(inout) :: m
    integer, intent(in) :: jk_start, jk_end, i_startblk, i_endblk
    integer :: jk
    do jk = jk_start, jk_end
      m%levelmask(jk) = ANY(m%levmask(i_startblk:i_endblk, jk))
    end do
  end subroutine

  subroutine kernel_flat(levmask, levelmask, nlev, nblks_c, jk_start, jk_end, i_startblk, i_endblk)
    integer, intent(in)    :: nlev, nblks_c, jk_start, jk_end, i_startblk, i_endblk
    logical, intent(in)    :: levmask(nblks_c, nlev)
    logical, intent(inout) :: levelmask(nlev)
    integer :: jk
    do jk = jk_start, jk_end
      levelmask(jk) = ANY(levmask(i_startblk:i_endblk, jk))
    end do
  end subroutine

end module

program loopnest_6_bench
  use loopnest_6_mod
  implicit none
  integer, parameter :: nlev = 64, nblks_c = 12, i_startblk = 2, i_endblk = 10
  integer, parameter :: jk_start = 3, jk_end = nlev - 3
  real(8), parameter :: TOL = 0.0d0   ! logical == means bit-exact

  logical, allocatable :: levmask(:,:), lm_s(:), lm_f(:)
  real(8) :: r
  integer :: sz, jk, jb
  integer, allocatable :: seed(:)
  type(mask_wrap_t) :: m
  logical :: mismatch

  call random_seed(size=sz); allocate(seed(sz)); seed = 6_4; call random_seed(put=seed)
  allocate(levmask(nblks_c, nlev))
  do jk = 1, nlev
    do jb = 1, nblks_c
      call random_number(r)
      levmask(jb, jk) = (r > 0.7d0)  ! sparse true — a realistic boundary flag
    end do
  end do
  allocate(lm_s(nlev)); lm_s = .false.
  allocate(lm_f(nlev)); lm_f = .false.

  allocate(m%levmask(nblks_c, nlev));  m%levmask   = levmask
  allocate(m%levelmask(nlev));         m%levelmask = .false.
  call kernel_struct(m, jk_start, jk_end, i_startblk, i_endblk)
  lm_s = m%levelmask

  call kernel_flat(levmask, lm_f, nlev, nblks_c, jk_start, jk_end, i_startblk, i_endblk)

  mismatch = any(lm_s .neqv. lm_f)
  if (mismatch) then; print *, "FAIL logical mismatch"; stop 1; end if
  print *, "OK (logical equality)"
end program
