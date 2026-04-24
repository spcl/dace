!==============================================================================
! E6 / Loopnest 2 — z_w_concorr_me direct stencil, partial vertical
!
! Lines 120-125 of velocity_advection_preprocessed.f90, jk=nflatlev..nlev:
!   z_w_concorr_me(je,jk,jb) = p_prog%vn(je,jk,jb)*p_metrics%ddxn_z_full(je,jk,jb)
!                            + p_diag%vt(je,jk,jb)*p_metrics%ddxt_z_full(je,jk,jb)
!==============================================================================

module loopnest_2_mod
  implicit none

  type :: patch_t
    integer :: pad = 0  ! placeholder — real p_patch unused by this kernel body
  end type

  type :: prog_t
    real(8), allocatable :: vn(:,:,:)
  end type

  type :: diag_t
    real(8), allocatable :: vt(:,:,:)
  end type

  type :: metrics_t
    real(8), allocatable :: ddxn_z_full(:,:,:)
    real(8), allocatable :: ddxt_z_full(:,:,:)
  end type

contains

  subroutine kernel_struct(p_prog, p_diag, p_metrics, z_w_concorr_me, &
                           nlev, nflatlev, i_startblk, i_endblk, i_startidx, i_endidx)
    type(prog_t),    intent(in)    :: p_prog
    type(diag_t),    intent(in)    :: p_diag
    type(metrics_t), intent(in)    :: p_metrics
    real(8),         intent(inout) :: z_w_concorr_me(:,:,:)
    integer,         intent(in)    :: nlev, nflatlev, i_startblk, i_endblk, i_startidx, i_endidx
    integer :: jb, jk, je
    do jb = i_startblk, i_endblk
      do jk = nflatlev, nlev
        do je = i_startidx, i_endidx
          z_w_concorr_me(je, jk, jb) = &
              p_prog%vn(je, jk, jb) * p_metrics%ddxn_z_full(je, jk, jb) + &
              p_diag%vt(je, jk, jb) * p_metrics%ddxt_z_full(je, jk, jb)
        end do
      end do
    end do
  end subroutine

  subroutine kernel_flat(vn, vt, ddxn, ddxt, z_w_concorr_me, &
                         nproma, nlev, nblks_e, nflatlev, &
                         i_startblk, i_endblk, i_startidx, i_endidx)
    integer, intent(in)    :: nproma, nlev, nblks_e, nflatlev
    integer, intent(in)    :: i_startblk, i_endblk, i_startidx, i_endidx
    real(8), intent(in)    :: vn(nproma, nlev, nblks_e), vt(nproma, nlev, nblks_e)
    real(8), intent(in)    :: ddxn(nproma, nlev, nblks_e), ddxt(nproma, nlev, nblks_e)
    real(8), intent(inout) :: z_w_concorr_me(nproma, nlev, nblks_e)
    integer :: jb, jk, je
    do jb = i_startblk, i_endblk
      do jk = nflatlev, nlev
        do je = i_startidx, i_endidx
          z_w_concorr_me(je, jk, jb) = vn(je,jk,jb)*ddxn(je,jk,jb) + vt(je,jk,jb)*ddxt(je,jk,jb)
        end do
      end do
    end do
  end subroutine

end module

program loopnest_2_bench
  use loopnest_2_mod
  implicit none
  integer, parameter :: nproma = 32, nlev = 16, nblks_e = 8, nflatlev = 4
  integer, parameter :: i_startidx = 1, i_endidx = nproma, i_startblk = 1, i_endblk = nblks_e
  real(8), parameter :: TOL = 1.0d-12

  real(8), allocatable :: vn(:,:,:), vt(:,:,:), ddxn(:,:,:), ddxt(:,:,:)
  real(8), allocatable :: z_s(:,:,:), z_f(:,:,:)
  type(prog_t)    :: p_prog
  type(diag_t)    :: p_diag
  type(metrics_t) :: p_metrics
  integer :: sz
  integer, allocatable :: seed(:)
  real(8) :: err

  call random_seed(size=sz); allocate(seed(sz)); seed = 2_4; call random_seed(put=seed)

  allocate(vn  (nproma, nlev, nblks_e), vt  (nproma, nlev, nblks_e))
  allocate(ddxn(nproma, nlev, nblks_e), ddxt(nproma, nlev, nblks_e))
  allocate(z_s (nproma, nlev, nblks_e), z_f (nproma, nlev, nblks_e))
  call random_number(vn); call random_number(vt)
  call random_number(ddxn); call random_number(ddxt)

  allocate(p_prog%vn           (nproma, nlev, nblks_e)); p_prog%vn          = vn
  allocate(p_diag%vt           (nproma, nlev, nblks_e)); p_diag%vt          = vt
  allocate(p_metrics%ddxn_z_full(nproma, nlev, nblks_e)); p_metrics%ddxn_z_full = ddxn
  allocate(p_metrics%ddxt_z_full(nproma, nlev, nblks_e)); p_metrics%ddxt_z_full = ddxt

  z_s = 0.0d0; z_f = 0.0d0
  call kernel_struct(p_prog, p_diag, p_metrics, z_s, nlev, nflatlev, i_startblk, i_endblk, i_startidx, i_endidx)
  call kernel_flat(vn, vt, ddxn, ddxt, z_f, nproma, nlev, nblks_e, nflatlev, &
                   i_startblk, i_endblk, i_startidx, i_endidx)

  err = maxval(abs(z_s - z_f))
  if (err > TOL) then; print *, "FAIL", err; stop 1; end if
  print *, "OK max_err=", err
end program
