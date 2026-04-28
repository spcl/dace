!==============================================================================
! E6 / Loopnest 5 — vn_ie horizontal boundary kernel
!
! Lines 127-136 of velocity_advection_preprocessed.f90 (non-nested branch):
!   vn_ie(je,1,jb)      = vn(je,1,jb)
!   z_vt_ie(je,1,jb)    = vt(je,1,jb)
!   z_kin_hor_e(je,1,jb)= 0.5*(vn(je,1,jb)**2 + vt(je,1,jb)**2)
!   vn_ie(je,nlevp1,jb) = wgtfacq_e(je,1,jb)*vn(je,nlev,jb)
!                       + wgtfacq_e(je,2,jb)*vn(je,nlev-1,jb)
!                       + wgtfacq_e(je,3,jb)*vn(je,nlev-2,jb)
!
! Single-level (je only) kernel — no jk loop.  Canonical horizontal-only
! boundary pattern.
!==============================================================================

module loopnest_5_mod
  implicit none

  type :: metrics_t
    real(8), allocatable :: wgtfacq_e(:,:,:)   ! (nproma, 3, nblks_e)
  end type
  type :: prog_t
    real(8), allocatable :: vn(:,:,:)
  end type
  type :: diag_t
    real(8), allocatable :: vt(:,:,:), vn_ie(:,:,:)
  end type

contains

  subroutine kernel_struct(p_metrics, p_prog, p_diag, z_vt_ie, z_kin_hor_e, &
                           nlev, nlevp1, i_startblk, i_endblk, i_startidx, i_endidx)
    type(metrics_t), intent(in)    :: p_metrics
    type(prog_t),    intent(in)    :: p_prog
    type(diag_t),    intent(inout) :: p_diag
    real(8),         intent(inout) :: z_vt_ie(:,:,:), z_kin_hor_e(:,:,:)
    integer,         intent(in)    :: nlev, nlevp1, i_startblk, i_endblk, i_startidx, i_endidx
    integer :: jb, je
    do jb = i_startblk, i_endblk
      do je = i_startidx, i_endidx
        p_diag%vn_ie(je, 1, jb)      = p_prog%vn(je, 1, jb)
        z_vt_ie(je, 1, jb)           = p_diag%vt(je, 1, jb)
        z_kin_hor_e(je, 1, jb)       = 0.5d0 * (p_prog%vn(je,1,jb)**2 + p_diag%vt(je,1,jb)**2)
        p_diag%vn_ie(je, nlevp1, jb) = &
            p_metrics%wgtfacq_e(je,1,jb)*p_prog%vn(je,nlev,  jb) + &
            p_metrics%wgtfacq_e(je,2,jb)*p_prog%vn(je,nlev-1,jb) + &
            p_metrics%wgtfacq_e(je,3,jb)*p_prog%vn(je,nlev-2,jb)
      end do
    end do
  end subroutine

  subroutine kernel_flat(vn, vt, wgtfacq_e, vn_ie, z_vt_ie, z_kin_hor_e, &
                         nproma, nlev, nlevp1, nblks_e, &
                         i_startblk, i_endblk, i_startidx, i_endidx)
    integer, intent(in)    :: nproma, nlev, nlevp1, nblks_e, i_startblk, i_endblk, i_startidx, i_endidx
    real(8), intent(in)    :: vn(nproma, nlev, nblks_e), vt(nproma, nlev, nblks_e)
    real(8), intent(in)    :: wgtfacq_e(nproma, 3, nblks_e)
    real(8), intent(inout) :: vn_ie(nproma, nlevp1, nblks_e)
    real(8), intent(inout) :: z_vt_ie(nproma, nlevp1, nblks_e), z_kin_hor_e(nproma, nlevp1, nblks_e)
    integer :: jb, je
    do jb = i_startblk, i_endblk
      do je = i_startidx, i_endidx
        vn_ie(je, 1, jb)       = vn(je, 1, jb)
        z_vt_ie(je, 1, jb)     = vt(je, 1, jb)
        z_kin_hor_e(je, 1, jb) = 0.5d0 * (vn(je,1,jb)**2 + vt(je,1,jb)**2)
        vn_ie(je, nlevp1, jb)  = wgtfacq_e(je,1,jb)*vn(je,nlev,jb) &
                               + wgtfacq_e(je,2,jb)*vn(je,nlev-1,jb) &
                               + wgtfacq_e(je,3,jb)*vn(je,nlev-2,jb)
      end do
    end do
  end subroutine

end module

program loopnest_5_bench
  use loopnest_5_mod
  implicit none
  integer, parameter :: nproma = 32, nlev = 16, nlevp1 = nlev + 1, nblks_e = 8
  integer, parameter :: i_startidx = 1, i_endidx = nproma, i_startblk = 1, i_endblk = nblks_e
  real(8), parameter :: TOL = 1.0d-12

  real(8), allocatable :: vn(:,:,:), vt(:,:,:), wgtfacq_e(:,:,:)
  real(8), allocatable :: vn_ie_s(:,:,:), vn_ie_f(:,:,:)
  real(8), allocatable :: z_vt_s(:,:,:), z_vt_f(:,:,:)
  real(8), allocatable :: z_k_s(:,:,:), z_k_f(:,:,:)
  type(metrics_t) :: p_metrics; type(prog_t) :: p_prog; type(diag_t) :: p_diag
  integer :: sz; integer, allocatable :: seed(:); real(8) :: err

  call random_seed(size=sz); allocate(seed(sz)); seed = 5_4; call random_seed(put=seed)
  allocate(vn(nproma,nlev,nblks_e), vt(nproma,nlev,nblks_e))
  allocate(wgtfacq_e(nproma,3,nblks_e))
  allocate(vn_ie_s(nproma,nlevp1,nblks_e), vn_ie_f(nproma,nlevp1,nblks_e))
  allocate(z_vt_s (nproma,nlevp1,nblks_e), z_vt_f (nproma,nlevp1,nblks_e))
  allocate(z_k_s  (nproma,nlevp1,nblks_e), z_k_f  (nproma,nlevp1,nblks_e))
  call random_number(vn); call random_number(vt); call random_number(wgtfacq_e)
  vn_ie_s = 0.0d0; vn_ie_f = 0.0d0; z_vt_s = 0.0d0; z_vt_f = 0.0d0; z_k_s = 0.0d0; z_k_f = 0.0d0

  allocate(p_metrics%wgtfacq_e(nproma,3,nblks_e));  p_metrics%wgtfacq_e = wgtfacq_e
  allocate(p_prog%vn(nproma,nlev,nblks_e));         p_prog%vn           = vn
  allocate(p_diag%vt(nproma,nlev,nblks_e));         p_diag%vt           = vt
  allocate(p_diag%vn_ie(nproma,nlevp1,nblks_e));    p_diag%vn_ie        = 0.0d0

  call kernel_struct(p_metrics, p_prog, p_diag, z_vt_s, z_k_s, &
                     nlev, nlevp1, i_startblk, i_endblk, i_startidx, i_endidx)
  vn_ie_s = p_diag%vn_ie
  call kernel_flat(vn, vt, wgtfacq_e, vn_ie_f, z_vt_f, z_k_f, &
                   nproma, nlev, nlevp1, nblks_e, &
                   i_startblk, i_endblk, i_startidx, i_endidx)

  err = max(maxval(abs(vn_ie_s - vn_ie_f)), maxval(abs(z_vt_s - z_vt_f)), maxval(abs(z_k_s - z_k_f)))
  if (err > TOL) then; print *, "FAIL", err; stop 1; end if
  print *, "OK max_err=", err
end program
