!==============================================================================
! E6 / Loopnest 3 — z_v_grad_w direct stencil, full vertical (deepatmo extension)
!
! Lines 174-178 of velocity_advection_preprocessed.f90:
!   z_v_grad_w = z_v_grad_w*deepatmo_gradh_ifc(jk)
!              + vn_ie*(vn_ie*deepatmo_invr_ifc(jk) - ft_e)
!              + z_vt_ie*(z_vt_ie*deepatmo_invr_ifc(jk) + fn_e)
!
! Representative of the "direct stencil, full vertical" class: no
! horizontal indirection, three (je,jb) coefficient arrays combined with
! one (jk)-only vertical profile.
!==============================================================================

module loopnest_3_mod
  implicit none

  type :: edges_t
    real(8), allocatable :: ft_e(:,:), fn_e(:,:)
  end type
  type :: patch_t
    type(edges_t) :: edges
  end type
  type :: diag_t
    real(8), allocatable :: vn_ie(:,:,:)
  end type
  type :: metrics_t
    real(8), allocatable :: deepatmo_gradh_ifc(:), deepatmo_invr_ifc(:)
  end type

contains

  subroutine kernel_struct(p_patch, p_diag, p_metrics, z_vt_ie, z_v_grad_w, &
                           nlev, i_startblk, i_endblk, i_startidx, i_endidx)
    type(patch_t),   intent(in)    :: p_patch
    type(diag_t),    intent(in)    :: p_diag
    type(metrics_t), intent(in)    :: p_metrics
    real(8),         intent(in)    :: z_vt_ie(:,:,:)
    real(8),         intent(inout) :: z_v_grad_w(:,:,:)
    integer,         intent(in)    :: nlev, i_startblk, i_endblk, i_startidx, i_endidx
    integer :: jb, jk, je
    do jb = i_startblk, i_endblk
      do jk = 1, nlev
        do je = i_startidx, i_endidx
          z_v_grad_w(je, jk, jb) = &
              z_v_grad_w(je, jk, jb) * p_metrics%deepatmo_gradh_ifc(jk) &
            + p_diag%vn_ie(je, jk, jb) * &
              ( p_diag%vn_ie(je, jk, jb) * p_metrics%deepatmo_invr_ifc(jk) - p_patch%edges%ft_e(je, jb) ) &
            + z_vt_ie(je, jk, jb) * &
              ( z_vt_ie(je, jk, jb) * p_metrics%deepatmo_invr_ifc(jk) + p_patch%edges%fn_e(je, jb) )
        end do
      end do
    end do
  end subroutine

  subroutine kernel_flat(vn_ie, z_vt_ie, ft_e, fn_e, gradh, invr, z_v_grad_w, &
                         nproma, nlev, nblks_e, i_startblk, i_endblk, i_startidx, i_endidx)
    integer, intent(in)    :: nproma, nlev, nblks_e, i_startblk, i_endblk, i_startidx, i_endidx
    real(8), intent(in)    :: vn_ie(nproma, nlev, nblks_e), z_vt_ie(nproma, nlev, nblks_e)
    real(8), intent(in)    :: ft_e(nproma, nblks_e), fn_e(nproma, nblks_e)
    real(8), intent(in)    :: gradh(nlev), invr(nlev)
    real(8), intent(inout) :: z_v_grad_w(nproma, nlev, nblks_e)
    integer :: jb, jk, je
    do jb = i_startblk, i_endblk
      do jk = 1, nlev
        do je = i_startidx, i_endidx
          z_v_grad_w(je, jk, jb) = z_v_grad_w(je, jk, jb) * gradh(jk) &
              + vn_ie(je, jk, jb) * (vn_ie(je, jk, jb)*invr(jk) - ft_e(je, jb)) &
              + z_vt_ie(je, jk, jb) * (z_vt_ie(je, jk, jb)*invr(jk) + fn_e(je, jb))
        end do
      end do
    end do
  end subroutine

end module

program loopnest_3_bench
  use loopnest_3_mod
  implicit none
  integer, parameter :: nproma = 32, nlev = 16, nblks_e = 8
  integer, parameter :: i_startidx = 1, i_endidx = nproma, i_startblk = 1, i_endblk = nblks_e
  real(8), parameter :: TOL = 1.0d-12

  real(8), allocatable :: vn_ie(:,:,:), z_vt_ie(:,:,:), ft_e(:,:), fn_e(:,:)
  real(8), allocatable :: gradh(:), invr(:)
  real(8), allocatable :: zs(:,:,:), zf(:,:,:), zinit(:,:,:)
  type(patch_t)   :: p_patch
  type(diag_t)    :: p_diag
  type(metrics_t) :: p_metrics
  integer :: sz; integer, allocatable :: seed(:); real(8) :: err

  call random_seed(size=sz); allocate(seed(sz)); seed = 3_4; call random_seed(put=seed)
  allocate(vn_ie(nproma,nlev,nblks_e), z_vt_ie(nproma,nlev,nblks_e))
  allocate(ft_e(nproma,nblks_e), fn_e(nproma,nblks_e))
  allocate(gradh(nlev), invr(nlev))
  allocate(zs(nproma,nlev,nblks_e), zf(nproma,nlev,nblks_e), zinit(nproma,nlev,nblks_e))
  call random_number(vn_ie); call random_number(z_vt_ie)
  call random_number(ft_e); call random_number(fn_e)
  call random_number(gradh); call random_number(invr); call random_number(zinit)
  zs = zinit; zf = zinit

  allocate(p_patch%edges%ft_e(nproma,nblks_e)); p_patch%edges%ft_e = ft_e
  allocate(p_patch%edges%fn_e(nproma,nblks_e)); p_patch%edges%fn_e = fn_e
  allocate(p_diag%vn_ie(nproma,nlev,nblks_e));  p_diag%vn_ie = vn_ie
  allocate(p_metrics%deepatmo_gradh_ifc(nlev)); p_metrics%deepatmo_gradh_ifc = gradh
  allocate(p_metrics%deepatmo_invr_ifc(nlev));  p_metrics%deepatmo_invr_ifc  = invr

  call kernel_struct(p_patch, p_diag, p_metrics, z_vt_ie, zs, nlev, i_startblk, i_endblk, i_startidx, i_endidx)
  call kernel_flat(vn_ie, z_vt_ie, ft_e, fn_e, gradh, invr, zf, &
                   nproma, nlev, nblks_e, i_startblk, i_endblk, i_startidx, i_endidx)

  err = maxval(abs(zs - zf))
  if (err > TOL) then; print *, "FAIL", err; stop 1; end if
  print *, "OK max_err=", err
end program
