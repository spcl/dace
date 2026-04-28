!==============================================================================
! E6 / Loopnest 4 — ddt_vn_apc_pc indirect stencil (non-deepatmo branch)
!
! Lines 309-317 of velocity_advection_preprocessed.f90:
!   ddt_vn_apc_pc(je,jk,jb,ntnd) = -( z_kin_hor_e*(cg1 - cg2)
!       + cg2*z_ekinh(icidx2, jk, icblk2) - cg1*z_ekinh(icidx1, jk, icblk1)
!       + vt*( f_e + 0.5*(zeta(ividx1,...) + zeta(ividx2,...)) )
!       + (c1*z_w_con_c_full(icidx1,...) + c2*z_w_con_c_full(icidx2,...))
!         * (vn_ie(jk) - vn_ie(jk+1)) / ddqz_z_full_e )
!==============================================================================

module loopnest_4_mod
  implicit none

  type :: edges_t
    real(8), allocatable :: f_e(:,:)
  end type
  type :: patch_t
    type(edges_t) :: edges
  end type
  type :: int_t
    real(8), allocatable :: c_lin_e(:,:,:)        ! (nproma, 2, nblks_e)
  end type
  type :: diag_t
    real(8), allocatable :: vt(:,:,:), vn_ie(:,:,:), ddt_vn_apc_pc(:,:,:,:)
  end type
  type :: metrics_t
    real(8), allocatable :: coeff_gradekin(:,:,:) ! (nproma, 2, nblks_e)
    real(8), allocatable :: ddqz_z_full_e(:,:,:)
  end type

contains

  subroutine kernel_struct(p_patch, p_int, p_diag, p_metrics, &
                           z_kin_hor_e, z_ekinh, zeta, z_w_con_c_full, &
                           icidx, icblk, ividx, ivblk, ntnd, nlev, &
                           i_startblk, i_endblk, i_startidx, i_endidx)
    type(patch_t),   intent(in)    :: p_patch
    type(int_t),     intent(in)    :: p_int
    type(diag_t),    intent(inout) :: p_diag
    type(metrics_t), intent(in)    :: p_metrics
    real(8),         intent(in)    :: z_kin_hor_e(:,:,:), z_ekinh(:,:,:), zeta(:,:,:), z_w_con_c_full(:,:,:)
    integer,         intent(in)    :: icidx(:,:,:), icblk(:,:,:), ividx(:,:,:), ivblk(:,:,:)
    integer,         intent(in)    :: ntnd, nlev, i_startblk, i_endblk, i_startidx, i_endidx
    integer :: jb, jk, je, c1i, c2i, c1b, c2b, v1i, v2i, v1b, v2b
    do jb = i_startblk, i_endblk
      do jk = 1, nlev
        do je = i_startidx, i_endidx
          c1i = icidx(je, jb, 1); c2i = icidx(je, jb, 2)
          c1b = icblk(je, jb, 1); c2b = icblk(je, jb, 2)
          v1i = ividx(je, jb, 1); v2i = ividx(je, jb, 2)
          v1b = ivblk(je, jb, 1); v2b = ivblk(je, jb, 2)
          p_diag%ddt_vn_apc_pc(je, jk, jb, ntnd) = -( &
              z_kin_hor_e(je, jk, jb) * (p_metrics%coeff_gradekin(je,1,jb) - p_metrics%coeff_gradekin(je,2,jb)) &
            + p_metrics%coeff_gradekin(je,2,jb) * z_ekinh(c2i, jk, c2b) &
            - p_metrics%coeff_gradekin(je,1,jb) * z_ekinh(c1i, jk, c1b) &
            + p_diag%vt(je,jk,jb) * ( p_patch%edges%f_e(je,jb) + 0.5d0 * &
              (zeta(v1i, jk, v1b) + zeta(v2i, jk, v2b)) ) &
            + (p_int%c_lin_e(je,1,jb) * z_w_con_c_full(c1i, jk, c1b) + &
               p_int%c_lin_e(je,2,jb) * z_w_con_c_full(c2i, jk, c2b)) * &
              (p_diag%vn_ie(je, jk, jb) - p_diag%vn_ie(je, jk+1, jb)) / p_metrics%ddqz_z_full_e(je, jk, jb) )
        end do
      end do
    end do
  end subroutine

  subroutine kernel_flat(vt, vn_ie, f_e, coeff_gradekin, c_lin_e, ddqz, &
                         z_kin_hor_e, z_ekinh, zeta, z_w_con_c_full, &
                         icidx, icblk, ividx, ivblk, ddt_vn_apc_pc, ntnd, &
                         nproma, nlev, nblks_e, nblks_c, nblks_v, nproma_tnd, &
                         i_startblk, i_endblk, i_startidx, i_endidx)
    integer, intent(in) :: ntnd, nproma, nlev, nblks_e, nblks_c, nblks_v, nproma_tnd
    integer, intent(in) :: i_startblk, i_endblk, i_startidx, i_endidx
    real(8), intent(in) :: vt(nproma, nlev, nblks_e), vn_ie(nproma, nlev+1, nblks_e)
    real(8), intent(in) :: f_e(nproma, nblks_e)
    real(8), intent(in) :: coeff_gradekin(nproma, 2, nblks_e), c_lin_e(nproma, 2, nblks_e)
    real(8), intent(in) :: ddqz(nproma, nlev, nblks_e)
    real(8), intent(in) :: z_kin_hor_e(nproma, nlev, nblks_e)
    real(8), intent(in) :: z_ekinh(nproma, nlev, nblks_c)
    real(8), intent(in) :: zeta(nproma, nlev, nblks_v)
    real(8), intent(in) :: z_w_con_c_full(nproma, nlev, nblks_c)
    integer, intent(in) :: icidx(nproma, nblks_e, 2), icblk(nproma, nblks_e, 2)
    integer, intent(in) :: ividx(nproma, nblks_e, 2), ivblk(nproma, nblks_e, 2)
    real(8), intent(inout) :: ddt_vn_apc_pc(nproma, nlev, nblks_e, nproma_tnd)
    integer :: jb, jk, je, c1i, c2i, c1b, c2b, v1i, v2i, v1b, v2b
    do jb = i_startblk, i_endblk
      do jk = 1, nlev
        do je = i_startidx, i_endidx
          c1i = icidx(je, jb, 1); c2i = icidx(je, jb, 2)
          c1b = icblk(je, jb, 1); c2b = icblk(je, jb, 2)
          v1i = ividx(je, jb, 1); v2i = ividx(je, jb, 2)
          v1b = ivblk(je, jb, 1); v2b = ivblk(je, jb, 2)
          ddt_vn_apc_pc(je, jk, jb, ntnd) = -( &
              z_kin_hor_e(je,jk,jb)*(coeff_gradekin(je,1,jb) - coeff_gradekin(je,2,jb)) &
            + coeff_gradekin(je,2,jb)*z_ekinh(c2i,jk,c2b) &
            - coeff_gradekin(je,1,jb)*z_ekinh(c1i,jk,c1b) &
            + vt(je,jk,jb)*(f_e(je,jb) + 0.5d0*(zeta(v1i,jk,v1b) + zeta(v2i,jk,v2b))) &
            + (c_lin_e(je,1,jb)*z_w_con_c_full(c1i,jk,c1b) + c_lin_e(je,2,jb)*z_w_con_c_full(c2i,jk,c2b)) &
              * (vn_ie(je,jk,jb) - vn_ie(je,jk+1,jb)) / ddqz(je,jk,jb) )
        end do
      end do
    end do
  end subroutine

  subroutine random_indirection_3d(idx, rng)
    integer, intent(out) :: idx(:,:,:)
    integer, intent(in)  :: rng
    real(8) :: r
    integer :: i, j, k
    do k = 1, size(idx,3); do j = 1, size(idx,2); do i = 1, size(idx,1)
      call random_number(r); idx(i,j,k) = max(1, min(rng, 1 + int(r*rng)))
    end do; end do; end do
  end subroutine

end module

program loopnest_4_bench
  use loopnest_4_mod
  implicit none
  integer, parameter :: nproma = 32, nlev = 16, nblks_e = 8, nblks_c = 8, nblks_v = 8, nproma_tnd = 3
  integer, parameter :: ntnd = 2, i_startidx = 1, i_endidx = nproma, i_startblk = 1, i_endblk = nblks_e
  real(8), parameter :: TOL = 1.0d-12

  real(8), allocatable :: vt(:,:,:), vn_ie(:,:,:), f_e(:,:)
  real(8), allocatable :: coeff_gradekin(:,:,:), c_lin_e(:,:,:), ddqz(:,:,:)
  real(8), allocatable :: z_kin_hor_e(:,:,:), z_ekinh(:,:,:), zeta(:,:,:), z_w_con_c_full(:,:,:)
  integer, allocatable :: icidx(:,:,:), icblk(:,:,:), ividx(:,:,:), ivblk(:,:,:)
  real(8), allocatable :: ddt_s(:,:,:,:), ddt_f(:,:,:,:)
  type(patch_t)   :: p_patch
  type(int_t)     :: p_int
  type(diag_t)    :: p_diag
  type(metrics_t) :: p_metrics
  integer :: sz; integer, allocatable :: seed(:); real(8) :: err

  call random_seed(size=sz); allocate(seed(sz)); seed = 4_4; call random_seed(put=seed)
  allocate(vt(nproma,nlev,nblks_e), vn_ie(nproma,nlev+1,nblks_e), f_e(nproma,nblks_e))
  allocate(coeff_gradekin(nproma,2,nblks_e), c_lin_e(nproma,2,nblks_e))
  allocate(ddqz(nproma,nlev,nblks_e))
  allocate(z_kin_hor_e(nproma,nlev,nblks_e), z_ekinh(nproma,nlev,nblks_c))
  allocate(zeta(nproma,nlev,nblks_v), z_w_con_c_full(nproma,nlev,nblks_c))
  allocate(icidx(nproma,nblks_e,2), icblk(nproma,nblks_e,2))
  allocate(ividx(nproma,nblks_e,2), ivblk(nproma,nblks_e,2))
  allocate(ddt_s(nproma,nlev,nblks_e,nproma_tnd), ddt_f(nproma,nlev,nblks_e,nproma_tnd))

  call random_number(vt); call random_number(vn_ie); call random_number(f_e)
  call random_number(coeff_gradekin); call random_number(c_lin_e); call random_number(ddqz)
  ddqz = ddqz + 0.1d0  ! avoid near-zero denominators
  call random_number(z_kin_hor_e); call random_number(z_ekinh)
  call random_number(zeta); call random_number(z_w_con_c_full)
  call random_indirection_3d(icidx, nproma)
  call random_indirection_3d(icblk, nblks_c)
  call random_indirection_3d(ividx, nproma)
  call random_indirection_3d(ivblk, nblks_v)

  allocate(p_patch%edges%f_e(nproma,nblks_e));             p_patch%edges%f_e             = f_e
  allocate(p_int%c_lin_e(nproma,2,nblks_e));               p_int%c_lin_e                 = c_lin_e
  allocate(p_diag%vt(nproma,nlev,nblks_e));                p_diag%vt                     = vt
  allocate(p_diag%vn_ie(nproma,nlev+1,nblks_e));           p_diag%vn_ie                  = vn_ie
  allocate(p_metrics%coeff_gradekin(nproma,2,nblks_e));    p_metrics%coeff_gradekin      = coeff_gradekin
  allocate(p_metrics%ddqz_z_full_e(nproma,nlev,nblks_e));  p_metrics%ddqz_z_full_e       = ddqz

  ddt_s = 0.0d0; ddt_f = 0.0d0
  allocate(p_diag%ddt_vn_apc_pc(nproma,nlev,nblks_e,nproma_tnd)); p_diag%ddt_vn_apc_pc = 0.0d0
  call kernel_struct(p_patch, p_int, p_diag, p_metrics, &
                     z_kin_hor_e, z_ekinh, zeta, z_w_con_c_full, &
                     icidx, icblk, ividx, ivblk, ntnd, nlev, &
                     i_startblk, i_endblk, i_startidx, i_endidx)
  ddt_s = p_diag%ddt_vn_apc_pc
  call kernel_flat(vt, vn_ie, f_e, coeff_gradekin, c_lin_e, ddqz, &
                   z_kin_hor_e, z_ekinh, zeta, z_w_con_c_full, &
                   icidx, icblk, ividx, ivblk, ddt_f, ntnd, &
                   nproma, nlev, nblks_e, nblks_c, nblks_v, nproma_tnd, &
                   i_startblk, i_endblk, i_startidx, i_endidx)

  err = maxval(abs(ddt_s - ddt_f))
  if (err > TOL) then; print *, "FAIL", err; stop 1; end if
  print *, "OK max_err=", err
end program
