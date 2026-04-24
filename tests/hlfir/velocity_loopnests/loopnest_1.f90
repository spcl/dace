!==============================================================================
! E6 / Loopnest 1 — z_v_grad_w indirect stencil
!
! Representative of the "indirect stencil" pattern class from ICON's
! velocity_advection::velocity_tendencies (lines 160-164 of
! velocity_advection_preprocessed.f90).
!
! Body per (je, jk, jb) with icidx/icblk/ividx/ivblk 3D indirection arrays:
!   z_v_grad_w(je,jk,jb) =
!       vn_ie(je,jk,jb) * inv_dual(je,jb) *
!         (w(icidx(je,jb,1), jk, icblk(je,jb,1)) -
!          w(icidx(je,jb,2), jk, icblk(je,jb,2)))
!     + z_vt_ie(je,jk,jb) * inv_primal(je,jb) * tangent(je,jb) *
!         (z_w_v(ividx(je,jb,1), jk, ivblk(je,jb,1)) -
!          z_w_v(ividx(je,jb,2), jk, ivblk(je,jb,2)))
!
! Ships a struct-typed and a flattened implementation and a driver that
! runs both against the same inputs and exits 0 iff their outputs
! match to within 1e-12.
!==============================================================================

module loopnest_1_mod
  implicit none

  type :: patch_edges_t
    real(8), allocatable :: inv_dual_edge_length(:,:)
    real(8), allocatable :: inv_primal_edge_length(:,:)
    real(8), allocatable :: tangent_orientation(:,:)
    integer, allocatable :: cell_idx(:,:,:), cell_blk(:,:,:)
    integer, allocatable :: vertex_idx(:,:,:), vertex_blk(:,:,:)
  end type

  type :: patch_t
    type(patch_edges_t) :: edges
  end type

  type :: diag_t
    real(8), allocatable :: vn_ie(:,:,:)
  end type

  type :: prog_t
    real(8), allocatable :: w(:,:,:)
  end type

contains

  subroutine kernel_struct(p_patch, p_prog, p_diag, z_vt_ie, z_w_v, z_v_grad_w, &
                           nlev, i_startblk, i_endblk, i_startidx, i_endidx)
    type(patch_t), intent(in)    :: p_patch
    type(prog_t),  intent(in)    :: p_prog
    type(diag_t),  intent(in)    :: p_diag
    real(8),       intent(in)    :: z_vt_ie(:,:,:), z_w_v(:,:,:)
    real(8),       intent(inout) :: z_v_grad_w(:,:,:)
    integer,       intent(in)    :: nlev, i_startblk, i_endblk, i_startidx, i_endidx
    integer :: jb, jk, je, ci0, ci1, cb0, cb1, vi0, vi1, vb0, vb1
    do jb = i_startblk, i_endblk
      do jk = 1, nlev
        do je = i_startidx, i_endidx
          ci0 = p_patch%edges%cell_idx(je, jb, 1)
          ci1 = p_patch%edges%cell_idx(je, jb, 2)
          cb0 = p_patch%edges%cell_blk(je, jb, 1)
          cb1 = p_patch%edges%cell_blk(je, jb, 2)
          vi0 = p_patch%edges%vertex_idx(je, jb, 1)
          vi1 = p_patch%edges%vertex_idx(je, jb, 2)
          vb0 = p_patch%edges%vertex_blk(je, jb, 1)
          vb1 = p_patch%edges%vertex_blk(je, jb, 2)
          z_v_grad_w(je, jk, jb) = &
               p_diag%vn_ie(je, jk, jb) * p_patch%edges%inv_dual_edge_length(je, jb) * &
               (p_prog%w(ci0, jk, cb0) - p_prog%w(ci1, jk, cb1)) &
             + z_vt_ie(je, jk, jb) * p_patch%edges%inv_primal_edge_length(je, jb) * &
               p_patch%edges%tangent_orientation(je, jb) * &
               (z_w_v(vi0, jk, vb0) - z_w_v(vi1, jk, vb1))
        end do
      end do
    end do
  end subroutine

  subroutine kernel_flat(vn_ie, inv_dual, inv_primal, tangent,  &
                         w, z_vt_ie, z_w_v,                     &
                         icidx, icblk, ividx, ivblk,            &
                         z_v_grad_w,                            &
                         nproma, nlev, nblks_e, nblks_c, nblks_v, &
                         i_startblk, i_endblk, i_startidx, i_endidx)
    integer, intent(in)    :: nproma, nlev, nblks_e, nblks_c, nblks_v
    integer, intent(in)    :: i_startblk, i_endblk, i_startidx, i_endidx
    real(8), intent(in)    :: vn_ie(nproma, nlev, nblks_e)
    real(8), intent(in)    :: inv_dual(nproma, nblks_e), inv_primal(nproma, nblks_e)
    real(8), intent(in)    :: tangent(nproma, nblks_e)
    real(8), intent(in)    :: w(nproma, nlev, nblks_c)
    real(8), intent(in)    :: z_vt_ie(nproma, nlev, nblks_e)
    real(8), intent(in)    :: z_w_v(nproma, nlev, nblks_v)
    integer, intent(in)    :: icidx(nproma, nblks_e, 2), icblk(nproma, nblks_e, 2)
    integer, intent(in)    :: ividx(nproma, nblks_e, 2), ivblk(nproma, nblks_e, 2)
    real(8), intent(inout) :: z_v_grad_w(nproma, nlev, nblks_e)
    integer :: jb, jk, je, ci0, ci1, cb0, cb1, vi0, vi1, vb0, vb1
    do jb = i_startblk, i_endblk
      do jk = 1, nlev
        do je = i_startidx, i_endidx
          ci0 = icidx(je, jb, 1); ci1 = icidx(je, jb, 2)
          cb0 = icblk(je, jb, 1); cb1 = icblk(je, jb, 2)
          vi0 = ividx(je, jb, 1); vi1 = ividx(je, jb, 2)
          vb0 = ivblk(je, jb, 1); vb1 = ivblk(je, jb, 2)
          z_v_grad_w(je, jk, jb) = &
               vn_ie(je, jk, jb) * inv_dual(je, jb) * &
               (w(ci0, jk, cb0) - w(ci1, jk, cb1)) &
             + z_vt_ie(je, jk, jb) * inv_primal(je, jb) * tangent(je, jb) * &
               (z_w_v(vi0, jk, vb0) - z_w_v(vi1, jk, vb1))
        end do
      end do
    end do
  end subroutine

  subroutine random_indirection_3d(idx, rng)
    integer, intent(out) :: idx(:,:,:)
    integer, intent(in)  :: rng
    real(8) :: r
    integer :: i, j, k
    do k = 1, size(idx, 3)
      do j = 1, size(idx, 2)
        do i = 1, size(idx, 1)
          call random_number(r)
          idx(i, j, k) = 1 + int(r * rng)
          if (idx(i, j, k) > rng) idx(i, j, k) = rng
        end do
      end do
    end do
  end subroutine

end module

program loopnest_1_bench
  use loopnest_1_mod
  implicit none

  integer, parameter :: nproma = 32, nlev = 16, nblks_e = 8, nblks_c = 8, nblks_v = 8
  integer, parameter :: i_startidx = 1, i_endidx = nproma
  integer, parameter :: i_startblk = 1, i_endblk = nblks_e
  real(8), parameter :: TOL = 1.0d-12

  real(8), allocatable :: vn_ie(:,:,:), inv_dual(:,:), inv_primal(:,:), tangent(:,:)
  real(8), allocatable :: w(:,:,:), z_vt_ie(:,:,:), z_w_v(:,:,:)
  integer, allocatable :: icidx(:,:,:), icblk(:,:,:), ividx(:,:,:), ivblk(:,:,:)
  real(8), allocatable :: z_v_grad_w_struct(:,:,:), z_v_grad_w_flat(:,:,:)
  type(patch_t) :: p_patch
  type(prog_t)  :: p_prog
  type(diag_t)  :: p_diag
  integer :: seed_size
  integer, allocatable :: seed(:)
  real(8) :: max_err

  call random_seed(size=seed_size)
  allocate(seed(seed_size))
  seed = 1_4
  call random_seed(put=seed)

  allocate(vn_ie(nproma, nlev, nblks_e), z_vt_ie(nproma, nlev, nblks_e))
  allocate(inv_dual(nproma, nblks_e), inv_primal(nproma, nblks_e), tangent(nproma, nblks_e))
  allocate(w(nproma, nlev, nblks_c), z_w_v(nproma, nlev, nblks_v))
  allocate(icidx(nproma, nblks_e, 2), icblk(nproma, nblks_e, 2))
  allocate(ividx(nproma, nblks_e, 2), ivblk(nproma, nblks_e, 2))
  allocate(z_v_grad_w_struct(nproma, nlev, nblks_e))
  allocate(z_v_grad_w_flat  (nproma, nlev, nblks_e))

  call random_number(vn_ie); call random_number(z_vt_ie)
  call random_number(inv_dual); call random_number(inv_primal); call random_number(tangent)
  call random_number(w); call random_number(z_w_v)
  call random_indirection_3d(icidx, nproma)
  call random_indirection_3d(icblk, nblks_c)
  call random_indirection_3d(ividx, nproma)
  call random_indirection_3d(ivblk, nblks_v)

  allocate(p_patch%edges%inv_dual_edge_length  (nproma, nblks_e))
  allocate(p_patch%edges%inv_primal_edge_length(nproma, nblks_e))
  allocate(p_patch%edges%tangent_orientation   (nproma, nblks_e))
  allocate(p_patch%edges%cell_idx  (nproma, nblks_e, 2))
  allocate(p_patch%edges%cell_blk  (nproma, nblks_e, 2))
  allocate(p_patch%edges%vertex_idx(nproma, nblks_e, 2))
  allocate(p_patch%edges%vertex_blk(nproma, nblks_e, 2))
  allocate(p_prog%w    (nproma, nlev, nblks_c))
  allocate(p_diag%vn_ie(nproma, nlev, nblks_e))
  p_patch%edges%inv_dual_edge_length   = inv_dual
  p_patch%edges%inv_primal_edge_length = inv_primal
  p_patch%edges%tangent_orientation    = tangent
  p_patch%edges%cell_idx   = icidx
  p_patch%edges%cell_blk   = icblk
  p_patch%edges%vertex_idx = ividx
  p_patch%edges%vertex_blk = ivblk
  p_prog%w      = w
  p_diag%vn_ie  = vn_ie

  z_v_grad_w_struct = 0.0d0
  z_v_grad_w_flat   = 0.0d0

  call kernel_struct(p_patch, p_prog, p_diag, z_vt_ie, z_w_v, z_v_grad_w_struct, &
                     nlev, i_startblk, i_endblk, i_startidx, i_endidx)
  call kernel_flat(vn_ie, inv_dual, inv_primal, tangent, w, z_vt_ie, z_w_v, &
                   icidx, icblk, ividx, ivblk, z_v_grad_w_flat, &
                   nproma, nlev, nblks_e, nblks_c, nblks_v, &
                   i_startblk, i_endblk, i_startidx, i_endidx)

  max_err = maxval(abs(z_v_grad_w_struct - z_v_grad_w_flat))
  if (max_err > TOL) then
    print *, "FAIL max_err=", max_err
    stop 1
  end if
  print *, "OK max_err=", max_err
end program
