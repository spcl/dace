! Reduced form of the ICON e_bln_c_s interpolation that maps edge-based
! kinetic energy onto cell centres (see mo_velocity_advection.f90's z_ekinh
! update).  Each cell samples three edges via the connectivity table
! edge_idx(jc, 1:3); this is the canonical "indirect access" shape.
!
!     z_ekinh(jc, jk) = sum_{k=1..3} e_bln(jc, k) * z_kin(edge_idx(jc, k), jk)
!
! After the HLFIR frontend runs, each indirect load of edge_idx must mint a
! fresh SDFG symbol and emit an interstate-edge assignment, producing a new
! state before the tasklet that consumes z_kin at that symbol.
subroutine kin_to_cell(z_kin, e_bln, edge_idx, z_ekinh, nc, ne, nk)
  implicit none
  integer, intent(in)  :: nc, ne, nk
  integer, intent(in)  :: edge_idx(nc, 3)
  real(8), intent(in)  :: e_bln(nc, 3)
  real(8), intent(in)  :: z_kin(ne, nk)
  real(8), intent(out) :: z_ekinh(nc, nk)
  integer :: jc, jk
  do jk = 1, nk
    do jc = 1, nc
      z_ekinh(jc, jk) = e_bln(jc, 1) * z_kin(edge_idx(jc, 1), jk) &
                      + e_bln(jc, 2) * z_kin(edge_idx(jc, 2), jk) &
                      + e_bln(jc, 3) * z_kin(edge_idx(jc, 3), jk)
    end do
  end do
end subroutine kin_to_cell
