! ICON velocity_advection loopnest 3 -- DERIVED-TYPE variant.
!
! Same math as ``loopnest_3.f90`` but inputs come through the ICON-style
! ``p_metrics`` / ``p_patch%edges`` derived-type chains.  Phase 1 of
! derived-type support flattens single-level structs whose members are
! flat scalars or fixed-shape arrays of scalars; for this kernel the
! struct shapes are:
!   p_metrics%deepatmo_gradh_ifc   shape (NLEV)
!   p_metrics%deepatmo_invr_ifc    shape (NLEV)
!   p_patch_edges%fn_e              shape (NPROMA, NBLKS)
!   p_patch_edges%ft_e              shape (NPROMA, NBLKS)
! The kernel's other 3-D inputs (``vn_ie``, ``z_vt_ie``, ``z_v_grad_w``)
! stay as plain arrays because in the original ICON source they live
! on a different struct chain (``p_diag``); separating per-struct
! lowers cleanly to per-field flat arrays in the SDFG.
MODULE icon_types_loopnest_3
  IMPLICIT NONE
  INTEGER, PARAMETER :: NPROMA = 32
  INTEGER, PARAMETER :: NLEV = 32
  INTEGER, PARAMETER :: NBLKS = 5
  TYPE :: t_metrics
    REAL(KIND=8) :: deepatmo_gradh_ifc(NLEV)
    REAL(KIND=8) :: deepatmo_invr_ifc(NLEV)
  END TYPE t_metrics
  TYPE :: t_patch_edges
    REAL(KIND=8) :: ft_e(NPROMA, NBLKS)
    REAL(KIND=8) :: fn_e(NPROMA, NBLKS)
  END TYPE t_patch_edges
END MODULE icon_types_loopnest_3

SUBROUTINE icon_loopnest_3_struct(jb, i_startidx, i_endidx, &
                                  vn_ie, z_vt_ie, p_metrics, p_patch_edges, z_v_grad_w)
  USE icon_types_loopnest_3
  IMPLICIT NONE
  INTEGER(KIND=4), VALUE :: jb, i_startidx, i_endidx
  REAL(KIND=8),         INTENT(IN)    :: vn_ie(NPROMA, NLEV, NBLKS)
  REAL(KIND=8),         INTENT(IN)    :: z_vt_ie(NPROMA, NLEV, NBLKS)
  TYPE(t_metrics),      INTENT(IN)    :: p_metrics
  TYPE(t_patch_edges),  INTENT(IN)    :: p_patch_edges
  REAL(KIND=8),         INTENT(INOUT) :: z_v_grad_w(NPROMA, NLEV, NBLKS)
  INTEGER(KIND=4) :: jk, je
  DO jk = 1, NLEV
    DO je = i_startidx, i_endidx
      z_v_grad_w(je, jk, jb) = z_v_grad_w(je, jk, jb) * p_metrics%deepatmo_gradh_ifc(jk) &
        + vn_ie(je, jk, jb) * (vn_ie(je, jk, jb) * p_metrics%deepatmo_invr_ifc(jk) - p_patch_edges%ft_e(je, jb)) &
        + z_vt_ie(je, jk, jb) * (z_vt_ie(je, jk, jb) * p_metrics%deepatmo_invr_ifc(jk) + p_patch_edges%fn_e(je, jb))
    END DO
  END DO
END SUBROUTINE icon_loopnest_3_struct
