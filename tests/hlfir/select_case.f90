! Fortran SELECT CASE - all four case-label shapes.  We expect the bridge
! to lower the fir.select_case terminator into a chain of nested
! kind="conditional" AST nodes, one per distinct branch, combining
! value-list cases with OR.
subroutine sel_all(x, out)
  implicit none
  integer, intent(in)    :: x
  integer, intent(inout) :: out
  select case (x)
  case (1)
    out = 100
  case (2, 3, 5)
    out = 200
  case (10:20)
    out = 300
  case (100:)
    out = 400
  case (:-1)
    out = 500
  case default
    out = 0
  end select
end subroutine sel_all
