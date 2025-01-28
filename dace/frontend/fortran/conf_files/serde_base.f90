module serde
  implicit none

  interface serialize
    module procedure int_2s, real_2s, double_precision_2s, logical_2s
  end interface serialize
contains
  subroutine write_to(path, s)
    character(len=*), intent(in) :: path
    character(len=*), intent(in) ::  s
    integer :: io
    open (NEWUNIT=io, FILE=path, STATUS="replace", ACTION="write")
    write (io, *) s
    close (UNIT=io)
  end subroutine write_to

  function int_2s(i) result(s)
    integer, intent(in) :: i
    character(len=:), allocatable :: s
    allocate (character(len=50) :: s)
    write (s, *) i
    s = trim(s)
  end function int_2s

  function real_2s(r) result(s)
    real, intent(in) :: r
    character(len=:), allocatable :: s
    allocate (character(len=50) :: s)
    write (s, *) r
    s = trim(s)
  end function real_2s

  function double_precision_2s(r) result(s)
    double precision, intent(in) :: r
    character(len=:), allocatable :: s
    allocate (character(len=50) :: s)
    write (s, *) r
    s = trim(s)
  end function double_precision_2s

  function logical_2s(l) result(s)
    logical, intent(in) :: l
    character(len=:), allocatable :: s
    allocate (character(len=50) :: s)
    write (s, *) l
    s = trim(s)
  end function logical_2s
end module serde
