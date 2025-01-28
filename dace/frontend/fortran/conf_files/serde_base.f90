module serde
  implicit none

  interface serialize
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
end module serde
