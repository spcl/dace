module serde
  implicit none

  interface serialize
    module procedure int_2s, int_a2s, int_aa2s, int_aaa2s
    module procedure real_2s, real_a2s, real_aa2s, real_aaa2s
    module procedure double_2s, double_a2s, double_aa2s, double_aaa2s
    module procedure logical_2s, logical_a2s, logical_aa2s, logical_aaa2s
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

  function rank_2s_(a) result(s)
    class(*), intent(in) :: a(*)
    character(len=:), allocatable :: s
    integer :: k
    s = "# rank"//new_line('A')//serialize(rank(a))//new_line('A')
    s = s//"# size"//new_line('A')
    do k = 1, rank(a)
      s = s//serialize(size(a, k))//new_line('A')
    end do
    s = s//"# lbound"//new_line('A')
    do k = 1, rank(a)
      s = s//serialize(lbound(a, k))//new_line('A')
    end do
  end function rank_2s_

  function int_2s(i) result(s)
    integer, intent(in) :: i
    character(len=:), allocatable :: s
    allocate (character(len=50) :: s)
    write (s, *) i
    s = trim(s)
  end function int_2s

  function int_a2s(i) result(s)
    integer, intent(in) :: i(:)
    character(len=:), allocatable :: s
    integer :: k1
    s = rank_2s_(i)//"# entries"//new_line('A')
    do k1 = 1, size(i, 1)
      s = s//serialize(i(k1))//new_line('A')
    end do
  end function int_a2s
  function int_aa2s(i) result(s)
    integer, intent(in) :: i(:, :)
    character(len=:), allocatable :: s
    integer :: k1, k2
    s = rank_2s_(i)//"# entries"//new_line('A')
    do k1 = 1, size(i, 1)
      do k2 = 1, size(i, 2)
        s = s//serialize(i(k1, k2))//new_line('A')
      end do
    end do
  end function int_aa2s
  function int_aaa2s(i) result(s)
    integer, intent(in) :: i(:, :, :)
    character(len=:), allocatable :: s
    integer :: k1, k2, k3
    s = rank_2s_(i)//"# entries"//new_line('A')
    do k1 = 1, size(i, 1)
      do k2 = 1, size(i, 2)
        do k3 = 1, size(i, 3)
          s = s//serialize(i(k1, k2, k3))//new_line('A')
        end do
      end do
    end do
  end function int_aaa2s

  function int_za2s(i) result(s)
    integer, allocatable, intent(in) :: i(:)
    character(len=:), allocatable :: s
    integer :: k1
    s = "# allocated"//new_line('A')//serialize(allocated(i))//new_line('A')
    if (allocated(i)) then
      s = rank_2s_(i)//"# entries"//new_line('A')
      do k1 = 1, size(i, 1)
        s = s//serialize(i(k1))//new_line('A')
      end do
    end if
  end function int_za2s
  function int_zaa2s(i) result(s)
    integer, allocatable, intent(in) :: i(:, :)
    character(len=:), allocatable :: s
    integer :: k1, k2
    s = "# allocated"//new_line('A')//serialize(allocated(i))//new_line('A')
    if (allocated(i)) then
      s = rank_2s_(i)//"# entries"//new_line('A')
      do k1 = 1, size(i, 1)
        do k2 = 1, size(i, 2)
          s = s//serialize(i(k1, k2))//new_line('A')
        end do
      end do
    end if
  end function int_zaa2s
  function int_zaaa2s(i) result(s)
    integer, allocatable, intent(in) :: i(:, :, :)
    character(len=:), allocatable :: s
    integer :: k1, k2, k3
    s = "# allocated"//new_line('A')//serialize(allocated(i))//new_line('A')
    if (allocated(i)) then
      s = rank_2s_(i)//"# entries"//new_line('A')
      do k1 = 1, size(i, 1)
        do k2 = 1, size(i, 2)
          do k3 = 1, size(i, 3)
            s = s//serialize(i(k1, k2, k3))//new_line('A')
          end do
        end do
      end do
    end if
  end function int_zaaa2s

  function real_2s(r) result(s)
    real, intent(in) :: r
    character(len=:), allocatable :: s
    allocate (character(len=50) :: s)
    write (s, *) r
    s = trim(s)
  end function real_2s

  function real_a2s(i) result(s)
    real, intent(in) :: i(:)
    character(len=:), allocatable :: s
    integer :: k1
    s = rank_2s_(i)//"# entries"//new_line('A')
    do k1 = 1, size(i, 1)
      s = s//serialize(i(k1))//new_line('A')
    end do
  end function real_a2s
  function real_aa2s(i) result(s)
    real, intent(in) :: i(:, :)
    character(len=:), allocatable :: s
    integer :: k1, k2
    s = rank_2s_(i)//"# entries"//new_line('A')
    do k1 = 1, size(i, 1)
      do k2 = 1, size(i, 2)
        s = s//serialize(i(k1, k2))//new_line('A')
      end do
    end do
  end function real_aa2s
  function real_aaa2s(i) result(s)
    real, intent(in) :: i(:, :, :)
    character(len=:), allocatable :: s
    integer :: k1, k2, k3
    s = rank_2s_(i)//"# entries"//new_line('A')
    do k1 = 1, size(i, 1)
      do k2 = 1, size(i, 2)
        do k3 = 1, size(i, 3)
          s = s//serialize(i(k1, k2, k3))//new_line('A')
        end do
      end do
    end do
  end function real_aaa2s

  function real_za2s(i) result(s)
    real, allocatable, intent(in) :: i(:)
    character(len=:), allocatable :: s
    integer :: k1
    s = "# allocated"//new_line('A')//serialize(allocated(i))//new_line('A')
    if (allocated(i)) then
      s = rank_2s_(i)//"# entries"//new_line('A')
      do k1 = 1, size(i, 1)
        s = s//serialize(i(k1))//new_line('A')
      end do
    end if
  end function real_za2s
  function real_zaa2s(i) result(s)
    real, allocatable, intent(in) :: i(:, :)
    character(len=:), allocatable :: s
    integer :: k1, k2
    s = "# allocated"//new_line('A')//serialize(allocated(i))//new_line('A')
    if (allocated(i)) then
      s = rank_2s_(i)//"# entries"//new_line('A')
      do k1 = 1, size(i, 1)
        do k2 = 1, size(i, 2)
          s = s//serialize(i(k1, k2))//new_line('A')
        end do
      end do
    end if
  end function real_zaa2s
  function real_zaaa2s(i) result(s)
    real, allocatable, intent(in) :: i(:, :, :)
    character(len=:), allocatable :: s
    integer :: k1, k2, k3
    s = "# allocated"//new_line('A')//serialize(allocated(i))//new_line('A')
    if (allocated(i)) then
      s = rank_2s_(i)//"# entries"//new_line('A')
      do k1 = 1, size(i, 1)
        do k2 = 1, size(i, 2)
          do k3 = 1, size(i, 3)
            s = s//serialize(i(k1, k2, k3))//new_line('A')
          end do
        end do
      end do
    end if
  end function real_zaaa2s

  function double_2s(r) result(s)
    double precision, intent(in) :: r
    character(len=:), allocatable :: s
    allocate (character(len=50) :: s)
    write (s, *) r
    s = trim(s)
  end function double_2s

  function double_a2s(i) result(s)
    double precision, intent(in) :: i(:)
    character(len=:), allocatable :: s
    integer :: k1
    s = rank_2s_(i)//"# entries"//new_line('A')
    do k1 = 1, size(i, 1)
      s = s//serialize(i(k1))//new_line('A')
    end do
  end function double_a2s
  function double_aa2s(i) result(s)
    double precision, intent(in) :: i(:, :)
    character(len=:), allocatable :: s
    integer :: k1, k2
    s = rank_2s_(i)//"# entries"//new_line('A')
    do k1 = 1, size(i, 1)
      do k2 = 1, size(i, 2)
        s = s//serialize(i(k1, k2))//new_line('A')
      end do
    end do
  end function double_aa2s
  function double_aaa2s(i) result(s)
    double precision, intent(in) :: i(:, :, :)
    character(len=:), allocatable :: s
    integer :: k1, k2, k3
    s = rank_2s_(i)//"# entries"//new_line('A')
    do k1 = 1, size(i, 1)
      do k2 = 1, size(i, 2)
        do k3 = 1, size(i, 3)
          s = s//serialize(i(k1, k2, k3))//new_line('A')
        end do
      end do
    end do
  end function double_aaa2s

  function double_za2s(i) result(s)
    double precision, allocatable, intent(in) :: i(:)
    character(len=:), allocatable :: s
    integer :: k1
    s = "# allocated"//new_line('A')//serialize(allocated(i))//new_line('A')
    if (allocated(i)) then
      s = rank_2s_(i)//"# entries"//new_line('A')
      do k1 = 1, size(i, 1)
        s = s//serialize(i(k1))//new_line('A')
      end do
    end if
  end function double_za2s
  function double_zaa2s(i) result(s)
    double precision, allocatable, intent(in) :: i(:, :)
    character(len=:), allocatable :: s
    integer :: k1, k2
    s = "# allocated"//new_line('A')//serialize(allocated(i))//new_line('A')
    if (allocated(i)) then
      s = rank_2s_(i)//"# entries"//new_line('A')
      do k1 = 1, size(i, 1)
        do k2 = 1, size(i, 2)
          s = s//serialize(i(k1, k2))//new_line('A')
        end do
      end do
    end if
  end function double_zaa2s
  function double_zaaa2s(i) result(s)
    double precision, allocatable, intent(in) :: i(:, :, :)
    character(len=:), allocatable :: s
    integer :: k1, k2, k3
    s = "# allocated"//new_line('A')//serialize(allocated(i))//new_line('A')
    if (allocated(i)) then
      s = rank_2s_(i)//"# entries"//new_line('A')
      do k1 = 1, size(i, 1)
        do k2 = 1, size(i, 2)
          do k3 = 1, size(i, 3)
            s = s//serialize(i(k1, k2, k3))//new_line('A')
          end do
        end do
      end do
    end if
  end function double_zaaa2s

  function logical_2s(l) result(s)
    logical, intent(in) :: l
    character(len=:), allocatable :: s
    allocate (character(len=50) :: s)
    write (s, *) l
    s = trim(s)
  end function logical_2s

  function logical_a2s(i) result(s)
    logical, intent(in) :: i(:)
    character(len=:), allocatable :: s
    integer :: k1
    s = rank_2s_(i)//"# entries"//new_line('A')
    do k1 = 1, size(i, 1)
      s = s//serialize(i(k1))//new_line('A')
    end do
  end function logical_a2s
  function logical_aa2s(i) result(s)
    logical, intent(in) :: i(:, :)
    character(len=:), allocatable :: s
    integer :: k1, k2
    s = rank_2s_(i)//"# entries"//new_line('A')
    do k1 = 1, size(i, 1)
      do k2 = 1, size(i, 2)
        s = s//serialize(i(k1, k2))//new_line('A')
      end do
    end do
  end function logical_aa2s
  function logical_aaa2s(i) result(s)
    logical, intent(in) :: i(:, :, :)
    character(len=:), allocatable :: s
    integer :: k1, k2, k3
    s = rank_2s_(i)//"# entries"//new_line('A')
    do k1 = 1, size(i, 1)
      do k2 = 1, size(i, 2)
        do k3 = 1, size(i, 3)
          s = s//serialize(i(k1, k2, k3))//new_line('A')
        end do
      end do
    end do
  end function logical_aaa2s

  function logical_za2s(i) result(s)
    logical, allocatable, intent(in) :: i(:)
    character(len=:), allocatable :: s
    integer :: k1
    s = "# allocated"//new_line('A')//serialize(allocated(i))//new_line('A')
    if (allocated(i)) then
      s = rank_2s_(i)//"# entries"//new_line('A')
      do k1 = 1, size(i, 1)
        s = s//serialize(i(k1))//new_line('A')
      end do
    end if
  end function logical_za2s
  function logical_zaa2s(i) result(s)
    logical, allocatable, intent(in) :: i(:, :)
    character(len=:), allocatable :: s
    integer :: k1, k2
    s = "# allocated"//new_line('A')//serialize(allocated(i))//new_line('A')
    if (allocated(i)) then
      s = rank_2s_(i)//"# entries"//new_line('A')
      do k1 = 1, size(i, 1)
        do k2 = 1, size(i, 2)
          s = s//serialize(i(k1, k2))//new_line('A')
        end do
      end do
    end if
  end function logical_zaa2s
  function logical_zaaa2s(i) result(s)
    logical, allocatable, intent(in) :: i(:, :, :)
    character(len=:), allocatable :: s
    integer :: k1, k2, k3
    s = "# allocated"//new_line('A')//serialize(allocated(i))//new_line('A')
    if (allocated(i)) then
      s = rank_2s_(i)//"# entries"//new_line('A')
      do k1 = 1, size(i, 1)
        do k2 = 1, size(i, 2)
          do k3 = 1, size(i, 3)
            s = s//serialize(i(k1, k2, k3))//new_line('A')
          end do
        end do
      end do
    end if
  end function logical_zaaa2s
end module serde
