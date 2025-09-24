# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
from dace.frontend.fortran.ast_desugaring import optimizations, types
from tests.fortran.desugaring.common import parse_and_improve
from tests.fortran.fortran_test_helper import SourceCodeBuilder


def test_constant_resolving_expressions():
    sources, main = SourceCodeBuilder().add_file("""
subroutine main
  implicit none
  integer, parameter :: k = 8
  integer :: a = -1, b = -1
  real, parameter :: pk = 4.1_k
  real(kind=selected_real_kind(5, 5)) :: p = 1.0_k

  if (k < 2) then
    a = k
    p = k*pk
  else if (k < 5) then
    b = k
    p = p + k*pk
  else
    a = k
    b = k
    p = a*p + k*pk
  end if
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = optimizations.const_eval_nodes(ast)

    got = ast.tofortran()
    want = """
SUBROUTINE main
  IMPLICIT NONE
  INTEGER, PARAMETER :: k = 8
  INTEGER :: a = - 1, b = - 1
  REAL, PARAMETER :: pk = 4.1D0
  REAL(KIND = 4) :: p = 1.0D0
  IF (.FALSE.) THEN
    a = 8
    p = 32.79999923706055D0
  ELSE IF (.FALSE.) THEN
    b = 8
    p = p + 32.79999923706055D0
  ELSE
    a = 8
    b = 8
    p = a * p + 32.79999923706055D0
  END IF
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_constant_expression_replacement():
    sources, main = SourceCodeBuilder().add_file("""
module main
  implicit none
  private
  real, parameter :: &
    three = 3.0
contains
  subroutine foo
    implicit none
    real :: res1, res2, res3, unk
    real, parameter :: &
        x = -(three + 4.0), &
        y = -4.0, &
        z = 3.0
    res1 = unk ** x
    res2 = unk ** y
    res3 = unk ** z
  end subroutine foo
end module main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = optimizations.const_eval_nodes(ast)

    got = ast.tofortran()
    want = """
MODULE main
  IMPLICIT NONE
  PRIVATE
  REAL, PARAMETER :: three = 3.0
  CONTAINS
  SUBROUTINE foo
    IMPLICIT NONE
    REAL :: res1, res2, res3, unk
    REAL, PARAMETER :: x = - (7.0), y = - 4.0, z = 3.0
    res1 = unk ** (- 7.0)
    res2 = unk ** (- 4.0)
    res3 = unk ** 3.0
  END SUBROUTINE foo
END MODULE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_constant_resolving_non_expressions():
    sources, main = SourceCodeBuilder().add_file("""
subroutine main
  implicit none
  integer, parameter :: k = 8
  integer :: i
  real :: a = 1
  do i = 2, k
    a = a + i * k
  end do
  a = fun(k)
  call not_fun(k, a)
  contains
  real function fun(x)
    integer, intent(in) :: x
    fun = x * k
  end function fun
  subroutine not_fun(x, y)
    integer, intent(in) :: x
    real, intent(out) :: y
    y = x * k
  end subroutine not_fun
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = optimizations.const_eval_nodes(ast)

    got = ast.tofortran()
    want = """
SUBROUTINE main
  IMPLICIT NONE
  INTEGER, PARAMETER :: k = 8
  INTEGER :: i
  REAL :: a = 1
  DO i = 2, 8
    a = a + i * 8
  END DO
  a = fun(8)
  CALL not_fun(8, a)
  CONTAINS
  REAL FUNCTION fun(x)
    INTEGER, INTENT(IN) :: x
    fun = x * 8
  END FUNCTION fun
  SUBROUTINE not_fun(x, y)
    INTEGER, INTENT(IN) :: x
    REAL, INTENT(OUT) :: y
    y = x * 8
  END SUBROUTINE not_fun
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_config_injection_type():
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
  type config
    integer :: a = 8
    real :: b = 2.0
    logical :: c = .false.
  end type config
  type big_config
    type(config) :: big
  end type big_config
  type(config) :: globalo
contains
  subroutine fun(this)
    implicit none
    type(config), intent(inout) :: this
    this%b = 5.1
  end subroutine fun
end module lib

subroutine main(cfg)
  use lib
  implicit none
  type(big_config), intent(in) :: cfg
  real :: a = 1
  a = cfg%big%b + a * globalo%a
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = optimizations.inject_const_evals(ast, [
        types.ConstTypeInjection(None, ('lib', 'config'), ('a', ), '42'),
        types.ConstTypeInjection(None, ('lib', 'config'), ('b', ), '10000.0')
    ])

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  TYPE :: config
    INTEGER :: a = 8
    REAL :: b = 2.0
    LOGICAL :: c = .FALSE.
  END TYPE config
  TYPE :: big_config
    TYPE(config) :: big
  END TYPE big_config
  TYPE(config) :: globalo
  CONTAINS
  SUBROUTINE fun(this)
    IMPLICIT NONE
    TYPE(config), INTENT(INOUT) :: this
    this % b = 5.1
  END SUBROUTINE fun
END MODULE lib
SUBROUTINE main(cfg)
  USE lib
  IMPLICIT NONE
  TYPE(big_config), INTENT(IN) :: cfg
  REAL :: a = 1
  a = 10000.0 + a * 42
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_config_injection_instance():
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
  type config
    integer :: a = 8
    real :: b = 2.0
    logical :: c = .false.
  end type config
  type big_config
    type(config) :: big
  end type big_config
  type(config) :: globalo
contains
  subroutine fun(this)
    implicit none
    type(config), intent(inout) :: this
    this%b = 5.1
  end subroutine fun
end module lib

subroutine main(cfg)
  use lib
  implicit none
  type(big_config), intent(in) :: cfg
  real :: a = 1
  a = cfg%big%b + a * globalo%a
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = optimizations.inject_const_evals(ast, [
        types.ConstInstanceInjection(None, ('lib', 'globalo'), ('a', ), '42'),
        types.ConstInstanceInjection(None, ('main', 'cfg'), ('big', 'b'), '10000.0')
    ])

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  TYPE :: config
    INTEGER :: a = 8
    REAL :: b = 2.0
    LOGICAL :: c = .FALSE.
  END TYPE config
  TYPE :: big_config
    TYPE(config) :: big
  END TYPE big_config
  TYPE(config) :: globalo
  CONTAINS
  SUBROUTINE fun(this)
    IMPLICIT NONE
    TYPE(config), INTENT(INOUT) :: this
    this % b = 5.1
  END SUBROUTINE fun
END MODULE lib
SUBROUTINE main(cfg)
  USE lib
  IMPLICIT NONE
  TYPE(big_config), INTENT(IN) :: cfg
  REAL :: a = 1
  a = 10000.0 + a * 42
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_config_injection_array():
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
  type config
    integer, allocatable :: a(:, :)
  end type config
contains
  real function fun(this)
    implicit none
    type(config), intent(inout) :: this
    if (allocated(this%a)) then  ! This will be replaced even though it is an out (i.e., beware of invalid injections).
      fun = 5.1
    else
      fun = -1
    endif
  end function fun
end module lib

subroutine main(cfg)
  use lib
  implicit none
  type(config), intent(in) :: cfg
  real :: a = 1
  if (allocated(cfg%a)) a = 7.2
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = optimizations.inject_const_evals(ast, [
        types.ConstTypeInjection(None, ('lib', 'config'), ('a_a', ), 'true'),
    ])

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  TYPE :: config
    INTEGER, ALLOCATABLE :: a(:, :)
  END TYPE config
  CONTAINS
  REAL FUNCTION fun(this)
    IMPLICIT NONE
    TYPE(config), INTENT(INOUT) :: this
    IF (.TRUE.) THEN
      fun = 5.1
    ELSE
      fun = - 1
    END IF
  END FUNCTION fun
END MODULE lib
SUBROUTINE main(cfg)
  USE lib
  IMPLICIT NONE
  TYPE(config), INTENT(IN) :: cfg
  REAL :: a = 1
  IF (.TRUE.) a = 7.2
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_config_injection_allocatable_fixing():
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
  type config
    integer, allocatable :: a(:, :)
  end type config
end module lib

subroutine main(cfg, b, c, d)
  use lib
  implicit none
  type(config), intent(in) :: cfg
  real, allocatable, intent(inout) :: b(:)
  real, allocatable, intent(inout) :: c(:, :)
  real, allocatable, intent(inout) :: d(:)
  real :: a = 1
  if (allocated(cfg%a)) a = 7.2
  if (allocated(b)) b = 7.2
  if (allocated(c)) c = 7.2
  if (allocated(d)) d = 7.2
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = optimizations.inject_const_evals(ast, [
        types.ConstTypeInjection(None, ('lib', 'config'), ('a_a', ), 'true'),
        types.ConstTypeInjection(None, ('lib', 'config'), ('__f2dace_SA_a_d_0_s', ), '3'),
        types.ConstTypeInjection(None, ('lib', 'config'), ('__f2dace_SOA_a_d_0_s', ), '1'),
        types.ConstTypeInjection(None, ('lib', 'config'), ('__f2dace_SA_a_d_1_s', ), '3'),
        types.ConstTypeInjection(None, ('lib', 'config'), ('__f2dace_SOA_a_d_1_s', ), '2'),
        types.ConstInstanceInjection(None, ('main', 'b_a'), tuple(), 'true'),
        types.ConstInstanceInjection(None, ('main', '__f2dace_SA_b_d_0_s'), tuple(), '4'),
        types.ConstInstanceInjection(None, ('main', '__f2dace_SOA_b_d_0_s'), tuple(), '1'),
        types.ConstInstanceInjection(None, ('main', 'c_a'), tuple(), 'true'),
        types.ConstInstanceInjection(None, ('main', '__f2dace_SA_c_d_0_s'), tuple(), '4'),
        types.ConstInstanceInjection(None, ('main', '__f2dace_SOA_c_d_0_s'), tuple(), '1'),
        types.ConstInstanceInjection(None, ('main', 'd_a'), tuple(), 'false'),
        types.ConstInstanceInjection(None, ('main', '__f2dace_SA_d_d_0_s'), tuple(), '4'),
        types.ConstInstanceInjection(None, ('main', '__f2dace_SOA_d_d_0_s'), tuple(), '1'),
    ])

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  TYPE :: config
    INTEGER(KIND = 4) :: a(1 : 3, 2 : 4)
  END TYPE config
END MODULE lib
SUBROUTINE main(cfg, b, c, d)
  USE lib
  IMPLICIT NONE
  TYPE(config), INTENT(IN) :: cfg
  REAL(KIND = 4), INTENT(INOUT) :: b(1 : 4)
  REAL, ALLOCATABLE, INTENT(INOUT) :: c(:, :)
  REAL(KIND = 4), INTENT(INOUT) :: d(1 : 4)
  REAL :: a = 1
  IF (.TRUE.) a = 7.2
  IF (.TRUE.) b = 7.2
  IF (.TRUE.) c = 7.2
  IF (.FALSE.) d = 7.2
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_practically_constant_arguments():
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
contains
  real function fun(cond, kwcond, opt)
    implicit none
    logical, intent(in) :: cond, kwcond
    logical, optional, intent(in) :: opt
    logical :: real_opt = .false.
    if (present(opt)) then
      real_opt = opt
    end if
    if (cond .and. kwcond .and. real_opt) then
      fun = -2.7
    else
      fun = 4.2
    end if
  end function fun

  real function not_fun(cond, kwcond, opt)
    implicit none
    logical, intent(in) :: cond, kwcond
    logical, optional, intent(in) :: opt
    logical :: real_opt = .false.
    if (present(opt)) then
      real_opt = opt
    end if
    if (cond .and. kwcond .and. real_opt) then
      not_fun = -500.1
    else
      not_fun = 9600.8
    end if
  end function not_fun

  subroutine user_1()
    implicit none
    real :: c
    c = fun(.false., kwcond=.false., opt=.true.)*not_fun(.false., kwcond=.false., opt=.false.)
  end subroutine user_1

  subroutine user_2()
    implicit none
    real :: c
    c = 3*fun(.false., kwcond=.false., opt=.true.)*not_fun(.true., kwcond=.true., opt=.true.)
  end subroutine user_2
end module lib

subroutine main()
  use lib
  implicit none
  call user_1()
  call user_2()
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = optimizations.make_practically_constant_arguments_constants(ast, [('main', )])

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  CONTAINS
  REAL FUNCTION fun(cond, kwcond, opt)
    IMPLICIT NONE
    LOGICAL, INTENT(IN) :: cond, kwcond
    LOGICAL, OPTIONAL, INTENT(IN) :: opt
    LOGICAL :: real_opt = .FALSE.
    IF (.TRUE.) THEN
      real_opt = .TRUE.
    END IF
    IF (.FALSE. .AND. .FALSE. .AND. real_opt) THEN
      fun = - 2.7
    ELSE
      fun = 4.2
    END IF
  END FUNCTION fun
  REAL FUNCTION not_fun(cond, kwcond, opt)
    IMPLICIT NONE
    LOGICAL, INTENT(IN) :: cond, kwcond
    LOGICAL, OPTIONAL, INTENT(IN) :: opt
    LOGICAL :: real_opt = .FALSE.
    IF (.TRUE.) THEN
      real_opt = opt
    END IF
    IF (cond .AND. kwcond .AND. real_opt) THEN
      not_fun = - 500.1
    ELSE
      not_fun = 9600.8
    END IF
  END FUNCTION not_fun
  SUBROUTINE user_1
    IMPLICIT NONE
    REAL :: c
    c = fun(.FALSE., kwcond = .FALSE., opt = .TRUE.) * not_fun(.FALSE., kwcond = .FALSE., opt = .FALSE.)
  END SUBROUTINE user_1
  SUBROUTINE user_2
    IMPLICIT NONE
    REAL :: c
    c = 3 * fun(.FALSE., kwcond = .FALSE., opt = .TRUE.) * not_fun(.TRUE., kwcond = .TRUE., opt = .TRUE.)
  END SUBROUTINE user_2
END MODULE lib
SUBROUTINE main
  USE lib
  IMPLICIT NONE
  CALL user_1
  CALL user_2
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_practically_constant_global_vars_constants():
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
  logical :: fixed_cond = .false.
  logical :: movable_cond = .false.
contains
  subroutine update(what)
    implicit none
    logical, intent(out) :: what
    what = .true.
  end subroutine update
end module lib

subroutine main
  use lib
  implicit none
  real :: a = 1.0
  call update(movable_cond)
  movable_cond = .not. movable_cond
  if (fixed_cond .and. movable_cond) a = 7.1
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = optimizations.make_practically_constant_global_vars_constants(ast)

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  LOGICAL, PARAMETER :: fixed_cond = .FALSE.
  LOGICAL :: movable_cond = .FALSE.
  CONTAINS
  SUBROUTINE update(what)
    IMPLICIT NONE
    LOGICAL, INTENT(OUT) :: what
    what = .TRUE.
  END SUBROUTINE update
END MODULE lib
SUBROUTINE main
  USE lib
  IMPLICIT NONE
  REAL :: a = 1.0
  CALL update(movable_cond)
  movable_cond = .NOT. movable_cond
  IF (fixed_cond .AND. movable_cond) a = 7.1
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_exploit_locally_constant_variables():
    sources, main = SourceCodeBuilder().add_file("""
subroutine main()
  implicit none
  logical :: cond = .true.
  real :: out = 0.
  integer :: i
  logical :: arr(5) = .true.

  ! `cond` is known in this block and doesn't change. `out` is unknown`, since it changes conditionally.
  if (cond) out = out + 1.
  out = out*2
  if (cond) then
    out = out + 1.
  else
    out = out - 1.
  end if

  ! `cond` is unknown after this, since it changes conditionally.
  if (out .gt. 20) cond = .false.
  if (cond) out = out + 100.

  ! `cond` is known again, and even `out` this time.
  cond = .true.
  out = 7.2
  out = out*2.0
  out = fun(.not. cond, out)

  ! A simple do loop with `i` as loop variable.
  do i=1, 20
    out = out + 1.
  end do
  ! TODO: `i` should be known at this point, since do loop is deterministic.
  i = i + 1

  ! A simple do-while loop with `i` as loop variable, `i` becomes unknown.
  i = 0
  do while (i < 10)
    out = out + 1
    i = i + 1
  end do

  ! Just making sure that `cond` is still known after all the loops.
  if (cond) out = out + 1.

  ! `cond` evaluation inside a branch should also happen.
  if (cond) then
    cond = .true.
    if (cond) then
      out = out + 1.
    else
      out = out + 7.
    end if
  end if

  ! The content of an array we don't track.
  arr = .false.
  do i=1, 5
    if (arrfun(arr) .or. arr(2)) then
      out = out + 3.14
    end if
  end do

contains
  real function fun(cond, out)
    implicit none
    logical, intent(in) :: cond
    real, intent(inout) :: out
    if (cond) out = out + 42
    fun = out + 1.0
  end function fun
  logical function arrfun(arr)
    implicit none
    logical, intent(in) :: arr(:)
    arrfun = arr(1)
  end function arrfun
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = optimizations.exploit_locally_constant_variables(ast)

    got = ast.tofortran()
    want = """
SUBROUTINE main
  IMPLICIT NONE
  LOGICAL :: cond = .TRUE.
  REAL :: out = 0.
  INTEGER :: i
  LOGICAL :: arr(5) = .TRUE.
  IF (.TRUE.) out = 0. + 1.
  out = out * 2
  IF (.TRUE.) THEN
    out = out + 1.
  ELSE
    out = out - 1.
  END IF
  IF (out .GT. 20) cond = .FALSE.
  IF (cond) out = out + 100.
  cond = .TRUE.
  out = 7.2
  out = 7.2 * 2.0
  out = fun(.NOT. .TRUE., out)
  DO i = 1, 20
    out = out + 1.
  END DO
  i = i + 1
  i = 0
  DO WHILE (i < 10)
    out = out + 1
    i = i + 1
  END DO
  IF (.TRUE.) out = out + 1.
  IF (.TRUE.) THEN
    cond = .TRUE.
    IF (.TRUE.) THEN
      out = out + 1.
    ELSE
      out = out + 7.
    END IF
  END IF
  arr = .FALSE.
  DO i = 1, 5
    IF (arrfun(arr) .OR. arr(2)) THEN
      out = out + 3.14
    END IF
  END DO
  CONTAINS
  REAL FUNCTION fun(cond, out)
    IMPLICIT NONE
    LOGICAL, INTENT(IN) :: cond
    REAL, INTENT(INOUT) :: out
    IF (cond) out = out + 42
    fun = out + 1.0
  END FUNCTION fun
  LOGICAL FUNCTION arrfun(arr)
    IMPLICIT NONE
    LOGICAL, INTENT(IN) :: arr(:)
    arrfun = arr(1)
  END FUNCTION arrfun
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_exploit_locally_constant_struct_members():
    sources, main = SourceCodeBuilder().add_file("""
subroutine main()
  implicit none
  type config
    logical :: cond = .true.
  end type config
  type(config) :: cond
  real :: out = 0.

  cond % cond = .true.
  if (cond % cond) out = out + 1.
  out = out*2
  if (cond % cond) then
    out = out + 1.
  else
    out = out - 1.
  end if

  if (out .gt. 20) cond % cond = .false.
  if (cond % cond) out = out + 100.

  cond % cond = .true.
  out = 7.2
  out = out*2.0
  out = fun(.not. cond % cond, out)

  if (cond % cond) out = out + 1.

contains
  real function fun(cond, out)
    implicit none
    logical, intent(in) :: cond
    real, intent(inout) :: out
    if (cond) out = out + 42
    fun = out + 1.0
  end function fun
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = optimizations.exploit_locally_constant_variables(ast)

    got = ast.tofortran()
    want = """
SUBROUTINE main
  IMPLICIT NONE
  TYPE :: config
    LOGICAL :: cond = .TRUE.
  END TYPE config
  TYPE(config) :: cond
  REAL :: out = 0.
  cond % cond = .TRUE.
  IF (.TRUE.) out = 0. + 1.
  out = out * 2
  IF (.TRUE.) THEN
    out = out + 1.
  ELSE
    out = out - 1.
  END IF
  IF (out .GT. 20) cond % cond = .FALSE.
  IF (cond % cond) out = out + 100.
  cond % cond = .TRUE.
  out = 7.2
  out = 7.2 * 2.0
  out = fun(.NOT. .TRUE., out)
  IF (.TRUE.) out = out + 1.
  CONTAINS
  REAL FUNCTION fun(cond, out)
    IMPLICIT NONE
    LOGICAL, INTENT(IN) :: cond
    REAL, INTENT(INOUT) :: out
    IF (cond) out = out + 42
    fun = out + 1.0
  END FUNCTION fun
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_exploit_locally_constant_pointers():
    sources, main = SourceCodeBuilder().add_file("""
subroutine main()
  implicit none
  type cfg
    real, pointer :: ptr => null()
  end type cfg
  type(cfg) :: c
  real, target :: data = 0.
  real, pointer :: ptr => null()
  integer, target :: i
  integer, pointer :: iptr => null()
  integer :: iarr(4) = 0

  ptr => data
  c % ptr => ptr
  iptr => i
  data = 2.
  ptr = data + 1.
  c % ptr = c % ptr + 7.
  if (c % ptr > 0.) c % ptr = 0.
  iptr = 4
  do i = 2, 4
    if (c % ptr > 0.) then
      c % ptr = c % ptr + 1.5
      iarr(iptr) = iarr(iptr-1) + 1
    end if
  end do
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = optimizations.exploit_locally_constant_variables(ast)

    got = ast.tofortran()
    want = """
SUBROUTINE main
  IMPLICIT NONE
  TYPE :: cfg
    REAL, POINTER :: ptr => NULL()
  END TYPE cfg
  TYPE(cfg) :: c
  REAL, TARGET :: data = 0.
  REAL, POINTER :: ptr => NULL()
  INTEGER, TARGET :: i
  INTEGER, POINTER :: iptr => NULL()
  INTEGER :: iarr(4) = 0
  ptr => data
  c % ptr => data
  iptr => i
  data = 2.
  data = 2.0 + 1.
  data = data + 7.
  IF (data > 0.) data = 0.
  i = 4
  DO i = 2, 4
    IF (data > 0.) THEN
      data = data + 1.5
      iarr(i) = iarr(i - 1) + 1
    END IF
  END DO
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_replace_case_selector_consts():
    sources, main = SourceCodeBuilder().add_file("""
module lib
  implicit none
  integer, parameter :: a = 1

contains
  subroutine foo(b)
    integer, intent(inout) :: b

    select case(b)
    case(a)
      b = 5
    end select
  end subroutine foo
end module lib

subroutine main()
  use lib
  implicit none
  integer :: b
  call foo(b)
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = optimizations.const_eval_nodes(ast)

    got = ast.tofortran()
    want = """
MODULE lib
  IMPLICIT NONE
  INTEGER, PARAMETER :: a = 1
  CONTAINS
  SUBROUTINE foo(b)
    INTEGER, INTENT(INOUT) :: b
    SELECT CASE (b)
    CASE (1)
      b = 5
    END SELECT
  END SUBROUTINE foo
END MODULE lib
SUBROUTINE main
  USE lib
  IMPLICIT NONE
  INTEGER :: b
  CALL foo(b)
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()


def test_constant_function_evaluation():
    sources, main = SourceCodeBuilder().add_file("""
subroutine main
  implicit none
  double precision :: a = sqrt(4.)
  double precision :: b = cos(0.)
  double precision :: c = abs(-3.)
end subroutine main
""").check_with_gfortran().get()
    ast = parse_and_improve(sources)
    ast = optimizations.const_eval_nodes(ast)

    got = ast.tofortran()
    want = """
SUBROUTINE main
  IMPLICIT NONE
  DOUBLE PRECISION :: a = 2.0
  DOUBLE PRECISION :: b = 1.0
  DOUBLE PRECISION :: c = 3.0
END SUBROUTINE main
""".strip()
    assert got == want
    SourceCodeBuilder().add_file(got).check_with_gfortran()
