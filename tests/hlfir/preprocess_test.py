"""Coverage for the Fortran source pre-processor: the ``IF (intvar)``
rewrite, the ``x**2`` / ``x**3`` -> explicit-multiply expansion, and the
single/default REAL literal -> double-precision promotion.
"""

from dace.frontend.hlfir.preprocess import (
    preprocess_fortran,
    promote_real_literals_to_double,
    rewrite_integer_powers,
)


def test_rewrites_bare_integer_if():
    src = """
SUBROUTINE legacy(flag)
  INTEGER :: flag
  IF (flag) THEN
    CALL do_thing()
  END IF
END SUBROUTINE
"""
    out = preprocess_fortran(src)
    assert "IF (flag /= 0)" in out
    assert "IF (flag)" not in out


def test_leaves_logical_if_alone():
    src = """
SUBROUTINE clean(p, q)
  LOGICAL :: p
  INTEGER :: q
  IF (p) THEN
    q = 1
  END IF
END SUBROUTINE
"""
    out = preprocess_fortran(src)
    # ``p`` is LOGICAL -- bridge must NOT rewrite it.
    assert "IF (p)" in out
    assert "IF (p /= 0)" not in out


def test_leaves_compound_condition_alone():
    src = """
SUBROUTINE compound(a, b)
  INTEGER :: a, b
  IF (a /= 0 .AND. b > 0) THEN
    CALL do_thing()
  END IF
END SUBROUTINE
"""
    out = preprocess_fortran(src)
    # The ``IF (a /= 0 .AND. ...)`` shape was already legal Fortran;
    # the rewriter only handles single-identifier conditions.
    assert "IF (a /= 0 .AND. b > 0)" in out


def test_rewrites_multi_decl_line():
    src = """
SUBROUTINE multi(flag1, flag2)
  INTEGER :: flag1, flag2
  IF (flag1) flag2 = 1
  IF (flag2) RETURN
END SUBROUTINE
"""
    out = preprocess_fortran(src)
    assert "IF (flag1 /= 0)" in out
    assert "IF (flag2 /= 0)" in out


def test_skips_integer_arrays():
    src = """
SUBROUTINE arr(a, n)
  INTEGER :: n
  INTEGER :: a(n)
  IF (n) RETURN
END SUBROUTINE
"""
    out = preprocess_fortran(src)
    # ``n`` is a true scalar INTEGER and should be rewritten; the
    # presence of the integer array ``a(n)`` declaration must not
    # confuse the scalar-name collector.
    assert "IF (n /= 0)" in out


def test_idempotent():
    src = """
SUBROUTINE leg(f)
  INTEGER :: f
  IF (f) RETURN
END SUBROUTINE
"""
    once = preprocess_fortran(src)
    twice = preprocess_fortran(once)
    assert once == twice


def test_no_integer_decls_passthrough():
    src = """
SUBROUTINE plain(x)
  REAL(8) :: x
  x = x + 1.0D0
END SUBROUTINE
"""
    assert preprocess_fortran(src) == src


# --------------------------------------------------------------------------
# rewrite_integer_powers -- only integer-valued REAL exponents become
# repeated multiplies; bare integers / fractional powers are untouched.
# --------------------------------------------------------------------------


def test_pow_real_two_and_three():
    # Minimal change: one outer pair only -- the base is a primary so
    # each factor needs no wrapping of its own.
    assert rewrite_integer_powers("y = x**2.0") == "y = (x*x)"
    assert rewrite_integer_powers("y = x**3.0_JPRB") == "y = (x*x*x)"


def test_pow_base_parenthesised_for_precedence():
    # Already-parenthesised base keeps its own parens; the single outer
    # pair preserves precedence (2.0*(t*t), a/(b*b), -(x*x)).
    assert rewrite_integer_powers("z = (a-b)**2.0") == "z = ((a-b)*(a-b))"
    assert rewrite_integer_powers("f = 2.0*t**2.0 + a/b**2.0") == "f = 2.0*(t*t) + a/(b*b)"
    assert rewrite_integer_powers("g = -x**2.0") == "g = -(x*x)"


def test_pow_call_and_array_bases_left_untouched():
    # Duplicating a function/array reference would call twice -- unsafe
    # for impure functions and shared inlined accumulators.  Such
    # powers are left for flang's own lowering.
    assert rewrite_integer_powers("h = arr(i,j)**2.0D0") == "h = arr(i,j)**2.0D0"
    assert rewrite_integer_powers("k = s%m(2)%v**3.0") == "k = s%m(2)%v**3.0"
    assert rewrite_integer_powers("q = custom_sum(d)**2.0") == "q = custom_sum(d)**2.0"
    # A pure designator chain (no call/subscript) is still safe.
    assert rewrite_integer_powers("w = a%b%c**2.0") == "w = (a%b%c*a%b%c)"


def test_pow_leaves_bare_integer_and_fractional_alone():
    # Bare integer exponent: flang lowers x**2 correctly itself.
    assert rewrite_integer_powers("c = z**2") == "c = z**2"
    # Genuine fractional powers must stay as pow().
    assert rewrite_integer_powers("d = r**0.5_JPRB") == "d = r**0.5_JPRB"
    assert rewrite_integer_powers("e = w**2.5") == "e = w**2.5"
    assert rewrite_integer_powers("p = rho**0.78") == "p = rho**0.78"


def test_pow_comment_untouched_and_idempotent():
    assert rewrite_integer_powers("z = a**2.0  ! b**2.0 keep") == "z = (a*a)  ! b**2.0 keep"
    once = rewrite_integer_powers("v = (p-q)**2.0 + zt**3.0_JPRB")
    assert rewrite_integer_powers(once) == once


# --------------------------------------------------------------------------
# promote_real_literals_to_double -- single/default REAL literals become
# explicit double; already-double and integers are left as-is.
# --------------------------------------------------------------------------


def test_double_bare_and_single_kind():
    assert promote_real_literals_to_double("x = 2.0") == "x = 2.0D0"
    assert promote_real_literals_to_double("y = 4.2_JPRM + 1.0_4") == "y = 4.2D0 + 1.0D0"
    assert promote_real_literals_to_double("b = 1.0e-3 + .5 + 1.") == "b = 1.0D-3 + .5D0 + 1.D0"


def test_double_leaves_already_double_and_integers():
    # _JPRB / _8 / D-exponent are already double.
    assert promote_real_literals_to_double("z = 0.85E5_JPRB + 1.5D0 + 1.0_8") == "z = 0.85E5_JPRB + 1.5D0 + 1.0_8"
    # Integers and kind selectors must not be touched.
    assert promote_real_literals_to_double("n = 137 + i*2") == "n = 137 + i*2"
    assert promote_real_literals_to_double("REAL(KIND=8) :: q") == "REAL(KIND=8) :: q"
    assert promote_real_literals_to_double("k = SELECTED_REAL_KIND(13,300)") == "k = SELECTED_REAL_KIND(13,300)"


def test_double_skips_identifiers_strings_comments():
    assert promote_real_literals_to_double("r = R2ES + X1 + a2b") == "r = R2ES + X1 + a2b"
    assert promote_real_literals_to_double("msg = 'keep 2.0 here'  ! 3.0 too") == "msg = 'keep 2.0 here'  ! 3.0 too"
    assert promote_real_literals_to_double("u = 6.0 ! 7.0 stays") == "u = 6.0D0 ! 7.0 stays"


def test_double_idempotent():
    once = promote_real_literals_to_double("v = 2.0 + 0.85E5 + 1.0_JPRM")
    assert promote_real_literals_to_double(once) == once
