"""Coverage for the optional Fortran source pre-processor (rewrites
``IF (intvar)`` to ``IF (intvar /= 0)`` for INTEGER scalars).
"""
from __future__ import annotations

from dace.frontend.hlfir.preprocess import preprocess_fortran


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
