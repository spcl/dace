"""Unit + e2e coverage for ``rewrite_integer_powers``.

The pass expands an *integer-valued REAL-literal* power (``x**2.0``,
``x**3.0_JPRB``) to an explicit multiply so the bridge's g++ codegen
and the gfortran reference emit byte-identical arithmetic.  Scope is
deliberately narrow:

* a *bare-integer* exponent (``x**2``) is left for flang's own
  integer-power lowering -- it is already bit-identical to gfortran;
* genuine fractional / symbolic powers (``**0.5``, ``**n``) must stay
  as ``pow()``;
* a base containing a function / array reference (``f(x)``,
  ``arr(i,j)``, ``a%b(i)%c``) is left untouched -- duplicating it
  would call twice (impure functions; the bridge's call-inlining
  shares the callee accumulator across copies);
* the rewrite adds exactly one outer pair of parentheses (the base is
  a primary), keeping the diff close to the source while preserving
  precedence.

The ``_eval_equivalent`` tests prove the precedence property
numerically: the rewritten arithmetic is also valid Python, so we
evaluate the original (``**``) and rewritten (``*``) strings under
random bindings and assert exact equality.
"""

from pathlib import Path

import numpy as np
import pytest

from dace.frontend.hlfir.preprocess import rewrite_integer_powers

# --- base extraction / minimal parenthesisation ---------------------


def test_simple_identifier_square():
    assert rewrite_integer_powers("y = x**2.0") == "y = (x*x)"


def test_simple_identifier_cube():
    assert rewrite_integer_powers("y = x**3.0_JPRB") == "y = (x*x*x)"


def test_parenthesised_base_keeps_its_parens():
    assert (rewrite_integer_powers("z = (a - b)**2.0") == "z = ((a - b)*(a - b))")


def test_pure_designator_chain_base():
    assert rewrite_integer_powers("w = a%b%c**2.0") == "w = (a%b%c*a%b%c)"


def test_parenthesised_exponent():
    assert rewrite_integer_powers("y = x**(2.0_JPRB)") == "y = (x*x)"


def test_double_exponent_form():
    assert rewrite_integer_powers("y = x**2.0D0") == "y = (x*x)"


def test_multiple_on_one_line():
    assert (rewrite_integer_powers("r = a**2.0 + b**3.0") == "r = (a*a) + (b*b*b)")


# --- things the pass must NOT touch ---------------------------------


@pytest.mark.parametrize(
    "expr",
    [
        "y = x**0.5_JPRB",  # genuine sqrt
        "y = x**1.5",  # genuine fractional
        "y = x**2",  # bare integer -> flang lowers it
        "y = x**3",  # bare integer
        "y = x**23",  # bare integer, not 2/3
        "y = x**n",  # symbolic exponent
        "y = x**2.5",  # non-integer real
        "rho = base**rcl_const4r",
        # call / array bases -- unsafe to duplicate.
        "z = arr(i, j)**2.0",
        "z = foealfa(ptare)**3.0",
        "z = a%b(i)%c**2.0",
        "d = custom_sum(d)**2.0",
    ])
def test_left_untouched(expr: str):
    assert rewrite_integer_powers(expr) == expr


def test_comment_left_untouched():
    src = "y = a**2.0   ! note: keep x**2.0 in the comment\n"
    out = rewrite_integer_powers(src)
    assert out == "y = (a*a)   ! note: keep x**2.0 in the comment\n"


def test_idempotent():
    once = rewrite_integer_powers("q = (p + 1)**3.0 * r**2.0")
    assert rewrite_integer_powers(once) == once


# --- precedence preserved (numeric proof via Python eval) -----------


@pytest.mark.parametrize("expr", [
    "2*x**2.0",
    "a/b**2.0",
    "-x**2.0",
    "c + x**3.0 + d",
    "a*b**2.0 + c",
    "(p - q)**2.0 / r",
    "x**2.0 * y**3.0",
    "1.0/(t - k)**2.0",
    "-(a - b)**3.0",
])
def test_eval_equivalent(expr: str):
    """Original ``**`` form and rewritten ``*`` form must be
    numerically close under arbitrary bindings -- this catches any
    missing parenthesis that would flip operator precedence (such a
    bug shifts the value by orders of magnitude).

    ``math.isclose`` rather than ``==``: ``x**2.0`` is libm ``pow``
    while the rewrite emits ``x*x``; they differ at the last ULP --
    which is exactly the rounding difference this pass exists to
    remove, not a precedence error."""
    import math

    rng = np.random.default_rng(0)
    env = {n: float(rng.uniform(1.5, 4.0)) for n in "abcdklpqrtxy"}
    rewritten = rewrite_integer_powers(expr)
    assert math.isclose(eval(rewritten, {}, env), eval(expr, {}, env),
                        rel_tol=1e-9), (f"{expr!r} -> {rewritten!r} changed the value")


# --- e2e: bridge vs gfortran on a base that needs wrapping ----------


def test_e2e_parenthesised_base_matches_gfortran(tmp_path: Path):
    """``(t(i) - k)**2.0`` -- a parenthesised base inside a quotient,
    plus a scalar ``s**3.0`` -- must stay numerically identical after
    the rewrite.  Build through the bridge (which applies the rewrite)
    and compare against a gfortran reference of the same source."""
    from _util import build_sdfg, f2py_compile, have_flang

    if not have_flang():
        pytest.skip("flang-new-21 not on PATH")

    src = """
subroutine pw(t, k, s, out, n)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in) :: t(n)
  real(8), intent(in) :: k
  real(8), intent(in) :: s
  real(8), intent(inout) :: out(n)
  integer :: i
  do i = 1, n
    out(i) = 1.0_8 / (t(i) - k)**2.0 + s**3.0
  end do
end subroutine pw
"""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir(parents=True, exist_ok=True)

    sdfg = build_sdfg(src, sdfg_dir, name="pw", entry="_QPpw").build()
    sdfg.validate()
    ref = f2py_compile(src, ref_dir, "pw_ref")

    rng = np.random.default_rng(42)
    t = np.asfortranarray(rng.uniform(2.0, 5.0, 16).astype(np.float64))
    k = np.float64(0.5)
    s = np.float64(1.7)
    o_ref = np.zeros(16, dtype=np.float64, order="F")
    o_sdfg = np.zeros(16, dtype=np.float64, order="F")
    ref.pw(t, k, s, o_ref)  # n auto-derived from shape(t)
    sdfg(n=np.int32(16), t=t, k=k, s=s, out=o_sdfg)
    np.testing.assert_array_equal(o_sdfg, o_ref)
