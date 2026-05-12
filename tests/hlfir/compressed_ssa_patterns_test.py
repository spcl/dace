"""Audit tests for SSA-tree-compressing bridge patterns.

Each test isolates one Fortran pattern that lowers to an MLIR shape
where ``buildExpr`` collapses several SSA ops into a single textual
call (or where a subscripted array read sits inside a non-arith op).
The bridge's ``collectReads`` and ``buildExprWithSubscripts`` must
keep textual occurrences and AccessInfo counts in lockstep, OR keep
subscripts on inner loads when an IF-condition lifts the expression
to an interstate edge.

Both the SDFG and the reference are compiled from the SAME Fortran
source — f2py-built reference per ``feedback_e2e_numerical``.
"""
from __future__ import annotations

import numpy as np
import pytest

from _util import build_sdfg, have_flang
from _helpers import f2py, sdfg_call_args

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _build_and_run(tmp_path, *, src: str, name: str, entry: str, fortran_call, int_args=None):
    """Compile Fortran via f2py + bridge, return (ref_module, sdfg)."""
    ref = f2py(src, tmp_path / 'ref', f'{name}_ref')
    sdfg_dir = tmp_path / 'sdfg'
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name=name, entry=entry).build()
    return ref, sdfg


def test_max_min_shared_load(tmp_path):
    """Shared array load reused in MAX and MIN of one expression.
    cloudsc line 2436 shape.  Was the original ``collectReads``
    cmp-skip bug.
    """
    src = """
SUBROUTINE cov_update(zcov, za, klon, klev)
integer :: klon, klev
double precision, intent(inout) :: zcov(klon)
double precision za(klon, klev)
integer jl, jk
DO jk = 2, klev
    DO jl = 1, klon
        zcov(jl) = 1.0d0 - (1.0d0 - zcov(jl)) &
            * (1.0d0 - MAX(za(jl, jk), za(jl, jk-1))) &
            / (1.0d0 - MIN(za(jl, jk-1), 1.0d0 - 1.0d-6))
    ENDDO
ENDDO
END SUBROUTINE cov_update
"""
    ref, sdfg = _build_and_run(tmp_path, src=src, name='cov_update', entry='_QPcov_update', fortran_call=None)
    rng = np.random.default_rng(7)
    klon, klev = 1, 6
    za = np.asfortranarray(rng.random((klon, klev)))
    zcov_in = np.asfortranarray(rng.random(klon))

    zcov_ref = zcov_in.copy()
    ref.cov_update(zcov_ref, za, klon, klev)

    zcov = zcov_in.copy()
    sdfg(zcov=zcov, za=za, klon=klon, klev=klev)
    np.testing.assert_allclose(zcov, zcov_ref, rtol=1e-12, atol=1e-12)


def test_modulo_intrinsic(tmp_path):
    """Fortran ``MODULO(i, n)`` lowers to a 9-op SSA tree
    (rem/xori/cmpi/cmpi/andi/addi/select) the bridge collapses to
    ``floor_mod(i, n)``.  Sibling-pattern to MIN/MAX: collectReads
    would over-count SSA references to ``i`` and ``n``.
    """
    src = """
SUBROUTINE wrap_mod(arr, mods, n_arr, n)
integer :: n_arr, n
integer :: arr(n_arr)
integer :: mods(n_arr)
integer i
DO i = 1, n_arr
    mods(i) = MODULO(arr(i), n)
ENDDO
END SUBROUTINE wrap_mod
"""
    ref, sdfg = _build_and_run(tmp_path, src=src, name='wrap_mod', entry='_QPwrap_mod', fortran_call=None)
    n_arr = 8
    n = 3
    arr = np.array([1, -1, 7, -7, 10, -10, 0, 11], dtype=np.int32)

    mods_ref = np.zeros(n_arr, dtype=np.int32)
    # f2py auto-derives shape symbols from array dims, so pass by kw.
    ref.wrap_mod(arr=arr, mods=mods_ref, n=n)

    mods = np.zeros(n_arr, dtype=np.int32)
    int_args = sdfg_call_args(sdfg, {'n_arr': n_arr, 'n': n})
    sdfg(arr=arr, mods=mods, **int_args)
    np.testing.assert_array_equal(mods, mods_ref)


def test_power_squared(tmp_path):
    """``a(i) ** 2`` → ``a*a``: two textual mul operands share one
    load SSA value.  Sanity check that 1:1 occurrence-to-connector
    contract still holds for the simple shared-load case.
    """
    src = """
SUBROUTINE sqrarr(a, out, n)
integer :: n
double precision a(n), out(n)
integer i
DO i = 1, n
    out(i) = a(i) ** 2
ENDDO
END SUBROUTINE sqrarr
"""
    ref, sdfg = _build_and_run(tmp_path, src=src, name='sqrarr', entry='_QPsqrarr', fortran_call=None)
    rng = np.random.default_rng(7)
    n = 5
    a = rng.random(n)
    out_ref = np.zeros(n)
    ref.sqrarr(a, out_ref, n)
    out = np.zeros(n)
    sdfg(a=a, out=out, n=n)
    np.testing.assert_allclose(out, out_ref, rtol=1e-12, atol=1e-12)


def test_abs_in_if_condition(tmp_path):
    """``ABS(a(i)) > ABS(b(i))`` in an IF condition.  The IF lifts
    the condition to an interstate-edge expression; the abs's
    operand load must KEEP its subscript or C++ emits
    ``abs(array_ptr)``.
    """
    src = """
SUBROUTINE which_bigger(a, b, out, n)
integer :: n
double precision a(n), b(n)
integer :: out(n)
integer i
DO i = 1, n
    IF (ABS(a(i)) > ABS(b(i))) THEN
        out(i) = 1
    ELSE
        out(i) = 0
    ENDIF
ENDDO
END SUBROUTINE which_bigger
"""
    ref, sdfg = _build_and_run(tmp_path, src=src, name='which_bigger', entry='_QPwhich_bigger', fortran_call=None)
    rng = np.random.default_rng(7)
    n = 8
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    out_ref = np.zeros(n, dtype=np.int32)
    ref.which_bigger(a, b, out_ref, n)
    out = np.zeros(n, dtype=np.int32)
    sdfg(a=a, b=b, out=out, n=n)
    np.testing.assert_array_equal(out, out_ref)


def test_sqrt_in_if_condition(tmp_path):
    """``SQRT(a(i)) > thr`` — sibling of ABS, exercises math.sqrt
    in the unary-intrinsic IF-condition peel.
    """
    src = """
SUBROUTINE flag_sqrt(a, thr, out, n)
integer :: n
double precision a(n)
double precision thr
integer :: out(n)
integer i
DO i = 1, n
    IF (SQRT(a(i)) > thr) THEN
        out(i) = 1
    ELSE
        out(i) = 0
    ENDIF
ENDDO
END SUBROUTINE flag_sqrt
"""
    ref, sdfg = _build_and_run(tmp_path, src=src, name='flag_sqrt', entry='_QPflag_sqrt', fortran_call=None)
    rng = np.random.default_rng(7)
    n = 8
    a = rng.random(n) * 4.0
    thr = 1.5
    out_ref = np.zeros(n, dtype=np.int32)
    ref.flag_sqrt(a, thr, out_ref, n)
    out = np.zeros(n, dtype=np.int32)
    from dace.data import Scalar
    thr_arg = thr if isinstance(sdfg.arglist().get('thr'), Scalar) else np.array([thr], dtype=np.float64)
    sdfg(a=a, thr=thr_arg, out=out, n=n)
    np.testing.assert_array_equal(out, out_ref)


def test_exp_in_if_condition(tmp_path):
    """``EXP(a(i)) > thr`` — math.exp peel through buildExprWithSubscripts."""
    src = """
SUBROUTINE flag_exp(a, thr, out, n)
integer :: n
double precision a(n)
double precision thr
integer :: out(n)
integer i
DO i = 1, n
    IF (EXP(a(i)) > thr) THEN
        out(i) = 1
    ELSE
        out(i) = 0
    ENDIF
ENDDO
END SUBROUTINE flag_exp
"""
    ref, sdfg = _build_and_run(tmp_path, src=src, name='flag_exp', entry='_QPflag_exp', fortran_call=None)
    rng = np.random.default_rng(7)
    n = 8
    a = rng.random(n) * 2.0
    thr = 3.0
    out_ref = np.zeros(n, dtype=np.int32)
    ref.flag_exp(a, thr, out_ref, n)
    out = np.zeros(n, dtype=np.int32)
    from dace.data import Scalar
    thr_arg = thr if isinstance(sdfg.arglist().get('thr'), Scalar) else np.array([thr], dtype=np.float64)
    sdfg(a=a, thr=thr_arg, out=out, n=n)
    np.testing.assert_array_equal(out, out_ref)


def test_log_in_if_condition(tmp_path):
    """``LOG(a(i)) > 0`` — math.log peel."""
    src = """
SUBROUTINE flag_log(a, out, n)
integer :: n
double precision a(n)
integer :: out(n)
integer i
DO i = 1, n
    IF (LOG(a(i)) > 0.0d0) THEN
        out(i) = 1
    ELSE
        out(i) = 0
    ENDIF
ENDDO
END SUBROUTINE flag_log
"""
    ref, sdfg = _build_and_run(tmp_path, src=src, name='flag_log', entry='_QPflag_log', fortran_call=None)
    rng = np.random.default_rng(7)
    n = 8
    a = rng.random(n) * 4.0 + 0.1  # positive
    out_ref = np.zeros(n, dtype=np.int32)
    ref.flag_log(a, out_ref, n)
    out = np.zeros(n, dtype=np.int32)
    sdfg(a=a, out=out, n=n)
    np.testing.assert_array_equal(out, out_ref)


def test_sin_cos_arith_in_if(tmp_path):
    """``SIN(a(i)) + COS(b(i)) > 0`` — two unary intrinsics on
    different array reads combined by addf, all inside an IF
    condition.  Both subscripts must survive the peel.
    """
    src = """
SUBROUTINE flag_trig(a, b, out, n)
integer :: n
double precision a(n), b(n)
integer :: out(n)
integer i
DO i = 1, n
    IF (SIN(a(i)) + COS(b(i)) > 0.0d0) THEN
        out(i) = 1
    ELSE
        out(i) = 0
    ENDIF
ENDDO
END SUBROUTINE flag_trig
"""
    ref, sdfg = _build_and_run(tmp_path, src=src, name='flag_trig', entry='_QPflag_trig', fortran_call=None)
    rng = np.random.default_rng(7)
    n = 8
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    out_ref = np.zeros(n, dtype=np.int32)
    ref.flag_trig(a, b, out_ref, n)
    out = np.zeros(n, dtype=np.int32)
    sdfg(a=a, b=b, out=out, n=n)
    np.testing.assert_array_equal(out, out_ref)


def test_same_element_two_arith_uses(tmp_path):
    """``a(i) + 2.0 * a(i)`` — CSE-shared load, 2 textual occurrences.
    Sanity that 1:1 mapping works without dedup.
    """
    src = """
SUBROUTINE triple(a, out, n)
integer :: n
double precision a(n), out(n)
integer i
DO i = 1, n
    out(i) = a(i) + 2.0d0 * a(i)
ENDDO
END SUBROUTINE triple
"""
    ref, sdfg = _build_and_run(tmp_path, src=src, name='triple', entry='_QPtriple', fortran_call=None)
    rng = np.random.default_rng(7)
    n = 5
    a = rng.standard_normal(n)
    out_ref = np.zeros(n)
    ref.triple(a, out_ref, n)
    out = np.zeros(n)
    sdfg(a=a, out=out, n=n)
    np.testing.assert_allclose(out, out_ref, rtol=1e-12, atol=1e-12)


def test_max_neighbor_pairs(tmp_path):
    """``MAX(a(i-1), a(i))`` — shared loads with potentially-swapped
    cmp operand order for stride-2 indexing.
    """
    src = """
SUBROUTINE max_neighbor(a, out, n)
integer :: n
double precision a(n)
double precision :: out(n)
integer i
out(1) = a(1)
DO i = 2, n
    out(i) = MAX(a(i-1), a(i))
ENDDO
END SUBROUTINE max_neighbor
"""
    ref, sdfg = _build_and_run(tmp_path, src=src, name='max_neighbor', entry='_QPmax_neighbor', fortran_call=None)
    rng = np.random.default_rng(7)
    n = 6
    a = rng.standard_normal(n)
    out_ref = np.zeros(n)
    ref.max_neighbor(a, out_ref, n)
    out = np.zeros(n)
    sdfg(a=a, out=out, n=n)
    np.testing.assert_allclose(out, out_ref, rtol=1e-12, atol=1e-12)
