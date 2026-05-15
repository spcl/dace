"""Unit tests for RAR / WAR / RAW / WAW data dependencies between
multiple tasklets in the same Fortran basic block.

Catches the bridge bug that surfaced in cloudsc Section 4.5
EVAPORATION (commit e85ec1f8f / project_hlfir_cloudsc_section_4_5_bisection):
the codegen scheduler reordered a ``ZQXFG -= ZEVAP`` tasklet to run
BEFORE the ``ZCOVPTOT = MAX(.., ZCOVPTOT-ZA)*ZEVAP/ZQXFG)`` tasklet
that read ZQXFG, because no SDFG edge connected them.  Fortran's
sequential WAR (write-after-read) ordering was violated.

Each test below pins one of the four standard dataflow-dependency
patterns:

  RAW (Read After Write)  -- writer must complete before reader sees value
  WAR (Write After Read)  -- reader must complete before writer overwrites
  WAW (Write After Write) -- later write wins; order matters
  RAR (Read After Read)   -- no dependency; either order safe

For each pattern: build the SDFG via the HLFIR bridge AND an f2py
reference from the same Fortran source on identical seeded inputs,
then assert numerical equivalence at strict ulp-level tolerance.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _f2py(src_text: str, out_dir: Path, mod_name: str):
    """f2py-compile inline ``src_text`` with strict FP flags."""
    if shutil.which("gfortran") is None:
        pytest.skip("gfortran not available")
    if shutil.which("meson") is None:
        pytest.skip("meson not available (f2py backend on Python>=3.12)")
    out_dir.mkdir(parents=True, exist_ok=True)
    src = out_dir / f"{mod_name}.f90"
    src.write_text(src_text)
    subprocess.check_call(
        [
            sys.executable, "-m", "numpy.f2py", "-c",
            str(src), "-m", mod_name, "--quiet", "--f90flags=-O0 -fno-fast-math -ffp-contract=off"
        ],
        cwd=out_dir,
    )
    if str(out_dir) not in sys.path:
        sys.path.insert(0, str(out_dir))
    __import__(mod_name)
    return sys.modules[mod_name]


def _sdfg_args(sdfg, int_vals):
    """Route ints to Scalar-or-length-1 by SDFG descriptor."""
    from dace.data import Scalar
    arglist = sdfg.arglist()
    out = {}
    for k, v in int_vals.items():
        desc = arglist.get(k)
        if desc is None or isinstance(desc, Scalar):
            out[k] = v
        else:
            out[k] = np.array([v], dtype=np.int32)
    return out


def test_raw_dependency(tmp_path):
    """RAW: writer must complete before reader sees the value.

    Fortran:
        x(i) = x(i) * 2.0       ! tasklet 1: writes x[i] (in-place)
        y(i) = x(i) + 1.0       ! tasklet 2: reads x[i] -- MUST see new value

    Expected: y[i] = (x_init[i] * 2.0) + 1.0
    Wrong (if bridge reorders): y[i] = x_init[i] + 1.0
    """
    src = """
SUBROUTINE raw_kernel(n, x, y)
  IMPLICIT NONE
  INTEGER(KIND=4), VALUE :: n
  REAL(KIND=8), INTENT(INOUT) :: x(n)
  REAL(KIND=8), INTENT(INOUT) :: y(n)
  INTEGER(KIND=4) :: i
  DO i = 1, n
    x(i) = x(i) * 2.0_8
    y(i) = x(i) + 1.0_8
  END DO
END SUBROUTINE raw_kernel
"""
    ref = _f2py(src, tmp_path / "ref", "raw_ref")
    (tmp_path / "sdfg").mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="raw_kernel", pipeline="hlfir-propagate-shapes").build()

    rng = np.random.default_rng(300)
    n = 32
    x_init = np.asfortranarray(rng.standard_normal(n).astype(np.float64))

    x_ref = np.array(x_init, order="F")
    y_ref = np.zeros(n, dtype=np.float64, order="F")
    ref.raw_kernel(x_ref, y_ref)

    x_sdfg = np.array(x_init, order="F")
    y_sdfg = np.zeros(n, dtype=np.float64, order="F")
    kw = dict(x=x_sdfg, y=y_sdfg)
    kw.update(_sdfg_args(sdfg, dict(n=n)))
    sdfg(**kw)

    np.testing.assert_array_equal(x_sdfg,
                                  x_ref,
                                  err_msg="RAW: x (in-place writer) diverged -- impossible unless tasklets reordered")
    np.testing.assert_array_equal(y_sdfg,
                                  y_ref,
                                  err_msg="RAW: y must equal (x_init * 2) + 1 -- bridge dropped the write-before-read")


def test_war_dependency(tmp_path):
    """WAR: reader must complete before writer overwrites.

    Fortran:
        y(i) = x(i) * 2.0       ! tasklet 1: reads x[i]
        x(i) = 99.0_8           ! tasklet 2: writes x[i] -- MUST not invalidate t1

    Expected: y[i] = x_init[i] * 2.0,  x[i] = 99.0
    Wrong (if bridge reorders): y[i] = 99.0 * 2.0 = 198.0
    """
    src = """
SUBROUTINE war_kernel(n, x, y)
  IMPLICIT NONE
  INTEGER(KIND=4), VALUE :: n
  REAL(KIND=8), INTENT(INOUT) :: x(n)
  REAL(KIND=8), INTENT(INOUT) :: y(n)
  INTEGER(KIND=4) :: i
  DO i = 1, n
    y(i) = x(i) * 2.0_8
    x(i) = 99.0_8
  END DO
END SUBROUTINE war_kernel
"""
    ref = _f2py(src, tmp_path / "ref", "war_ref")
    (tmp_path / "sdfg").mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="war_kernel", pipeline="hlfir-propagate-shapes").build()

    rng = np.random.default_rng(301)
    n = 32
    x_init = np.asfortranarray(rng.standard_normal(n).astype(np.float64))

    x_ref = np.array(x_init, order="F")
    y_ref = np.zeros(n, dtype=np.float64, order="F")
    ref.war_kernel(x_ref, y_ref)

    x_sdfg = np.array(x_init, order="F")
    y_sdfg = np.zeros(n, dtype=np.float64, order="F")
    kw = dict(x=x_sdfg, y=y_sdfg)
    kw.update(_sdfg_args(sdfg, dict(n=n)))
    sdfg(**kw)

    np.testing.assert_array_equal(x_sdfg, x_ref, err_msg="WAR: x final values should all be 99.0")
    np.testing.assert_array_equal(y_sdfg,
                                  y_ref,
                                  err_msg="WAR: y must equal x_init * 2 -- bridge reordered the write "
                                  "before the read (this is the cloudsc Section 4.5 bug shape)")


def test_waw_dependency(tmp_path):
    """WAW: two writes to same location; later write wins.

    Fortran:
        x(i) = 1.0_8            ! tasklet 1: writes x[i]
        x(i) = 2.0_8            ! tasklet 2: writes x[i]

    Expected: x[i] = 2.0  (tasklet 2's value)
    Wrong (if bridge reorders): x[i] = 1.0
    """
    src = """
SUBROUTINE waw_kernel(n, x)
  IMPLICIT NONE
  INTEGER(KIND=4), VALUE :: n
  REAL(KIND=8), INTENT(INOUT) :: x(n)
  INTEGER(KIND=4) :: i
  DO i = 1, n
    x(i) = 1.0_8
    x(i) = 2.0_8
  END DO
END SUBROUTINE waw_kernel
"""
    ref = _f2py(src, tmp_path / "ref", "waw_ref")
    (tmp_path / "sdfg").mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="waw_kernel", pipeline="hlfir-propagate-shapes").build()

    n = 32
    x_ref = np.zeros(n, dtype=np.float64, order="F")
    ref.waw_kernel(x_ref)
    x_sdfg = np.zeros(n, dtype=np.float64, order="F")
    kw = dict(x=x_sdfg)
    kw.update(_sdfg_args(sdfg, dict(n=n)))
    sdfg(**kw)

    np.testing.assert_array_equal(x_sdfg,
                                  x_ref,
                                  err_msg="WAW: final x must equal 2.0 (later write) -- bridge "
                                  "reordered the two writes")


def test_rar_dependency(tmp_path):
    """RAR: two reads of same value; no ordering constraint.

    Fortran:
        y(i) = x(i) * 2.0       ! tasklet 1: reads x[i]
        z(i) = x(i) + 1.0       ! tasklet 2: reads x[i] (same value)

    Expected: y[i] = x[i]*2, z[i] = x[i]+1.  Any tasklet order is OK.
    """
    src = """
SUBROUTINE rar_kernel(n, x, y, z)
  IMPLICIT NONE
  INTEGER(KIND=4), VALUE :: n
  REAL(KIND=8), INTENT(IN)    :: x(n)
  REAL(KIND=8), INTENT(INOUT) :: y(n)
  REAL(KIND=8), INTENT(INOUT) :: z(n)
  INTEGER(KIND=4) :: i
  DO i = 1, n
    y(i) = x(i) * 2.0_8
    z(i) = x(i) + 1.0_8
  END DO
END SUBROUTINE rar_kernel
"""
    ref = _f2py(src, tmp_path / "ref", "rar_ref")
    (tmp_path / "sdfg").mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="rar_kernel", pipeline="hlfir-propagate-shapes").build()

    rng = np.random.default_rng(303)
    n = 32
    x = np.asfortranarray(rng.standard_normal(n).astype(np.float64))

    y_ref = np.zeros(n, dtype=np.float64, order="F")
    z_ref = np.zeros(n, dtype=np.float64, order="F")
    ref.rar_kernel(x, y_ref, z_ref)

    y_sdfg = np.zeros(n, dtype=np.float64, order="F")
    z_sdfg = np.zeros(n, dtype=np.float64, order="F")
    kw = dict(x=x, y=y_sdfg, z=z_sdfg)
    kw.update(_sdfg_args(sdfg, dict(n=n)))
    sdfg(**kw)

    np.testing.assert_array_equal(y_sdfg, y_ref, err_msg="RAR: y mismatch")
    np.testing.assert_array_equal(z_sdfg, z_ref, err_msg="RAR: z mismatch")


# --------------------------------------------------------------------
# The cloudsc-shape regression test: combines RAW + WAR in a single
# basic block, exactly mirroring 4.5a Abel-Boutle ZCOVPTOT update.
# --------------------------------------------------------------------
def test_cloudsc_shape_war_via_division(tmp_path):
    """Reproducer matching cloudsc.F90:3174-3188 structure:

        e = MIN(d, f)             ! e = evap (clamped to f when d>f)
        s_qv_qr += e              ! source/sink accumulators (RAR on e)
        s_qr_qv -= e
        cv = MAX(g, cv - MAX(0, (cv - z) * e / f))    ! reads f
        f = f - e                                     ! writes f -- WAR vs above

    With ``d > f`` initially, ``e = f``.  Then in the cv update, ``e/f
    = 1.0`` exactly -> inner term = (cv - z), so cv = MAX(g, z).

    But if the bridge reorders ``f -= e`` before the cv update, ``f``
    is decreased -- often to a tiny positive number close to 0 (since
    e ≈ f).  Then ``e / f_new`` explodes -> cv clamps to g (the floor).
    This is exactly the cloudsc behavior: SDFG clamps to RCOVPMIN
    while gfortran computes the correct MAX(g, z).
    """
    src = """
SUBROUTINE cloudsc_war_shape(n, d, f, z, g, cv, e_out)
  IMPLICIT NONE
  INTEGER(KIND=4), VALUE :: n
  REAL(KIND=8), INTENT(IN)    :: d(n)         ! ZDPEVAP-like
  REAL(KIND=8), INTENT(INOUT) :: f(n)         ! ZQXFG-like
  REAL(KIND=8), INTENT(IN)    :: z(n)         ! ZA-like
  REAL(KIND=8), VALUE         :: g            ! RCOVPMIN-like floor
  REAL(KIND=8), INTENT(INOUT) :: cv(n)        ! ZCOVPTOT-like
  REAL(KIND=8), INTENT(INOUT) :: e_out(n)     ! capture evap
  INTEGER(KIND=4) :: i
  REAL(KIND=8)    :: e
  DO i = 1, n
    e = MIN(d(i), f(i))
    cv(i) = MAX(g, cv(i) - MAX(0.0_8, (cv(i) - z(i)) * e / f(i)))
    f(i)  = f(i) - e
    e_out(i) = e
  END DO
END SUBROUTINE cloudsc_war_shape
"""
    ref = _f2py(src, tmp_path / "ref", "cloudsc_war_ref")
    (tmp_path / "sdfg").mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="cloudsc_war_shape", pipeline="hlfir-propagate-shapes").build()

    rng = np.random.default_rng(304)
    n = 32
    # Force the clamp branch: d > f so e = f.  Then post-update f ≈ 0,
    # so any reorder explodes the e/f ratio.
    f_init = np.asfortranarray(0.1 + 0.3 * rng.random(n, dtype=np.float64))
    d = np.asfortranarray(f_init + 0.5)  # always > f
    z = np.asfortranarray(rng.random(n, dtype=np.float64) * 0.5)
    cv_init = np.asfortranarray(0.5 + 0.4 * rng.random(n, dtype=np.float64))
    g = 1e-5

    f_ref = np.array(f_init, order="F")
    cv_ref = np.array(cv_init, order="F")
    e_ref = np.zeros(n, dtype=np.float64, order="F")
    ref.cloudsc_war_shape(d, f_ref, z, g, cv_ref, e_ref)

    f_sdfg = np.array(f_init, order="F")
    cv_sdfg = np.array(cv_init, order="F")
    e_sdfg = np.zeros(n, dtype=np.float64, order="F")
    kw = dict(d=d, f=f_sdfg, z=z, g=g, cv=cv_sdfg, e_out=e_sdfg)
    kw.update(_sdfg_args(sdfg, dict(n=n)))
    sdfg(**kw)

    np.testing.assert_array_equal(cv_sdfg,
                                  cv_ref,
                                  err_msg="cloudsc-WAR-shape: cv diverges -- bridge moved `f -= e` "
                                  "before the cv update so the cv tasklet reads post-update f")
    np.testing.assert_array_equal(f_sdfg, f_ref, err_msg="f final mismatch")
    np.testing.assert_array_equal(e_sdfg, e_ref, err_msg="e mismatch")


# --------------------------------------------------------------------
# DEFAULT_PIPELINE variants: same patterns through the FULL pipeline
# (inline-all + flatten-structs + lift-cf-to-scf + sccp + canonicalize
# + cse + ...).  The cloudsc divergence only manifests under
# DEFAULT_PIPELINE, so these are the suite that should expose the bug.
# --------------------------------------------------------------------
def test_war_dependency_default_pipeline(tmp_path):
    """Same WAR pattern but built with DEFAULT_PIPELINE.  If this fails
    while the minimal-pipeline WAR test passes, one of the extra passes
    is reordering the tasklets."""
    src = """
SUBROUTINE war_default(n, x, y)
  IMPLICIT NONE
  INTEGER(KIND=4), VALUE :: n
  REAL(KIND=8), INTENT(INOUT) :: x(n)
  REAL(KIND=8), INTENT(INOUT) :: y(n)
  INTEGER(KIND=4) :: i
  DO i = 1, n
    y(i) = x(i) * 2.0_8
    x(i) = 99.0_8
  END DO
END SUBROUTINE war_default
"""
    ref = _f2py(src, tmp_path / "ref", "war_default_ref")
    (tmp_path / "sdfg").mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="war_default").build()

    rng = np.random.default_rng(305)
    n = 32
    x_init = np.asfortranarray(rng.standard_normal(n).astype(np.float64))

    x_ref = np.array(x_init, order="F")
    y_ref = np.zeros(n, dtype=np.float64, order="F")
    ref.war_default(x_ref, y_ref)

    x_sdfg = np.array(x_init, order="F")
    y_sdfg = np.zeros(n, dtype=np.float64, order="F")
    kw = dict(x=x_sdfg, y=y_sdfg)
    kw.update(_sdfg_args(sdfg, dict(n=n)))
    sdfg(**kw)

    np.testing.assert_array_equal(x_sdfg, x_ref, err_msg="WAR (DEFAULT_PIPELINE): x final mismatch")
    np.testing.assert_array_equal(y_sdfg,
                                  y_ref,
                                  err_msg="WAR (DEFAULT_PIPELINE): y mismatch -- "
                                  "DEFAULT_PIPELINE pass reordered the write before the read")


def test_cloudsc_shape_war_default_pipeline(tmp_path):
    """The cloudsc-shape WAR pattern under DEFAULT_PIPELINE.  Should
    fail and reproduce the bottom_upper / cloudsc_full bug at strict
    tolerance.  Inputs chosen so MIN(d, f) returns f -> e/f_new explodes
    in the bug case."""
    src = """
SUBROUTINE cw_default(n, d, f, z, g, cv, e_out)
  IMPLICIT NONE
  INTEGER(KIND=4), VALUE :: n
  REAL(KIND=8), INTENT(IN)    :: d(n)
  REAL(KIND=8), INTENT(INOUT) :: f(n)
  REAL(KIND=8), INTENT(IN)    :: z(n)
  REAL(KIND=8), VALUE         :: g
  REAL(KIND=8), INTENT(INOUT) :: cv(n)
  REAL(KIND=8), INTENT(INOUT) :: e_out(n)
  INTEGER(KIND=4) :: i
  REAL(KIND=8)    :: e
  DO i = 1, n
    e = MIN(d(i), f(i))
    cv(i) = MAX(g, cv(i) - MAX(0.0_8, (cv(i) - z(i)) * e / f(i)))
    f(i)  = f(i) - e
    e_out(i) = e
  END DO
END SUBROUTINE cw_default
"""
    ref = _f2py(src, tmp_path / "ref", "cw_default_ref")
    (tmp_path / "sdfg").mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="cw_default").build()

    rng = np.random.default_rng(306)
    n = 32
    f_init = np.asfortranarray(0.1 + 0.3 * rng.random(n, dtype=np.float64))
    d = np.asfortranarray(f_init + 0.5)
    z = np.asfortranarray(rng.random(n, dtype=np.float64) * 0.5)
    cv_init = np.asfortranarray(0.5 + 0.4 * rng.random(n, dtype=np.float64))
    g = 1e-5

    f_ref = np.array(f_init, order="F")
    cv_ref = np.array(cv_init, order="F")
    e_ref = np.zeros(n, dtype=np.float64, order="F")
    ref.cw_default(d, f_ref, z, g, cv_ref, e_ref)

    f_sdfg = np.array(f_init, order="F")
    cv_sdfg = np.array(cv_init, order="F")
    e_sdfg = np.zeros(n, dtype=np.float64, order="F")
    kw = dict(d=d, f=f_sdfg, z=z, g=g, cv=cv_sdfg, e_out=e_sdfg)
    kw.update(_sdfg_args(sdfg, dict(n=n)))
    sdfg(**kw)

    np.testing.assert_array_equal(cv_sdfg,
                                  cv_ref,
                                  err_msg="cloudsc-WAR (DEFAULT_PIPELINE): cv diverges -- the cv "
                                  "tasklet reads post-update f because one of the DEFAULT_PIPELINE "
                                  "passes reordered the write-before-read.  Same bug as cloudsc_full.")
    np.testing.assert_array_equal(f_sdfg, f_ref, err_msg="f final mismatch")
    np.testing.assert_array_equal(e_sdfg, e_ref, err_msg="e mismatch")


@pytest.mark.xfail(
    strict=False,
    reason="Closest reproducer of the cloudsc Section 4.5 bug: multiple "
    "tasklets inside nested IF (outer IEVAPRAIN-style + inner LLO1-style), "
    "DEFAULT_PIPELINE, multi-JK loop with loop-carried cv state.  This "
    "structure may be what triggers the WAR-ordering violation that the "
    "simpler patterns above don't catch.",
)
def test_cloudsc_full_shape_nested_if(tmp_path):
    """Mirrors cloudsc.F90 4.5 structure more closely:
      outer IF (mode == 2) THEN  ! IEVAPRAIN-like
        DO JK = 1, KLEV
          ...compute condition...
          DO i = 1, n
            inner LLO1-like predicate
            IF (llo1) THEN
              e = MIN(d(i,jk), f(i))           ! reads f
              cv(i) = MAX(g, cv(i) - MAX(0.0_8, (cv(i)-z(i))*e/f(i)))  ! reads f
              f(i) = f(i) - e                  ! writes f (must be LAST)
            END IF
          END DO
        END DO
      END IF
    """
    src = """
SUBROUTINE cw_nested(n, klev, mode, d, f, z, g, cv)
  IMPLICIT NONE
  INTEGER(KIND=4), VALUE :: n, klev, mode
  REAL(KIND=8), INTENT(IN)    :: d(n, klev)
  REAL(KIND=8), INTENT(INOUT) :: f(n)
  REAL(KIND=8), INTENT(IN)    :: z(n)
  REAL(KIND=8), VALUE         :: g
  REAL(KIND=8), INTENT(INOUT) :: cv(n)
  INTEGER(KIND=4) :: i, jk
  REAL(KIND=8)    :: e, eps
  LOGICAL :: llo1
  eps = 1.0E-12_8
  IF (mode == 2) THEN
    DO jk = 1, klev
      DO i = 1, n
        llo1 = (cv(i) > eps) .AND. (f(i) > eps)
        IF (llo1) THEN
          e = MIN(d(i, jk), f(i))
          cv(i) = MAX(g, cv(i) - MAX(0.0_8, (cv(i) - z(i)) * e / f(i)))
          f(i) = f(i) - e
        END IF
      END DO
    END DO
  END IF
END SUBROUTINE cw_nested
"""
    ref = _f2py(src, tmp_path / "ref", "cw_nested_ref")
    (tmp_path / "sdfg").mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="cw_nested").build()

    rng = np.random.default_rng(307)
    n, klev = 8, 20
    mode = 2  # active branch (like IEVAPRAIN=2)
    f_init = np.asfortranarray(0.1 + 0.3 * rng.random(n, dtype=np.float64))
    d = np.asfortranarray(f_init[:, None] + 0.5 * rng.random((n, klev), dtype=np.float64))
    z = np.asfortranarray(rng.random(n, dtype=np.float64) * 0.5)
    cv_init = np.asfortranarray(0.5 + 0.4 * rng.random(n, dtype=np.float64))
    g = 1e-5

    f_ref = np.array(f_init, order="F")
    cv_ref = np.array(cv_init, order="F")
    ref.cw_nested(mode, d, f_ref, z, g, cv_ref)

    f_sdfg = np.array(f_init, order="F")
    cv_sdfg = np.array(cv_init, order="F")
    kw = dict(d=d, f=f_sdfg, z=z, g=g, cv=cv_sdfg)
    kw.update(_sdfg_args(sdfg, dict(n=n, klev=klev, mode=mode)))
    sdfg(**kw)

    np.testing.assert_allclose(cv_sdfg,
                               cv_ref,
                               rtol=1e-12,
                               atol=1e-12,
                               err_msg="cloudsc-nested-IF (DEFAULT_PIPELINE): cv diverges -- "
                               "the WAR ordering bug from cloudsc Section 4.5 reproduced here.")
    np.testing.assert_allclose(f_sdfg, f_ref, rtol=1e-12, atol=1e-12, err_msg="f final mismatch")
