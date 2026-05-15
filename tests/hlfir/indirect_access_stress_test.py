"""Stress tests for indirect-array access patterns.

Targets the bridge's indirect-symbol synthesis path
(``access.py::collect_indirect`` + ``__sym_<arr>_<n>`` minting).  Each
test:

* writes a small Fortran kernel with a known-correct semantics,
* compiles a gfortran/f2py reference and verifies it produces the
  expected numeric output (so the test's *intent* is validated
  independently of the bridge),
* builds the SDFG and asserts the bridge output matches the f2py
  reference bitwise.

Patterns covered (gather + scatter):

* 1-level indirect read:   ``B(i) = A(idx(i))``
* 1-level indirect write:  ``C(idx(i)) = A(i)``
* 2-level nested gather:   ``B(i) = A(idx1(idx2(i)))``
* 3-level nested gather:   ``B(i) = A(idx1(idx2(idx3(i))))``
* 3-level nested scatter:  ``C(idx1(idx2(idx3(i)))) = A(i)``
* High-dim indirect read (up to 6-D): different combinations of
  direct + indirect indices on each axis.
* High-dim indirect scatter (up to 6-D): scatter into a 6-D buffer
  with one indirect axis.

A pure ``builds_and_runs`` test runs first per pattern -- if the
SDFG can't even be compiled, the deeper numeric assertion is moot.
"""

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, f2py_compile, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

# ---------------------------------------------------------------------------
# 1-level gather: B(i) = A(idx(i))
# ---------------------------------------------------------------------------

_1L_GATHER_SRC = """
SUBROUTINE k_1l_gather(a, idx, b, n)
  IMPLICIT NONE
  INTEGER(KIND=4), VALUE :: n
  REAL(KIND=8), INTENT(IN)  :: a(n)
  INTEGER(KIND=4), INTENT(IN) :: idx(n)
  REAL(KIND=8), INTENT(OUT) :: b(n)
  INTEGER :: i
  DO i = 1, n
    b(i) = a(idx(i))
  END DO
END SUBROUTINE
"""


def test_indirect_1l_gather(tmp_path: Path):
    n = 8
    rng = np.random.default_rng(11)
    a = np.asfortranarray(rng.standard_normal(n, dtype=np.float64))
    idx = np.asfortranarray(rng.integers(1, n + 1, size=n, dtype=np.int32))

    # Reference (f2py turns ``intent(out) :: b`` into a return value).
    mod = f2py_compile(_1L_GATHER_SRC, tmp_path / "ref", "k_1l_gather_ref")
    b_ref = mod.k_1l_gather(a, idx)
    b_expected = a[idx - 1]
    np.testing.assert_array_equal(b_ref, b_expected)

    # SDFG
    sdfg = build_sdfg(_1L_GATHER_SRC, tmp_path, name="k_1l_gather").build()
    b_sdfg = np.zeros(n, dtype=np.float64, order="F")
    sdfg(a=a, idx=idx, b=b_sdfg, n=n)
    np.testing.assert_array_equal(b_sdfg, b_ref)


# ---------------------------------------------------------------------------
# 1-level scatter: C(idx(i)) = A(i)
# ---------------------------------------------------------------------------

_1L_SCATTER_SRC = """
SUBROUTINE k_1l_scatter(a, idx, c, n)
  IMPLICIT NONE
  INTEGER(KIND=4), VALUE :: n
  REAL(KIND=8), INTENT(IN)  :: a(n)
  INTEGER(KIND=4), INTENT(IN) :: idx(n)
  REAL(KIND=8), INTENT(OUT) :: c(n)
  INTEGER :: i
  c = 0.0D0
  DO i = 1, n
    c(idx(i)) = a(i)
  END DO
END SUBROUTINE
"""


def test_indirect_1l_scatter(tmp_path: Path):
    n = 8
    rng = np.random.default_rng(12)
    a = np.asfortranarray(rng.standard_normal(n, dtype=np.float64))
    # Permutation so each idx(i) is unique and the scatter is bijective.
    idx = np.asfortranarray((rng.permutation(n) + 1).astype(np.int32))

    mod = f2py_compile(_1L_SCATTER_SRC, tmp_path / "ref", "k_1l_scatter_ref")
    c_ref = mod.k_1l_scatter(a, idx)
    c_expected = np.zeros(n, dtype=np.float64)
    c_expected[idx - 1] = a
    np.testing.assert_array_equal(c_ref, c_expected)

    sdfg = build_sdfg(_1L_SCATTER_SRC, tmp_path, name="k_1l_scatter").build()
    c_sdfg = np.zeros(n, dtype=np.float64, order="F")
    sdfg(a=a, idx=idx, c=c_sdfg, n=n)
    np.testing.assert_array_equal(c_sdfg, c_ref)


# ---------------------------------------------------------------------------
# 2-level nested gather: B(i) = A(idx1(idx2(i)))
# ---------------------------------------------------------------------------

_2L_GATHER_SRC = """
SUBROUTINE k_2l_gather(a, idx1, idx2, b, n)
  IMPLICIT NONE
  INTEGER(KIND=4), VALUE :: n
  REAL(KIND=8), INTENT(IN)  :: a(n)
  INTEGER(KIND=4), INTENT(IN) :: idx1(n), idx2(n)
  REAL(KIND=8), INTENT(OUT) :: b(n)
  INTEGER :: i
  DO i = 1, n
    b(i) = a(idx1(idx2(i)))
  END DO
END SUBROUTINE
"""


def test_indirect_2l_nested_gather(tmp_path: Path):
    n = 8
    rng = np.random.default_rng(13)
    a = np.asfortranarray(rng.standard_normal(n, dtype=np.float64))
    idx1 = np.asfortranarray(rng.integers(1, n + 1, size=n, dtype=np.int32))
    idx2 = np.asfortranarray(rng.integers(1, n + 1, size=n, dtype=np.int32))

    mod = f2py_compile(_2L_GATHER_SRC, tmp_path / "ref", "k_2l_gather_ref")
    b_ref = mod.k_2l_gather(a, idx1, idx2)
    b_expected = a[idx1[idx2 - 1] - 1]
    np.testing.assert_array_equal(b_ref, b_expected)

    sdfg = build_sdfg(_2L_GATHER_SRC, tmp_path, name="k_2l_gather").build()
    b_sdfg = np.zeros(n, dtype=np.float64, order="F")
    sdfg(a=a, idx1=idx1, idx2=idx2, b=b_sdfg, n=n)
    np.testing.assert_array_equal(b_sdfg, b_ref)


# ---------------------------------------------------------------------------
# 3-level nested gather: B(i) = A(idx1(idx2(idx3(i))))
# ---------------------------------------------------------------------------

_3L_GATHER_SRC = """
SUBROUTINE k_3l_gather(a, idx1, idx2, idx3, b, n)
  IMPLICIT NONE
  INTEGER(KIND=4), VALUE :: n
  REAL(KIND=8), INTENT(IN)  :: a(n)
  INTEGER(KIND=4), INTENT(IN) :: idx1(n), idx2(n), idx3(n)
  REAL(KIND=8), INTENT(OUT) :: b(n)
  INTEGER :: i
  DO i = 1, n
    b(i) = a(idx1(idx2(idx3(i))))
  END DO
END SUBROUTINE
"""


def test_indirect_3l_nested_gather(tmp_path: Path):
    n = 8
    rng = np.random.default_rng(14)
    a = np.asfortranarray(rng.standard_normal(n, dtype=np.float64))
    idx1 = np.asfortranarray(rng.integers(1, n + 1, size=n, dtype=np.int32))
    idx2 = np.asfortranarray(rng.integers(1, n + 1, size=n, dtype=np.int32))
    idx3 = np.asfortranarray(rng.integers(1, n + 1, size=n, dtype=np.int32))

    mod = f2py_compile(_3L_GATHER_SRC, tmp_path / "ref", "k_3l_gather_ref")
    b_ref = mod.k_3l_gather(a, idx1, idx2, idx3)
    b_expected = a[idx1[idx2[idx3 - 1] - 1] - 1]
    np.testing.assert_array_equal(b_ref, b_expected)

    sdfg = build_sdfg(_3L_GATHER_SRC, tmp_path, name="k_3l_gather").build()
    b_sdfg = np.zeros(n, dtype=np.float64, order="F")
    sdfg(a=a, idx1=idx1, idx2=idx2, idx3=idx3, b=b_sdfg, n=n)
    np.testing.assert_array_equal(b_sdfg, b_ref)


# ---------------------------------------------------------------------------
# 3-level nested scatter: C(idx1(idx2(idx3(i)))) = A(i)
# ---------------------------------------------------------------------------

_3L_SCATTER_SRC = """
SUBROUTINE k_3l_scatter(a, idx1, idx2, idx3, c, n)
  IMPLICIT NONE
  INTEGER(KIND=4), VALUE :: n
  REAL(KIND=8), INTENT(IN)  :: a(n)
  INTEGER(KIND=4), INTENT(IN) :: idx1(n), idx2(n), idx3(n)
  REAL(KIND=8), INTENT(OUT) :: c(n)
  INTEGER :: i
  c = 0.0D0
  DO i = 1, n
    c(idx1(idx2(idx3(i)))) = a(i)
  END DO
END SUBROUTINE
"""


def test_indirect_3l_nested_scatter(tmp_path: Path):
    n = 8
    rng = np.random.default_rng(15)
    a = np.asfortranarray(rng.standard_normal(n, dtype=np.float64))
    # Build chained permutations so the composite map is bijective and
    # every element of A lands in a distinct C slot.  This avoids the
    # "last writer wins" non-determinism of overlapping scatter.
    p1 = (rng.permutation(n) + 1).astype(np.int32)
    p2 = (rng.permutation(n) + 1).astype(np.int32)
    p3 = (rng.permutation(n) + 1).astype(np.int32)
    idx1 = np.asfortranarray(p1)
    idx2 = np.asfortranarray(p2)
    idx3 = np.asfortranarray(p3)

    mod = f2py_compile(_3L_SCATTER_SRC, tmp_path / "ref", "k_3l_scatter_ref")
    c_ref = mod.k_3l_scatter(a, idx1, idx2, idx3)
    # Compute expected by hand: c[idx1[idx2[idx3[i]-1]-1]-1] = a[i].
    c_expected = np.zeros(n, dtype=np.float64)
    for i in range(n):
        j = idx1[idx2[idx3[i] - 1] - 1] - 1
        c_expected[j] = a[i]
    np.testing.assert_array_equal(c_ref, c_expected)

    sdfg = build_sdfg(_3L_SCATTER_SRC, tmp_path, name="k_3l_scatter").build()
    c_sdfg = np.zeros(n, dtype=np.float64, order="F")
    sdfg(a=a, idx1=idx1, idx2=idx2, idx3=idx3, c=c_sdfg, n=n)
    np.testing.assert_array_equal(c_sdfg, c_ref)


# ---------------------------------------------------------------------------
# 3-D indirect gather (one indirect axis per dim).
# ---------------------------------------------------------------------------

_3D_GATHER_SRC = """
SUBROUTINE k_3d_gather(a, ix, iy, iz, b, n)
  IMPLICIT NONE
  INTEGER(KIND=4), VALUE :: n
  REAL(KIND=8), INTENT(IN)  :: a(n, n, n)
  INTEGER(KIND=4), INTENT(IN) :: ix(n), iy(n), iz(n)
  REAL(KIND=8), INTENT(OUT) :: b(n)
  INTEGER :: i
  DO i = 1, n
    b(i) = a(ix(i), iy(i), iz(i))
  END DO
END SUBROUTINE
"""


def test_indirect_3d_gather(tmp_path: Path):
    n = 5
    rng = np.random.default_rng(16)
    a = np.asfortranarray(rng.standard_normal((n, n, n), dtype=np.float64))
    ix = np.asfortranarray(rng.integers(1, n + 1, size=n, dtype=np.int32))
    iy = np.asfortranarray(rng.integers(1, n + 1, size=n, dtype=np.int32))
    iz = np.asfortranarray(rng.integers(1, n + 1, size=n, dtype=np.int32))

    mod = f2py_compile(_3D_GATHER_SRC, tmp_path / "ref", "k_3d_gather_ref")
    b_ref = mod.k_3d_gather(a, ix, iy, iz)
    b_expected = np.array([a[ix[i] - 1, iy[i] - 1, iz[i] - 1] for i in range(n)])
    np.testing.assert_array_equal(b_ref, b_expected)

    sdfg = build_sdfg(_3D_GATHER_SRC, tmp_path, name="k_3d_gather").build()
    b_sdfg = np.zeros(n, dtype=np.float64, order="F")
    sdfg(a=a, ix=ix, iy=iy, iz=iz, b=b_sdfg, n=n)
    np.testing.assert_array_equal(b_sdfg, b_ref)


# ---------------------------------------------------------------------------
# 6-D indirect gather (one direct + five indirect axes).  Stresses high-rank
# memlet subset construction and deep linearisation.
# ---------------------------------------------------------------------------

_6D_GATHER_SRC = """
SUBROUTINE k_6d_gather(a, i1, i2, i3, i4, i5, b, n)
  IMPLICIT NONE
  INTEGER(KIND=4), VALUE :: n
  REAL(KIND=8), INTENT(IN)  :: a(n, n, n, n, n, n)
  INTEGER(KIND=4), INTENT(IN) :: i1(n), i2(n), i3(n), i4(n), i5(n)
  REAL(KIND=8), INTENT(OUT) :: b(n)
  INTEGER :: i
  DO i = 1, n
    b(i) = a(i, i1(i), i2(i), i3(i), i4(i), i5(i))
  END DO
END SUBROUTINE
"""


def test_indirect_6d_gather(tmp_path: Path):
    n = 4
    rng = np.random.default_rng(17)
    a = np.asfortranarray(rng.standard_normal((n, n, n, n, n, n), dtype=np.float64))
    i1 = np.asfortranarray(rng.integers(1, n + 1, size=n, dtype=np.int32))
    i2 = np.asfortranarray(rng.integers(1, n + 1, size=n, dtype=np.int32))
    i3 = np.asfortranarray(rng.integers(1, n + 1, size=n, dtype=np.int32))
    i4 = np.asfortranarray(rng.integers(1, n + 1, size=n, dtype=np.int32))
    i5 = np.asfortranarray(rng.integers(1, n + 1, size=n, dtype=np.int32))

    mod = f2py_compile(_6D_GATHER_SRC, tmp_path / "ref", "k_6d_gather_ref")
    b_ref = mod.k_6d_gather(a, i1, i2, i3, i4, i5)
    b_expected = np.array([a[i, i1[i] - 1, i2[i] - 1, i3[i] - 1, i4[i] - 1, i5[i] - 1] for i in range(n)])
    np.testing.assert_array_equal(b_ref, b_expected)

    sdfg = build_sdfg(_6D_GATHER_SRC, tmp_path, name="k_6d_gather").build()
    b_sdfg = np.zeros(n, dtype=np.float64, order="F")
    sdfg(a=a, i1=i1, i2=i2, i3=i3, i4=i4, i5=i5, b=b_sdfg, n=n)
    np.testing.assert_array_equal(b_sdfg, b_ref)


# ---------------------------------------------------------------------------
# 6-D indirect scatter: C(i1(i), i2(i), i3(i), i4(i), i5(i), i) = A(i).
# ---------------------------------------------------------------------------

_6D_SCATTER_SRC = """
SUBROUTINE k_6d_scatter(a, i1, i2, i3, i4, i5, c, n)
  IMPLICIT NONE
  INTEGER(KIND=4), VALUE :: n
  REAL(KIND=8), INTENT(IN)  :: a(n)
  INTEGER(KIND=4), INTENT(IN) :: i1(n), i2(n), i3(n), i4(n), i5(n)
  REAL(KIND=8), INTENT(OUT) :: c(n, n, n, n, n, n)
  INTEGER :: i
  c = 0.0D0
  DO i = 1, n
    c(i1(i), i2(i), i3(i), i4(i), i5(i), i) = a(i)
  END DO
END SUBROUTINE
"""


def test_indirect_6d_scatter(tmp_path: Path):
    n = 3
    rng = np.random.default_rng(18)
    a = np.asfortranarray(rng.standard_normal(n, dtype=np.float64))
    # Use permutations on each axis so the (i1,i2,i3,i4,i5,i) tuple is
    # unique across i and there are no overlapping writes.
    i1 = np.asfortranarray((rng.permutation(n) + 1).astype(np.int32))
    i2 = np.asfortranarray((rng.permutation(n) + 1).astype(np.int32))
    i3 = np.asfortranarray((rng.permutation(n) + 1).astype(np.int32))
    i4 = np.asfortranarray((rng.permutation(n) + 1).astype(np.int32))
    i5 = np.asfortranarray((rng.permutation(n) + 1).astype(np.int32))

    mod = f2py_compile(_6D_SCATTER_SRC, tmp_path / "ref", "k_6d_scatter_ref")
    c_ref = mod.k_6d_scatter(a, i1, i2, i3, i4, i5)
    c_expected = np.zeros((n, n, n, n, n, n), dtype=np.float64, order="F")
    for i in range(n):
        c_expected[i1[i] - 1, i2[i] - 1, i3[i] - 1, i4[i] - 1, i5[i] - 1, i] = a[i]
    np.testing.assert_array_equal(c_ref, c_expected)

    sdfg = build_sdfg(_6D_SCATTER_SRC, tmp_path, name="k_6d_scatter").build()
    c_sdfg = np.zeros((n, n, n, n, n, n), dtype=np.float64, order="F")
    sdfg(a=a, i1=i1, i2=i2, i3=i3, i4=i4, i5=i5, c=c_sdfg, n=n)
    np.testing.assert_array_equal(c_sdfg, c_ref)


# ---------------------------------------------------------------------------
# ICON loopnest_4 minimal reproducer: mixed direct/indirect axes inside a
# nested IF inside a DO loop -- the bridge previously left the indirect
# array names bare in the memlet subset because ``collect_indirect`` was
# only called from ``emit_loop``'s batch path, not from individual
# ``emit_assign`` calls inside an IF body.  This reproducer keeps the
# same shape but trims the kernel to the bare minimum so any future
# regression points at the right code path immediately.
# ---------------------------------------------------------------------------

_ICON4_REPRO_SRC = """
SUBROUTINE icon4_repro(vn, iqidx, iqblk, mask, out, nproma, nlev, nblks, jb)
  IMPLICIT NONE
  INTEGER(KIND=4), VALUE :: nproma, nlev, nblks, jb
  REAL(KIND=8), INTENT(IN)  :: vn(nproma, nlev, nblks)
  INTEGER(KIND=4), INTENT(IN) :: iqidx(nproma, nblks), iqblk(nproma, nblks)
  INTEGER(KIND=4), INTENT(IN) :: mask(nlev)
  REAL(KIND=8), INTENT(OUT) :: out(nproma, nlev)
  INTEGER :: jk, je
  out = 0.0D0
  DO jk = 1, nlev
    IF (mask(jk) /= 0) THEN
      DO je = 1, nproma
        out(je, jk) = vn(iqidx(je, jb), jk, iqblk(je, jb))
      END DO
    END IF
  END DO
END SUBROUTINE
"""


def test_indirect_icon4_minimal_repro(tmp_path: Path):
    nproma, nlev, nblks = 4, 5, 3
    rng = np.random.default_rng(42)
    vn = np.asfortranarray(rng.standard_normal((nproma, nlev, nblks)))
    iqidx = np.asfortranarray(rng.integers(1, nproma + 1, size=(nproma, nblks), dtype=np.int32))
    iqblk = np.asfortranarray(rng.integers(1, nblks + 1, size=(nproma, nblks), dtype=np.int32))
    mask = np.asfortranarray(rng.integers(0, 2, size=nlev, dtype=np.int32))
    jb = 2

    mod = f2py_compile(_ICON4_REPRO_SRC, tmp_path / "ref", "icon4_repro_ref")
    out_ref = mod.icon4_repro(vn, iqidx, iqblk, mask, nproma=nproma, nlev=nlev, nblks=nblks, jb=jb)

    sdfg = build_sdfg(_ICON4_REPRO_SRC, tmp_path, name="icon4_repro").build()
    out_sdfg = np.zeros((nproma, nlev), dtype=np.float64, order="F")
    sdfg(vn=vn,
         iqidx=iqidx,
         iqblk=iqblk,
         mask=mask,
         out=out_sdfg,
         nproma=nproma,
         nlev=nlev,
         nblks=nblks,
         jb=jb,
         jk=0,
         je=0)
    np.testing.assert_array_equal(out_sdfg, out_ref)
