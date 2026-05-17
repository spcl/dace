"""Tests for ``IntegerizePowerExponents``.

The pass retypes integer-valued float ``**`` exponents (``base**2.0``)
in tasklets to ``int`` so C++ codegen takes the deterministic
repeated-multiply ``dace::math::ipow`` path instead of libm
``dace::math::pow`` -- bit-matching a Fortran reference.

Two levels of coverage:

* a direct pass unit test on a hand-built SDFG tasklet (independent of
  the HLFIR bridge), and
* an end-to-end Fortran-vs-gfortran numerical test through the bridge,
  using an *array-reference* base (``a(jl)**2.0``).  The source-level
  ``rewrite_integer_powers`` preprocessor deliberately skips a base
  containing an array/function reference, so the float-exponent ``**``
  survives into a tasklet -- exactly the case this SDFG pass exists to
  fix.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import dace

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_pass_retypes_integer_valued_float_exponents():
    """The pass turns ``a ** 2.0`` / ``a ** -3.0`` into ``int``
    exponents and leaves fractional / already-int powers alone."""
    from dace.frontend.hlfir.integer_power_exponents import IntegerizePowerExponents

    sdfg = dace.SDFG("ipow_pass")
    sdfg.add_array("A", [1], dace.float64)
    sdfg.add_array("B", [1], dace.float64)
    state = sdfg.add_state()
    r = state.add_read("A")
    w = state.add_write("B")
    t = state.add_tasklet("p", {"a"}, {"b"}, "b = (a ** 2.0) + (a ** -3.0) + (a ** 0.5) + (a ** 2)")
    state.add_edge(r, None, t, "a", dace.Memlet("A[0]"))
    state.add_edge(t, "b", w, None, dace.Memlet("B[0]"))

    n = IntegerizePowerExponents().apply_pass(sdfg, {})

    assert n == 2  # the 2.0 and the -3.0; 0.5 and 2 untouched
    code = t.code.as_string
    assert "a ** 2.0" not in code and "a ** 2" in code
    assert "a ** -3" in code
    assert "a ** 0.5" in code  # genuine fractional left alone


_ARR_POW = """
subroutine kern(n, a, y)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in)  :: a(n)
  real(8), intent(out) :: y(n)
  integer :: jl
  do jl = 1, n
    y(jl) = a(jl)**2.0 + 3.0d0 * a(jl)**3.0
  end do
end subroutine kern
"""


def _f2py(src_text: str, out_dir: Path, mod: str):
    if shutil.which("gfortran") is None:
        pytest.skip("gfortran not available")
    if shutil.which("meson") is None:
        pytest.skip("meson not available (f2py backend on Python>=3.12)")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{mod}.f90").write_text(src_text)
    subprocess.check_call(
        [
            sys.executable, "-m", "numpy.f2py", "-c", f"{mod}.f90", "-m", mod, "--quiet",
            "--f90flags=-O0 -fno-fast-math -ffp-contract=off"
        ],
        cwd=out_dir,
    )
    if str(out_dir) not in sys.path:
        sys.path.insert(0, str(out_dir))
    __import__(mod)
    return sys.modules[mod]


def test_array_base_float_power_matches_gfortran(tmp_path: Path):
    """``a(jl)**2.0`` (array-ref base, skipped by the source
    preprocessor) goes through the bridge; the SDFG pass retypes the
    exponent so codegen emits ``ipow``, and the result is bit-identical
    to gfortran."""
    ref = _f2py(_ARR_POW, tmp_path / "ref", "ipow_ref")
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_ARR_POW, sdfg_dir, name="kern", entry="_QPkern").build()

    # The exponents must have been integerised: no libm float ``pow``
    # of an integer-valued exponent should remain in any tasklet.
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.Tasklet):
            assert "** 2.0" not in node.code.as_string
            assert "** 3.0" not in node.code.as_string

    n = 7
    rng = np.random.default_rng(0)
    a = np.asfortranarray(rng.standard_normal(n))
    yr = ref.kern(a)

    ys = np.zeros(n, order="F")
    from dace.data import Scalar
    al = sdfg.arglist()
    nkw = {"n": n if isinstance(al.get("n"), Scalar) else np.array([n], np.int32)}
    sdfg(a=a.copy(order="F"), y=ys, **nkw)

    np.testing.assert_allclose(ys, yr, rtol=1e-12, atol=1e-12)
