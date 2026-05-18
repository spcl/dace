"""Bottom-LOWER reproducer (companion to bottom-half / bottom-upper).

Drops the bottom-upper physics (sedimentation + autoconversion + melting
+ freezing + evaporation, lines 2620-3355 of cloudsc.F90) and keeps the
solvers + flux + tendency block (lines 3356-3710).

ZSOLQA / ZSOLQB / ZCOVPTOT remain at zero (from their reset above the
deleted region), and the solvers + flux/tendency chain runs on those
zeros.

Expectation: this PASSES at rtol=atol=1e-15.  The bisection chain says:
  * test_cloudsc_top_half          PASS  -> bug in bottom half
  * test_cloudsc_bottom_half       FAIL  -> bug confirmed in bottom half
  * test_cloudsc_bottom_upper      FAIL  -> bug in bottom-upper physics
  * test_cloudsc_bottom_upper_a    PASS  -> bug specifically in 4.5 EVAP
  * test_cloudsc_bottom_lower      PASS  -> bug NOT in solvers/flux/tendency

So this loopnest's role is to confirm the negative: the solvers + flux +
tendency portion of cloudsc lowers bit-correctly under the bridge.
"""
from pathlib import Path
import numpy as np
import pytest
from _util import f2py_compile, have_flang
from cloudsc_full._registries import CLOUDSC_F90FLAGS, program_outputs
from cloudsc_full._harness import run_cloudsc

_HERE = Path(__file__).resolve().parent
pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


@pytest.fixture(scope="module")
def _f2py_lo(tmp_path_factory):
    src = (_HERE / "cloudsc_bottom_lower.F90").read_text()
    ref_dir = tmp_path_factory.mktemp("cloudsc_bottom_lower_ref")
    return f2py_compile(
        src,
        ref_dir,
        "cloudsc_bottom_lower_ref",
        # ``-ffree-line-length-none`` is the sole intentionally
        # gfortran-only flag: it is a non-semantic parser necessity for
        # the long-line cloudsc source; LLVM-flang has no line limit and
        # needs no equivalent.  The FP set is the flang-portable core.
        extra_f90flags=CLOUDSC_F90FLAGS,
        only=("cloudscouter", ))


def test_cloudsc_bottom_lower_numerical(tmp_path, _f2py_lo, _strict_fp_cpu_args):
    src = (_HERE / "cloudsc_bottom_lower.F90").read_text()
    outputs_sdfg, outputs_ref = run_cloudsc(src, "cloudsc_bottom_lower", _f2py_lo, tmp_path / "sdfg")

    # Expectation: PASS at strict tolerance.  Compare every program_output
    # (the solvers + flux + tendency chain writes them all).
    for name in program_outputs:
        np.testing.assert_allclose(outputs_sdfg[name.lower()],
                                   outputs_ref[name.lower()],
                                   rtol=1e-15,
                                   atol=1e-15,
                                   err_msg=f"PCOVPTOT mismatch in bottom-lower (solvers/flux/tendency only): {name}")
