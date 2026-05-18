"""Sub-bisection: bottom-UPPER-A (drops 4.5 EVAPORATION section).

After test_cloudsc_bottom_upper.py FAILS with 51/548 PCOVPTOT mismatch,
this drops the 4.5 EVAPORATION OF RAIN/SNOW section (~330 lines of
the bottom-upper) and keeps only the Sedimentation (4.2) +
Autoconversion (4.3a/b) + Melting (4.4a) + Freezing (4.4b/c) sections.

If this passes, the bug is in 4.5 EVAPORATION.
If it fails, the bug is in 4.2-4.4 (Sedimentation/Autoconv/Melt/Freeze).
"""
from pathlib import Path
import numpy as np
import pytest
from _util import f2py_compile, have_flang
from cloudsc_full._registries import CLOUDSC_F90FLAGS
from cloudsc_full._harness import run_cloudsc

_HERE = Path(__file__).resolve().parent
pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


@pytest.fixture(scope="module")
def _f2py_a(tmp_path_factory):
    src = (_HERE / "cloudsc_bottom_upper_a.F90").read_text()
    ref_dir = tmp_path_factory.mktemp("cloudsc_bottom_upper_a_ref")
    return f2py_compile(
        src,
        ref_dir,
        "cloudsc_bottom_upper_a_ref",
        # ``-ffree-line-length-none`` is the sole intentionally
        # gfortran-only flag: it is a non-semantic parser necessity for
        # the long-line cloudsc source; LLVM-flang has no line limit and
        # needs no equivalent.  The FP set is the flang-portable core.
        extra_f90flags=CLOUDSC_F90FLAGS,
        only=("cloudscouter", ))


# Physical (NaN-free) inputs: the bridge matches gfortran to tight tolerance here.
def test_cloudsc_bottom_upper_a_numerical(tmp_path, _f2py_a, _strict_fp_cpu_args):
    src = (_HERE / "cloudsc_bottom_upper_a.F90").read_text()
    outputs_sdfg, outputs_ref = run_cloudsc(src, "cloudsc_bottom_upper_a", _f2py_a, tmp_path / "sdfg")

    np.testing.assert_allclose(outputs_sdfg["pcovptot"],
                               outputs_ref["pcovptot"],
                               rtol=1e-15,
                               atol=1e-15,
                               err_msg="PCOVPTOT mismatch in bottom-upper-A (Sed/Autoconv/Melt/Freeze)")
