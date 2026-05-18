"""Bisection step: bottom-UPPER reproducer.

After test_cloudsc_bottom_half.py FAILS with the same 26/548 PCOVPTOT
divergence as test_cloudsc_full, we split the bottom half at the
boundary between physics (sections 4.2-4.5) and solvers/flux/tendency
(sections 5.x, 6.x).

This reproducer keeps the bottom-half physics (sedimentation +
autoconversion + melting + freezing + evaporation, lines 2620-3355
of original cloudsc.F90) and drops the solvers/flux/tendency
(lines 3356-3710), replacing them with a direct
``PCOVPTOT(JL,JK) = ZCOVPTOT(JL)`` writeback.

If this PASSES, the cloudsc bug is in the bottom-lower (solvers +
flux + tendency, lines 3356-3710).  If it FAILS with the same
26/548 PCOVPTOT mismatch, the bug is in the bottom-upper physics.
"""

from pathlib import Path

import numpy as np
import pytest

from _util import f2py_compile, have_flang
from cloudsc_full._registries import (
    CLOUDSC_F90FLAGS, )
from cloudsc_full._harness import run_cloudsc

_HERE = Path(__file__).resolve().parent
pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


@pytest.fixture(scope="module")
def _f2py_bottom_upper(tmp_path_factory):
    src = (_HERE / "cloudsc_bottom_upper.F90").read_text()
    ref_dir = tmp_path_factory.mktemp("cloudsc_bottom_upper_ref")
    return f2py_compile(
        src,
        ref_dir,
        "cloudsc_bottom_upper_ref",
        # ``-ffree-line-length-none`` is the sole intentionally
        # gfortran-only flag: it is a non-semantic parser necessity for
        # the long-line cloudsc source; LLVM-flang has no line limit and
        # needs no equivalent.  The FP set is the flang-portable core.
        extra_f90flags=CLOUDSC_F90FLAGS,
        only=("cloudscouter", ),
    )


# Physical (NaN-free) inputs: the bridge is bit-identical to gfortran here.
def test_cloudsc_bottom_upper_numerical(tmp_path, _f2py_bottom_upper, _strict_fp_cpu_args):
    src = (_HERE / "cloudsc_bottom_upper.F90").read_text()
    outputs_sdfg, outputs_ref = run_cloudsc(src, "cloudsc_bottom_upper", _f2py_bottom_upper, tmp_path / "sdfg")

    # Only PCOVPTOT is meaningful here (other outputs zero -- their writes
    # got dropped along with the solver/flux section).
    np.testing.assert_allclose(
        outputs_sdfg["pcovptot"],
        outputs_ref["pcovptot"],
        rtol=1e-15,
        atol=1e-15,
        err_msg="PCOVPTOT mismatch in bottom-upper (sedimentation/physics) reproducer",
    )
