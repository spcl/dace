"""Bottom-half-CLOUDSC reproducer for the cloudsc_full xfail.

Companion to ``test_cloudsc_top_half.py``.

The top-half reproducer (lines 1766-2617 of CLOUDSC's body, source/
sink accumulation into ZSOLQA/ZSOLQB) passes at rtol=atol=1e-15 -- the
bridge lowers it bit-correctly.  Therefore the cloudsc_full divergence
has to come from the bottom half (sedimentation + LU solver + flux/
tendency updates, lines 2620-3700).

This reproducer is ``cloudsc.F90`` with the source/sink accumulation
block deleted (lines 1879-2617 of the original).  ZSOLQA / ZSOLQB stay
at the zero values they were reset to just above the deletion point;
the LU solver then factors a near-identity matrix.  The
sedimentation/LU/flux chain still runs over all 122 JK iterations and
produces meaningful PCOVPTOT + flux outputs.

The basic init (ZQXFG, ZTP1, ZRHO, ZA, saturation values) is preserved
above line 1879, so the bottom half has all the local state it needs.

Compares SDFG vs f2py on the same source with identical seeded inputs.
If they disagree at strict tolerance, the bug is in the bottom-half
lowering itself.  If they agree, the cloudsc_full bug only manifests
when source/sink and sedimentation/LU/flux are stitched together --
i.e. it's a cross-talk-only bug.
"""

from pathlib import Path

import numpy as np
import pytest

from _util import f2py_compile, have_flang
from cloudsc_full._registries import (
    CLOUDSC_F90FLAGS,
    program_outputs,
)
from cloudsc_full._harness import run_cloudsc

_HERE = Path(__file__).resolve().parent
pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


@pytest.fixture(scope="module")
def _f2py_bottom_half(tmp_path_factory):
    src = (_HERE / "cloudsc_bottom_half.F90").read_text()
    ref_dir = tmp_path_factory.mktemp("cloudsc_bottom_half_ref")
    return f2py_compile(
        src,
        ref_dir,
        "cloudsc_bottom_half_ref",
        # ``-ffree-line-length-none`` is the sole intentionally
        # gfortran-only flag: it is a non-semantic parser necessity for
        # the long-line cloudsc source; LLVM-flang has no line limit and
        # needs no equivalent.  The FP set is the flang-portable core.
        extra_f90flags=CLOUDSC_F90FLAGS,
        only=("cloudscouter", ),
    )


# Physical (NaN-free) inputs: the bridge matches gfortran to tight tolerance here.
def test_cloudsc_bottom_half_numerical(tmp_path, _f2py_bottom_half, _strict_fp_cpu_args):
    src = (_HERE / "cloudsc_bottom_half.F90").read_text()
    outputs_sdfg, outputs_ref = run_cloudsc(src, "cloudsc_bottom_half", _f2py_bottom_half, tmp_path / "sdfg")

    for name in program_outputs:
        np.testing.assert_allclose(
            outputs_sdfg[name.lower()],
            outputs_ref[name.lower()],
            rtol=1e-15,
            atol=1e-15,
            err_msg=f"mismatch on output {name}",
        )
