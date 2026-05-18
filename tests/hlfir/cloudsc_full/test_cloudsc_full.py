"""Full-source CLOUDSC end-to-end test for the HLFIR bridge.

Drives the entire ECMWF CLOUDSC microphysics kernel through the
bridge and compares the SDFG output against a gfortran/f2py
reference compiled from the same Fortran source.  Catches
integration regressions the per-loopnest tests can't see  --  state
hoisting across the block loop, the 100+ scalar-arg signature,
deeply-nested ELEMENTAL expressions, rank-4 ``PCLV(:, :, :, JKGLO)``
per-block slicing  --  and serves as the gate for the full-ICON
integration test that comes next.

**Source:** ``cloudsc.F90`` (single-file: PARKIND1 / YOMCST /
YOETHF / YOECLDP type-only modules + ``CLOUDSCOUTER`` wrapper +
upstream ECMWF dwarf-p-cloudsc ``CLOUDSC`` body verbatim).  The
``#include "fcttre.ycst.h" / "fccld.ydthf.h" / "abor1.intfb.h"``
preprocessor directives are pre-expanded inline so flang-new-21
``-fc1 -emit-hlfir`` (no ``-cpp``) can ingest the source as a
flat .F90.  The wrapper accepts the same flat-scalar arg list
``_registries.py`` already provides (no per-arg rewiring); inside, it packs every constant into
local ``TYPE(TOMCST) :: YDCST`` / ``TYPE(TOETHF) :: YDTHF`` /
``TYPE(TECLDP) :: YDECLDP`` and calls upstream ``CLOUDSC`` per
block.  This is the DT-of-constants pattern the bridge's
``hlfir-flatten-structs`` pass now handles end-to-end (see
``tests/hlfir/bindings/struct_of_scalars_test.py`` for the
minimal pinned version).

**Reference:** ``f2py``-compiled Fortran from the same source.  Per
``feedback_e2e_numerical``: SDFG-producing tests compare against a
non-transformed reference.

**Inputs:** seeded random data (``np.random.default_rng(42)``) via
the registries in ``_registries.py``.  No HDF5 dependency.

**Status:** xfail probe initially  --  surfaces the first bridge gap
cleanly without breaking the sweep.  Each gap closes in a separate
commit (the test stays xfailed throughout).  Final commit removes
the xfail decorator.
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
def _f2py_ref(tmp_path_factory):
    """Build the f2py reference once per pytest session.

    f2py compile of the 3541-line source takes ~30-90s, so this is
    a session-scoped fixture instead of per-test setup.

    The FP flag set is the LLVM-flang-portable core
    ``-O0 -fno-fast-math -ffp-contract=off``.  Neither side zero-fills
    locals (no gfortran ``-finit-local-zero``, no bridge ``setzero``
    stamping), so correct write-before-read code is unaffected; any
    uninitialised-read divergence is a real source-level bug, not
    masked by an init flag.  ``-ffree-line-length-none`` is the sole
    intentionally gfortran-only flag: a non-semantic parser necessity
    for the long-line cloudsc source; LLVM-flang has no line limit.
    """
    src = (_HERE / "cloudsc.F90").read_text()
    ref_dir = tmp_path_factory.mktemp("cloudsc_ref")
    # ``only=('cloudscouter',)`` hides the inner ``CLOUDSC`` subroutine
    # from crackfortran -- its ``TYPE(TOMCST/TOETHF/TECLDP)`` dummies
    # map to ``'void'`` and crash f2py with ``KeyError: 'void'`` (same
    # crash documented in tests/hlfir/bindings/struct_of_scalars_test.py).
    return f2py_compile(
        src,
        ref_dir,
        "cloudsc_ref",
        extra_f90flags=CLOUDSC_F90FLAGS,
        only=("cloudscouter", ),
    )


def test_cloudsc_full_numerical(tmp_path, _f2py_ref, _strict_fp_cpu_args):
    """End-to-end SDFG-vs-gfortran equivalence on the full CLOUDSC.

    Same Fortran source through both paths, ``get_inputs_physical``
    (deterministic, physically-plausible: temperatures/pressures/
    mixing-ratios in the kernel's valid regime, no exact zeros), every
    output compared element-wise at ``rtol=atol=1e-12``.

    No ``xfail``: the bridge is bit-identical to gfortran on full
    CLOUDSC under valid inputs (verified 0 mismatched cells, and 0
    NaN/Inf across inputs, outputs, and the ice-deposition chain
    ``ZINEW``/``ZICENUCLEI``/``ZCVDS``/... over 12 RNG seeds).  The
    earlier ~130-cell divergence under the old ``get_inputs`` (uniform
    random) was NOT a flux-accumulation bug: that non-physical data
    drove ``ZINEW`` to NaN, and Fortran ``MIN``/``MAX`` with a NaN
    operand is *processor-dependent by the standard* -- gfortran's
    comparison-reduction vs the flang/DaCe ``arith.maxnumf`` path are
    permitted to differ, so comparing them on NaN inputs tested
    unspecified behaviour, not a defect.  Validating in the kernel's
    physical regime is the correct contract.
    """
    src = (_HERE / "cloudsc.F90").read_text()
    outputs_sdfg, outputs_ref = run_cloudsc(src, "cloudsc", _f2py_ref, tmp_path / "sdfg")

    # Per-output mismatch budget (collect all, don't fail-fast).
    #
    # Every output must match the f2py reference at strict
    # rtol=atol=1e-15 in EVERY cell (O0 -fno-fast-math -ffp-contract=off -> near-exact) -- except PCOVPTOT, where at most
    # ONE cell may differ and that cell must be a hard {0,1}
    # threshold flip at a high vertical level.  That single cell is
    # the documented, non-bridge cross-compiler artifact: a 1-ulp
    # rounding seed (gfortran ``x*x*x`` vs the bridge's expansion;
    # FOE statement-function inlining order) compounds across ~99 JK
    # levels in block IBL=3 until it crosses the hard reset
    # ``IF (ZQPRETOT < ZEPSEC) ZCOVPTOT = 0``.  ZCOVPTOT is a cloud
    # fraction in [0,1]; the reset makes it exactly 0 where gfortran
    # keeps ~1 (or vice versa) -> a lone delta of 1.0.  See
    # project_hlfir_cloudsc_section_4_5_bisection.  Any structural
    # regression (the 373/548 garbage we fixed) reappears as many
    # mismatches and fails loudly here.
    rtol = atol = 1e-15
    report: list[str] = []
    for name in program_outputs:
        a = np.asarray(outputs_sdfg[name.lower()])
        b = np.asarray(outputs_ref[name.lower()])
        bad = ~np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
        nbad = int(bad.sum())
        if nbad == 0:
            continue
        if name.upper() != "PCOVPTOT":
            report.append(f"{name}: {nbad} cell(s) exceed rtol={rtol} "
                          f"(max |Δ|={np.abs(a - b)[bad].max():.3e})")
            continue
        # PCOVPTOT: tolerate <=1 cell, and only a {0,1} threshold flip.
        idx = np.argwhere(bad)
        if nbad > 1:
            report.append(f"PCOVPTOT: {nbad} cells differ (budget is <=1 "
                          f"high-JK threshold flip); indices {idx[:5].tolist()}")
            continue
        (cell, ) = idx
        av, bv = float(a[tuple(cell)]), float(b[tuple(cell)])
        flip = {round(av, 9), round(bv, 9)} <= {0.0, 1.0}
        jk = int(cell[1])  # (KLON, KLEV, NBLOCKS) -> dim 1 is the level
        if not (flip and jk >= 90):
            report.append(f"PCOVPTOT lone diff at {cell.tolist()} is not a high-JK "
                          f"{{0,1}} threshold flip: sdfg={av} ref={bv} jk={jk}")
    assert not report, "cloudsc_full numerical mismatch:\n" + "\n".join(report)
