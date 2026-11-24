# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""CloudSC from Fortran, canonicalized and rendered as CPF: every loop and map is classed, and the open ones only shrink.

dace-fortran parses ECMWF's ``cloudsc.F90``; canonicalization specializes ``nclv = 5`` as the dace-fortran driver
does, and the CPF rendering classes every loop (it raises on one without a class). The per-column 5x5 solve -- the LU
pivots over ``zqlhs`` and the forward and backward substitutions over ``zqxn`` -- is a genuine recurrence and must
read ``sequential`` with a proof. ``UNKNOWN_BOUND`` is the number of loops still ``unknown``: a change may lower it,
never raise it.

Needs a dace-fortran checkout in ``DACE_FORTRAN_ROOT`` with dace-fortran installed over this DaCe; the
``dace-fortran`` workflow provides both.
"""

import os
import re
from pathlib import Path

import pytest

from dace.codegen.cpf import render
from dace.transformation.passes.canonicalize.pipeline import canonicalize

pytestmark = pytest.mark.long

#: Loops left ``unknown`` in the rendering's header (measured 2026-11-24): may go down, never up.
UNKNOWN_BOUND = 15

#: Loops of CloudSC's per-column 5x5 solve proven sequential, per array: the four substitution loops over ``zqxn``
#: and the LU pivot loop over ``zqlhs``.
SOLVE_PROOFS = {"zqxn": 4, "zqlhs": 1}


def cloudsc_source() -> Path:
    root = Path(os.environ.get("DACE_FORTRAN_ROOT", ""))
    assert (root / "dace_fortran").is_dir(), "DACE_FORTRAN_ROOT must name a dace-fortran checkout"
    return root / "tests" / "cloudsc" / "full" / "cloudsc.F90"


def header_loops(code: str, kind: str) -> dict[str, str]:
    """``{loop: reason}`` for every ``kind`` loop the rendering's summary header lists."""
    return dict(re.findall(rf"^//   {kind}: ([^,\n]+), (.*)$", code, re.MULTILINE))


def test_cloudsc_classes_every_loop_and_proves_its_column_solve_sequential():
    import dace_fortran  # Optional: only the dace-fortran workflow installs it

    sdfg = dace_fortran.build_sdfg(cloudsc_source().read_text(), entry="cloudscouter", name="cloudsc_kinds")
    canonicalize(sdfg, specialize_constants={"nclv": 5})
    code = render(sdfg, language="c").code
    sequential = header_loops(code, "sequential").values()
    proofs = {
        name: sum(reason.startswith(f"proven dependence on {name}[") for reason in sequential) for name in SOLVE_PROOFS
    }
    assert proofs == SOLVE_PROOFS, sequential
    unknown = header_loops(code, "unknown")
    assert len(unknown) <= UNKNOWN_BOUND, unknown


if __name__ == "__main__":
    test_cloudsc_classes_every_loop_and_proves_its_column_solve_sequential()
