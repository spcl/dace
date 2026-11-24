# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Statement distribution through a guarded branch (ICON velocity_tendencies' CFL clipping, ``mo_velocity_advection``):
each statement gets its own copy of the guard and recomputes the scalar temporary, so the column loop distributes into
one loop per statement."""

import os

import numpy as np

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.split_statements import SplitStatements
from tests.codegen.cpf.conftest import assert_matches

NC = dace.symbol("NC")
KL = dace.symbol("KL")


@dace.program
def guarded_column(
    z: dace.float64[NC], dd: dace.float64[NC], clipping: dace.bool_[NC], flag: dace.bool_[2], peak: dace.float64[1]
):
    for c in range(NC):
        if clipping[c]:
            flag[1] = True
            vcfl = z[c] * 2.0 / dd[c]
            peak[0] = max(peak[0], abs(vcfl))
            z[c] = vcfl * 0.5 * dd[c]


def test_a_guarded_body_distributes_one_loop_per_statement():
    """Each statement keeps its own copy of the guard and recomputes ``vcfl``; the store to ``z`` runs last, after the
    two statements that read ``z`` through ``vcfl``."""
    sdfg = guarded_column.to_sdfg(simplify=True)
    sdfg.name = "guarded_column"
    assert sum(isinstance(r, LoopRegion) for r in sdfg.all_control_flow_regions(recursive=True)) == 1
    assert SplitStatements().apply_pass(sdfg, {}) == 1
    assert sum(isinstance(r, LoopRegion) for r in sdfg.all_control_flow_regions(recursive=True)) == 3, sdfg
    rng = np.random.default_rng(3)
    z, dd, clipping = rng.normal(size=64), rng.random(64) + 0.5, rng.random(64) > 0.5
    got = {"z": z.copy(), "flag": np.zeros(2, dtype=np.bool_), "peak": np.zeros(1)}
    sdfg(**got, dd=dd, clipping=clipping, NC=64)
    vcfl = z * 2.0 / dd
    want = {
        "z": np.where(clipping, vcfl * 0.5 * dd, z),
        "flag": np.array([False, True]),
        "peak": np.array([np.abs(vcfl)[clipping].max()]),
    }
    assert_matches(want, got)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    test_a_guarded_body_distributes_one_loop_per_statement()
