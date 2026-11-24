# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Statement distribution through a guarded branch (ICON velocity_tendencies' CFL clipping, ``mo_velocity_advection``):
each statement gets its own copy of the guard and recomputes the scalar temporary, so the column loop distributes into
one loop per statement."""

import ctypes
import os

import numpy as np

import dace
from dace.codegen.cpf import render
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.canonicalize.split_statements import SplitStatements
from tests.codegen.cpf.conftest import assert_matches, build_standalone, call_standalone

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


@dace.program
def cfl_clipping(
    z: dace.float64[KL, NC],
    dd: dace.float64[KL, NC],
    clipping: dace.bool_[KL, NC],
    levmask: dace.bool_[KL],
    vcflmax: dace.float64[1],
    dtime: dace.float64,
    wlim: dace.float64,
):
    for k in range(KL):
        levmask[k] = False
    maxvcfl = 0.0
    for k in range(KL):
        clip_count = 0
        for c in range(NC):
            clipping[k, c] = abs(z[k, c]) > wlim * dd[k, c]
            if clipping[k, c]:
                clip_count = clip_count + 1
        if clip_count > 0:
            for c in range(NC):
                if clipping[k, c]:
                    levmask[k] = True
                    vcfl = z[k, c] * dtime / dd[k, c]
                    maxvcfl = max(maxvcfl, abs(vcfl))
                    if vcfl < -0.85:
                        z[k, c] = -0.85 * dd[k, c] / dtime
                    elif vcfl > 0.85:
                        z[k, c] = 0.85 * dd[k, c] / dtime
    vcflmax[0] = maxvcfl


def cfl_reference(z: np.ndarray, dd: np.ndarray, dtime: float, wlim: float) -> dict[str, np.ndarray]:
    clipping = np.abs(z) > wlim * dd
    vcfl = z * dtime / dd
    clipped = np.where(clipping & (vcfl < -0.85), -0.85 * dd / dtime, z)
    clipped = np.where(clipping & (vcfl > 0.85), 0.85 * dd / dtime, clipped)
    return {
        "z": clipped,
        "levmask": clipping.any(axis=1),
        "vcflmax": np.array([np.abs(vcfl)[clipping].max(initial=0.0)]),
    }


def test_the_cfl_clipping_loop_distributes_into_maps_and_reductions():
    sdfg = cfl_clipping.to_sdfg(simplify=True)
    sdfg.name = "guarded_cfl_clipping"
    canonicalize(sdfg, target="cpu")
    finalize_for_target(sdfg, "cpu")
    result = render(sdfg, language="c")
    header = [line for line in result.code.splitlines() if line.startswith("// cpf: proven")]
    assert header == ["// cpf: proven: all"], header
    assert "// unknown -- " not in result.code and "// sequential -- " not in result.code, result.code
    library = build_standalone(result.code, sdfg.name, language="c")
    rng = np.random.default_rng(1)
    kl, nc = 7, 29
    z, dd = rng.normal(size=(kl, nc)), rng.random((kl, nc)) + 0.5
    z[2, :] = 0.0
    want = cfl_reference(z, dd, 2.0, 0.6)
    for threads in (1, 3, 8):
        ctypes.CDLL("libgomp.so.1").omp_set_num_threads(threads)
        got = {"z": z.copy(), "levmask": np.ones(kl, dtype=np.bool_), "vcflmax": np.zeros(1)}
        arguments = {**got, "dd": dd, "clipping": np.zeros((kl, nc), dtype=np.bool_), "dtime": 2.0, "wlim": 0.6}
        call_standalone(library, result.sdfg, {**arguments, "KL": kl, "NC": nc})
        assert_matches(want, got, f"{threads} threads")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    test_a_guarded_body_distributes_one_loop_per_statement()
    test_the_cfl_clipping_loop_distributes_into_maps_and_reductions()
