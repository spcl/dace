# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A workspace zero-filled ahead of a loop that writes all of it before reading it every iteration loses the fill, and
the loop then privatizes it; a fill any read can observe stays (quatrex_rgf's energy loop stayed sequential)."""

import numpy as np

import dace
from dace.libraries.standard.nodes import FillLibraryNode
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate.loop_to_map import LoopToMap
from dace.transformation.passes.assignment_and_copy_kernel_to_memset_and_memcpy import (
    AssignmentAndCopyKernelToMemsetAndMemcpy,
)
from dace.transformation.passes.canonicalize.dead_loop_initialization import DeadLoopInitialization
from dace.transformation.passes.simplify import SimplifyPass

NE, NB, BS = dace.symbol("NE"), dace.symbol("NB"), dace.symbol("BS")


@dace.program
def sweeps(a: dace.float64[NE, NB, BS], out: dace.float64[NE, NB, BS]):
    w = np.zeros((NB, BS), dtype=np.float64)
    for e in range(NE):
        w[0, :] = a[e, 0, :]
        for i in range(NB - 1):
            w[i + 1, :] = w[i, :] * 0.5 + a[e, i + 1, :]
        for i in range(NB - 1):
            w[NB - i - 2, :] = w[NB - i - 2, :] + w[NB - i - 1, :]
        out[e, :, :] = w[:, :]


@dace.program
def partial_rows(a: dace.float64[NE, NB, BS], out: dace.float64[NE, NB, BS]):
    w = np.zeros((NB, BS), dtype=np.float64)
    for e in range(NE):
        for i in range(NB - 1):
            w[i + 1, :] = a[e, i + 1, :]
        out[e, :, :] = w[:, :]


@dace.program
def read_first(a: dace.float64[NE, BS], out: dace.float64[NE, BS]):
    w = np.zeros((BS,), dtype=np.float64)
    for e in range(NE):
        out[e, :] = w[:]
        w[:] = a[e, :]


@dace.program
def read_after(a: dace.float64[NE, BS], out: dace.float64[BS]):
    w = np.zeros((BS,), dtype=np.float64)
    for e in range(NE):
        w[:] = a[e, :]
    out[:] = w[:]


def prepared(program) -> dace.SDFG:
    """Simplified, with the zero-fill maps turned into the ``FillLibraryNode`` canonicalization makes of them."""
    sdfg = program.to_sdfg(simplify=False)
    SimplifyPass().apply_pass(sdfg, {})
    AssignmentAndCopyKernelToMemsetAndMemcpy().apply_pass(sdfg, {})
    return sdfg


def fills(sdfg: dace.SDFG) -> int:
    return sum(1 for node, _ in sdfg.all_nodes_recursive() if isinstance(node, FillLibraryNode))


def test_a_workspace_tiled_every_iteration_loses_its_fill_and_the_loop_maps():
    sdfg = prepared(sweeps)
    reference = dace.SDFG.from_json(sdfg.to_json())
    reference.name = "sweeps_reference"
    assert fills(sdfg) == 1
    assert DeadLoopInitialization().apply_pass(sdfg, {}) == 1
    assert fills(sdfg) == 0
    outer = next(r for r in sdfg.nodes() if isinstance(r, LoopRegion))
    xform = LoopToMap()
    xform.setup_match(sdfg, sdfg.cfg_id, -1, {LoopToMap.loop: sdfg.node_id(outer)}, 0, override=True)
    assert xform.can_be_applied(sdfg, 0, sdfg), xform.last_refusal
    rng = np.random.default_rng(0)
    a = rng.random((5, 6, 3))
    want, got = np.zeros((5, 6, 3)), np.zeros((5, 6, 3))
    reference(a=a, out=want, NE=5, NB=6, BS=3)
    sdfg(a=a, out=got, NE=5, NB=6, BS=3)
    assert np.allclose(got, want)


def test_a_fill_an_unwritten_row_shows_stays():
    """Row 0 is never written in the loop: its zeros reach ``out``."""
    sdfg = prepared(partial_rows)
    assert fills(sdfg) == 1
    assert DeadLoopInitialization().apply_pass(sdfg, {}) is None
    assert fills(sdfg) == 1


def test_a_fill_the_first_iteration_reads_stays():
    sdfg = prepared(read_first)
    assert fills(sdfg) == 1
    assert DeadLoopInitialization().apply_pass(sdfg, {}) is None
    assert fills(sdfg) == 1


def test_a_fill_read_after_the_loop_stays():
    """A zero-trip loop leaves the fill as the result."""
    sdfg = prepared(read_after)
    assert fills(sdfg) == 1
    assert DeadLoopInitialization().apply_pass(sdfg, {}) is None
    assert fills(sdfg) == 1


if __name__ == "__main__":
    test_a_workspace_tiled_every_iteration_loses_its_fill_and_the_loop_maps()
    test_a_fill_an_unwritten_row_shows_stays()
    test_a_fill_the_first_iteration_reads_stays()
    test_a_fill_read_after_the_loop_stays()
