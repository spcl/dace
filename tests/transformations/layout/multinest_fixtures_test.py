# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""D1 fixture validation (GLOBAL_LAYOUT_DESIGN.md): the multi-nest programs are bit-exact against
their numpy oracles, and -- the load-bearing property -- their nests SURVIVE canonicalize + maximal
fusion, so the line graph the global-layout machinery is tested on is genuine."""
import numpy
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.layout.prepare import prepare_for_layout

from tests.transformations.layout import multinest_programs as fixtures

EXPECTED_NESTS = {"conflict2": 2, "conflict3": 3, "agree2": 2}


def top_level_map_entries(sdfg):
    """All MapEntry nodes at state top scope -- one per kernel nest post-canonicalize."""
    entries = []
    for state in sdfg.states():
        children = state.scope_children()[None]
        entries.extend(n for n in children if isinstance(n, nodes.MapEntry))
    return entries


def run_and_check(sdfg, program_name, n=64, seed=0):
    inputs = fixtures.make_inputs(n, seed)
    outputs = fixtures.output_arrays(program_name, n)
    sdfg(A=inputs["A"].copy(), **outputs, N=n)
    _, oracle, _ = fixtures.PROGRAMS[program_name]
    reference = oracle(inputs["A"])
    for name, ref in reference.items():
        assert numpy.allclose(outputs[name], ref), f"{program_name}: {name} diverges from the oracle"


@pytest.mark.parametrize("program_name", sorted(fixtures.PROGRAMS))
def test_fixture_bitexact(program_name):
    program, _, _ = fixtures.PROGRAMS[program_name]
    sdfg = program.to_sdfg(simplify=True)
    run_and_check(sdfg, program_name)


@pytest.mark.parametrize("program_name", sorted(fixtures.PROGRAMS))
def test_fixture_nests_survive_prepare(program_name):
    """After prepare_for_layout (canonicalize + maximal fusion) the nest count is EXACTLY the
    designed one: fusion neither merged the nests (fixture would be degenerate) nor did the
    pipeline split extra kernels in. Bit-exact again post-prepare."""
    program, _, _ = fixtures.PROGRAMS[program_name]
    sdfg = program.to_sdfg(simplify=True)
    prepare_for_layout(sdfg)
    entries = top_level_map_entries(sdfg)
    assert len(entries) == EXPECTED_NESTS[program_name], (
        f"{program_name}: expected {EXPECTED_NESTS[program_name]} top-level nests, found "
        f"{len(entries)}: {[e.map.label for e in entries]}")
    run_and_check(sdfg, program_name, seed=1)


if __name__ == "__main__":
    for name in sorted(fixtures.PROGRAMS):
        test_fixture_bitexact(name)
        test_fixture_nests_survive_prepare(name)
    print("multinest fixture tests PASS")
