# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A1 externalize_nest (GLOBAL_LAYOUT_DESIGN.md): each nest of a multi-nest program, cut out into a
standalone SDFG, is bit-exact against its PER-NEST numpy oracle -- inputs produced by earlier nests
are provided (promoted to non-transient by the cutout), outputs start as deterministic noise and
must be fully overwritten."""
import numpy
import pytest

from dace.sdfg import nodes
from dace.transformation.layout.externalize import (externalize_nest, nest_arguments, nest_entries, written_array_names)
from dace.transformation.layout.prepare import prepare_for_layout

from tests.transformations.layout import multinest_programs as fixtures

# Each fixture nest writes exactly one array; its name indexes the per-nest oracle.
NEST_INDEX_BY_OUTPUT = {"B": 0, "C": 1, "D": 2}


def all_nests(sdfg):
    """Every (state, top-level map entry) pair in the program, any order."""
    return [(state, entry) for state in sdfg.states() for entry in nest_entries(state)]


@pytest.mark.parametrize("program_name", ["conflict2", "conflict3", "agree2"])
def test_externalized_nests_match_per_nest_oracles(program_name, n=48):
    program, oracle, nest_oracles = fixtures.PROGRAMS[program_name]
    sdfg = program.to_sdfg(simplify=True)
    prepare_for_layout(sdfg)

    inputs = fixtures.make_inputs(n, seed=3)
    chain = {"A": inputs["A"], **oracle(inputs["A"])}  # every array's oracle value, feeds nest inputs

    nests = all_nests(sdfg)
    assert len(nests) == len(nest_oracles)
    covered = set()
    for state, entry in nests:
        ext = externalize_nest(state, entry, name=f"{program_name}_{entry.map.label}_ext")
        written = written_array_names(ext)
        assert len(written) == 1, f"fixture nests write exactly one array, got {written}"
        out_name = written.pop()
        covered.add(out_name)

        provided = {name: chain[name] for name in ext.arrays if name in chain and name != out_name}
        args = nest_arguments(ext, symbols={"N": n}, provided=provided, seed=7)
        assert not numpy.allclose(args[out_name], chain[out_name])  # output starts as noise
        ext(**args, N=n)

        reference = nest_oracles[NEST_INDEX_BY_OUTPUT[out_name]](**chain)
        assert numpy.allclose(args[out_name], reference[out_name]), \
            f"{program_name}: externalized nest writing {out_name} diverges from its oracle"
    assert len(covered) == len(nest_oracles)  # each nest hit a DISTINCT oracle (KeyError above on a foreign one)


def test_externalize_refuses_ambiguous_state_without_map_entry():
    """A state holding several nests must be refused when no map_entry picks one."""
    import dace
    sdfg = dace.SDFG("two_nests_one_state")
    sdfg.add_array("X", [16], dace.float64)
    sdfg.add_array("Y", [16], dace.float64)
    state = sdfg.add_state("s", is_start_block=True)
    for arr in ("X", "Y"):
        me, mx = state.add_map(f"m_{arr}", {"i": "0:16"})
        t = state.add_tasklet(f"t_{arr}", {"a"}, {"b"}, "b = a + 1.0")
        state.add_memlet_path(state.add_read(arr), me, t, dst_conn="a", memlet=dace.Memlet(f"{arr}[i]"))
        state.add_memlet_path(t, mx, state.add_write(arr), src_conn="b", memlet=dace.Memlet(f"{arr}[i]"))
    with pytest.raises(ValueError, match="top-level"):
        externalize_nest(state)


if __name__ == "__main__":
    for name in ["conflict2", "conflict3", "agree2"]:
        test_externalized_nests_match_per_nest_oracles(name)
    print("externalize_nest tests PASS")
