# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``EliminateTrivialMaps`` is ``apply_transformations_repeated(TeamSafeTrivialMapElimination)`` in one sweep per
state: the same maps go, the same team maps stay, and the generated code is the same.

The repeated form matches the whole SDFG again after every application, which on a program with a thousand trivial
maps (CloudSC as NestForge canonicalizes it) takes most of a CI job. A sweep that missed a map, or touched a node an
earlier elimination removed, would change the code silently; these tests compare the two forms on the shapes that
matter: maps inside maps, maps in nested SDFGs freed by an outer elimination, and a thread team."""

import copy

import dace
from dace.codegen import cpf
from dace.codegen.cpf import EliminateTrivialMaps, TeamSafeTrivialMapElimination
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.canonicalize.pipeline import canonicalize

N = dace.symbol("N")


@dace.program
def one_trip_outer(a: dace.float64[N, 1], b: dace.float64[N, 1]):
    for i in dace.map[0:1]:
        for j in dace.map[0:N]:
            for k in dace.map[0:1]:
                b[j, k] = a[j, k] * 2.0 + i


@dace.program
def partly_trivial(a: dace.float64[N, 3], b: dace.float64[N, 3]):
    for j, k in dace.map[0:N, 1:2]:
        b[j, k] = a[j, k] + 1.0


@dace.program
def team_sweep(level: dace.float64[N], T: dace.int64):
    for d in range(T):
        scratch = level * 2.0 + d
        for i in dace.map[0:N]:
            level[i] = scratch[N - 1 - i] * 0.5


def both_ways(sdfg: dace.SDFG) -> tuple[tuple[int, str], tuple[int, str]]:
    """The count and the generated code after the repeated transformation and after the sweep, on copies."""
    old, new = copy.deepcopy(sdfg), copy.deepcopy(sdfg)
    old_count = old.apply_transformations_repeated(TeamSafeTrivialMapElimination, validate=False)
    new_count = EliminateTrivialMaps().apply_pass(new, {}) or 0
    old.validate()
    new.validate()
    return (old_count, old.generate_code()[0].clean_code), (new_count, new.generate_code()[0].clean_code)


def maps(sdfg: dace.SDFG) -> int:
    return sum(isinstance(n, nodes.MapEntry) for n, parent in sdfg.all_nodes_recursive())


def test_nested_one_trip_maps_go_as_the_repeated_form_removes_them():
    sdfg = one_trip_outer.to_sdfg(simplify=False)
    assert maps(sdfg) >= 3

    old, new = both_ways(sdfg)

    assert new == old
    assert old[0] >= 2, "both one-trip maps are eliminated"


def test_a_partly_trivial_map_keeps_its_other_dimension():
    old, new = both_ways(partly_trivial.to_sdfg(simplify=True))

    assert new == old and old[0] == 1


def test_a_thread_team_keeps_its_maps_and_its_rendering():
    sdfg = team_sweep.to_sdfg(simplify=True)
    canonicalize(sdfg, target="cpu")
    finalize_for_target(sdfg, "cpu")

    old, new = both_ways(sdfg)

    assert new == old
    rendered = cpf.render(copy.deepcopy(sdfg), language="c++").code
    assert "#pragma omp parallel" in rendered, "the team survives the sweep inside prepare"


def test_a_second_run_finds_nothing():
    sdfg = one_trip_outer.to_sdfg(simplify=False)
    EliminateTrivialMaps().apply_pass(sdfg, {})
    assert EliminateTrivialMaps().apply_pass(sdfg, {}) is None, "a second run finds nothing"
