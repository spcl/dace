# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""LoopToMap must keep a loop-carried-dependency loop sequential.

These are *non-vectorizable* TSVC kernels: a recurrence along one loop
dimension (e.g. ``aa[j, i] = aa[j - 1, i] + bb[j, i]``) makes that loop
unparallelisable. ``LoopToMap`` must leave the carried-dependency loop a
``LoopRegion`` -- the orthogonal independent loop may still map. If
``LoopToMap`` ever turns a carried-dependency loop into a ``Map``, the
downstream vectorizer would parallelise a recurrence and produce wrong
results.

This is a lightweight structural guard: it runs ``LoopToMap`` without any
vectorization options and asserts only the loop / map shape, with no
compile or numeric run. It is the regression net for the rule "every loop
that must stay a loop stays a loop". The kernels are reused from
:mod:`tests.passes.vectorization.tsvc_2d.test_2d` rather than redefined.
"""
import copy

import dace
import pytest

from dace.sdfg.state import LoopRegion
from dace.transformation.interstate import LoopToMap
from tests.passes.vectorization.tsvc_1d.test_misc import s481_d_single, s482_d_single
from tests.passes.vectorization.tsvc_2d.test_2d import (s1119_d_single, s2101_d_single, s2111_d_single, s231_d_single,
                                                        s2275_d_single, s232_d_single, s235_d_single, s256_d_single,
                                                        s257_d_single)

# (kernel, loop variables carrying a real dependency that MUST stay sequential).
_CARRIED_DEP = [
    (s1119_d_single, {"i"}),       # aa[i, j] = aa[i - 1, j] + bb[i, j]
    (s231_d_single, {"j"}),        # aa[j, i] = aa[j - 1, i] + bb[j, i]
    (s232_d_single, {"i"}),        # aa[j, i] = aa[j, i - 1] ** 2 + bb[j, i]
    (s235_d_single, {"j"}),        # aa[j, i] = aa[j - 1, i] + bb[j, i] * a[i]
    (s2111_d_single, {"i", "j"}),  # Gauss-Seidel: aa[j, i - 1] and aa[j - 1, i]
    (s256_d_single, {"j"}),        # a[j] = 1.0 - a[j - 1]
    (s257_d_single, {"i"}),        # a[i] = aa[j, i] - a[i - 1]
]

# Data-dependent ``break`` makes the loop trip count runtime-dependent: a
# later iteration only runs if no earlier one broke, so the loop cannot be
# parallelised. LoopToMap must keep it sequential.
_DATA_DEPENDENT_BREAK = [
    (s481_d_single, {"i"}),        # if d[i] < 0.0: break  (before the write)
    (s482_d_single, {"i"}),        # ...; if c[i] > b[i]: break  (after the write)
]

# Fully-parallel TSVC kernels (control): every loop SHOULD map, so the guard
# above is not trivially satisfied by a LoopToMap that never fires.
_FULLY_PARALLEL = [
    (s2101_d_single, {"i"}),       # diagonal aa[i, i] -- independent across i
    (s2275_d_single, {"i", "j"}),  # column aa[j, i] -- both dims independent
]


def _loops_and_maps_after_l2map(prog):
    """Apply simplify + LoopToMap and return the surviving loop variables and map params.

    Walks nested SDFGs too: a carried-dependency loop nested inside a
    mapped independent loop lives in the map's body SDFG, not the top-level
    control-flow graph.

    :param prog: The ``@dace.program`` kernel to lower.
    :returns: ``(loop_variables, map_params)`` as two sets of strings.
    """
    sdfg = copy.deepcopy(prog.to_sdfg(simplify=False))
    sdfg.simplify(validate=True, validate_all=True)
    sdfg.apply_transformations_repeated(LoopToMap())
    sdfg.simplify()

    loops, map_params = set(), set()
    for sub in sdfg.all_sdfgs_recursive():
        for region in sub.all_control_flow_regions():
            if isinstance(region, LoopRegion):
                loops.add(region.loop_variable)
        for state in sub.states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.MapEntry):
                    map_params.update(node.map.params)
    return loops, map_params


@pytest.mark.parametrize("prog,carried_vars", _CARRIED_DEP, ids=[p[0].name for p in _CARRIED_DEP])
def test_loop_to_map_keeps_carried_dep_sequential(prog, carried_vars):
    loops, map_params = _loops_and_maps_after_l2map(prog)
    wrongly_mapped = carried_vars & map_params
    assert not wrongly_mapped, (
        f"{prog.name}: LoopToMap parallelised carried-dependency loop(s) {wrongly_mapped} into a Map "
        f"(map_params={map_params}); this would vectorize a recurrence and produce wrong results.")
    missing = carried_vars - loops
    assert not missing, (
        f"{prog.name}: carried-dependency loop(s) {missing} are no longer a LoopRegion (loops_kept={loops}).")


@pytest.mark.parametrize("prog,break_vars", _DATA_DEPENDENT_BREAK, ids=[p[0].name for p in _DATA_DEPENDENT_BREAK])
def test_loop_to_map_keeps_data_dependent_break_sequential(prog, break_vars):
    loops, map_params = _loops_and_maps_after_l2map(prog)
    wrongly_mapped = break_vars & map_params
    assert not wrongly_mapped, (
        f"{prog.name}: LoopToMap parallelised a loop with a data-dependent break {wrongly_mapped} "
        f"(map_params={map_params}); the early exit makes the trip count runtime-dependent.")
    missing = break_vars - loops
    assert not missing, (
        f"{prog.name}: break loop(s) {missing} are no longer a LoopRegion (loops_kept={loops}).")


@pytest.mark.parametrize("prog,parallel_vars", _FULLY_PARALLEL, ids=[p[0].name for p in _FULLY_PARALLEL])
def test_loop_to_map_parallelises_independent_loops(prog, parallel_vars):
    loops, map_params = _loops_and_maps_after_l2map(prog)
    not_mapped = parallel_vars - map_params
    assert not not_mapped, (
        f"{prog.name}: independent loop(s) {not_mapped} were NOT mapped (map_params={map_params}, "
        f"loops_kept={loops}); LoopToMap should parallelise them.")
