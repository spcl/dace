# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Extra tests for NormalizeScheduleForLayout.

Complements ``normalize_schedule_test.py`` by exercising branches it does not touch: direct detection
of the inner block width from POINT ``Mod(param, b)`` accesses, refusal when a parameter carries two
conflicting widths, asymmetric per-dimension tile sizes, the ``divides_evenly`` flag's effect on the
inner tile range, the ``_already_tiled`` idempotence guard in isolation, and multiple top-level maps
tiled in a single pass. Where the SDFG compiles the result is checked bit-exact against numpy."""
import numpy

import dace
from dace.transformation import pass_pipeline as ppl
from dace.transformation.layout.split_dimensions import SplitDimensions
from dace.transformation.layout.normalize_schedule import normalize_schedule_for_layout, NormalizeScheduleForLayout

N = dace.symbol("N")


@dace.program
def ew1d(A: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N] @ dace.ScheduleType.Sequential:
        C[i] = A[i] * 2.0 + 1.0


@dace.program
def ew2d(A: dace.float64[N, N], C: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N] @ dace.ScheduleType.Sequential:
        C[i, j] = A[i, j] * 3.0


@dace.program
def two_ops(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N] @ dace.ScheduleType.Sequential:
        B[i] = A[i] + B[i]


@dace.program
def chain2(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N] @ dace.ScheduleType.Sequential:
        B[i] = A[i] + 1.0
    for i in dace.map[0:N] @ dace.ScheduleType.Sequential:
        C[i] = B[i] * 2.0


def count_maps(sdfg):
    return sum(1 for st in sdfg.all_states() for n in st.nodes() if isinstance(n, dace.nodes.MapEntry))


def first_top_map(sdfg):
    """Return the (state, MapEntry) of the first top-level (scope==None) map, or (None, None)."""
    for st in sdfg.all_states():
        scope = st.scope_dict()
        for n in st.nodes():
            if isinstance(n, dace.nodes.MapEntry) and scope[n] is None:
                return st, n
    return None, None


def child_map_of(state, entry):
    """Return the single map nested directly inside ``entry`` (the inner tile map after tiling)."""
    scope = state.scope_dict()
    children = [n for n in state.nodes() if isinstance(n, dace.nodes.MapEntry) and scope[n] is entry]
    assert len(children) == 1, children
    return children[0]


def test_block_width_detects_point_mod_1d():
    """After blocking A and C by 4, ``_block_width`` reads the POINT ``Mod(i, 4)`` offsets and reports
    exactly one width per map parameter; the pass then tiles that single map bit-exactly."""
    sdfg = ew1d.to_sdfg(simplify=True)
    sdfg.name = "nse_blockwidth"
    SplitDimensions(split_map={"A": ([True], [4]), "C": ([True], [4])}).apply_pass(sdfg, {})

    state, me = first_top_map(sdfg)
    detected = NormalizeScheduleForLayout()._block_width(state, me)
    assert detected == {"i": 4}, detected

    assert normalize_schedule_for_layout(sdfg, divides_evenly=True) == 1

    _N = 16
    A = numpy.random.rand(_N // 4, 4)
    C = numpy.zeros((_N // 4, 4))
    sdfg(A=A, C=C, N=_N)
    assert numpy.allclose(C, A * 2.0 + 1.0)


def test_conflicting_block_widths_untouched():
    """A single parameter that indexes one operand blocked by 4 and another blocked by 8 has two
    candidate widths; the map is not uniformly blocked, so ``_block_width`` returns None and the pass
    leaves the schedule completely untouched (no map added)."""
    sdfg = two_ops.to_sdfg(simplify=True)
    SplitDimensions(split_map={"A": ([True], [4]), "B": ([True], [8])}).apply_pass(sdfg, {})

    state, me = first_top_map(sdfg)
    assert NormalizeScheduleForLayout()._block_width(state, me) is None

    before = count_maps(sdfg)
    assert normalize_schedule_for_layout(sdfg) == 0
    assert count_maps(sdfg) == before


def test_normalize_asymmetric_widths_bitexact():
    """A 2D map whose two dimensions are blocked by DIFFERENT widths (4 and 2) is tiled by the exact
    per-parameter tuple (4, 2); the inner tile map spans both blocks and the result is unchanged."""
    sdfg = ew2d.to_sdfg(simplify=True)
    sdfg.name = "nse_asym2d"
    SplitDimensions(split_map={"A": ([True, True], [4, 2]), "C": ([True, True], [4, 2])}).apply_pass(sdfg, {})

    state, me = first_top_map(sdfg)
    assert NormalizeScheduleForLayout()._block_width(state, me) == {"i": 4, "j": 2}

    before = count_maps(sdfg)
    assert normalize_schedule_for_layout(sdfg, divides_evenly=True) == 1
    assert count_maps(sdfg) == before + 1

    state, outer = first_top_map(sdfg)
    inner = child_map_of(state, outer)
    # The inner tile map spans a 4x2 block: each dimension has extent equal to its width.
    sizes = [int(dace.symbolic.simplify(e - b + 1)) for (b, e, s) in inner.map.range.ranges]
    assert sizes == [4, 2], sizes

    _N = 8
    A = numpy.random.rand(_N // 4, _N // 2, 4, 2)
    C = numpy.zeros((_N // 4, _N // 2, 4, 2))
    sdfg(A=A, C=C, N=_N)
    assert numpy.allclose(C, A * 3.0)


def test_divides_evenly_flag_shapes_inner_range():
    """``divides_evenly=True`` yields a clean constant-extent inner tile (no clamping); ``False`` keeps
    a ``Min(...)`` remainder guard in the inner range. Both are bit-exact when N is a multiple of 4."""
    inner_strs = {}
    for divides in (True, False):
        sdfg = ew1d.to_sdfg(simplify=True)
        sdfg.name = "nse_divides_true" if divides else "nse_divides_false"
        SplitDimensions(split_map={"A": ([True], [4]), "C": ([True], [4])}).apply_pass(sdfg, {})
        assert normalize_schedule_for_layout(sdfg, divides_evenly=divides) == 1

        state, outer = first_top_map(sdfg)
        inner = child_map_of(state, outer)
        inner_strs[divides] = str(inner.map.range)

        _N = 16
        A = numpy.random.rand(_N // 4, 4)
        C = numpy.zeros((_N // 4, 4))
        sdfg(A=A, C=C, N=_N)
        assert numpy.allclose(C, A * 2.0 + 1.0)

    assert "Min" not in inner_strs[True] and "N" not in inner_strs[True], inner_strs[True]
    assert "Min" in inner_strs[False], inner_strs[False]
    assert inner_strs[True] != inner_strs[False]


def test_idempotent_bitexact_and_map_count_stable():
    """Running the pass a second time tiles nothing, does not add a map, and leaves the (already
    correct) numerical result untouched -- running twice equals running once."""
    sdfg = ew1d.to_sdfg(simplify=True)
    sdfg.name = "nse_idempotent"
    SplitDimensions(split_map={"A": ([True], [4]), "C": ([True], [4])}).apply_pass(sdfg, {})

    before = count_maps(sdfg)
    assert normalize_schedule_for_layout(sdfg, divides_evenly=True) == 1
    after_once = count_maps(sdfg)
    assert after_once == before + 1

    _N = 16
    A = numpy.random.rand(_N // 4, 4)
    C = numpy.zeros((_N // 4, 4))
    sdfg(A=A, C=C, N=_N)
    assert numpy.allclose(C, A * 2.0 + 1.0)

    # Second (and third) application must be a no-op in both structure and numerics.
    assert normalize_schedule_for_layout(sdfg, divides_evenly=True) == 0
    assert normalize_schedule_for_layout(sdfg, divides_evenly=True) == 0
    assert count_maps(sdfg) == after_once

    C2 = numpy.zeros((_N // 4, 4))
    sdfg(A=A, C=C2, N=_N)
    assert numpy.allclose(C2, A * 2.0 + 1.0)


def test_already_tiled_guard_in_isolation():
    """``_already_tiled`` recognizes a map that already iterates exactly a ``0:b`` tile and rejects a
    map still iterating the full symbolic extent -- the guard that makes re-runs idempotent."""
    p = NormalizeScheduleForLayout()
    sdfg = dace.SDFG("guard")
    state = sdfg.add_state()
    tiled_entry, _ = state.add_map("tiled", {"i": "0:4"})
    full_entry, _ = state.add_map("full", {"i": "0:N"})

    assert p._already_tiled(tiled_entry, {"i": 4}) is True
    assert p._already_tiled(full_entry, {"i": 4}) is False
    # A tile map measured against a DIFFERENT width is not considered already-tiled.
    assert p._already_tiled(tiled_entry, {"i": 8}) is False


def test_two_top_level_maps_both_tiled_bitexact():
    """Two independent top-level maps that both carry a uniform block-4 signal are each tiled in a
    single pass (count == 2, two maps added) and the fused pipeline stays bit-exact."""
    sdfg = chain2.to_sdfg(simplify=True)
    sdfg.name = "nse_twomaps"
    SplitDimensions(split_map={"A": ([True], [4]), "B": ([True], [4]), "C": ([True], [4])}).apply_pass(sdfg, {})

    before = count_maps(sdfg)
    assert normalize_schedule_for_layout(sdfg, divides_evenly=True) == 2
    assert count_maps(sdfg) == before + 2

    _N = 16
    A = numpy.random.rand(_N // 4, 4)
    B = numpy.zeros((_N // 4, 4))
    C = numpy.zeros((_N // 4, 4))
    sdfg(A=A, B=B, C=C, N=_N)
    assert numpy.allclose(C, (A + 1.0) * 2.0)


def test_pass_contract_metadata():
    """The pass declares it mutates nodes, memlets and scopes, and opts out of automatic reapplication
    (it is a one-shot normalization, not a fixpoint pass)."""
    p = NormalizeScheduleForLayout()
    mods = p.modifies()
    assert bool(mods & ppl.Modifies.Nodes)
    assert bool(mods & ppl.Modifies.Memlets)
    assert bool(mods & ppl.Modifies.Scopes)
    assert p.should_reapply(ppl.Modifies.Everything) is False
    assert p.should_reapply(mods) is False


if __name__ == "__main__":
    test_block_width_detects_point_mod_1d()
    test_conflicting_block_widths_untouched()
    test_normalize_asymmetric_widths_bitexact()
    test_divides_evenly_flag_shapes_inner_range()
    test_idempotent_bitexact_and_map_count_stable()
    test_already_tiled_guard_in_isolation()
    test_two_top_level_maps_both_tiled_bitexact()
    test_pass_contract_metadata()
    print("normalize_schedule extra tests PASS")
