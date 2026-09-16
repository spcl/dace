# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from unittest import mock

import dace
from dace.transformation.dataflow import MapFusionVertical, TrivialMapElimination
from dace.transformation.passes import pattern_matching


def _make_sdfg() -> dace.SDFG:
    """Three maps in a row, the last one with a single iteration, so both transformations match."""
    sdfg = dace.SDFG("order_by_transformation")
    for array, transient in [("a", False), ("t1", True), ("t2", True), ("b", False)]:
        sdfg.add_array(array, shape=(10, ), dtype=dace.float64, transient=transient)
    state = sdfg.add_state(is_start_block=True)
    previous = state.add_access("a")
    for target, rng in [("t1", "0:10"), ("t2", "0:10"), ("b", "0:1")]:
        _, _, exit_node = state.add_mapped_tasklet(
            f"comp_{target}",
            map_ranges={"__i": rng},
            inputs={"__in": dace.Memlet(f"{previous.data}[__i]")},
            outputs={"__out": dace.Memlet(f"{target}[__i]")},
            code="__out = __in + 1.0",
            input_nodes={previous},
            external_edges=True,
        )
        previous = next(e.dst for e in state.out_edges(exit_node))
    sdfg.validate()
    return sdfg


def _apply(sdfg: dace.SDFG, order_by_transformation: bool) -> tuple[int, int]:
    """Returns the number of applied transformations and of pattern enumerations."""
    original = pattern_matching.match_patterns
    with mock.patch.object(pattern_matching, "match_patterns", side_effect=original) as spy:
        applied = sdfg.apply_transformations_repeated(
            [MapFusionVertical(), TrivialMapElimination()],
            validate=False,
            order_by_transformation=order_by_transformation,
        )
    return applied, spy.call_count


def test_order_by_transformation_applies_the_same_matches():
    ordered, unordered = _make_sdfg(), _make_sdfg()

    applied_ordered, enumerations_ordered = _apply(ordered, True)
    applied_unordered, enumerations_unordered = _apply(unordered, False)

    assert applied_ordered > 0
    assert applied_ordered == applied_unordered
    assert ordered.hash_sdfg() == unordered.hash_sdfg()
    # Matching runs on the metadata of all transformations of the pass, so ordering by
    #  transformation cannot narrow it: one enumeration per application, plus the final empty one.
    assert enumerations_ordered == applied_ordered + 1
    assert enumerations_unordered == applied_unordered + 1


if __name__ == "__main__":
    test_order_by_transformation_applies_the_same_matches()
