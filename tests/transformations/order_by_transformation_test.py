# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Any, List, Tuple
from unittest import mock

import pytest

import dace
from dace.sdfg import validation
from dace.transformation import dataflow, transformation as xf
from dace.transformation.passes import pattern_matching

# The transformations of the pass, in this order. Each one matches once in the SDFG of `_make_sdfg()`.
_TRANSFORMATIONS = (dataflow.MapFusionVertical, dataflow.TrivialMapElimination)


def _add_maps(state: dace.SDFGState, chain: List[Tuple[str, str]]) -> None:
    """Adds a chain of maps to `state`, where each `(array, range)` pair writes `array` over `range`."""
    previous = state.add_access(chain[0][0])
    for target, rng in chain[1:]:
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


def _make_sdfg() -> dace.SDFG:
    """
    Two states: the first one only matches `TrivialMapElimination`, the second one only matches
    `MapFusionVertical`. Since matching visits states before transformations, the first match found
    over all transformations is `TrivialMapElimination`, even if it is listed last.
    """
    sdfg = dace.SDFG("order_by_transformation")
    for array, transient in [("a", False), ("b", False), ("c", False), ("t", True), ("d", False)]:
        sdfg.add_array(array, shape=(10, ), dtype=dace.float64, transient=transient)
    first = sdfg.add_state(is_start_block=True)
    _add_maps(first, [("a", ""), ("b", "0:1")])
    second = sdfg.add_state_after(first)
    _add_maps(second, [("c", ""), ("t", "0:10"), ("d", "0:10")])
    sdfg.validate()
    return sdfg


def _apply(sdfg: dace.SDFG, order_by_transformation: bool) -> Tuple[List[str], int]:
    """Returns the names of the applied transformations, in order, and the number of pattern enumerations."""
    applied = []
    original_apply = pattern_matching.PatternMatchAndApplyRepeated._apply_and_validate

    def _record(self: pattern_matching.PatternMatchAndApplyRepeated, match: xf.PatternTransformation, *args: Any,
                **kwargs: Any) -> None:
        applied.append(type(match).__name__)
        original_apply(self, match, *args, **kwargs)

    with (mock.patch.object(pattern_matching.PatternMatchAndApplyRepeated, "_apply_and_validate", _record),
          mock.patch.object(pattern_matching, "match_patterns", side_effect=pattern_matching.match_patterns) as spy):
        sdfg.apply_transformations_repeated(
            [xform() for xform in _TRANSFORMATIONS],
            validate=False,
            order_by_transformation=order_by_transformation,
        )
    return applied, spy.call_count


def test_order_by_transformation() -> None:
    ordered, unordered = _make_sdfg(), _make_sdfg()

    applied_ordered, enumerations_ordered = _apply(ordered, True)
    applied_unordered, enumerations_unordered = _apply(unordered, False)

    # Ordered: each transformation is exhausted before moving to the next one.
    assert applied_ordered == ["MapFusionVertical", "TrivialMapElimination"]
    # Unordered: the first match found over all transformations is applied.
    assert applied_unordered == ["TrivialMapElimination", "MapFusionVertical"]
    # The matches are independent, so both orders lead to the same result.
    assert ordered.hash_sdfg() == unordered.hash_sdfg()

    # Ordered: per transformation, one enumeration per application plus the final empty one, and
    #  since something was applied, one more round of empty enumerations.
    assert enumerations_ordered == len(applied_ordered) + 2 * len(_TRANSFORMATIONS)
    # Unordered: one enumeration per application, plus the final empty one.
    assert enumerations_unordered == len(applied_unordered) + 1


def test_apply_first_match_per_transformation() -> None:
    # `PatternMatchAndApply` applies, for each transformation in order, the first match of that transformation.
    sdfg = _make_sdfg()
    applied = pattern_matching.PatternMatchAndApply([xform() for xform in _TRANSFORMATIONS],
                                                    validate=False).apply_pass(sdfg, {})
    assert list(applied.keys()) == ["MapFusionVertical", "TrivialMapElimination"]
    assert all(len(results) == 1 for results in applied.values())


@pytest.mark.parametrize("order_by_transformation, last_applied", [(True, "TrivialMapElimination"),
                                                                   (False, "MapFusionVertical")])
def test_validation_failure_names_last_applied_transformation(order_by_transformation: bool, last_applied: str) -> None:
    # Both transformations apply, in the order `test_order_by_transformation` established for this SDFG and
    #  mode; `last_applied` here is the second (last) one of that order, not just any applied transformation.
    sdfg = _make_sdfg()
    failure = validation.InvalidSDFGError("invalid", sdfg, None)

    with mock.patch.object(dace.SDFG, "validate", side_effect=failure):
        with pytest.raises(validation.InvalidSDFGError, match=f"after applying {last_applied}") as info:
            sdfg.apply_transformations_repeated(
                [xform() for xform in _TRANSFORMATIONS],
                validate=True,
                order_by_transformation=order_by_transformation,
            )
    assert info.value.__cause__ is failure


if __name__ == "__main__":
    test_order_by_transformation()
    test_apply_first_match_per_transformation()
    test_validation_failure_names_last_applied_transformation(True, "TrivialMapElimination")
    test_validation_failure_names_last_applied_transformation(False, "MapFusionVertical")
