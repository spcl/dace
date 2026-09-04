# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The reusable extract -> evaluate -> best wrapper (nest_eval): every wrap-mode candidate of a
fixture nest verifies against the pristine-copy reference, timing records median + spread metadata,
and the identity-first tie-break law is enforced."""
import numpy
import pytest

import dace
from dace.transformation.layout import assignment_costs, nest_eval
from dace.transformation.layout.externalize import nest_entries
from dace.transformation.layout.nest_eval import (IDENTITY_TAG, default_permutation_candidates, evaluate_nest)
from dace.transformation.layout.prepare import prepare_for_layout
from dace.transformation.layout.timing import compute_region_stats_timer

from tests.transformations.layout import multinest_programs as fixtures


def prepared_nests(program_name):
    program, oracle, _ = fixtures.PROGRAMS[program_name]
    sdfg = program.to_sdfg(simplify=True)
    prepare_for_layout(sdfg)
    return sdfg, [(state, entry) for state in sdfg.states() for entry in nest_entries(state)]


def test_conflict2_nests_evaluate_and_rank(n=48):
    """Both conflict2 nests: all permutation candidates verify, all are timed with spread
    metadata, and a best candidate exists."""
    _, nests = prepared_nests("conflict2")
    assert len(nests) == 2
    inputs = fixtures.make_inputs(n, seed=5)
    chain = {"A": inputs["A"], **fixtures.conflict2_oracle(inputs["A"])}

    for state, entry in nests:
        ev = evaluate_nest(state,
                           entry,
                           symbols={"N": n},
                           provided=chain,
                           reps=3,
                           warmup=1,
                           timer=compute_region_stats_timer,
                           name=f"ev_c2_{entry.map.label}")
        assert all(r.correct for r in ev.results), [(r.name, r.error) for r in ev.results]
        names = [r.name for r in ev.results]
        # identity + one non-identity permutation per 2-D non-transient array of the nest
        two_d = [a for a, d in ev.ext.arrays.items() if not d.transient and len(d.shape) == 2]
        assert IDENTITY_TAG in names and len(names) == 1 + len(two_d), names
        for r in ev.results:
            assert r.time is not None and r.time >= 0.0
            assert "spread" in r.metadata and "contended" in r.metadata
        assert ev.best() is not None


def test_provided_inputs_reach_the_reference(n=32):
    """The reference outputs come from the caller-provided inputs, not from random fill: nest 1 of
    agree2 (B = 2A) must reproduce 2 * the provided A exactly."""
    _, nests = prepared_nests("agree2")
    inputs = fixtures.make_inputs(n, seed=9)
    by_output = {}
    for state, entry in nests:
        ev = evaluate_nest(state,
                           entry,
                           symbols={"N": n},
                           provided={
                               "A": inputs["A"],
                               **fixtures.agree2_oracle(inputs["A"])
                           },
                           reps=2,
                           warmup=1,
                           name=f"ev_a2_{entry.map.label}")
        by_output.update({out: ev for out in ev.reference})
    assert numpy.allclose(by_output["B"].reference["B"], 2.0 * inputs["A"])
    ref = fixtures.agree2_oracle(inputs["A"])
    assert numpy.allclose(by_output["C"].reference["C"], ref["C"])


def test_identity_must_come_first(n=16):
    """A candidate dict that does not enumerate identity first is refused (the tie-break law is an
    invariant, not a convention)."""
    _, nests = prepared_nests("conflict2")
    state, entry = nests[0]
    with pytest.raises(ValueError, match="tie-break"):
        evaluate_nest(state,
                      entry,
                      symbols={"N": n},
                      candidates={
                          "permute_first": lambda sdfg: None,
                          IDENTITY_TAG: lambda sdfg: None
                      },
                      name="ev_order_check")


def test_default_candidates_identity_first():
    _, nests = prepared_nests("conflict2")
    state, entry = nests[0]
    from dace.transformation.layout.externalize import externalize_nest
    ext = externalize_nest(state, entry, name="ev_defaults_check")
    cands = default_permutation_candidates(ext)
    assert next(iter(cands)) == IDENTITY_TAG
    assert all(tag.startswith("permute_") for tag in list(cands)[1:])


def test_rank_above_max_permute_ndim_is_refused():
    """A non-transient array of rank > MAX_PERMUTE_NDIM is refused loudly. The enumeration was
    unguarded, so a rank-d array yielded d! candidates -- 766 measured on one nest -- each deepcopy'd,
    compiled, RUN and TIMED, while the modelled path refused the same rank outright. The transient
    check still comes first: a transient of any rank is simply skipped."""
    over_rank = dace.SDFG("nest_eval_rank4")
    over_rank.add_array("X", [2, 2, 2, 2], dace.float64)
    with pytest.raises(NotImplementedError, match="rank"):
        default_permutation_candidates(over_rank)

    transient_only = dace.SDFG("nest_eval_rank4_transient")
    transient_only.add_array("T", [2, 2, 2, 2], dace.float64, transient=True)
    assert list(default_permutation_candidates(transient_only)) == [IDENTITY_TAG]


def test_max_permute_ndim_is_shared_with_the_modelled_path():
    """MAX_PERMUTE_NDIM lives in nest_eval and assignment_costs imports it, so the MEASURED and the
    MODELLED candidate spaces cannot drift: both must accept the same top rank -- with the same
    candidate count -- and refuse the next one."""
    assert assignment_costs.MAX_PERMUTE_NDIM is nest_eval.MAX_PERMUTE_NDIM
    top = nest_eval.MAX_PERMUTE_NDIM

    accepted = dace.SDFG("nest_eval_shared_top")
    accepted.add_array("X", [2] * top, dace.float64)
    assert len(default_permutation_candidates(accepted)) == len(assignment_costs.permutation_layouts(top))

    refused = dace.SDFG("nest_eval_shared_over")
    refused.add_array("X", [2] * (top + 1), dace.float64)
    with pytest.raises(NotImplementedError, match="rank"):
        default_permutation_candidates(refused)
    with pytest.raises(NotImplementedError, match="rank"):
        assignment_costs.permutation_layouts(top + 1)


def test_rank3_array_still_enumerates_every_permutation():
    """The rank guard must not shrink the space it allows: a rank-3 non-transient array still yields
    identity plus all 5 non-identity permutations, identity first (the tie-break law)."""
    sdfg = dace.SDFG("nest_eval_rank3")
    sdfg.add_array("X", [2, 2, 2], dace.float64)
    assert list(default_permutation_candidates(sdfg)) == [
        IDENTITY_TAG, "permute_X_021", "permute_X_102", "permute_X_120", "permute_X_201", "permute_X_210"
    ]


N = dace.symbol("N")


@dace.program
def transpose_only(A: dace.float64[N, N], B: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        B[i, j] = A[j, i]


def test_pure_copy_nest_identity_is_timed(n=48):
    """A transpose nest's kernel tasklet is a pure copy: its sole externalized compute state used to
    classify as a relayout boundary (copy-in AND copy-out at once), leaving the IDENTITY candidate
    untimed -- ranked last as inf while the permute candidates timed fine, silently inverting the
    identity-first law for exactly the kernel class layout matters most for. The timing boundary is
    the symmetric difference now, so identity times like every other candidate."""
    sdfg = transpose_only.to_sdfg(simplify=True)
    prepare_for_layout(sdfg)
    nests = [(state, entry) for state in sdfg.states() for entry in nest_entries(state)]
    assert len(nests) == 1
    state, entry = nests[0]
    ev = evaluate_nest(state,
                       entry,
                       symbols={"N": n},
                       reps=3,
                       warmup=1,
                       timer=compute_region_stats_timer,
                       name="ev_transpose")
    assert all(r.correct for r in ev.results), [(r.name, r.error) for r in ev.results]
    assert all(r.time is not None for r in ev.results), [(r.name, r.time) for r in ev.results]
    assert ev.best() is not None


if __name__ == "__main__":
    test_conflict2_nests_evaluate_and_rank()
    test_provided_inputs_reach_the_reference()
    test_identity_must_come_first()
    test_default_candidates_identity_first()
    test_rank_above_max_permute_ndim_is_refused()
    test_max_permute_ndim_is_shared_with_the_modelled_path()
    test_rank3_array_still_enumerates_every_permutation()
    test_pure_copy_nest_identity_is_timed()
    print("nest_eval tests PASS")
