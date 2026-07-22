# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""C1/C2/C3 solver tests (GLOBAL_LAYOUT_DESIGN.md) on synthetic cost tables: the DP matches the
brute-force oracle everywhere, the greedy baseline loses exactly where it should (edge-blind), both
edge regimes work, ties resolve toward identity, and every refusal is loud."""
import random

import pytest

from dace.libraries.layout.algebra import Permute
from dace.transformation.layout.apply_assignment import IDENTITY_LAYOUT, Layout
from dace.transformation.layout.global_assign import (AssignmentCosts, brute_force_trajectories, conflict_report,
                                                      format_conflict_report, greedy_assignment, per_array_dp,
                                                      to_assignment, trajectory_cost)

CM = Layout("perm10", (Permute((1, 0)), ))

# A THREE-layout candidate set. With two layouts identity sits at index 0, so almost any tie-break looks
# identity-first by accident; three is the smallest set that can tell a whole-trajectory lexicographic
# tie-break apart from a per-kernel one.
P120 = Layout("perm120", (Permute((1, 2, 0)), ))
P201 = Layout("perm201", (Permute((2, 0, 1)), ))
THREE = {"A": [IDENTITY_LAYOUT, P120, P201]}


def three_layout_table(node, relayout, **kwargs):
    return AssignmentCosts(layouts=THREE, node_cost=node, relayout_cost=relayout, **kwargs)


def flat_three(n_kernels, relayout_cost=0.0):
    """Every layout equally good at every kernel, every conversion free: EVERY trajectory ties."""
    node = {("A", k, l.tag): 1.0 for k in range(n_kernels) for l in THREE["A"]}
    rel = {("A", a.tag, b.tag): relayout_cost for a in THREE["A"] for b in THREE["A"] if a is not b}
    return node, rel


def table(node, relayout, layouts=None):
    layouts = layouts or {"A": [IDENTITY_LAYOUT, CM]}
    return AssignmentCosts(layouts=layouts, node_cost=node, relayout_cost=relayout)


def k17_table(n_kernels=4, delta=0.5, relayout=2.0):
    """Alternating preferences: kernel k prefers identity (even) / colmajor (odd) by ``delta``;
    a boundary change costs ``relayout``."""
    node = {}
    for k in range(n_kernels):
        preferred, other = ("identity", "perm10") if k % 2 == 0 else ("perm10", "identity")
        node[("A", k, preferred)] = 1.0
        node[("A", k, other)] = 1.0 + delta
    rel = {("A", "identity", "perm10"): relayout, ("A", "perm10", "identity"): relayout}
    return table(node, rel)


def test_greedy_pays_for_edge_blindness():
    """relayout > delta: greedy flips every boundary; the DP keeps one layout. The k17 figure."""
    costs = k17_table(n_kernels=4, delta=0.5, relayout=2.0)
    greedy = greedy_assignment(costs, 4)["A"]
    dp = per_array_dp(costs, 4)["A"]
    oracle = brute_force_trajectories(costs, 4)["A"]
    assert greedy.tags == ["identity", "perm10", "identity", "perm10"]
    assert greedy.cost == pytest.approx(4 * 1.0 + 3 * 2.0)
    assert dp.tags == ["identity"] * 4  # tie between all-identity and all-cm -> identity-first law
    assert dp.cost == pytest.approx(oracle.cost) and oracle.cost == pytest.approx(1 + 1.5 + 1 + 1.5)


def test_trajectory_wins_when_relayout_is_cheap():
    """delta > relayout: the paid change beats carrying the wrong layout (conflict3's shape:
    one producer nest, then two transposed readers)."""
    node = {
        ("A", 0, "identity"): 1.0,
        ("A", 0, "perm10"): 3.0,
        ("A", 1, "identity"): 3.0,
        ("A", 1, "perm10"): 1.0,
        ("A", 2, "identity"): 3.0,
        ("A", 2, "perm10"): 1.0
    }
    rel = {("A", "identity", "perm10"): 0.5, ("A", "perm10", "identity"): 0.5}
    costs = table(node, rel)
    dp = per_array_dp(costs, 3)["A"]
    assert dp.tags == ["identity", "perm10", "perm10"] and dp.cost == pytest.approx(3.5)
    single = per_array_dp(costs, 3, allow_changes=False)["A"]
    assert single.tags == ["perm10"] * 3 and single.cost == pytest.approx(5.0)
    assert brute_force_trajectories(costs, 3)["A"].cost == pytest.approx(dp.cost)


def test_dp_matches_oracle_on_random_tables():
    """Random node/edge tables AND random liveness facts (entry needed / last-write position): the
    flag-state DP must equal the enumeration oracle under the full objective in both regimes."""
    rng = random.Random(42)
    layouts = {"A": [IDENTITY_LAYOUT, CM, Layout("perm01x", (Permute((0, 1)), ))]}
    tags = [l.tag for l in layouts["A"]]
    for trial in range(40):
        n = rng.randint(2, 5)
        node = {("A", k, t): rng.uniform(0.5, 5.0) for k in range(n) for t in tags}
        rel = {("A", a, b): rng.uniform(0.1, 3.0) for a in tags for b in tags if a != b}
        lw = rng.choice([None] + list(range(n)))
        costs = AssignmentCosts(layouts=layouts,
                                node_cost=node,
                                relayout_cost=rel,
                                entry_conversion_needed={"A": rng.random() < 0.5},
                                last_write_kernel={} if lw is None else {"A": lw})
        for allow in (True, False):
            dp = per_array_dp(costs, n, allow_changes=allow)["A"]
            oracle = brute_force_trajectories(costs, n, allow_changes=allow)["A"]
            assert dp.cost == pytest.approx(oracle.cost), (trial, allow)
            assert dp.tags == oracle.tags, (trial, allow, dp.tags, oracle.tags)
            assert dp.cost == pytest.approx(trajectory_cost(costs, "A", dp.tags))
            if not allow:
                assert dp.changes() == 0
        # A random subset of transitions is locked (a loop span): the DP must still equal the oracle
        # restricted to lock-respecting trajectories, and must not change layout across any locked transition.
        locked = {k for k in range(1, n) if rng.random() < 0.5}
        dp_locked = per_array_dp(costs, n, locked_before=locked)["A"]
        oracle_locked = brute_force_trajectories(costs, n, locked_before=locked)["A"]
        assert dp_locked.cost == pytest.approx(oracle_locked.cost), (trial, sorted(locked))
        assert all(dp_locked.tags[k] == dp_locked.tags[k - 1] for k in locked), (trial, sorted(locked))


def test_body_uniform_pins_a_loop_span():
    """Kernels 1,2 are a loop body (transition into 2 locked). With relayout cheap, the unconstrained DP flips
    at every boundary; body-uniform forces kernels 1,2 to one layout while the prologue/epilogue stay free."""
    node = {}
    for k in range(4):  # alternating single-layout preference
        preferred, other = ("identity", "perm10") if k % 2 == 0 else ("perm10", "identity")
        node[("A", k, preferred)] = 1.0
        node[("A", k, other)] = 2.0
    rel = {("A", "identity", "perm10"): 0.1, ("A", "perm10", "identity"): 0.1}
    costs = table(node, rel)

    free = per_array_dp(costs, 4)["A"]
    assert free.tags[1] != free.tags[2]  # cheap relayout -> unconstrained flips through the "body"

    locked = {2}  # kernels 1,2 share one enclosing loop; the internal transition is locked
    dp = per_array_dp(costs, 4, locked_before=locked)["A"]
    assert dp.tags[1] == dp.tags[2]  # body-uniform
    assert dp.cost == pytest.approx(brute_force_trajectories(costs, 4, locked_before=locked)["A"].cost)


def test_entry_conversion_flips_marginal_preference():
    """W read live-in only in kernel 1 of 3, never written: a marginal permute advantage must NOT
    win once the mandatory entry conversion is priced -- the edge-blind objective picked the SLOWER
    assignment here (the review's decision-flip case, exact numbers)."""
    node = {("W", k, t): 0.0 for k in (0, 2) for t in ("identity", "perm10")}
    node[("W", 1, "identity")] = 2.586e-6
    node[("W", 1, "perm10")] = 2.2e-6
    rel = {("W", "identity", "perm10"): 6.554e-7, ("W", "perm10", "identity"): 6.554e-7}
    layouts = {"W": [IDENTITY_LAYOUT, CM]}
    blind = AssignmentCosts(layouts=layouts, node_cost=node, relayout_cost=rel)
    assert per_array_dp(blind, 3)["W"].tags == ["perm10"] * 3  # no facts: the entry ride is free

    costs = AssignmentCosts(layouts=layouts, node_cost=node, relayout_cost=rel, entry_conversion_needed={"W": True})
    dp = per_array_dp(costs, 3)["W"]
    assert dp.tags == ["identity"] * 3
    assert dp.cost == pytest.approx(2.586e-6)
    assert dp.cost == pytest.approx(brute_force_trajectories(costs, 3)["W"].cost)
    single = per_array_dp(costs, 3, allow_changes=False)["W"]
    assert single.tags == ["identity"] * 3  # the single-layout regime pays the entry too


def test_exit_conversion_priced_once():
    """B WRITTEN under perm10 in kernel 0: [perm10, identity] pays exactly ONE conversion (the
    boundary restore IS the exit, moved onto the boundary), [perm10, perm10] pays one exit, and
    the restore-then-leave sandwich prices every real conversion -- no double-charge, no free
    oscillation."""
    rel = {("B", "identity", "perm10"): 0.4, ("B", "perm10", "identity"): 0.4}
    node = {("B", 0, "identity"): 2.0, ("B", 0, "perm10"): 1.0, ("B", 1, "identity"): 1.0, ("B", 1, "perm10"): 1.0}
    costs = AssignmentCosts(layouts={"B": [IDENTITY_LAYOUT, CM]},
                            node_cost=node,
                            relayout_cost=rel,
                            entry_conversion_needed={"B": False},
                            last_write_kernel={"B": 0})
    assert trajectory_cost(costs, "B", ["perm10", "identity"]) == pytest.approx(2.4)
    assert trajectory_cost(costs, "B", ["perm10", "perm10"]) == pytest.approx(2.4)
    assert trajectory_cost(costs, "B", ["identity", "identity"]) == pytest.approx(3.0)
    dp = per_array_dp(costs, 2)["B"]
    assert dp.cost == pytest.approx(2.4)
    assert dp.tags == ["perm10", "identity"]  # tie vs [perm10, perm10]: identity enumerated first
    assert dp.cost == pytest.approx(brute_force_trajectories(costs, 2)["B"].cost)

    node3 = {("B", k, t): 1.0 for k in range(3) for t in ("identity", "perm10")}
    costs3 = AssignmentCosts(layouts={"B": [IDENTITY_LAYOUT, CM]},
                             node_cost=node3,
                             relayout_cost=rel,
                             entry_conversion_needed={"B": False},
                             last_write_kernel={"B": 0})
    assert trajectory_cost(costs3, "B", ["perm10", "identity", "identity"]) == pytest.approx(3.4)
    assert trajectory_cost(costs3, "B", ["perm10", "perm10", "identity"]) == pytest.approx(3.4)
    assert trajectory_cost(costs3, "B", ["perm10", "identity", "perm10"]) == pytest.approx(3.8)
    assert trajectory_cost(costs3, "B", ["perm10", "perm10", "perm10"]) == pytest.approx(3.4)
    dp3 = per_array_dp(costs3, 3)["B"]
    assert dp3.cost == pytest.approx(brute_force_trajectories(costs3, 3)["B"].cost)


def test_conflict_report_triad():
    costs = k17_table()
    rows = conflict_report(costs, 4)
    assert len(rows) == 1 and rows[0].array == "A"
    assert rows[0].conflicting and rows[0].greedy_cost > rows[0].global_cost
    assert rows[0].global_cost == pytest.approx(rows[0].single_layout_cost)  # sticking wins here
    assert "A" in format_conflict_report(rows)


def test_to_assignment_drops_identity_only():
    costs = k17_table()
    dp = per_array_dp(costs, 4)
    assert to_assignment(dp, costs.layouts) == {}  # all-identity -> nothing to apply
    node = {("A", 0, "identity"): 5.0, ("A", 0, "perm10"): 1.0}
    costs1 = table(node, {("A", "identity", "perm10"): 9.0, ("A", "perm10", "identity"): 9.0})
    dp1 = per_array_dp(costs1, 1)
    assignment = to_assignment(dp1, costs1.layouts)
    assert [l.tag for l in assignment["A"]] == ["perm10"]


def test_refusals_are_loud():
    with pytest.raises(ValueError, match="identity/baseline"):
        per_array_dp(table({("A", 0, "perm10"): 1.0}, {}, layouts={"A": [CM]}), 1)
    with pytest.raises(ValueError, match="missing node cost"):
        per_array_dp(table({("A", 0, "identity"): 1.0}, {}), 1)
    complete = k17_table(n_kernels=2)
    with pytest.raises(ValueError, match="exceed"):
        brute_force_trajectories(complete, 2, cap=3)
    missing_edge = table(
        {
            ("A", 0, "identity"): 1.0,
            ("A", 0, "perm10"): 1.0,
            ("A", 1, "identity"): 1.0,
            ("A", 1, "perm10"): 1.0
        }, {})
    with pytest.raises(ValueError, match="missing relayout"):
        per_array_dp(missing_edge, 2)
    per_array_dp(missing_edge, 2, allow_changes=False)  # no edges needed in the no-change regime


def test_dp_tie_breaks_toward_identity_not_toward_the_last_kernel():
    """The regression pin for the tie-break. Both trajectories below cost exactly 5.0, but only one
    starts at identity -- and starting at identity is what makes the entry conversion disappear, so the
    tie is not cosmetic. Resolving the final tie on the LAST kernel's layout index (or on the
    identity-visited flag) picks all-perm120 here, which pays for a clone the plan does not need."""
    node = {
        ("A", 0, "identity"): 1.5, ("A", 0, "perm120"): 1.0, ("A", 0, "perm201"): 1.5,
        ("A", 1, "identity"): 1.5, ("A", 1, "perm120"): 1.5, ("A", 1, "perm201"): 1.5,
        ("A", 2, "identity"): 1.5, ("A", 2, "perm120"): 1.5, ("A", 2, "perm201"): 1.0,
        ("A", 3, "identity"): 1.5, ("A", 3, "perm120"): 1.0, ("A", 3, "perm201"): 1.0,
    }  # yapf: disable
    rel = {
        ("A", "identity", "perm120"): 0.0, ("A", "identity", "perm201"): 0.0,
        ("A", "perm120", "identity"): 0.5, ("A", "perm120", "perm201"): 0.5,
        ("A", "perm201", "identity"): 0.0, ("A", "perm201", "perm120"): 0.5,
    }  # yapf: disable
    costs = three_layout_table(node, rel, entry_conversion_needed={"A": True})
    dp = per_array_dp(costs, 4)["A"]
    oracle = brute_force_trajectories(costs, 4)["A"]
    assert dp.cost == pytest.approx(5.0) and oracle.cost == pytest.approx(5.0)
    assert dp.tags == oracle.tags
    assert dp.tags == ["identity", "identity", "perm201", "perm201"]
    assert trajectory_cost(costs, "A", ["perm120"] * 4) == pytest.approx(5.0)  # the equal-cost rival


def test_dp_tie_breaks_lexicographically_with_three_layouts():
    """Identity-first is a WHOLE-TRAJECTORY law, and it still holds when every trajectory ties."""
    node, rel = flat_three(3)
    dp = per_array_dp(three_layout_table(node, rel), 3)["A"]
    oracle = brute_force_trajectories(three_layout_table(node, rel), 3)["A"]
    assert dp.tags == ["identity"] * 3 and dp.tags == oracle.tags
    # ...and it is genuinely lexicographic, not "always identity": let kernel 0 strictly prefer perm201
    node[("A", 0, "perm201")] = 0.5
    dp2 = per_array_dp(three_layout_table(node, rel), 3)["A"]
    oracle2 = brute_force_trajectories(three_layout_table(node, rel), 3)["A"]
    assert dp2.tags == oracle2.tags
    assert dp2.tags == ["perm201", "identity", "identity"]  # forced first, then identity-first again


def test_dp_tie_break_survives_the_exit_conversion_flag():
    """The identity-visited flag doubles every DP state; the tie-break must order on the trajectory, not
    on that bookkeeping bit."""
    node, rel = flat_three(2)
    costs = three_layout_table(node, rel, last_write_kernel={"A": 0})
    dp = per_array_dp(costs, 2)["A"]
    oracle = brute_force_trajectories(costs, 2)["A"]
    assert dp.tags == oracle.tags and dp.tags == ["identity", "identity"]


def test_dp_tags_match_the_oracle_with_three_layouts():
    """Asserting COST agreement alone is what let the tie-break bug through -- assert the TRAJECTORY.
    Costs come from a tiny discrete set so ties are common AND every sum is exact in binary (the DP
    accumulates incrementally, the oracle does not; inexact costs would tie-break on the last ulp)."""
    rng = random.Random(2027)
    for trial in range(400):
        n = rng.randint(1, 4)
        node = {("A", k, l.tag): rng.choice([1.0, 1.5]) for k in range(n) for l in THREE["A"]}
        rel = {("A", a.tag, b.tag): rng.choice([0.0, 0.5]) for a in THREE["A"] for b in THREE["A"] if a is not b}
        lw = rng.choice([None, 0, n - 1])
        costs = three_layout_table(node,
                                   rel,
                                   entry_conversion_needed={"A": rng.random() < 0.5},
                                   last_write_kernel={} if lw is None else {"A": lw})
        for allow in (True, False):
            dp = per_array_dp(costs, n, allow_changes=allow)["A"]
            oracle = brute_force_trajectories(costs, n, allow_changes=allow)["A"]
            assert dp.cost == pytest.approx(oracle.cost), (trial, allow)
            assert dp.tags == oracle.tags, (trial, allow, dp.tags, oracle.tags)


def test_conflict_report_respects_loop_locks():
    """The report is the user-facing decision surface. Without the loop locks it advertises a plan that
    changes layout INSIDE a loop body -- which apply_assignment refuses outright -- and quotes its cost."""
    costs = k17_table(n_kernels=4, delta=0.5, relayout=0.1)
    free = {r.array: r for r in conflict_report(costs, 4)}["A"]
    locked = {r.array: r for r in conflict_report(costs, 4, locked_before={2})}["A"]
    assert free.chosen == ["identity", "perm10", "identity", "perm10"]  # flips freely: relayout is cheap
    assert locked.chosen[1] == locked.chosen[2]  # kernels 1,2 are one loop body -> one layout
    assert locked.chosen == per_array_dp(costs, 4, locked_before={2})["A"].tags
    assert locked.global_cost > free.global_cost  # the feasible plan is dearer, and now quoted honestly
    # the greedy baseline must stay applicable too, else the triad compares against an infeasible plan
    assert locked.greedy_cost == pytest.approx(greedy_assignment(costs, 4, {2})["A"].cost)
    assert greedy_assignment(costs, 4, {2})["A"].tags[1] == greedy_assignment(costs, 4, {2})["A"].tags[2]


def test_check_requires_only_the_conversions_it_charges():
    """Entry and exit are charged independently in the single-layout regime, so demanding BOTH pairs
    falsely refuses a complete table."""
    node = {("A", 0, "identity"): 1.0, ("A", 0, "perm10"): 1.0}
    layouts = {"A": [IDENTITY_LAYOUT, CM]}
    exit_only = AssignmentCosts(layouts=layouts,
                                node_cost=node,
                                relayout_cost={("A", "perm10", "identity"): 2.0},
                                entry_conversion_needed={"A": False},
                                last_write_kernel={"A": 0})
    assert per_array_dp(exit_only, 1, allow_changes=False)["A"].tags == ["identity"]
    entry_only = AssignmentCosts(layouts=layouts,
                                 node_cost=node,
                                 relayout_cost={("A", "identity", "perm10"): 2.0},
                                 entry_conversion_needed={"A": True})
    assert per_array_dp(entry_only, 1, allow_changes=False)["A"].tags == ["identity"]
    # ...while an edge the solver WILL consult is still refused loudly
    with pytest.raises(ValueError, match="missing relayout"):
        per_array_dp(AssignmentCosts(layouts=layouts, node_cost=node, entry_conversion_needed={"A": True}),
                     1,
                     allow_changes=False)


def test_single_kernel_charges_entry_and_exit_together():
    """n=1 with an entry conversion AND a last write is the only path charging both on one kernel; the
    random oracle draws n >= 2 and never reaches it."""
    node = {("A", 0, "identity"): 5.0, ("A", 0, "perm10"): 1.0}
    rel = {("A", "identity", "perm10"): 1.0, ("A", "perm10", "identity"): 1.0}
    costs = AssignmentCosts(layouts={"A": [IDENTITY_LAYOUT, CM]},
                            node_cost=node,
                            relayout_cost=rel,
                            entry_conversion_needed={"A": True},
                            last_write_kernel={"A": 0})
    dp = per_array_dp(costs, 1)["A"]
    oracle = brute_force_trajectories(costs, 1)["A"]
    assert dp.tags == oracle.tags and dp.tags == ["perm10"]  # 1 (node) + 1 (entry) + 1 (exit) beats 5
    assert dp.cost == pytest.approx(3.0)
    assert oracle.cost == pytest.approx(3.0)


if __name__ == "__main__":
    test_greedy_pays_for_edge_blindness()
    test_trajectory_wins_when_relayout_is_cheap()
    test_dp_matches_oracle_on_random_tables()
    test_entry_conversion_flips_marginal_preference()
    test_exit_conversion_priced_once()
    test_conflict_report_triad()
    test_to_assignment_drops_identity_only()
    test_refusals_are_loud()
    test_dp_tie_breaks_toward_identity_not_toward_the_last_kernel()
    test_dp_tie_breaks_lexicographically_with_three_layouts()
    test_dp_tie_break_survives_the_exit_conversion_flag()
    test_dp_tags_match_the_oracle_with_three_layouts()
    test_conflict_report_respects_loop_locks()
    test_check_requires_only_the_conversions_it_charges()
    test_single_kernel_charges_entry_and_exit_together()
    print("global_assign tests PASS")
