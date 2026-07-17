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
            assert dp.cost == pytest.approx(trajectory_cost(costs, "A", dp.tags))
            if not allow:
                assert dp.changes() == 0


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


if __name__ == "__main__":
    test_greedy_pays_for_edge_blindness()
    test_trajectory_wins_when_relayout_is_cheap()
    test_dp_matches_oracle_on_random_tables()
    test_entry_conversion_flips_marginal_preference()
    test_exit_conversion_priced_once()
    test_conflict_report_triad()
    test_to_assignment_drops_identity_only()
    test_refusals_are_loud()
    print("global_assign tests PASS")
