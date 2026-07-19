# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Global layout assignment over the line graph: per-array Viterbi DP, brute-force oracle, per-op greedy baseline, and conflict report."""
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from dace.transformation.layout.apply_assignment import Layout


@dataclass
class AssignmentCosts:
    """Pluggable cost table: per-array layouts, node/relayout costs, entry/exit conversion flags, and untrusted markers."""
    layouts: Dict[str, List[Layout]]
    node_cost: Dict[Tuple[str, int, str], float]
    relayout_cost: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    entry_conversion_needed: Dict[str, bool] = field(default_factory=dict)
    last_write_kernel: Dict[str, Optional[int]] = field(default_factory=dict)
    untrusted: set = field(default_factory=set)

    def check(self, n_kernels: int, changes_allowed: bool = True) -> None:
        """Validate completeness and the identity-first law before any solve."""
        for array, layouts in self.layouts.items():
            if not layouts:
                raise ValueError(f"AssignmentCosts: '{array}' has no candidate layouts")
            if not layouts[0].is_identity:
                raise ValueError(f"AssignmentCosts: '{array}' must enumerate its identity/baseline "
                                 f"layout FIRST (tie-break law); got '{layouts[0].tag}'")
            tags = [l.tag for l in layouts]
            if len(set(tags)) != len(tags):
                raise ValueError(f"AssignmentCosts: duplicate layout tags for '{array}': {tags}")
            for k in range(n_kernels):
                for layout in layouts:
                    if (array, k, layout.tag) not in self.node_cost:
                        raise ValueError(f"AssignmentCosts: missing node cost ({array}, kernel {k}, "
                                         f"'{layout.tag}')")
            lw = self.last_write_kernel.get(array)
            if lw is not None and not 0 <= lw < n_kernels:
                raise ValueError(f"AssignmentCosts: last_write_kernel[{array!r}] = {lw} out of "
                                 f"range for {n_kernels} kernels")
            if len(layouts) > 1:
                if changes_allowed:
                    pairs = itertools.permutations(tags, 2)
                elif self.entry_conversion_needed.get(array, False) or lw is not None:
                    # single-layout regime still prices entry/exit conversions
                    pairs = [(tags[0], t) for t in tags[1:]] + [(t, tags[0]) for t in tags[1:]]
                else:
                    pairs = []
                for a, b in pairs:
                    if (array, a, b) not in self.relayout_cost:
                        raise ValueError(f"AssignmentCosts: missing relayout cost ({array}, "
                                         f"'{a}' -> '{b}')")


@dataclass
class ArrayTrajectory:
    """One array's solved trajectory: layout tag per kernel and its total cost."""
    array: str
    tags: List[str]
    cost: float

    def changes(self) -> int:
        return sum(1 for a, b in zip(self.tags, self.tags[1:]) if a != b)


def trajectory_cost(costs: AssignmentCosts, array: str, tags: List[str]) -> float:
    """Trajectory cost: node costs + boundary changes + entry/exit conversions (first post-write switch back to identity is free)."""
    identity = costs.layouts[array][0].tag
    lw = costs.last_write_kernel.get(array)
    total = sum(costs.node_cost[(array, k, tag)] for k, tag in enumerate(tags))
    if costs.entry_conversion_needed.get(array, False) and tags[0] != identity:
        total += costs.relayout_cost[(array, identity, tags[0])]
    if lw is not None and tags[lw] != identity:
        total += costs.relayout_cost[(array, tags[lw], identity)]
    restored = lw is not None and tags[lw] == identity
    for k in range(len(tags) - 1):
        if tags[k] != tags[k + 1]:
            exit_moved_here = (lw is not None and k >= lw and tags[k + 1] == identity and not restored)
            if not exit_moved_here:
                total += costs.relayout_cost[(array, tags[k], tags[k + 1])]
        if lw is not None and k + 1 >= lw and tags[k + 1] == identity:
            restored = True
    return total


def per_array_dp(costs: AssignmentCosts,
                 n_kernels: int,
                 allow_changes: bool = True,
                 locked_before: Optional[Set[int]] = None) -> Dict[str, ArrayTrajectory]:
    """Viterbi DP: cheapest per-array layout trajectory under `trajectory_cost`; ties resolve to the
    earlier-enumerated layout. ``locked_before`` holds kernel indices ``k`` whose transition from ``k-1``
    may not change layout (loop-span internal transitions, body-uniform); the layout at such ``k`` is forced
    equal to ``k-1``, so a loop body ends up with one layout and its back-edge is a genuine no-op."""
    locked_before = locked_before or set()
    costs.check(n_kernels, changes_allowed=allow_changes)
    solution: Dict[str, ArrayTrajectory] = {}
    for array, layouts in costs.layouts.items():
        tags = [l.tag for l in layouts]
        identity = tags[0]
        lw = costs.last_write_kernel.get(array)
        entry = costs.entry_conversion_needed.get(array, False)

        def node(k: int, j: int) -> float:
            c = costs.node_cost[(array, k, tags[j])]
            if k == 0 and entry and tags[j] != identity:
                c += costs.relayout_cost[(array, identity, tags[j])]
            if lw is not None and k == lw and tags[j] != identity:
                c += costs.relayout_cost[(array, tags[j], identity)]
            return c

        # dp[j][flag]: best cost at kernel k, layout j; flag = identity visited in tags[lw..k]
        dp: List[List[Optional[float]]] = [[None, None] for _ in tags]
        for j in range(len(tags)):
            flag = lw == 0 and tags[j] == identity
            dp[j][flag] = node(0, j)
        back: List[List[List[Optional[Tuple[int, int]]]]] = [[[None, None] for _ in tags]]
        for k in range(1, n_kernels):
            new_dp: List[List[Optional[float]]] = [[None, None] for _ in tags]
            new_back: List[List[Optional[Tuple[int, int]]]] = [[None, None] for _ in tags]
            for j in range(len(tags)):
                new_flag_base = lw is not None and k >= lw and tags[j] == identity
                for i in range(len(tags)):
                    for flag in (False, True):
                        if dp[i][flag] is None:
                            continue
                        if i != j:
                            if not allow_changes or k in locked_before:
                                continue
                            exit_moved_here = (lw is not None and k - 1 >= lw and tags[j] == identity and not flag)
                            edge = 0.0 if exit_moved_here else costs.relayout_cost[(array, tags[i], tags[j])]
                        else:
                            edge = 0.0
                        new_flag = flag or new_flag_base
                        c = dp[i][flag] + edge
                        if new_dp[j][new_flag] is None or c < new_dp[j][new_flag]:  # strict <
                            new_dp[j][new_flag] = c
                            new_back[j][new_flag] = (i, flag)
                for flag in (False, True):
                    if new_dp[j][flag] is not None:
                        new_dp[j][flag] += node(k, j)
            dp = new_dp
            back.append(new_back)
        final = min(((j, flag) for j in range(len(tags)) for flag in (False, True) if dp[j][flag] is not None),
                    key=lambda jf: (dp[jf[0]][jf[1]], jf[0], jf[1]))
        chosen = [final]
        for k in range(n_kernels - 1, 0, -1):
            j, flag = chosen[-1]
            chosen.append(back[k][j][flag])
        trajectory = [tags[j] for j, _ in reversed(chosen)]
        solution[array] = ArrayTrajectory(array, trajectory, dp[final[0]][final[1]])
    return solution


def brute_force_trajectories(costs: AssignmentCosts,
                             n_kernels: int,
                             allow_changes: bool = True,
                             cap: int = 1_000_000,
                             locked_before: Optional[Set[int]] = None) -> Dict[str, ArrayTrajectory]:
    """Enumeration oracle: every trajectory per array, raising if the space exceeds `cap`. ``locked_before``
    (see :func:`per_array_dp`) filters out trajectories that change layout across a locked transition, so this
    stays a valid oracle for the body-uniform DP."""
    locked_before = locked_before or set()
    costs.check(n_kernels, changes_allowed=allow_changes)
    solution: Dict[str, ArrayTrajectory] = {}
    for array, layouts in costs.layouts.items():
        tags = [l.tag for l in layouts]
        space = len(tags)**n_kernels if allow_changes else len(tags)
        if space > cap:
            raise ValueError(f"brute_force_trajectories: {space} trajectories for '{array}' exceed "
                             f"the cap ({cap}); raise it explicitly or use the DP")
        candidates = (itertools.product(tags, repeat=n_kernels) if allow_changes else
                      ((tag, ) * n_kernels for tag in tags))
        best: Optional[ArrayTrajectory] = None
        for trajectory in candidates:
            if any(trajectory[k] != trajectory[k - 1] for k in locked_before):
                continue
            c = trajectory_cost(costs, array, list(trajectory))
            if best is None or c < best.cost:  # strict <: enumeration order breaks ties
                best = ArrayTrajectory(array, list(trajectory), c)
        solution[array] = best
    return solution


def greedy_assignment(costs: AssignmentCosts, n_kernels: int) -> Dict[str, ArrayTrajectory]:
    """Greedy baseline: each kernel picks its lowest node-cost layout, paying whatever boundary conversions that implies."""
    costs.check(n_kernels, changes_allowed=True)
    solution: Dict[str, ArrayTrajectory] = {}
    for array, layouts in costs.layouts.items():
        tags = [l.tag for l in layouts]
        trajectory = []
        for k in range(n_kernels):
            trajectory.append(min(tags, key=lambda tag: (costs.node_cost[(array, k, tag)], tags.index(tag))))
        solution[array] = ArrayTrajectory(array, trajectory, trajectory_cost(costs, array, trajectory))
    return solution


@dataclass
class ConflictRow:
    """One array's conflict-report row: per-kernel preferences, chosen trajectory, cost triad, and untrusted (contended) flag."""
    array: str
    per_kernel_preference: List[str]
    conflicting: bool
    chosen: List[str]
    greedy_cost: float
    global_cost: float
    single_layout_cost: float
    untrusted: bool = False


def conflict_report(costs: AssignmentCosts, n_kernels: int) -> List[ConflictRow]:
    """Per-array report: per-kernel preferences, whether they disagree, and the greedy/global/single cost triad."""
    greedy = greedy_assignment(costs, n_kernels)
    dp = per_array_dp(costs, n_kernels, allow_changes=True)
    single = per_array_dp(costs, n_kernels, allow_changes=False)
    rows = []
    for array in sorted(costs.layouts):
        preferences = greedy[array].tags
        chosen = dp[array].tags
        rows.append(
            ConflictRow(array=array,
                        per_kernel_preference=preferences,
                        conflicting=len(set(preferences)) > 1,
                        chosen=chosen,
                        greedy_cost=greedy[array].cost,
                        global_cost=dp[array].cost,
                        single_layout_cost=single[array].cost,
                        untrusted=any((array, k, tag) in costs.untrusted for k, tag in enumerate(chosen))))
    return rows


def format_conflict_report(rows: List[ConflictRow]) -> str:
    lines = [
        f"{'array':<10} {'conflict':<8} {'prefers':<28} {'chosen':<28} "
        f"{'greedy':>12} {'global':>12} {'single':>12}  trust"
    ]
    for r in rows:
        lines.append(f"{r.array:<10} {str(r.conflicting):<8} {'/'.join(r.per_kernel_preference):<28} "
                     f"{'/'.join(r.chosen):<28} {r.greedy_cost:>12.4g} {r.global_cost:>12.4g} "
                     f"{r.single_layout_cost:>12.4g}  {'CONTENDED' if r.untrusted else 'ok'}")
    return "\n".join(lines)


def to_assignment(trajectories: Dict[str, ArrayTrajectory], layouts: Dict[str,
                                                                          List[Layout]]) -> Dict[str, List[Layout]]:
    """Convert trajectories to `apply_assignment` input; drops arrays whose trajectory is all-identity."""
    by_tag = {array: {l.tag: l for l in ls} for array, ls in layouts.items()}
    assignment = {}
    for array, trajectory in trajectories.items():
        if any(tag != layouts[array][0].tag for tag in trajectory.tags):
            assignment[array] = [by_tag[array][tag] for tag in trajectory.tags]
    return assignment
