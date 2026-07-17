# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Global layout assignment over the line graph: per-array Viterbi DP, the brute-force oracle, the
per-op greedy baseline, and the conflict report (GLOBAL_LAYOUT_DESIGN.md, tasks C1 + C2 + C3).

The assignment algorithm is COST-PROVIDER AGNOSTIC: it consumes an :class:`AssignmentCosts` table
(node cost per ``(array, kernel, layout)``, relayout cost per ``(array, from, to)``) and does not
care whether the numbers came from the tier-0/tier-2 cost model or from measured per-nest timings
-- the two ranking modes of the design share this one solver.

Laws encoded here, not in callers:

  * **Identity-first tie-break** -- each array's layout list must enumerate its baseline layout
    first; every argmin uses strict ``<`` scanning in enumeration order, so ties resolve toward the
    earlier (ultimately the identity) candidate. Enumeration order is load-bearing, so it is
    validated, not assumed.
  * **Both edge regimes come free** -- ``allow_changes=False`` is the same DP with infinite edge
    cost on a layout change.
  * **The brute force is capped and loud** -- above ``cap`` trajectories per array it refuses with
    the count, never silently samples.

The separability caveat stands (the DP treats arrays independently; the tier-2 nest cost couples
them through the max): the mandatory mitigation is downstream -- re-score/re-time the COMPOSED
assignment (D3) and compare against :func:`brute_force_trajectories`.
"""
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from dace.transformation.layout.apply_assignment import Layout


@dataclass
class AssignmentCosts:
    """The pluggable cost table the assignment algorithms consume.

    :ivar layouts: per array, its candidate layouts -- the BASELINE (identity) layout FIRST.
    :ivar node_cost: ``(array, kernel_index, layout_tag) -> cost`` -- the array's access cost in
                     that kernel under that layout (others at baseline; one consistent unit).
    :ivar relayout_cost: ``(array, from_tag, to_tag) -> cost`` of converting between two layouts on
                         a boundary (same unit as node costs).
    :ivar entry_conversion_needed: per array, whether ``apply_assignment`` would insert an ENTRY
                                   conversion into a non-identity first segment (the first touching
                                   kernel reads live-in, or its write does not provably cover the
                                   array). Missing key = no entry charge.
    :ivar last_write_kernel: per array, the last kernel writing it (``None``/missing = never
                             written): the trajectory must then return the value to the ORIGINAL
                             descriptor, priced as the EXIT conversion.
    :ivar untrusted: ``(array, kernel, layout_tag)`` node costs whose measurement was CONTENDED
                     (spread above the threshold) -- kept in the table per the measurement
                     protocol, but carried so the conflict report can flag decisions that consumed
                     them. Empty for model-derived tables.
    """
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
                    # Even the single-layout regime pays the entry/exit conversions of a
                    # non-identity layout, so the identity edges must be priced.
                    pairs = [(tags[0], t) for t in tags[1:]] + [(t, tags[0]) for t in tags[1:]]
                else:
                    pairs = []
                for a, b in pairs:
                    if (array, a, b) not in self.relayout_cost:
                        raise ValueError(f"AssignmentCosts: missing relayout cost ({array}, "
                                         f"'{a}' -> '{b}')")


@dataclass
class ArrayTrajectory:
    """One array's solved trajectory: the layout tag per kernel and its total (access + relayout)
    cost."""
    array: str
    tags: List[str]
    cost: float

    def changes(self) -> int:
        return sum(1 for a, b in zip(self.tags, self.tags[1:]) if a != b)


def trajectory_cost(costs: AssignmentCosts, array: str, tags: List[str]) -> float:
    """Total cost of one trajectory: node costs, every boundary change, plus the ENTRY and EXIT
    conversions ``apply_assignment`` actually inserts (the liveness facts of the table):

      * entry -- converting the live-in value into a non-identity first layout;
      * exit  -- returning the last-written value to the original descriptor, priced as a
        surcharge at the last write; the FIRST later switch back to identity is then free (it IS
        the exit conversion, moved onto that boundary), every further switch is real again.

    Approximation stated once: the model prices every tag switch, while the applier skips
    conversions for untouched/aliasing segments and provably-covered first writes -- the optimum is
    unaffected (the skipped shapes have a cost-equal constant twin), the mandatory D3 re-score
    covers the rest."""
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


def per_array_dp(costs: AssignmentCosts, n_kernels: int, allow_changes: bool = True) -> Dict[str, ArrayTrajectory]:
    """The Viterbi DP (C1): per array, the cheapest layout trajectory over the kernel line, under
    the FULL :func:`trajectory_cost` objective (node costs, boundary changes, entry charge, exit
    surcharge with its one free restore switch). ``allow_changes=False`` forbids boundary changes
    (infinite edge cost), yielding the best SINGLE global layout -- still paying its entry/exit
    conversions, so single-vs-trajectory comparisons stay fair. Ties resolve toward the
    earlier-enumerated layout at every step.

    The DP state is ``(layout, restored)`` -- ``restored`` records whether identity was visited at
    or after the last write, which decides whether a later switch back to identity is the (already
    surcharged) exit conversion moved onto that boundary (free, once) or a real conversion."""
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

        # dp[j][flag] = best cost ending at kernel k with layout j; flag = identity visited in
        # tags[lw..k]. None = unreachable state.
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
                            if not allow_changes:
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
                             cap: int = 1_000_000) -> Dict[str, ArrayTrajectory]:
    """The enumeration oracle (C2): every trajectory per array, capped and loud. With additive
    per-array costs the joint optimum decomposes per array, so per-array enumeration IS the oracle
    for the table (the coupling the table cannot see is D3's re-score job)."""
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
            c = trajectory_cost(costs, array, list(trajectory))
            if best is None or c < best.cost:  # strict <: enumeration order breaks ties
                best = ArrayTrajectory(array, list(trajectory), c)
        solution[array] = best
    return solution


def greedy_assignment(costs: AssignmentCosts, n_kernels: int) -> Dict[str, ArrayTrajectory]:
    """The per-op greedy baseline (the k17 antagonist): each kernel independently picks the layout
    with the lowest node cost, then PAYS whatever boundary conversions that implies -- greedy
    optimizes nodes and is blind to edges."""
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
    """One array's conflict-report row: what each kernel wants, what was chosen, and the k17 triad
    of costs (greedy / global DP / no-changes single layout). ``untrusted`` marks a chosen
    trajectory that consumed at least one CONTENDED measurement -- the decision stands, the flag
    travels with it."""
    array: str
    per_kernel_preference: List[str]
    conflicting: bool
    chosen: List[str]
    greedy_cost: float
    global_cost: float
    single_layout_cost: float
    untrusted: bool = False


def conflict_report(costs: AssignmentCosts, n_kernels: int) -> List[ConflictRow]:
    """The C3 report: per array, the per-kernel preferences, whether they disagree, and the
    greedy/global/single-layout cost triad."""
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
    """Convert solved trajectories to ``apply_assignment``'s input: ``{array: [Layout per kernel]}``.
    Arrays whose trajectory is all-identity are dropped (nothing to apply)."""
    by_tag = {array: {l.tag: l for l in ls} for array, ls in layouts.items()}
    assignment = {}
    for array, trajectory in trajectories.items():
        if any(tag != layouts[array][0].tag for tag in trajectory.tags):
            assignment[array] = [by_tag[array][tag] for tag in trajectory.tags]
    return assignment
