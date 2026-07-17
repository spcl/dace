# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Apply a chosen global layout ASSIGNMENT -- one layout trajectory per array -- end to end
(GLOBAL_LAYOUT_DESIGN.md, task A5; the audit blocker: without this, the greedy-vs-global figure
cannot be measured).

An assignment gives every array one :class:`Layout` per kernel of the line graph. Consecutive equal
layouts form a SEGMENT. The planner walks the segments carrying the LIVE holder -- the descriptor
that currently materializes the array's value: untouched segments stay unmaterialized, a touched
segment whose layout matches the live holder's ALIASES onto it, and every other touched segment
materializes a holder (a transient clone ``B__seg1_perm10``, or the original for identity), gets
its kernel states rewritten onto it (the shared ``rewrite_state_for_permute`` core, with
``PermuteDimensions``' copy-retranspose bookkeeping), and is wired with ``LayoutChange``
conversions:

  * entry conversion  -- chained from the LIVE holder, decided at the segment's first TOUCHING
    kernel; skipped ONLY on proof that this kernel fully produces the array before any read
    (``writes_cover_array``; a partial write, a WCR accumulation, or a live-in read all pre-fill
    the holder, since converting redundantly is correct and skipping wrongly is a miscompile);
  * exit conversion   -- inserted when the ORIGINAL array does not hold the post-last-write value
    at program exit: the last write landed in a clone and no later entry conversion restored the
    original. The program interface stays logical and bit-exact against the untransformed program.

v1 applies PERMUTE trajectories (the k17 conflict class); a Block op in a trajectory is refused
loudly -- blocked layouts are still scored/timed per nest on externalized copies, and the global
application of blocked trajectories is the documented extension.
"""
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import dace
from dace import SDFG
from dace.libraries.layout import add_layout_change
from dace.libraries.layout.algebra import Permute, simplify_ops
from dace.sdfg import nodes
from dace.transformation.layout.line_graph import KernelState, check_kernel_per_state
from dace.transformation.layout.permute_dimensions import (note_copy_side, retranspose_copies,
                                                           rewrite_state_for_permute, spanned_dims)


@dataclass(frozen=True)
class Layout:
    """One layout in a trajectory: a stable tag (candidate identity) plus the layout-algebra op
    sequence from the packed-C identity to this layout."""
    tag: str
    ops: Tuple = ()

    @property
    def is_identity(self) -> bool:
        return len(self.ops) == 0


IDENTITY_LAYOUT = Layout("identity", ())


def composed_permutation(ops, ndim: int) -> List[int]:
    """The single axis permutation an op sequence amounts to (``new[i] = old[perm[i]]``).
    Refuses any non-Permute op -- v1 trajectories are permute-only."""
    perm = list(range(ndim))
    for op in ops:
        if not isinstance(op, Permute):
            raise NotImplementedError(f"apply_assignment: v1 applies Permute trajectories only, got "
                                      f"{type(op).__name__} (blocked trajectories are deferred)")
        if len(op.perm) != ndim:
            raise ValueError(f"apply_assignment: Permute{op.perm} does not match rank {ndim}")
        perm = [perm[op.perm[i]] for i in range(ndim)]
    return perm


def segments_of(trajectory: List[Layout]) -> List[Tuple[int, int, Layout]]:
    """Runs of equal layout: ``[(first_kernel, last_kernel_exclusive, layout), ...]``."""
    segments = []
    start = 0
    for k in range(1, len(trajectory) + 1):
        if k == len(trajectory) or trajectory[k].tag != trajectory[start].tag:
            segments.append((start, k, trajectory[start]))
            start = k
    return segments


def reads_before_write(state: dace.SDFGState, array: str) -> bool:
    """True iff ``state`` reads ``array``: through a SOURCE access node (data live-in), or through a
    WCR write -- WCR reads-modifies the destination, so the segment needs the live-in values even
    when the access node has only in-edges. The wcr sits on the INNER memlet-tree edge in canonical
    output (the outer MapExit->AccessNode edge has ``wcr=None``), so the whole state's edges are
    scanned."""
    if any(node.data == array and state.in_degree(node) == 0 for node in state.data_nodes()):
        return True
    return any(e.data is not None and e.data.data == array and e.data.wcr is not None for e in state.edges())


def state_touches(state: dace.SDFGState, array: str) -> bool:
    """True iff ``state`` has an access node for ``array``."""
    return any(node.data == array for node in state.data_nodes())


def writes_cover_array(state: dace.SDFGState, array: str) -> bool:
    """Conservative PROOF that ``state`` writes EVERY element of ``array`` (so a segment starting
    here may skip its entry conversion). Returns False whenever coverage is unprovable -- a
    redundant entry conversion is always semantically correct, a skipped necessary one is a silent
    miscompile.

    The proof: the array's single sink access node is fed by one top-level ``MapExit``, and some
    INNER write memlet (no WCR) indexes a distinct map parameter per array dimension with each
    parameter's map range spanning that dimension's full extent. The propagated OUTER memlet is
    deliberately not consulted: for a partial writer it is over-approximated to the full array, so
    any coverage test on it is unsound."""
    desc = state.sdfg.arrays[array]
    sinks = [n for n in state.data_nodes() if n.data == array and state.in_degree(n) > 0]
    if len(sinks) != 1:
        return False
    edges_in = state.in_edges(sinks[0])
    if len(edges_in) != 1 or not isinstance(edges_in[0].src, nodes.MapExit):
        return False
    exit_node = edges_in[0].src
    if state.scope_dict()[sinks[0]] is not None:  # the sink (and so the map) must be top-level
        return False
    param_ranges = dict(zip(exit_node.map.params, exit_node.map.range.ranges))
    for leaf in state.memlet_tree(edges_in[0]).leaves():
        memlet = leaf.data
        if (memlet is None or memlet.wcr is not None or not isinstance(memlet.subset, dace.subsets.Range)
                or len(memlet.subset.ranges) != len(desc.shape)):
            continue
        used = set()
        proven = True
        for d, (begin, end, _) in enumerate(memlet.subset.ranges):
            if dace.symbolic.simplify(end - begin) != 0:
                proven = False
                break
            param = str(begin)
            if param not in param_ranges or param in used:
                proven = False
                break
            used.add(param)
            range_begin, range_end, range_step = param_ranges[param]
            if (dace.symbolic.simplify(range_begin) != 0 or dace.symbolic.simplify(range_step - 1) != 0
                    or dace.symbolic.simplify(range_end - (desc.shape[d] - 1)) != 0):
                proven = False
                break
        if proven:
            return True
    return False


def refuse_interstate_references(sdfg: SDFG, arrays) -> None:
    """v1 rewrites kernel states only; an interstate edge mentioning a reassigned array would keep
    reading the ORIGINAL layout silently -- refuse loudly instead."""
    for edge in sdfg.all_interstate_edges():
        text = "; ".join([f"{k} = {v}" for k, v in edge.data.assignments.items()] + [edge.data.condition.as_string])
        for array in arrays:
            if re.search(rf"\b{re.escape(array)}\b", text):
                raise NotImplementedError(f"apply_assignment: interstate edge references array "
                                          f"'{array}' ({text!r}); the v1 segment rewrite does not "
                                          f"cover interstate edges")


@dataclass
class AppliedAssignment:
    """What the application did: per array its segment names, plus the inserted conversion states."""
    segment_names: Dict[str, List[str]]
    boundary_states: List[dace.SDFGState]
    exit_state: Optional[dace.SDFGState]


def apply_assignment(sdfg: SDFG, kernels: List[KernelState], assignment: Dict[str, List[Layout]]) -> AppliedAssignment:
    """Apply one layout trajectory per array across the line graph, with paid conversions on the
    boundaries. The SDFG is modified in place; the program interface stays logical (segment clones
    are transient, originals untouched at entry/exit).

    :param sdfg: the kernel-per-state line-graph SDFG (the A6 invariant is re-checked).
    :param kernels: ``line_graph(sdfg)``'s kernel list (positions = trajectory indices).
    :param assignment: ``{array: [Layout per kernel]}``; arrays not mentioned keep packed-C.
    :return: the :class:`AppliedAssignment` summary.
    """
    check_kernel_per_state(sdfg)
    refuse_interstate_references(sdfg, [a for a, traj in assignment.items() if any(not l.is_identity for l in traj)])
    for array, trajectory in assignment.items():
        if array not in sdfg.arrays:
            raise ValueError(f"apply_assignment: unknown array '{array}'")
        if len(trajectory) != len(kernels):
            raise ValueError(f"apply_assignment: trajectory for '{array}' has {len(trajectory)} "
                             f"entries for {len(kernels)} kernels")

    # Plan first (liveness reads pre-rewrite state), then rewrite, then insert conversion states.
    # boundary_changes[kernel_index] = {in_name: (out_name, delta_ops)}
    boundary_changes: Dict[int, Dict[str, Tuple[str, List]]] = {}
    exit_changes: Dict[str, Tuple[str, List]] = {}
    segment_names: Dict[str, List[str]] = {}
    rewrites: List[Tuple[int, str, str, List[int]]] = []  # (kernel_index, array, seg_name, perm)

    for array in sorted(assignment):
        trajectory = assignment[array]
        desc = sdfg.arrays[array]
        ndim = len(desc.shape)
        segments = segments_of(trajectory)

        # Walk the segments carrying the LIVE holder -- the (name, ops) that currently materializes
        # the array's value. Untouched segments stay unmaterialized (no clone, no conversion, the
        # value keeps living where it was); a touched segment whose layout equals the live holder's
        # ALIASES onto it (the tag differs, the physical layout does not -- e.g. perm10 segments
        # separated by an untouched identity run); everything else materializes a holder and chains
        # its entry conversion from the LIVE holder, decided at the segment's first TOUCHING kernel.
        live_name, live_ops = array, []
        holders: List[Tuple[str, List]] = []  # per segment: the holder materializing the value
        entry_targets: List[Tuple[int, str]] = []  # (kernel_position, out_name) of planned entries
        for si, (start, end, seg_layout) in enumerate(segments):
            touched = [k for k in range(start, end) if state_touches(kernels[k].state, array)]
            if not touched:
                holders.append((live_name, live_ops))
                continue
            delta = simplify_ops([op.inverse() for op in reversed(live_ops)] + list(seg_layout.ops))
            if not delta:
                # Same physical layout as the live holder: rewrite onto it, no conversion, no clone.
                name, ops = live_name, live_ops
            else:
                name = array if seg_layout.is_identity else f"{array}__seg{si}_{seg_layout.tag}"
                ops = list(seg_layout.ops)
                if not seg_layout.is_identity:
                    perm = composed_permutation(ops, ndim)
                    sdfg.add_array(name=name,
                                   shape=[desc.shape[perm[i]] for i in range(ndim)],
                                   dtype=desc.dtype,
                                   storage=desc.storage,
                                   transient=True,
                                   lifetime=desc.lifetime,
                                   find_new_name=False)
                # The entry conversion may only be skipped on PROOF that the segment's first
                # touching kernel fully produces the array before any read; anything weaker
                # (partial write, WCR, live-in read) pre-fills the holder from the live one.
                first_touch = kernels[touched[0]].state
                if reads_before_write(first_touch, array) or not writes_cover_array(first_touch, array):
                    boundary_changes.setdefault(start, {})[live_name] = (name, delta)
                    entry_targets.append((start, name))
            holders.append((name, ops))
            live_name, live_ops = name, ops
            if name != array:
                perm = composed_permutation(ops, ndim)
                for k in touched:
                    rewrites.append((k, array, name, perm))
        segment_names[array] = [holder_name for holder_name, _ in holders]

        last_write = max((k.index for k in kernels if any(node.data == array and k.state.in_degree(node) > 0
                                                          for node in k.state.data_nodes())),
                         default=None)

        # Exit conversion: needed when the ORIGINAL array does not hold the post-last-write value at
        # program exit -- the last write landed in a clone AND no later entry conversion restored the
        # original (any entry with out_name == array past the last write copies the value back; no
        # write can follow it, or last_write would be larger).
        if last_write is not None and not desc.transient:
            si_lw = next(si for si, (start, end, _) in enumerate(segments) if start <= last_write < end)
            holder_name, holder_ops = holders[si_lw]
            restored = any(pos > last_write and out_name == array for pos, out_name in entry_targets)
            if holder_name != array and not restored:
                exit_changes[holder_name] = (array, simplify_ops([op.inverse() for op in reversed(holder_ops)]))

    rewrites_by_state: Dict[int, List[Tuple[str, str, List[int]]]] = {}
    for kernel_index, array, seg_name, perm in rewrites:
        rewrites_by_state.setdefault(kernel_index, []).append((array, seg_name, perm))
    for kernel_index in sorted(rewrites_by_state):
        state = kernels[kernel_index].state
        arrays_here = {array for array, _, _ in rewrites_by_state[kernel_index]}
        for node in state.nodes():
            if isinstance(node, nodes.NestedSDFG):
                # A unit-element edge (scalar slice) is layout-transparent: permuting the outer
                # subset is the complete rewrite. A SPANNING edge hands the nested SDFG a >=1-D
                # window whose inner descriptor keeps the old dimension order -- silently wrong on
                # equal-extent shapes -- so it is refused (recursion is the documented extension).
                carried = sorted({
                    e.data.data
                    for e in state.all_edges(node)
                    if e.data is not None and e.data.data in arrays_here and spanned_dims(e.data) > 0
                })
                if carried:
                    raise NotImplementedError(
                        f"apply_assignment: kernel state '{state.label}' passes reassigned "
                        f"array(s) {carried} into a NestedSDFG through a spanning memlet; the v1 "
                        f"segment rewrite does not recurse into nested SDFGs (deferred) -- expand "
                        f"the nest first or drop the array from the assignment.")
        # A copy with ONE relaid operand becomes transposing; the shared PermuteDimensions
        # bookkeeping converts it to a TensorTranspose (or refuses sub-region copies loudly).
        # Sides are merged across this state's per-array rewrites so a copy whose two operands are
        # both reassigned is judged on both permutations.
        sides: Dict = {}
        for array, seg_name, perm in rewrites_by_state[kernel_index]:
            noted = rewrite_state_for_permute(state, {array: seg_name}, {array: perm}, note_copy_side)
            for copy_node, side in noted.items():
                sides.setdefault(copy_node, {}).update(side)
        retranspose_copies(state, sides, context="apply_assignment")

    boundary_states = []
    for kernel_index in sorted(boundary_changes):
        target = kernels[kernel_index].state
        boundary = sdfg.add_state_before(target,
                                         label=f"relayout_before_{target.label}",
                                         is_start_block=(sdfg.start_block is target))
        for in_name in sorted(boundary_changes[kernel_index]):
            out_name, delta = boundary_changes[kernel_index][in_name]
            add_layout_change(sdfg, boundary, in_name, out_name, delta, create_output=False)
        boundary_states.append(boundary)

    exit_state = None
    if exit_changes:
        exit_state = sdfg.add_state_after(kernels[-1].state, label="relayout_exit")
        for in_name in sorted(exit_changes):
            out_name, delta = exit_changes[in_name]
            add_layout_change(sdfg, exit_state, in_name, out_name, delta, create_output=False)

    return AppliedAssignment(segment_names=segment_names, boundary_states=boundary_states, exit_state=exit_state)
