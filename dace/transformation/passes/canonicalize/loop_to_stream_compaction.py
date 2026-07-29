# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lift a conditional-append (stream compaction) loop to a parallel scan + scatter.

The sequential shape -- TSVC ``s341`` / ``s342`` / ``s343`` -- carries an output
cursor across iterations and bumps it only on the taken branch::

    j = -1
    for i in range(N):
        if b[i] > 0.0:
            j = j + 1
            a[j] = b[i]

``j`` is the ONLY loop-carried value, so ``LoopToMap`` sees a carried scalar plus
a data-dependent write subset ``a[j]`` and refuses; the loop stays sequential.
Neither ``LoopToScan`` (the carried value is not the written array -- it is the
write INDEX) nor ``LoopToReduce`` / ``LiftLoopCarriedReduction`` (there is no
associative combine on an accumulator) claims this shape, and
``ScatterToGuardedMaps`` only guards an idx array the loop already READS.

The lift replaces the cursor with its closed form. Let ``mask[i] = cond(i)`` and
``rank[i] = mask[0] + ... + mask[i-1]`` (the EXCLUSIVE prefix sum). The value of
the cursor on entry to iteration ``i`` is exactly ``c_in + K * rank[i]`` for a
per-taken-iteration increment ``K`` -- a closed form with no carry -- so every
iteration can compute its own cursor and the guarded writes land on slots that
are DISJOINT BY CONSTRUCTION (``rank`` is strictly increasing on taken
iterations, and a non-taken iteration writes nothing).

Soundness -- the lift is value-preserving when, for the loop ``L``, the guard
``cond``, the branch body ``B`` and the cursor ``c``:

* ``c`` is bumped by exactly ONE interstate assignment ``c = c + K`` in all of
  ``L``, that assignment lives inside ``B``, and ``K`` is loop-invariant. Two
  bumps, a bump outside the guard, or a data-dependent ``K`` all break
  ``cursor(i) = c_in + K * rank[i]``;
* ``c`` is free of the loop bounds, of the preamble that computes ``cond``, and
  of ``cond`` itself -- otherwise phase 1 cannot compute ``mask`` without
  already knowing the cursor;
* every array written through ``c`` is never read in ``L`` (an in-place
  compaction ``Z[c] = Z[i]`` aliases its own input: sequentially ``c <= i`` so
  it is safe, in parallel it is a read/write race) and is written ONLY through
  ``c``; every array read through ``c`` is never written in ``L``;
* an array the guard reads and the body writes is touched at ONE point subset,
  identical on both sides and injective-affine in the loop variable, so
  hoisting all guard reads (phase 1) ahead of all body writes (phase 3) cannot
  reorder a real dependence;
* the residual body -- with the cursor modelled by an injective affine index --
  passes ``LoopToMap.can_be_applied`` non-permissively. This pass owns the proof
  that the CURSOR-indexed accesses are disjoint; ``LoopToMap`` owns everything
  else, so no dependence is waived by assumption.

Rewrite (``T`` = trip count; ``mask`` / ``rank`` are fresh ``T``-element
transients and ``total`` a fresh scalar -- each with exactly ONE writer, which
is what keeps state fusion from splitting a buffer across two access nodes and
leaving the reader on the wrong one)::

    BEFORE                                    AFTER

    -- {c: INIT} -->                          -- {c_in: c} -->
    for i in start:end                        for i in start:end          (phase 1, parallel)
    |  preamble  -- {s: b[i]} -->             |  preamble -- {s: b[i]} -->
    +- if (cond(s))                           +- mask[i - start] = cond(s)
       +- -- {c: c + K} -->                   rank  = exclusive_scan(mask) (phase 2, Scan)
          a[c] = f(i)                         total = sum(mask)            (phase 2, Reduce)
                                              for i in start:end          (phase 3, parallel)
                                              |  preamble -- {s: b[i]} -->
                                              +- -- {c: c_in + K*rank[i - start]} -->
                                                 if (cond(s))
                                                 +- -- {c: c + K} -->
                                                    a[c] = f(i)
                                              -- {c: c_in + K*total} -->

Phase 3 keeps the branch body VERBATIM, including the ``c = c + K`` bump:
rebinding ``c`` on the edge into the guard gives every iteration the right
entry value, and the bump then reproduces the original in-iteration value at
every use, whether the source bumped before or after the writes. ``total``
comes from a ``Reduce``, not from a spare scan slot, so a zero-trip loop reads
nothing out of bounds and publishes the untouched ``c_in``.

Algorithm (pseudocode)::

    for loop in every LoopRegion:
        m = match(loop)                       # unit stride, one guard, one cursor bump,
        if m is None: continue                # purity, alias and dependence gates, probe
        mask, rank, total = fresh transients
        emit phase 1 loop  (preamble copy + mask store)
        emit Scan(SUM, exclusive, identity 0) and Reduce(+, identity 0) off one mask read
        emit phase 3 loop  (loop copy + cursor rebind on the guard in-edge)
        publish c = c_in + K * total
        force LoopToMap on phase 1 and phase 3 (the disjointness proof is ours)

Refusals -- each names the miscompile it prevents:

* Non-unit stride / unanalyzable bounds -- ``rank`` is indexed by ``i - start``
  with unit steps; any other stride mis-indexes ``mask``.
* Guard is not a single one-armed ``ConditionalBlock``, or the body is not a
  linear chain -- ``rank`` counts ONE guard; an ``else`` arm or a second guard
  would append on a path the mask does not model.
* Zero or several cursor bumps, a bump outside the guard, a bump by a
  data-dependent amount -- the closed form ``c_in + K * rank[i]`` is wrong.
* The cursor reaches the loop bounds, the preamble or the guard -- phase 1
  would need the cursor it is being computed to replace.
* ``break`` / ``continue`` / ``return``, a nested SDFG, a side-effecting
  tasklet, a WCR write, or a stream inside the loop -- phase 1 re-executes the
  preamble and phase 3 re-executes the body, so any effect would run twice or
  out of order.
* The guard reads a data container directly instead of an interstate-bound
  symbol -- the mask tasklet emits the guard as symbol-only code; a bare array
  name would be a dangling read.
* The preamble writes a non-transient array -- phase 1 and phase 3 both run it.
* An array written through the cursor is read anywhere in the loop (in-place
  compaction), or written by anything but the cursor; an array read through the
  cursor is written in the loop -- the scatter would race against its own
  input.
* A guard-read array is written by the body at anything other than the single
  identical injective point subset -- hoisting the guard reads would then see a
  value the sequential loop had already overwritten.
* The affine-cursor probe fails ``LoopToMap.can_be_applied`` -- the residual
  body carries a dependence this pass does not model.
"""

import ast
import copy
from typing import Dict, List, NamedTuple, Optional, Tuple

import dace
from dace import SDFG, data, dtypes, memlet as mm, properties, subsets, symbolic
from dace.sdfg import nodes
from dace.sdfg.state import (AbstractControlFlowRegion, BreakBlock, ConditionalBlock, ContinueBlock, ControlFlowBlock,
                             ControlFlowRegion, LoopRegion, ReturnBlock, SDFGState)
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf
from dace.transformation.passes.analysis import loop_analysis

#: Prefixes for the transients and symbol this pass introduces.
MASK_PREFIX = 'compaction_mask_'
IDX_PREFIX = 'compaction_rank_'
TOTAL_PREFIX = 'compaction_total_'
BASE_PREFIX = 'compaction_base_'

#: Index dtype of the mask / rank buffers.
COUNT_DTYPE = dtypes.int64


class CompactionMatch(NamedTuple):
    """A matched conditional-append loop and everything the rewrite touches.

    :param loop: The compaction loop itself.
    :param parent: ``loop.parent_graph`` -- where the phase chain is spliced in.
    :param sdfg: The SDFG owning ``loop``, where the transients are declared.
    :param cond_block: The single one-armed guard inside the loop body.
    :param cond_str: The guard condition, as a symbol-only Python expression.
    :param cursor: Name of the loop-carried output cursor symbol.
    :param step: The per-taken-iteration cursor increment ``K`` (loop-invariant).
    :param start: First value of the loop variable.
    :param trip: Number of iterations, ``end - start + 1``.
    """

    loop: LoopRegion
    parent: AbstractControlFlowRegion
    sdfg: SDFG
    cond_block: ConditionalBlock
    cond_str: str
    cursor: str
    step: symbolic.SymbolicType
    start: symbolic.SymbolicType
    trip: symbolic.SymbolicType


@properties.make_properties
@xf.explicit_cf_compatible
class LoopToStreamCompaction(ppl.Pass):
    """Lift conditional-append loops to a parallel mask + exclusive scan + guarded scatter."""

    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return {}

    def apply_pass(self, sdfg: SDFG, _pipeline_results) -> Optional[int]:
        lifted = 0
        for sd in list(sdfg.all_sdfgs_recursive()):
            for region in list(sd.all_control_flow_regions()):
                if not (isinstance(region, LoopRegion) and region.loop_variable):
                    continue
                # Stale-snapshot guard: an earlier rewrite in this sweep may have removed the block.
                if region.parent_graph is None or region not in region.parent_graph.nodes():
                    continue
                m = self.match_loop(region, sd)
                if m is None:
                    continue
                self.rewrite(m)
                lifted += 1
        return lifted or None

    # ----------------------------------- match -----------------------------------

    def match_loop(self, loop: LoopRegion, sdfg: SDFG) -> Optional[CompactionMatch]:
        if loop.pinned_sequential:
            return None  # an earlier pass pinned this loop; honour it
        start, trip = self.loop_extent(loop)
        if start is None:
            return None  # unanalyzable bounds or non-unit stride -> mask indexing would be wrong

        chain = self.body_chain(loop)
        if chain is None:
            return None  # branching loop body -> the guard is not the only append path
        guards = [b for b in chain if isinstance(b, ConditionalBlock)]
        if len(guards) != 1:
            return None  # zero or several guards -> rank counts exactly one
        cond_block = guards[0]
        if len(cond_block.branches) != 1 or cond_block.branches[0][0] is None:
            return None  # an else arm appends on a path the mask does not model
        cond_str = cond_block.branches[0][0].as_string.strip()
        branch = cond_block.branches[0][1]

        pos = chain.index(cond_block)
        preamble = chain[:pos]
        for blk in chain[pos + 1:]:
            if not (isinstance(blk, SDFGState) and blk.number_of_nodes() == 0):
                return None  # a non-empty tail would run before phase 3's guard on the copy

        if not self.body_is_pure(loop, sdfg):
            return None  # break/continue/return, nested SDFG, side effect, WCR or stream

        cursor = self.cursor_increment(loop, branch)
        if cursor is None:
            return None  # not exactly one loop-invariant cursor bump inside the guard
        cursor_name, step = cursor

        if not self.cursor_is_isolated(loop, cond_block, preamble, cursor_name, cond_str):
            return None  # the cursor reaches the bounds / preamble / guard -> mask is not computable

        if self.expression_reads_containers(cond_str, sdfg):
            return None  # the mask tasklet emits the guard as symbol-only code

        for blk in preamble:
            if isinstance(blk, SDFGState) and self.writes_nontransient(blk, sdfg):
                return None  # phase 1 and phase 3 both re-run the preamble

        if not self.cursor_arrays_are_isolated(loop, cursor_name, sdfg):
            return None  # in-place compaction / the scatter would race its own input

        if not self.guard_reads_survive_hoisting(loop, cond_block, preamble, sdfg):
            return None  # hoisting the guard reads ahead of the body writes reorders a dependence

        # Expensive last: model the cursor by an injective affine index and let LoopToMap rule on
        # every dependence this pass does NOT prove itself.
        if not self.affine_cursor_probe(loop, sdfg, cursor_name, step, start):
            return None

        return CompactionMatch(loop=loop,
                               parent=loop.parent_graph,
                               sdfg=sdfg,
                               cond_block=cond_block,
                               cond_str=cond_str,
                               cursor=cursor_name,
                               step=step,
                               start=start,
                               trip=trip)

    def loop_extent(self, loop: LoopRegion) -> Tuple[Optional[symbolic.SymbolicType], Optional[symbolic.SymbolicType]]:
        """Return ``(start, trip)`` for a unit-stride loop, ``(None, None)`` otherwise."""
        start = loop_analysis.get_init_assignment(loop)
        end = loop_analysis.get_loop_end(loop)
        stride = loop_analysis.get_loop_stride(loop)
        if start is None or end is None or stride is None:
            return None, None
        if symbolic.simplify(stride - 1) != 0:
            return None, None
        return start, symbolic.simplify(end - start + 1)

    def body_chain(self, loop: LoopRegion) -> Optional[List[ControlFlowBlock]]:
        """Return the loop body as a linear chain of blocks, or ``None`` if it branches."""
        blocks = loop.nodes()
        if not blocks:
            return None
        chain: List[ControlFlowBlock] = [loop.start_block]
        seen = {id(loop.start_block)}
        while True:
            out = loop.out_edges(chain[-1])
            if not out:
                break
            if len(out) != 1 or len(loop.in_edges(out[0].dst)) != 1:
                return None  # a fork or a join -> not a single path through the body
            nxt = out[0].dst
            if id(nxt) in seen:
                return None  # a cycle inside the body region
            seen.add(id(nxt))
            chain.append(nxt)
        if len(chain) != len(blocks):
            return None  # unreachable blocks -> the chain does not describe the whole body
        return chain

    def body_is_pure(self, loop: LoopRegion, sdfg: SDFG) -> bool:
        """Reject anything phase 1's re-execution or phase 3's reordering could break."""
        for blk in loop.all_control_flow_blocks(recursive=False):
            if isinstance(blk, (BreakBlock, ContinueBlock, ReturnBlock)):
                return False
        for state in loop.all_states():
            for node in state.nodes():
                if isinstance(node, nodes.NestedSDFG):
                    return False  # opaque body; purity and dependences are not analyzable here
                if isinstance(node, nodes.CodeNode) and node.has_side_effects(sdfg):
                    return False
            for edge in state.edges():
                if edge.data.is_empty():
                    continue
                if edge.data.wcr is not None:
                    return False  # a conflict resolution is an accumulation, not an append
                desc = sdfg.arrays.get(edge.data.data)
                if isinstance(desc, data.Stream):
                    return False  # stream push order is observable
        return True

    def cursor_increment(self, loop: LoopRegion,
                         branch: ControlFlowRegion) -> Optional[Tuple[str, symbolic.SymbolicType]]:
        """Find the unique ``c = c + K`` interstate assignment; it must live inside ``branch``."""
        assigned: Dict[str, int] = {}
        bumps: List[Tuple[str, symbolic.SymbolicType, bool]] = []
        branch_edges = {id(e) for e in branch.all_interstate_edges(recursive=False)}
        for edge in loop.all_interstate_edges(recursive=False):
            inside = id(edge) in branch_edges
            for name, rhs in edge.data.assignments.items():
                assigned[name] = assigned.get(name, 0) + 1
                expr = symbolic.pystr_to_symbolic(rhs)
                sym = symbolic.pystr_to_symbolic(name)
                if sym not in expr.free_symbols:
                    continue
                bumps.append((name, symbolic.simplify(expr - sym), inside))
        if len(bumps) != 1:
            return None  # zero or several carried cursors -> no single closed form
        name, step, inside = bumps[0]
        if not inside:
            return None  # a bump outside the guard advances on non-taken iterations too
        if assigned[name] != 1:
            return None  # the cursor is assigned elsewhere in the loop as well
        if step == 0:
            return None  # a zero bump is not an append cursor
        invariant = {loop.loop_variable, *assigned}
        if any(str(s) in invariant for s in step.free_symbols):
            return None  # a data-dependent or iteration-dependent step breaks c_in + K*rank[i]
        return name, step

    def cursor_is_isolated(self, loop: LoopRegion, cond_block: ConditionalBlock, preamble: List[ControlFlowBlock],
                           cursor: str, cond_str: str) -> bool:
        """The mask must be computable without the cursor."""
        meta = [loop.loop_condition.as_string, loop.init_statement.as_string, loop.update_statement.as_string, cond_str]
        for code in meta:
            if cursor in self.expression_names(code):
                return False
        for blk in preamble:
            if isinstance(blk, SDFGState) and cursor in blk.free_symbols:
                return False
        for blk in [*preamble, cond_block]:
            for edge in loop.in_edges(blk):
                for rhs in edge.data.assignments.values():
                    if cursor in self.expression_names(rhs):
                        return False
        return True

    def cursor_arrays_are_isolated(self, loop: LoopRegion, cursor: str, sdfg: SDFG) -> bool:
        """Cursor-indexed arrays must be write-only-through-the-cursor, or read-only."""
        cursor_written: Dict[str, int] = {}
        cursor_read: Dict[str, int] = {}
        plain_written: Dict[str, int] = {}
        read: Dict[str, int] = {}
        for state in loop.all_states():
            for edge in state.edges():
                if edge.data.is_empty() or edge.data.data is None:
                    continue
                name = edge.data.data
                uses_cursor = cursor in {str(s) for s in edge.data.subset.free_symbols}
                writes = isinstance(edge.dst, nodes.AccessNode) and edge.dst.data == name
                target = (cursor_written if uses_cursor else plain_written) if writes else (
                    cursor_read if uses_cursor else read)
                target[name] = target.get(name, 0) + 1
        # Interstate edges and branch conditions read data too; a compaction whose source read is
        # bound to a symbol (``t = Z[i]``) aliases its target exactly as a dataflow read would.
        for memlet in self.meta_reads(loop, sdfg):
            uses_cursor = cursor in {str(s) for s in memlet.subset.free_symbols}
            target = cursor_read if uses_cursor else read
            target[memlet.data] = target.get(memlet.data, 0) + 1
        for name in cursor_written:
            if name in read or name in cursor_read:
                return False  # in-place compaction: the parallel scatter races its own input
            if name in plain_written:
                return False  # a second, non-cursor write to the compaction target
        for name in cursor_read:
            if name in plain_written:
                return False  # the gathered source is rewritten inside the loop
        return True

    def guard_reads_survive_hoisting(self, loop: LoopRegion, cond_block: ConditionalBlock,
                                     preamble: List[ControlFlowBlock], sdfg: SDFG) -> bool:
        """Phase 1 runs every guard read before every body write; that must not reorder a dependence."""
        guard_reads: Dict[str, subsets.Subset] = {}
        for blk in [*preamble, cond_block]:
            for edge in loop.in_edges(blk):
                for read in edge.data.get_read_memlets(sdfg.arrays):
                    if read.data in guard_reads and guard_reads[read.data] != read.subset:
                        return False  # two distinct guard reads of one array -> no single point
                    guard_reads[read.data] = read.subset
        for blk in preamble:
            if not isinstance(blk, SDFGState):
                continue
            for node in blk.data_nodes():
                if blk.out_degree(node) == 0:
                    continue
                for edge in blk.out_edges(node):
                    if edge.data.is_empty() or edge.data.data != node.data:
                        continue
                    if node.data in guard_reads and guard_reads[node.data] != edge.data.subset:
                        return False
                    guard_reads[node.data] = edge.data.subset
        if not guard_reads:
            return True
        loop_var = symbolic.pystr_to_symbolic(loop.loop_variable)
        for state in loop.all_states():
            for edge in state.edges():
                if edge.data.is_empty() or edge.data.data is None:
                    continue
                if not (isinstance(edge.dst, nodes.AccessNode) and edge.dst.data == edge.data.data):
                    continue
                name = edge.data.data
                if name not in guard_reads:
                    continue
                if edge.data.subset != guard_reads[name]:
                    return False  # the body overwrites an element the guard read at another index
                if not self.subset_is_injective_point(edge.data.subset, loop_var):
                    return False  # two iterations share the element -> the hoist changes the value read
        return True

    def subset_is_injective_point(self, subset: subsets.Subset, loop_var: symbolic.SymbolicType) -> bool:
        """A single point whose index is ``a * loop_var + b`` with ``a != 0`` in exactly one dimension."""
        if not isinstance(subset, subsets.Range):
            return False
        carrying = 0
        for begin, end, step in subset.ranges:
            if symbolic.simplify(begin - end) != 0 or symbolic.simplify(step - 1) != 0:
                return False
            if loop_var not in begin.free_symbols:
                continue
            lead = begin.coeff(loop_var, 1)
            if symbolic.simplify(begin - (lead * loop_var + begin.coeff(loop_var, 0))) != 0:
                return False  # non-affine in the loop variable
            if not lead.is_Integer or lead == 0:
                return False
            carrying += 1
        return carrying == 1

    def affine_cursor_probe(self, loop: LoopRegion, sdfg: SDFG, cursor: str, step: symbolic.SymbolicType,
                            start: symbolic.SymbolicType) -> bool:
        """Ask ``LoopToMap`` about the residual body with the cursor modelled as an injective affine index.

        The real phase-3 cursor ``c_in + K * rank[i]`` hits a SUBSET of the elements the affine model
        ``K * (i - start)`` hits, and only on taken iterations, so a disjointness verdict on the model
        carries over. Everything the cursor does not index is judged by ``LoopToMap`` itself, without
        the permissive waiver.
        """
        from dace.transformation.interstate.loop_to_map import LoopToMap
        probe_sdfg = copy.deepcopy(sdfg)
        probe_loop = self.locate_corresponding(probe_sdfg, loop)
        if probe_loop is None:
            return False
        model = symbolic.symstr(step * (symbolic.pystr_to_symbolic(loop.loop_variable) - start))
        for edge in probe_loop.all_interstate_edges(recursive=False):
            if cursor in edge.data.assignments:
                del edge.data.assignments[cursor]
        # Substitute, do not rebind: LoopToMap reads subsets, not interstate symbol definitions.
        probe_loop.replace_dict({cursor: model})
        instance = LoopToMap()
        instance.loop = probe_loop
        return bool(instance.can_be_applied(probe_loop.parent_graph, 0, probe_sdfg, permissive=False))

    def locate_corresponding(self, target: SDFG, loop: LoopRegion) -> Optional[LoopRegion]:
        """Re-find a loop across a deepcopy -- object identity does not survive it.

        Labels are unique per region but not across nested SDFGs, so the iterator and the body
        size have to agree too; an ambiguous lookup would probe the wrong loop.
        """
        found: List[LoopRegion] = []
        for region in target.all_control_flow_regions():
            if (isinstance(region, LoopRegion) and region.label == loop.label
                    and region.loop_variable == loop.loop_variable
                    and region.number_of_nodes() == loop.number_of_nodes()):
                found.append(region)
        return found[0] if len(found) == 1 else None

    # -------------------------------- match helpers --------------------------------

    def meta_reads(self, loop: LoopRegion, sdfg: SDFG) -> List[mm.Memlet]:
        """Every data read outside dataflow: interstate assignments/conditions and branch guards."""
        out: List[mm.Memlet] = []
        for region in loop.all_control_flow_regions():
            for edge in region.edges():
                out.extend(edge.data.get_read_memlets(sdfg.arrays))
            if isinstance(region, ConditionalBlock):
                out.extend(region.get_meta_read_memlets(sdfg.arrays))
        return out

    def expression_names(self, code: str) -> List[str]:
        """Every bare identifier in a Python expression / statement."""
        if not code:
            return []
        return [n.id for n in ast.walk(ast.parse(code, mode='exec')) if isinstance(n, ast.Name)]

    def expression_reads_containers(self, code: str, sdfg: SDFG) -> bool:
        return any(name in sdfg.arrays for name in self.expression_names(code))

    def writes_nontransient(self, state: SDFGState, sdfg: SDFG) -> bool:
        for node in state.data_nodes():
            if state.in_degree(node) == 0:
                continue
            desc = sdfg.arrays.get(node.data)
            if desc is not None and not desc.transient:
                return True
        return False

    # ---------------------------------- rewrite ----------------------------------

    def rewrite(self, m: CompactionMatch) -> None:
        sdfg, parent, loop = m.sdfg, m.parent, m.loop
        mask_name, _ = sdfg.add_array(MASK_PREFIX + loop.label, [m.trip],
                                      COUNT_DTYPE,
                                      transient=True,
                                      find_new_name=True)
        idx_name, _ = sdfg.add_array(IDX_PREFIX + loop.label, [m.trip], COUNT_DTYPE, transient=True, find_new_name=True)
        total_name, _ = sdfg.add_scalar(TOTAL_PREFIX + loop.label, COUNT_DTYPE, transient=True, find_new_name=True)
        base_sym = BASE_PREFIX + loop.label
        while base_sym in sdfg.symbols:
            base_sym += '_'
        sdfg.add_symbol(base_sym, sdfg.symbols.get(m.cursor, COUNT_DTYPE))

        in_edges = list(parent.in_edges(loop))
        out_edges = list(parent.out_edges(loop))
        was_start = parent.start_block is loop

        entry = parent.add_state(loop.label + '_compaction_entry', is_start_block=was_start)
        mask_loop = self.clone_loop(m, '_compaction_mask')
        scan = parent.add_state(loop.label + '_compaction_scan')
        scatter_loop = self.clone_loop(m, '_compaction_scatter')
        exit_state = parent.add_state(loop.label + '_compaction_exit')
        self.build_mask_loop(m, mask_loop, mask_name)
        self.build_scatter_loop(m, scatter_loop, idx_name, base_sym)

        parent.add_edge(entry, mask_loop, dace.InterstateEdge(assignments={base_sym: m.cursor}))
        parent.add_edge(mask_loop, scan, dace.InterstateEdge())
        parent.add_edge(scan, scatter_loop, dace.InterstateEdge())
        live_out = f'({base_sym}) + ({symbolic.symstr(m.step)}) * {total_name}'
        parent.add_edge(scatter_loop, exit_state, dace.InterstateEdge(assignments={m.cursor: live_out}))

        self.emit_scan_and_count(scan, mask_name, idx_name, total_name, m.trip)

        for edge in in_edges:
            parent.remove_edge(edge)
            parent.add_edge(edge.src, entry, edge.data)
        for edge in out_edges:
            parent.remove_edge(edge)
            parent.add_edge(exit_state, edge.dst, edge.data)
        parent.remove_node(loop)
        sdfg.reset_cfg_list()

        self.parallelize(mask_loop, sdfg, permissive=False)
        self.parallelize(scatter_loop, sdfg, permissive=True)

    def clone_loop(self, m: CompactionMatch, suffix: str) -> LoopRegion:
        """Copy the matched loop into the parent region; ownership is what makes it editable."""
        loop = copy.deepcopy(m.loop)
        loop.label = m.loop.label + suffix
        m.parent.add_node(loop, ensure_unique_name=True)
        return loop

    def rename_iterator(self, loop: LoopRegion, sdfg: SDFG, suffix: str) -> str:
        """Give a phase copy its own iterator: LoopToMap refuses a counter another block still names."""
        new_name = f'compaction_it_{suffix}'
        while new_name in sdfg.symbols or new_name in sdfg.arrays:
            new_name += '_'
        sdfg.add_symbol(new_name, dtypes.int64)
        repl = {loop.loop_variable: new_name}
        loop.replace_meta_accesses(repl)
        loop.replace_dict(repl)
        loop.loop_variable = new_name
        return new_name

    def build_mask_loop(self, m: CompactionMatch, loop: LoopRegion, mask_name: str) -> None:
        """Phase 1: the loop with its guard replaced by a store of the guard's value."""
        itervar = self.rename_iterator(loop, m.sdfg, loop.label)
        guard = next(b for b in loop.nodes() if b.label == m.cond_block.label)
        store = loop.add_state(loop.label + '_store', is_start_block=loop.start_block is guard)
        for edge in list(loop.in_edges(guard)):
            loop.remove_edge(edge)
            loop.add_edge(edge.src, store, edge.data)
        for block in [b for b in loop.nodes() if b is not store and self.reaches(loop, guard, b)]:
            loop.remove_node(block)
        index = f'({itervar}) - ({symbolic.symstr(m.start)})'
        tasklet = store.add_tasklet(loop.label + '_mask', {}, {'__out'}, f'__out = ({m.cond_str})')
        write = store.add_write(mask_name)
        store.add_edge(tasklet, '__out', write, None,
                       mm.Memlet(data=mask_name, subset=subsets.Range([(index, index, 1)])))

    def build_scatter_loop(self, m: CompactionMatch, loop: LoopRegion, idx_name: str, base_sym: str) -> None:
        """Phase 3: the loop verbatim, with the cursor rebound from the scan on the guard's in-edge."""
        itervar = self.rename_iterator(loop, m.sdfg, loop.label)
        guard = next(b for b in loop.nodes() if b.label == m.cond_block.label)
        index = f'({itervar}) - ({symbolic.symstr(m.start)})'
        binding = f'({base_sym}) + ({symbolic.symstr(m.step)}) * {idx_name}[{index}]'
        edges = loop.in_edges(guard)
        if not edges:
            # No preamble: the guard is the body's first block, so there is no edge to bind on.
            # Dropping the rebinding would leave the cursor at its stale carried value.
            bind = loop.add_state(loop.label + '_bind', is_start_block=True)
            loop.add_edge(bind, guard, dace.InterstateEdge(assignments={m.cursor: binding}))
            return
        for edge in edges:
            edge.data.assignments[m.cursor] = binding

    def reaches(self, region: AbstractControlFlowRegion, src: ControlFlowBlock, dst: ControlFlowBlock) -> bool:
        """True if ``dst`` is ``src`` or lies downstream of it in ``region``."""
        frontier = [src]
        seen = {id(src)}
        while frontier:
            block = frontier.pop()
            if block is dst:
                return True
            for edge in region.out_edges(block):
                if id(edge.dst) not in seen:
                    seen.add(id(edge.dst))
                    frontier.append(edge.dst)
        return False

    def emit_scan_and_count(self, state: SDFGState, mask: str, rank: str, total: str,
                            trip: symbolic.SymbolicType) -> None:
        """Phase 2: ``rank = exclusive_scan(mask)`` and ``total = sum(mask)``, off one mask read.

        Every buffer here has exactly ONE writer. A partial second write to ``mask`` -- padding a
        spare slot to carry the total, say -- survives this pass but not state fusion, which splits
        the two writers onto two access nodes and leaves the reader on the wrong one.
        """
        from dace.libraries.standard.nodes.reduce import Reduce
        from dace.libraries.standard.nodes.scan import (INPUT_CONNECTOR_NAME, OUTPUT_CONNECTOR_NAME, Scan, ScanOp)
        read = state.add_read(mask)
        span = subsets.Range([(0, trip - 1, 1)])

        scan = Scan(name=state.label + '_scan', op=ScanOp.SUM, exclusive=True, identity=0)
        state.add_node(scan)
        state.add_edge(read, None, scan, INPUT_CONNECTOR_NAME, mm.Memlet(data=mask, subset=copy.deepcopy(span)))
        state.add_edge(scan, OUTPUT_CONNECTOR_NAME, state.add_write(rank), None,
                       mm.Memlet(data=rank, subset=copy.deepcopy(span)))

        count = Reduce(name=state.label + '_count', wcr='lambda a, b: a + b', axes=[0], identity=0)
        count.add_in_connector('_in')
        count.add_out_connector('_out')
        state.add_node(count)
        state.add_edge(read, None, count, '_in', mm.Memlet(data=mask, subset=copy.deepcopy(span)))
        state.add_edge(count, '_out', state.add_write(total), None,
                       mm.Memlet(data=total, subset=subsets.Range([(0, 0, 1)])))

    def parallelize(self, loop: LoopRegion, sdfg: SDFG, permissive: bool) -> None:
        """Best effort -- the three-phase form is correct whether or not the phases become Maps.

        ``permissive`` is the cursor-disjointness proof this pass owns and ``LoopToMap`` cannot
        reconstruct: phase 3 writes through ``rank``, which is strictly increasing on the taken
        iterations. Phase 1 needs no waiver, so it is lifted strictly.
        """
        from dace.transformation.interstate.loop_to_map import LoopToMap
        instance = LoopToMap()
        instance.loop = loop
        parent = loop.parent_graph
        if not instance.can_be_applied(parent, 0, sdfg, permissive=permissive):
            return
        instance.apply(parent, sdfg)


__all__ = ['LoopToStreamCompaction']
