# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lift a conditional-append (stream compaction) loop to a parallel scan + scatter."""

import ast
import copy
from typing import Any, NamedTuple

import sympy

import dace
from dace import SDFG, data, dtypes, properties, subsets, symbolic
from dace import memlet as mm
from dace.optionals import required
from dace.sdfg import nodes
from dace.sdfg.narrowing import as_basic, as_expr
from dace.sdfg.state import (
    AbstractControlFlowRegion,
    BreakBlock,
    ConditionalBlock,
    ContinueBlock,
    ControlFlowBlock,
    ControlFlowRegion,
    LoopRegion,
    ReturnBlock,
    SDFGState,
    SymbolResolver,
)
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf
from dace.transformation.passes.analysis import loop_analysis, scopes

#: Prefixes for the transients and symbol this pass introduces.
MASK_PREFIX = "compaction_mask_"
IDX_PREFIX = "compaction_rank_"
TOTAL_PREFIX = "compaction_total_"
BASE_PREFIX = "compaction_base_"

#: Index dtype of the rank buffer, the total and the cursor base symbol. An index stays int64.
COUNT_DTYPE = dtypes.int64

#: The rank buffer and the total narrow to this when every cursor offset ``K * rank`` fits it: half the scan's write and
#: phase 3's read.
NARROW_COUNT_DTYPE = dtypes.int32
NARROW_COUNT_LIMIT = 2**31 - 1

#: Element type of the predicate mask.
MASK_DTYPE = dtypes.int8


class NestLevel(NamedTuple):
    """One unit-stride loop level of the matched compaction nest."""

    loop: LoopRegion
    start: symbolic.SymbolicType
    trip: symbolic.SymbolicType
    preamble: list[ControlFlowBlock]
    child: ControlFlowBlock
    tail: tuple[ControlFlowBlock, ...] = ()


class CompactionMatch(NamedTuple):
    """A matched conditional-append nest and everything the rewrite touches."""

    root: LoopRegion
    levels: tuple[NestLevel, ...]
    parent: AbstractControlFlowRegion
    sdfg: SDFG
    cond_block: ConditionalBlock
    cond_str: str
    cursor: str
    step: symbolic.SymbolicType
    accumulators: frozenset[str] = frozenset()


@properties.make_properties
@xf.explicit_cf_compatible
class LoopToStreamCompaction(ppl.Pass):
    """Lift conditional-append loops to a parallel mask + exclusive scan + guarded scatter."""

    CATEGORY: str = "Optimization Preparation"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> list[type[ppl.Pass] | ppl.Pass]:
        return []

    def apply_pass(self, sdfg: SDFG, _pipeline_results: dict[str, Any]) -> int | None:
        lifted = 0
        # ONE resolver for the whole run; each lift invalidates the SDFG it rewrote.
        resolver = scopes.ScopedSymbolResolver()
        for sd in list(sdfg.all_sdfgs_recursive()):
            for region in list(sd.all_control_flow_regions()):
                if not (isinstance(region, LoopRegion) and region.loop_variable):
                    continue
                if not self.is_attached(region, sd):
                    continue
                m = self.match_loop(region, sd)
                if m is None:
                    continue
                self.rewrite(m, resolver, self.count_width(m))
                lifted += 1
        return lifted or None

    # match

    def match_loop(self, loop: LoopRegion, sdfg: SDFG) -> CompactionMatch | None:
        nest = self.match_nest(loop)
        if nest is None:
            return None  # not a rectangular unit-stride chain of loops down to one guard
        levels, cond_block = nest

        if len(cond_block.branches) != 1 or cond_block.branches[0][0] is None:
            return None  # an else arm appends on a path the mask does not model
        cond_str = cond_block.branches[0][0].as_string.strip()
        branch = cond_block.branches[0][1]

        if not self.body_is_pure(loop, sdfg, levels):
            return None  # break/continue/return, nested SDFG, side effect, WCR or stream

        cursor = self.cursor_increment(loop, branch, levels)
        if cursor is None:
            return None  # not exactly one nest-invariant cursor bump inside the guard
        cursor_name, step = cursor

        if not self.cursor_is_isolated(levels, cursor_name, cond_str):
            return None  # the cursor reaches the bounds / preamble / guard -> mask is not computable

        if self.expression_reads_containers(cond_str, sdfg):
            return None  # the mask tasklet emits the guard as symbol-only code

        for level in levels:
            for blk in level.preamble:
                if isinstance(blk, SDFGState) and self.writes_nontransient(blk, sdfg):
                    return None  # phase 1 and phase 3 both re-run the preamble

        if not self.cursor_arrays_are_isolated(loop, cursor_name, sdfg):
            return None  # in-place compaction / the scatter would race its own input

        if self.cursor_writes_move_with_iteration(loop, levels, cursor_name):
            return None  # e.g. ``b[2*c - i]``: the affine cursor model hides a real collision

        if not self.guard_reads_survive_hoisting(loop, levels, sdfg):
            return None  # hoisting the guard reads ahead of the body writes reorders a dependence

        # Expensive last: model the cursor by an injective affine index and let LoopToMap rule on
        # every dependence this pass does NOT prove itself.
        accumulators = self.affine_cursor_probe(loop, cursor_name, step, levels)
        if accumulators is None:
            return None

        return CompactionMatch(
            root=loop,
            levels=levels,
            parent=loop.parent_graph,
            sdfg=sdfg,
            cond_block=cond_block,
            cond_str=cond_str,
            cursor=cursor_name,
            step=step,
            accumulators=accumulators,
        )

    def is_attached(self, region: AbstractControlFlowRegion, sdfg: SDFG) -> bool:
        """Stale-snapshot guard: an earlier rewrite in this sweep may have detached a whole nest."""
        cur: AbstractControlFlowRegion | None = region
        while cur is not sdfg:
            parent = required(cur).parent_graph
            if parent is None or cur not in parent.nodes():
                return False
            cur = parent
        return True

    def match_nest(self, loop: LoopRegion) -> tuple[tuple[NestLevel, ...], ConditionalBlock] | None:
        """Walk down a rectangular chain of unit-stride loops to the single guard."""
        levels: list[NestLevel] = []
        outer_vars: list[str] = []
        cur = loop
        while True:
            if cur.pinned_sequential:
                return None  # an earlier pass pinned this loop; honour it
            start, trip = self.loop_extent(cur)
            if start is None:
                return None  # unanalyzable bounds or non-unit stride -> mask indexing would be wrong
            if any(name in outer_vars for name in symbolic.symlist([start, trip])):
                return None  # non-rectangular: the level has no constant extent to give mask a shape
            chain = self.body_chain(cur)
            if chain is None:
                return None  # branching loop body -> the guard is not the only append path
            guards = [b for b in chain if isinstance(b, ConditionalBlock)]
            inner = [b for b in chain if isinstance(b, LoopRegion) and b.loop_variable]
            if len(guards) == 1 and not inner:
                child: ControlFlowBlock = guards[0]
            elif not guards and len(inner) == 1:
                child = inner[0]
            else:
                return None  # zero or several guards, or a level that both loops and guards
            pos = chain.index(child)
            tail = tuple(chain[pos + 1 :])
            if isinstance(child, LoopRegion) and not all(
                isinstance(blk, SDFGState) and blk.number_of_nodes() == 0 for blk in tail
            ):
                return None  # phase 1 keeps an outer level's tail and would run its effects twice
            levels.append(NestLevel(loop=cur, start=start, trip=trip, preamble=chain[:pos], child=child, tail=tail))
            # A bound assigned in this level's preamble varies per iteration just as its iterator does
            outer_vars += [cur.loop_variable, *(k for e in cur.edges() for k in e.data.assignments)]
            if isinstance(child, ConditionalBlock):
                return tuple(levels), child
            cur = child

    def nest_index(
        self, levels: tuple[NestLevel, ...], loops: list[LoopRegion], iterator_dtypes: list[dtypes.typeclass]
    ) -> list[symbolic.SymbolicType]:
        """The origin-shifted iteration vector, one component per level, SYMBOLIC."""
        return [
            symbolic.symbol(loop.loop_variable, dtype) - level.start
            for level, loop, dtype in zip(levels, loops, iterator_dtypes)
        ]

    def point_subset(self, index: list[symbolic.SymbolicType]) -> subsets.Range:
        """One point of a shaped buffer. ``Range`` keeps an already-symbolic bound verbatim."""
        return subsets.Range([(comp, comp, 1) for comp in index])

    def loop_extent(self, loop: LoopRegion) -> tuple[symbolic.SymbolicType | None, symbolic.SymbolicType | None]:
        """Return ``(start, trip)`` for a unit-stride loop, ``(None, None)`` otherwise."""
        start = loop_analysis.get_init_assignment(loop)
        end = loop_analysis.get_loop_end(loop)
        stride = loop_analysis.get_loop_stride(loop)
        if start is None or end is None or stride is None:
            return None, None
        if symbolic.simplify(stride - 1) != 0:
            return None, None
        return start, symbolic.simplify(end - start + 1)

    def body_chain(self, loop: LoopRegion) -> list[ControlFlowBlock] | None:
        """Return the loop body as a linear chain of blocks, or ``None`` if it branches."""
        blocks = loop.nodes()
        if not blocks:
            return None
        chain: list[ControlFlowBlock] = [loop.start_block]
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

    def body_is_pure(self, loop: LoopRegion, sdfg: SDFG, levels: tuple[NestLevel, ...]) -> bool:
        """Reject anything phase 1's re-execution or phase 3's reordering could break."""
        for blk in loop.all_control_flow_blocks(recursive=False):
            if isinstance(blk, (BreakBlock, ContinueBlock, ReturnBlock)):
                return False
        rerun = {id(state) for state in self.preamble_states(levels)}
        for state in loop.states():
            for node in state.nodes():
                if isinstance(node, nodes.NestedSDFG):
                    return False  # opaque body; purity and dependences are not analyzable here
                if isinstance(node, nodes.CodeNode) and node.has_side_effects(sdfg):
                    return False
            for edge in state.edges():
                if edge.data.is_empty():
                    continue
                if edge.data.wcr is not None and id(state) in rerun:
                    return False  # phase 1 re-runs the preamble: the accumulation would happen twice
                desc = sdfg.arrays.get(edge.data.data)
                if isinstance(desc, data.Stream):
                    return False  # stream push order is observable
        return True

    def cursor_increment(
        self, loop: LoopRegion, branch: ControlFlowRegion, levels: tuple[NestLevel, ...]
    ) -> tuple[str, symbolic.SymbolicType] | None:
        """Find the unique ``c = c + K`` interstate assignment; it must live inside ``branch``."""
        assigned: dict[str, int] = {}
        bumps: list[tuple[str, symbolic.SymbolicType, bool]] = []
        # all_interstate_edges descends into nested guards and loops
        on_path = self.body_chain(branch) is not None
        branch_edges = {id(e) for e in branch.edges() if on_path and e.data.is_unconditional()}
        for edge in loop.all_interstate_edges(recursive=False):
            inside = id(edge) in branch_edges
            for name, rhs in edge.data.assignments.items():
                assigned[name] = assigned.get(name, 0) + 1
                expr = symbolic.pystr_to_symbolic(rhs)
                sym = symbolic.pystr_to_symbolic(name)
                # membership and the subtraction both go through identity, so equalize first
                expr, sym = symbolic.equalize_symbols_across(expr, sym)
                if sym not in expr.free_symbols:
                    continue
                bumps.append((name, symbolic.simplify(as_expr(expr) - as_expr(sym)), inside))
        if len(bumps) != 1:
            return None  # zero or several carried cursors -> no single closed form
        name, step, inside = bumps[0]
        if not inside:
            return None  # a bump outside the guard advances on non-taken iterations too
        if assigned[name] != 1:
            return None  # the cursor is assigned elsewhere in the loop as well
        if step == 0:
            return None  # a zero bump is not an append cursor
        invariant = {*(level.loop.loop_variable for level in levels), *assigned}
        if any(str(s) in invariant for s in as_basic(step).free_symbols):
            return None  # a data-dependent or iteration-dependent step breaks c_in + K*rank[i]
        return name, step

    def cursor_is_isolated(self, levels: tuple[NestLevel, ...], cursor: str, cond_str: str) -> bool:
        """The mask must be computable without the cursor, at every level of the nest."""
        meta = [cond_str]
        for level in levels:
            meta += [
                level.loop.loop_condition.as_string,
                required(level.loop.init_statement).as_string,
                required(level.loop.update_statement).as_string,
            ]
        for code in meta:
            if cursor in self.expression_names(code):
                return False
        for level in levels:
            for blk in level.preamble:
                if isinstance(blk, SDFGState) and cursor in blk.free_symbols:
                    return False
            for blk in [*level.preamble, level.child]:
                for edge in level.loop.in_edges(blk):
                    for rhs in edge.data.assignments.values():
                        if cursor in self.expression_names(rhs):
                            return False
            # A tail runs on every iteration, after the guard: on a non-taken one the cursor names the slot of
            # the next taken iteration, so a tail access through it would collide with that iteration's append.
            for blk in level.tail:
                if cursor in blk.free_symbols:
                    return False
                for edge in level.loop.in_edges(blk):
                    if cursor in edge.data.used_symbols(all_symbols=True, union_lhs_symbols=True):
                        return False
        return True

    def cursor_arrays_are_isolated(self, loop: LoopRegion, cursor: str, sdfg: SDFG) -> bool:
        """Cursor-indexed arrays must be write-only-through-the-cursor, or read-only."""
        cursor_written, cursor_read, plain_written, read = self.cursor_accesses(loop, cursor, sdfg)
        for name in cursor_written:
            if name in read or name in cursor_read:
                return False  # in-place compaction: the parallel scatter races its own input
            if name in plain_written:
                return False  # a second, non-cursor write to the compaction target
        for name in cursor_read:
            if name in plain_written:
                return False  # the gathered source is rewritten inside the loop
        return True

    def cursor_accesses(
        self, loop: LoopRegion, cursor: str, sdfg: SDFG
    ) -> tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, int]]:
        """Access counts per container in ``loop``: written through ``cursor``, read through it, written
        otherwise, read otherwise."""
        cursor_written: dict[str, int] = {}
        cursor_read: dict[str, int] = {}
        plain_written: dict[str, int] = {}
        read: dict[str, int] = {}
        for state in loop.states():
            for edge in state.edges():
                if edge.data.is_empty() or edge.data.data is None:
                    continue
                # An access-node destination is written at its own side of the memlet, which for a copy between two
                # access nodes is the other subset (``Z[i] -> Z[c]`` writes ``Z[c]``); its source is read at the other.
                accesses = []
                if isinstance(edge.dst, nodes.AccessNode):
                    accesses.append((edge.dst.data, edge.data.get_dst_subset(edge, state), True))
                    if isinstance(edge.src, nodes.AccessNode):
                        accesses.append((edge.src.data, edge.data.get_src_subset(edge, state), False))
                else:
                    accesses.append((edge.data.data, edge.data.subset, False))
                for name, sub, writes in accesses:
                    uses_cursor = cursor in {str(s) for s in required(sub).free_symbols}
                    target = (
                        (cursor_written if uses_cursor else plain_written)
                        if writes
                        else (cursor_read if uses_cursor else read)
                    )
                    target[name] = target.get(name, 0) + 1
        # Interstate edges and branch conditions read data too
        for memlet in self.meta_reads(loop, sdfg):
            uses_cursor = cursor in {str(s) for s in required(memlet.subset).free_symbols}
            target = cursor_read if uses_cursor else read
            target[memlet.data] = target.get(memlet.data, 0) + 1
        return cursor_written, cursor_read, plain_written, read

    def cursor_writes_move_with_iteration(self, loop: LoopRegion, levels: tuple[NestLevel, ...], cursor: str) -> bool:
        """Does a cursor-indexed write also read a symbol that changes from one iteration to the next?"""
        moving = {level.loop.loop_variable for level in levels}
        for edge in loop.all_interstate_edges():
            moving.update(edge.data.assignments.keys())
        moving.discard(cursor)
        for state in loop.states():
            for edge in state.edges():
                if edge.data.is_empty() or edge.data.data is None:
                    continue
                # A copy named after its source writes its destination at the other subset
                if not isinstance(edge.dst, nodes.AccessNode):
                    continue
                written = edge.data.get_dst_subset(edge, state)
                if written is None:
                    continue
                names = {str(sym) for sym in written.free_symbols}
                if cursor in names and names & moving:
                    return True
        return False

    def guard_reads_survive_hoisting(self, loop: LoopRegion, levels: tuple[NestLevel, ...], sdfg: SDFG) -> bool:
        """Phase 1 runs every guard read before every body write; that must not reorder a dependence."""
        guard_reads: dict[str, list[subsets.Subset]] = {}
        for level in levels:
            for blk in [*level.preamble, level.child]:
                for edge in level.loop.in_edges(blk):
                    for read in edge.data.get_read_memlets(sdfg.arrays):
                        guard_reads.setdefault(read.data, []).append(read.subset)
            for blk in level.preamble:
                if not isinstance(blk, SDFGState):
                    continue
                for node in blk.data_nodes():
                    for edge in blk.out_edges(node):
                        if edge.data.is_empty():
                            continue
                        # A copy named after its destination reads its source at the other subset, or all of it
                        read = edge.data.get_src_subset(edge, blk)
                        if read is None:
                            read = subsets.Range.from_array(sdfg.arrays[node.data])
                        guard_reads.setdefault(node.data, []).append(read)
        if not guard_reads:
            return True
        loop_vars = [symbolic.pystr_to_symbolic(level.loop.loop_variable) for level in levels]
        for state in loop.states():
            for edge in state.edges():
                if edge.data.is_empty() or edge.data.data is None:
                    continue
                if not isinstance(edge.dst, nodes.AccessNode):
                    continue
                name = edge.dst.data
                if name not in guard_reads:
                    continue
                written = edge.data.get_dst_subset(edge, state)
                # Reads of an array the body never writes hoist freely, at as many points as they like
                if written is None or any(read != written for read in guard_reads[name]):
                    return False  # the body overwrites an element the guard read at another index
                if not self.subset_is_injective_point(written, loop_vars):
                    return False  # two iterations share the element -> the hoist changes the value read
        return True

    def subset_is_injective_point(self, subset: subsets.Subset, loop_vars: list[symbolic.SymbolicType]) -> bool:
        """A single point that is an injective affine image of the nest's iteration vector."""
        if not isinstance(subset, subsets.Range):
            return False
        carried: list[symbolic.SymbolicType] = []
        for begin, end, step in subset.ranges:
            if symbolic.simplify(begin - end) != 0 or symbolic.simplify(step - 1) != 0:
                return False
            # local, not a rebind of loop_vars: this runs per range, and re-equalizing the outer list
            # against a different begin each time could change which instance is kept.
            begin, *here_vars = symbolic.equalize_symbols_across(begin, *loop_vars)
            here = [var for var in here_vars if var in begin.free_symbols]
            if not here:
                continue
            if len(here) != 1:
                return False  # two iterators in one index -> not separably injective
            lead = begin.coeff(here[0], 1)
            if symbolic.simplify(begin - (lead * here[0] + begin.coeff(here[0], 0))) != 0:
                return False  # non-affine in the loop variable
            if not required(lead).is_Integer or lead == 0:
                return False
            carried.append(here[0])
        # by name: carried holds equalized instances, loop_vars the originals
        return all(sum(1 for var in carried if str(var) == str(outer)) == 1 for outer in loop_vars)

    def affine_cursor_probe(
        self, loop: LoopRegion, cursor: str, step: symbolic.SymbolicType, levels: tuple[NestLevel, ...]
    ) -> frozenset[str] | None:
        """Ask ``LoopToMap`` about the residual body with the cursor modelled as an injective affine index."""
        from dace.transformation.interstate.loop_to_map import LoopLift, Refusal
        from dace.transformation.passes.canonicalize.carried_accumulation import all_convertible
        from dace.transformation.passes.loop_to_reduce import only_read_modify_written
        from dace.transformation.passes.parallelization.parallelize_loops import ParallelizeLoops

        outer = levels[0]
        closed = {cursor: step * (symbolic.pystr_to_symbolic(outer.loop.loop_variable) - outer.start)}
        probe = LoopLift()
        while not ParallelizeLoops().accepts(loop, probe, closed_forms=closed):
            refusal = probe.last_refusal
            name = None if refusal is None else refusal.container
            if (
                name is None
                or name in probe.assume_wcr
                or refusal.kind is not Refusal.NON_UNIQUE_WRITE
                or any(node.data == name for state in self.preamble_states(levels) for node in state.data_nodes())
                or not only_read_modify_written(loop, name)
                or not all_convertible(loop, {name})
            ):
                return None
            probe.assume_wcr = probe.assume_wcr | {name}
        return probe.assume_wcr

    def preamble_states(self, levels: tuple[NestLevel, ...]) -> list[SDFGState]:
        """The states of every level's preamble: what phase 1 runs as well as phase 3."""
        return [
            state
            for level in levels
            for blk in level.preamble
            for state in ([blk] if isinstance(blk, SDFGState) else blk.states())
        ]

    # match helpers

    def meta_reads(self, loop: LoopRegion, sdfg: SDFG) -> list[mm.Memlet]:
        """Every data read outside dataflow: interstate assignments/conditions and branch guards."""
        out: list[mm.Memlet] = []
        for region in loop.all_control_flow_regions():
            for edge in region.edges():
                out.extend(edge.data.get_read_memlets(sdfg.arrays))
            if isinstance(region, ConditionalBlock):
                out.extend(region.get_meta_read_memlets(sdfg.arrays))
        return out

    def expression_names(self, code: str) -> list[str]:
        """Every bare identifier in a Python expression / statement."""
        if not code:
            return []
        return [n.id for n in ast.walk(ast.parse(code, mode="exec")) if isinstance(n, ast.Name)]

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

    # rewrite

    def rank_shape(self, m: CompactionMatch) -> list[symbolic.SymbolicType]:
        """One extent per level; a level past its end runs nothing, but an array sized by its bare count would be negative."""
        facts = SymbolResolver().facts_at(m.parent)
        return [
            level.trip if symbolic.provably_nonnegative(level.trip, facts) else sympy.Max(0, level.trip)
            for level in m.levels
        ]

    def count_width(self, m: CompactionMatch) -> dtypes.typeclass:
        """The rank dtype: 32-bit when the reach ``K * iterations`` provably fits it, else 64-bit. One nest either
        way, never a runtime choice between two copies."""
        reach = m.step * sympy.Mul(*self.rank_shape(m))
        fits = symbolic.provably_le(reach, NARROW_COUNT_LIMIT, SymbolResolver().facts_at(m.parent))
        return NARROW_COUNT_DTYPE if fits else COUNT_DTYPE

    def rewrite(
        self, m: CompactionMatch, resolver: scopes.ScopedSymbolResolver, count_dtype: dtypes.typeclass = COUNT_DTYPE
    ) -> None:
        from dace.transformation.passes.canonicalize.carried_accumulation import convert_updates

        sdfg, parent, root = m.sdfg, m.parent, m.root
        # Read before anything is added: the phase copies rename each level's iterator, and a
        # rename must carry the dtype the ORIGINAL level declares, level by level.
        iterator_dtypes = [self.iterator_dtype(level.loop, sdfg, resolver) for level in m.levels]
        shape = self.rank_shape(m)
        targets = frozenset(self.cursor_accesses(root, m.cursor, sdfg)[0])
        mask_name, _ = sdfg.add_array(MASK_PREFIX + root.label, shape, MASK_DTYPE, transient=True, find_new_name=True)
        idx_name, _ = sdfg.add_array(IDX_PREFIX + root.label, shape, count_dtype, transient=True, find_new_name=True)
        total_name, _ = sdfg.add_scalar(TOTAL_PREFIX + root.label, count_dtype, transient=True, find_new_name=True)
        base_sym = BASE_PREFIX + root.label
        while base_sym in sdfg.symbols:
            base_sym += "_"
        sdfg.add_symbol(base_sym, sdfg.symbols.get(m.cursor, COUNT_DTYPE))

        in_edges = list(parent.in_edges(root))
        out_edges = list(parent.out_edges(root))
        was_start = parent.start_block is root

        entry = parent.add_state(root.label + "_compaction_entry", is_start_block=was_start)
        mask_nest = self.clone_loop(m, "_compaction_mask")
        scan = parent.add_state(root.label + "_compaction_scan")
        scatter_nest = self.clone_loop(m, "_compaction_scatter")
        exit_state = parent.add_state(root.label + "_compaction_exit")
        self.build_mask_loop(m, mask_nest, mask_name, iterator_dtypes)
        self.build_scatter_loop(m, scatter_nest, idx_name, base_sym, iterator_dtypes)

        parent.add_edge(entry, mask_nest, dace.InterstateEdge(assignments={base_sym: m.cursor}))
        parent.add_edge(mask_nest, scan, dace.InterstateEdge())
        parent.add_edge(scan, scatter_nest, dace.InterstateEdge())
        live_out = f"({base_sym}) + ({symbolic.symstr(m.step)}) * {total_name}"
        parent.add_edge(scatter_nest, exit_state, dace.InterstateEdge(assignments={m.cursor: live_out}))

        self.emit_scan_and_count(scan, mask_name, idx_name, total_name, shape)

        for edge in in_edges:
            parent.remove_edge(edge)
            parent.add_edge(edge.src, entry, edge.data)
        for edge in out_edges:
            parent.remove_edge(edge)
            parent.add_edge(exit_state, edge.dst, edge.data)
        parent.remove_node(root)

        self.parallelize(mask_nest, sdfg)
        # The updates the probe took as WCR; only phase 3 runs them (phase 1 dropped them with the guard and tail).
        for name in sorted(m.accumulators):
            convert_updates(scatter_nest, name)
        self.parallelize(scatter_nest, sdfg, targets)
        # Symbols, descriptors, regions and interstate edges all changed
        resolver.invalidate_sdfg(sdfg)

    def clone_loop(self, m: CompactionMatch, suffix: str) -> LoopRegion:
        """Copy the matched nest into the parent region; ownership is what makes it editable."""
        loop = copy.deepcopy(m.root)
        loop.label = m.root.label + suffix
        m.parent.add_node(loop, ensure_unique_name=True)
        own_scratch(loop, m.root, m.sdfg)
        return loop

    def clone_path(self, m: CompactionMatch, root: LoopRegion) -> tuple[list[LoopRegion], ControlFlowBlock]:
        """Re-find the nest levels and the guard inside a clone -- identity does not survive the copy."""
        loops = [root]
        while len(loops) < len(m.levels):
            loops.append(next(b for b in loops[-1].nodes() if isinstance(b, LoopRegion) and b.loop_variable))
        guard = next(b for b in loops[-1].nodes() if b.label == m.cond_block.label)
        return loops, guard

    def iterator_dtype(self, loop: LoopRegion, sdfg: SDFG, resolver: scopes.ScopedSymbolResolver) -> dtypes.typeclass:
        """The dtype ``loop``'s iterator is declared with."""
        body = next(iter(loop.states()), None)
        if body is not None:
            scoped = resolver.tabulate(body)[None].get(loop.loop_variable)
            if scoped is not None:
                return scoped
        return resolver.resolve_dtype(loop.loop_variable, sdfg)

    def rename_iterator(self, loop: LoopRegion, sdfg: SDFG, suffix: str, dtype: dtypes.typeclass) -> str:
        """Give a phase copy its own iterator: LoopToMap refuses a counter another block still names."""
        new_name = f"compaction_it_{suffix}"
        while new_name in sdfg.symbols or new_name in sdfg.arrays:
            new_name += "_"
        repl = {loop.loop_variable: new_name}
        loop.replace_meta_accesses(repl)
        loop.replace_dict(repl)
        loop.loop_variable = new_name
        return new_name

    def build_mask_loop(
        self, m: CompactionMatch, root: LoopRegion, mask_name: str, iterator_dtypes: list[dtypes.typeclass]
    ) -> None:
        """Phase 1: the nest with its guard replaced by a store of the guard's value."""
        loops, guard = self.clone_path(m, root)
        for loop, dtype in zip(loops, iterator_dtypes):
            self.rename_iterator(loop, m.sdfg, loop.label, dtype)
        # The clone's guard carries the renamed iterators; ``m.cond_str`` still names the original ones.
        cond_str = guard.branches[0][0].as_string.strip()
        inner = loops[-1]
        store = inner.add_state(root.label + "_store", is_start_block=inner.start_block is guard)
        for edge in list(inner.in_edges(guard)):
            inner.remove_edge(edge)
            inner.add_edge(edge.src, store, edge.data)
        for block in [b for b in inner.nodes() if b is not store and self.reaches(inner, guard, b)]:
            inner.remove_node(block)
        index = self.nest_index(m.levels, loops, iterator_dtypes)
        tasklet = store.add_tasklet(root.label + "_mask", {}, {"__out"}, f"__out = ({cond_str})")
        write = store.add_write(mask_name)
        store.add_edge(tasklet, "__out", write, None, mm.Memlet(data=mask_name, subset=self.point_subset(index)))

    def build_scatter_loop(
        self,
        m: CompactionMatch,
        root: LoopRegion,
        idx_name: str,
        base_sym: str,
        iterator_dtypes: list[dtypes.typeclass],
    ) -> None:
        """Phase 3: the nest verbatim, with the cursor rebound from the scan on the guard's in-edge."""
        loops, guard = self.clone_path(m, root)
        for loop, dtype in zip(loops, iterator_dtypes):
            self.rename_iterator(loop, m.sdfg, loop.label, dtype)
        inner = loops[-1]
        # An interstate assignment IS text
        index = ", ".join(symbolic.symstr(comp) for comp in self.nest_index(m.levels, loops, iterator_dtypes))
        binding = f"({base_sym}) + ({symbolic.symstr(m.step)}) * {idx_name}[{index}]"
        edges = inner.in_edges(guard)
        if not edges:
            # No preamble: the guard is the body's first block, so there is no edge to bind on.
            bind = inner.add_state(root.label + "_bind", is_start_block=True)
            inner.add_edge(bind, guard, dace.InterstateEdge(assignments={m.cursor: binding}))
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

    def emit_scan_and_count(
        self, state: SDFGState, mask: str, rank: str, total: str, shape: list[symbolic.SymbolicType]
    ) -> None:
        """Phase 2: ``rank = exclusive_scan(mask)`` and ``total = sum(mask)``, off one mask read."""
        from dace.libraries.standard.nodes.reduce import Reduce
        from dace.libraries.standard.nodes.scan import INPUT_CONNECTOR_NAME, OUTPUT_CONNECTOR_NAME, Scan, ScanOp

        read = state.add_read(mask)
        span = subsets.Range([(0, extent - 1, 1) for extent in shape])

        scan = Scan(name=state.label + "_scan", op=ScanOp.SUM, exclusive=True, identity=0)
        state.add_node(scan)
        state.add_edge(read, None, scan, INPUT_CONNECTOR_NAME, mm.Memlet(data=mask, subset=copy.deepcopy(span)))
        state.add_edge(
            scan, OUTPUT_CONNECTOR_NAME, state.add_write(rank), None, mm.Memlet(data=rank, subset=copy.deepcopy(span))
        )

        count = Reduce(name=state.label + "_count", wcr="lambda a, b: a + b", axes=list(range(len(shape))), identity=0)
        count.add_in_connector("_in")
        count.add_out_connector("_out")
        state.add_node(count)
        state.add_edge(read, None, count, "_in", mm.Memlet(data=mask, subset=copy.deepcopy(span)))
        state.add_edge(
            count, "_out", state.add_write(total), None, mm.Memlet(data=total, subset=subsets.Range([(0, 0, 1)]))
        )

    def parallelize(self, loop: LoopRegion, sdfg: SDFG, targets: frozenset[str] = frozenset()) -> None:
        """Best effort -- the three-phase form is correct whether or not the phases become Maps."""
        from dace.transformation.passes.parallelization.parallelize_loops import ParallelizeLoops

        parallelize = ParallelizeLoops()
        parallelize.expand_buffers = True  # a scalar each iteration writes before it reads becomes private
        parallelize.parallelize_loop(sdfg, loop, contract_arrays=targets)


def own_scratch(copy_: LoopRegion, original: LoopRegion, sdfg: SDFG) -> None:
    """Give ``copy_``, a deep copy of ``original`` placed in the graph, its own copy of every transient that only
    ``original`` accesses."""
    inside = {node.data for state in original.states() for node in state.data_nodes()}
    outside: set[str] = set()
    for state in sdfg.states():
        if not within(state, original) and not within(state, copy_):
            outside |= {node.data for node in state.data_nodes()}
    for region in sdfg.all_control_flow_regions():
        if all(region is not mine and not within(region, mine) for mine in (original, copy_)):
            for edge in region.edges():
                outside |= edge.data.free_symbols
    names = sorted(
        name
        for name in inside - outside
        if sdfg.arrays[name].transient and not isinstance(sdfg.arrays[name], data.View)
    )
    renames = {name: sdfg.add_datadesc(name, copy.deepcopy(sdfg.arrays[name]), find_new_name=True) for name in names}
    if renames:
        copy_.replace_dict(renames)


def within(block: ControlFlowBlock, region: AbstractControlFlowRegion) -> bool:
    """Whether ``region`` encloses ``block`` in ``block``'s own SDFG."""
    parent = block.parent_graph
    while isinstance(parent, AbstractControlFlowRegion) and not isinstance(parent, SDFG):
        if parent is region:
            return True
        parent = parent.parent_graph
    return False


__all__ = ["LoopToStreamCompaction"]
