# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

from dace import SDFG, SDFGState, symbolic
from dace.ordered import OrderedSet
from dace.sdfg import nodes
from dace.sdfg.graph import Edge
from dace.sdfg.replace import replace_in_codeblock
from dace.sdfg.state import (BreakBlock, ConditionalBlock, ContinueBlock, ControlFlowBlock, ControlFlowRegion,
                             LoopRegion, ReturnBlock)
from dace.utils import find_new_name
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes import analysis as ap


@transformation.explicit_cf_compatible
class StrictSymbolSSA(ppl.ControlFlowRegionPass):
    """
    Perform an SSA transformation on all symbols in the SDFG in a strict manner, i.e., without introducing phi nodes.
    """

    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Symbols | ppl.Modifies.Edges | ppl.Modifies.Nodes | ppl.Modifies.States

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.Symbols | ppl.Modifies.Edges | ppl.Modifies.Nodes | ppl.Modifies.States

    def depends_on(self):
        return [ap.SymbolWriteScopes]

    def apply(self, region, pipeline_results) -> Optional[Dict[str, Set[str]]]:
        """
        Rename symbols in a restricted SSA manner.

        :param region: The control flow region to modify.
        :param pipeline_results: If in the context of a ``Pipeline``, a dictionary that is populated with prior Pass
                                 results as ``{Pass subclass name: returned object from pass}``. If not run in a
                                 pipeline, an empty dictionary is expected.
        :return: A dictionary mapping the original name to a set of all new names created for each symbol.
        """
        results: Dict[str, Set[str]] = defaultdict(lambda: set())
        sdfg = region if isinstance(region, SDFG) else region.sdfg

        symbol_scope_dict: ap.SymbolScopeDict = pipeline_results[ap.SymbolWriteScopes.__name__][region.cfg_id]

        for name, scope_dict in symbol_scope_dict.items():
            # If there is only one scope, don't do anything.
            if len(scope_dict) <= 1:
                continue

            for write, shadowed_reads in scope_dict.items():
                if write is not None:
                    newname = sdfg.find_new_symbol(name)
                    sdfg.symbols[newname] = sdfg.symbols[name]

                    # Replace the write to this symbol with a write to the new symbol.
                    try:
                        write.data.assignments[newname] = write.data.assignments[name]
                        del write.data.assignments[name]
                    except KeyError:
                        # Ignore.
                        pass

                    # Replace all dominated reads.
                    for read in shadowed_reads:
                        if isinstance(read, ControlFlowBlock):
                            read.replace(name, newname)
                        else:
                            if read not in scope_dict:
                                read.data.replace(name, newname)
                            else:
                                read.data.replace(name, newname, replace_keys=False)

                    results[name].add(newname)

        if len(results) == 0:
            return None
        else:
            return results

    def report(self, pass_retval: Any) -> Optional[str]:
        return f'Renamed {len(pass_retval)} symbols: {pass_retval}.'


#: A program point of the flattened control flow: ``(kind, block_or_edge)``.
Point = Tuple[str, Any]


class FlatControlFlow:
    """One SDFG's hierarchical control flow flattened into program points, nested SDFGs excluded.

    ``in``/``out`` bracket every block, ``rin``/``rout`` every region body, ``head``/``latch`` a
    loop's condition and update, and ``edge`` an interstate edge (reads happen before its
    assignments). ``reads`` holds the symbols each point itself reads.
    """

    def __init__(self, sdfg: SDFG):
        self.succ: Dict[Point, List[Point]] = defaultdict(list)
        self.reads: Dict[Point, Set[str]] = {}
        self.entry: Point = ('rin', sdfg)
        self.exit: Point = ('rout', sdfg)
        self.add_region(sdfg, None)

    def link(self, src: Point, dst: Point):
        self.succ[src].append(dst)

    def add_region(self, region: ControlFlowRegion, loop: Optional[LoopRegion]):
        if region.number_of_nodes() == 0:
            self.link(('rin', region), ('rout', region))
            return
        self.link(('rin', region), ('in', region.start_block))
        for block in region.nodes():
            self.add_block(block, loop)
            if region.out_degree(block) == 0 and not isinstance(block, (BreakBlock, ContinueBlock, ReturnBlock)):
                self.link(('out', block), ('rout', region))
        for edge in region.edges():
            self.link(('out', edge.src), ('edge', edge))
            self.link(('edge', edge), ('in', edge.dst))
            self.reads[('edge', edge)] = edge.data.read_symbols()

    def add_block(self, block: ControlFlowBlock, loop: Optional[LoopRegion]):
        entry, leave = ('in', block), ('out', block)
        if isinstance(block, SDFGState):
            self.reads[entry] = block.used_symbols(all_symbols=True)
            self.link(entry, leave)
        elif isinstance(block, BreakBlock):
            self.link(entry, ('out', loop))
        elif isinstance(block, ContinueBlock):
            self.link(entry, ('latch', loop))
        elif isinstance(block, ReturnBlock):
            self.link(entry, self.exit)
        elif isinstance(block, ConditionalBlock):
            self.reads[entry] = {s for cond, _ in block.branches if cond is not None for s in cond.get_free_symbols()}
            for _, branch in block.branches:
                self.link(entry, ('rin', branch))
                self.add_region(branch, loop)
                self.link(('rout', branch), leave)
            if all(cond is not None for cond, _ in block.branches):
                self.link(entry, leave)
        elif isinstance(block, LoopRegion):
            head, latch, body_in, body_out = ('head', block), ('latch', block), ('rin', block), ('rout', block)
            for point, code in ((entry, block.init_statement), (head, block.loop_condition), (latch,
                                                                                              block.update_statement)):
                self.reads[point] = code.get_free_symbols() if code is not None else set()
            if not block.inverted:
                chain = [entry, head, body_in], [body_out, latch, head]
            elif block.update_before_condition:
                chain = [entry, body_in], [body_out, latch, head, body_in]
            else:
                chain = [entry, body_in], [body_out, head, latch, body_in]
            for path in chain:
                for src, dst in zip(path, path[1:]):
                    self.link(src, dst)
            self.link(head, leave)
            self.add_region(block, block)
        else:
            self.link(entry, ('rin', block))
            self.add_region(block, loop)
            self.link(('rout', block), leave)

    def reaching(self, gen: Dict[Point, int], incoming: int) -> Dict[Point, Set[int]]:
        """Definitions live on entry to each point; ``incoming`` stands for the value held at SDFG entry."""
        arriving: Dict[Point, Set[int]] = defaultdict(set)
        arriving[self.entry].add(incoming)
        work = deque([self.entry])
        while work:
            point = work.popleft()
            out = {gen[point]} if point in gen else arriving[point]
            for nxt in self.succ[point]:
                if not out <= arriving[nxt]:
                    arriving[nxt] |= out
                    work.append(nxt)
        return arriving


@transformation.explicit_cf_compatible
class SymbolSSA(ppl.Pass):
    """Give every web of an interstate symbol its own name, so one name has one definition.

    Unrolling replays a body and peeling copies it, and each copy reassigns the same
    frontend-materialized ``idx = arr[k]``. One name carrying several values makes the copies
    look dependent: nothing may be reordered around the chain, and a loop whose peel reuses the
    body's names appears to export them (CloudSC's ``llfall_index_*`` blocked ``MoveLoopIntoMap``).

    Reaching definitions are solved per SDFG over :class:`FlatControlFlow`; a use unites every
    definition reaching it into one web, so a join or a loop-carried value stays one web (no phi)
    under one name. Exactly one defining web keeps the original name: the one merged with the
    incoming value if any, else the last one live at the SDFG exit (the final version of a chain),
    else the first. Every other web is renamed. Loop variables, symbols in descriptor shapes, and
    symbols an edge both assigns and reads in another assignment are left alone.
    """

    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Symbols | ppl.Modifies.Edges | ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & (ppl.Modifies.Symbols | ppl.Modifies.Edges | ppl.Modifies.CFG))

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Set[str]]]:
        """Version every symbol web that can carry its own name.

        :param sdfg: The SDFG to transform in place, nested SDFGs included.
        :param pipeline_results: Prior pass results; unused.
        :returns: ``{original name: {new names}}``, or ``None`` if nothing was renamed.
        """
        results: Dict[str, OrderedSet] = defaultdict(OrderedSet)
        for nested in sdfg.all_sdfgs_recursive():
            for name, new_names in self.version_sdfg(nested).items():
                results[name] |= new_names
        return dict(results) or None

    @staticmethod
    def candidates(sdfg: SDFG) -> Dict[str, List[Edge]]:
        """Symbols assigned on several of ``sdfg``'s own interstate edges that may be renamed."""
        defs: Dict[str, List[Edge]] = defaultdict(list)
        blocked: Set[str] = {str(s) for desc in sdfg.arrays.values() for s in desc.used_symbols(True)}
        for region in sdfg.all_control_flow_regions():
            if isinstance(region, LoopRegion):
                blocked.add(region.loop_variable)
            if isinstance(region, ConditionalBlock):
                continue
            for edge in region.edges():
                assignments = edge.data.assignments
                for name in assignments:
                    defs[name].append(edge)
                    if any(name in symbolic.free_symbols_and_functions(rhs) for other, rhs in assignments.items()
                           if other != name):
                        blocked.add(name)
        return {name: edges for name, edges in defs.items() if len(edges) > 1 and name not in blocked}

    @staticmethod
    def taken_names(sdfg: SDFG) -> Set[str]:
        taken = set(sdfg.arrays) | set(sdfg.symbols) | set(sdfg.constants_prop)
        for region in sdfg.all_control_flow_regions():
            if isinstance(region, LoopRegion):
                taken.add(region.loop_variable)
            if isinstance(region, ConditionalBlock):
                continue
            for edge in region.edges():
                taken |= set(edge.data.assignments)
        for state in sdfg.states():
            for node in state.nodes():
                if isinstance(node, nodes.MapEntry):
                    taken |= set(node.map.params)
        return taken

    def version_sdfg(self, sdfg: SDFG) -> Dict[str, OrderedSet]:
        renamed: Dict[str, OrderedSet] = defaultdict(OrderedSet)
        candidates = self.candidates(sdfg)
        if not candidates:
            return renamed
        flat = FlatControlFlow(sdfg)
        taken = self.taken_names(sdfg)
        for name, defs in candidates.items():
            incoming = len(defs)
            arriving = flat.reaching({('edge', edge): index for index, edge in enumerate(defs)}, incoming)
            parent = list(range(len(defs) + 1))

            def find(i: int) -> int:
                while parent[i] != i:
                    parent[i] = parent[parent[i]]
                    i = parent[i]
                return i

            uses = [point for point, reads in flat.reads.items() if name in reads and arriving[point]]
            for point in uses:
                first, *rest = arriving[point]
                for other in rest:
                    parent[find(other)] = find(first)
            roots = [find(d) for d in range(len(defs))]
            if find(incoming) in roots:
                keeper = find(incoming)
            else:
                live = [roots[d] for d in range(len(defs)) if d in arriving[flat.exit]]
                keeper = live[-1] if live else roots[0]
            fresh: Dict[int, str] = {}
            for root in roots:
                if root != keeper and root not in fresh:
                    fresh[root] = find_new_name(name, taken)
                    taken.add(fresh[root])
                    if name in sdfg.symbols:
                        sdfg.symbols[fresh[root]] = sdfg.symbols[name]
                    renamed[name].add(fresh[root])
            if not fresh:
                continue
            for point in uses:
                new_name = fresh.get(find(next(iter(arriving[point]))))
                if new_name is not None:
                    self.rename_use(point, name, new_name)
            for index, edge in enumerate(defs):
                new_name = fresh.get(find(index))
                if new_name is not None:
                    items = list(edge.data.assignments.items())
                    edge.data.assignments.clear()
                    edge.data.assignments.update((new_name if k == name else k, v) for k, v in items)
        return renamed

    @staticmethod
    def rename_use(point: Point, name: str, new_name: str):
        kind, obj = point
        repl = {name: new_name}
        if kind == 'edge':
            obj.data.replace_dict(repl, replace_keys=False)
        elif isinstance(obj, SDFGState):
            obj.replace_dict(repl)
        elif isinstance(obj, ConditionalBlock):
            for cond, _ in obj.branches:
                if cond is not None:
                    replace_in_codeblock(cond, repl)
        else:
            code = {'in': obj.init_statement, 'head': obj.loop_condition, 'latch': obj.update_statement}[kind]
            replace_in_codeblock(code, repl)


__all__ = ['FlatControlFlow', 'StrictSymbolSSA', 'SymbolSSA']
