# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from dace import SDFG
from dace.ordered import OrderedSet
from dace.sdfg.state import ConditionalBlock, ControlFlowBlock, ControlFlowRegion
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


@transformation.explicit_cf_compatible
class SymbolSSA(ppl.Pass):
    """Give each interstate-edge definition of a repeatedly-assigned symbol its own name.

    Short-loop unrolling replays a body ``N`` times, and every replay reassigns the same
    frontend-materialized index symbol on the edge feeding its copy::

        s0 -[idx = arr[0]]-> s1 -[idx = arr[1]]-> s2 -[idx = arr[2]]-> s3

    One name carrying ``N`` values makes the chain position-dependent: nothing may be reordered,
    hoisted, or re-guarded around it, which is what stops ``MoveIfIntoLoop`` from distributing a
    guard over such a chain. Versioning the definitions removes the false dependence.

    Strict SSA, no phi nodes: a definition is renamed only where the renaming is unambiguous.
    Reaching definitions are solved per :class:`ControlFlowRegion` over that region's own block
    graph -- ``ControlFlowBlock.replace`` already rewrites a whole subtree, nested SDFG symbol
    mappings included, so the CFG is the natural granularity and no whole-SDFG walk is needed.

    Two things block a definition from being renamed, and both are decided without looking outside
    the region:

    - **Ambiguity.** A use reached by more than one definition would need a phi, so every
      definition reaching it keeps the original name.
    - **Escape.** A definition still live at the region's exit may be read by anything downstream,
      so it keeps the original name. Blocking every exit-live definition is what makes the pass
      sound without a whole-SDFG liveness scan; it costs at most the last version of a chain.

    Distinct from :class:`StrictSymbolSSA`, which drives the ``SymbolWriteScopes`` analysis and
    only supports state-machine SDFGs.
    """

    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Symbols | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & (ppl.Modifies.Symbols | ppl.Modifies.Edges | ppl.Modifies.CFG))

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Set[str]]]:
        """Version every unambiguously-renameable symbol definition in ``sdfg``.

        :param sdfg: The SDFG to transform in place.
        :param pipeline_results: Prior pass results; unused, this pass runs its own analysis.
        :returns: ``{original name: {new names}}``, or ``None`` if nothing was renamed.
        """
        results: Dict[str, Set[str]] = defaultdict(set)
        for region in [sdfg, *sdfg.all_control_flow_regions(recursive=True)]:
            if isinstance(region, ConditionalBlock):
                continue  # holds branches, not a block graph of its own
            for name, new_names in self.version_region(region).items():
                results[name] |= new_names
        return dict(results) or None

    def version_region(self, region: ControlFlowRegion) -> Dict[str, Set[str]]:
        """Rename what can be renamed among ``region``'s own interstate-edge definitions."""
        renamed: Dict[str, Set[str]] = defaultdict(set)
        edges = region.edges()
        defs_by_symbol: Dict[str, List[Any]] = defaultdict(list)
        for edge in edges:
            for name in edge.data.assignments:
                defs_by_symbol[name].append(edge)

        owner = region if isinstance(region, SDFG) else region.sdfg
        for name, defs in defs_by_symbol.items():
            if len(defs) < 2:
                continue  # a single definition is already its own version
            reaching = self.reaching_definitions(region, name)
            blocked = OrderedSet()

            # A use reached by several definitions cannot be renamed to any one of them.
            uses: List[Tuple[Any, OrderedSet]] = []
            for block in region.nodes():
                if name in block.used_symbols(all_symbols=True):
                    uses.append((block, reaching[block]))
            for edge in edges:
                if name in edge.data.free_symbols:
                    uses.append((edge, reaching[edge.src]))
            for _use, arriving in uses:
                if len(arriving) > 1:
                    blocked |= {d for d in arriving if d is not None}

            # A definition live where control leaves the region may be read downstream.
            sinks = [b for b in region.nodes() if region.out_degree(b) == 0]
            if not sinks:
                continue  # no identifiable exit, so no definition can be proven local
            for sink in sinks:
                blocked |= {d for d in reaching[sink] if d is not None}

            fresh: Dict[Any, str] = {}
            for edge in defs:
                if edge in blocked:
                    continue
                new_name = owner.find_new_symbol(name)
                if name in owner.symbols:
                    owner.symbols[new_name] = owner.symbols[name]
                fresh[edge] = new_name
                renamed[name].add(new_name)
            if not fresh:
                continue

            for edge, new_name in fresh.items():
                edge.data.assignments[new_name] = edge.data.assignments.pop(name)
            for use, arriving in uses:
                if len(arriving) != 1:
                    continue
                single = next(iter(arriving))
                new_name = fresh.get(single)
                if new_name is None:
                    continue
                if isinstance(use, ControlFlowBlock):
                    use.replace(name, new_name)
                else:
                    # Not the keys: an assignment on this edge defines its own version, and the
                    # loop above already renamed the ones that earned a new name.
                    use.data.replace(name, new_name, replace_keys=False)
        return renamed

    @staticmethod
    def reaching_definitions(region: ControlFlowRegion, name: str) -> Dict[ControlFlowBlock, OrderedSet]:
        """The definitions of ``name`` live on entry to each block of ``region``.

        ``None`` stands for "whatever ``name`` held when control entered the region". Only the
        region's own edges define, so a block neither generates nor kills and simply forwards.

        :param region: The region whose block graph is solved.
        :param name: The symbol to trace.
        :returns: block -> set of defining edges (and/or ``None``) reaching its entry.
        """
        arriving: Dict[ControlFlowBlock, OrderedSet] = {b: OrderedSet() for b in region.nodes()}
        arriving[region.start_block].add(None)
        changed = True
        while changed:
            changed = False
            for block in region.nodes():
                before = len(arriving[block])
                for edge in region.in_edges(block):
                    if name in edge.data.assignments:
                        arriving[block].add(edge)  # this edge kills everything upstream
                    else:
                        arriving[block] |= arriving[edge.src]
                if len(arriving[block]) != before:
                    changed = True
        return arriving


__all__ = ['StrictSymbolSSA', 'SymbolSSA']
