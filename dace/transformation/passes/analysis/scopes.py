# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Analysis passes precomputing per-scope tables that code generation would otherwise rederive per node."""
import collections
from typing import Dict, List, Optional, Set

from dace import properties
from dace.sdfg import nodes
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import SDFGState, enclosing_region_symbols, sdfg_scope_symbols
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.passes.analysis.analysis import StateReachability

#: Per state, the symbols visible at each scope entry; ``None`` keys the state's own top level.
StateSymbolScopes = Dict[SDFGState, Dict[Optional[nodes.EntryNode], Dict[str, 'dace.dtypes.typeclass']]]


@properties.make_properties
@transformation.explicit_cf_compatible
class SymbolScopes(ppl.Pass):
    """For each scope of each state, the symbols visible there (the per-node answer of ``symbols_defined_at``)."""

    CATEGORY: str = 'Analysis'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified
                    & (ppl.Modifies.Descriptors | ppl.Modifies.Symbols | ppl.Modifies.CFG | ppl.Modifies.Scopes))

    def apply_pass(self, top_sdfg: SDFG, pipeline_res: Dict) -> Dict[int, StateSymbolScopes]:
        """
        :return: A dictionary mapping each CFG id to its states' per-scope symbol tables.
        """
        result: Dict[int, StateSymbolScopes] = {}
        for sdfg in top_sdfg.all_sdfgs_recursive():
            base = sdfg_scope_symbols(sdfg)
            per_sdfg: StateSymbolScopes = {}
            for state in sdfg.states():
                per_scope = {None: enclosing_region_symbols(state, base)}
                children = state.scope_children()
                stack: List[Optional[nodes.EntryNode]] = [None]
                while stack:  # outer to inner: each entry inherits its parent's finished table
                    parent = stack.pop()
                    for node in children[parent]:
                        if not isinstance(node, nodes.EntryNode):
                            continue
                        symbols = collections.OrderedDict(per_scope[parent])
                        symbols.update(node.new_symbols(sdfg, state, symbols))
                        per_scope[node] = symbols
                        stack.append(node)
                per_sdfg[state] = per_scope
            result[sdfg.cfg_id] = per_sdfg
        return result


def defined_at(scopes: Dict[int, StateSymbolScopes], state: SDFGState,
               node: Optional[nodes.Node]) -> Dict[str, 'dace.dtypes.typeclass']:
    """Table for ``node``'s innermost enclosing scope; falls back to ``symbols_defined_at`` on a miss."""
    if node is None:
        return collections.OrderedDict()
    per_scope = scopes.get(state.sdfg.cfg_id, {}).get(state)
    if per_scope is None:
        return state.symbols_defined_at(node)
    table = per_scope.get(state.entry_node(node))
    if table is None:
        return state.symbols_defined_at(node)
    return table


@properties.make_properties
@transformation.explicit_cf_compatible
class AllocationScopes(ppl.Pass):
    """Lookup tables for :meth:`DaCeCodeGenerator.determine_allocation_lifetime`, replacing its per-descriptor scans."""

    CATEGORY: str = 'Analysis'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & (ppl.Modifies.Descriptors | ppl.Modifies.Symbols | ppl.Modifies.CFG
                                | ppl.Modifies.AccessNodes | ppl.Modifies.Scopes))

    def apply_pass(self, top_sdfg: SDFG, pipeline_res: Dict) -> Dict[str, Dict]:
        """
        :return: ``data_states``, ``root_data_states`` and ``meta_symbols`` keyed by CFG id.
        """
        data_states: Dict[int, Dict[str, List[SDFGState]]] = {}
        root_data_states: Dict[int, Dict[str, Set[SDFGState]]] = {}
        meta_symbols: Dict[int, Set[str]] = {}

        for sdfg in top_sdfg.all_sdfgs_recursive():
            by_data: Dict[str, List[SDFGState]] = collections.defaultdict(list)
            by_root: Dict[str, Set[SDFGState]] = collections.defaultdict(set)
            # ``sdfg.states()`` order (not topological); the consumer relies on it to place allocations.
            for state in sdfg.states():
                seen: Set[str] = set()
                for node in state.data_nodes():
                    if node.data not in seen:
                        seen.add(node.data)
                        by_data[node.data].append(state)
                    by_root[node.root_data].add(state)

            meta: Set[str] = set()
            for isedge in sdfg.all_interstate_edges():
                meta |= isedge.data.free_symbols
            for cfg in sdfg.all_control_flow_regions():
                meta |= cfg.used_symbols(all_symbols=True, with_contents=False)

            data_states[sdfg.cfg_id] = by_data
            root_data_states[sdfg.cfg_id] = by_root
            meta_symbols[sdfg.cfg_id] = meta

        return {
            'data_states': data_states,
            'root_data_states': root_data_states,
            'meta_symbols': meta_symbols,
        }


@properties.make_properties
@transformation.explicit_cf_compatible
class AccessInstances(ppl.Pass):
    """Per container, the states using it in block-topological order, plus each SDFG's shared transients."""

    CATEGORY: str = 'Analysis'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & (ppl.Modifies.Descriptors | ppl.Modifies.CFG | ppl.Modifies.AccessNodes
                                | ppl.Modifies.Tasklets | ppl.Modifies.InterstateEdges))

    def apply_pass(self, top_sdfg: SDFG, pipeline_res: Dict) -> Dict[str, Dict]:
        """
        :return: ``access_instances``, ``code_instances`` and ``shared_transients``, keyed by CFG id.
                 The consumer takes the first and last ``access_instances`` entry, so the order is
                 load-bearing and must stay ``blockorder_topological_sort``.
        """
        from dace.sdfg.analysis import cfg as cfg_analysis

        access_instances: Dict[int, Dict[str, List]] = {}
        code_instances: Dict[int, Dict[str, List]] = {}
        shared_transients: Dict[int, List[str]] = {}

        for sdfg in top_sdfg.all_sdfgs_recursive():
            shared_transients[sdfg.cfg_id] = sdfg.shared_transients(check_toplevel=False, include_nested_data=True)
            instances: Dict[str, List] = collections.defaultdict(list)
            code_uses: Dict[str, List] = collections.defaultdict(list)
            array_names = sdfg.arrays.keys()

            for state in cfg_analysis.blockorder_topological_sort(sdfg, ignore_nonstate_blocks=True):
                for node in state.data_nodes():
                    if node.data not in array_names:
                        continue
                    instances[node.data].append((state, node))

                # A code node naming a container as a free symbol (no memlet) is also a use; a
                #  synthetic AccessNode records it since the consumer only needs the state.
                for node in state.nodes():
                    if not isinstance(node, nodes.CodeNode):
                        continue
                    for used in (node.free_symbols & array_names):
                        instances[used].append((state, nodes.AccessNode(used)))
                        code_uses[used].append((state, node))

                for e in state.parent_graph.all_edges(state):
                    for edge_array in e.data.used_arrays(sdfg.arrays):
                        instances[edge_array].append((state, nodes.AccessNode(edge_array)))

            access_instances[sdfg.cfg_id] = instances
            code_instances[sdfg.cfg_id] = code_uses

        return {
            'access_instances': access_instances,
            'code_instances': code_instances,
            'shared_transients': shared_transients,
        }


class CodegenAnalysisPipeline(ppl.Pipeline):
    """The read-only analyses code generation runs on a frozen SDFG before emitting; shares ``depends_on`` results."""

    def __init__(self):
        super().__init__([StateReachability(), SymbolScopes(), AllocationScopes(), AccessInstances()])
