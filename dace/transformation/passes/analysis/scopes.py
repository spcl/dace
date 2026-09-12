# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Analysis passes precomputing per-scope tables that code generation would otherwise rederive per node."""
import collections
from collections.abc import Mapping
from typing import Dict, Final, List, NoReturn, Optional, Set

import dace
from dace import properties
from dace.dtypes import typeclass
from dace.memlet import Memlet
from dace.sdfg import graph as dgraph
from dace.sdfg import nodes
from dace.sdfg.sdfg import SDFG, InterstateEdge
from dace.sdfg.state import SDFGState, enclosing_region_symbols, sdfg_scope_symbols
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.passes.analysis.analysis import StateReachability

#: The scopes of one state mapped to the symbols visible there; ``None`` keys the state's own top level.
StateScopeTables = dict[nodes.EntryNode | None, dict[str, typeclass]]

#: Per state, the symbols visible at each scope entry; ``None`` keys the state's own top level.
StateSymbolScopes = Dict[SDFGState, StateScopeTables]


def state_scope_symbol_tables(sdfg: SDFG, state: SDFGState, base: dict[str, typeclass]) -> StateScopeTables:
    """
    One table per scope of ``state``, each the answer ``symbols_defined_at`` would give for a node in
    that scope. Built outer to inner so every entry inherits its parent's finished table and
    ``new_symbols`` runs once per scope instead of once per node.

    :param sdfg: The SDFG owning ``state``.
    :param state: The state to tabulate.
    :param base: :func:`~dace.sdfg.state.sdfg_scope_symbols` of ``sdfg``, which is per-SDFG invariant
                 and therefore worth hoisting out of a loop over states.
    :return: Scope entry node (``None`` for the state's top level) to its visible symbols.
    """
    per_scope: StateScopeTables = {None: enclosing_region_symbols(state, base)}
    children = state.scope_children()
    stack: list[nodes.EntryNode | None] = [None]
    while stack:  # outer to inner: each entry inherits its parent's finished table
        parent = stack.pop()
        for node in children[parent]:
            if not isinstance(node, nodes.EntryNode):
                continue
            symbols = collections.OrderedDict(per_scope[parent])
            symbols.update(node.new_symbols(sdfg, state, symbols))
            per_scope[node] = symbols
            stack.append(node)
    return per_scope


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
                per_sdfg[state] = state_scope_symbol_tables(sdfg, state, base)
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


class UndeterminedDType:
    """The "nothing declares this name" answer, unusable as a value: truth-testing raises, so
    ``resolve_dtype_or_undetermined(...) or dace.int64`` cannot compile a guess into the graph. ``dtype`` is part of
    symbol identity here, so a guessed int64 against an int32 map parameter is a SECOND symbol of the same name:
    ``Min(i, i)`` stops folding, ``i - i`` stops cancelling. An undeterminable dtype is a finding, never a default."""

    __slots__ = ()

    def __bool__(self) -> NoReturn:
        raise TypeError('an undetermined symbol dtype has no truth value; report it, never default it')

    def __repr__(self) -> str:
        return 'UNDETERMINED'


#: Singleton :class:`UndeterminedDType`; compare with ``is``, since it refuses ``bool()``.
UNDETERMINED: Final[UndeterminedDType] = UndeterminedDType()

#: What an UNTYPED connector holds: ``typeclass(None)``, printing as ``void`` and not Python ``None``, so a rung
#: testing only for ``None`` answers ``void`` instead of falling through.
UNTYPED_CONNECTOR: Final[typeclass] = typeclass(None)


class UndeterminedSymbolDType(Exception):
    """No rung of :meth:`ScopedSymbolResolver.resolve_dtype`'s ladder declares the name."""

    def __init__(self, name: str, sdfg_label: str) -> None:
        super().__init__(f'cannot determine the dtype of symbol "{name}" in SDFG "{sdfg_label}": it is not a data '
                         f'descriptor, a typed connector, a memlet descriptor, an interstate-edge assignment, a '
                         f'scoped (loop or map) symbol, nor a declared SDFG symbol. Report this, do not default it')
        self.name = name


class StaleScopeCache(Exception):
    """A node sits in a dataflow scope the cached tables never saw: a pass added one and did not invalidate."""

    def __init__(self, state_label: str) -> None:
        super().__init__(f'node in state "{state_label}" sits in a dataflow scope absent from the cached tables. A '
                         f'pass added a scoped symbol; call invalidate_state or invalidate_sdfg before querying again')


class ScopedSymbolResolver:
    """
    The shared scoped-symbol and dtype oracle for canonicalization, vectorization and tileops.

    :meth:`defined_at` answers :meth:`~dace.sdfg.state.SDFGState.symbols_defined_at` without paying for it per node:
    that call rebuilds :func:`~dace.sdfg.state.sdfg_scope_symbols`, :func:`~dace.sdfg.state.enclosing_region_symbols`
    and ``scope_dict()`` before reaching the cheap scope walk, and all three are invariant per ``(sdfg, state)``.
    :meth:`resolve_dtype` puts the dtype ladder behind one entry point. Construct one per pass run; tabulation is
    lazy, so a pass touching three states of four hundred pays for three.

    **Invalidation contract.** Adding a scoped symbol -- a map parameter, a ``LoopRegion`` iterator, an interstate
    assignment -- makes the cache stale. The resolver never detects that, never self-heals, never re-derives behind
    the caller: the pass that added one calls :meth:`invalidate_state` for a changed dataflow scope, or
    :meth:`invalidate_sdfg` for a changed symbol, descriptor, interstate edge or region, all of which feed the
    per-SDFG base every state's table starts from. A node landing in an unseen scope raises
    :class:`StaleScopeCache`; a declaration that merely CHANGED under unchanged scope structure carries no signal
    at all and the old answer is served, so invalidate on every mutation, not only the ones that raise.
    """

    __slots__ = ('sdfg_bases', 'state_tables')

    def __init__(self) -> None:
        self.sdfg_bases: dict[SDFG, dict[str, typeclass]] = {}
        self.state_tables: dict[SDFGState, StateScopeTables] = {}

    def defined_at(self, state: SDFGState, node: nodes.Node | None) -> Mapping[str, typeclass]:
        """The symbols visible at ``node``, equal to ``state.symbols_defined_at(node)`` (empty for ``None``, as the
        core call is). Returns the cached table itself, so it is read-only to the caller."""
        if node is None:
            return collections.OrderedDict()
        tables = self.state_tables.get(state)
        if tables is None:
            tables = self.tabulate(state)
        table = tables.get(state.entry_node(node))
        if table is None:
            raise StaleScopeCache(state.label)
        return table

    def tabulate(self, state: SDFGState) -> StateScopeTables:
        """Build and cache ``state``'s scope tables, reusing this SDFG's base if another state already paid."""
        sdfg = state.sdfg
        base = self.sdfg_bases.get(sdfg)
        if base is None:
            base = sdfg_scope_symbols(sdfg)
            self.sdfg_bases[sdfg] = base
        tables = state_scope_symbol_tables(sdfg, state, base)
        self.state_tables[state] = tables
        return tables

    def resolve_dtype(self,
                      name: str,
                      sdfg: SDFG,
                      state: SDFGState | None = None,
                      node: nodes.Node | None = None,
                      connector: str | None = None,
                      edge: dgraph.MultiConnectorEdge[Memlet] | None = None,
                      interstate_edge: InterstateEdge | None = None) -> typeclass:
        """:meth:`resolve_dtype_or_undetermined`, raising :class:`UndeterminedSymbolDType` rather than returning
        something a caller could mistake for a dtype."""
        resolved = self.resolve_dtype_or_undetermined(name, sdfg, state, node, connector, edge, interstate_edge)
        if resolved is UNDETERMINED:
            raise UndeterminedSymbolDType(name, sdfg.label)
        return resolved

    def resolve_dtype_or_undetermined(self,
                                      name: str,
                                      sdfg: SDFG,
                                      state: SDFGState | None = None,
                                      node: nodes.Node | None = None,
                                      connector: str | None = None,
                                      edge: dgraph.MultiConnectorEdge[Memlet] | None = None,
                                      interstate_edge: InterstateEdge | None = None) -> typeclass | UndeterminedDType:
        """The declared dtype of ``name``, from the most specific source that knows it, else :data:`UNDETERMINED`.

        Rungs, most specific first, each skipped when the caller supplies nothing for it: the data descriptor,
        ``connector`` of ``node``, ``edge``'s descriptor, ``interstate_edge``'s assignment (CloudSC's ``zlcrit`` is
        bound on four edges and declared nowhere, so only that rung answers for it), the scoped table at ``state``
        and ``node`` -- the only source seeing a map parameter or loop iterator -- and finally ``sdfg.symbols``."""
        desc = sdfg.arrays.get(name)
        if desc is not None:
            return desc.dtype

        if connector is not None and node is not None:
            declared = node.in_connectors.get(connector)
            if declared is None:
                declared = node.out_connectors.get(connector)
            if declared is not None and declared != UNTYPED_CONNECTOR:
                return declared

        if edge is not None and edge.data.data is not None:
            edge_desc = sdfg.arrays.get(edge.data.data)
            if edge_desc is not None:
                return edge_desc.dtype

        if interstate_edge is not None:
            bound = interstate_edge.new_symbols(sdfg, sdfg.symbols).get(name)
            if bound is not None:
                return bound

        # Ahead of ``sdfg.symbols`` though dearer: the scoped table is seeded from ``sdfg_scope_symbols``, so it is
        # a superset, and on a shadowed name it is the right answer. Correctness orders this rung, not cost.
        if state is not None and node is not None:
            scoped = self.defined_at(state, node).get(name)
            if scoped is not None:
                return scoped

        declared_symbol = sdfg.symbols.get(name)
        if declared_symbol is not None:
            return declared_symbol

        return UNDETERMINED

    def invalidate_state(self, state: SDFGState) -> None:
        """Drop ``state``'s tables; call after adding or reshaping a dataflow scope in it."""
        self.state_tables.pop(state, None)

    def invalidate_sdfg(self, sdfg: SDFG) -> None:
        """Drop ``sdfg``'s base and every table built from it."""
        self.sdfg_bases.pop(sdfg, None)
        for state in [s for s in self.state_tables if s.sdfg is sdfg]:
            del self.state_tables[state]

    def invalidate_all(self) -> None:
        """Drop everything."""
        self.sdfg_bases.clear()
        self.state_tables.clear()
