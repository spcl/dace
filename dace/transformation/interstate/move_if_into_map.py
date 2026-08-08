# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Transformation that pushes a conditional block through one level of map nesting.

When a ``Map`` whose body is a single ``NestedSDFG`` guards an inner ``Map`` via
a conditional block, the outer guard can be pushed inside the inner map's body.
This exposes the two maps to fusion/collapsing passes that otherwise refuse to
cross the conditional block.
"""
import copy
from typing import Dict, List, Optional, Set, Tuple

from dace import dtypes, memlet as mm, symbolic
from dace import sdfg as sd
from dace.properties import CodeBlock
from dace.sdfg import utils as sdutil
from dace.sdfg.nodes import MapEntry, MapExit, NestedSDFG, AccessNode, Tasklet
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, SDFGState
from dace.sdfg.utils import set_nested_sdfg_parent_references
from dace.sdfg.graph import SubgraphView
from dace.transformation import transformation
from dace.transformation import helpers as xfh


@transformation.explicit_cf_compatible
class MoveIfIntoMap(transformation.MultiStateTransformation):
    """
    Pushes a conditional block through a level of map nesting.

    Matches the following structure::

        outer_state (in some outer SDFG):
            ... -> outer_map_entry -> outer_nsdfg -> outer_map_exit -> ...

        outer_nsdfg.sdfg contents (possibly preceded by empty states):
            ConditionalBlock(<cond>):
                single branch with a control-flow region containing
                    (optional empty states) -> branch_state
                    branch_state contains one or more sibling top-level maps,
                    each whose body is a single NestedSDFG:
                        ... -> inner_map_entry_k -> inner_nsdfg_k -> inner_map_exit_k -> ...

    After the transformation, the conditional block at the outer nested SDFG
    level is replaced by ``branch_state`` (so the inner maps become neighbours
    of the surrounding maps), and a copy of the original conditional block is
    placed *inside* every ``inner_nsdfg_k.sdfg``, wrapping its original
    contents. The guard's free symbols are threaded into each ``inner_nsdfg_k``
    via ``symbol_mapping`` (and any arrays it reads are piped in), so the guard
    is re-evaluated inside the inner map with the same values it had at the
    outer conditional.

    Design -- patterns this transformation accepts and the soundness argument:

    * **Single guarded map** (the original supported case). One sibling map
      under a one-branch condition.
    * **Multiple sibling maps under one condition** (``if c: map1; map2; ...``).
      The single guard is pushed independently into each sibling map's nested
      SDFG. This is the highest-value extension: it turns ``k`` maps that a
      blocking conditional kept apart into ``k`` adjacent maps that fusion /
      ``MapCollapse`` can then combine. Soundness holds because the guard is
      identical for every sibling and merely replicated inside each inner body
      -- the per-map semantics are unchanged.

    Soundness: the guard is pushed DOWN into the inner map, not hoisted up out
    of the outer map. It sits above the inner map, so it never references the
    inner map's parameters; replicating it inside the inner body therefore
    evaluates it per inner-map iteration to the identical value it had at the
    original (per-outer-iteration) position. This holds even when the condition
    varies with an outer-map parameter (``if i < threshold``): ``i`` is the
    outer map's parameter, constant across the inner iterations, and threaded
    in through ``symbol_mapping`` -- so the per-element guard reproduces the
    original whole-inner-map guard exactly.

    Not accepted (deliberately out of scope to keep the change surgical):
    conditions with an ``else``/``elif`` branch (``len(branches) != 1``).
    """

    cond_block = transformation.PatternNode(ConditionalBlock)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.cond_block)]

    @staticmethod
    def _single_meaningful_state(region: ControlFlowRegion) -> Optional[SDFGState]:
        """Returns the single non-empty ``SDFGState`` inside a region.

        :param region: The control-flow region to inspect.
        :returns: The single non-empty ``SDFGState`` if every other block is an
                  empty ``SDFGState``, else ``None``.
        """
        blocks = list(region.nodes())
        if not blocks:
            return None
        for b in blocks:
            if not isinstance(b, SDFGState):
                return None
        non_empty = [s for s in blocks if not s.is_empty()]
        if len(non_empty) != 1:
            return None
        return non_empty[0]

    @staticmethod
    def _find_all_inner_map_pieces(branch_state: SDFGState) -> Optional[List[Tuple[MapEntry, MapExit, NestedSDFG]]]:
        """Returns the list of ``(map_entry, map_exit, inner_nsdfg)`` tuples for
        every top-level map in ``branch_state``.

        The state must contain one or more top-level maps; every top-level node
        must be a map entry/exit or an :class:`AccessNode` tap, and each map's
        body must be a single NestedSDFG (again with access-node taps allowed).
        These are exactly the maps that the Python frontend emits for sibling
        ``dace.map`` loops under one condition.

        :param branch_state: The state to inspect.
        :returns: A non-empty list of per-map pieces, or ``None`` if the state
                  does not match the expected sibling-maps shape.
        """
        map_entries = [n for n in branch_state.nodes() if isinstance(n, MapEntry)]
        if len(map_entries) < 1:
            return None

        map_pairs = [(me, branch_state.exit_node(me)) for me in map_entries]
        scope_nodes: Set = set()
        for me, mx in map_pairs:
            scope_nodes.add(me)
            scope_nodes.add(mx)

        for node in branch_state.nodes():
            if branch_state.entry_node(node) is not None:
                continue
            if node in scope_nodes:
                continue
            if isinstance(node, AccessNode):
                continue
            return None

        pieces: List[Tuple[MapEntry, MapExit, NestedSDFG]] = []
        for map_entry, map_exit in map_pairs:
            body = list(branch_state.all_nodes_between(map_entry, map_exit))
            nsdfgs = [n for n in body if isinstance(n, NestedSDFG)]
            if len(nsdfgs) != 1:
                return None
            inner_nsdfg = nsdfgs[0]
            for node in body:
                if node is inner_nsdfg:
                    continue
                if isinstance(node, AccessNode):
                    continue
                return None
            pieces.append((map_entry, map_exit, inner_nsdfg))

        return pieces

    @staticmethod
    def _inner_maps_shape_ok(branch_state: SDFGState) -> bool:
        """Returns whether ``branch_state`` is the sibling-maps shape.

        Requires at least one top-level map, every top-level node to be a map
        entry/exit or an :class:`AccessNode` tap, and every map to have a
        non-empty body. Unlike :meth:`_find_all_inner_map_pieces` the body need
        not already be a single ``NestedSDFG``; :meth:`_normalize_inner_map_bodies`
        wraps plain ``Tasklet`` bodies (the common Python-frontend shape).

        :param branch_state: The conditional branch state to inspect.
        :returns: ``True`` if the shape is the normalizable sibling-maps shape.
        """
        map_entries = [n for n in branch_state.nodes() if isinstance(n, MapEntry)]
        if len(map_entries) < 1:
            return False
        # Only sibling top-level maps. A nested map (``if c: map j: map k:``)
        # would be accepted here but breaks the rewrite: normalizing the outer
        # map nests the inner one away, so the snapshot loop in
        # ``_normalize_inner_map_bodies`` then calls ``exit_node`` on a node
        # already removed from the state (a KeyError from the plain-dict scope
        # lookup) -- and by then the branch state has been mutated, so the
        # transformation would raise having already changed the graph.
        if any(branch_state.entry_node(me) is not None for me in map_entries):
            return False
        scope_nodes: Set = set()
        for me in map_entries:
            scope_nodes.add(me)
            scope_nodes.add(branch_state.exit_node(me))
        for node in branch_state.nodes():
            if branch_state.entry_node(node) is not None:
                continue
            if node in scope_nodes or isinstance(node, AccessNode):
                continue
            return False
        for me in map_entries:
            body = list(branch_state.all_nodes_between(me, branch_state.exit_node(me)))
            if not any(not isinstance(n, AccessNode) for n in body):
                return False
        return True

    @staticmethod
    def _normalize_inner_map_bodies(branch_state: SDFGState, enclosing_sdfg: sd.SDFG):
        """Ensures every top-level inner map's body is a single ``NestedSDFG``.

        The Python frontend emits sibling ``dace.map`` bodies as plain
        ``Tasklet`` subgraphs, whereas the per-map rewrite expects a single
        ``NestedSDFG``. Any map whose body is not already exactly one
        ``NestedSDFG`` has its body nested into one in place, after which the
        proven NestedSDFG path applies uniformly.

        :param branch_state: The conditional branch state holding the maps.
        :param enclosing_sdfg: The SDFG that owns ``branch_state``.
        """
        for me in [n for n in branch_state.nodes() if isinstance(n, MapEntry)]:
            body = list(branch_state.all_nodes_between(me, branch_state.exit_node(me)))
            nsdfgs = [n for n in body if isinstance(n, NestedSDFG)]
            non_access = [n for n in body if not isinstance(n, AccessNode)]
            if len(nsdfgs) == 1 and len(non_access) == 1:
                continue
            xfh.nest_state_subgraph(enclosing_sdfg, branch_state, SubgraphView(branch_state, body))

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        cond_block: ConditionalBlock = self.cond_block

        if len(cond_block.branches) != 1:
            return False
        branch_cond, branch_cfr = cond_block.branches[0]
        if branch_cond is None:
            return False

        parent_graph = cond_block.parent_graph
        if not isinstance(parent_graph, sd.SDFG):
            return False
        enclosing_sdfg: sd.SDFG = parent_graph

        if enclosing_sdfg.parent_nsdfg_node is None or enclosing_sdfg.parent is None:
            return False
        outer_state: SDFGState = enclosing_sdfg.parent
        outer_nsdfg: NestedSDFG = enclosing_sdfg.parent_nsdfg_node

        if outer_state.entry_node(outer_nsdfg) is None:
            return False

        for b in enclosing_sdfg.nodes():
            if b is cond_block:
                continue
            if not isinstance(b, SDFGState) or not b.is_empty():
                return False

        if not isinstance(branch_cfr, ControlFlowRegion):
            return False
        branch_state = self._single_meaningful_state(branch_cfr)
        if branch_state is None:
            return False

        if not self._inner_maps_shape_ok(branch_state):
            return False

        # The condition is pushed INTO each inner map's body, where it is
        # evaluated per inner-map iteration with the outer map's parameters and
        # symbols threaded in through ``symbol_mapping`` (and any read arrays
        # piped in). The guard sits above the inner map, so it never references
        # the inner map's own parameters; per-iteration evaluation inside the
        # inner map therefore yields the identical result to the original
        # per-outer-iteration guard -- including when the condition varies with
        # an outer-map parameter (``if i < threshold``). The only requirement
        # is that the condition's free symbols can be read at all; a condition
        # that cannot be parsed is rejected (``apply`` could not thread it).
        try:
            cond_syms = {str(s) for s in branch_cond.get_free_symbols()}
        except Exception:
            return False

        # The condition is usually a symbol assigned on the entry edge
        # (``__tmp0 = (__tmp_r > thr)``), and ``apply`` moves that assignment
        # INTO the inner body so the guard can be re-evaluated there. That only
        # works while the assignment reads symbols: an assignment reading a
        # data container turns those names into free symbols of the inner SDFG,
        # which the nested node cannot satisfy through ``symbol_mapping`` --
        # the result fails validation with "Missing symbols on nested SDFG".
        # Refuse rather than emit an invalid SDFG.
        #
        # The data-reading case is the common data-dependent guard, so this
        # gives up real fusion opportunities; the proper fix is to leave the
        # assignment on the edge feeding the replacement state and thread only
        # the resulting boolean in, which needs no data piping at all.
        # A guard symbol defined on more than one incoming edge has no single
        # definition to move inside. ``apply`` collects them into a flat dict,
        # so the last one seen wins -- while every edge's copy is deleted. Both
        # predecessors would then evaluate the guard with whichever definition
        # the iteration order happened to land on. Silently wrong; refuse.
        guard_defs: Dict[str, int] = {}
        for e in enclosing_sdfg.in_edges(cond_block):
            for lhs in e.data.assignments:
                if lhs in cond_syms:
                    guard_defs[lhs] = guard_defs.get(lhs, 0) + 1
        if any(count > 1 for count in guard_defs.values()):
            return False

        for e in enclosing_sdfg.in_edges(cond_block):
            for lhs, rhs in e.data.assignments.items():
                if lhs not in cond_syms:
                    continue
                try:
                    rhs_names = {str(s) for s in symbolic.pystr_to_symbolic(rhs).free_symbols}
                except Exception:
                    return False
                if any(n in enclosing_sdfg.arrays for n in rhs_names):
                    return False
        if any(s in enclosing_sdfg.arrays for s in cond_syms):
            return False

        return True

    def _rewrite_inner_sdfg(self, cond_block: ConditionalBlock, branch_cond: CodeBlock, enclosing_sdfg: sd.SDFG,
                            inner_nsdfg: NestedSDFG, cond_free_syms: Set[str],
                            moved_assignments: Dict[str, str]) -> Set[str]:
        """Wraps the body of one inner NestedSDFG in a copy of the moved
        condition and threads in the symbols/arrays it needs.

        This is the per-map core of :meth:`apply`; it is invoked once for every
        sibling map under the condition so that each inner body receives its own
        guard. The guard is identical across siblings (same ``branch_cond``),
        which keeps the per-map semantics unchanged.

        :param cond_block: The conditional block being pushed in (used for
                           naming the new blocks).
        :param branch_cond: The condition to replicate inside the inner body.
        :param enclosing_sdfg: The SDFG that currently owns ``cond_block``.
        :param inner_nsdfg: The NestedSDFG whose body is wrapped.
        :param cond_free_syms: Free symbol names of the condition.
        :param moved_assignments: Interstate assignments pulled out of the
                                  outer edges that the condition depends on.
        :returns: The set of array names that had to be piped into
                  ``inner_nsdfg`` (so the caller can wire them on the copied
                  branch state).
        """
        inner_sdfg: sd.SDFG = inner_nsdfg.sdfg

        # Identify arrays referenced by the moved assignments -- they must
        # be piped as inputs into the inner NSDFG so the moved-inside
        # assignments can still evaluate.
        arrays_to_pipe: Set[str] = set()
        for sym, rhs in moved_assignments.items():
            tmp_edge = InterstateEdge(assignments={sym: rhs})
            for mmlt in tmp_edge.get_read_memlets(enclosing_sdfg.arrays):
                arrays_to_pipe.add(mmlt.data)

        piped_array_shape_syms: Set[str] = set()
        for arr_name in arrays_to_pipe:
            if arr_name not in inner_sdfg.arrays:
                desc = copy.deepcopy(enclosing_sdfg.arrays[arr_name])
                desc.transient = False
                inner_sdfg.add_datadesc(arr_name, desc)
            if arr_name not in inner_nsdfg.in_connectors:
                inner_nsdfg.add_in_connector(arr_name)
            piped_array_shape_syms.update(str(s) for s in enclosing_sdfg.arrays[arr_name].free_symbols)

        # Pipe free symbols that the condition and moved RHS expressions
        # reference. Skip any names that now live as data descriptors OR
        # that are themselves defined inside (LHS of a moved assignment).
        syms_needed: Set[str] = set(cond_free_syms)
        for rhs in moved_assignments.values():
            try:
                for s in symbolic.pystr_to_symbolic(rhs).free_symbols:
                    syms_needed.add(str(s))
            except Exception:
                pass
        syms_needed |= piped_array_shape_syms
        syms_needed -= arrays_to_pipe
        syms_needed -= set(moved_assignments.keys())
        for sym_name in syms_needed:
            if sym_name in enclosing_sdfg.symbols:
                if sym_name not in inner_sdfg.symbols:
                    inner_sdfg.add_symbol(sym_name, enclosing_sdfg.symbols[sym_name])
                if sym_name not in inner_nsdfg.symbol_mapping:
                    inner_nsdfg.symbol_mapping[sym_name] = sym_name

        # Declare the moved-assignment LHS symbols inside the inner SDFG.
        # They're defined by the pre-state's interstate edge below.
        for sym in moved_assignments:
            if sym not in inner_sdfg.symbols:
                dtype = enclosing_sdfg.symbols.get(sym, dtypes.int32)
                inner_sdfg.add_symbol(sym, dtype)

        new_body_cfr = ControlFlowRegion(label=f"{cond_block.label}_body")
        body_blocks = list(inner_sdfg.nodes())
        original_start = inner_sdfg.start_block
        copy_mapping = {}
        for b in body_blocks:
            new_b = copy.deepcopy(b)
            copy_mapping[b] = new_b
            new_body_cfr.add_node(new_b, is_start_block=(b is original_start))
        for e in inner_sdfg.edges():
            new_body_cfr.add_edge(copy_mapping[e.src], copy_mapping[e.dst], copy.deepcopy(e.data))

        new_cond_block = ConditionalBlock(label=f"{cond_block.label}_moved")
        new_cond_block.add_branch(CodeBlock(branch_cond.as_string), new_body_cfr)

        inner_sdfg.remove_nodes_from(body_blocks)
        inner_sdfg.add_node(new_cond_block, is_start_block=True, ensure_unique_name=True)

        # If there are assignments to move, prepend a pre-state that
        # materialises them via an interstate edge to ``new_cond_block``.
        if moved_assignments:
            inner_pre = inner_sdfg.add_state(f"{cond_block.label}_materialize")
            inner_sdfg.add_edge(inner_pre, new_cond_block, InterstateEdge(assignments=dict(moved_assignments)))
            inner_sdfg.start_block = inner_sdfg.node_id(inner_pre)

        return arrays_to_pipe

    def apply(self, graph: ControlFlowRegion, sdfg: sd.SDFG):
        cond_block: ConditionalBlock = self.cond_block
        enclosing_sdfg: sd.SDFG = cond_block.parent_graph  # type: ignore[assignment]

        branch_cond: CodeBlock = cond_block.branches[0][0]
        branch_cfr: ControlFlowRegion = cond_block.branches[0][1]
        branch_state: SDFGState = self._single_meaningful_state(branch_cfr)  # type: ignore[assignment]

        # Wrap plain-Tasklet inner-map bodies (the Python-frontend shape) into a
        # single NestedSDFG so the per-map rewrite below applies uniformly.
        self._normalize_inner_map_bodies(branch_state, enclosing_sdfg)

        all_pieces = self._find_all_inner_map_pieces(branch_state)  # type: ignore[assignment]

        try:
            cond_free_syms = {str(s) for s in branch_cond.get_free_symbols()}
        except Exception:
            cond_free_syms = set()

        # Pull out any interstate-edge assignment feeding ``cond_block``
        # whose LHS is referenced by the guard condition. Those assignments
        # must move inside the inner NSDFG together with the condition they
        # define so that none of them is left orphaned on the outer edges
        # (they would otherwise add an always-true pre-state to the outer
        # map body, blocking collapse).
        in_edges = list(enclosing_sdfg.in_edges(cond_block))
        moved_assignments: Dict[str, str] = {}
        for e in in_edges:
            for k in list(e.data.assignments.keys()):
                if k in cond_free_syms:
                    moved_assignments[k] = e.data.assignments[k]
                    del e.data.assignments[k]

        # Wrap each sibling map's inner body with its own copy of the guard.
        # All siblings share the identical condition and moved assignments;
        # the union of arrays that needed piping is rewired below.
        arrays_to_pipe: Set[str] = set()
        for _, _, inner_nsdfg in all_pieces:
            arrays_to_pipe |= self._rewrite_inner_sdfg(cond_block, branch_cond, enclosing_sdfg, inner_nsdfg,
                                                       cond_free_syms, moved_assignments)

        new_branch_state = copy.deepcopy(branch_state)
        enclosing_sdfg.add_node(new_branch_state, ensure_unique_name=True)

        # Wire the newly-needed array inputs through every copied inner map
        # to its copied inner NSDFG node. The copied ``new_branch_state`` has
        # its own MapEntry/NestedSDFG nodes; we look them up again.
        if arrays_to_pipe:
            copied_pieces = self._find_all_inner_map_pieces(new_branch_state)
            for copied_entry, _, copied_nsdfg in copied_pieces:
                for arr_name in arrays_to_pipe:
                    if arr_name not in copied_nsdfg.sdfg.arrays:
                        continue
                    if arr_name not in copied_nsdfg.in_connectors:
                        copied_nsdfg.add_in_connector(arr_name)
                    if "IN_" + arr_name not in copied_entry.in_connectors:
                        copied_entry.add_in_connector("IN_" + arr_name)
                    if "OUT_" + arr_name not in copied_entry.out_connectors:
                        copied_entry.add_out_connector("OUT_" + arr_name)
                    arr_read = new_branch_state.add_read(arr_name)
                    new_branch_state.add_edge(arr_read, None, copied_entry, "IN_" + arr_name,
                                              mm.Memlet.from_array(arr_name, enclosing_sdfg.arrays[arr_name]))
                    new_branch_state.add_edge(copied_entry, "OUT_" + arr_name, copied_nsdfg, arr_name,
                                              mm.Memlet.from_array(arr_name, enclosing_sdfg.arrays[arr_name]))

        out_edges = list(enclosing_sdfg.out_edges(cond_block))
        was_start = enclosing_sdfg.start_block is cond_block

        # Wire in_edges to new_branch_state. If an edge has been emptied
        # by the moved-inside step (no assignments, no condition), drop it
        # and also drop its source state when the source is an empty
        # placeholder that exists only to feed the ConditionalBlock.
        states_to_try_remove: Set[SDFGState] = set()
        for e in in_edges:
            if e.data.is_unconditional() and not e.data.assignments:
                if isinstance(e.src, SDFGState):
                    states_to_try_remove.add(e.src)
                enclosing_sdfg.remove_edge(e)
            else:
                enclosing_sdfg.add_edge(e.src, new_branch_state, copy.deepcopy(e.data))
                enclosing_sdfg.remove_edge(e)

        for e in out_edges:
            enclosing_sdfg.add_edge(new_branch_state, e.dst, copy.deepcopy(e.data))
            enclosing_sdfg.remove_edge(e)

        enclosing_sdfg.remove_node(cond_block)

        # Repoint the start AFTER the removal: ``start_block`` is stored as a
        # node index, and removing a node renumbers the ones after it -- so
        # assigning it first would leave the index dangling.
        if was_start or not in_edges:
            enclosing_sdfg.start_block = enclosing_sdfg.node_id(new_branch_state)

        # Drop the moved symbols from the enclosing SDFG's symbol table if
        # no interstate edge or nested SDFG symbol-mapping still uses them.
        # Otherwise the enclosing SDFG would appear to require them as free
        # symbols from its own parent, but they're now defined (and
        # consumed) purely inside the inner NSDFG.
        def _still_references(sdfg: sd.SDFG, name: str) -> bool:
            # Only scan the enclosing SDFG's own top-level edges and
            # direct-child NestedSDFG symbol_mappings; anything deeper is
            # encapsulated and doesn't pull ``name`` out of this scope.
            for e in sdfg.edges():
                if not isinstance(e.data, InterstateEdge):
                    continue
                for v in e.data.assignments.values():
                    try:
                        if name in {str(s) for s in symbolic.pystr_to_symbolic(v).free_symbols}:
                            return True
                    except Exception:
                        pass
                if e.data.condition is not None:
                    try:
                        if name in {str(s) for s in e.data.condition.get_free_symbols()}:
                            return True
                    except Exception:
                        pass
            # Dataflow uses the symbol too, and dropping it there is what makes
            # the removal dangerous: ``SDFG.remove_symbol`` also strips the name
            # from the parent NestedSDFG's symbol_mapping, so a map range like
            # ``map[0:n2]`` or a memlet subset mentioning ``n2`` would be left
            # referring to a symbol that no longer exists at any level.
            for state in sdfg.states():
                for n in state.nodes():
                    if isinstance(n, NestedSDFG):
                        for v in n.symbol_mapping.values():
                            try:
                                if name in {str(s) for s in symbolic.pystr_to_symbolic(str(v)).free_symbols}:
                                    return True
                            except Exception:
                                pass
                    if isinstance(n, MapEntry):
                        if name in {str(s) for s in n.map.range.free_symbols}:
                            return True
                    if isinstance(n, Tasklet):
                        if name in {str(s) for s in n.free_symbols}:
                            return True
                for e in state.edges():
                    if e.data is None or e.data.is_empty():
                        continue
                    if name in {str(s) for s in e.data.free_symbols}:
                        return True
            # Descriptor shapes and strides can be symbolic as well.
            for desc in sdfg.arrays.values():
                if name in {str(s) for s in desc.free_symbols}:
                    return True
            return False

        for sym in list(moved_assignments.keys()):
            if sym in enclosing_sdfg.symbols and not _still_references(enclosing_sdfg, sym):
                enclosing_sdfg.remove_symbol(sym)

        # Drop empty placeholder pre-states that are no longer reachable.
        #
        # The start block is captured ONCE, before any removal: ``start_block``
        # resolves a cached node *index*, and removing a node invalidates it --
        # reading the property again mid-loop raises ``NodeNotFoundError``.
        start_before = enclosing_sdfg.start_block if enclosing_sdfg.nodes() else None
        removed_start = False
        for s in states_to_try_remove:
            if (s in enclosing_sdfg.nodes() and s.is_empty() and enclosing_sdfg.in_degree(s) == 0
                    and enclosing_sdfg.out_degree(s) == 0):
                removed_start = removed_start or s is start_before
                enclosing_sdfg.remove_node(s)
        if removed_start:
            enclosing_sdfg.start_block = enclosing_sdfg.node_id(new_branch_state)

        set_nested_sdfg_parent_references(sdfg)
