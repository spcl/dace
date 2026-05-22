# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Hoist a map-invariant guarding conditional out of its enclosing map.

The map analogue of ``MoveLoopInvariantIfUp`` and the inverse of
``MoveIfIntoMap``: a parallel map whose body is a single ``NestedSDFG`` that
contains only a guarding ``ConditionalBlock`` whose condition does not depend
on the map parameters::

    map[p0, p1]: { NSDFG: if c: A else: B }

becomes, with one copy of the map per branch::

    if c: { map[p0, p1]: A }  else: { map[p0, p1]: B }

The condition leaves the per-element body and is evaluated once, above the
map. This is value-preserving exactly when ``c`` is invariant w.r.t. the map
parameters (it picks the same branch for every element), so the whole map runs
one branch or the other -- identical to guarding each element with the same
invariant condition.

Conservative -- only fires when:

* a state contains exactly one top-level map and nothing else but the
  ``AccessNode`` s wired through it (so the whole state can be replaced by the
  hoisted conditional);
* that map's body is a single ``NestedSDFG`` whose only non-empty block is the
  ``ConditionalBlock`` (any number of branches, with or without an ``else``);
* every branch condition, resolved through the ``NestedSDFG`` ``symbol_mapping``
  to the enclosing scope's symbols, references no map parameter.

Anything else is a no-op.
"""
import copy
from typing import Any, Dict, Optional, Set, Tuple

from dace import SDFG
from dace import properties, symbolic
from dace.properties import CodeBlock
from dace.sdfg.nodes import AccessNode, MapEntry, MapExit, NestedSDFG
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, SDFGState
from dace.sdfg.utils import set_nested_sdfg_parent_references
from dace.transformation import pass_pipeline as ppl, transformation


def _single_meaningful_conditional(inner: SDFG) -> Optional[ConditionalBlock]:
    """The inner SDFG's sole ``ConditionalBlock`` if every other block is an
    empty ``SDFGState``; else ``None``.

    :param inner: A map body's nested SDFG.
    :returns: The guarding ``ConditionalBlock`` or ``None``.
    """
    cbs = [b for b in inner.nodes() if isinstance(b, ConditionalBlock)]
    if len(cbs) != 1:
        return None
    for b in inner.nodes():
        if b is cbs[0]:
            continue
        if not isinstance(b, SDFGState) or b.number_of_nodes() != 0:
            return None
    return cbs[0]


def _resolve_through_mapping(cond: CodeBlock, symbol_mapping: Dict[str, Any]) -> Tuple[str, Set[str]]:
    """Rewrite ``cond`` from inner-SDFG symbol names to the enclosing scope's
    names via ``symbol_mapping`` and return its resolved free symbols.

    :param cond: The branch condition (in inner-SDFG symbol names).
    :param symbol_mapping: The nested SDFG's inner-to-outer symbol mapping.
    :returns: ``(resolved_condition_string, resolved_free_symbol_names)``.
    """
    expr = symbolic.pystr_to_symbolic(cond.as_string)
    free = {str(s) for s in expr.free_symbols}
    repl = {
        symbolic.symbol(k): symbolic.pystr_to_symbolic(str(v))
        for k, v in symbol_mapping.items() if k in free and str(v) != k
    }
    resolved = expr.subs(repl) if repl else expr
    return symbolic.symstr(resolved), {str(s) for s in resolved.free_symbols}


def _match(sdfg: SDFG) -> Optional[Tuple[SDFGState, MapEntry, NestedSDFG, ConditionalBlock]]:
    """Find a single-map state whose map body is one ``NestedSDFG`` guarding a
    map-invariant ``ConditionalBlock``.

    :param sdfg: The SDFG to scan (recursively).
    :returns: ``(state, map_entry, nsdfg, cond_block)`` or ``None``.
    """
    for st in sdfg.all_states():
        top = [n for n in st.nodes() if st.entry_node(n) is None]
        entries = [n for n in top if isinstance(n, MapEntry)]
        if len(entries) != 1:
            continue
        if any(not isinstance(n, (MapEntry, MapExit, AccessNode)) for n in top):
            continue
        me = entries[0]
        body = [n for n in st.nodes() if st.entry_node(n) is me]
        nsdfgs = [n for n in body if isinstance(n, NestedSDFG)]
        if len(nsdfgs) != 1 or any(not isinstance(n, (NestedSDFG, MapExit, AccessNode)) for n in body):
            continue
        ns = nsdfgs[0]
        cb = _single_meaningful_conditional(ns.sdfg)
        if cb is None:
            continue
        map_params = {str(p) for p in me.map.params}
        # Symbols assigned by an interstate edge anywhere inside the map body
        # are per-element values (e.g. ``__tmp0 = a[i]`` for a data-dependent
        # ``if a[i] > thr`` mask). A condition reading one of those is NOT
        # map-invariant -- it varies per element -- so it must stay inside.
        inner_assigned: Set[str] = set()
        for e in ns.sdfg.all_interstate_edges():
            inner_assigned |= set(e.data.assignments.keys())
        # Only hoist a guard over a LEAF perfect nest: the branches must hold
        # plain (tasklet-level) computation, no further maps. A guard whose
        # body still contains a nested map guards an imperfect/multi-level nest
        # where the guard was pushed IN to enable that inner fusion (the dual
        # MoveIfIntoMap direction); hoisting it back out would undo that. The
        # collapsed fully-parallel L-A shape (``map[i, j]: { if c: A else B }``)
        # has leaf branches and is hoisted; ``map i: { if c: { c[i]; map j } }``
        # is left alone.
        if any(isinstance(n, MapEntry) for _c, branch in cb.branches for st in branch.all_states() for n in st.nodes()):
            continue
        invariant = True
        for cond, _branch in cb.branches:
            if cond is None:
                continue  # else branch carries no condition
            raw_syms = {str(s) for s in cond.get_free_symbols()}
            # Every condition symbol must be threaded in from outside via the
            # symbol mapping (so it has a value at the parent scope), defined
            # nowhere inside the body, and -- once resolved -- reference no map
            # parameter.
            if not raw_syms <= set(ns.symbol_mapping.keys()) or (raw_syms & inner_assigned):
                invariant = False
                break
            _resolved, resolved_syms = _resolve_through_mapping(cond, ns.symbol_mapping)
            if resolved_syms & map_params:
                invariant = False
                break
        if not invariant:
            continue
        return st, me, ns, cb
    return None


def _replace_conditional_with_branch(inner: SDFG, cb_label: str, branch: ControlFlowRegion):
    """Splice ``branch``'s body into ``inner`` in place of the conditional
    block named ``cb_label`` (the inner SDFG's only meaningful block).

    :param inner: The map body's nested SDFG (a copy local to one branch).
    :param cb_label: Label of the conditional block to remove.
    :param branch: The branch control-flow region whose body replaces it.
    """
    cb = next(b for b in inner.nodes() if isinstance(b, ConditionalBlock) and b.label == cb_label)
    cb_was_start = inner.start_block is cb
    for e in list(inner.in_edges(cb)) + list(inner.out_edges(cb)):
        inner.remove_edge(e)
    inner.remove_node(cb)

    bb = list(branch.nodes())
    bstart = branch.start_block
    node_map = {}
    for b in bb:
        nb = copy.deepcopy(b)
        node_map[b] = nb
        inner.add_node(nb, is_start_block=(cb_was_start and b is bstart), ensure_unique_name=True)
    for e in branch.edges():
        inner.add_edge(node_map[e.src], node_map[e.dst], copy.deepcopy(e.data))
    if cb_was_start:
        inner.start_block = inner.node_id(node_map[bstart])


@properties.make_properties
@transformation.explicit_cf_compatible
class MoveMapInvariantIfUp(ppl.Pass):
    """Hoist a map-invariant guarding conditional out of its map (fixpoint).

    The inverse of ``MoveIfIntoMap`` and the map analogue of
    ``MoveLoopInvariantIfUp``. Replicates the map once per branch and lifts the
    conditional to the map's parent control-flow graph.
    """
    CATEGORY: str = 'Canonicalization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Hoist invariant guards out of their maps until none remain.

        :param sdfg: The SDFG to transform in place.
        :returns: Number of guards hoisted, or ``None`` if none.
        """
        count = 0
        while True:
            m = _match(sdfg)
            if m is None:
                break
            self._move(*m)
            count += 1
        if count:
            set_nested_sdfg_parent_references(sdfg)
        return count or None

    @staticmethod
    def _move(st: SDFGState, me: MapEntry, ns: NestedSDFG, cb: ConditionalBlock):
        """Replace ``st`` with ``if c: { map: A } [else: { map: B }]`` in its
        parent CFG, one map copy per branch of ``cb``.

        :param st: The single-map state being replaced.
        :param me: The map entry whose body holds the guard.
        :param ns: The map body's nested SDFG.
        :param cb: The map-invariant guarding conditional block.
        """
        cfg = st.parent_graph
        in_edges = list(cfg.in_edges(st))
        out_edges = list(cfg.out_edges(st))
        is_start = cfg.start_block is st

        outer_cb = ConditionalBlock(label=f"{st.label}_guard")
        for cond, branch in cb.branches:
            new_st = copy.deepcopy(st)
            new_ns = next(n for n in new_st.nodes() if isinstance(n, NestedSDFG) and n.sdfg.label == ns.sdfg.label)
            _replace_conditional_with_branch(new_ns.sdfg, cb.label, branch)
            wrap = ControlFlowRegion(label=f"{st.label}_branch")
            wrap.add_node(new_st, is_start_block=True, ensure_unique_name=True)
            outer_cond = None
            if cond is not None:
                resolved, _syms = _resolve_through_mapping(cond, ns.symbol_mapping)
                outer_cond = CodeBlock(resolved)
            outer_cb.add_branch(outer_cond, wrap)

        cfg.add_node(outer_cb, ensure_unique_name=True)
        for e in in_edges:
            cfg.add_edge(e.src, outer_cb, copy.deepcopy(e.data))
        for e in out_edges:
            cfg.add_edge(outer_cb, e.dst, copy.deepcopy(e.data))
        for e in in_edges + out_edges:
            cfg.remove_edge(e)
        cfg.remove_node(st)
        if is_start:
            cfg.start_block = cfg.node_id(outer_cb)
