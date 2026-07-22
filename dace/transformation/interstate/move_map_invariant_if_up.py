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

Invariance is judged **per map**, not per nest. In a chain the guard belongs
directly outside the innermost map whose parameters it does not read::

    map i: { map j: { if a[i] > 0: ... } }

``a[i]`` varies with ``i`` but is constant across ``j``, so the guard is hoisted
out of ``j`` and lands between the two maps::

    map i: { if a[i] > 0: { map j: ... } }

Repeated application walks a guard outward one map at a time and stops at the
first level whose parameters it reads.

Conservative -- only fires when:

* the map's body is a single ``NestedSDFG`` whose only non-empty block is the
  ``ConditionalBlock`` (any number of branches, with or without an ``else``);
* every branch condition is invariant w.r.t. *that map's* parameters -- either
  because it resolves through the ``NestedSDFG`` ``symbol_mapping`` to symbols
  that are not map parameters, or because it reads only interstate-assigned
  symbols whose defining expressions in turn read data through memlets whose
  subsets do not mention a map parameter;
* for a map at the top level of its state, the state holds nothing but that map
  and the ``AccessNode`` s wired through it (so the whole state can be replaced
  by the hoisted conditional).

Anything else is a no-op.
"""
import copy
from typing import Any, Dict, List, Optional, Set, Tuple

import re

from dace import SDFG, dtypes
from dace import properties, symbolic
from dace.properties import CodeBlock
from dace.sdfg.nodes import AccessNode, MapEntry, MapExit, NestedSDFG
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, SDFGState
from dace.sdfg.utils import set_nested_sdfg_parent_references
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.helpers import nest_state_subgraph


def _identifiers(expr: str) -> Set[str]:
    """Every bare identifier appearing in an expression source.

    Used instead of sympy free symbols because a lifted expression may contain
    a subscripted data access (``a[i]``), which sympy does not read as a symbol.

    :param expr: The expression source.
    :returns: The identifier names it mentions.
    """
    return set(re.findall(r'\b[A-Za-z_]\w*\b', expr))


def _is_predicate(expr: str) -> bool:
    """Report whether an expression source is a boolean predicate.

    :param expr: The expression source.
    :returns: ``True`` if it reads as a comparison or logical combination.
    """
    return bool(re.search(r'[<>]|==|!=|\band\b|\bor\b|\bnot\b', expr))


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


def _inner_assignments(ns: NestedSDFG) -> Dict[str, str]:
    """Every symbol assigned on an interstate edge inside a map body, mapped to
    its defining expression.

    :param ns: The map body's nested SDFG.
    :returns: ``{symbol: rhs_expression}``; a symbol assigned more than once
              maps to ``None`` (it is not a single stable definition).
    """
    defs: Dict[str, str] = {}
    for e in ns.sdfg.all_interstate_edges():
        for lhs, rhs in e.data.assignments.items():
            defs[lhs] = None if lhs in defs else rhs
    return defs


def _connector_reads(ns: NestedSDFG, state: SDFGState) -> Dict[str, Set[str]]:
    """The free symbols of the memlet subset feeding each of ``ns``'s inputs.

    An interstate assignment inside the body may read one of these connectors
    by name; whether that read is map-invariant is decided by whether the
    memlet bringing the data in mentions a map parameter.

    :param ns: The nested SDFG node.
    :param state: The state ``ns`` lives in.
    :returns: ``{connector_name: symbols_used_by_its_incoming_memlet}``.
    """
    reads: Dict[str, Set[str]] = {}
    for e in state.in_edges(ns):
        if e.dst_conn is None or e.data.is_empty():
            continue
        syms = {str(s) for s in e.data.subset.free_symbols} if e.data.subset is not None else set()
        reads[e.dst_conn] = syms
    return reads


def _condition_invariant(cond: CodeBlock,
                         ns: NestedSDFG,
                         state: SDFGState,
                         map_params: Set[str],
                         allow_inner_defs: bool = True) -> bool:
    """Report whether ``cond`` picks the same branch for every element of the map.

    A condition symbol qualifies in one of two ways:

    * it is threaded in through ``symbol_mapping``, and once resolved to the
      enclosing scope's names it mentions no map parameter; or
    * it is defined by an interstate assignment inside the body, and every name
      that definition reads is itself invariant -- a data container entering
      through a connector whose memlet subset is free of map parameters (the
      ``a[i]`` read inside ``map j``), or another qualifying symbol.

    Anything else -- a symbol defined nowhere visible, a multiply-assigned
    symbol, or a read whose memlet is indexed by a map parameter (a genuine
    per-element mask) -- is not invariant.

    :param cond: The branch condition, in inner-SDFG names.
    :param ns: The map body's nested SDFG.
    :param state: The state ``ns`` lives in.
    :param map_params: Parameter names of the map being hoisted out of.
    :param allow_inner_defs: Whether a condition symbol may be defined by an
        interstate assignment inside the body. Only the isolation path may
        allow this -- it lifts those definitions to the new level. The plain
        hoist re-expresses the condition through ``symbol_mapping`` alone, so
        accepting a body-defined symbol there would move the guard and strand
        its definition behind, leaving the symbol undefined at the new scope.
    :returns: ``True`` if the condition is invariant w.r.t. ``map_params``.
    """
    defs = _inner_assignments(ns)
    reads = _connector_reads(ns, state)
    mapping = ns.symbol_mapping

    def qualifies(name: str, depth: int) -> bool:
        if depth > 8:  # pathological definition chain; refuse rather than recurse
            return False
        # A body-local definition SHADOWS the value threaded in through
        # ``symbol_mapping``, so it has to be consulted first. Reading the
        # mapping first would call a symbol invariant on the strength of its
        # outer value while the body reassigns it per element.
        if name in defs:
            rhs = defs[name]
            if rhs is None or not allow_inner_defs:
                return False
            return all(qualifies(s, depth + 1) for s in _free_names(rhs))
        if name in mapping:
            _resolved, resolved_syms = _resolve_through_mapping(CodeBlock(name), mapping)
            return not (resolved_syms & map_params)
        if name in reads:
            # A data container read through a connector: invariant exactly when
            # the memlet that brings it in does not index by a map parameter.
            return not (reads[name] & map_params)
        # Not a map parameter and not defined anywhere we can see: it is an
        # outer-scope symbol, constant for the whole map.
        return name not in map_params

    return all(qualifies(s, 0) for s in {str(x) for x in cond.get_free_symbols()})


def _free_names(expr: str) -> Set[str]:
    """Free symbol names of an interstate-assignment right-hand side.

    :param expr: The expression source.
    :returns: The set of names it reads.
    """
    return {str(s) for s in symbolic.pystr_to_symbolic(expr).free_symbols}


def _match(sdfg: SDFG,
           require_full_hoist: bool = False) -> Optional[Tuple[SDFGState, MapEntry, NestedSDFG, ConditionalBlock]]:
    """Find a single-map state whose map body is one ``NestedSDFG`` guarding a
    map-invariant ``ConditionalBlock``.

    :param sdfg: The SDFG to scan (recursively).
    :returns: ``(state, map_entry, nsdfg, cond_block)`` or ``None``.
    """
    # ``allow_inner_defs=False``: this path re-expresses the condition through
    # ``symbol_mapping`` only, so a body-defined condition symbol would be left
    # behind undefined. Those reach the guard via ``_match_inner`` instead.
    for st, me, ns, cb in _candidates(sdfg, allow_inner_defs=False, require_full_hoist=require_full_hoist):
        if st.entry_node(me) is not None:
            continue  # an inner map: handled by _match_inner
        top = [n for n in st.nodes() if st.entry_node(n) is None]
        if len([n for n in top if isinstance(n, MapEntry)]) != 1:
            continue
        if any(not isinstance(n, (MapEntry, MapExit, AccessNode)) for n in top):
            continue
        # Every top-level node must actually be wired through the map. Being an
        # AccessNode is not enough: a second, map-free component in the same
        # state (the ``a -> b`` copy StateFusion merges in for ``b[:] = a[:]``)
        # would be swept into the conditional along with the map, so with a
        # single-branch guard it would stop running when the guard is false --
        # silently wrong, since it was unconditional before.
        mx = st.exit_node(me)
        for n in top:
            if n is me or n is mx:
                continue
            if not any(e.dst is me for e in st.out_edges(n)) and not any(e.src is mx for e in st.in_edges(n)):
                break
        else:
            return st, me, ns, cb
    return None


def _branch_holds_a_map(cb: ConditionalBlock) -> bool:
    """Report whether any branch of the conditional contains a map.

    :param cb: The guarding conditional block.
    :returns: ``True`` if a ``MapEntry`` appears in any branch.
    """
    return any(
        isinstance(n, MapEntry) for _c, branch in cb.branches for bst in branch.all_states() for n in bst.nodes())


def _enclosing_map_params(st: SDFGState, me: MapEntry) -> Set[str]:
    """Parameters of every map enclosing ``me`` in its own state.

    :param st: The state holding the map.
    :param me: The innermost map entry.
    :returns: The union of the enclosing maps' parameter names.
    """
    params: Set[str] = set()
    scope = st.entry_node(me)
    while scope is not None:
        params |= {str(p) for p in scope.map.params}
        scope = st.entry_node(scope)
    return params


def _candidates(sdfg: SDFG, allow_inner_defs: bool = True, require_full_hoist: bool = False):
    """Yield every ``(state, map_entry, nsdfg, cond_block)`` whose guard is
    invariant w.r.t. that map's own parameters, at any scope depth.

    :param sdfg: The SDFG to scan (recursively).
    :param allow_inner_defs: Whether a condition symbol may be body-defined.
    :param require_full_hoist: Accept only guards that clear the *whole* map
        chain -- invariant w.r.t. every enclosing map's parameters too, so the
        nest is not split partway up.
    :returns: An iterator of candidate tuples.
    """
    # ``all_states`` stops at the nested-SDFG boundary, so recurse explicitly:
    # after an inner map is isolated, its guard lives one nesting level down.
    for sd in sdfg.all_sdfgs_recursive():
        for st in sd.all_states():
            for me in [n for n in st.nodes() if isinstance(n, MapEntry)]:
                cand = _candidate_at(st, me, allow_inner_defs)
                if cand is None:
                    continue
                # A guard whose branch still contains a map is exactly what
                # ``MoveIfIntoMap`` produces when it pushes a guard down to
                # co-locate inner maps for fusion. Hoisting it back one level
                # would undo that and the two passes would trade the guard back
                # and forth. Taking it only when it clears the ENTIRE chain
                # breaks the tie: the guard leaves the nest altogether, which
                # is a strict win and not a move ``MoveIfIntoMap`` will reverse
                # (it only ever pushes a guard into maps below it).
                if require_full_hoist or _branch_holds_a_map(cand[3]):
                    outer = _enclosing_map_params(st, me)
                    if outer and not all(
                            _condition_invariant(c, cand[2], st, outer, allow_inner_defs)
                            for c, _b in cand[3].branches if c is not None):
                        continue
                yield cand


def _candidate_at(st: SDFGState,
                  me: MapEntry,
                  allow_inner_defs: bool = True) -> Optional[Tuple[SDFGState, MapEntry, NestedSDFG, ConditionalBlock]]:
    """The candidate rooted at one map entry, if it qualifies.

    :param st: The state holding the map.
    :param me: The map entry to test.
    :returns: The candidate tuple, or ``None``.
    """
    body = [n for n in st.nodes() if st.entry_node(n) is me]
    nsdfgs = [n for n in body if isinstance(n, NestedSDFG)]
    if len(nsdfgs) != 1 or any(not isinstance(n, (NestedSDFG, MapExit, AccessNode)) for n in body):
        return None
    ns = nsdfgs[0]
    cb = _single_meaningful_conditional(ns.sdfg)
    if cb is None:
        return None
    # A branch containing further maps is NOT a reason to refuse. The condition
    # is evaluated at the conditional, outside every branch-nested map, so it
    # cannot depend on their parameters; invariance w.r.t. the map being hoisted
    # past is the whole soundness requirement and is checked below. Refusing
    # here would pin a fully invariant guard inside the first level of a chain,
    # since after one hoist the guard's own branch always contains the map it
    # just cleared.
    map_params = {str(p) for p in me.map.params}
    if not all(
            _condition_invariant(cond, ns, st, map_params, allow_inner_defs)
            for cond, _b in cb.branches if cond is not None):
        return None
    # A branch with no blocks has no start block to splice in its place, and
    # ``start_block`` raises on a node-less region rather than returning None.
    if any(not branch.nodes() for _c, branch in cb.branches):
        return None
    return st, me, ns, cb


def _match_inner(
        sdfg: SDFG,
        require_full_hoist: bool = False) -> Optional[Tuple[SDFGState, MapEntry, NestedSDFG, ConditionalBlock]]:
    """Find a guard that is invariant w.r.t. an *inner* map of a chain, so it
    can be hoisted to sit between that map and its parent.

    :param sdfg: The SDFG to scan (recursively).
    :returns: ``(state, map_entry, nsdfg, cond_block)`` or ``None``.
    """
    for st, me, ns, cb in _candidates(sdfg, require_full_hoist=require_full_hoist):
        if st.entry_node(me) is None:
            continue
        if _liftable_prelude(ns, st, {str(p) for p in me.map.params}) is None:
            continue
        return st, me, ns, cb
    return None


def _liftable_prelude(ns: NestedSDFG, state: SDFGState, map_params: Set[str]) -> Optional[Dict[str, str]]:
    """Re-express the body's interstate assignments in the scope *outside* the map.

    A condition symbol defined inside the body (``__tmp0 = __conn > 0``) has to
    be computable one level out before the guard can move there. Each connector
    name in the definition is replaced by the data expression its memlet reads
    (``__conn`` -> ``a[i]``), which is valid outside the map precisely because
    the invariance check already established that subset mentions no map
    parameter.

    :param ns: The map body's nested SDFG.
    :param state: The state ``ns`` lives in.
    :param map_params: Parameter names of the map being hoisted out of.
    :returns: ``{symbol: outer_expression}``, or ``None`` if any definition
              cannot be expressed outside the map.
    """
    conn_expr: Dict[str, str] = {}
    for e in state.in_edges(ns):
        if e.dst_conn is None or e.data.is_empty() or e.data.data is None:
            continue
        if e.data.subset is not None and ({str(s) for s in e.data.subset.free_symbols} & map_params):
            continue
        subset = str(e.data.subset) if e.data.subset is not None else ''
        conn_expr[e.dst_conn] = f'{e.data.data}[{subset}]' if subset else e.data.data

    lifted: Dict[str, str] = {}
    for lhs, rhs in _inner_assignments(ns).items():
        if rhs is None:
            return None
        names = _free_names(rhs)
        if names & map_params:
            return None
        expr = rhs
        for name in names:
            if name in conn_expr:
                expr = _substitute_name(expr, name, conn_expr[name])
            elif name in ns.symbol_mapping:
                expr = _substitute_name(expr, name, str(ns.symbol_mapping[name]))
            elif name in lifted:
                continue
            elif name in _inner_assignments(ns):
                return None  # depends on a definition we have not lifted yet
        lifted[lhs] = expr
    return lifted


def _substitute_name(expr: str, name: str, replacement: str) -> str:
    """Replace a whole-word symbol name inside an expression source.

    Done textually rather than symbolically because the replacement may be a
    subscripted data access (``a[i]``), which is not a sympy symbol.

    :param expr: The expression source.
    :param name: The name to replace.
    :param replacement: The text to put in its place.
    :returns: The rewritten expression source.
    """
    return re.sub(rf'\b{re.escape(name)}\b', f'({replacement})', expr)


def _replace_conditional_with_branch(inner: SDFG, cb_label: str, branch: ControlFlowRegion):
    """Splice ``branch``'s body into ``inner`` in place of the conditional
    block named ``cb_label`` (the inner SDFG's only meaningful block).

    :param inner: The map body's nested SDFG (a copy local to one branch).
    :param cb_label: Label of the conditional block to remove.
    :param branch: The branch control-flow region whose body replaces it.
    """
    cb = next(b for b in inner.nodes() if isinstance(b, ConditionalBlock) and b.label == cb_label)
    cb_was_start = inner.start_block is cb
    in_edges = [(e.src, copy.deepcopy(e.data)) for e in inner.in_edges(cb)]
    out_edges = [(e.dst, copy.deepcopy(e.data)) for e in inner.out_edges(cb)]
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

    # Reconnect the spliced body in place of the conditional. Dropping these
    # would strand it: the guard is only the start block when nothing precedes
    # it, and a prelude state assigning the condition always does.
    for src, data in in_edges:
        inner.add_edge(src, node_map[bstart], data)
    if out_edges:
        sinks = [node_map[b] for b in bb if branch.out_degree(b) == 0] or [node_map[bstart]]
        for dst, data in out_edges:
            for sink in sinks:
                inner.add_edge(sink, dst, copy.deepcopy(data))


@properties.make_properties
@transformation.explicit_cf_compatible
class MoveMapInvariantIfUp(ppl.Pass):
    """Hoist a map-invariant guarding conditional out of its map (fixpoint).

    The inverse of ``MoveIfIntoMap`` and the map analogue of
    ``MoveLoopInvariantIfUp``. Replicates the map once per branch and lifts the
    conditional to the map's parent control-flow graph.

    :param require_full_hoist: All-or-nothing mode: hoist only a guard that
        clears the *entire* enclosing map chain. A guard that stalls between
        two levels of a chain splits the nest there, which on GPU turns one
        kernel into a branch around several launches -- so that target takes
        the hoist only when the whole nest stays intact. On CPU each level
        cleared removes a re-evaluation from an enclosing iteration, so a
        partial hoist is still a win and this stays off.
    """
    CATEGORY: str = 'Canonicalization'

    require_full_hoist = properties.Property(dtype=bool,
                                             default=False,
                                             desc='Hoist only guards that clear the entire enclosing map chain.')

    def __init__(self, require_full_hoist: bool = False):
        super().__init__()
        self.require_full_hoist = require_full_hoist

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
        structural_changes = 0
        isolated: Set[Tuple[int, str]] = set()
        while True:
            m = _match(sdfg, self.require_full_hoist)
            if m is not None:
                self._move(*m)
                count += 1
                continue
            # No top-level candidate left; try to expose one by isolating an
            # inner map of a chain into its own nested SDFG. That makes its
            # state a single-map state, which the top-level path above then
            # hoists on the next iteration.
            inner = _match_inner(sdfg, self.require_full_hoist)
            if inner is None:
                break
            # Isolation nests the inner map before it can know whether the
            # guard's definitions re-express against the wrapper's re-based
            # memlets. That nesting is semantics-preserving but structural, so
            # a later refusal must still be reported as a change -- returning
            # "nothing happened" on a modified graph would let downstream
            # stages skip their invalidation. ``isolated`` tracks maps already
            # wrapped so a refusal cannot spin on the same candidate.
            key = (id(inner[0]), inner[1].map.label)
            if key in isolated:
                break
            isolated.add(key)
            progressed = self._isolate_inner_map(*inner)
            set_nested_sdfg_parent_references(sdfg)
            if not progressed:
                structural_changes += 1
                break
        if count:
            set_nested_sdfg_parent_references(sdfg)
        return (count + structural_changes) or None

    @staticmethod
    def _isolate_inner_map(st: SDFGState, me: MapEntry, ns: NestedSDFG, cb: ConditionalBlock) -> bool:
        """Wrap an inner map in a ``NestedSDFG`` and lift its guard's defining
        assignments to that new level.

        A ``ConditionalBlock`` cannot live inside a map scope -- only dataflow
        nodes can -- so the guard cannot simply be placed between the two maps.
        Wrapping the inner map gives the guard a control-flow graph to land in
        that is still inside the parent map, and reduces the chain case to the
        single-map case handled by ``_move``.

        :param st: The state holding the map chain.
        :param me: The inner map's entry node.
        :param ns: The inner map body's nested SDFG.
        :param cb: The invariant guarding conditional.
        :returns: ``True`` if the map was isolated (caller should re-match).
        """
        map_params = {str(p) for p in me.map.params}
        if _liftable_prelude(ns, st, map_params) is None:
            return False

        subgraph = st.scope_subgraph(me, include_entry=True, include_exit=True)
        wrapper = nest_state_subgraph(st.sdfg, st, subgraph, name=f'{me.map.label}_guard_scope')

        # The guard's definitions now belong one level out, in the wrapper's
        # own CFG; leaving copies behind would keep the condition looking
        # body-defined (and so non-invariant) to the next match.
        inner_ns = next(n for w in wrapper.sdfg.all_states() for n in w.nodes()
                        if isinstance(n, NestedSDFG) and n.sdfg is ns.sdfg)
        inner_st = next(w for w in wrapper.sdfg.all_states() if inner_ns in w.nodes())
        # Recompute against the wrapper's own memlets: nesting re-bases data
        # names and subsets, so expressions built from the outer state would
        # read the wrong element here.
        # ``{}`` means the condition needs nothing lifted (it already reads only
        # outer-scope symbols); only ``None`` means it cannot be expressed here.
        lifted = _liftable_prelude(inner_ns, inner_st, map_params)
        if lifted is None:
            return False
        for e in inner_ns.sdfg.all_interstate_edges():
            for lhs in list(e.data.assignments.keys()):
                if lhs in lifted:
                    del e.data.assignments[lhs]
        for lhs in lifted:
            inner_ns.symbol_mapping[lhs] = symbolic.pystr_to_symbolic(lhs)
            if lhs not in inner_ns.sdfg.symbols:
                inner_ns.sdfg.add_symbol(lhs, dtypes.bool_ if _is_predicate(lifted[lhs]) else dtypes.int64)

        if not lifted:
            return True  # nothing to lift; isolating the map was the whole job

        # Put the lifted definitions on the wrapper's entry edge, so they are
        # evaluated once per parent-map element instead of once per inner one.
        pre = wrapper.sdfg.add_state(f'{me.map.label}_guard_pre', is_start_block=True)
        wrapper.sdfg.add_edge(pre, inner_st, InterstateEdge(assignments=lifted))
        for lhs in lifted:
            if lhs not in wrapper.sdfg.symbols:
                wrapper.sdfg.add_symbol(lhs, dtypes.bool_ if _is_predicate(lifted[lhs]) else dtypes.int64)

        # The lifted expressions were written in the *outer* scope's names, so
        # any symbol they read (the parent map's parameter in ``a[i]``) has to
        # be threaded into the wrapper -- nesting ran before they existed.
        outer_symbols = st.sdfg.symbols
        for expr in lifted.values():
            for name in _identifiers(expr):
                if name in wrapper.sdfg.arrays or name in lifted:
                    continue
                if name not in wrapper.symbol_mapping:
                    wrapper.symbol_mapping[name] = symbolic.pystr_to_symbolic(name)
                if name not in wrapper.sdfg.symbols:
                    wrapper.sdfg.add_symbol(name, outer_symbols.get(name, dtypes.int64))
        return True

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
            wrap = ControlFlowRegion(label=f"{st.label}_branch_{len(outer_cb.branches)}")
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
