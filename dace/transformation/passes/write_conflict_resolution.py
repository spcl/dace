# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Write-conflict detection and resolution over the dataflow graph.

A write emitted inside a parallel scope runs concurrently in every iteration of
the enclosing map (or consume) scope. It is correct only if it provably cannot
collide — distinct iterations touch distinct elements — or if it carries a
conflict-resolution function. Deciding that is a property of the *graph*: which
memlet writes where, and what the value being written was computed from.

Frontends have historically decided it from their own syntax instead, which has
a floor they cannot rise above: a Python-level marker cannot see a read that
was hoisted into a separate statement, cannot see through a call whose body is
built by a library implementation, and does not exist at all for graphs that
were not produced by that frontend. This pass asks the question of the graph,
of every write, so every producer of parallel dataflow benefits.

Each conflicting write falls into one of four classes:

- **partitioned** — the write subset varies with every enclosing parallel
  parameter, so iterations touch disjoint elements. Nothing to do.
- **read-modify-write** — the written value is computed from the same element.
  The region feeding the write is composed into a single update expression over
  the element and one contribution, the self-read is dropped, and the write
  carries the equivalent conflict resolution (``b[i] (CR: Sum) = x``), which
  the code generator applies atomically. Composition sees the WHOLE region, so
  a self-read sitting in an earlier tasklet — invisible to a per-statement
  marker, and the cause of conflict resolutions that silently fold a stale
  value — is folded in rather than dropped.
- **order-dependent read-modify-write** — the composed update is race-free but
  its combiner does not commute (``b[i] = a[j] - b[i]``, or a chained
  ``b[i] += b[i] + a[j]``, whose update is ``2x + y``). It is still resolved,
  because an atomic update is strictly better than a racing one, and reported:
  the PROGRAM is order-dependent and no lowering can give it one answer.
- **conflicting overwrite** — concurrent iterations write the same element with
  values that do not depend on it (``out[0] = f(A[i])``). No conflict
  resolution can express "last writer wins"; reported.
- **undecidable** — the graph does not say which element an iteration writes,
  so a collision can be neither proved nor ruled out. Two sources: a scatter
  through a whole-array pointer connector, where the index lives in the
  tasklet's code (``__arr[__ind] = __val``); and anything involving a
  :class:`~dace.data.Reference`, which points wherever it was last set at
  runtime. References matter beyond their own writes: everything here matches
  reads to writes BY CONTAINER NAME, and that stops holding as soon as another
  container can alias one, so writes to a referenced container are left alone
  rather than resolved on a name basis. Reported, but not warned about by
  default — a warning that fires on every scatter teaches people to ignore
  warnings.

The order-independence policy lives in :data:`WCR_OPERATORS` and
:data:`WCR_CALL_COMBINERS`: a combiner is order-independent only when
``f(f(x, a), b) == f(f(x, b), a)``, because conflict resolution applies it in
arbitrary thread order. Note that ``%`` is deliberately absent — it is in the
classic Python frontend's table, which is why that frontend miscompiles it.
"""
import ast
import copy
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from dace import SDFG, SDFGState, data, dtypes, properties
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.graph import MultiConnectorEdge
from dace.transformation import pass_pipeline as ppl
from dace.transformation.transformation import explicit_cf_compatible

#: Binary operators that may become a conflict-resolution lambda.
#:
#: This is the classic Python frontend's ``augassign_ops`` table MINUS ``%``,
#: which is not order-independent (``(30 % 17) % 27 == 13`` but
#: ``(30 % 27) % 17 == 3``). ``/`` is kept: it reorders only within
#: floating-point rounding, the same tolerance already accepted for ``+``
#: and ``*``.
WCR_OPERATORS: Dict[type, str] = {
    ast.Add: '+',
    ast.Sub: '-',
    ast.Mult: '*',
    ast.Div: '/',
    ast.FloorDiv: '//',
    ast.Pow: '**',
    ast.LShift: '<<',
    ast.RShift: '>>',
    ast.BitOr: '|',
    ast.BitXor: '^',
    ast.BitAnd: '&',
}

#: Two-argument functions that fold their accumulator order-independently, and
#: so may become a conflict-resolution lambda just like :data:`WCR_OPERATORS`.
WCR_CALL_COMBINERS: Tuple[str, ...] = ('min', 'max')

#: Operators whose operands may be exchanged without changing the result, used
#: to present a combiner canonically (the element first).
_COMMUTATIVE_OPERATORS = (ast.Add, ast.Mult, ast.BitOr, ast.BitXor, ast.BitAnd)


@dataclass
class ConflictReport:
    """One conflicting write, and what the pass did about it."""
    state: str
    data: str
    subset: str
    reason: str
    resolved: bool = False
    """Whether the write now carries a conflict resolution (is race-free)."""
    order_independent: bool = True
    """Whether the result is independent of the order iterations apply the
    update. A resolved write that is NOT order-independent is race-free, but
    its result depends on thread order — the program itself is order-dependent
    and no lowering can change that."""
    decidable: bool = True
    """Whether the graph says which element each iteration writes. A scatter
    through a data-dependent index does not: it may or may not collide, and
    only the values of the index array decide. Reported, but not warned about
    by default — a warning that fires on every scatter teaches people to ignore
    warnings."""

    def __str__(self) -> str:
        return f'{self.data}[{self.subset}] in state "{self.state}": {self.reason}'


@properties.make_properties
@explicit_cf_compatible
class ResolveWriteConflicts(ppl.Pass):
    """
    Find writes that concurrent iterations of a parallel scope may aim at the
    same element, resolve the ones that are order-independent
    read-modify-writes into conflict-resolution edges, and report the rest.

    See the module docstring for the classification and the policy tables.
    """

    CATEGORY: str = 'Correctness'

    warn = properties.Property(dtype=bool,
                               default=True,
                               desc='Emit a UserWarning for each conflicting write that could not be resolved')

    warn_undecidable = properties.Property(
        dtype=bool,
        default=False,
        desc='Also warn about writes whose target element the graph does not determine (a scatter '
        'through a data-dependent index), which may or may not collide depending on runtime values')

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Memlets | ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & (ppl.Modifies.Memlets | ppl.Modifies.Nodes | ppl.Modifies.Scopes))

    def apply_pass(self, sdfg: SDFG, _) -> Optional[Dict[str, List[ConflictReport]]]:
        """
        :return: ``{'resolved': [...], 'unresolved': [...]}`` describing the
                 conflicting writes found, or None if there were none.
        """
        resolved: List[ConflictReport] = []
        unresolved: List[ConflictReport] = []
        undecidable: List[ConflictReport] = []

        aliased = _reference_targets(sdfg)
        for state in sdfg.all_states():
            for edge in _conflicting_writes(state, sdfg):
                report = _classify_and_resolve(state, edge, sdfg, aliased)
                if report is None:
                    continue
                if not report.decidable:
                    undecidable.append(report)
                else:
                    (resolved if report.resolved else unresolved).append(report)

        if self.warn:
            for report in unresolved:
                warnings.warn(
                    f'Possible write conflict: {report}. Concurrent iterations of the enclosing '
                    f'parallel scope write the same element, and the update has no conflict-resolution '
                    f'equivalent. The generated code contains a data race.', UserWarning)
            for report in (item for item in resolved if not item.order_independent):
                # Race-free now, but the program itself is order-dependent: no
                # lowering can give it a single well-defined answer, so say so
                # rather than let it look settled.
                warnings.warn(
                    f'Order-dependent update: {report}. The write is now applied atomically, but the '
                    f'combiner does not commute, so the result depends on the order in which the '
                    f'iterations of the enclosing parallel scope apply it.', UserWarning)

        if self.warn_undecidable:
            for report in undecidable:
                warnings.warn(f'Possibly conflicting write: {report}.', UserWarning)

        if not resolved and not unresolved and not undecidable:
            return None
        return {'resolved': resolved, 'unresolved': unresolved, 'undecidable': undecidable}

    def report(self, pass_retval: Dict[str, List[ConflictReport]]) -> str:
        return (f'Resolved {len(pass_retval["resolved"])} write conflict(s), '
                f'{len(pass_retval["unresolved"])} left unresolved.')


def enclosing_parallel_params(state: SDFGState, node: nodes.Node) -> Set[str]:
    """The parameters of every parallel scope enclosing a node: map parameters,
    and a consume scope's processing-element index."""
    params: Set[str] = set()
    entry = state.entry_node(node)
    while entry is not None:
        if isinstance(entry, nodes.MapEntry):
            params.update(entry.map.params)
        elif isinstance(entry, nodes.ConsumeEntry):
            if entry.consume.pe_index:
                params.add(entry.consume.pe_index)
        entry = state.entry_node(entry)
    return params


def _conflicting_writes(state: SDFGState, sdfg: SDFG) -> List[MultiConnectorEdge[Memlet]]:
    """
    The innermost write edges whose subset does not partition across the
    enclosing parallel scopes.

    The innermost edge of a write path carries the per-iteration subset; the
    outer edges of the same path have already been widened to the whole scope
    range, which would make every write look like a collision.
    """
    result: List[MultiConnectorEdge[Memlet]] = []
    for node in state.data_nodes():
        for edge in state.in_edges(node):
            if edge.data.is_empty():
                continue
            inner = state.memlet_path(edge)[0]
            if isinstance(inner.src, nodes.EntryNode):
                continue  # A pass-through read, not a write produced in this scope
            params = enclosing_parallel_params(state, inner.src)
            if not params:
                continue
            # A write arriving from a view carries the view's whole window, so
            # the per-iteration subset has to be read off the writes into it.
            written = _write_subsets(inner, state)
            if not written:
                continue
            if all(_partitions(subset, params) for subset in written):
                continue  # Partitioned: iterations write disjoint elements
            # A nested SDFG's connector memlet covers the whole container until
            # propagation narrows it, so the per-iteration subset is only
            # visible inside. Ask there before calling it a collision.
            if isinstance(inner.src, nodes.NestedSDFG):
                nested = _nested_write_subsets(inner.src, inner.src_conn)
                if nested and all(_partitions(nested_subset, params) for nested_subset in nested):
                    continue
            if isinstance(sdfg.arrays.get(node.data), data.Stream):
                # A stream is concurrent by construction: every push appends,
                # so "the same element" is not a collision but the whole point.
                continue
            if _is_scope_private(state, sdfg, node.data, inner.src):
                continue  # Per-iteration storage: every instance has its own copy
            result.append(inner)
    return result


def _is_scope_private(state: SDFGState, sdfg: SDFG, data_name: str, writer: nodes.Node) -> bool:
    """
    Whether a container is per-iteration storage rather than shared memory.

    A transient with the default ``AllocationLifetime.Scope`` is allocated in
    the innermost scope that uses it (see
    ``codegen/targets/framecode.py``), so if every access lies inside the
    writer's own innermost parallel scope, each iteration gets its own copy and
    concurrent writes cannot collide. Staging temporaries a frontend emits
    inside a map body are all of this kind, and reporting them would bury the
    real conflicts in noise.
    """
    descriptor = sdfg.arrays.get(data_name)
    if descriptor is None or not descriptor.transient:
        return False
    if descriptor.lifetime != dtypes.AllocationLifetime.Scope:
        return False
    scope = state.entry_node(writer)
    if scope is None:
        return False
    for other_state in sdfg.all_states():
        for node in other_state.data_nodes():
            if node.data != data_name:
                continue
            if other_state is not state:
                return False
            if _inside_scope(other_state, node, scope):
                continue
            # An access node outside the scope is only harmless if nothing
            # reads it: the value never leaves the iteration that wrote it, so
            # concurrent writes are unobservable. (Frontends route staging
            # temporaries out through the scope exit even when the value is
            # consumed only inside; that outward write is dead.)
            if other_state.out_degree(node) > 0:
                return False
    for edge in sdfg.all_interstate_edges():
        if data_name in edge.data.free_symbols:
            return False
    return True


def _inside_scope(state: SDFGState, node: nodes.Node, scope: nodes.EntryNode) -> bool:
    """Whether a node lies (transitively) inside a scope."""
    entry = state.entry_node(node)
    while entry is not None:
        if entry is scope:
            return True
        entry = state.entry_node(entry)
    return False


def _classify_and_resolve(state: SDFGState, edge: MultiConnectorEdge[Memlet], sdfg: SDFG,
                          aliased: Set[str]) -> Optional[ConflictReport]:
    """Classify one conflicting write and resolve it if possible."""
    data_name = edge.data.data
    subset = edge.data.get_dst_subset(edge, state) or edge.data.subset
    location = ConflictReport(state=state.label, data=data_name, subset=str(subset), reason='')

    # References break the assumption every step below relies on -- that a read
    # and a write of the same element name the same container. A reference
    # points wherever it was last set, at runtime, so a write THROUGH one has
    # an unknown target, and a write TO a container some reference points at
    # may be aliased by writes that never mention its name. Neither can be
    # decided here, and resolving them by name could attach conflict
    # resolution to the wrong update.
    if isinstance(sdfg.arrays.get(data_name), data.Reference):
        location.reason = ('the write goes through a reference, whose target is set at runtime, so the '
                           'element it lands on is unknown')
        location.decidable = False
        return location
    if data_name in aliased:
        location.reason = (f'"{data_name}" is the target of a reference, so reads and writes of it '
                           f'cannot be matched by container name')
        location.decidable = False
        return location

    if _writes_through_pointer(edge) or _writes_somewhere_in(edge, state):
        location.reason = ('the written element is chosen at runtime (a data-dependent index), so a '
                           'collision can be neither proved nor ruled out')
        location.decidable = False
        return location

    self_reads = _self_reads(state, edge, data_name, str(subset))

    # A write produced inside a nested SDFG (a map body) carries its conflict
    # resolution on the INNER edge; the connector edge only gains it once
    # memlet propagation runs, which is after this pass. Ask the inner graph
    # rather than depending on that ordering.
    resolution = edge.data.wcr or _nested_write_wcr(edge.src, edge.src_conn)

    if resolution is not None and not self_reads:
        return None  # Resolved and sound

    if isinstance(edge.src, nodes.NestedSDFG):
        # The write is produced inside a nested SDFG -- a map body, typically.
        # The COLLISION was decided out here, where the enclosing scopes are
        # visible; the read-modify-write itself is in there, where the region
        # feeding it can be seen. Descend and resolve it at its own level.
        return _resolve_inside(edge.src, edge.src_conn, location, aliased)

    if not self_reads:
        location.reason = ('concurrent iterations overwrite the same element with values that do not depend '
                           'on it; no conflict resolution can express this')
        return location

    # A read-modify-write. The single-tasklet fold is the common case and keeps
    # the graph closest to what it was; anything longer (a chain the frontend's
    # ANF left behind, possibly with a conflict resolution ALREADY on the write
    # that silently folds a stale value) goes through region composition.
    if resolution is None:
        combiner = _extract_combiner(state, edge, self_reads)
        if combiner is not None:
            _apply_conflict_resolution(state, edge, combiner, self_reads)
            location.reason = f'resolved into conflict resolution "{combiner}"'
            location.resolved = True
            return location

    composed = _compose_region(state, edge, self_reads, data_name)
    if composed is None:
        location.reason = ('the read-modify-write does not reduce to a combiner over the element and a '
                           'single contribution')
        return location

    _apply_composed_resolution(state, edge, composed)
    location.reason = f'resolved into conflict resolution "{composed.combiner}"'
    location.resolved = True
    location.order_independent = composed.order_independent
    return location


def _self_reads(state: SDFGState, edge: MultiConnectorEdge[Memlet], data_name: str,
                subset: str) -> List[MultiConnectorEdge[Memlet]]:
    """
    The edges that read the same element this write targets and feed the value
    being written, within the same parallel scope.

    This is the "dependent region" of the write: following the dataflow
    backwards from the writer catches a self-read wherever it sits, including
    one hoisted into an earlier tasklet of the same scope.
    """
    reads: List[MultiConnectorEdge[Memlet]] = []
    visited: Set[int] = set()
    frontier: List[nodes.Node] = [edge.src]
    while frontier:
        node = frontier.pop()
        if id(node) in visited:
            continue
        visited.add(id(node))
        if isinstance(node, nodes.NestedSDFG) and node is edge.src:
            # A nested SDFG is not opaque: it reads and writes whole
            # containers through connectors, but only some of its outputs
            # actually depend on a given input. Asking the inner graph avoids
            # reporting a read that merely sits in the same body as the write
            # (a dead self-read the frontend leaves for DCE, for instance).
            for in_edge in state.in_edges(node):
                if in_edge.data.data != data_name or in_edge.dst_conn is None:
                    continue
                read_subset = in_edge.data.get_src_subset(in_edge, state) or in_edge.data.subset
                if read_subset is None or str(read_subset) != subset:
                    continue
                if _depends_within(node.sdfg, in_edge.dst_conn, edge.src_conn):
                    reads.append(in_edge)
            continue
        for in_edge in state.in_edges(node):
            if isinstance(in_edge.src, nodes.EntryNode):
                # Reads entering the scope from outside: follow the memlet to
                # its origin to see what container it names.
                path = state.memlet_path(in_edge)
                source = path[0].src
                if isinstance(source, nodes.AccessNode) and source.data == data_name:
                    read_subset = in_edge.data.get_src_subset(in_edge, state) or in_edge.data.subset
                    if read_subset is not None and str(read_subset) == subset:
                        reads.append(in_edge)
                continue
            if isinstance(in_edge.src, nodes.AccessNode):
                if in_edge.src.data == data_name:
                    read_subset = in_edge.data.get_src_subset(in_edge, state) or in_edge.data.subset
                    if read_subset is not None and str(read_subset) == subset:
                        reads.append(in_edge)
                    continue
                frontier.append(in_edge.src)
                continue
            frontier.append(in_edge.src)
    return reads


def _resolve_inside(node: nodes.NestedSDFG, connector: Optional[str], location: ConflictReport,
                    aliased: Set[str]) -> Optional[ConflictReport]:
    """
    Resolve a conflicting write that a nested SDFG produces, from inside it.

    Whether the write collides is a question about the ENCLOSING scopes, which
    are only visible outside; what the write actually is — a plain overwrite, a
    fold, a chain with a hoisted self-read — is only visible inside. So the
    caller decides the first and delegates the second here, running the same
    classification on the inner write edges. The conflict resolution installed
    inside reaches the outer edges through memlet propagation.

    A frontend's accumulation lands in exactly this shape whenever the map body
    became a nested SDFG, which is why resolving it here is the prerequisite
    for a frontend not having to decide conflicts itself.
    """
    if connector is None:
        return location
    inner_sdfg = node.sdfg
    reports: List[ConflictReport] = []
    for inner_state in inner_sdfg.all_states():
        for access in inner_state.data_nodes():
            if access.data != connector:
                continue
            for in_edge in list(inner_state.in_edges(access)):
                if in_edge.data.is_empty():
                    continue
                inner = inner_state.memlet_path(in_edge)[0]
                if isinstance(inner.src, nodes.EntryNode):
                    continue
                report = _classify_and_resolve(inner_state, inner, inner_sdfg, aliased)
                if report is not None:
                    reports.append(report)
    if not reports:
        # Nothing recognizable in there: report the write as the caller saw it.
        location.reason = ('concurrent iterations write the same element from inside a nested SDFG, and '
                           'the update was not recognizable there')
        return location
    if all(report.resolved for report in reports):
        location.resolved = True
        location.order_independent = all(report.order_independent for report in reports)
        location.reason = '; '.join(sorted({report.reason for report in reports}))
        return location
    unresolved = [report for report in reports if not report.resolved]
    location.reason = '; '.join(sorted({report.reason for report in unresolved}))
    location.decidable = all(report.decidable for report in unresolved)
    return location


def _reference_targets(sdfg: SDFG) -> Set[str]:
    """
    Containers that some :class:`~dace.data.Reference` in the SDFG is pointed
    at (the source of an edge into a ``'set'`` connector).

    A write to one of these may be aliased by a write through the reference,
    which names a different container entirely -- so name-based matching, which
    everything in this pass relies on, does not hold for them.
    """
    targets: Set[str] = set()
    for state in sdfg.all_states():
        for edge in state.edges():
            if edge.dst_conn == 'set' and edge.data.data is not None:
                targets.add(edge.data.data)
    for node in sdfg.all_sdfgs_recursive():
        if node is sdfg:
            continue
        targets |= _reference_targets(node)
    return targets


def _partitions(subset, params: Set[str]) -> bool:
    """Whether a write subset varies with every enclosing parallel parameter,
    so that distinct iterations touch distinct elements."""
    free_symbols = {str(symbol) for symbol in subset.free_symbols}
    return all(param in free_symbols for param in params)


def _nested_write_subsets(node: nodes.Node, connector: Optional[str]) -> List:
    """
    The subsets a nested SDFG writes to the container behind one of its output
    connectors, as seen from the inside — where the enclosing map's parameters
    are in scope, so a per-iteration write is recognizable as one.
    """
    if not isinstance(node, nodes.NestedSDFG) or connector is None:
        return []
    result = []
    for state in node.sdfg.all_states():
        for access in state.data_nodes():
            if access.data != connector:
                continue
            for in_edge in state.in_edges(access):
                result.extend(_write_subsets(in_edge, state))
    return result


def _write_subsets(edge: MultiConnectorEdge[Memlet], state: SDFGState) -> List:
    """
    The subsets one write edge actually touches, in the written container's
    coordinates.

    Normally that is just the edge's own destination subset. The exception is a
    write arriving from a VIEW: the edge out of a view carries the view's whole
    window (``y -> a`` is ``a[0:10]`` no matter which element was written), so
    the per-iteration subset lives on the writes INTO the view and has to be
    composed back through the window to be comparable.
    """
    if isinstance(edge.src, nodes.AccessNode) and isinstance(state.sdfg.arrays.get(edge.src.data), data.View):
        through_view = _view_write_subsets(state, edge.src)
        if through_view:
            return through_view
    subset = edge.data.get_dst_subset(edge, state) or edge.data.subset
    return [subset] if subset is not None else []


def _view_write_subsets(state: SDFGState, view: nodes.AccessNode) -> List:
    """
    The subsets written through a view, expressed in the viewed container's
    coordinates.

    A view node has two kinds of incoming edge: the one naming another
    container is the window it looks through, and the ones naming the view
    itself are the writes into it. Composing the second through the first gives
    the element the write really lands on (``y[__i0]`` through ``a[0:10]`` is
    ``a[__i0]``). Views may chain, so a window that is itself a view recurses.
    """
    window = None
    writes = []
    for edge in state.in_edges(view):
        if edge.data.is_empty():
            continue
        if edge.data.data is not None and edge.data.data != view.data:
            window = edge.data.get_src_subset(edge, state) or edge.data.subset
            if isinstance(edge.src, nodes.AccessNode) and isinstance(state.sdfg.arrays.get(edge.src.data), data.View):
                # A chained view: resolve this window in ITS source first.
                outer = _view_write_subsets(state, edge.src)
                if len(outer) == 1:
                    window = outer[0]
        else:
            subset = edge.data.get_dst_subset(edge, state) or edge.data.subset
            if subset is not None:
                writes.append(subset)
    if window is None:
        return writes
    result = []
    for subset in writes:
        try:
            result.append(window.compose(subset))
        except (TypeError, ValueError, NotImplementedError):
            # Not composable (mismatched dimensionality, a non-Range subset):
            # the write subset alone still names every symbol the composition
            # would, which is all the partition test reads.
            result.append(subset)
    return result


def _writes_through_pointer(edge: MultiConnectorEdge[Memlet]) -> bool:
    """
    Whether a write goes through a whole-array POINTER connector, which means
    the element it targets is chosen by the tasklet's own code
    (``__arr[__ind0] = __val0``, the shape a frontend emits for a scatter).

    The memlet then names the whole container, so nothing in the graph says
    which element each iteration writes: a collision can be neither proved nor
    ruled out.
    """
    if not isinstance(edge.src, nodes.Tasklet) or edge.src_conn is None:
        return False
    return isinstance(edge.src.out_connectors.get(edge.src_conn), dtypes.pointer)


def _writes_somewhere_in(edge: MultiConnectorEdge[Memlet], state: SDFGState) -> bool:
    """
    Whether a write touches FEWER elements than its subset spans — the
    signature of "somewhere in here", which is how a scatter through an index
    array is expressed (subset ``A[0:N]``, volume 1).

    Such a memlet deliberately does not say which element is written, so a
    collision depends on the index values and cannot be settled statically.
    """
    subset = edge.data.get_dst_subset(edge, state) or edge.data.subset
    if subset is None or edge.data.volume is None:
        return False
    try:
        return bool(subset.num_elements() > edge.data.volume > 0)
    except TypeError:
        return False  # Symbolic comparison with no definite answer


def _nested_write_wcr(node: nodes.Node, connector: Optional[str]) -> Optional[str]:
    """
    The conflict resolution a nested SDFG applies to the container behind one
    of its output connectors, or None when it applies none.

    A write inside a map body emitted as a nested SDFG carries its conflict
    resolution on the inner edge. The connector edge only gains it when memlet
    propagation runs, so reading it from the inside makes this pass independent
    of whether propagation has happened yet.
    """
    if not isinstance(node, nodes.NestedSDFG) or connector is None:
        return None
    resolutions = {
        in_edge.data.wcr
        for state in node.sdfg.all_states()
        for access in state.data_nodes() if access.data == connector for in_edge in state.in_edges(access)
    }
    if len(resolutions) != 1:
        return None  # No writes, or an inconsistent mix: do not assume
    return next(iter(resolutions))


def _depends_within(sdfg: SDFG, source: str, target: str) -> bool:
    """
    Whether the value written to ``target`` inside an SDFG depends on the value
    read from ``source``.

    Asking plain container reachability is not enough when the two names are
    the same one (an input and an output connector for the same container, the
    read-modify-write shape this pass exists for). So the question is asked of
    what the writers of ``target`` actually consume: their direct inputs, and
    whether ``source`` reaches any of those.

    The traversal ignores state order, which over-approximates dependence — the
    safe direction, since missing one would let a genuine race pass unreported.
    """
    dependencies: Dict[str, Set[str]] = {}
    consumed_by_writers: Set[str] = set()
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, nodes.AccessNode):
                for edge in state.out_edges(node):
                    if isinstance(edge.dst, nodes.AccessNode):
                        dependencies.setdefault(node.data, set()).add(edge.dst.data)
                        if edge.dst.data == target:
                            consumed_by_writers.add(node.data)
                continue
            written = {edge.data.data for edge in state.out_edges(node) if edge.data.data is not None}
            read = {edge.data.data for edge in state.in_edges(node) if edge.data.data is not None}
            for name in read:
                dependencies.setdefault(name, set()).update(written)
            if target in written:
                consumed_by_writers.update(read)

    if source in consumed_by_writers:
        return True  # A direct read-modify-write of the same container

    frontier = [source]
    seen: Set[str] = set()
    while frontier:
        name = frontier.pop()
        if name in seen:
            continue
        seen.add(name)
        if name in consumed_by_writers:
            return True
        frontier.extend(dependencies.get(name, ()))
    return False


#: Placeholder names used while composing a read-modify-write region: the
#: element's current value, and the value each iteration contributes.
_ACCUMULATOR = '__wcr_x'
_CONTRIBUTION = '__wcr_y'


@dataclass
class ComposedUpdate:
    """A read-modify-write region reduced to a single conflict resolution."""
    combiner: str
    """The conflict-resolution lambda, over (element value, contribution)."""
    contribution: str
    """Tasklet expression computing what one iteration contributes."""
    inputs: List[Tuple[str, MultiConnectorEdge[Memlet]]]
    """(connector name, edge) for each value the contribution reads."""
    region: List[nodes.Node]
    """The nodes the region occupied, to be replaced."""
    order_independent: bool
    """Whether applying the combiner in any order gives the same result."""


def _compose_region(state: SDFGState, edge: MultiConnectorEdge[Memlet], self_reads: List[MultiConnectorEdge[Memlet]],
                    data_name: str) -> Optional[ComposedUpdate]:
    """
    Reduce a whole read-modify-write region to one conflict resolution.

    The region is the chain of tasklets feeding the write. Inlining it gives
    the new value as an expression over the element's current value and the
    iteration's other inputs; folding the write's EXISTING conflict resolution
    into that expression (if it has one) gives the total update. Rewriting the
    total as ``f(element, contribution)`` is what makes it expressible as a
    single atomic update.

    This is what a per-statement marker cannot do: after ANF the self-read sits
    in an earlier tasklet, so the frontend sees a plain accumulation and emits
    a conflict resolution that folds a STALE value — correct-looking and
    silently wrong. Here the whole region is visible at once.

    :return: The composed update, or None if the region is not a chain of
             single-expression tasklets, or its total does not factor into a
             combiner over one contribution.
    """
    if not isinstance(edge.src, nodes.Tasklet):
        return None
    region: List[nodes.Node] = []
    inputs: List[Tuple[str, MultiConnectorEdge[Memlet]]] = []
    consumed: Set[int] = set()
    value = _inline_producer(state, edge.src, self_reads, region, inputs, consumed)
    if value is None:
        return None

    # Safety: every self-read must have been inlined into the expression. If
    # one is still out there -- feeding a node the inliner could not absorb --
    # the composed update would silently omit it, which is exactly the
    # miscompilation this pass exists to catch. Report instead.
    if len(consumed) != len(self_reads) or _ACCUMULATOR not in _names(value):
        return None

    total = value
    if edge.data.wcr is not None:
        total = _fold_existing_wcr(edge.data.wcr, value)
        if total is None:
            return None

    factored = _factor_contribution(total)
    if factored is None:
        return None
    total, contribution = factored

    total = _canonical_operand_order(total)
    combiner = f'lambda x, y: {ast.unparse(_substitute_names(total, {_ACCUMULATOR: "x", _CONTRIBUTION: "y"}))}'
    return ComposedUpdate(combiner=combiner,
                          contribution=ast.unparse(contribution),
                          inputs=inputs,
                          region=region,
                          order_independent=_is_order_independent(total))


def _inline_producer(state: SDFGState, tasklet: nodes.Tasklet, self_reads: List[MultiConnectorEdge[Memlet]],
                     region: List[nodes.Node], inputs: List[Tuple[str, MultiConnectorEdge[Memlet]]],
                     consumed: Set[int]) -> Optional[ast.expr]:
    """
    The expression a tasklet produces, with its inputs inlined: the element's
    own value becomes :data:`_ACCUMULATOR`, a value produced by another tasklet
    of the region is inlined recursively, and anything else becomes an input of
    the composed contribution.
    """
    expression = _single_output_expression(tasklet)
    if expression is None:
        return None
    region.append(tasklet)

    substitutions: Dict[str, ast.expr] = {}
    for in_edge in state.in_edges(tasklet):
        if in_edge.dst_conn is None:
            continue
        if any(read is in_edge for read in self_reads):
            consumed.add(id(in_edge))
            substitutions[in_edge.dst_conn] = ast.Name(id=_ACCUMULATOR, ctx=ast.Load())
            continue
        producer = _region_producer(state, in_edge)
        if producer is not None:
            inlined = _inline_producer(state, producer, self_reads, region, inputs, consumed)
            if inlined is None:
                return None
            substitutions[in_edge.dst_conn] = inlined
            continue
        name = f'__wcr_in{len(inputs)}'
        inputs.append((name, in_edge))
        substitutions[in_edge.dst_conn] = ast.Name(id=name, ctx=ast.Load())
    return _substitute_expressions(expression, substitutions)


def _region_producer(state: SDFGState, in_edge: MultiConnectorEdge[Memlet]) -> Optional[nodes.Tasklet]:
    """
    The tasklet that produced this input, when it belongs to the same region: a
    transient access node written by exactly one tasklet in the same scope, and
    read only here.

    Requiring a single reader keeps the rewrite local — inlining a value that
    something else also consumes would leave that consumer without a producer.
    """
    if not isinstance(in_edge.src, nodes.AccessNode):
        return None
    access = in_edge.src
    if state.in_degree(access) != 1:
        return None
    # Other consumers are tolerated only when they are the scope exit: a
    # frontend routes a staging temporary out of the scope even when nothing
    # outside reads it, and that dead write must not block the rewrite.
    others = [out for out in state.out_edges(access) if out is not in_edge]
    if any(not isinstance(out.dst, nodes.ExitNode) for out in others):
        return None
    producer_edge = next(iter(state.in_edges(access)))
    if not isinstance(producer_edge.src, nodes.Tasklet):
        return None
    if state.entry_node(producer_edge.src) is not state.entry_node(in_edge.dst):
        return None
    return producer_edge.src


def _fold_existing_wcr(wcr: str, value: ast.expr) -> Optional[ast.expr]:
    """
    Apply a write's existing conflict resolution to the value the region
    produces: the memory already holds the element, so the effective update is
    ``wcr(element, value)``.
    """
    try:
        parsed = ast.parse(wcr, mode='eval').body
    except SyntaxError:
        return None
    if not isinstance(parsed, ast.Lambda) or len(parsed.args.args) != 2:
        return None
    accumulator, contribution = (argument.arg for argument in parsed.args.args)
    return _substitute_expressions(parsed.body, {
        accumulator: ast.Name(id=_ACCUMULATOR, ctx=ast.Load()),
        contribution: value,
    })


def _factor_contribution(total: ast.expr) -> Optional[Tuple[ast.expr, ast.expr]]:
    """
    Rewrite an update expression as a combiner over the element and ONE
    contribution: every maximal subexpression that does not read the element
    must be the same one, which then becomes :data:`_CONTRIBUTION`.

    ``x + (x + a)`` factors (the only element-free subexpression is ``a``);
    ``x * a + b`` does not, since a single contribution cannot stand for both.
    """
    subtrees: List[ast.expr] = []

    def collect(node: ast.expr) -> None:
        if _ACCUMULATOR not in _names(node):
            subtrees.append(node)
            return
        if isinstance(node, ast.Call):
            # Only the arguments are operands: the callee is part of the
            # combiner itself, and counting ``max`` as a contribution would
            # make ``max(x, a)`` look like it had two of them.
            children = list(node.args) + [keyword.value for keyword in node.keywords]
        else:
            children = [child for child in ast.iter_child_nodes(node) if isinstance(child, ast.expr)]
        for child in children:
            collect(child)

    collect(total)
    distinct = {ast.unparse(subtree) for subtree in subtrees}
    if len(distinct) != 1:
        return None
    contribution = subtrees[0]
    replacement = ast.Name(id=_CONTRIBUTION, ctx=ast.Load())
    source = ast.unparse(contribution)

    class _Replacer(ast.NodeTransformer):

        def visit(self, node):
            if isinstance(node, ast.expr) and ast.unparse(node) == source:
                return replacement
            return super().generic_visit(node)

    return ast.fix_missing_locations(_Replacer().visit(total)), contribution


def _canonical_operand_order(total: ast.expr) -> ast.expr:
    """
    Put the element first in a commutative combiner, so that ``min(A[i], b[0])``
    and ``min(b[0], A[i])`` produce the same conflict resolution rather than
    two spellings of it. Only reorders where the operator commutes, so the
    meaning is untouched.
    """
    if isinstance(total, ast.Call) and isinstance(total.func, ast.Name):
        if total.func.id in WCR_CALL_COMBINERS and len(total.args) == 2:
            if ast.unparse(total.args[0]) == _CONTRIBUTION and ast.unparse(total.args[1]) == _ACCUMULATOR:
                total.args.reverse()
    elif isinstance(total, ast.BinOp) and isinstance(total.op, _COMMUTATIVE_OPERATORS):
        if ast.unparse(total.left) == _CONTRIBUTION and ast.unparse(total.right) == _ACCUMULATOR:
            total.left, total.right = total.right, total.left
    return total


def _is_order_independent(total: ast.expr) -> bool:
    """
    Whether applying an update in any order yields the same result:
    ``f(f(x, a), b) == f(f(x, b), a)``.

    Only the shapes the policy tables vouch for qualify. Everything else is
    still emitted — an atomic update is strictly better than a racing one — but
    the caller says out loud that the result depends on the order.
    """
    if isinstance(total, ast.BinOp) and type(total.op) in WCR_OPERATORS:
        left, right = ast.unparse(total.left), ast.unparse(total.right)
        if left == _ACCUMULATOR and right == _CONTRIBUTION:
            # Folding the accumulator on the left is what the table vouches
            # for: (x OP a) OP b == (x OP b) OP a.
            return True
        # The other order only folds for a commutative operator: ``y - x`` is
        # "contribution minus element", which does not accumulate at all.
        return (left == _CONTRIBUTION and right == _ACCUMULATOR and isinstance(total.op, _COMMUTATIVE_OPERATORS))
    if isinstance(total, ast.Call) and isinstance(total.func, ast.Name):
        if total.func.id not in WCR_CALL_COMBINERS or len(total.args) != 2:
            return False
        return {ast.unparse(argument) for argument in total.args} == {_ACCUMULATOR, _CONTRIBUTION}
    return False


def _substitute_expressions(expression: ast.expr, substitutions: Dict[str, ast.expr]) -> ast.expr:
    """Replace names in an expression with whole expressions."""

    class _Substituter(ast.NodeTransformer):

        def visit_Name(self, node: ast.Name) -> ast.AST:
            replacement = substitutions.get(node.id)
            return node if replacement is None else copy.deepcopy(replacement)

    return ast.fix_missing_locations(_Substituter().visit(copy.deepcopy(expression)))


def _substitute_names(expression: ast.expr, renaming: Dict[str, str]) -> ast.expr:
    """Rename names in an expression."""
    return _substitute_expressions(expression, {
        name: ast.Name(id=new, ctx=ast.Load())
        for name, new in renaming.items()
    })


def _apply_composed_resolution(state: SDFGState, edge: MultiConnectorEdge[Memlet], composed: ComposedUpdate) -> None:
    """
    Turn a read-modify-write region into a conflict-resolved write: the writing
    tasklet computes only the contribution, and the composed combiner goes on
    the write path.

    The rest of the region is left in place rather than deleted. It no longer
    feeds the write, so it is dead code that dead-dataflow elimination removes
    — and leaving it avoids surgery on scope connectivity, which is easy to get
    subtly wrong and impossible to notice until code generation.
    """
    writer: nodes.Tasklet = edge.src
    # Sources the contribution still needs must survive the teardown below,
    # connectors included -- pruning one and re-attaching to it would leave the
    # edge pointing at a connector that no longer exists.
    reused = {(id(source_edge.src), source_edge.src_conn) for _, source_edge in composed.inputs}
    for in_edge in list(state.in_edges(writer)):
        state.remove_edge(in_edge)
        if (id(in_edge.src), in_edge.src_conn) not in reused:
            _prune_dangling_read(state, in_edge)
    for connector in list(writer.in_connectors):
        writer.remove_in_connector(connector)

    for name, source_edge in composed.inputs:
        writer.add_in_connector(name)
        state.add_edge(source_edge.src, source_edge.src_conn, writer, name, copy.deepcopy(source_edge.data))

    output = next(iter(writer.out_connectors))
    writer.code.code = ast.parse(f'{output} = {composed.contribution}').body

    for path_edge in state.memlet_path(edge):
        path_edge.data.wcr = composed.combiner


def _extract_combiner(state: SDFGState, edge: MultiConnectorEdge[Memlet],
                      self_reads: List[MultiConnectorEdge[Memlet]]) -> Optional[str]:
    """
    The conflict-resolution lambda equivalent to a read-modify-write, or None
    when the region does not reduce to one.

    Recognized shape: a single tasklet computing ``__out = <self> OP <other>``
    (or ``__out = f(<self>, <other>)`` for an order-independent ``f``), reading
    the element exactly once. Anything else — several self-reads, a longer
    chain of tasklets, a non-single-expression body — is left to the caller to
    report, since silently guessing a combiner would be a miscompilation.
    """
    writer = edge.src
    if not isinstance(writer, nodes.Tasklet) or len(self_reads) != 1:
        return None
    read = self_reads[0]
    if read.dst is not writer or read.dst_conn is None:
        return None  # The self-read feeds an earlier node, not this tasklet
    expression = _single_output_expression(writer)
    if expression is None:
        return None
    return _combiner_of(expression, read.dst_conn)


def _single_output_expression(tasklet: nodes.Tasklet) -> Optional[ast.expr]:
    """The right-hand side of a tasklet whose body is one assignment to its one
    output connector, or None."""
    if tasklet.language != dtypes.Language.Python or len(tasklet.out_connectors) != 1:
        return None
    try:
        body = ast.parse(tasklet.code.as_string).body
    except SyntaxError:
        return None
    if len(body) != 1 or not isinstance(body[0], ast.Assign) or len(body[0].targets) != 1:
        return None
    target = body[0].targets[0]
    if not isinstance(target, ast.Name) or target.id not in tasklet.out_connectors:
        return None
    return body[0].value


def _combiner_of(expression: ast.expr, accumulator: str) -> Optional[str]:
    """
    The conflict-resolution lambda for ``expression``, where ``accumulator``
    names the connector holding the element's current value.

    Both operand orders are accepted for a commutative operator; for a
    non-commutative one the accumulator must be on the left, which is the only
    position that folds (``x = x - a`` accumulates, ``x = a - x`` does not).
    """
    if isinstance(expression, ast.BinOp):
        symbol = WCR_OPERATORS.get(type(expression.op))
        if symbol is None:
            return None
        if _names(expression.left) == {accumulator} and accumulator not in _names(expression.right):
            return f'lambda x, y: x {symbol} y'
        commutative = isinstance(expression.op, _COMMUTATIVE_OPERATORS)
        if (commutative and _names(expression.right) == {accumulator} and accumulator not in _names(expression.left)):
            return f'lambda x, y: x {symbol} y'
        return None
    if isinstance(expression, ast.Call) and isinstance(expression.func, ast.Name):
        if expression.func.id not in WCR_CALL_COMBINERS or len(expression.args) != 2 or expression.keywords:
            return None
        left, right = expression.args
        if _names(left) == {accumulator} and accumulator not in _names(right):
            return f'lambda x, y: {expression.func.id}(x, y)'
        if _names(right) == {accumulator} and accumulator not in _names(left):
            return f'lambda x, y: {expression.func.id}(x, y)'
    return None


def _names(expression: ast.expr) -> Set[str]:
    """Every name read by an expression."""
    return {node.id for node in ast.walk(expression) if isinstance(node, ast.Name)}


def _apply_conflict_resolution(state: SDFGState, edge: MultiConnectorEdge[Memlet], combiner: str,
                               self_reads: List[MultiConnectorEdge[Memlet]]) -> None:
    """
    Rewrite a read-modify-write into a conflict-resolved write: the whole write
    path carries the combiner, the self-read is dropped, and the tasklet
    computes only the contribution.
    """
    read = self_reads[0]
    writer = edge.src
    accumulator = read.dst_conn

    for path_edge in state.memlet_path(edge):
        path_edge.data.wcr = combiner

    expression = _single_output_expression(writer)
    output = next(iter(writer.out_connectors))
    contribution = _contribution_of(expression, accumulator)
    writer.code.code = ast.parse(f'{output} = {contribution}').body

    # Drop the self-read: its connector is gone, and so is the path feeding it
    # (up to the scope entry, where other consumers may still need it).
    state.remove_edge(read)
    writer.remove_in_connector(accumulator)
    _prune_dangling_read(state, read)


def _contribution_of(expression: ast.expr, accumulator: str) -> str:
    """The operand of a combiner expression that is NOT the accumulator: the
    value each iteration contributes."""
    if isinstance(expression, ast.BinOp):
        other = expression.right if _names(expression.left) == {accumulator} else expression.left
    else:  # Call form, validated by _combiner_of
        left, right = expression.args
        other = right if _names(left) == {accumulator} else left
    return ast.unparse(other)


def _prune_dangling_read(state: SDFGState, read: MultiConnectorEdge[Memlet]) -> None:
    """Remove the nodes and connectors that only existed to feed a self-read
    that conflict resolution made unnecessary."""
    source = read.src
    if isinstance(source, nodes.EntryNode):
        if read.src_conn is None:
            return
        if any(out.src_conn == read.src_conn for out in state.out_edges(source)):
            return  # Still feeding something else inside the scope
        incoming = [e for e in state.in_edges(source) if e.dst_conn == 'IN_' + read.src_conn[4:]]
        source.remove_out_connector(read.src_conn)
        for in_edge in incoming:
            state.remove_edge(in_edge)
            source.remove_in_connector(in_edge.dst_conn)
            # The read may have been routed through several enclosing scopes;
            # each one now has a connector pair with nothing behind it, which
            # memlet propagation trips over ("no internal edge for this
            # external connector"). Keep unwinding outwards.
            _prune_dangling_read(state, in_edge)
        return
    if isinstance(source, nodes.AccessNode) and state.degree(source) == 0:
        state.remove_node(source)
