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
- **resolvable read-modify-write** — the written value is computed from the
  same element by an order-independent combiner (``b[i] = b[i] + x``). The
  self-read is removed and the write carries the equivalent conflict resolution
  (``b[i] (CR: Sum) = x``), which the code generator emits atomically.
- **unresolvable read-modify-write** — a self-referential update whose combiner
  is not order-independent, or whose region this pass cannot reduce to one.
  Reported; the generated code races.
- **conflicting overwrite** — concurrent iterations write the same element with
  values that do not depend on it (``out[0] = f(A[i])``). No conflict
  resolution can express "last writer wins"; reported.

The order-independence policy lives in :data:`WCR_OPERATORS` and
:data:`WCR_CALL_COMBINERS`: a combiner qualifies only when repeatedly folding
the accumulator is order-independent, ``f(f(x, a), b) == f(f(x, b), a)``,
because conflict resolution applies the combiner in arbitrary thread order.
"""
import ast
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from dace import SDFG, SDFGState, dtypes, properties
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


@dataclass
class ConflictReport:
    """One conflicting write the pass could not resolve."""
    state: str
    data: str
    subset: str
    reason: str

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

        for state in sdfg.all_states():
            for edge in _conflicting_writes(state, sdfg):
                report = _classify_and_resolve(state, edge)
                if report is None:
                    continue
                (resolved if report.reason.startswith('resolved') else unresolved).append(report)

        for report in unresolved:
            if self.warn:
                warnings.warn(
                    f'Possible write conflict: {report}. Concurrent iterations of the enclosing '
                    f'parallel scope write the same element, and the update has no conflict-resolution '
                    f'equivalent. The generated code contains a data race.', UserWarning)

        if not resolved and not unresolved:
            return None
        return {'resolved': resolved, 'unresolved': unresolved}

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
            subset = inner.data.get_dst_subset(inner, state) or inner.data.subset
            if subset is None:
                continue
            free_symbols = {str(symbol) for symbol in subset.free_symbols}
            if all(param in free_symbols for param in params):
                continue  # Partitioned: iterations write disjoint elements
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


def _classify_and_resolve(state: SDFGState, edge: MultiConnectorEdge[Memlet]) -> Optional[ConflictReport]:
    """Classify one conflicting write and resolve it if possible."""
    data_name = edge.data.data
    subset = edge.data.get_dst_subset(edge, state) or edge.data.subset
    location = ConflictReport(state=state.label, data=data_name, subset=str(subset), reason='')

    self_reads = _self_reads(state, edge, data_name, str(subset))

    if edge.data.wcr is not None:
        if self_reads:
            # The write is conflict-resolved, but the value it contributes was
            # itself computed from an UNSYNCHRONIZED read of the same element.
            # The combiner is applied to a stale value, so the accumulation
            # double-counts or loses updates -- the shape a syntactic marker
            # cannot see, because the read lives in a different statement.
            location.reason = ('the conflict-resolved value is computed from an unsynchronized read of the '
                               'same element (the read is outside the atomic update)')
            return location
        return None  # Resolved and sound

    if not self_reads:
        location.reason = ('concurrent iterations overwrite the same element with values that do not depend '
                           'on it; no conflict resolution can express this')
        return location

    combiner = _extract_combiner(state, edge, self_reads)
    if combiner is None:
        location.reason = 'the read-modify-write does not reduce to an order-independent combiner'
        return location

    _apply_conflict_resolution(state, edge, combiner, self_reads)
    location.reason = f'resolved into conflict resolution "{combiner}"'
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
        commutative = isinstance(expression.op, (ast.Add, ast.Mult, ast.BitOr, ast.BitXor, ast.BitAnd))
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
        if read.src_conn is not None and state.out_degree(source) > 0:
            if not any(out.src_conn == read.src_conn for out in state.out_edges(source)):
                incoming = [e for e in state.in_edges(source) if e.dst_conn == 'IN_' + read.src_conn[4:]]
                source.remove_out_connector(read.src_conn)
                for in_edge in incoming:
                    state.remove_edge(in_edge)
                    source.remove_in_connector(in_edge.dst_conn)
                    if isinstance(in_edge.src, nodes.AccessNode) and state.degree(in_edge.src) == 0:
                        state.remove_node(in_edge.src)
        return
    if isinstance(source, nodes.AccessNode) and state.degree(source) == 0:
        state.remove_node(source)
