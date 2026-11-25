# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Array facts: what an index array holds where a loop reads it, and the write-uniqueness proofs they admit.

A fact is a property of an array's contents at one program point:

* ``INJECTIVE``: the entries other than ``sentinel`` are pairwise distinct, each within ``[lo, hi]``
  (a permutation, the output of a counting-sort scatter);
* ``NONDECREASING``: ``A[x] <= A[y]`` for ``lo <= x <= y <= hi`` (a prefix sum, CSR row pointers);
* ``NONNEGATIVE``: every entry is ``>= 0``;
* ``BOUNDED``: every entry lies in ``[lo, hi]`` (an index array every store into which is in range).

Facts are derived from the producer blocks preceding the reading loop (:func:`facts_at`), and enter the
consumer's conflict query as axioms instantiated over the array reads the query contains
(:func:`unique_write_under_facts`).
"""

import ast
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import sympy

from dace import data as dt
from dace import dtypes, subsets, symbolic
from dace.libraries.standard.nodes.fill.node import FillLibraryNode
from dace.libraries.standard.nodes.scan import Scan, ScanOp, segmented
from dace.optionals import required
from dace.sdfg import nodes
from dace.sdfg.analysis.cfg import collect_enclosing_conditions
from dace.sdfg.narrowing import as_expr
from dace.sdfg.sdfg import SDFG, interstate_bound_names
from dace.sdfg.state import (
    ConditionalBlock,
    ControlFlowBlock,
    ControlFlowRegion,
    LoopRegion,
    SDFGState,
    SymbolResolver,
    is_written,
)
from dace.transformation.passes.analysis import canon_stats, loop_analysis, smt_dependence
from dace.transformation.passes.analysis.smt_dependence import z3

#: What :func:`collect_enclosing_conditions` returns for an unguarded block.
UNGUARDED = symbolic.pystr_to_symbolic("True")

#: Rounds of bound-symbol substitution before a subset counts as not expressible over the iterator.
SUBSTITUTION_DEPTH = 8


#: The kinds are the IR's (:class:`~dace.symbolic_facts.DataFactKind`), so declared and derived facts are one thing.
FactKind = symbolic.DataFactKind


@dataclass(frozen=True, slots=True)
class ArrayFact:
    """A property of ``array``'s contents; ``origin`` names the blocks (or declaration) that establish it."""

    kind: FactKind
    array: str
    lo: Any = None
    hi: Any = None
    sentinel: int | None = None
    origin: str = ""
    segments: str | None = None

    def describe(self) -> str:
        text = symbolic.DataFact(self.kind, self.lo, self.hi, self.sentinel, self.segments).text(self.array)
        return f"{text} ({self.origin})" if self.origin else text


# ---------------------------------------------------------------------------------------------------------------------
# Producer side


def states_of(block: ControlFlowBlock) -> list[SDFGState]:
    if isinstance(block, SDFGState):
        return [block]
    return [b for b in block.all_control_flow_blocks() if isinstance(b, SDFGState)]


def write_edges(block: ControlFlowBlock | SDFG, name: str) -> list[tuple[SDFGState, Any]]:
    """Every non-empty edge into an access node of ``name`` in ``block``."""
    return [
        (st, e)
        for st in states_of(block)
        for dn in st.data_nodes()
        if dn.data == name
        for e in st.in_edges(dn)
        if e.data is not None and not e.data.is_empty()
    ]


def writes(block: ControlFlowBlock, name: str) -> bool:
    return any(dn.data == name and is_written(st, dn) for st in states_of(block) for dn in st.data_nodes())


def assigns(block: ControlFlowBlock, name: str) -> bool:
    """Whether an interstate edge inside ``block``, or leaving it, assigns ``name``."""
    region = block.parent_graph
    if any(name in e.data.assignments for e in region.out_edges(block)):
        return True
    return not isinstance(block, SDFGState) and name in assigned_in(block)


def preceding_chain(block: ControlFlowBlock) -> tuple[list[ControlFlowBlock], bool]:
    """The blocks that run before ``block`` on every path to it, nearest first, and whether the walk reached the
    SDFG's entry. It stops at a merge and at the start of an enclosing loop's body: what ran before either is not
    known at ``block``."""
    chain: list[ControlFlowBlock] = []
    current = block
    while True:
        region = current.parent_graph
        preds = region.predecessors(current)
        if len(preds) > 1:
            return chain, False
        if preds:
            current = preds[0]
            chain.append(current)
            continue
        if isinstance(region, SDFG):
            return chain, True
        if isinstance(region, LoopRegion):
            return chain, False
        owner = region.parent_graph
        current = owner if isinstance(owner, ConditionalBlock) else region


def inner_iterators(state: SDFGState, stop: ControlFlowRegion) -> set[str]:
    """The iterators of the loops other than ``stop`` that enclose ``state`` below ``stop``."""
    names: set[str] = set()
    region = state.parent_graph
    while region is not None and region is not stop:
        if isinstance(region, LoopRegion):
            names.add(region.loop_variable)
        region = region.parent_graph
    return names


def enclosing_loops(state: SDFGState, stop: ControlFlowRegion) -> bool:
    """Whether a loop other than ``stop`` encloses ``state`` below ``stop``."""
    region = state.parent_graph
    while region is not None and region is not stop:
        if isinstance(region, LoopRegion):
            return True
        region = region.parent_graph
    return False


def runs_once(state: SDFGState, top: ControlFlowRegion) -> bool:
    """Whether ``state`` runs exactly once each time ``top`` does: every region between them is an unconditional
    chain, neither a loop nor a branch."""
    region = state.parent_graph
    while region is not top:
        if region is None or isinstance(region, LoopRegion) or isinstance(region.parent_graph, ConditionalBlock):
            return False
        if not straight(region):
            return False
        region = region.parent_graph
    return straight(top)


def straight(region: ControlFlowRegion) -> bool:
    """Whether ``region``'s blocks form one chain of unconditional edges."""
    return all(region.in_degree(b) <= 1 and region.out_degree(b) <= 1 for b in region.nodes()) and all(
        e.data.is_unconditional() for e in region.edges()
    )


def assignment_value(tasklet: nodes.Tasklet) -> str | None:
    """The right-hand side of a tasklet that is one assignment to its single output, else ``None``."""
    if len(tasklet.out_connectors) != 1:
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
    return ast.unparse(body[0].value)


def integer_constant(text: str) -> int | None:
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def producer(state: SDFGState, edge: Any) -> tuple[Any, Any]:
    """The node computing the value ``edge`` stores, looking through map exits, and the edge leaving it."""
    if isinstance(edge.src, nodes.MapExit):
        inner = [e for e in state.in_edges(edge.src) if e.dst_conn == "IN_" + edge.src_conn[len("OUT_") :]]
        if len(inner) != 1:
            return None, None
        return producer(state, inner[0])
    return edge.src, edge


def loop_range(loop: LoopRegion) -> tuple[Any, Any, Any] | None:
    start, end = loop_analysis.get_init_assignment(loop), loop_analysis.get_loop_end(loop)
    step = loop_analysis.get_loop_stride(loop)
    if start is None or end is None or step is None:
        return None
    return start, end, step


def assigned_in(region: ControlFlowRegion) -> Counter:
    """How many interstate edges inside ``region`` assign each name."""
    counts: Counter = Counter()
    for edge in region.all_interstate_edges():
        counts.update(edge.data.assignments.keys())
    return counts


def covers_whole(subset: Any, desc: dt.Data) -> bool:
    from dace.transformation.passes.analysis.analysis import covers_full_extent

    return covers_full_extent(subset, desc)


class Store(NamedTuple):
    """The one store into an array a loop or map iteration performs: its element, and what it stores, either
    ``value`` (an expression of the iterator and of names the iteration does not assign) or the unchanged contents
    of the single-element container ``source``."""

    itervar: str
    bounds: tuple[Any, Any, Any]
    index: Any
    value: Any
    source: str | None
    once: bool


def stored(state: SDFGState, edge: Any) -> tuple[Any, str | None] | None:
    """``(value, source)`` of the tasklet or copy feeding ``edge``, see :class:`Store`."""
    node = edge.src
    if isinstance(node, nodes.AccessNode):
        return None, node.data
    if not isinstance(node, nodes.Tasklet):
        return None
    text = assignment_value(node)
    if text is None:
        return None
    if not node.in_connectors:
        value = symbolic.pystr_to_symbolic(text)
        return (None if value.atoms(symbolic.Subscript) else value), None
    if len(node.in_connectors) != 1 or text != next(iter(node.in_connectors)):
        return None
    sources = [e for e in state.in_edges(node) if e.dst_conn == text]
    if len(sources) != 1 or sources[0].data.subset.num_elements() != 1:
        return None
    return None, sources[0].data.data


def nested_store(node: nodes.NestedSDFG, conn: str) -> tuple[Any, Any, bool] | None:
    """``(index, value, once)`` of the one store ``node``'s body performs into ``conn``, in the caller's names."""
    edges = write_edges(node.sdfg, conn)
    if len(edges) != 1:
        return None
    state, edge = edges[0]
    if state.entry_node(edge.src) is not None or edge.data.subset.num_elements() != 1 or edge.data.wcr is not None:
        return None
    if enclosing_loops(state, node.sdfg):
        return None
    what = stored(state, edge)
    if what is None or what[0] is None:
        return None
    mapping = {symbolic.symbol(k): symbolic.pystr_to_symbolic(str(v)) for k, v in node.symbol_mapping.items()}
    index = symbolic.pystr_to_symbolic(str(edge.data.subset[0][0]))
    value = what[0]
    if any(str(s) not in node.symbol_mapping for s in set(index.free_symbols) | set(value.free_symbols)):
        return None
    return index.subs(mapping), value.subs(mapping), runs_once(state, node.sdfg)


def iteration_store(block: ControlFlowBlock, name: str) -> Store | None:
    """The :class:`Store` of ``block`` into ``name``: a loop whose body, or a state whose one-dimensional map,
    stores into ``name`` through one edge, at most once per iteration."""
    edges = write_edges(block, name)
    if len(edges) != 1:
        return None
    state, edge = edges[0]
    if edge.data.wcr is not None:
        return None
    if isinstance(block, LoopRegion):
        bounds = loop_range(block)
        if bounds is None or enclosing_loops(state, block) or state.entry_node(edge.src) is not None:
            return None
        if edge.data.subset.num_elements() != 1:
            return None
        what = stored(state, edge)
        if what is None:
            return None
        index = symbolic.pystr_to_symbolic(str(edge.data.subset[0][0]))
        return Store(block.loop_variable, bounds, index, what[0], what[1], runs_once(state, block))
    if not isinstance(block, SDFGState) or not isinstance(edge.src, nodes.MapExit):
        return None
    node, inner = producer(state, edge)
    entry = state.entry_node(node) if node is not None else None
    if entry is None or state.entry_node(entry) is not None or len(entry.map.params) != 1:
        return None
    param = entry.map.params[0]
    bounds = entry.map.range[0]
    if isinstance(node, nodes.NestedSDFG):
        found = nested_store(node, inner.src_conn)
        if found is None:
            return None
        index, value, once = found
        return Store(param, bounds, index, value, None, once)
    if inner.data.subset.num_elements() != 1:
        return None
    what = stored(state, inner)
    if what is None:
        return None
    index = symbolic.pystr_to_symbolic(str(inner.data.subset[0][0]))
    return Store(param, bounds, index, what[0], what[1], True)


def fill_value(block: ControlFlowBlock, name: str, desc: dt.Data) -> int | None:
    """The constant ``block`` sets every element of the one-dimensional ``name`` to, when that is all it stores."""
    if len(desc.shape) != 1:
        return None
    edges = write_edges(block, name)
    if len(edges) == 1 and isinstance(edges[0][1].src, FillLibraryNode):
        state, edge = edges[0]
        node = edge.src
        if node.value_edge(state) is not None or not covers_whole(edge.data.subset, desc):
            return None
        return integer_constant(repr(node.value))
    store = iteration_store(block, name)
    if store is None or not store.once or store.value is None or not store.value.is_Integer:
        return None
    if str(store.index) != store.itervar or not covers_whole(subsets.Range([store.bounds]), desc):
        return None
    return int(store.value)


def scatter_store(block: ControlFlowBlock, name: str) -> Store | None:
    """The :class:`Store` of a loop or map storing into ``name``, at most once per iteration and at any element, a
    value of its iterator and of names it does not assign."""
    store = iteration_store(block, name)
    if store is None or store.value is None:
        return None
    assigned = assigned_in(block) if isinstance(block, LoopRegion) else Counter()
    if any(str(s) in assigned for s in store.value.free_symbols):
        return None
    return store


def injective_fact(chain: list[ControlFlowBlock], name: str, desc: dt.Data) -> ArrayFact | None:
    """``INJECTIVE`` from the nearest writers of ``name``: a scatter after a constant fill.

    Each surviving non-sentinel entry was stored by a distinct iteration, since every iteration stores at most one
    entry and the value names its iteration; overwrites only drop values. The solver proves that the value is
    injective in the iterator and never equals the fill constant."""
    writers = [b for b in chain if writes(b, name)]
    if len(writers) < 2:
        return None
    store = scatter_store(writers[0], name)
    sentinel = fill_value(writers[1], name, desc)
    if store is None or sentinel is None:
        return None
    start, end, step = store.bounds
    if smt_dependence.prove_injective_write(store.value, store.itervar, start, end, step) is not True:
        return None
    if prove_never_equal(store.value, sentinel, store.itervar, start, end, step) is not True:
        return None
    lo, hi = value_range(store.value, store.itervar, start, end)
    origin = f"scatter {writers[0].label} after fill {writers[1].label}"
    return ArrayFact(FactKind.INJECTIVE, name, lo, hi, sentinel, origin)


def prove_never_equal(value: Any, constant: int, itervar: str, start: Any, end: Any, step: Any) -> bool | None:
    """Whether ``value`` differs from ``constant`` on every iteration of ``[start, end]`` by ``step``."""
    z = required(z3)
    i = z.Int(itervar)
    term = smt_dependence._sympy_to_z3(value, {itervar: i}, {})
    domain = smt_dependence._iter_bounds(i, start, end, step)
    if term is None or not domain or not z.is_int(term):
        return None
    return smt_dependence.prove_unsat(z.And(*domain), term != constant)


def value_range(value: Any, itervar: str, start: Any, end: Any) -> tuple[Any, Any]:
    """``(lo, hi)`` of an affine ``value`` over ``[start, end]`` with a constant slope, else ``(None, None)``."""
    itersym = symbolic.symbol(itervar)
    expr = symbolic.pystr_to_symbolic(str(value))
    slope = symbolic.simplify(expr.diff(itersym))
    if not slope.is_Integer or slope == 0:
        return None, None
    first, last = expr.subs(itersym, start), expr.subs(itersym, end)
    return (first, last) if slope > 0 else (last, first)


def accumulator(store: Store) -> str | None:
    """The container or symbol whose current value ``store`` stores unchanged."""
    if store.source is not None:
        return store.source
    if store.value is not None and store.value.is_Symbol:
        return str(store.value)
    return None


def adds_nonnegative(expr: ast.AST, own: set[str], names: set[str]) -> bool:
    """Whether ``expr`` is ``x + d`` (either order) with ``x`` one of ``own`` and ``d`` nonnegative."""
    if not (isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add)):
        return False
    for x, d in ((expr.left, expr.right), (expr.right, expr.left)):
        if isinstance(x, ast.Name) and x.id in own and nonnegative_expr(d, names):
            return True
    return False


def only_grows(loop: LoopRegion, acc: str, nonneg: set[str]) -> bool:
    """Whether every update of the container or symbol ``acc`` inside ``loop`` adds a nonnegative value to it."""
    for edge in loop.all_interstate_edges():
        value = edge.data.assignments.get(acc)
        if value is None:
            continue
        try:
            expr = ast.parse(value, mode="eval").body
        except SyntaxError:
            return False
        if not adds_nonnegative(expr, {acc}, nonneg):
            return False
    for state, edge in write_edges(loop, acc):
        node = edge.src
        if not isinstance(node, nodes.Tasklet) or edge.data.wcr is not None:
            return False
        value = assignment_value(node)
        if value is None:
            return False
        sources = {e.dst_conn: e.data.data for e in state.in_edges(node) if e.dst_conn}
        own = {c for c, d in sources.items() if d == acc}
        names = {c for c, d in sources.items() if d in nonneg} | (nonneg - set(sources))
        try:
            if not adds_nonnegative(ast.parse(value, mode="eval").body, own, names):
                return False
        except SyntaxError:
            return False
    return True


def scan_fact(state: SDFGState, name: str, nonneg: set[str]) -> ArrayFact | None:
    """``NONDECREASING`` over the elements a prefix ``Scan``, ``state``'s one store into ``name``, writes: a running
    maximum, or a running sum of nonnegative values (a seed or an exclusive scan's identity only shifts it)."""
    edges = write_edges(state, name)
    if len(edges) != 1 or not isinstance(edges[0][1].src, Scan):
        return None
    edge = edges[0][1]
    node = edge.src
    if node.chains != 1 or segmented(node) or not symbolic.equal_valued(1, node.stride):
        return None
    if node.op is ScanOp.SUM:
        inputs = [e.data.data for e in state.in_edges(node) if e.dst_conn == Scan.INPUT_CONNECTOR_NAME]
        if len(inputs) != 1 or not (
            inputs[0] in nonneg or any(f.kind is FactKind.NONNEGATIVE for f in declared_facts(state, inputs[0]))
        ):
            return None
    elif node.op is not ScanOp.MAX:
        return None
    subset = edge.data.dst_subset or edge.data.subset
    if subset.dims() != 1 or subset[0][2] != 1:
        return None
    return ArrayFact(
        FactKind.NONDECREASING, name, subset[0][0], subset[0][1], None, f"prefix {node.op.value} {node.label}"
    )


def nondecreasing_fact(chain: list[ControlFlowBlock], name: str, desc: dt.Data, nonneg: set[str]) -> ArrayFact | None:
    """``NONDECREASING`` from the nearest writers of ``name``: a prefix ``Scan`` (:func:`scan_fact`), or a running-sum
    loop storing the sum at ``i + b`` on every iteration, optionally followed by one store of the final sum at
    ``end + b + 1``."""
    if len(desc.shape) != 1:
        return None
    writers = [b for b in chain if writes(b, name)]
    if writers and isinstance(writers[0], SDFGState) and (fact := scan_fact(writers[0], name, nonneg)) is not None:
        return fact
    tail = writers[0] if writers and isinstance(writers[0], SDFGState) else None
    if tail is not None:
        writers = writers[1:]
    if not writers or not isinstance(writers[0], LoopRegion):
        return None
    loop = writers[0]
    store = iteration_store(loop, name)
    if store is None or not store.once or store.bounds[2] != 1:
        return None
    start, end, _ = store.bounds
    offset = symbolic.simplify(store.index - symbolic.symbol(store.itervar))
    if any(str(s) == store.itervar or str(s) in assigned_in(loop) for s in offset.free_symbols):
        return None
    acc = accumulator(store)
    if acc is None or not only_grows(loop, acc, nonneg):
        return None
    lo, hi = start + offset, end + offset
    origin = f"running sum {loop.label}"
    if tail is not None:
        edges = write_edges(tail, name)
        if len(edges) != 1 or edges[0][1].data.subset.num_elements() != 1 or edges[0][1].data.wcr is not None:
            return None
        state, edge = edges[0]
        what = stored(state, edge)
        source = (
            None if what is None else (what[1] or (str(what[0]) if what[0] is not None and what[0].is_Symbol else None))
        )
        between = chain[chain.index(tail) + 1 : chain.index(loop) + 1]
        if source != acc or any(writes(b, acc) or assigns(b, acc) for b in between if b is not loop):
            return None
        if assigns(loop, acc) and any(acc in e.data.assignments for e in loop.parent_graph.out_edges(loop)):
            return None
        index = symbolic.pystr_to_symbolic(str(edge.data.subset[0][0]))
        if symbolic.simplify(index - (hi + 1)) != 0:
            return None
        hi = hi + 1
        origin += f" and its total in {tail.label}"
    return ArrayFact(FactKind.NONDECREASING, name, lo, hi, None, origin)


def nonnegative_expr(expr: ast.AST, names: set[str]) -> bool:
    """Whether ``expr`` is a nonnegative constant, one of ``names`` or an element of one, or a sum, product,
    ``min``, ``max`` or conditional choice of those."""
    if isinstance(expr, ast.Constant):
        return isinstance(expr.value, (int, float)) and not isinstance(expr.value, bool) and expr.value >= 0
    if isinstance(expr, ast.Name):
        return expr.id in names
    if isinstance(expr, ast.Subscript):
        return isinstance(expr.value, ast.Name) and expr.value.id in names
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, (ast.Add, ast.Mult)):
        return nonnegative_expr(expr.left, names) and nonnegative_expr(expr.right, names)
    if isinstance(expr, ast.IfExp):
        return nonnegative_expr(expr.body, names) and nonnegative_expr(expr.orelse, names)
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id in ("min", "max") and expr.args:
        verdicts = [nonnegative_expr(a, names) for a in expr.args]
        return all(verdicts) if expr.func.id == "min" else any(verdicts)
    return False


def parses_nonnegative(text: str, names: set[str]) -> bool:
    try:
        return nonnegative_expr(ast.parse(text, mode="eval").body, names)
    except SyntaxError:
        return False


def nonnegative_containers(
    sdfg: SDFG, assumed: frozenset[str] = frozenset(), outputs: frozenset[str] = frozenset()
) -> set[str]:
    """The transients and interstate symbols of ``sdfg`` (and ``outputs``) every store into or assignment of which
    is nonnegative, assuming the same of what it reads and of ``assumed``.

    The largest such set: it starts from every candidate and drops each one with a store or assignment that is not
    provably nonnegative under the current set, until none drops. A loop iterator counts when it starts nonnegative
    and only grows. Holds for every store the program performs, so at every point. A nested SDFG's store is followed
    inside it; a library node other than a constant fill drops the target.
    """
    containers = {name for name, desc in sdfg.arrays.items() if desc.transient} | set(outputs)
    stores: dict[str, list[tuple[SDFGState, Any]]] = {name: [] for name in containers}
    for state in sdfg.states():
        for dn in state.data_nodes():
            if dn.data in stores:
                stores[dn.data].extend((state, e) for e in state.in_edges(dn) if not e.data.is_empty())
    assignments: dict[str, list[str]] = {}
    for edge in sdfg.all_interstate_edges():
        for name, value in edge.data.assignments.items():
            assignments.setdefault(name, []).append(value)
    for loop in sdfg.all_control_flow_regions():
        if isinstance(loop, LoopRegion) and loop.loop_variable:
            start = loop_analysis.get_init_assignment(loop)
            step = loop_analysis.get_loop_stride(loop)
            ok = start is not None and step is not None and step.is_Integer and step > 0
            assignments.setdefault(loop.loop_variable, []).append(str(start) if ok else "-1")
    candidates = containers | set(assignments)

    def stores_nonnegative(state: SDFGState, edge: Any, known: set[str]) -> bool:
        if edge.data.wcr is not None and "+" not in edge.data.wcr:
            return False
        node, inner = producer(state, edge)
        if isinstance(node, nodes.AccessNode):
            return node.data in known
        if isinstance(node, FillLibraryNode):
            constant = integer_constant(repr(node.value))
            return node.value_edge(state) is None and constant is not None and constant >= 0
        if isinstance(node, Scan):
            # Sums, products, minima and maxima of nonnegative values, from a nonnegative seed, are nonnegative
            seeded = not node.exclusive or node.identity is None or parses_nonnegative(str(node.identity), known)
            reads = [e.data.data for e in state.in_edges(node) if not e.data.is_empty()]
            return node.op is not ScanOp.AFFINE and seeded and all(data in known for data in reads)
        if isinstance(node, nodes.NestedSDFG):
            inputs = {e.dst_conn for e in state.in_edges(node) if e.dst_conn and e.data.data in known}
            inputs |= {k for k, v in node.symbol_mapping.items() if parses_nonnegative(str(v), known)}
            conn = inner.src_conn
            return conn in nonnegative_containers(node.sdfg, frozenset(inputs), frozenset({conn}))
        if not isinstance(node, nodes.Tasklet):
            return False
        value = assignment_value(node)
        if value is None:
            return False
        conns = {e.dst_conn: e.data.data for e in state.in_edges(node) if e.dst_conn}
        names = {c for c, d in conns.items() if d in known} | (known - set(conns))
        return parses_nonnegative(value, names)

    known = candidates | set(assumed)
    changed = True
    while changed:
        changed = False
        for name in sorted(candidates & known):
            ok = all(parses_nonnegative(v, known) for v in assignments.get(name, ())) and all(
                stores_nonnegative(st, e, known) for st, e in stores.get(name, ())
            )
            if not ok:
                known.discard(name)
                changed = True
    return known - (set(assumed) - candidates)


def extreme(value: Any, ranges: list[tuple[str, Any, Any]], upper: bool) -> Any | None:
    """The largest (``upper``) or smallest value of ``value`` while each ``(name, lo, hi)`` of ``ranges``, innermost
    first, ranges over ``[lo, hi]`` (which may read the names of the ranges after it); None unless ``value`` is
    affine in each of them with an integer slope."""
    for name, lo, hi in ranges:
        sym = next((s for s in value.free_symbols if str(s) == name), None)
        if sym is None:
            continue
        slope = symbolic.simplify(value.diff(sym))
        if not slope.is_Integer:
            return None
        value = value.subs(sym, hi if (slope > 0) == upper else lo)
    return value


#: The conditional expressions a tasklet's right-hand side spells a select with: ``(condition, then, else)``.
SELECTS = (symbolic.IfExpr, symbolic.ITE)


def integer_polynomial(value: Any, clamps: bool) -> bool:
    """Whether ``value`` is a polynomial with integer constants, or (``clamps``) a ``Min``, ``Max`` or select of such."""
    if clamps and isinstance(value, (sympy.Min, sympy.Max)):
        return all(integer_polynomial(a, clamps) for a in value.args)
    if clamps and isinstance(value, SELECTS):
        return all(integer_polynomial(a, clamps) for a in value.args[1:])
    # A boolean (a select of ``True`` and ``False``) is not a number: nothing to bound
    return (
        isinstance(value, sympy.Expr)
        and value.is_polynomial()
        and all(number.is_Integer for number in value.atoms(sympy.Number))
    )


def stored_value(state: SDFGState, edge: Any, clamps: bool = False) -> Any | None:
    """What ``edge`` stores, when it is the same integer polynomial every time: a constant fill, or the expression a
    tasklet without inputs (or a nested SDFG's one store, see :func:`nested_store`) assigns. A stored float would be
    truncated, a ``Mod`` or ``floor`` has no slope. With ``clamps``, a ``Min``, ``Max`` or select of polynomials
    also counts."""
    node, inner = producer(state, edge)
    if node is None or edge.data.wcr is not None or inner.data.wcr is not None:
        return None
    if isinstance(node, FillLibraryNode):
        constant = integer_constant(repr(node.value))
        return None if node.value_edge(state) is not None or constant is None else sympy.Integer(constant)
    if isinstance(node, nodes.NestedSDFG):
        nested = nested_store(node, inner.src_conn)
        value = None if nested is None else nested[1]
    else:
        text = assignment_value(node) if isinstance(node, nodes.Tasklet) and not node.in_connectors else None
        value = None if text is None else symbolic.pystr_to_symbolic(text)
    return value if isinstance(value, sympy.Expr) and integer_polynomial(value, clamps) else None


def clamped_range(value: Any, ranges: list[tuple[str, Any, Any]]) -> tuple[Any, Any] | None:
    """The smallest and largest value of ``value``, a polynomial or a ``Min``, ``Max`` or select of them, while each of
    ``ranges`` ranges over its bounds (:func:`extreme`); None unless every polynomial in it is affine."""
    if not isinstance(value, (sympy.Min, sympy.Max, *SELECTS)):
        lo, hi = extreme(value, ranges, False), extreme(value, ranges, True)
        return None if lo is None or hi is None else (lo, hi)
    select = isinstance(value, SELECTS)
    parts = [clamped_range(a, ranges) for a in (value.args[1:] if select else value.args)]
    if None in parts:
        return None
    # A select takes either branch; Min and Max are monotone in each argument
    low, high = (sympy.Min, sympy.Max) if select else (type(value), type(value))
    return low(*(p[0] for p in parts)), high(*(p[1] for p in parts))


def scope_ranges(sdfg: SDFG, state: SDFGState, node: nodes.Node) -> list[tuple[str, Any, Any]] | None:
    """``(name, lo, hi)`` of each map parameter and loop iterator around ``node``, innermost first, whose range holds
    and whose step is positive (a negative loop step swaps the bounds); None for a loop whose step sign is
    unknown."""
    ranges = []
    entry = state.entry_node(node)
    while entry is not None:
        if isinstance(entry, nodes.MapEntry):
            for name, (lo, hi, step) in zip(entry.map.params, entry.map.range.ranges, strict=True):
                if as_expr(step).is_positive:
                    ranges.append((name, lo, hi))
        entry = state.entry_node(entry)
    region = state.parent_graph
    while isinstance(region, ControlFlowRegion) and not isinstance(region, SDFG):
        bounds = loop_range(region) if isinstance(region, LoopRegion) and not region.inverted else None
        if bounds is not None and loop_analysis.range_holds(region, bounds, sdfg):
            start, end, step = bounds
            if step.is_positive is None:
                return None
            ranges.append((region.loop_variable, *((start, end) if step.is_positive else (end, start))))
        region = region.parent_graph
    return ranges


def store_bounds(sdfg: SDFG, state: SDFGState, edge: Any, invariant: set[str]) -> tuple[Any, Any] | None:
    """``(lo, hi)`` of every value ``edge`` stores (:func:`stored_value`), affine in the enclosing map parameters and
    loop iterators (:func:`scope_ranges`), at their extremes; None unless both read only ``invariant`` names."""
    value = stored_value(state, edge, clamps=True)
    ranges = None if value is None else scope_ranges(sdfg, state, producer(state, edge)[0])
    bounds = None if ranges is None else clamped_range(value, ranges)
    if bounds is None or any(b.atoms(symbolic.Subscript) for b in bounds):
        return None
    lo, hi = bounds
    return (lo, hi) if {str(s) for b in (lo, hi) for s in b.free_symbols} <= invariant else None


def widest(bounds: list[Any], upper: bool, facts: symbolic.Facts) -> Any | None:
    """The one of ``bounds`` that ``facts`` prove at least (``upper``), or at most, every other one."""
    for c in bounds:
        if all(
            symbolic.ask(symbolic.Relation(symbolic.RelationKind.LE, *((b, c) if upper else (c, b))), facts)
            is symbolic.Truth.TRUE
            for b in bounds
            if b != c
        ):
            return c
    return None


def bounded_facts(sdfg: SDFG) -> dict[str, ArrayFact]:
    """``BOUNDED`` of each one-dimensional integer transient of ``sdfg`` every store into which is bounded
    (:func:`store_bounds`): over the smallest of their lower bounds and the largest of their upper bounds, if the
    SDFG's facts order them. Holds for every value the program stores, so at every point. Derived in one walk over
    the SDFG's states, for the span of one :func:`facts_at` caller, as :func:`nonnegative_containers` is."""
    candidates = {
        name
        for name, desc in sdfg.arrays.items()
        if desc.transient and len(desc.shape) == 1 and desc.dtype in dtypes.INTEGER_TYPES
    }
    # A view or reference stores into a container under another name
    if not candidates or any(isinstance(d, (dt.View, dt.Reference)) for d in sdfg.arrays.values()):
        return {}
    invariant = sdfg.symbols.keys() - interstate_bound_names(sdfg)
    stores: dict[str, list[tuple[Any, Any] | None]] = {}
    for state in sdfg.states():
        for dn in state.data_nodes():
            if dn.data in candidates:
                stores.setdefault(dn.data, []).extend(
                    store_bounds(sdfg, state, e, invariant) for e in state.in_edges(dn) if not e.data.is_empty()
                )
    found: dict[str, ArrayFact] = {}
    facts = None
    for name, bounds in stores.items():
        if not bounds or None in bounds:
            continue
        facts = sdfg.facts() if facts is None else facts
        lo, hi = widest([b[0] for b in bounds], False, facts), widest([b[1] for b in bounds], True, facts)
        if lo is not None and hi is not None:
            found[name] = ArrayFact(FactKind.BOUNDED, name, lo, hi, origin="every store is bounded")
    return found


def facts_at(
    sdfg: SDFG,
    block: ControlFlowBlock,
    names: Iterable[str],
    nonneg: set[str] | None = None,
    bounded: dict[str, ArrayFact] | None = None,
) -> list[ArrayFact]:
    """The facts known of each of ``names`` on entry to ``block``: derived from the nearest writers before it, or,
    when nothing before it writes the container, the ones assumed on entry to the SDFG (:func:`declared_facts`).
    ``nonneg`` and ``bounded`` are :func:`nonnegative_containers` and :func:`bounded_facts` of ``sdfg``, when the
    caller holds them while ``sdfg`` is unchanged; derived here on first need otherwise."""
    chain, _ = preceding_chain(block)
    found: list[ArrayFact] = []
    for name in names:
        desc = sdfg.arrays.get(name)
        if desc is None:
            continue
        fact = injective_fact(chain, name, desc)
        if fact is None:
            if nonneg is None:
                nonneg = nonnegative_containers(sdfg)
            fact = nondecreasing_fact(chain, name, desc, nonneg)
        if fact is not None:
            found.append(fact)
        elif nonneg is not None and name in nonneg:
            found.append(ArrayFact(FactKind.NONNEGATIVE, name, origin="every store is nonnegative"))
        if bounded is None:
            bounded = bounded_facts(sdfg)
        if name in bounded:
            found.append(bounded[name])
        found.extend(f for f in declared_facts(block, name) if f not in found)
    return found


def declared_facts(block: ControlFlowBlock, name: str) -> list[ArrayFact]:
    """The facts assumed of ``name`` on entry to ``block`` (:meth:`SymbolResolver.data_facts_at`)."""
    return [
        ArrayFact(f.kind, name, f.lo, f.hi, f.sentinel, "declared", f.segments)
        for f in SymbolResolver().data_facts_at(block, name)
    ]


# ---------------------------------------------------------------------------------------------------------------------
# Consumer side


def sympy_logic(value: Any) -> Any:
    """``value`` with DaCe's ``AND``/``OR`` as sympy's ``And``/``Or``: the same truth value, and one a sympy ``And``
    or ``Not`` guard accepts as an argument (it refuses a DaCe function with a ``TypeError``)."""
    if not isinstance(value, sympy.Basic):
        return value
    return value.replace(symbolic.AND, sympy.And).replace(symbolic.OR, sympy.Or)


def bound_expression(expr: Any, bindings: dict[str, Any], assigned: Counter) -> Any | None:
    """``expr`` with every name the loop assigns once replaced by its value, until only the iterator and names the
    loop does not assign remain; ``None`` if a name the loop assigns more than once is left."""
    for _ in range(SUBSTITUTION_DEPTH):
        names = {str(s) for s in expr.free_symbols} | {str(a.args[0]) for a in expr.atoms(symbolic.Subscript)}
        pending = {n: sympy_logic(bindings[n]) for n in names if n in bindings}
        if not pending:
            break
        try:
            expr = expr.subs({symbolic.symbol(n): v for n, v in pending.items()})
        except TypeError:
            # A guard name bound to a non-boolean value (``if_cond = clipping[i]``, a truth test) inside ``And``
            return None
    names = {str(s) for s in expr.free_symbols}
    if any(n in assigned and n not in bindings for n in names) or names & bindings.keys():
        return None
    return expr


def loop_bindings(loop: LoopRegion) -> tuple[dict[str, Any], Counter]:
    """The names ``loop``'s interstate edges assign exactly once, with their values, and the assignment counts."""
    assigned = assigned_in(loop)
    bindings = {}
    for edge in loop.all_interstate_edges():
        for name, value in edge.data.assignments.items():
            if assigned[name] == 1:
                try:
                    bindings[name] = symbolic.pystr_to_symbolic(value)
                except Exception:
                    continue
    return bindings, assigned


def stored_accesses(state: SDFGState, edge: Any) -> list[tuple[subsets.Subset, Any]]:
    """What the store ``edge`` really writes, each with the branch conditions inside the nested SDFG it runs under (or
    ``None``): the accesses of the nested SDFG behind a map exit (one box each), else the subset of ``edge``."""
    from dace.transformation.interstate.loop_to_map import exact_accesses  # avoid an import loop

    return [(a.subset, a.guard) for a in exact_accesses(state, edge) or []] or [(edge.data.subset, None)]


def write_bounds(
    loop: LoopRegion,
    state: SDFGState,
    edge: Any,
    bindings: dict[str, Any],
    assigned: Counter,
    subset: subsets.Subset,
    inner: Any = None,
) -> tuple[list[tuple[Any, Any]], Any] | None:
    """``(box, guard)`` of the store of ``subset`` through ``edge``, ``box`` its ``(lo, hi)`` per dimension, over the
    iterator and names ``loop`` does not assign, else ``None``."""
    if edge.data.wcr is not None:
        return None
    box = []
    for rb, re_, stride in subset.ndrange():
        lo = bound_expression(symbolic.pystr_to_symbolic(str(rb)), bindings, assigned)
        hi = bound_expression(symbolic.pystr_to_symbolic(str(re_)), bindings, assigned)
        if lo is None or hi is None:
            return None
        box.append((lo, hi))
    guard = collect_enclosing_conditions(state, stop=loop)
    if guard != UNGUARDED:
        guard = bound_expression(guard, bindings, assigned)
        if guard is None:
            return None
    if inner is not None:
        # The branches inside the nested SDFG the access sits in
        inner = bound_expression(inner, bindings, assigned)
        if inner is not None:
            guard = inner if guard == UNGUARDED else symbolic.pystr_to_symbolic(f"({guard}) and ({inner})")
    # An inner loop repeats the access at the same place: the store is the same box in every one of its iterations
    # unless its index or branch names the inner iterator
    around = inner_iterators(state, loop)
    named = {str(x) for lo, hi in box for x in (*lo.free_symbols, *hi.free_symbols)}
    if guard != UNGUARDED:
        named |= {str(x) for x in guard.free_symbols}
    if around & named:
        return None
    return (box, None) if guard == UNGUARDED else (box, guard)


def resolved_range(loop: LoopRegion) -> tuple[Any, Any, Any] | None:
    """``loop``'s range with each name its bounds read replaced by the value the nearest assignment on every path to
    the loop gives it, while nothing between that assignment and the loop changes what the value reads."""
    bounds = loop_range(loop)
    if bounds is None:
        return None
    chain, _ = preceding_chain(loop)
    values: dict[str, Any] = {}
    changed: set[str] = set()
    for block in [loop, *chain]:
        for edge in block.parent_graph.in_edges(block):
            for name, text in edge.data.assignments.items():
                if name in values or name in changed:
                    continue
                value = symbolic.pystr_to_symbolic(text)
                reads = {str(x) for x in value.free_symbols} | {str(a.args[0]) for a in value.atoms(symbolic.Subscript)}
                if not reads & changed:
                    values[name] = value
            changed |= set(edge.data.assignments)
        if block is not loop:
            changed |= {dn.data for st in states_of(block) for dn in st.data_nodes() if is_written(st, dn)}
            if not isinstance(block, SDFGState):
                changed |= set(assigned_in(block))
    start, end, step = bounds
    mapping = {symbolic.symbol(n): v for n, v in values.items()}
    return start.subs(mapping), end.subs(mapping), step


def read_bounds(
    loop: LoopRegion, name: str, bindings: dict[str, Any], assigned: Counter
) -> list[tuple[list[tuple[Any, Any]], Any]] | None:
    """``(box, guard)`` of every read of ``name`` in ``loop``, as :func:`write_bounds` gives them for stores; ``None`` if
    one cannot be written over the iterator and names ``loop`` does not assign, or a read sits in an interstate edge or
    a header, which has no access node to bound."""
    arrays = loop.sdfg.arrays
    if any(m.data == name for e in loop.all_interstate_edges() for m in e.data.get_read_memlets(arrays)):
        return None
    headers = [loop, *(b for b in loop.all_control_flow_blocks() if isinstance(b, (LoopRegion, ConditionalBlock)))]
    if any(m.data == name for b in headers for m in b.get_meta_read_memlets()):
        return None
    from dace.transformation.interstate.loop_to_map import exact_accesses  # avoid an import loop

    found = []
    for st in states_of(loop):
        for dn in st.data_nodes():
            if dn.data != name:
                continue
            for e in st.out_edges(dn):
                if e.data is None or e.data.is_empty():
                    continue
                for subset, inner in [(a.subset, a.guard) for a in exact_accesses(st, e) or []] or [
                    (e.data.get_src_subset(e, st), None)
                ]:
                    read = None if subset is None else write_bounds(loop, st, e, bindings, assigned, subset, inner)
                    if read is None:
                        return None
                    found.append(read)
    return found


def constant_table(sdfg: SDFG, name: str) -> tuple[int, ...] | None:
    """The values of the one-dimensional integer constant ``name`` of ``sdfg`` or an SDFG around it, else ``None``."""
    while sdfg is not None:
        if name in sdfg.constants:
            values = np.asarray(sdfg.constants[name])
            return (
                tuple(int(v) for v in values) if values.ndim == 1 and np.issubdtype(values.dtype, np.integer) else None
            )
        sdfg = sdfg.parent_sdfg
    return None


def runs_at_most_once(loop: LoopRegion, bounds: tuple[Any, Any, Any]) -> bool:
    """Whether ``loop``, of unit stride up or down, provably runs at most once. No two of its iterations exist, so any
    proof that none of them store alike holds for nothing, and a solver verdict on it says nothing about the body."""
    start, end, step = bounds
    if step not in (1, -1):
        return False
    low, high = (end, start) if step == 1 else (start, end)
    relation = symbolic.Relation(symbolic.RelationKind.LE, low, high)
    return symbolic.ask(relation, loop.sdfg.facts()) is symbolic.Truth.TRUE


def conflict_free_under_facts(sdfg: SDFG, loop: LoopRegion, name: str) -> list[ArrayFact] | None:
    """The facts under which the solver proves that no two iterations of ``loop`` store into one element of
    ``name``, whose reads of other iterations' stores the solver also excludes (:func:`read_bounds`); ``None`` if there are
    none.

    A store's subset may name symbols the loop binds once per iteration from arrays it does not write
    (``row = perm[r]``); they are replaced by their values, as are the names the loop's bounds read
    (:func:`resolved_range`). The branch conditions guarding a store are kept for both iterations, and every array
    read gets the facts known of it on entry to ``loop`` (:func:`facts_at`). Every read of ``name``
    is shown clear of the stores of the other iterations in the same way.
    """
    if not smt_dependence.has_z3():
        return None
    bounds = resolved_range(loop)
    if bounds is None or runs_at_most_once(loop, bounds):
        return None
    reads, written = loop.read_and_write_sets()
    edges = write_edges(loop, name)
    if not edges:
        return None
    bindings, assigned = loop_bindings(loop)
    stores = []
    for state, edge in edges:
        for subset, inner in stored_accesses(state, edge):
            store = write_bounds(loop, state, edge, bindings, assigned, subset, inner)
            if store is None or len(store[0]) != edges[0][1].data.subset.dims():
                return None
            stores.append(store)
    # A read meets a store of another iteration unless the solver shows its box clear of it under both guards: a read at
    # the very subset a store writes is no proof, since another iteration may read it without storing
    seen = read_bounds(loop, name, bindings, assigned) if name in reads else []
    if seen is None:
        return None
    exprs = [e for box, guard in stores + seen for e in [*(x for lh in box for x in lh), guard] if e is not None]
    exprs += list(bounds[:2])
    arrays = {str(a.args[0]) for e in exprs for a in e.atoms(symbolic.Subscript)}
    if arrays & set(written):
        return None
    # A table of compile-time constants (``FoldConstantTables``) is read as it is, not as an array with facts
    tables = {n: table for n in sorted(arrays) if (table := constant_table(sdfg, n)) is not None}
    facts = facts_at(sdfg, loop, sorted(arrays - tables.keys()))
    start, end, step = bounds
    pairs = [(first, second) for x, first in enumerate(stores) for second in stores[x:]]
    pairs += [(read, store) for read in seen for store in stores]
    with canon_stats.timed("z3", sdfg):
        for first, second in pairs:
            if (
                prove_disjoint_under_facts(first, second, loop.loop_variable, start, end, step, facts, tables)
                is not True
            ):
                return None
    return facts


def fact_axioms(facts: list[ArrayFact], arrays: dict[str, Any], terms: list[Any], sym_cache: dict[str, Any]) -> list:
    """The instances of each fact over the array reads appearing in ``terms``."""
    z = required(z3)
    reads: dict[str, list[Any]] = {}
    for term in terms:
        smt_dependence.selects(term, reads)
    axioms = []
    for fact in facts:
        if fact.array not in reads:
            continue
        arr = z.Array(fact.array, z.IntSort(), z.IntSort())
        idx = list({x.sexpr(): x for x in reads[fact.array]}.values())
        bound = [
            None if b is None else smt_dependence._sympy_to_z3(symbolic.pystr_to_symbolic(str(b)), sym_cache, arrays)
            for b in (fact.lo, fact.hi)
        ]
        if any(b is not None and not z.is_int(b) for b in bound):
            continue
        lo, hi = bound
        if fact.kind is FactKind.SEGMENT_INJECTIVE:
            if fact.segments not in reads:
                continue
            seg = z.Array(fact.segments, z.IntSort(), z.IntSort())
            rows = list({r.sexpr(): r for r in reads[fact.segments]}.values())
            for x in idx:
                for y in idx:
                    if x is y:
                        continue
                    for r in rows:
                        inside = z.And(seg[r] <= x, x < seg[r + 1], seg[r] <= y, y < seg[r + 1])
                        axioms.append(z.Implies(z.And(x != y, inside), arr[x] != arr[y]))
            continue
        for x in idx:
            ax = arr[x]
            if fact.kind is FactKind.NONNEGATIVE:
                axioms.append(ax >= 0)
            elif fact.kind is FactKind.BOUNDED:
                axioms += [c for c in (lo is None or lo <= ax, hi is None or ax <= hi) if c is not True]
            elif fact.kind is FactKind.INJECTIVE and lo is not None:
                inside = z.And(lo <= ax, ax <= hi)
                axioms.append(inside if fact.sentinel is None else z.Or(ax == fact.sentinel, inside))
            for y in idx:
                if x is y:
                    continue
                ay = arr[y]
                if fact.kind is FactKind.INJECTIVE:
                    premise = x != y if fact.sentinel is None else z.And(x != y, ax != fact.sentinel)
                    axioms.append(z.Implies(premise, ax != ay))
                elif fact.kind is FactKind.NONDECREASING:
                    axioms.append(z.Implies(z.And(lo <= x, x <= y, y <= hi), ax <= ay))
    return axioms


def prove_disjoint_under_facts(
    first: tuple[list[tuple[Any, Any]], Any],
    second: tuple[list[tuple[Any, Any]], Any],
    itervar: str,
    start: Any,
    end: Any,
    step: Any,
    facts: list[ArrayFact],
    tables: dict[str, tuple[int, ...]] | None = None,
) -> bool | None:
    """Whether a store ``first`` of one iteration and a store ``second`` of another, each ``(box, guard)`` and each
    passing its guard, write disjoint boxes, given ``facts`` about the arrays they and the loop bounds read and the
    ``tables`` of constants they read, which hold exactly their values."""
    z = required(z3)
    sym_cache: dict[str, Any] = {}
    arrays: dict[str, Any] = {}
    sides = []
    for suffix, (box, guard) in (("_1", first), ("_2", second)):
        i = z.Int(itervar + suffix)
        sym_cache[itervar] = i
        dims = [
            (smt_dependence._sympy_to_z3(lo, sym_cache, arrays), smt_dependence._sympy_to_z3(hi, sym_cache, arrays))
            for lo, hi in box
        ]
        cond = True if guard is None else smt_dependence._bool_to_z3(guard, sym_cache, arrays)
        domain = smt_dependence._iter_bounds(i, start, end, step)
        if cond is None or not domain or any(x is None or not z.is_int(x) for lh in dims for x in lh):
            return None
        sides.append((i, dims, cond, domain))
    del sym_cache[itervar]
    (i1, d1, c1, b1), (i2, d2, c2, b2) = sides
    conds = [c for c in (c1, c2) if c is not True]
    terms = [x for lh in d1 + d2 for x in lh] + conds + list(b1) + list(b2)
    axioms = fact_axioms(facts, arrays, terms, sym_cache)
    reads: dict[str, list[Any]] = {}
    for term in terms:
        smt_dependence.selects(term, reads)
    for name, values in (tables or {}).items():
        table = smt_dependence.table_term(values)
        for x in reads.get(name, ()):
            axioms += [0 <= x, x < len(values), z.Select(arrays[name], x) == z.Select(table, x)]
    overlap = z.And(*(z.And(lo1 <= hi2, lo2 <= hi1) for (lo1, hi1), (lo2, hi2) in zip(d1, d2)))
    antecedent = z.And(i1 != i2, *b1, *b2, *conds, *axioms)
    return smt_dependence.prove_unsat(antecedent, z.Not(overlap))
