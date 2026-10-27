# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Range-privatize the data-dependent scatter reductions of parallel CPU maps.

A parallel map accumulating ``A[f(i, B), ...] (op)= v`` would otherwise lower to an OpenMP array-section
``reduction(op:A[0:n])``: every thread fills and combines a private copy of all of ``A``. Range privatization
gives each thread a copy of only the rows ``[lo, hi]`` its static chunk writes, found by a pre-pass that
re-evaluates the index computation of every write over the chunk; the copies are combined by row owner.
The pass owns the analysis and sets ``Map.omp_range_privatize``; ``CPUCodeGen`` renders it.
"""

import ast
from dataclasses import dataclass, field
from typing import Any

from dace import SDFG, data, dtypes, properties, symbolic
from dace.frontend import operations
from dace.sdfg import nodes
from dace.sdfg.state import (
    BreakBlock,
    ConditionalBlock,
    ContinueBlock,
    ControlFlowRegion,
    LoopRegion,
    ReturnBlock,
    SDFGState,
)
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation

#: Operators the lowering combines: each has an identity and a ``wcr_fixed`` update.
RANGE_PRIVATIZABLE_OPS = frozenset(
    {
        dtypes.ReductionType.Sum,
        dtypes.ReductionType.Product,
        dtypes.ReductionType.Min,
        dtypes.ReductionType.Max,
        dtypes.ReductionType.Logical_And,
        dtypes.ReductionType.Logical_Or,
        dtypes.ReductionType.Bitwise_And,
        dtypes.ReductionType.Bitwise_Or,
        dtypes.ReductionType.Bitwise_Xor,
    }
)

#: Operators that need an ordered element type.
ORDERED_OPS = frozenset({dtypes.ReductionType.Min, dtypes.ReductionType.Max})

#: Functions an index computation may call: pure functions of their arguments.
PURE_CALLS = frozenset(
    {
        "abs",
        "min",
        "max",
        "int",
        "float",
        "bool",
        "round",
        "Abs",
        "Min",
        "Max",
        "Mod",
        "floor",
        "ceil",
        "trunc",
        "int_floor",
        "int_ceil",
        "py_mod",
        "py_floor",
        "sqrt",
        "exp",
        "log",
        "log2",
        "log10",
        "sin",
        "cos",
        "tan",
        "pow",
        "fmod",
    }
)

#: Expression nodes an index computation may contain besides names, subscripts and calls.
PURE_NODES = (
    ast.Constant,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    ast.Tuple,
    ast.Load,
    ast.operator,
    ast.unaryop,
    ast.boolop,
    ast.cmpop,
)


@dataclass(frozen=True)
class Let:
    """``const dtype name = expr;``, ``expr`` symbolic or Python source over earlier names and read-only arrays."""

    name: str
    dtype: dtypes.typeclass
    expr: Any


@dataclass(frozen=True)
class Loop:
    """``for (dtype var = init; cond; var = update) body``, each a Python expression."""

    var: str
    dtype: dtypes.typeclass
    init: str
    cond: str
    update: str
    body: tuple


@dataclass(frozen=True)
class MapLoop:
    """One sequential loop per variable: ``var = begin; var <= end; var += step`` (symbolic, step positive)."""

    vars: tuple[str, ...]
    dtypes: tuple[dtypes.typeclass, ...]
    ranges: tuple[tuple[Any, Any, Any], ...]
    body: tuple


@dataclass(frozen=True)
class Guard:
    """``if (cond) body``, ``cond`` a Python expression."""

    cond: str
    body: tuple


@dataclass(frozen=True)
class Row:
    """The row a write lands in, symbolic over earlier names, and how many writes it stands for."""

    expr: Any
    weight: Any = 1


@dataclass(frozen=True)
class ScatterIndexSources:
    """The rows one accumulator is written at: per write, a statement tree evaluated in one map iteration.

    :ivar sites: One tuple of :class:`Let` / :class:`Loop` / :class:`MapLoop` / :class:`Guard` / :class:`Row`
                 per write; every row written is the value of a :class:`Row` of one of them.
    """

    sites: tuple


class Refuse(Exception):
    """The index computation cannot be re-evaluated."""


def _contiguous_rows(desc: data.Data) -> bool:
    """Whether ``desc`` is a plain C-contiguous array, so its rows are consecutive ``strides[0]``-long spans."""
    if type(desc) is not data.Array or desc.start_offset != 0:
        return False
    expected = 1
    for size, stride in reversed(list(zip(desc.shape, desc.strides))):
        if symbolic.equal(stride, expected) is not True:
            return False
        expected = expected * size
    return symbolic.equal(desc.total_size, expected) is True


def _same_layout(inner: data.Data, outer: data.Data) -> bool:
    """Whether ``inner`` indexes the memory of ``outer`` element for element."""
    if len(inner.shape) != len(outer.shape) or inner.start_offset != outer.start_offset:
        return False
    pairs = list(zip(inner.shape, outer.shape)) + list(zip(inner.strides, outer.strides))
    return all(symbolic.equal(a, b) is True for a, b in pairs)


def _is_scalar(desc: data.Data) -> bool:
    return isinstance(desc, data.Scalar) or (
        type(desc) is data.Array and desc.transient and symbolic.simplify(desc.total_size - 1) == 0
    )


def _single_element(desc: data.Data | None) -> bool:
    """Whether ``desc`` is an array of one element, so any in-bounds subscript reads that element."""
    return type(desc) is data.Array and symbolic.simplify(desc.total_size - 1) == 0


def _sequential_entry(state: SDFGState, entry: nodes.MapEntry) -> bool:
    """A sequential map with a static, positive-step range."""
    if entry.map.schedule != dtypes.ScheduleType.Sequential:
        return False
    if any(not c.startswith("IN_") for c in entry.in_connectors):
        return False
    return all((step > 0) == True for _b, _e, step in entry.map.range)  # noqa: E712


def _pure_call(func: ast.expr) -> bool:
    if isinstance(func, ast.Name):
        return func.id in PURE_CALLS
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        if func.value.id == "dace":
            return func.attr in dtypes.TYPECLASS_BY_NAME
        return func.value.id == "math" and func.attr in PURE_CALLS
    return False


@dataclass(slots=True)
class Nest:
    """One SDFG on the path: the parallel map's own (``node`` None) or a nest called on it."""

    sdfg: SDFG
    node: nodes.NestedSDFG | None
    state: SDFGState  # the state that calls ``node`` (or holds the parallel map)
    parent: "Nest | None"
    frame: int  # the frame the call happens in
    assigned: dict[str, list] = field(default_factory=dict)  # symbol -> [(region, edge)]
    writers: dict[str, list] = field(default_factory=dict)  # data -> [(state, access node)]


@dataclass(slots=True)
class Frame:
    """One level of the path: a region's chain, a loop, a guard or a sequential map."""

    kind: str
    nest: Nest
    stmts: list = field(default_factory=list)
    vars: dict[str, str] = field(default_factory=dict)  # name in the nest -> local
    used: set[str] = field(default_factory=set)  # names of ``vars`` the index computation reads
    order: dict = field(default_factory=dict)  # region frame: chain block -> index
    edges: dict = field(default_factory=dict)  # region frame: chain edge -> index
    payload: Any = None


class SiteSlicer:
    """Slices the index computation of one write out of the scope of a parallel map, as a statement tree."""

    def __init__(self, state: SDFGState, map_entry: nodes.MapEntry, written: set[str]):
        self.state = state
        self.map_entry = map_entry
        self.written = written
        self.frames: list[Frame] = []
        self.memo: dict = {}
        self.locals: dict[str, dtypes.typeclass] = {}
        self.top = Nest(state.sdfg, None, state, None, 0)

    # Names

    def fresh(self, name: str, dtype: dtypes.typeclass) -> str:
        base = f"__rp_{name}"
        local, n = base, 0
        while local in self.locals:
            n += 1
            local = f"{base}_{n}"
        self.locals[local] = dtype
        return local

    def let(self, frame: int, name: str, dtype: dtypes.typeclass, expr: Any) -> str:
        local = self.fresh(name, dtype)
        self.frames[frame].stmts.append(Let(local, dtype, expr))
        return local

    @staticmethod
    def visible(defined: tuple[int, int], at: tuple[int, float]) -> bool:
        return defined[0] < at[0] or (defined[0] == at[0] and defined[1] < at[1])

    def position_of_block(self, nest: Nest, block, before: bool = False) -> tuple[int, int] | None:
        """Position of a chain block of ``nest`` (``before``: of the edge into it), or None if off the path."""
        for i, f in enumerate(self.frames):
            if f.nest is nest and f.kind == "region" and block in f.order:
                return i, 2 * f.order[block] - (1 if before else 0)
        return None

    # Values

    def symbol(self, nest: Nest, name: str, at: tuple[int, float], bound: dict[str, str]) -> str:
        if name in bound:
            return bound[name]
        for i in range(min(at[0], len(self.frames) - 1), -1, -1):
            f = self.frames[i]
            if f.nest is nest and name in f.vars:
                f.used.add(name)
                return f.vars[name]
        if nest.node is None:
            if name in nest.sdfg.arrays:
                raise Refuse(name)
            return name
        key = ("sym", id(nest), name)
        if key not in self.memo:
            self.memo[key] = self.define_symbol(nest, name)
        local, defined = self.memo[key]
        if not self.visible(defined, at):
            raise Refuse(name)
        return local

    def define_symbol(self, nest: Nest, name: str) -> tuple[str, tuple[int, int]]:
        sites = nest.assigned.get(name, [])
        dtype = nest.sdfg.symbols.get(name)
        if not sites:
            if name not in nest.node.symbol_mapping or dtype is None:
                raise Refuse(name)
            expr = self.sym_expr(nest.parent, nest.node.symbol_mapping[name], (nest.frame, float("inf")), {})
            if isinstance(expr, symbolic.symbol):
                return str(expr), (nest.frame, -1)
            return self.let(nest.frame, name, dtype, expr), (nest.frame, -1)
        if len(sites) != 1 or name in nest.node.symbol_mapping:
            raise Refuse(name)
        region, edge = sites[0]
        if edge is None:
            raise Refuse(name)
        pos = self.position_of_block(nest, edge.dst, before=True)
        if pos is None or self.frames[pos[0]].edges.get(edge) is None:
            raise Refuse(name)
        dtype = dtype or edge.data.new_symbols(nest.sdfg, nest.sdfg.symbols).get(name)
        if dtype is None:
            raise Refuse(name)
        expr = self.python(nest, edge.data.assignments[name], pos, {})
        return self.let(pos[0], name, dtype, expr), pos

    def scalar(self, nest: Nest, name: str, at: tuple[int, float]) -> str:
        """The value a scalar container holds at ``at``: its one writer must run in an earlier chain state."""
        desc = nest.sdfg.arrays[name]
        if nest.node is not None and not desc.transient:
            if name in nest.node.out_connectors or name not in nest.node.in_connectors:
                raise Refuse(name)
            feeds = [e for e in nest.state.in_edges(nest.node) if e.dst_conn == name]
            if len(feeds) != 1:
                raise Refuse(name)
            return self.edge_value(nest.parent, nest.state, feeds[0], (nest.frame, float("inf")))
        if nest.node is None:
            raise Refuse(name)
        writers = nest.writers.get(name, [])
        if len(writers) != 1:
            raise Refuse(name)
        wstate, wnode = writers[0]
        pos = self.position_of_block(nest, wstate)
        if pos is None or not self.visible(pos, at) or pos[0] < at[0] and self.child_of(pos[0]) is wstate:
            raise Refuse(name)
        return self.node_value(nest, wstate, wnode, pos)

    def child_of(self, frame: int):
        f = self.frames[frame]
        return max(f.order, key=f.order.get) if f.order else None

    def node_value(self, nest: Nest, state: SDFGState, node: nodes.AccessNode, at: tuple[int, float]) -> str:
        """The value of the scalar ``node`` in ``state``, written there or before."""
        if state.in_degree(node) == 0:
            if any(w[1] is not node and w[0] is state for w in nest.writers.get(node.data, [])):
                raise Refuse(node.data)
            return self.scalar(nest, node.data, at)
        key = ("node", id(node))
        if key in self.memo:
            return self.memo[key]
        # At the top, a value the map body computes before the call runs once per iteration, as the call does.
        scope = self.map_entry if nest.node is None else None
        if state.in_degree(node) != 1 or state.entry_node(node) is not scope:
            raise Refuse(node.data)
        edge = state.in_edges(node)[0]
        dtype = nest.sdfg.arrays[node.data].dtype
        if isinstance(edge.src, nodes.AccessNode):
            value = self.edge_value(nest, state, edge, at)
            local = (
                value if nest.sdfg.arrays[edge.src.data].dtype == dtype else self.let(at[0], node.data, dtype, value)
            )
        elif isinstance(edge.src, nodes.Tasklet):
            local = self.let(at[0], node.data, dtype, self.tasklet(nest, state, edge.src, node.data, at))
        else:
            raise Refuse(node.data)
        self.memo[key] = local
        return local

    def edge_value(self, nest: Nest, state: SDFGState, edge, at: tuple[int, float]) -> str:
        """The one element ``edge`` reads, as a local or outer name."""
        mpath = state.memlet_path(edge)
        src = mpath[0].src
        if not isinstance(src, nodes.AccessNode) or edge.data.data != src.data or edge.data.wcr is not None:
            raise Refuse(str(edge.data))
        desc = nest.sdfg.arrays[src.data]
        if _is_scalar(desc):
            return self.node_value(nest, state, src, at)
        subset = edge.data.get_src_subset(edge, state)
        if subset is None or subset.num_elements() != 1:
            raise Refuse(str(edge.data))
        array = self.array(nest, src.data)
        index = [self.sym_expr(nest, b, at, {}) for b, _e, _s in subset]
        return self.let(at[0], src.data, desc.dtype, f"{array}[{', '.join(map(symbolic.symstr, index))}]")

    def sym_expr(self, nest: Nest, expr, at: tuple[int, float], bound: dict[str, str]):
        """``expr`` over the names of ``nest`` as an expression over locals and the map's scope."""
        expr = symbolic.pystr_to_symbolic(str(expr) if isinstance(expr, int) else expr)
        subs = {}
        for s in expr.free_symbols:
            local = self.symbol(nest, str(s), at, bound)
            if local != str(s):
                subs[s] = symbolic.symbol(local, self.locals[local])
        return expr.subs(subs) if subs else expr

    def array(self, nest: Nest, name: str) -> str:
        """The name in the map's SDFG of the read-only array ``name`` of ``nest``."""
        key = ("array", id(nest), name)
        if key in self.memo:
            return self.memo[key]
        desc = nest.sdfg.arrays.get(name)
        if not isinstance(desc, data.Array) or isinstance(desc, data.View):
            raise Refuse(name)
        if nest.node is None:
            if name in self.written:
                raise Refuse(name)
            self.memo[key] = name
            return name
        if desc.transient or name in nest.node.out_connectors or nest.writers.get(name):
            raise Refuse(name)
        feeds = [e for e in nest.state.in_edges(nest.node) if e.dst_conn == name]
        if len(feeds) != 1:
            raise Refuse(name)
        src = nest.state.memlet_path(feeds[0])[0].src
        if not isinstance(src, nodes.AccessNode) or feeds[0].data.data != src.data:
            raise Refuse(name)
        if not _same_layout(desc, nest.parent.sdfg.arrays[src.data]):
            raise Refuse(name)
        outer = self.array(nest.parent, src.data)
        self.memo[key] = outer
        return outer

    def python(self, nest: Nest, code: str, at: tuple[int, float], bound: dict[str, str]) -> str:
        """``code``, a pure expression of ``nest``, with its names replaced by locals and outer arrays."""
        tree = ast.parse(code, mode="eval").body
        slicer = self

        class Rename(ast.NodeTransformer):
            def visit_Call(self, node: ast.Call):
                if not _pure_call(node.func) or node.keywords:
                    raise Refuse(ast.unparse(node))
                node.args = [self.visit(a) for a in node.args]
                return node

            def visit_Subscript(self, node: ast.Subscript):
                if not isinstance(node.value, ast.Name):
                    raise Refuse(ast.unparse(node))
                if _single_element(nest.sdfg.arrays.get(node.value.id)):
                    return ast.Name(id=slicer.scalar(nest, node.value.id, at), ctx=ast.Load())
                node.value = ast.Name(id=slicer.array(nest, node.value.id), ctx=ast.Load())
                node.slice = self.visit(node.slice)
                return node

            def visit_Name(self, node: ast.Name):
                if node.id in nest.sdfg.arrays:
                    if not _is_scalar(nest.sdfg.arrays[node.id]):
                        raise Refuse(node.id)
                    return ast.Name(id=slicer.scalar(nest, node.id, at), ctx=ast.Load())
                return ast.Name(id=slicer.symbol(nest, node.id, at, bound), ctx=ast.Load())

            def generic_visit(self, node):
                if not isinstance(node, PURE_NODES):
                    raise Refuse(ast.unparse(node))
                return super().generic_visit(node)

        return ast.unparse(Rename().visit(tree))

    def tasklet(self, nest: Nest, state: SDFGState, tasklet: nodes.Tasklet, written: str, at: tuple[int, float]) -> str:
        """The value ``tasklet`` writes to the scalar ``written``: one pure Python assignment of its inputs, by
        connector or (once code generation inlined them) by container name."""
        if tasklet.code.language != dtypes.Language.Python or len(tasklet.out_connectors) != 1:
            raise Refuse(tasklet.label)
        body = tasklet.code.code
        if len(body) != 1 or not isinstance(body[0], ast.Assign) or len(body[0].targets) != 1:
            raise Refuse(tasklet.label)
        target = body[0].targets[0]
        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            target = target.value if target.value.id == written else target
        elif isinstance(target, ast.Name) and target.id in tasklet.out_connectors:
            target = ast.Name(id=written)
        if not isinstance(target, ast.Name) or target.id != written:
            raise Refuse(tasklet.label)
        feeds: dict[str, Any] = {}
        for e in state.in_edges(tasklet):
            if not e.data.is_empty():
                for key in (e.dst_conn, e.data.data):
                    if key in feeds:
                        raise Refuse(tasklet.label)
                    feeds[key] = e
        slicer = self

        def scalar_feed(name: str | None) -> bool:
            return name in feeds and name in nest.sdfg.arrays and _is_scalar(nest.sdfg.arrays[name])

        class Inputs(ast.NodeTransformer):
            def value(self, name: str) -> ast.Name:
                return ast.Name(id=slicer.edge_value(nest, state, feeds[name], at), ctx=ast.Load())

            def visit_Subscript(self, node: ast.Subscript):
                if isinstance(node.value, ast.Name) and scalar_feed(node.value.id):
                    return self.value(node.value.id)
                return self.generic_visit(node)

            def visit_Name(self, node: ast.Name):
                if node.id in tasklet.in_connectors:
                    if node.id not in feeds:
                        raise Refuse(node.id)
                    return self.value(node.id)
                if scalar_feed(node.id):
                    return self.value(node.id)
                if node.id in nest.sdfg.arrays and node.id not in feeds:
                    raise Refuse(node.id)
                return node

        tree = Inputs().visit(ast.parse(ast.unparse(body[0].value), mode="eval").body)
        bound = {n.id: n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id in self.locals}
        return self.python(nest, ast.unparse(tree), at, bound)

    # Path

    def push(self, kind: str, nest: Nest, **kwargs) -> int:
        self.frames.append(Frame(kind, nest, **kwargs))
        return len(self.frames) - 1

    def build(self) -> tuple:
        def tree(i: int, weight) -> tuple:
            if i == len(self.frames):
                return ()
            f = self.frames[i]
            below = self.frames[i:]
            # Without a guard or value below, a loop variable no row reads only multiplies the rows' writes.
            free = not any(g.kind == "guard" or any(isinstance(s, Let) for s in g.stmts) for g in below)
            kept = []
            if f.kind == "map":
                read = {f.vars[n] for n in f.used}
                kept = [k for k, v in enumerate(f.payload[0]) if v in read or not free]
                for k, (begin, end, step) in enumerate(f.payload[2]):
                    if k not in kept:
                        weight = weight * symbolic.sympy.Max(0, symbolic.int_floor(end - begin, step) + 1)
            stmts = tuple(Row(s.expr, weight) if isinstance(s, Row) else s for s in f.stmts)
            inner = stmts + tree(i + 1, weight)
            if f.kind in ("region", "top", "nest"):
                return inner
            if f.kind == "guard":
                return (Guard(f.payload, inner),)
            if f.kind == "loop":
                if not f.used and free:
                    return inner
                var, dtype, init, cond, update = f.payload
                return (Loop(var, dtype, init, cond, update, inner),)
            if not kept:
                return inner
            return (MapLoop(*(tuple(x[k] for k in kept) for x in f.payload), inner),)

        return tree(0, 1)


def _nest_tables(nest: Nest) -> None:
    for region in nest.sdfg.all_control_flow_regions():
        for block in region.nodes():
            if isinstance(block, (BreakBlock, ContinueBlock, ReturnBlock)):
                raise Refuse(block.label)
        for edge in region.edges():
            for sym in edge.data.assignments:
                nest.assigned.setdefault(sym, []).append((region, edge))
    for region in nest.sdfg.all_control_flow_regions():
        if isinstance(region, LoopRegion):
            nest.assigned.setdefault(region.loop_variable, []).append((region, None))
    for st in nest.sdfg.states():
        for n in st.data_nodes():
            if st.in_degree(n) > 0:
                nest.writers.setdefault(n.data, []).append((st, n))


def _chain(region: ControlFlowRegion, child) -> tuple[list, list]:
    """The blocks and edges ``region`` runs, unconditionally and once each, from its start up to ``child``."""
    blocks, edges = [], []
    block = region.start_block
    if region.in_degree(block) != 0:
        raise Refuse(region.label)
    while block is not child:
        out = region.out_edges(block)
        if len(out) != 1 or not out[0].data.is_unconditional():
            raise Refuse(region.label)
        blocks.append(block)
        edges.append(out[0])
        block = out[0].dst
        if region.in_degree(block) != 1 or block in blocks:
            raise Refuse(region.label)
    return blocks, edges


def _region_path(sdfg: SDFG, state: SDFGState) -> list:
    """``(region, child)`` from ``sdfg`` down to ``state``, ConditionalBlocks as ``(block, branch index)``."""
    path = []
    block = state
    while block is not sdfg:
        parent = block.parent_graph
        if isinstance(parent, ConditionalBlock):
            index = next(i for i, (_c, b) in enumerate(parent.branches) if b is block)
            path.append((parent, index))
        else:
            path.append((parent, block))
        block = parent
    return path[::-1]


class ScatterAnalysis:
    """Finds the writes of one accumulator of a parallel map and slices each one's index computation."""

    def __init__(self, state: SDFGState, map_entry: nodes.MapEntry, name: str, redtype: dtypes.ReductionType):
        self.state = state
        self.map_entry = map_entry
        self.name = name
        self.redtype = redtype
        self.shape = state.sdfg.arrays[name].shape
        exit_node = state.exit_node(map_entry)
        self.written = {e.data.data for e in state.out_edges(exit_node) if e.data.data is not None}
        self.written |= {
            n.data
            for n in state.scope_subgraph(map_entry).nodes()
            if isinstance(n, nodes.AccessNode) and state.in_degree(n) > 0
        }

    def wcr_path(self, state: SDFGState, edge, stop=None) -> tuple[list, Any]:
        """The sequential map entries ``edge``'s memlet path crosses up to ``stop`` (outermost first) and its
        source edge."""
        path = state.memlet_path(edge)
        entries = []
        for e in path:
            if e.data.wcr is None or operations.detect_reduction_type(e.data.wcr) != self.redtype:
                raise Refuse(str(e.data))
            if e.dst is stop or e is path[-1]:
                break
            if not isinstance(e.dst, nodes.MapExit):
                raise Refuse(str(e.dst))
            entry = state.entry_node(e.dst)
            if not _sequential_entry(state, entry):
                raise Refuse(entry.map.label)
            entries.append(entry)
        return entries[::-1], path[0]

    def sites(self) -> tuple:
        state = self.state
        exits = [e for e in state.in_edges(state.exit_node(self.map_entry)) if e.data.data == self.name]
        if len(exits) != 1:
            raise Refuse(self.name)
        # A region that moves with the map's own variables is written per iteration, not reduced over.
        if {str(s) for s in exits[0].data.subset.free_symbols} & set(self.map_entry.map.params):
            raise Refuse(self.name)
        entries, first = self.wcr_path(state, exits[0], state.exit_node(self.map_entry))
        if not isinstance(first.src, nodes.NestedSDFG):
            raise Refuse(self.name)
        head = [("map", state, e) for e in entries]
        steps = self.paths(head, state, first.src, first.src_conn, state.sdfg.arrays[self.name])
        return tuple(self.slice(s) for s in steps)

    def paths(self, head: list, state: SDFGState, node: nodes.NestedSDFG, acc: str, outer: data.Data):
        """Each write of ``acc`` inside ``node``, called in ``state`` after the steps ``head``, as path steps."""
        inner = node.sdfg
        if acc not in inner.arrays or not _same_layout(inner.arrays[acc], outer):
            raise Refuse(acc)
        for region in inner.all_control_flow_regions():
            for code in region.get_meta_codeblocks():
                if acc in code.get_free_symbols():
                    raise Refuse(acc)
            for edge in region.edges():
                if acc in edge.data.read_symbols():
                    raise Refuse(acc)
        head = head + [("nest", state, node)]
        found = False
        for istate in inner.states():
            for an in istate.data_nodes():
                if an.data != acc:
                    continue
                if any(not e.data.is_empty() for e in istate.out_edges(an)):
                    raise Refuse(acc)
                for e in istate.in_edges(an):
                    ientries, leaf = self.wcr_path(istate, e)
                    steps = head + [("region", r, c) for r, c in _region_path(inner, istate)]
                    steps += [("map", istate, x) for x in ientries]
                    found = True
                    if isinstance(leaf.src, nodes.NestedSDFG):
                        yield from self.paths(steps, istate, leaf.src, leaf.src_conn, inner.arrays[acc])
                    elif isinstance(leaf.src, (nodes.Tasklet, nodes.AccessNode)):
                        yield steps + [("leaf", istate, leaf)]
                    else:
                        raise Refuse(acc)
        if not found:
            raise Refuse(acc)

    def slice(self, steps: list) -> tuple:
        slicer = SiteSlicer(self.state, self.map_entry, self.written)
        nest = slicer.top
        slicer.push("top", nest)
        for kind, a, b in steps:
            last = len(slicer.frames) - 1
            if kind == "map":
                state, entry = a, b
                at = (last, float("inf"))
                types = entry.new_symbols(nest.sdfg, state, dict(nest.sdfg.symbols))
                ranges = tuple(tuple(slicer.sym_expr(nest, x, at, {}) for x in rng) for rng in entry.map.range)
                names = tuple(slicer.fresh(p, types[p]) for p in entry.map.params)
                slicer.push(
                    "map",
                    nest,
                    vars=dict(zip(entry.map.params, names)),
                    payload=(names, tuple(types[p] for p in entry.map.params), ranges),
                )
            elif kind == "nest":
                child = Nest(b.sdfg, b, a, nest, last)
                _nest_tables(child)
                nest = child
                slicer.push("nest", nest)
            elif kind == "region" and isinstance(a, ConditionalBlock):
                cond = self.branch_condition(slicer, nest, a, b, (last, float("inf")))
                slicer.push("guard", nest, payload=cond)
            elif kind == "region":
                region, child = a, b
                if isinstance(region, LoopRegion):
                    self.push_loop(slicer, nest, region)
                blocks, edges = _chain(region, child)
                order = {blk: i for i, blk in enumerate(blocks)}
                order[child] = len(blocks)
                slicer.push("region", nest, order=order, edges={e: i for i, e in enumerate(edges)})
            else:
                state, leaf = a, b
                subset = leaf.data.get_dst_subset(leaf, state)
                if subset is None or subset.dims() != len(self.shape):
                    raise Refuse(str(leaf.data))
                row, row_end, _ = subset[0]
                if symbolic.simplify(row - row_end) != 0:
                    raise Refuse(str(leaf.data))
                slicer.frames[last].stmts.append(Row(slicer.sym_expr(nest, row, (last, float("inf")), {})))
        return slicer.build()

    @staticmethod
    def branch_condition(slicer: SiteSlicer, nest: Nest, block: ConditionalBlock, index: int, at) -> str:
        terms = []
        for i, (cond, _branch) in enumerate(block.branches[: index + 1]):
            if cond is None:
                if i != index:
                    raise Refuse(block.label)
                continue
            text = slicer.python(nest, cond.as_string, at, {})
            terms.append(f"({text})" if i == index else f"(not ({text}))")
        return " and ".join(terms) or "True"

    @staticmethod
    def push_loop(slicer: SiteSlicer, nest: Nest, loop: LoopRegion) -> None:
        if loop.inverted or loop.init_statement is None or loop.update_statement is None:
            raise Refuse(loop.label)
        var = loop.loop_variable
        dtype = loop.new_symbols(dict(nest.sdfg.symbols)).get(var) or nest.sdfg.symbols.get(var)
        # The loop's frame binds ``var`` for its body; a sibling loop over the same name does not reach it, a body
        # edge assigning it would.
        if dtype is None or any(edge is not None for _region, edge in nest.assigned.get(var, [])):
            raise Refuse(loop.label)
        init, update = loop.init_statement.code, loop.update_statement.code
        if len(init) != 1 or len(update) != 1:
            raise Refuse(loop.label)
        for stmt in (init[0], update[0]):
            if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                raise Refuse(loop.label)
            if not isinstance(stmt.targets[0], ast.Name) or stmt.targets[0].id != var:
                raise Refuse(loop.label)
        at = (len(slicer.frames) - 1, float("inf"))
        local = slicer.fresh(var, dtype)
        bound = {var: local}
        start = slicer.python(nest, ast.unparse(init[0].value), at, {})
        cond = slicer.python(nest, loop.loop_condition.as_string, at, bound)
        step = slicer.python(nest, ast.unparse(update[0].value), at, bound)
        slicer.push("loop", nest, vars={var: local}, payload=(local, dtype, start, cond, step))


def scatter_index_sources(
    state: SDFGState, map_entry: nodes.MapEntry, name: str, redtype: dtypes.ReductionType
) -> ScatterIndexSources | None:
    """Index sources of the ``redtype`` WCR accumulator ``name`` of ``map_entry``, or ``None``.

    Requires: one WCR edge chain from a nested SDFG through sequential maps to the exit; inside the
    nests the accumulator is only written, with ``redtype``, at a first index whose value is a pure
    function of map and loop variables, symbols and arrays the map does not write; every value it
    depends on is computed once on the path to the write (no break, continue or return in a nest).
    """
    desc = state.sdfg.arrays[name]
    if not _contiguous_rows(desc):
        return None
    if redtype in ORDERED_OPS and desc.dtype in (dtypes.complex64, dtypes.complex128):
        return None
    try:
        return ScatterIndexSources(ScatterAnalysis(state, map_entry, name, redtype).sites())
    except Refuse:
        return None


def _in_parallel_scope(state: SDFGState, node: nodes.Node) -> bool:
    """Whether ``node`` runs inside a parallel CPU scope, in its SDFG or any SDFG that nests it."""
    parallel = (dtypes.ScheduleType.CPU_Multicore, dtypes.ScheduleType.CPU_Persistent)
    while state is not None:
        scope = state.entry_node(node)
        while scope is not None:
            if isinstance(scope, nodes.MapEntry) and scope.map.schedule in parallel:
                return True
            scope = state.entry_node(scope)
        sdfg = state.sdfg
        node, state = sdfg.parent_nsdfg_node, sdfg.parent
    return False


def range_private_accumulators(state: SDFGState, map_entry: nodes.MapEntry) -> dict[str, tuple]:
    """``{accumulator: (reduction type, ScatterIndexSources)}`` of the scatters ``map_entry`` can range-privatize.

    The map must be a statically scheduled ``CPU_Multicore`` map outside any other parallel scope, and an
    accumulator a transient-or-argument array the map does not also read.
    """
    m = map_entry.map
    if m.schedule != dtypes.ScheduleType.CPU_Multicore or m.omp_schedule not in (
        dtypes.OMPScheduleType.Default,
        dtypes.OMPScheduleType.Static,
    ):
        return {}
    if symbolic.issymbolic(m.omp_chunk_size) or int(m.omp_chunk_size) > 0:
        return {}
    if _in_parallel_scope(state, map_entry):
        return {}
    read = {
        e.src.data for e in state.in_edges(map_entry) if isinstance(e.src, nodes.AccessNode) and not e.data.is_empty()
    }
    out = {}
    for edge in state.out_edges(state.exit_node(map_entry)):
        name = edge.data.data
        if edge.data.wcr is None or name is None or name in out or name in read:
            continue
        desc = state.sdfg.arrays[name]
        if desc.may_alias or desc.lifetime in (
            dtypes.AllocationLifetime.Persistent,
            dtypes.AllocationLifetime.External,
        ):
            continue
        redtype = operations.detect_reduction_type(edge.data.wcr)
        if redtype not in RANGE_PRIVATIZABLE_OPS:
            continue
        sources = scatter_index_sources(state, map_entry, name, redtype)
        if sources is not None:
            out[name] = (redtype, sources)
    return out


@properties.make_properties
@transformation.explicit_cf_compatible
class RangePrivatizeScatterReductions(ppl.Pass):
    """Sets ``Map.omp_range_privatize`` on every parallel CPU map with a range-privatizable array scatter."""

    CATEGORY: str = "Optimization Preparation"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & (ppl.Modifies.States | ppl.Modifies.Nodes | ppl.Modifies.Memlets))

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> set[tuple[int, str]] | None:
        """Marks every qualifying map in ``sdfg`` and its nested SDFGs.

        :return: The marked maps as ``(cfg id, map label)``, or ``None`` if none qualified.
        """
        marked = set()
        for nsdfg in sdfg.all_sdfgs_recursive():
            for state in nsdfg.states():
                for node in state.nodes():
                    if isinstance(node, nodes.MapEntry) and not node.map.omp_range_privatize:
                        if range_private_accumulators(state, node):
                            node.map.omp_range_privatize = True
                            marked.add((nsdfg.cfg_id, node.map.label))
        return marked or None
