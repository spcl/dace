# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Write a prepared SDFG as one Python function over NumPy arrays.

Every name lives in the one namespace of the function (:class:`~dace.transformation.passes.unique_names.UniqueNames`),
so the walker never opens a scope. A ``Scalar`` is a one-element array, so a tasklet reads and writes it as
``s[0]`` like any element. A view is ``as_strided`` over the first element of its subset with the view's own shape
and strides, the pointer the C++ backend would pass; a nested SDFG connector aliases the outer container.
"""

import ast
import contextlib
from collections.abc import Iterator

import numpy as np

from dace import SDFG, data, dtypes, subsets, symbolic
from dace.codegen.exceptions import CodegenError
from dace.codegen.python import CUPY_STORAGES
from dace.config import Config
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import (
    BreakBlock,
    ConditionalBlock,
    ContinueBlock,
    ControlFlowBlock,
    ControlFlowRegion,
    LoopRegion,
    ReturnBlock,
    SDFGState,
)
from dace.sdfg.utils import dfs_topological_sort, get_view_edge
from dace.transformation.passes.analysis import loop_analysis

#: The module head: the names SymPy prints and DaCe tasklets call.
PRELUDE = """\
import math

import numpy as np
from numpy.lib.stride_tricks import as_strided

import dace


def int_floor(a, b):
    return a // b


def int_ceil(a, b):
    return -(-a // b)


def ipow(a, b):
    return a**b


def py_mod(a, b):
    return a % b


def c_mod(a, b):
    if isinstance(a, int) and isinstance(b, int):
        remainder = abs(a) % abs(b)
        return -remainder if a < 0 else remainder
    return np.fmod(a, b)


CMod = FtnMod = c_mod
Mod = PyMod = FtnModulo = py_mod


Min, Max, Abs = min, max, abs
inf, nan = math.inf, math.nan


def abort():
    raise RuntimeError("an assumption of the program does not hold")
"""


class Modulos(ast.NodeTransformer):
    """DaCe text spells C's truncating modulo ``%`` and the floored one ``Mod``; Python's ``%`` floors."""

    def visit_BinOp(self, node: ast.BinOp) -> ast.expr:
        self.generic_visit(node)
        if isinstance(node.op, ast.Mod):
            return ast.Call(ast.Name("c_mod", ast.Load()), [node.left, node.right], [])
        return node


def c_modulo(code: str) -> str:
    """``code`` with C's truncating ``%``, the one every SDFG text means."""
    if "%" not in code:
        return code
    return ast.unparse(Modulos().visit(ast.parse(code)))


def expr(value) -> str:
    """A symbolic expression as Python."""
    return c_modulo(symbolic.symstr(value, cpp_mode=False))


def starts(desc: data.Data, subset: subsets.Subset) -> list[str]:
    """The per-dim index of the first element of ``subset`` in ``desc``."""
    return [expr(begin + offset) for (begin, _, _), offset in zip(subset.ndrange(), desc.offset)]


def element(desc: data.Data, subset: subsets.Subset) -> str:
    """The index of the first element of ``subset`` in ``desc``."""
    return ", ".join(starts(desc, subset))


def slices(desc: data.Data, subset: subsets.Subset) -> str:
    """``subset`` of ``desc`` as NumPy slices that keep every dim."""
    parts = []
    for (begin, end, step), offset in zip(subset.ndrange(), desc.offset):
        stride = "" if step == 1 else f":{expr(step)}"
        parts.append(f"{expr(begin + offset)}:{expr(end + offset + 1)}{stride}")
    return ", ".join(parts)


def on_gpu(desc: data.Data) -> bool:
    return desc.storage in CUPY_STORAGES


def strided(base: str, base_desc: data.Data, subset: subsets.Subset, desc: data.Data) -> str:
    """``desc`` laid over ``base`` from the first element of ``subset``: a view with ``desc``'s shape and strides."""
    start = ", ".join(f"{index}:" for index in starts(base_desc, subset))
    shape = ", ".join(expr(s) for s in desc.shape)
    strides = ", ".join(f"{expr(s)} * {base}.itemsize" for s in desc.strides)
    function = "cupy.lib.stride_tricks.as_strided" if on_gpu(base_desc) else "as_strided"
    return f"{function}({base}[{start}], ({shape},), ({strides},))"


def packed_strides(shape) -> tuple:
    """The strides of a C-contiguous array of ``shape``."""
    strides = [1]
    for extent in reversed(shape[1:]):
        strides.insert(0, strides[0] * extent)
    return tuple(strides)


def allocation(desc: data.Data) -> str:
    """A new array for ``desc`` on its side, with its strides when they are not the packed ones."""
    dtype = f"np.{desc.dtype.as_numpy_dtype().name}"
    xp = "cupy" if on_gpu(desc) else "np"
    if isinstance(desc, data.Scalar):
        return f"{xp}.empty(1, {dtype})"
    shape = ", ".join(expr(s) for s in desc.shape)
    if tuple(desc.strides) == packed_strides(desc.shape):
        return f"{xp}.empty(({shape},), {dtype})"
    strides = ", ".join(f"{expr(s)} * {desc.dtype.bytes}" for s in desc.strides)
    function = "cupy.lib.stride_tricks.as_strided" if on_gpu(desc) else "as_strided"
    return f"{function}({xp}.empty({expr(desc.total_size)}, {dtype}), ({shape},), ({strides},))"


def ancestors(block: ControlFlowBlock) -> list[ControlFlowBlock]:
    """``block`` and the blocks around it, innermost first, up to its SDFG."""
    chain = [block]
    while not isinstance(chain[-1], SDFG):
        chain.append(chain[-1].parent_graph)
    return chain


def assigned_in(region: ControlFlowRegion) -> set[str]:
    """The symbols ``region`` assigns: on its interstate edges and as loop iterators."""
    names = {str(name) for edge in region.all_interstate_edges() for name in edge.data.assignments}
    loops = (r for r in [region, *region.all_control_flow_regions()] if isinstance(r, LoopRegion))
    return names | {loop.loop_variable for loop in loops if loop.loop_variable}


def allocation_point(states: list[SDFGState], shape_symbols: set[str]) -> tuple[ControlFlowRegion, ControlFlowBlock]:
    """The region and the block of it before which a transient used in ``states`` is allocated."""
    chains = [list(reversed(ancestors(state))) for state in states]
    depth = 0
    while all(len(chain) > depth + 1 for chain in chains) and len({id(chain[depth + 1]) for chain in chains}) == 1:
        depth += 1
    # The innermost common block; a region the emitter walks block by block is a ControlFlowRegion
    region = chains[0][depth]
    while not isinstance(region, ControlFlowRegion) or isinstance(region, ConditionalBlock):
        depth -= 1
        region = chains[0][depth]
    # Out of each loop that does not reshape the transient, so its data outlives the iteration
    while isinstance(region, LoopRegion) and not (shape_symbols & assigned_in(region)) and depth > 0:
        depth -= 1
        region = chains[0][depth]
        while not isinstance(region, ControlFlowRegion) or isinstance(region, ConditionalBlock):
            depth -= 1
            region = chains[0][depth]
    first = min((chain for chain in chains), key=lambda chain: block_order(region, chain[depth + 1]))
    return region, first[depth + 1]


def block_order(region: ControlFlowRegion, block: ControlFlowBlock) -> int:
    """The position of ``block`` in the chain of blocks the structured ``region`` runs."""
    position, current = 0, region.start_block
    while current is not block:
        position, current = position + 1, region.out_edges(current)[0].dst
    return position


def kernel_containers(sdfg: SDFG) -> set[str]:
    """The containers of ``sdfg`` that only GPU scopes access: the tile kernels hold them, the host never does."""
    inside, outside = set(), set()
    for state in sdfg.states():
        scope = state.scope_dict()
        for node in state.data_nodes():
            parent = scope[node]
            while parent is not None and parent.map.schedule not in dtypes.GPU_SCHEDULES:
                parent = scope[parent]
            (inside if parent is not None else outside).add(node.data)
    return inside - outside


def is_view(sdfg: SDFG, node: nodes.Node) -> bool:
    return isinstance(node, nodes.AccessNode) and isinstance(node.desc(sdfg), data.View)


def used_names(code: str) -> set[str]:
    return {n.id for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Name)}


class ScalarReads(ast.NodeTransformer):
    """Reads a ``Scalar`` container by its element: an interstate expression names it bare."""

    def __init__(self, scalars: set[str]):
        self.scalars = scalars

    def visit_Subscript(self, node: ast.Subscript) -> ast.Subscript:
        node.slice = self.visit(node.slice)
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.scalars and isinstance(node.ctx, ast.Load):
            return ast.Subscript(value=node, slice=ast.Constant(0), ctx=ast.Load())
        return node


class Emitter:
    """Writes the lines of the function, one block of the SDFG tree at a time."""

    __slots__ = ("sdfg", "lines", "depth", "allocated", "loops", "view_edges", "kernels", "pending")

    def __init__(self, sdfg: SDFG):
        self.sdfg = sdfg
        self.lines: list[str] = []
        self.depth = 0
        self.allocated: set[str] = set()
        #: Per enclosing ``while``-form loop, the lines a ``continue`` runs first.
        self.loops: list[list[str]] = []
        #: The edges of the current state that bind a view to the container it views, which move no data.
        self.view_edges: set[MultiConnectorEdge[Memlet]] = set()
        #: The tile kernels the function launches, each the source of one module-level function.
        self.kernels: list[str] = []
        #: The transients to allocate just before a block of a region (:meth:`plan_allocations`).
        self.pending: dict[tuple[ControlFlowRegion, ControlFlowBlock], list[str]] = {}

    def line(self, text: str) -> None:
        self.lines.append("    " * self.depth + text)

    @contextlib.contextmanager
    def block(self, header: str) -> Iterator[None]:
        self.line(header)
        start = len(self.lines)
        self.depth += 1
        yield
        if len(self.lines) == start:
            self.line("pass")
        self.depth -= 1

    def code(self, sdfg: SDFG, text: str) -> str:
        """Interstate code, with its ``Scalar`` containers read by element."""
        scalars = {name for name, desc in sdfg.arrays.items() if isinstance(desc, data.Scalar)}
        if scalars & used_names(text):
            text = ast.unparse(ScalarReads(scalars).visit(ast.parse(text)))
        return c_modulo(text)

    def function(self) -> str:
        sdfg = self.sdfg
        arglist = sdfg.arglist()
        with self.block(f"def {sdfg.name}({', '.join(arglist)}):"):
            for name, desc in arglist.items():
                if isinstance(desc, data.Scalar) and name in sdfg.arrays:
                    self.line(f"{name} = np.asarray({name}, np.{desc.dtype.as_numpy_dtype().name}).reshape(1)")
            self.constants(sdfg)
            self.allocated |= kernel_containers(sdfg)
            self.allocate_ready(sdfg, set(arglist) | set(sdfg.constants))
            self.plan_allocations(sdfg)
            self.region(sdfg, sdfg)
        return "\n".join(self.lines)

    def constants(self, sdfg: SDFG) -> None:
        """Bind the constants of ``sdfg``, which neither the caller passes nor a state writes."""
        for name, value in sdfg.constants.items():
            if isinstance(value, np.ndarray):
                self.line(f"{name} = np.asarray({value.tolist()!r}, np.{value.dtype.name})")
            else:
                # A NumPy scalar becomes a plain number, which a tile kernel can also take
                self.line(f"{name} = {value.item() if isinstance(value, np.generic) else value!r}")

    def plan_allocations(self, sdfg: SDFG) -> None:
        """Place every transient the function entry cannot allocate: ahead of the first block that uses it, in the
        innermost region holding all its uses, and outside a loop whose iterations keep its shape."""
        uses: dict[str, list[ControlFlowBlock]] = {}
        for state in sdfg.states():
            for node in state.data_nodes():
                desc = sdfg.arrays[node.data]
                if desc.transient and not isinstance(desc, data.View) and node.data not in self.allocated:
                    uses.setdefault(node.data, []).append(state)
        for name, states in uses.items():
            region, block = allocation_point(states, {str(s) for s in sdfg.arrays[name].free_symbols})
            self.pending.setdefault((region, block), []).append(name)

    def allocate_ready(self, sdfg: SDFG, defined: set[str]) -> None:
        """Allocate the transients whose shape the ``defined`` names fix already."""
        for name, desc in sdfg.arrays.items():
            if desc.transient and name not in self.allocated and not isinstance(desc, data.View):
                if {str(s) for s in desc.free_symbols} <= defined:
                    self.allocate(name, desc)

    def allocate(self, name: str, desc: data.Data) -> None:
        self.allocated.add(name)
        self.line(f"{name} = {allocation(desc)}")

    def region(self, sdfg: SDFG, region: ControlFlowRegion) -> None:
        block = region.start_block
        while block is not None:
            for name in self.pending.pop((region, block), []):
                if name not in self.allocated:
                    self.allocate(name, sdfg.arrays[name])
            self.control_block(sdfg, block)
            out_edges = region.out_edges(block)
            if not out_edges:
                break
            for name, value in out_edges[0].data.assignments.items():
                self.line(f"{name} = {self.code(sdfg, value)}")
            block = out_edges[0].dst

    def control_block(self, sdfg: SDFG, block: ControlFlowBlock) -> None:
        if isinstance(block, SDFGState):
            self.state(sdfg, block)
        elif isinstance(block, LoopRegion):
            self.loop(sdfg, block)
        elif isinstance(block, ConditionalBlock):
            keyword = "if"
            for condition, branch in block.branches:
                header = "else:" if condition is None else f"{keyword} {self.code(sdfg, condition.as_string)}:"
                with self.block(header):
                    self.region(sdfg, branch)
                keyword = "elif"
        elif isinstance(block, BreakBlock):
            self.line("break")
        elif isinstance(block, ContinueBlock):
            for text in self.loops[-1] if self.loops else []:
                self.line(text)
            self.line("continue")
        elif isinstance(block, ReturnBlock):
            self.line("return")
        elif isinstance(block, ControlFlowRegion):
            self.region(sdfg, block)
        else:
            raise CodegenError(f"Python backend: no Python form for {type(block).__name__} {block.label!r}")

    def loop(self, sdfg: SDFG, loop: LoopRegion) -> None:
        counted = self.counted_range(loop)
        if counted is not None:
            self.loops.append([])
            with self.block(f"for {loop.loop_variable} in range({counted}):"):
                self.region(sdfg, loop)
            self.loops.pop()
            return
        if loop.init_statement is not None:
            self.line(self.code(sdfg, loop.init_statement.as_string))
        update = [self.code(sdfg, loop.update_statement.as_string)] if loop.update_statement is not None else []
        check = f"if not ({self.code(sdfg, loop.loop_condition.as_string)}):"
        self.loops.append(update + ([check, "    break"] if loop.inverted else []))
        with self.block("while True:"):
            if not loop.inverted:
                with self.block(check):
                    self.line("break")
            self.region(sdfg, loop)
            for text in update:
                self.line(text)
            if loop.inverted:
                with self.block(check):
                    self.line("break")
        self.loops.pop()

    def counted_range(self, loop: LoopRegion) -> str | None:
        """``start, stop, step`` of a loop a Python ``range`` runs, or ``None``."""
        if loop.inverted or not loop.loop_variable:
            return None
        start = loop_analysis.get_init_assignment(loop)
        end = loop_analysis.get_loop_end(loop)
        step = loop_analysis.get_loop_stride(loop)
        if start is None or end is None or step is None or not step.is_Number or step == 0:
            return None
        if any(loop_analysis.rebinds_counter(block, loop.loop_variable) for block in loop.nodes()):
            return None
        stop = end + 1 if step > 0 else end - 1
        return ", ".join([expr(start), expr(stop)] + ([] if step == 1 else [expr(step)]))

    def state(self, sdfg: SDFG, state: SDFGState) -> None:
        accessed = {n.data for n in state.data_nodes()}
        for name in sorted(accessed - self.allocated):
            desc = sdfg.arrays[name]
            if desc.transient and not isinstance(desc, data.View):
                self.allocate(name, desc)
        scope = state.scope_dict()
        order = [n for n in dfs_topological_sort(state) if not isinstance(n, nodes.ExitNode)]
        views = [n for n in order if is_view(sdfg, n)]
        self.view_edges = {get_view_edge(state, view) for view in views}
        self.scope(sdfg, state, scope, order, None)

    def scope(self, sdfg: SDFG, state: SDFGState, scope: dict, order: list, entry: nodes.EntryNode | None) -> None:
        self.bind_views(sdfg, state, [n for n in order if scope[n] is entry and is_view(sdfg, n)])
        for node in order:
            if scope[node] is entry:
                self.dispatch(sdfg, state, scope, order, node)

    def bind_views(self, sdfg: SDFG, state: SDFGState, views: list[nodes.AccessNode]) -> None:
        """Bind ``views``, a view of a view after the view it reads through."""
        while views:
            ready = [view for view in views if self.viewed(state, view) not in {v.data for v in views}]
            if not ready:
                raise CodegenError(f"Python backend: the views {[v.data for v in views]} view each other")
            for view in ready:
                self.view(sdfg, state, view)
            views = [view for view in views if view not in ready]

    def dispatch(self, sdfg: SDFG, state: SDFGState, scope: dict, order: list, node: nodes.Node) -> None:
        if isinstance(node, nodes.MapEntry):
            self.map(sdfg, state, scope, order, node)
        elif isinstance(node, nodes.Tasklet):
            self.tasklet(sdfg, state, node)
        elif isinstance(node, nodes.AccessNode):
            self.access(sdfg, state, node)
        elif isinstance(node, nodes.NestedSDFG):
            self.nested(sdfg, state, node)
        elif isinstance(node, nodes.LibraryNode):
            self.library(sdfg, state, node)
        else:
            raise CodegenError(f"Python backend: no Python form for {type(node).__name__} {node.label!r}")

    def library(self, sdfg: SDFG, state: SDFGState, node: nodes.LibraryNode) -> None:
        raise CodegenError(f"Python backend: library node {node.label!r} outside a tile kernel was not expanded")

    def map(self, sdfg: SDFG, state: SDFGState, scope: dict, order: list, entry: nodes.MapEntry) -> None:
        if entry.map.schedule in dtypes.GPU_SCHEDULES:
            self.tile_kernel(sdfg, state, scope, order, entry)
            return
        for edge in state.in_edges(entry):
            if edge.dst_conn and not edge.dst_conn.startswith("IN_"):
                self.line(f"{edge.dst_conn} = {self.read(sdfg, state, edge)}")
        with contextlib.ExitStack() as stack:
            for param, (begin, end, step) in zip(entry.map.params, entry.map.range):
                stop = expr(end + 1)
                stride = "" if step == 1 else f", {expr(step)}"
                stack.enter_context(self.block(f"for {param} in range({expr(begin)}, {stop}{stride}):"))
            self.scope(sdfg, state, scope, order, entry)

    def tile_kernel(self, sdfg: SDFG, state: SDFGState, scope: dict, order: list, entry: nodes.MapEntry) -> None:
        """Write the map at ``entry`` as a kernel of the tile dialect and launch it, one program per iteration."""
        # Avoid import loop
        from dace.codegen.python.kernel import holds_tile_ops

        if not holds_tile_ops(sdfg, state, entry):
            raise CodegenError(
                f"Python backend: GPU map {entry.map.label!r} holds no tile ops; vectorize with "
                "VectorizeMultiDim(device=GPU, tile_group=BLOCK) first"
            )
        dialect = tile_dialect()
        # Labels repeat; the SDFG, state and node ids do not, as the CUDA generator names its kernels
        name = f"kernel_{entry.map.label}_{sdfg.cfg_id}_{state.block_id}_{state.node_id(entry)}"
        kernel = dialect.emitter_type(sdfg, dialect)
        arrays, symbols = kernel.kernel(sdfg, state, scope, order, entry)
        grid = [f"int_ceil({expr(end - begin + 1)}, {expr(step)})" for begin, end, step in reversed(entry.map.range)]
        self.kernels.append(dialect.define(name, kernel, arrays, symbols, grid))
        # An empty map launches nothing; cuTile refuses a grid of zero programs
        with self.block(f"if {' and '.join(f'{size} > 0' for size in grid)}:"):
            self.line(dialect.launch(name, grid, arrays, symbols))

    def read(self, sdfg: SDFG, state: SDFGState, edge: MultiConnectorEdge[Memlet]) -> str:
        """The value or view an edge brings into a connector."""
        name = edge.data.data
        desc = sdfg.arrays[name]
        subset = edge.data.subset
        if subset.num_elements() == 1 and not isinstance(desc, data.Scalar):
            return f"{name}[{element(desc, subset)}]"
        if isinstance(desc, data.Scalar):
            return f"{name}[0]"
        return f"{name}[{slices(desc, subset)}]"

    def body(self, sdfg: SDFG, node: nodes.Tasklet) -> str:
        return c_modulo(node.code.as_string)

    def tasklet(self, sdfg: SDFG, state: SDFGState, node: nodes.Tasklet) -> None:
        body = self.body(sdfg, node)
        used = used_names(body)
        for edge in state.in_edges(node):
            if edge.dst_conn in used and edge.data.data is not None:
                self.line(f"{edge.dst_conn} = {self.read(sdfg, state, edge)}")
        writes = [e for e in state.out_edges(node) if e.src_conn in used and e.data.data is not None]
        for edge in writes:
            if edge.src_conn in node.in_connectors:
                continue
            if edge.data.dynamic and edge.data.wcr is not None and edge.data.subset.num_elements() == 1:
                # A path that does not write must not accumulate: an unwritten connector stays None
                self.line(f"{edge.src_conn} = None")
            elif edge.data.subset.num_elements() != 1 or edge.data.dynamic:
                self.line(f"{edge.src_conn} = {self.read(sdfg, state, edge)}")
        self.statements(body)
        for edge in writes:
            self.write(sdfg, edge)

    def statements(self, body: str) -> None:
        """Write the Python statements of a tasklet body."""
        for text in body.splitlines():
            self.line(text)

    def write(self, sdfg: SDFG, edge: MultiConnectorEdge[Memlet]) -> None:
        name = edge.data.data
        desc = sdfg.arrays[name]
        subset = edge.data.subset
        if subset.num_elements() != 1:
            return  # the connector is a view of the array, written in place
        target = f"{name}[0]" if isinstance(desc, data.Scalar) else f"{name}[{element(desc, subset)}]"
        value = edge.src_conn
        if edge.data.wcr is None:
            self.line(f"{target} = {value}")
            return
        update = f"{target} = ({edge.data.wcr})({target}, {value})"
        if edge.data.dynamic:
            with self.block(f"if {value} is not None:"):
                self.line(update)
        else:
            self.line(update)

    def access(self, sdfg: SDFG, state: SDFGState, node: nodes.AccessNode) -> None:
        for edge in state.in_edges(node):
            if self.moves_data(edge) and isinstance(edge.src, (nodes.AccessNode, nodes.EntryNode)):
                self.copy(sdfg, state, edge)
        for edge in state.out_edges(node):
            if self.moves_data(edge) and isinstance(edge.dst, nodes.ExitNode):
                self.copy(sdfg, state, edge)

    def moves_data(self, edge: MultiConnectorEdge[Memlet]) -> bool:
        return edge not in self.view_edges and not edge.data.is_empty()

    def viewed(self, state: SDFGState, view: nodes.AccessNode) -> str:
        """The container ``view`` views."""
        edge = get_view_edge(state, view)
        if edge is None:
            raise CodegenError(f"Python backend: view {view.data!r} has no viewed container")
        path = state.memlet_path(edge)
        return path[0].src.data if edge.dst is view else path[-1].dst.data

    def view(self, sdfg: SDFG, state: SDFGState, node: nodes.AccessNode) -> None:
        """Bind a view before any node of its scope reads or writes it."""
        edge = get_view_edge(state, node)
        viewed = self.viewed(state, node)
        subset = edge.data.get_src_subset(edge, state) if edge.dst is node else edge.data.get_dst_subset(edge, state)
        self.line(f"{node.data} = {strided(viewed, sdfg.arrays[viewed], subset, node.desc(sdfg))}")
        self.allocated.add(node.data)

    def copy(self, sdfg: SDFG, state: SDFGState, edge: MultiConnectorEdge[Memlet]) -> None:
        path = state.memlet_path(edge)
        src, dst = path[0].src.data, path[-1].dst.data
        src_desc, dst_desc = sdfg.arrays[src], sdfg.arrays[dst]
        src_subset = edge.data.get_src_subset(edge, state) or subsets.Range.from_array(src_desc)
        dst_subset = edge.data.get_dst_subset(edge, state) or subsets.Range.from_array(dst_desc)
        value = f"{src}[{slices(src_desc, src_subset)}]"
        if on_gpu(src_desc) != on_gpu(dst_desc):
            value = f"cupy.asarray({value})" if on_gpu(dst_desc) else f"cupy.asnumpy({value})"
        if src_subset.size() != dst_subset.size():
            value = f"np.reshape({value}, ({', '.join(expr(s) for s in dst_subset.size())},))"
        self.line(f"{dst}[{slices(dst_desc, dst_subset)}] = {value}")

    def binds(self, sdfg: SDFG, node: nodes.NestedSDFG, edge: MultiConnectorEdge[Memlet], conn: str) -> bool:
        """Whether the connector ``conn`` of ``node`` takes the outer container of ``edge`` before the body runs."""
        return True

    def nested(self, sdfg: SDFG, state: SDFGState, node: nodes.NestedSDFG) -> None:
        inner = node.sdfg
        for name, value in node.symbol_mapping.items():
            if str(value) != name:
                self.line(f"{name} = {expr(value)}")
        connectors = set()
        for edge in (*state.in_edges(node), *state.out_edges(node)):
            conn = edge.dst_conn if edge.dst is node else edge.src_conn
            # An in-out connector has an edge on each side and binds once
            if conn is None or edge.data.is_empty() or conn in connectors:
                continue
            connectors.add(conn)
            # The connector's descriptor is the outer container's (``NestedSDFG.validate``): an alias. Bound for
            # every nested SDFG anew, which may give one connector name another container than the last did
            if conn != edge.data.data and self.binds(sdfg, node, edge, conn):
                self.line(f"{conn} = {edge.data.data}")
        self.constants(inner)
        ready = set(self.allocated) | connectors | set(node.symbol_mapping) | set(inner.constants)
        self.allocate_ready(inner, ready)
        self.plan_allocations(inner)
        self.region(inner, inner)


def tile_dialect():
    """The tile DSL ``compiler.python.tile_dialect`` selects."""
    # Avoid import loop
    from dace.codegen.python.dialects import DIALECTS

    return DIALECTS[Config.get("compiler", "python", "tile_dialect")]()


def generate(sdfg: SDFG) -> str:
    """The Python module of a prepared ``sdfg``: the prelude, its tile kernels and one function named after it."""
    emitter = Emitter(sdfg)
    function = emitter.function()
    prelude = PRELUDE
    if "cupy." in function or emitter.kernels:
        prelude = prelude.replace("import numpy as np\n", "import cupy\nimport numpy as np\n")
    if emitter.kernels:
        prelude += f"\n\n{tile_dialect().imports}"
    return "\n\n\n".join([prelude, *emitter.kernels, function]) + "\n"
