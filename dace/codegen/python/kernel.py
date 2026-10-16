# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tile kernels: a GPU map over BLOCK tile nodes, written in a tile DSL (Triton, cuTile, TileLang).

One program of the kernel runs one iteration of the map, whose parameters it recovers from its program id. The body is
walked as the host function is (:class:`~dace.codegen.python.emit.Emitter`), with three differences: a register
container is a value, not an array, so a copy rebinds it; a global array is the pointer or tensor the DSL takes and
only a tile node reads or writes it; a tile node is one statement of the :class:`Dialect`.
"""

import ast
import re
import textwrap
from typing import NamedTuple

from dace import SDFG, data, dtypes, symbolic
from dace.codegen.exceptions import CodegenError
from dace.codegen.python.emit import Emitter, expr
from dace.frontend.operations import detect_reduction_type
from dace.libraries.standard.nodes.copy.node import CopyLibraryNode
from dace.libraries.tileops import ops
from dace.libraries.tileops.dispatch import TileGroup
from dace.libraries.tileops.kinds import SYMBOL, TILE
from dace.libraries.tileops.nodes import (
    MaskedCopyLibraryNode,
    TileBinop,
    TileFMA,
    TileIota,
    TileITE,
    TileMaskGen,
    TileMMA,
    TileReduce,
    TileScan,
    TileUnop,
)
from dace.libraries.tileops.nodes.tile_op import TileOp
from dace.sdfg import nodes
from dace.sdfg.state import SDFGState

#: The prelude functions SymPy prints, which a kernel calls under the dialect's own name.
HELPERS = ("int_floor", "int_ceil", "ipow", "Min", "Max", "Abs")
HELPER_CALL = re.compile(rf"\b({'|'.join(HELPERS)})\(")
#: The lane index ``__l<k>`` of a per-lane expression of a tile node.
LANE = re.compile(r"__l(\d+)")
#: A C++ function-style cast, ``double(x)``: the statement casts its result to the output type anyway.
CPP_CAST = re.compile(r"\b(?:double|float|bool|char|short|int|long|unsigned|u?int(?:8|16|32|64)_t|dace::\w+)\s*\(")
#: What a masked lane of a reduction contributes.
IDENTITIES = {"+": "0", "*": "1", "min": "float('inf')", "max": "float('-inf')"}


class Element(NamedTuple):
    """The elements of a global array an access addresses: the index along each of its dims, a lane tensor along
    the dims a tile spans, and the strides of the array. A DSL that takes pointers flattens them (:meth:`flat`)."""

    indices: list[str]
    strides: list[str]
    #: Whether a lane tensor spans a dim, so the access moves a tile rather than one element.
    tile: bool = False

    def flat(self) -> str:
        return " + ".join(f"({index}) * {stride}" for index, stride in zip(self.indices, self.strides))


class Dialect:
    """What one tile DSL spells its own way. ``dialect[name]`` is the DSL's function ``name``."""

    #: The module the generated code imports, and the helpers its kernels call.
    imports = ""
    #: The decorator of a kernel function.
    decorator = ""
    #: The module whose functions ``dialect[name]`` names by default.
    module = ""
    #: The functions spelled other than ``<module>.<name>``.
    functions: dict[str, str] = {}
    #: The indentation of the kernel body: one under the ``def``, more under a launch context.
    body_depth = 1

    @property
    def emitter_type(self) -> type["KernelEmitter"]:
        """The walker of a kernel body; a dialect that holds tiles in buffers walks its own way."""
        return KernelEmitter

    @property
    def statement_type(self) -> type["TileStatement"]:
        """The statement a tile node is; a dialect that holds tiles in buffers writes its own."""
        return TileStatement

    def define(self, name: str, kernel: "KernelEmitter", arrays: list[str], symbols: list[str], grid: list[str]) -> str:
        """The source of the kernel function ``name`` around the body ``kernel`` wrote."""
        return "\n".join([self.decorator, f"def {name}({', '.join(arrays + symbols)}):", *kernel.lines])

    def __getitem__(self, name: str) -> str:
        return self.functions.get(name, f"{self.module}.{name}")

    def program_id(self, axis: int) -> str:
        raise NotImplementedError

    def lane(self, widths: list[int], dim: int) -> str:
        """The index of each lane along ``dim``, broadcast over a tile of ``widths``."""
        raise NotImplementedError

    def load(self, base: str, at: "Element", mask: str | None, dtype: dtypes.typeclass) -> str:
        """The elements ``at`` addresses of ``base``, whose elements are ``dtype``; masked lanes read 0."""
        raise NotImplementedError

    def store(self, base: str, at: "Element", value: str, mask: str | None) -> str:
        raise NotImplementedError

    def cast(self, value: str, dtype: dtypes.typeclass) -> str:
        raise NotImplementedError

    def select(self, condition: str, value: str, otherwise: str) -> str:
        """``value`` where ``condition`` holds, else ``otherwise``."""
        return f"{self['where']}({condition}, {value}, {otherwise})"

    def full(self, widths: list[int], value: str, dtype: dtypes.typeclass) -> str:
        """A tile of ``widths`` every lane of which is the scalar ``value``."""
        raise NotImplementedError

    def dot(self, a: str, b: str) -> str:
        raise NotImplementedError

    def identity(self, op: str) -> str:
        """What a masked lane contributes to a reduction with ``op``."""
        return IDENTITIES[op]

    def reduce(self, op: str, value: str, axis: int | None, rank: int) -> str:
        """``value``, a tile of ``rank`` dims, reduced with ``op`` along ``axis``, or fully."""
        return f"{self[ops.PY_REDUCE_OPS[op]]}({value}{'' if axis is None else f', {axis}'})"

    def scan(self, op: str, value: str) -> str:
        """The inclusive scan of the one-dim tile ``value`` with ``op``."""
        if op not in ops.PY_SCAN_OPS:
            raise CodegenError(f"Python backend: no tile scan combines with {op!r}")
        return f"{self[ops.PY_SCAN_OPS[op]]}({value}, 0)"

    def atomic_add(self, base: str, at: "Element", value: str, mask: str | None) -> str:
        """Add ``value`` to the elements ``at`` addresses, atomically against the other programs."""
        raise NotImplementedError

    def launch(self, kernel: str, grid: list[str], arrays: list[str], symbols: list[str]) -> str:
        raise NotImplementedError


def holds_tile_ops(sdfg: SDFG, state: SDFGState, entry: nodes.MapEntry) -> bool:
    """Whether the scope of ``entry``, nested SDFGs included, holds a BLOCK tile node."""
    for node in state.scope_subgraph(entry).nodes():
        if isinstance(node, TileOp) and node.group is not TileGroup.THREAD:
            return True
        if isinstance(node, nodes.NestedSDFG):
            if any(isinstance(n, TileOp) for n, _ in node.sdfg.all_nodes_recursive()):
                return True
    return False


#: The storages of a container a tile kernel holds itself, as a value or a buffer of its own.
KERNEL_STORAGES = (dtypes.StorageType.Register, dtypes.StorageType.Default, dtypes.StorageType.GPU_Shared)


def is_register(desc: data.Data) -> bool:
    return desc.transient and desc.storage in KERNEL_STORAGES


class KernelEmitter(Emitter):
    """Writes the kernel of one tile map in a :class:`Dialect`."""

    __slots__ = ("dialect", "arrays")

    def __init__(self, sdfg: SDFG, dialect: Dialect):
        super().__init__(sdfg)
        self.dialect = dialect
        #: The global arrays the kernel takes.
        self.arrays: set[str] = set()

    def line(self, text: str) -> None:
        for part in text.split("\n"):
            super().line(HELPER_CALL.sub(lambda m: f"{self.dialect[m.group(1)]}(", part))

    def code(self, sdfg: SDFG, text: str) -> str:
        return text  # a scalar is a value in a kernel

    def allocate(self, name: str, desc: data.Data) -> None:
        if not is_register(desc):
            raise CodegenError(f"Python backend: a tile kernel cannot allocate {name!r} in {desc.storage}")
        self.allocated.add(name)

    def kernel(
        self, sdfg: SDFG, state: SDFGState, scope: dict, order: list, entry: nodes.MapEntry
    ) -> tuple[list[str], list[str]]:
        """Write the body of the kernel of the map at ``entry``; return its global arrays and its symbols."""
        for name in (e.data.data for e in (*state.in_edges(entry), *state.out_edges(state.exit_node(entry)))):
            if name is not None:
                self.arrays.add(name)
        self.depth = self.dialect.body_depth
        # The registers of the map scope itself; a nested SDFG's states allocate their own
        for name in sorted({n.data for n in state.scope_subgraph(entry).data_nodes()} - self.allocated):
            if sdfg.arrays[name].transient:
                self.allocate(name, sdfg.arrays[name])
        params = list(entry.map.params)
        for axis, (param, (begin, _, step)) in enumerate(zip(reversed(params), reversed(entry.map.range))):
            self.line(f"{param} = {expr(begin)} + {self.dialect.program_id(axis)} * {expr(step)}")
        self.scope(sdfg, state, scope, order, entry)
        names, assigned = self.bindings()
        # The grid is the map range, which a dialect may compute in the kernel
        names |= {str(s) for s in entry.map.range.free_symbols}
        symbols = (names & (set(sdfg.symbols) | set(sdfg.arglist()))) - self.arrays - assigned
        return sorted(self.arrays), sorted(symbols)

    def bindings(self) -> tuple[set[str], set[str]]:
        """The names the kernel body reads or writes, and those it assigns."""
        tree = ast.parse(textwrap.dedent("\n".join(self.lines)))
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        return names, {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}

    def scope(self, sdfg, state, scope, order, entry) -> None:
        for node in order:
            if scope[node] is entry and isinstance(node, TileOp):
                if node.group is TileGroup.THREAD:
                    raise CodegenError(f"Python backend: tile kernels take BLOCK tiles; {node.label!r} is THREAD")
        super().scope(sdfg, state, scope, order, entry)

    def library(self, sdfg: SDFG, state: SDFGState, node: nodes.LibraryNode) -> None:
        if isinstance(node, CopyLibraryNode) and not isinstance(node, TileOp):
            # A copy the copy lowering inserted between two registers rebinds the value
            source, target = state.in_edges(node)[0].data.data, state.out_edges(node)[0].data.data
            if not (is_register(sdfg.arrays[source]) and is_register(sdfg.arrays[target])):
                raise CodegenError(f"Python backend: a tile kernel copies {node.label!r} through global memory")
            self.line(f"{target} = {source}")
            return
        if not isinstance(node, TileOp):
            raise CodegenError(f"Python backend: no tile kernel form for {type(node).__name__} {node.label!r}")
        self.dialect.statement_type(self, sdfg, state, node).emit()

    def map(self, sdfg, state, scope, order, entry) -> None:
        if entry.map.schedule in dtypes.GPU_SCHEDULES:
            raise CodegenError(f"Python backend: map {entry.map.label!r} nests a GPU schedule in a tile kernel")
        super().map(sdfg, state, scope, order, entry)

    def read(self, sdfg, state, edge) -> str:
        name, desc = edge.data.data, sdfg.arrays[edge.data.data]
        if is_register(desc):
            return name
        return self.dialect.load(name, self.element_offset(desc, edge.data.subset), None, desc.dtype)

    def write(self, sdfg, edge) -> None:
        name, desc = edge.data.data, sdfg.arrays[edge.data.data]
        if is_register(desc):
            value = edge.src_conn if edge.data.wcr is None else f"({edge.data.wcr})({name}, {edge.src_conn})"
            self.line(f"{name} = {value}")
            return
        store = self.dialect.atomic_add if accumulates(edge.data) else self.dialect.store
        self.line(store(name, self.element_offset(desc, edge.data.subset), edge.src_conn, None))

    def element_offset(self, desc: data.Data, subset) -> Element:
        """The one element ``subset`` reads; a kernel moves a global array one element at a time
        outside tile nodes."""
        if subset.num_elements() != 1:
            raise CodegenError("Python backend: outside tile nodes a tile kernel moves one global element at a time")
        return element_at(desc, [expr(begin) for begin, _, _ in subset.ndrange()])

    def body(self, sdfg, node) -> str:
        # An inlined connector reads its array in the body, which the kernel spells as the dialect does
        tree = KernelAccesses(sdfg, self.dialect).visit(ast.parse(node.code.as_string))
        return ast.unparse(ast.fix_missing_locations(tree))

    def copy(self, sdfg, state, edge) -> None:
        path = state.memlet_path(edge)
        src, dst = path[0].src.data, path[-1].dst.data
        for name in (src, dst):
            if not is_register(sdfg.arrays[name]):
                raise CodegenError(f"Python backend: in a tile kernel only tile nodes touch the global array {name!r}")
        self.line(f"{dst} = {src}")

    def view(self, sdfg, state, node) -> None:
        raise CodegenError(f"Python backend: a tile kernel cannot take the view {node.data!r}")


def accumulates(memlet) -> bool:
    """Whether ``memlet`` adds into a global array, which programs of a kernel must do atomically."""
    if memlet.wcr is None:
        return False
    if detect_reduction_type(memlet.wcr) is not dtypes.ReductionType.Sum:
        raise CodegenError(
            f"Python backend: a tile kernel accumulates into global memory by sums only, not {memlet.wcr}"
        )
    return True


def element_at(desc: data.Data, indices: list[str]) -> Element:
    """The element at ``indices`` of ``desc``, with its index offset applied."""
    shifted = [index if offset == 0 else f"{index} + {expr(offset)}" for index, offset in zip(indices, desc.offset)]
    return Element(shifted, [expr(stride) for stride in desc.strides])


class KernelAccesses(ast.NodeTransformer):
    """A tasklet body in a kernel: a register scalar is a value, a global element is a dialect load or store."""

    def __init__(self, sdfg: SDFG, dialect: Dialect):
        self.sdfg, self.dialect = sdfg, dialect

    def container(self, node: ast.AST) -> tuple[str, data.Data] | None:
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in self.sdfg.arrays:
            return node.value.id, self.sdfg.arrays[node.value.id]
        return None

    def offset(self, node: ast.Subscript, desc: data.Data) -> Element:
        index = node.slice
        indices = index.elts if isinstance(index, ast.Tuple) else [index]
        return element_at(desc, [ast.unparse(self.visit(i)) for i in indices])

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        found = self.container(node)
        if found is None:
            return self.generic_visit(node)
        name, desc = found
        if is_register(desc):
            if desc.total_size != 1:
                raise CodegenError(f"Python backend: a kernel tasklet indexes the register tile {name!r}")
            return ast.Name(id=name, ctx=node.ctx)
        return ast.parse(self.dialect.load(name, self.offset(node, desc), None, desc.dtype), mode="eval").body

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        node.value = self.visit(node.value)
        found = self.container(node.targets[0]) if len(node.targets) == 1 else None
        if found is None or is_register(found[1]):
            node.targets = [self.visit(target) for target in node.targets]
            return node
        name, desc = found
        store = self.dialect.store(name, self.offset(node.targets[0], desc), ast.unparse(node.value), None)
        return ast.parse(store, mode="exec").body[0]


def per_lane(text: str, dialect: Dialect, widths: list[int]) -> str:
    """A per-lane C++ expression of a tile node over whole tiles: lanes become lane tensors, logic becomes bitwise."""
    text = CPP_CAST.sub("(", text).replace("&&", " & ").replace("||", " | ")
    text = re.sub(r"!(?!=)", "~", text)
    return LANE.sub(lambda m: dialect.lane(widths, int(m.group(1))), text)


def scalar(text: str) -> str:
    """A scalar expression of the symbols in scope, as Python."""
    return expr(symbolic.pystr_to_symbolic(text))


class TileStatement:
    """The statement one tile node is in a kernel."""

    __slots__ = ("emitter", "dialect", "sdfg", "state", "node", "inputs", "outputs")

    def __init__(self, emitter: KernelEmitter, sdfg: SDFG, state: SDFGState, node: TileOp):
        self.emitter, self.dialect, self.sdfg, self.state, self.node = emitter, emitter.dialect, sdfg, state, node
        self.inputs = {e.dst_conn: e for e in state.in_edges(node) if not e.data.is_empty()}
        self.outputs = {e.src_conn: e for e in state.out_edges(node) if not e.data.is_empty()}

    def emit(self) -> None:
        node = self.node
        if isinstance(node, MaskedCopyLibraryNode):
            self.masked_copy()
            return
        if isinstance(node, (TileBinop, TileUnop, TileFMA, TileITE)):
            out, value = self.elementwise()
        elif isinstance(node, TileMaskGen):
            out, value = "_o", self.mask_gen()
        elif isinstance(node, TileIota):
            if node.extra_inputs:
                raise CodegenError(f"Python backend: {node.label!r} is an indirect iota; tile kernels take affine ones")
            out, value = "_dst", per_lane(node.expr, self.dialect, node.widths)
            if not LANE.search(node.expr):
                value = self.dialect.full(node.widths, value, self.sdfg.arrays[self.outputs[out].data.data].dtype)
        elif isinstance(node, TileReduce):
            out, value = "_dst", self.reduce()
        elif isinstance(node, TileScan):
            out, value = "_dst", self.dialect.scan(node.op, self.input("_src"))
        elif isinstance(node, TileMMA):
            out, value = "_c", self.mma()
        else:
            raise CodegenError(f"Python backend: no tile kernel form for {type(node).__name__} {node.label!r}")
        self.assign(out, value)

    def input(self, conn: str) -> str:
        name = self.inputs[conn].data.data
        if not is_register(self.sdfg.arrays[name]):
            raise CodegenError(f"Python backend: {self.node.label!r} reads the global array {name!r} as an operand")
        return name

    def assign(self, conn: str, value: str) -> None:
        name = self.outputs[conn].data.data
        self.emitter.line(f"{name} = {self.dialect.cast(value, self.sdfg.arrays[name].dtype)}")

    def operand(self, conn: str, kind: str, text: str | None) -> str:
        return f"({scalar(text)})" if kind == SYMBOL else self.input(conn)

    def masked(self, value: str, zero: str = "0") -> str:
        if not getattr(self.node, "has_mask", False):
            return value
        return self.dialect.select(self.input("_mask"), value, zero)

    def elementwise(self) -> tuple[str, str]:
        node = self.node
        values = {o.conn: self.operand(o.conn, o.kind, o.expr) for o in node.operands()}
        if isinstance(node, TileBinop):
            value, out = ops.PY_BINARY_OPS[node.op].format(a=values["_a"], b=values["_b"], f=self.dialect), "_c"
        elif isinstance(node, TileUnop):
            out = "_c"
            value = (
                values["_a"]
                if node.op in ops.CAST_OPS
                else ops.PY_UNARY_OPS[node.op].format(a=values["_a"], f=self.dialect)
            )
        elif isinstance(node, TileFMA):
            value, out = f"{self.dialect['fma']}({values['_a']}, {values['_b']}, {values['_c']})", "_o"
        else:
            value, out = f"{self.dialect['where']}({values['_mask']}, {values['_t']}, {values['_e']})", "_o"
        # A lane-invariant output is one value every active lane agrees on; masked lanes only consume it
        if isinstance(self.sdfg.arrays[self.outputs[out].data.data], data.Scalar):
            return out, value
        if all(o.kind != TILE for o in node.operands()):
            value = self.dialect.full(node.widths, value, self.sdfg.arrays[self.outputs[out].data.data].dtype)
        return out, self.masked(value)

    def lane_mask(self) -> str | None:
        """The lanes in range: ``_mask`` and the ``lane_bounds`` of a tile at the end of its map."""
        widths = self.node.widths
        terms = [f"({self.dialect.lane(widths, k)} < ({scalar(b)}))" for k, b in enumerate(self.node.lane_bounds)]
        if getattr(self.node, "has_mask", False):
            terms.append(self.input("_mask"))
        return " & ".join(terms) or None

    def offsets(self, name: str, subset) -> Element:
        """The elements of the lanes of a window ``subset`` of the global array ``name``: the dims of extent other
        than 1 are the lanes of the tile, in order."""
        desc = self.sdfg.arrays[name]
        widths, lane, indices = self.node.widths, 0, []
        for (begin, end, step), offset in zip(subset.ndrange(), desc.offset):
            index = expr(begin + offset)
            if (end - begin + 1) != 1:
                index = f"{index} + {self.dialect.lane(widths, lane)}" + ("" if step == 1 else f" * {expr(step)}")
                lane += 1
            indices.append(index)
        return Element(indices, [expr(stride) for stride in desc.strides], tile=True)

    def masked_copy(self) -> None:
        source, target = self.inputs["_cpy_in"], self.outputs["_cpy_out"]
        src, dst = source.data.data, target.data.data
        src_register, dst_register = is_register(self.sdfg.arrays[src]), is_register(self.sdfg.arrays[dst])
        if src_register and dst_register:
            self.emitter.line(f"{dst} = {self.masked(src)}")
        elif dst_register:
            offsets = self.offsets(src, source.data.subset)
            self.emitter.line(
                f"{dst} = {self.dialect.load(src, offsets, self.lane_mask(), self.sdfg.arrays[src].dtype)}"
            )
        else:
            subset = target.data.subset if target.data.data == dst else target.data.other_subset
            offsets = self.offsets(dst, subset)
            store = self.dialect.atomic_add if accumulates(target.data) else self.dialect.store
            self.emitter.line(store(dst, offsets, src, self.lane_mask()))

    def mask_gen(self) -> str:
        node, lanes = self.node, []
        for k, (var, bound) in enumerate(zip(node.iter_vars, node.global_ubs)):
            lanes.append(f"(({scalar(var)}) + {self.dialect.lane(node.widths, k)} < ({scalar(bound)}))")
        if node.guard_predicate:
            lanes.append(f"({per_lane(node.guard_predicate, self.dialect, node.widths)})")
        return " & ".join(lanes)

    def reduce(self) -> str:
        node = self.node
        if node.op not in ops.PY_REDUCE_OPS:
            raise CodegenError(f"Python backend: no tile reduction combines with {node.op!r}")
        source = self.masked(self.input("_src"), self.dialect.identity(node.op))
        return self.dialect.reduce(node.op, source, node.axis, len(node.widths))

    def mma(self) -> str:
        node = self.node
        value = f"({node.alpha}) * {self.dialect.dot(self.input('_a'), self.input('_b'))}"
        if "_cin" in self.inputs:
            value += f" + ({node.beta}) * {self.input('_cin')}"
        return value
