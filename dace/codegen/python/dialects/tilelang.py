# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""TileLang: a kernel holds tiles in buffers and writes them one ``T.Parallel`` lane loop at a time.

Triton and cuTile compute with tile values; TileLang computes with buffers, so a tile node is a lane loop over its
output (``out[lane0] = a[lane0] + b[lane0]``), a register tile a fragment allocated at the top of the kernel, and a
register scalar a ``T.alloc_var`` every thread holds. Every thread runs a statement outside a lane loop, so a
write to global memory outside one runs in ``T.Parallel(1)``, on one thread.
"""

import ast
import re

from dace import data, dtypes, symbolic
from dace.codegen.exceptions import CodegenError
from dace.codegen.python.emit import expr
from dace.codegen.python.kernel import (
    Dialect,
    Element,
    KernelEmitter,
    TileStatement,
    accumulates,
    is_register,
    tile_extent,
)
from dace.libraries.standard.nodes.copy.node import CopyLibraryNode
from dace.libraries.tileops.nodes import MaskedCopyLibraryNode, TileMMA, TileReduce, TileScan

IMPORTS = """\
import torch
import tilelang
import tilelang.language as T
"""

#: The reductions and scans TileLang has; a product has neither, a minimum has no scan.
REDUCTIONS = {"+": "T.reduce_sum", "max": "T.reduce_max", "min": "T.reduce_min"}
SCANS = {"+": "T.cumsum", "max": "T.cummax"}
#: The block ids ``T.Kernel`` binds, by axis.
BLOCK_IDS = ("bx", "by", "bz")
SERIAL = re.compile(r"\bin range\(")


def dtype_name(dtype: dtypes.typeclass) -> str:
    return {dtypes.bool_: "bool"}.get(dtype, dtype.to_string())


def lanes(widths: list[int]) -> list[str]:
    return [f"lane{dim}" for dim in range(len(widths))]


def lane_loop(widths: list[int]) -> str:
    return f"for {', '.join(lanes(widths))} in T.Parallel({', '.join(map(str, widths))}):"


def one_thread(statement: str) -> str:
    return f"for lane in T.Parallel(1):\n    {statement}"


class TileLangEmitter(KernelEmitter):
    """Allocates every register container at the top of the kernel, a tile as a buffer and a scalar as a var."""

    __slots__ = ("allocations", "threads", "mma_threads", "shared")

    def __init__(self, sdfg, dialect):
        super().__init__(sdfg, dialect)
        self.allocations: list[str] = []
        self.threads = 128
        #: The threads of a kernel with an MMA, whose warps must each own whole 16 x 8 pieces of the output.
        self.mma_threads: int | None = None
        #: The tiles an MMA reads, which ``T.gemm`` takes from shared memory.
        self.shared = {
            edge.data.data
            for node, state in sdfg.all_nodes_recursive()
            if isinstance(node, TileMMA)
            for edge in state.in_edges(node)
            if edge.dst_conn in ("_a", "_b")
        }

    def line(self, text: str) -> None:
        super().line(SERIAL.sub("in T.serial(", text))

    def allocate(self, name: str, desc: data.Data) -> None:
        super().allocate(name, desc)
        if desc.total_size == 1:
            self.allocations.append(f'{name} = T.alloc_var("{dtype_name(desc.dtype)}")')
            return
        kind = "shared" if name in self.shared else "fragment"
        # A register has no ``ONE`` dim: the copy filling an index tile writes only the dims it varies over
        extents = [str(tile_extent(s)) for s in desc.shape if not symbolic.has_one_marker(s)] or ["1"]
        shape = ", ".join(extents if is_register(desc) else (expr(s) for s in desc.shape))
        self.allocations.append(f'{name} = T.alloc_{kind}(({shape},), "{dtype_name(desc.dtype)}")')

    def copy(self, sdfg, state, edge) -> None:
        path = state.memlet_path(edge)
        src, dst = path[0].src.data, path[-1].dst.data
        super().copy(sdfg, state, edge)  # a register copy writes ``dst = src``
        if sdfg.arrays[dst].total_size != 1:
            self.lines[-1] = self.lines[-1].replace(f"{dst} = {src}", f"T.copy({src}, {dst})")


class TileLangStatement(TileStatement):
    """A tile node as lane loops over buffers."""

    def __init__(self, emitter, sdfg, state, node):
        super().__init__(emitter, sdfg, state, node)
        emitter.threads = max(emitter.threads, node.group_threads())
        if isinstance(node, TileMMA):
            m, _, n = node.widths
            warps = max(1, min(node.num_warps, (m // 16) * (n // 8)))
            emitter.mma_threads = min(emitter.mma_threads or 32 * warps, 32 * warps)

    def indexed(self, value: str, widths: list[int]) -> str:
        """``value`` read at the lanes: a register tile becomes its element at the lane indices, lane 0 along a dim
        of extent 1, which an index tile has on the dims it does not vary over."""
        tiles = {n: d.shape for n, d in self.sdfg.arrays.items() if is_register(d) and d.total_size != 1}

        class Lanes(ast.NodeTransformer):
            def visit_Name(self, node: ast.Name) -> ast.AST:
                if node.id not in tiles:
                    return node
                # A ``ONE`` dim is not allocated (``allocate``); a literal extent 1 is, and reads lane 0
                kept = [(lane, e) for lane, e in zip(lanes(widths), tiles[node.id]) if not symbolic.has_one_marker(e)]
                at = [lane if extent != 1 else "0" for lane, extent in kept] or ["0"]
                index = ast.parse(", ".join(at), mode="eval").body
                return ast.Subscript(value=node, slice=index, ctx=node.ctx)

        return ast.unparse(Lanes().visit(ast.parse(value, mode="eval")))

    def assign(self, conn: str, value: str) -> None:
        name = self.outputs[conn].data.data
        desc = self.sdfg.arrays[name]
        widths = list(self.node.widths)
        cast = self.dialect.cast(self.indexed(value, widths), desc.dtype)
        if desc.total_size == 1:
            self.emitter.line(f"{name} = {cast}")
        else:
            self.emitter.line(f"{lane_loop(widths)}\n    {name}[{', '.join(lanes(widths))}] = {cast}")

    def emit(self) -> None:
        node = self.node
        if isinstance(node, MaskedCopyLibraryNode):
            self.masked_copy()
        elif isinstance(node, TileReduce):
            self.reduction()
        elif isinstance(node, TileScan):
            self.prefix()
        elif isinstance(node, TileMMA):
            self.gemm()
        else:
            super().emit()

    def masked_copy(self) -> None:
        source, target = (
            self.inputs[CopyLibraryNode.INPUT_CONNECTOR_NAME],
            self.outputs[CopyLibraryNode.OUTPUT_CONNECTOR_NAME],
        )
        src, dst = source.data.data, target.data.data
        widths = list(self.node.widths)
        mask = self.lane_mask()
        if is_register(self.sdfg.arrays[dst]):
            if is_register(self.sdfg.arrays[src]):
                self.assign(CopyLibraryNode.OUTPUT_CONNECTOR_NAME, self.masked(src))
            else:
                self.assign(
                    CopyLibraryNode.OUTPUT_CONNECTOR_NAME,
                    self.dialect.load(src, self.offsets(src, source.data.subset), mask, self.sdfg.arrays[src].dtype),
                )
            return
        subset = target.data.subset if target.data.data == dst else target.data.other_subset
        self.write(dst, self.offsets(dst, subset), src, accumulates(target.data))

    def write(self, name: str, at: Element, value: str, accumulate: bool) -> None:
        widths, mask = list(self.node.widths), self.lane_mask()
        address = self.indexed(f"{name}[{at.flat()}]", widths)
        value = self.indexed(value, widths)
        write = f"T.atomic_add({address}, {value})" if accumulate else f"{address} = {value}"
        body = write if mask is None else f"if {self.indexed(mask, widths)}:\n        {write}"
        self.emitter.line(f"{lane_loop(widths)}\n    {body}")

    def reduction(self) -> None:
        node = self.node
        if node.op not in REDUCTIONS:
            raise CodegenError(f"Python backend: TileLang has no tile reduction with {node.op!r}")
        if node.axis is None and len(node.widths) != 1:
            raise CodegenError(f"Python backend: TileLang reduces {node.label!r} fully only over one dim")
        source, out = self.input("_src"), self.outputs["_dst"].data.data
        out_desc = self.sdfg.arrays[out]
        if node.has_mask:
            masked = f"{out}_masked"
            self.emitter.allocate(masked, self.sdfg.arrays[source])
            self.assign_buffer(masked, self.masked(source, IDENTITY[node.op]))
            source = masked
        target = out
        if out_desc.total_size == 1:
            target = f"{out}_reduced"
            self.emitter.allocations.append(f'{target} = T.alloc_fragment((1,), "{dtype_name(out_desc.dtype)}")')
        self.emitter.line(f"{REDUCTIONS[node.op]}({source}, {target}, dim={node.axis or 0})")
        if target != out:
            self.emitter.line(f"{out} = {target}[0]")

    def assign_buffer(self, name: str, value: str) -> None:
        widths = list(self.node.widths)
        self.emitter.line(
            f"{lane_loop(widths)}\n    {name}[{', '.join(lanes(widths))}] = {self.indexed(value, widths)}"
        )

    def prefix(self) -> None:
        if self.node.op not in SCANS:
            raise CodegenError(f"Python backend: TileLang has no tile scan with {self.node.op!r}")
        self.emitter.line(f"{SCANS[self.node.op]}({self.input('_src')}, {self.outputs['_dst'].data.data}, dim=0)")

    def gemm(self) -> None:
        node = self.node
        if str(node.alpha) not in ("1", "1.0"):
            raise CodegenError(f"Python backend: TileLang multiplies {node.label!r} with alpha 1 only")
        out = self.outputs["_c"].data.data
        if "_cin" in self.inputs:
            if str(node.beta) not in ("1", "1.0"):
                raise CodegenError(f"Python backend: TileLang accumulates {node.label!r} with beta 1 only")
            if self.input("_cin") != out:
                self.emitter.line(f"T.copy({self.input('_cin')}, {out})")
        else:
            self.emitter.line(f"T.clear({out})")
        self.emitter.line(f"T.gemm({self.input('_a')}, {self.input('_b')}, {out})")


#: What a masked lane of a reduction contributes.
IDENTITY = {"+": "0", "max": "-T.infinity('float64')", "min": "T.infinity('float64')"}


class TileLang(Dialect):
    imports = IMPORTS
    decorator = "@T.prim_func"
    module = "T"
    body_depth = 2
    functions = {
        "int_floor": "T.floordiv",
        "py_mod": "T.floormod",
        "c_mod": "T.truncmod",
        "int_ceil": "T.ceildiv",
        "Min": "T.min",
        "Max": "T.max",
        "Abs": "T.abs",
        "ipow": "T.pow",
        "minimum": "T.min",
        "maximum": "T.max",
        "where": "T.if_then_else",
    }

    @property
    def emitter_type(self) -> type[KernelEmitter]:
        return TileLangEmitter

    @property
    def statement_type(self) -> type[TileStatement]:
        return TileLangStatement

    def program_id(self, axis: int) -> str:
        return BLOCK_IDS[axis]

    def index_tile(self, name: str, shape: list[int], widths: list[int]) -> str:
        # Read lane by lane: ``indexed`` takes lane 0 on the dims the index tile does not vary over
        return name

    def lane(self, widths: list[int], dim: int) -> str:
        return f"lane{dim}"

    def load(self, base: str, at: Element, mask: str | None, dtype: dtypes.typeclass) -> str:
        element = f"{base}[{at.flat()}]"
        return element if mask is None else f"T.if_then_else({mask}, {element}, 0)"

    def store(self, base: str, at: Element, value: str, mask: str | None) -> str:
        return one_thread(f"{base}[{at.flat()}] = {value}")

    def atomic_add(self, base: str, at: Element, value: str, mask: str | None) -> str:
        return one_thread(f"T.atomic_add({base}[{at.flat()}], {value})")

    def cast(self, value: str, dtype: dtypes.typeclass) -> str:
        return f'T.cast({value}, "{dtype_name(dtype)}")'

    def full(self, widths: list[int], value: str, dtype: dtypes.typeclass) -> str:
        return value  # the lane loop of the statement broadcasts it

    def define(self, name: str, kernel: TileLangEmitter, arrays: list[str], symbols: list[str], grid: list[str]) -> str:
        sdfg = kernel.sdfg
        params = [f'{a}: T.Tensor((T.dynamic("{a}_size"),), "{dtype_name(sdfg.arrays[a].dtype)}")' for a in arrays]
        params += [f"{s}: T.{dtype_name(sdfg.symbols.get(s, dtypes.int64))}" for s in symbols]
        launch = f"T.Kernel({', '.join(grid)}, threads={kernel.mma_threads or kernel.threads})"
        lines = [
            self.decorator,
            f"def {name}({', '.join(params)}):",
            f"    with {launch} as ({', '.join(BLOCK_IDS[: len(grid)])},):",
            *kernel.lines[: len(grid)],
            *(f"        {line}" for line in kernel.allocations),
            *kernel.lines[len(grid) :],
            "",
            "",
            f"{name} = tilelang.compile({name})",
        ]
        return "\n".join(line.replace("int_ceil(", "T.ceildiv(") for line in lines)

    def launch(self, kernel: str, grid: list[str], arrays: list[str], symbols: list[str]) -> str:
        args = [f"torch.from_dlpack({a}).view(-1)" for a in arrays] + symbols
        return f"{kernel}({', '.join(args)})"
