# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""CUDA Tile C++: a ``__tile_global__`` kernel of the C++/CUDA code generator, one tile block per map iteration.

The kernel body is walked as the Python backend walks a tile kernel (:class:`~dace.codegen.python.kernel.
KernelEmitter`), which writes Python; :class:`CppKernelEmitter` turns each statement into C++ as it is written. The
dialect spells the Tile C++ API in Python a parser accepts and :func:`cpp` maps back: ``ct.sum[0](x)`` is
``ct::sum<0>(x)``, ``ct.tile[float, ct.shape[16, 16]]`` is ``ct::tile<float, ct::shape<16, 16>>``. Tiles are
immutable values, as in Triton, and a pointer plus an offset tile is the tile of pointers ``ct::load`` reads.
"""

import ast
import contextlib
import re
from collections.abc import Iterator

import dace.library
from dace import SDFG, dtypes
from dace.codegen import cppunparse
from dace.codegen.exceptions import CodegenError
from dace.codegen.python.kernel import Dialect, Element, KernelEmitter, holds_tile_ops
from dace.config import Config
from dace.libraries.standard.nodes.copy.node import CopyLibraryNode
from dace.libraries.tileops.nodes.tile_op import TileOp
from dace.sdfg import nodes

#: A name the dialect spells with template arguments in brackets, ``ct.sum[0]``.
TEMPLATED = re.compile(r"\b(?:(?:ct|dace_tile)(?:\.\w+)+|static_cast)\[")
#: A dotted name in the ``ct`` namespace, ``ct.memory_order.relaxed``.
NAMESPACED = re.compile(r"\b(?:ct|std|dace_tile)(?:\.\w+)+")
#: The lane of a one-element access, which Tile C++ moves as a one-lane tile.
ONE_LANE = "ct.iota[ct.tile[int, ct.shape[1]]]()"
#: Written ahead of the tile kernels of a translation unit: the header, and a cast that takes a scalar (a C++
#: ``static_cast``) or a tile (``element_cast``).
PRELUDE = """\
#include "cuda_tile.h"
#include <type_traits>
namespace dace_tile {
template <typename T, typename X>
__tile__ inline auto cast(X x) {
    if constexpr (std::is_arithmetic_v<X>) {
        return static_cast<T>(x);
    } else {
        return cuda::tiles::element_cast<T>(x);
    }
}
}  // namespace dace_tile
"""
#: The C++ element type of each DaCe type, spelled so a Python parser takes it (``std.int64_t``).
CPP_TYPES = {
    dtypes.float32: "float",
    dtypes.float64: "double",
    dtypes.bool_: "bool",
    dtypes.int8: "std.int8_t",
    dtypes.int16: "std.int16_t",
    dtypes.int32: "std.int32_t",
    dtypes.int64: "std.int64_t",
    dtypes.uint8: "std.uint8_t",
    dtypes.uint16: "std.uint16_t",
    dtypes.uint32: "std.uint32_t",
    dtypes.uint64: "std.uint64_t",
}


def matching(text: str, start: int) -> int:
    """The index of the bracket closing the one at ``start``."""
    depth = 0
    for index in range(start, len(text)):
        depth += {"[": 1, "]": -1}.get(text[index], 0)
        if depth == 0:
            return index
    raise CodegenError(f"Python backend: unbalanced brackets in {text!r}")


def cpp(text: str) -> str:
    """C++ from the Python spelling of the Tile C++ API: template brackets and namespaces, after ``cppunparse``."""
    found = TEMPLATED.search(text)
    while found is not None:
        close = matching(text, found.end() - 1)
        inner = text[found.end() : close]
        if inner.startswith("std::make_tuple(") and inner.endswith(")"):
            inner = inner[len("std::make_tuple(") : -1]
        text = f"{text[: found.end() - 1]}<{inner}>{text[close + 1 :]}"
        found = TEMPLATED.search(text)
    return NAMESPACED.sub(lambda m: m.group(0).replace(".", "::"), text)


def cpp_type(dtype: dtypes.typeclass) -> str:
    if dtype not in CPP_TYPES:
        raise CodegenError(f"Python backend: CUDA Tile C++ takes no {dtype} elements")
    return CPP_TYPES[dtype]


class CppKernelEmitter(KernelEmitter):
    """Writes the kernel body in C++: each Python statement of the walker becomes one C++ statement."""

    __slots__ = ("declared", "python")

    def __init__(self, sdfg, dialect):
        super().__init__(sdfg, dialect)
        #: Per open C++ block, the names it declares; a name is declared ``auto`` where it is first assigned.
        self.declared: list[set[str]] = [set()]
        #: The Python statements the body was translated from, which say what the kernel binds.
        self.python: list[ast.AST] = []

    def raw(self, text: str) -> None:
        self.lines.append("    " * self.depth + text)

    def known(self, name: str) -> bool:
        return any(name in scope for scope in self.declared)

    def line(self, text: str) -> None:
        for part in text.split("\n"):
            tree = ast.parse(part.strip())
            self.python.append(tree)
            self.statement(tree.body[0])

    def statement(self, node: ast.stmt) -> None:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name, value = node.targets[0].id, self.expression(node.value)
            if self.known(name):
                self.raw(f"{name} = {value};")
            else:
                self.declared[-1].add(name)
                self.raw(f"auto {name} = {value};")
        elif isinstance(node, (ast.Break, ast.Continue, ast.Return)):
            self.raw(f"{type(node).__name__.lower()};")
        elif isinstance(node, ast.Pass):
            return
        elif isinstance(node, ast.Expr):
            self.raw(f"{self.expression(node.value)};")
        else:
            raise CodegenError(f"Python backend: no CUDA Tile C++ form for {ast.unparse(node)!r}")

    def expression(self, node: ast.expr) -> str:
        return cpp(cppunparse.pyexpr2cpp(ast.unparse(node)))

    @contextlib.contextmanager
    def block(self, header: str) -> Iterator[None]:
        self.raw(f"{self.header(header)} {{")
        self.depth += 1
        self.declared.append(set())
        yield
        self.declared.pop()
        self.depth -= 1
        self.raw("}")

    def bindings(self) -> tuple[set[str], set[str]]:
        found = [n for tree in self.python for n in ast.walk(tree) if isinstance(n, ast.Name)]
        return {n.id for n in found}, {n.id for n in found if isinstance(n.ctx, ast.Store)}

    def header(self, header: str) -> str:
        """The C++ head of a block the walker opens with the Python ``header``."""
        if header not in ("else:", "while True:"):
            self.python.append(ast.parse(f"{header.removeprefix('el')}\n    pass"))
        header = header.rstrip(":")
        if header == "else":
            return "else"
        if header == "while True":
            return "while (true)"
        if header.startswith("elif "):
            return f"else if ({self.expression(ast.parse(header[5:], mode='eval').body)})"
        if header.startswith("if "):
            return f"if ({self.expression(ast.parse(header[3:], mode='eval').body)})"
        loop = ast.parse(f"{header}:\n    pass").body[0]
        if isinstance(loop, ast.For) and isinstance(loop.iter, ast.Call) and len(loop.iter.args) >= 2:
            variable = loop.target.id
            start, stop, *step = (self.expression(arg) for arg in loop.iter.args)
            step = step[0] if step else "1"
            compare = ">" if step.lstrip("(").startswith("-") else "<"
            self.declared[-1].add(variable)
            return f"for (auto {variable} = {start}; {variable} {compare} {stop}; {variable} += {step})"
        raise CodegenError(f"Python backend: no CUDA Tile C++ form for the block {header!r}")

    def statements(self, body: str) -> None:
        # A tasklet body is compound Python: cppunparse translates it whole, declaring its new locals
        self.python.append(ast.parse(body))
        known = dict.fromkeys(set().union(*self.declared))
        for text in cppunparse.py2cpp(body, defined_symbols=known).splitlines():
            self.raw(cpp(text))


class CudaTileCpp(Dialect):
    """The Tile C++ API, spelled in Python for :func:`cpp`."""

    module = "ct"
    functions = {
        "minimum": "ct.min",
        "maximum": "ct.max",
        "where": "ct.select",
        "Min": "ct.min",
        "Max": "ct.max",
        "Abs": "ct.abs",
        "ipow": "ct.pow",
    }

    @property
    def emitter_type(self) -> type[KernelEmitter]:
        return CppKernelEmitter

    def program_id(self, axis: int) -> str:
        # ``bid`` is unsigned, and an unsigned index would not mix with the signed lane tiles
        return f"static_cast[int](ct.bid().{'xyz'[axis]})"

    def tile_type(self, widths: list[int], dtype: dtypes.typeclass) -> str:
        return f"ct.tile[{cpp_type(dtype)}, ct.shape[{', '.join(map(str, widths))}]]"

    def lane(self, widths: list[int], dim: int) -> str:
        lanes = f"ct.iota[ct.tile[int, ct.shape[{widths[dim]}]]]()"
        if len(widths) == 1:
            return lanes
        column = ", ".join(str(width) if k == dim else "1" for k, width in enumerate(widths))
        return f"ct.broadcast[ct.shape[{', '.join(map(str, widths))}]](ct.reshape[ct.shape[{column}]]({lanes}))"

    def pointers(self, base: str, at: Element) -> str:
        """The pointer tile of ``at``; one element is a one-lane tile, since a bare pointer is no storeable tile."""
        return f"{base} + {at.flat()}" + ("" if at.tile else f" + {ONE_LANE}")

    def load(self, base: str, at: Element, mask: str | None, dtype: dtypes.typeclass) -> str:
        if not at.tile:
            return f"ct.reshape[ct.shape[()]](ct.load({self.pointers(base, at)}))"
        if mask is None:
            return f"ct.load({self.pointers(base, at)})"
        return f"ct.load_masked({self.pointers(base, at)}, {mask}, static_cast[{cpp_type(dtype)}](0))"

    def store(self, base: str, at: Element, value: str, mask: str | None) -> str:
        if not at.tile:
            value = f"ct.broadcast[ct.shape[1]]({value})"
        if mask is None:
            return f"ct.store({self.pointers(base, at)}, {value})"
        return f"ct.store_masked({self.pointers(base, at)}, {value}, {mask})"

    def cast(self, value: str, dtype: dtypes.typeclass) -> str:
        return f"dace_tile.cast[{cpp_type(dtype)}]({value})"

    def select(self, condition: str, value: str, otherwise: str) -> str:
        # Both arms are one tile type: the other arm is a tile of that type
        return f"ct.select({condition}, {value}, ct.full[decltype({value})]({otherwise}))"

    def full(self, widths: list[int], value: str, dtype: dtypes.typeclass) -> str:
        return f"ct.full[{self.tile_type(widths, dtype)}](static_cast[{cpp_type(dtype)}]({value}))"

    def dot(self, a: str, b: str) -> str:
        return f"ct.matmul({a}, {b})"

    def identity(self, op: str) -> str:
        return {"min": "INFINITY", "max": "-INFINITY"}.get(op, super().identity(op))

    def reduce(self, op: str, value: str, axis: int | None, rank: int) -> str:
        function = {"+": "sum", "*": "prod", "max": "reduce_max", "min": "reduce_min"}[op]
        # A full reduction folds the last dim first, so each fold leaves the dims before it in place
        for dim in [axis] if axis is not None else reversed(range(rank)):
            value = f"ct.{function}[{dim}]({value})"
        return value

    def scan(self, op: str, value: str) -> str:
        if op not in ("+", "*"):
            raise CodegenError(f"Python backend: CUDA Tile C++ has no tile scan with {op!r}")
        return f"ct.partial_{'sum' if op == '+' else 'prod'}[0]({value})"

    def atomic_add(self, base: str, at: Element, value: str, mask: str | None) -> str:
        function = "atomic_add" if mask is None else "atomic_add_masked"
        masked = "" if mask is None else f", {mask}"
        if not at.tile:
            value = f"ct.broadcast[ct.shape[1]]({value})"
        return f"ct.{function}[ct.memory_order.relaxed]({self.pointers(base, at)}, {value}{masked})"


def is_tile_kernel(state, entry: nodes.MapEntry) -> bool:
    """Whether the CUDA generator writes the map at ``entry`` as a CUDA Tile C++ kernel, which the GPU preprocessing
    passes leave alone: a ``GPU_Device`` map over BLOCK tile nodes under ``compiler.cuda.tile_backend: cuda_tile``."""
    return (
        Config.get("compiler", "cuda", "tile_backend") == "cuda_tile"
        and entry.map.schedule is dtypes.ScheduleType.GPU_Device
        and holds_tile_ops(state.sdfg, state, entry)
    )


def pin_tile_blocks(sdfg: SDFG) -> None:
    """Give every tile kernel its launch block: one tile thread, which runs the whole tile."""
    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.MapEntry) and is_tile_kernel(state, node):
            node.map.gpu_block_size = [1, 1, 1]


def tile_kernel_nodes(sdfg: SDFG) -> set[nodes.LibraryNode]:
    """The library nodes a CUDA Tile C++ kernel writes itself, which stay unexpanded: the tile nodes and the copies of
    a ``GPU_Device`` map over BLOCK tiles."""
    found: set[TileOp] = set()
    for node, state in sdfg.all_nodes_recursive():
        if not (isinstance(node, nodes.MapEntry) and is_tile_kernel(state, node)):
            continue
        for inner in state.scope_subgraph(node).nodes():
            nested = inner.sdfg.all_nodes_recursive() if isinstance(inner, nodes.NestedSDFG) else [(inner, state)]
            found |= {n for n, _ in nested if isinstance(n, (TileOp, CopyLibraryNode))}
    return found


def check_tile_toolchain(needed: bool) -> None:
    """Refuse a configuration tile code cannot build in: it takes C++20 and sm_80 or newer."""
    if not needed:
        return
    architectures = [a for a in Config.get("compiler", "cuda", "cuda_arch").replace(";", ",").split(",") if a.strip()]
    if any(int(re.sub(r"\D", "", a)) < 80 for a in architectures):
        raise CodegenError(f"CUDA Tile C++ needs sm_80 or newer, not compiler.cuda.cuda_arch={architectures}")
    if int(Config.get("compiler", "cpp_standard")) < 20:
        raise CodegenError("CUDA Tile C++ needs compiler.cpp_standard 20 or newer")


@dace.library.environment
class CudaTileCppFlags:
    """The nvcc arguments of tile code (``compiler.cuda.tile_args``), which a build takes only when the CUDA
    generator writes a tile kernel into it."""

    cmake_minimum_version = None
    cmake_packages: list[str] = []
    cmake_includes: list[str] = []
    cmake_libraries: list[str] = []
    cmake_compile_flags: list[str] = []
    cmake_link_flags: list[str] = []
    cmake_files: list[str] = []
    headers: dict[str, list[str]] = {}
    state_fields: list[str] = []
    init_code = ""
    finalize_code = ""
    dependencies: list[str] = []

    @staticmethod
    def cmake_variables() -> dict[str, str]:
        return {
            "CMAKE_CUDA_FLAGS": "_DACE_CMAKE_EXPAND{CMAKE_CUDA_FLAGS} " + Config.get("compiler", "cuda", "tile_args")
        }
