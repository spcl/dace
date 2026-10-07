# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lowering snapshot of every tile-op library node.

Builds a tiny SDFG per node configuration (operand kinds, op, dtypes, widths, mask, shapes), then for every target ISA
resolves the implementation through ``select_tile_implementation`` and records the tasklet the expansion emits together
with the environments it declares. The result of each node type is hashed, so a restructuring of the library that must
not change what is emitted can be checked against ``golden_lowering_digests.json`` in one run.

``python golden_lowering.py dump <file>`` writes every recorded lowering as JSON, which is how a digest mismatch is
narrowed to the configuration that changed; ``python golden_lowering.py update`` rewrites the digest file.
"""

import hashlib
import itertools
import json
import os
import sys
import warnings
from collections.abc import Callable, Iterator
from typing import NamedTuple
from unittest import mock

import dace
from dace.libraries.tileops import (
    MaskedCopyLibraryNode,
    TileBinop,
    TileFMA,
    TileIota,
    TileITE,
    TileGather,
    TileMaskGen,
    TileMMA,
    TileReduce,
    TileScatter,
    TileUnop,
)
import dace.libraries.tileops.dispatch as dispatch
from dace.libraries.tileops.dispatch import ISA
from dace.libraries.tileops.alignment import STRIDE_GUARD_PREFIX, TILE_GUARD_STATE_LABEL, TILE_MAIN_MARKER

DIGEST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_lowering_digests.json")

#: The implementation-selection ISAs; ``AUTO`` resolves to the host ISA, pinned below.
ISAS = (ISA.SCALAR, ISA.AVX512, ISA.AVX2, ISA.ARM_SVE, ISA.ARM_NEON, ISA.CUDA, ISA.AUTO)
#: What the host-feature probes answer while lowering, so the snapshot does not depend on the machine.
HOST_ISAS = frozenset({ISA.AVX512, ISA.AVX2, ISA.ARM_SVE, ISA.ARM_NEON, ISA.SCALAR})

BINARY_OPS = (
    "+",
    "-",
    "*",
    "/",
    "%",
    "py_mod",
    "<",
    "<=",
    ">",
    ">=",
    "==",
    "!=",
    "&&",
    "||",
    "&",
    "|",
    "^",
    "min",
    "max",
    "**",
    "pow",
    "ipow",
    "atan2",
    "hypot",
    "fmod",
)
UNARY_OPS = (
    "neg",
    "not",
    "abs",
    "exp",
    "log",
    "sqrt",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "floor",
    "ceil",
    "tanh",
    "sign_numpy_2",
)
CAST_OPS = ("float64", "float32", "int32", "int64", "float16")
REDUCE_OPS = ("+", "*", "min", "max")

F64, F32, F16, I32, I64, BOOL, C128 = (
    dace.float64,
    dace.float32,
    dace.float16,
    dace.int32,
    dace.int64,
    dace.bool_,
    dace.complex128,
)
#: ``(operand dtype, output dtype)`` pairs every op is lowered with; the first at every width, all at the primary width.
DTYPES_PRIMARY = ((F64, F64), (I32, I32), (F64, BOOL))
DTYPES_EXTRA = ((F32, F32), (I64, I64), (F16, F16), (I32, F64), (F32, F64), (C128, C128))
WIDTHS_PRIMARY = (8,)
WIDTHS_EXTRA = ((1,), (4, 8), (2, 2, 4))

#: Operand kinds. ``ScalarTile`` is a ``Scalar``-kind operand bound to a tile-shaped array (a transient an upstream tile
#: op widened), ``SymbolLit`` a ``Symbol`` operand holding a literal instead of a free symbol.
BINARY_KINDS = (
    ("Tile", "Tile"),
    ("Tile", "Symbol"),
    ("Symbol", "Tile"),
    ("Tile", "SymbolLit"),
    ("Tile", "Scalar"),
    ("Scalar", "Tile"),
    ("Tile", "ScalarTile"),
    ("ScalarTile", "Tile"),
    ("Scalar", "Scalar"),
    ("Scalar", "Symbol"),
    ("Symbol", "Scalar"),
    ("ScalarTile", "ScalarTile"),
)
UNARY_KINDS = ("Tile", "Scalar", "ScalarTile", "Symbol", "SymbolLit")
FMA_KINDS = (
    *(kinds for kinds in itertools.product(("Tile", "Scalar", "Symbol"), repeat=3) if "Tile" in kinds),
    ("ScalarTile", "Tile", "Symbol"),
    ("Tile", "ScalarTile", "ScalarTile"),
)
ITE_KINDS = tuple(itertools.product(("Tile", "Scalar", "Symbol"), ("Tile", "Scalar", "Symbol"), ("Tile", "Symbol")))


def node_kind(kind: str) -> str:
    return {"ScalarTile": "Scalar", "SymbolLit": "Symbol"}.get(kind, kind)


def literal(dtype: dace.typeclass) -> str:
    return "2.5" if dtype in (F64, F32, F16, C128) else "2"


def tile_name(sdfg: dace.SDFG, dtype: dace.typeclass, widths: tuple[int, ...], hint: str) -> str:
    return sdfg.add_array(hint, widths, dtype, transient=True, storage=dace.StorageType.Register, find_new_name=True)[0]


def full_subset(widths: tuple[int, ...]) -> str:
    return ", ".join(f"0:{w}" for w in widths)


class Builder:
    """One SDFG with one tile node in its single state, wired operand by operand."""

    def __init__(self, label: str, widths: tuple[int, ...]):
        self.sdfg = dace.SDFG(label)
        self.state = self.sdfg.add_state()
        self.widths = tuple(widths)
        self.node = None

    def place(self, node):
        self.state.add_node(node)
        self.node = node
        return node

    def read_tile(self, conn: str, dtype: dace.typeclass, shape: tuple[int, ...] | None = None) -> None:
        shape = self.widths if shape is None else shape
        name = tile_name(self.sdfg, dtype, shape, f"in{conn}")
        self.state.add_edge(
            self.state.add_read(name), None, self.node, conn, dace.Memlet(f"{name}[{full_subset(shape)}]")
        )

    def read_scalar(self, conn: str, dtype: dace.typeclass) -> None:
        name = self.sdfg.add_scalar(f"in{conn}", dtype, transient=True, find_new_name=True)[0]
        self.state.add_edge(self.state.add_read(name), None, self.node, conn, dace.Memlet(f"{name}[0]"))

    def write_tile(self, conn: str, dtype: dace.typeclass, shape: tuple[int, ...] | None = None) -> None:
        shape = self.widths if shape is None else shape
        name = tile_name(self.sdfg, dtype, shape, f"out{conn}")
        self.state.add_edge(
            self.node, conn, self.state.add_write(name), None, dace.Memlet(f"{name}[{full_subset(shape)}]")
        )

    def write_element(self, conn: str, dtype: dace.typeclass) -> None:
        name = self.sdfg.add_scalar(f"out{conn}", dtype, transient=True, find_new_name=True)[0]
        self.state.add_edge(self.node, conn, self.state.add_write(name), None, dace.Memlet(f"{name}[0]"))

    def mask(self, conn: str = "_mask") -> None:
        self.read_tile(conn, BOOL)

    def operand(self, conn: str, kind: str, dtype: dace.typeclass) -> str | None:
        """Wire operand ``conn`` as ``kind``; returns the inline expression of a symbol operand."""
        if kind in ("Symbol", "SymbolLit"):
            if kind == "SymbolLit":
                return literal(dtype)
            symbol = f"s{conn}"
            if symbol not in self.sdfg.symbols:
                self.sdfg.add_symbol(symbol, dtype)
            return symbol
        if kind in ("Tile", "ScalarTile"):
            self.read_tile(conn, dtype)
        else:
            self.read_scalar(conn, dtype)
        return None


def lower_everywhere(builder: Builder) -> dict[str, object]:
    """Lowering of the builder's node for each ISA: implementation, tasklet and environments, or the error."""
    results = {}
    for isa in ISAS:
        builder.node.target_isa = isa
        entry: dict[str, object] = {}
        try:
            implementation = dispatch.select_tile_implementation(builder.node, builder.state)
            entry["implementation"] = implementation
            expansion = builder.node.implementations[implementation]
            tasklet = expansion.expansion(builder.node, builder.state, builder.sdfg)
            entry["tasklet"] = {
                "label": tasklet.label,
                "inputs": sorted(tasklet.in_connectors),
                "outputs": sorted(tasklet.out_connectors),
                "language": str(tasklet.language),
                "code": tasklet.code.as_string,
            }
            entry["environments"] = [
                (env.__name__, list(env.cmake_compile_flags), {key: list(value) for key, value in env.headers.items()})
                for env in expansion.environments
            ]
        except Exception as error:  # the error a configuration raises is part of its lowering
            entry["error"] = f"{type(error).__name__}: {str(error).splitlines()[0] if str(error) else ''}"
        results[isa.name] = entry
    return results


def widths_and_dtypes(ops: tuple[str, ...]) -> Iterator[tuple[str, tuple[int, ...], tuple]]:
    for op in ops:
        for dtypes in DTYPES_PRIMARY + DTYPES_EXTRA:
            yield op, WIDTHS_PRIMARY, dtypes
        for widths in WIDTHS_EXTRA:
            yield op, widths, DTYPES_PRIMARY[0]


def dtype_tag(dtypes: tuple) -> str:
    return f"{dtypes[0].to_string()}-{dtypes[1].to_string()}"


def binop_cases() -> Iterator[tuple[str, Builder]]:
    for op, widths, (dtype, out_dtype) in widths_and_dtypes(BINARY_OPS):
        for kind_a, kind_b in BINARY_KINDS:
            has_tile = "Tile" in (kind_a, kind_b)
            for out_variant in ("tile",) if has_tile else ("tile", "element"):
                for has_mask in (False, True):
                    name = (
                        f"{op}|{widths}|{dtype_tag((dtype, out_dtype))}|{kind_a}-{kind_b}|{out_variant}|mask={has_mask}"
                    )
                    builder = Builder("golden_binop", widths)
                    node = TileBinop(
                        "binop",
                        widths,
                        op=op,
                        has_mask=has_mask,
                        kind_a=node_kind(kind_a),
                        kind_b=node_kind(kind_b),
                        expr_a=builder.operand("_a", kind_a, dtype) if kind_a.startswith("Symbol") else "x",
                        expr_b=builder.operand("_b", kind_b, dtype) if kind_b.startswith("Symbol") else "x",
                    )
                    builder.place(node)
                    for conn, kind in (("_a", kind_a), ("_b", kind_b)):
                        if not kind.startswith("Symbol"):
                            builder.operand(conn, kind, dtype)
                    if has_mask:
                        builder.mask()
                    (builder.write_tile if out_variant == "tile" else builder.write_element)("_c", out_dtype)
                    yield name, builder


def unop_cases() -> Iterator[tuple[str, Builder]]:
    ops = UNARY_OPS + CAST_OPS
    for op, widths, (dtype, out_dtype) in widths_and_dtypes(ops):
        for kind in UNARY_KINDS:
            for out_variant in ("tile",) if kind == "Tile" else ("tile", "element"):
                for has_mask in (False, True):
                    name = f"{op}|{widths}|{dtype_tag((dtype, out_dtype))}|{kind}|{out_variant}|mask={has_mask}"
                    builder = Builder("golden_unop", widths)
                    expr = builder.operand("_a", kind, dtype) if kind.startswith("Symbol") else "x"
                    builder.place(
                        TileUnop("unop", widths, op=op, has_mask=has_mask, kind_a=node_kind(kind), expr_a=expr)
                    )
                    if not kind.startswith("Symbol"):
                        builder.operand("_a", kind, dtype)
                    if has_mask:
                        builder.mask()
                    (builder.write_tile if out_variant == "tile" else builder.write_element)("_c", out_dtype)
                    yield name, builder


def fma_cases() -> Iterator[tuple[str, Builder]]:
    for widths, (dtype, out_dtype) in itertools.chain(
        ((WIDTHS_PRIMARY, d) for d in DTYPES_PRIMARY + DTYPES_EXTRA), ((w, DTYPES_PRIMARY[0]) for w in WIDTHS_EXTRA)
    ):
        for kinds in FMA_KINDS:
            has_tile = "Tile" in kinds
            for out_variant in ("tile",) if has_tile else ("tile", "element"):
                for has_mask in (False, True):
                    name = f"{widths}|{dtype_tag((dtype, out_dtype))}|{'-'.join(kinds)}|{out_variant}|mask={has_mask}"
                    builder = Builder("golden_fma", widths)
                    exprs = [
                        builder.operand(conn, kind, dtype) if kind.startswith("Symbol") else "x"
                        for conn, kind in zip(("_a", "_b", "_c"), kinds, strict=True)
                    ]
                    builder.place(
                        TileFMA(
                            "fma",
                            widths,
                            has_mask=has_mask,
                            kind_a=node_kind(kinds[0]),
                            kind_b=node_kind(kinds[1]),
                            kind_c=node_kind(kinds[2]),
                            expr_a=exprs[0],
                            expr_b=exprs[1],
                            expr_c=exprs[2],
                        )
                    )
                    for conn, kind in zip(("_a", "_b", "_c"), kinds, strict=True):
                        if not kind.startswith("Symbol"):
                            builder.operand(conn, kind, dtype)
                    if has_mask:
                        builder.mask()
                    (builder.write_tile if out_variant == "tile" else builder.write_element)("_o", out_dtype)
                    yield name, builder


def ite_cases() -> Iterator[tuple[str, Builder]]:
    for widths, (dtype, out_dtype) in itertools.chain(
        ((WIDTHS_PRIMARY, d) for d in DTYPES_PRIMARY + DTYPES_EXTRA), ((w, DTYPES_PRIMARY[0]) for w in WIDTHS_EXTRA)
    ):
        for kind_t, kind_e, kind_mask in ITE_KINDS:
            has_tile = "Tile" in (kind_t, kind_e, kind_mask)
            for out_variant in ("tile",) if has_tile else ("tile", "element"):
                name = (
                    f"{widths}|{dtype_tag((dtype, out_dtype))}|then={kind_t}|else={kind_e}|cond={kind_mask}|"
                    f"{out_variant}"
                )
                builder = Builder("golden_ite", widths)
                expr_t = builder.operand("_t", kind_t, dtype) if kind_t.startswith("Symbol") else None
                expr_e = builder.operand("_e", kind_e, dtype) if kind_e.startswith("Symbol") else None
                expr_mask = "cond_symbol" if kind_mask == "Symbol" else None
                builder.place(
                    TileITE(
                        "ite",
                        widths,
                        kind_t=node_kind(kind_t),
                        kind_e=node_kind(kind_e),
                        expr_t=expr_t,
                        expr_e=expr_e,
                        kind_mask=kind_mask,
                        expr_mask=expr_mask,
                    )
                )
                if kind_t == "Tile":
                    builder.read_tile("_t", dtype)
                if kind_t == "Scalar":
                    builder.read_scalar("_t", dtype)
                if kind_e == "Tile":
                    builder.read_tile("_e", dtype)
                if kind_e == "Scalar":
                    builder.read_scalar("_e", dtype)
                if kind_mask == "Tile":
                    builder.mask()
                if kind_mask == "Scalar":
                    builder.read_scalar("_mask", BOOL)
                (builder.write_tile if out_variant == "tile" else builder.write_element)("_o", out_dtype)
                yield name, builder


def reduce_cases() -> Iterator[tuple[str, Builder]]:
    for op, widths, dtypes in widths_and_dtypes(REDUCE_OPS):
        dtype = dtypes[0]
        for axis in (None, *range(len(widths))):
            for has_mask in (False, True):
                for out_variant in ("element", "array") if axis is None else ("tile",):
                    name = f"{op}|{widths}|{dtype.to_string()}|axis={axis}|mask={has_mask}|{out_variant}"
                    builder = Builder("golden_reduce", widths)
                    builder.place(TileReduce("reduce", widths, op=op, axis=axis, has_mask=has_mask))
                    builder.read_tile("_src", dtype)
                    if has_mask:
                        builder.mask()
                    if axis is None and out_variant == "element":
                        builder.write_element("_dst", dtype)
                    elif axis is None:
                        builder.write_tile("_dst", dtype, (1,))
                    else:
                        kept = tuple(w for d, w in enumerate(widths) if d != axis) or (1,)
                        builder.write_tile("_dst", dtype, kept)
                    yield name, builder


def mask_gen_cases() -> Iterator[tuple[str, Builder]]:
    for widths in ((1,), (8,), (16,), (4, 8), (2, 2, 4)):
        for guard in (None, "flag > 0"):
            name = f"{widths}|guard={guard}"
            builder = Builder("golden_mask_gen", widths)
            names = tuple(f"i{d}" for d in range(len(widths)))
            bounds = tuple(f"n{d}" for d in range(len(widths)))
            builder.place(TileMaskGen("mask_gen", widths, names, bounds, guard_predicate=guard))
            builder.write_tile("_o", BOOL)
            yield name, builder


def iota_cases() -> Iterator[tuple[str, Builder]]:
    for widths, extra in itertools.product(((1,), (1, 1), (8,), (4, 8), (2, 2, 4)), ((), ("_idx",))):
        expr = "__l0 + " + ("_idx[0]" if extra else "1")
        name = f"{widths}|extra={extra}"
        builder = Builder("golden_iota", widths)
        builder.place(TileIota("iota", widths, expr, extra_inputs=extra))
        for conn in extra:
            builder.read_tile(conn, I64, (1,))
        builder.write_tile("_dst", I64)
        yield name, builder


def mma_cases() -> Iterator[tuple[str, Builder]]:
    for dtype, (alpha, beta) in itertools.product((F64, F32), ((1, 0), (1, 1), (2, 0), (1, 3), (2.5, 3))):
        name = f"{dtype.to_string()}|alpha={alpha}|beta={beta}"
        builder = Builder("golden_mma", (4, 4, 4))
        builder.place(TileMMA("mma", (4, 4, 4), alpha=alpha, beta=beta))
        builder.read_tile("_a", dtype, (4, 4))
        builder.read_tile("_b", dtype, (4, 4))
        if beta != 0:
            builder.read_tile("_cin", dtype, (4, 4))
        builder.write_tile("_c", dtype, (4, 4))
        yield name, builder


#: ``(label, source subset)`` of the window a tile load / store addresses. ``i`` is the induction variable of the
#: enclosing tile map, which the alignment proof reads the base offset from.
WINDOWS_1D = (("whole", "0:8"), ("shifted_odd", "3:11"), ("shifted_even", "4:12"), ("map_param", "i:i + 8"))


def with_tile_map(builder: Builder, in_map: bool):
    """Wrap the node in a tile-main map over ``i`` when ``in_map``; returns ``(entry, exit)`` or ``(None, None)``."""
    if not in_map:
        return None, None
    entry, exit_ = builder.state.add_map(f"golden{TILE_MAIN_MARKER}", {"i": "0:1016:8"})
    return entry, exit_


def strided_window(shape: tuple, widths: tuple, strides: tuple | None) -> str:
    """The window of an array of ``shape`` that a tile of ``widths`` copies, at ``strides`` along the tile dims."""
    steps = strides or (1,) * len(widths)
    leading = [None] * (len(shape) - len(widths))
    return ", ".join(
        "0" if step is None else f"0:{width * step}:{step}" if step != 1 else f"0:{width}"
        for step, width in zip((*leading, *steps), (*leading, *widths))
    )


def load_store_cases(is_load: bool, node_type: type) -> Iterator[tuple[str, Builder]]:
    """Structured, strided, transposed, replicated, broadcast and gathering tile loads or stores.

    ``MaskedCopyLibraryNode`` takes the structured and strided ones, with the stride as the step of its window, and
    the nodes of the other two types the rest.
    """
    copies = node_type is MaskedCopyLibraryNode
    source_conn, destination_conn = node_type.INPUT_CONNECTOR_NAME, node_type.OUTPUT_CONNECTOR_NAME
    dims_name = "src_dims" if is_load else "dst_dims"
    N = dace.symbol("N", dtype=dace.int64)
    shapes = {
        "1d": ((1024,), (1,)),
        "1d_symbolic": ((N,), (1,)),
        "2d": ((64, 64), (64, 1)),
        "2d_symbolic": ((N, N), (N, 1)),
    }

    def build(
        name: str,
        widths: tuple,
        dtype: dace.typeclass,
        array: str,
        window: str | None,
        in_map: bool,
        storage: dace.StorageType,
        mask: bool,
        **node_kwargs,
    ):
        shape, strides = shapes[array]
        builder = Builder(f"golden_{'load' if is_load else 'store'}", widths)
        if "i" in str(window) or in_map:
            builder.sdfg.add_symbol("N", dace.int64)
        node = builder.place(node_type(f"{'load' if is_load else 'store'}", widths, has_mask=mask, **node_kwargs))
        builder.sdfg.add_array("A", shape, dtype, storage=storage, strides=strides)
        subset = window if window is not None else ", ".join(f"0:{s}" for s in shape)
        entry, exit_ = with_tile_map(builder, in_map)
        access = builder.state.add_access("A")
        tile = tile_name(builder.sdfg, dtype, widths, "tile")
        tile_access = builder.state.add_access(tile)
        if is_load:
            if entry is None:
                builder.state.add_edge(access, None, node, source_conn, dace.Memlet(f"A[{subset}]"))
            else:
                builder.state.add_memlet_path(
                    access, entry, node, dst_conn=source_conn, memlet=dace.Memlet(f"A[{subset}]")
                )
            builder.state.add_edge(
                node, destination_conn, tile_access, None, dace.Memlet(f"{tile}[{full_subset(widths)}]")
            )
            if entry is not None:
                builder.state.add_edge(entry, None, node, None, dace.Memlet())
        else:
            builder.state.add_edge(tile_access, None, node, source_conn, dace.Memlet(f"{tile}[{full_subset(widths)}]"))
            if exit_ is None:
                builder.state.add_edge(node, destination_conn, access, None, dace.Memlet(f"A[{subset}]"))
            else:
                builder.state.add_edge(entry, None, tile_access, None, dace.Memlet())
                builder.state.add_memlet_path(
                    node, exit_, access, src_conn=destination_conn, memlet=dace.Memlet(f"A[{subset}]")
                )
        if mask:
            builder.mask()
        return name, builder

    # Plain contiguous / strided tiles of every dtype, with and without a mask, on 1-D and 2-D arrays.
    for dtype in (F64, F32, I32, F16, I64):
        for array, widths in (
            ("1d", (8,)),
            ("1d", (1,)),
            ("1d", (16,)),
            ("2d", (8,)),
            ("2d", (4, 8)),
            ("1d_symbolic", (8,)),
            ("2d_symbolic", (8,)),
            ("2d_symbolic", (4, 8)),
        ):
            for mask in (False, True):
                for strides in (None, (2,), (4, 2)):
                    if strides is not None and len(strides) != len(widths):
                        continue
                    kwargs = {} if strides is None or copies else {"dim_strides": strides}
                    window = strided_window(shapes[array][0], widths, strides) if copies else None
                    name = f"{array}|{widths}|{dtype.to_string()}|mask={mask}|dim_strides={strides}"
                    yield build(name, widths, dtype, array, window, False, dace.StorageType.CPU_Heap, mask, **kwargs)
    # A transposed tile.
    for dtype in () if copies else (F64, F16):
        for mask in (False, True):
            name = f"transposed|{dtype.to_string()}|mask={mask}"
            yield build(name, (4, 8), dtype, "2d", None, False, dace.StorageType.CPU_Heap, mask, **{dims_name: (1, 0)})
    # Alignment proof inputs: a half-precision tile on device memory at several base offsets, with and without the
    # tile map the proof reads the divisibility guarantee from.
    for label, window in WINDOWS_1D:
        for in_map in (False, True):
            for storage in (dace.StorageType.GPU_Global, dace.StorageType.CPU_Heap):
                for dtype in (F16, F32):
                    for mask in (False, True):
                        name = f"alignment|{label}|map={in_map}|{storage.name}|{dtype.to_string()}|mask={mask}"
                        yield build(name, (8,), dtype, "1d", window, in_map or "i" in window, storage, mask)
    # A half-precision window on a symbolic row stride: only the extents of the tiled maps and the stride-divisibility
    # guards the vectorizer emits make its parity decidable.
    for guard, (cols, row) in itertools.product(
        (None, 2, 8),
        (
            ("0:N:8", "i, j:j + 8"),
            ("0:M:8", "i, j:j + 8"),
            ("0:M:8", "i, j + 1:j + 9"),
            ("1:N - 1:8", "i, j:j + 8"),
            ("1:N - 1:8", "i, j + 1:j + 9"),
        ),
    ):
        for storage in (dace.StorageType.GPU_Global, dace.StorageType.CPU_Heap):
            name = f"symbolic_stride|guard={guard}|cols={cols}|{row}|{storage.name}"
            builder = Builder(f"golden_{'load' if is_load else 'store'}_symbolic_stride", (8,))
            builder.sdfg.add_symbol("N", dace.int64)
            builder.sdfg.add_symbol("M", dace.int64)
            builder.sdfg.add_array("A", (N, N), F16, storage=storage, strides=(N, 1))
            if guard is not None:
                guard_state = builder.sdfg.add_state(TILE_GUARD_STATE_LABEL)
                guard_state.add_tasklet(f"{STRIDE_GUARD_PREFIX}N_{guard}", set(), set(), "")
            node = builder.place(node_type("symbolic", (8,), has_mask=False) if copies else node_type("symbolic", (8,)))
            outer, outer_exit = builder.state.add_map("rows", {"i": "0:N"})
            inner, inner_exit = builder.state.add_map(f"cols{TILE_MAIN_MARKER}", {"j": cols})
            access = builder.state.add_access("A")
            tile = tile_name(builder.sdfg, F16, (8,), "tile")
            if is_load:
                builder.state.add_memlet_path(
                    access, outer, inner, node, dst_conn=source_conn, memlet=dace.Memlet(f"A[{row}]")
                )
                builder.state.add_edge(
                    node, destination_conn, builder.state.add_access(tile), None, dace.Memlet(f"{tile}[0:8]")
                )
            else:
                tile_access = builder.state.add_access(tile)
                builder.state.add_edge(outer, None, inner, None, dace.Memlet())
                builder.state.add_edge(inner, None, tile_access, None, dace.Memlet())
                builder.state.add_edge(tile_access, None, node, source_conn, dace.Memlet(f"{tile}[0:8]"))
                builder.state.add_memlet_path(
                    node, inner_exit, outer_exit, access, src_conn=destination_conn, memlet=dace.Memlet(f"A[{row}]")
                )
            yield name, builder
    if copies:
        return
    # Gathers and scatters addressed through index tiles.
    for dtype in (F64, I32):
        for idx_dtype in (I32, I64, dace.uint32):
            for array, widths, gather in (
                ("1d", (8,), (0,)),
                ("2d", (8,), (0,)),
                ("2d", (8,), (1,)),
                ("2d", (8,), (0, 1)),
                ("2d", (4, 8), (0,)),
            ):
                for mask in (False, True):
                    name = (
                        f"gather|{array}|{widths}|{gather}|{dtype.to_string()}|idx={idx_dtype.to_string()}|mask={mask}"
                    )
                    shape, strides = shapes[array]
                    builder = Builder(f"golden_{'load' if is_load else 'store'}_gather", widths)
                    node = builder.place(node_type("gather", widths, has_mask=mask, gather_dims=gather))
                    builder.sdfg.add_array("A", shape, dtype, strides=strides)
                    subset = ", ".join(f"0:{s}" for s in shape)
                    tile = tile_name(builder.sdfg, dtype, widths, "tile")
                    if is_load:
                        builder.state.add_edge(
                            builder.state.add_access("A"), None, node, "_src", dace.Memlet(f"A[{subset}]")
                        )
                        builder.state.add_edge(
                            node,
                            "_dst",
                            builder.state.add_access(tile),
                            None,
                            dace.Memlet(f"{tile}[{full_subset(widths)}]"),
                        )
                    else:
                        builder.state.add_edge(
                            builder.state.add_access(tile),
                            None,
                            node,
                            "_src",
                            dace.Memlet(f"{tile}[{full_subset(widths)}]"),
                        )
                        builder.state.add_edge(
                            node, "_dst", builder.state.add_access("A"), None, dace.Memlet(f"A[{subset}]")
                        )
                    for d in gather:
                        if len(widths) > 1 and "ONE" not in builder.sdfg.constants_prop:
                            builder.sdfg.add_constant("ONE", 1, dace.data.Scalar(dace.int32))
                        idx = tile_name(
                            builder.sdfg, idx_dtype, (widths[0],) + (dace.symbolic.ONE,) * (len(widths) - 1), f"idx{d}"
                        )
                        shape_idx = tuple(builder.sdfg.arrays[idx].shape)
                        builder.state.add_edge(
                            builder.state.add_read(idx),
                            None,
                            node,
                            f"_idx_{d}",
                            dace.Memlet(f"{idx}[{full_subset(shape_idx)}]"),
                        )
                    if mask:
                        builder.mask()
                    yield name, builder
    # Broadcast sources.
    for kind in ("Scalar", "Symbol"):
        for dtype in (F64, I32, F16):
            for mask in (False, True):
                for widths in ((8,), (4, 8), (1,)):
                    name = f"broadcast|{kind}|{dtype.to_string()}|{widths}|mask={mask}"
                    builder = Builder(f"golden_{'load' if is_load else 'store'}_broadcast", widths)
                    kwargs = {"src_kind": kind, "src_expr": "alpha" if kind == "Symbol" else None}
                    if kind == "Symbol":
                        builder.sdfg.add_symbol("alpha", dtype)
                    node = builder.place(node_type("broadcast", widths, has_mask=mask, **kwargs))
                    if is_load:
                        tile = tile_name(builder.sdfg, dtype, widths, "tile")
                        if kind == "Scalar":
                            builder.read_scalar("_src", dtype)
                        builder.state.add_edge(
                            node,
                            "_dst",
                            builder.state.add_access(tile),
                            None,
                            dace.Memlet(f"{tile}[{full_subset(widths)}]"),
                        )
                    else:
                        builder.sdfg.add_array("A", (1024,), dtype)
                        if kind == "Scalar":
                            builder.read_scalar("_src", dtype)
                        builder.state.add_edge(
                            node,
                            "_dst",
                            builder.state.add_access("A"),
                            None,
                            dace.Memlet(f"A[0:{widths[-1]}]" if len(widths) == 1 else "A[0:1024]"),
                        )
                    if mask:
                        builder.mask()
                    yield name, builder
    # A replicated tile (lanes sharing a source element), with the ``int_floor`` base the phase-aware offset reads.
    if is_load:
        for factor in (2, 3, 4):
            for width in (8, 12):
                name = f"replicate|{factor}|{width}"
                builder = Builder("golden_load_replicate", (width,))
                builder.sdfg.add_symbol("N", dace.int64)
                node = builder.place(TileGather("replicate", (width,), replicate_factor_per_dim=(factor,)))
                builder.sdfg.add_array("A", (1024,), F64)
                entry = builder.state.add_map(f"golden{TILE_MAIN_MARKER}", {"i": f"0:N:{width}"})[0]
                tile = tile_name(builder.sdfg, F64, (width,), "tile")
                begin = dace.symbolic.pystr_to_symbolic(f"int_floor(i, {factor})")
                end = dace.symbolic.pystr_to_symbolic(f"int_floor(i, {factor}) + {-(-width // factor)}")
                builder.state.add_memlet_path(
                    builder.state.add_access("A"),
                    entry,
                    node,
                    dst_conn="_src",
                    memlet=dace.Memlet(subset=dace.subsets.Range([(begin, end - 1, 1)]), data="A"),
                )
                builder.state.add_edge(entry, None, node, None, dace.Memlet())
                builder.state.add_edge(
                    node, "_dst", builder.state.add_access(tile), None, dace.Memlet(f"{tile}[{full_subset((width,))}]")
                )
                yield name, builder


def masked_copy_cases() -> Iterator[tuple[str, Builder]]:
    for direction, is_load in (("load", True), ("store", False)):
        for name, builder in load_store_cases(is_load, MaskedCopyLibraryNode):
            yield f"{direction}|{name}", builder


CASES: dict[str, Callable[[], Iterator[tuple[str, Builder]]]] = {
    "TileBinop": binop_cases,
    "TileUnop": unop_cases,
    "TileFMA": fma_cases,
    "TileITE": ite_cases,
    "TileReduce": reduce_cases,
    "TileMaskGen": mask_gen_cases,
    "TileIota": iota_cases,
    "TileMMA": mma_cases,
    "TileGather": lambda: load_store_cases(True, TileGather),
    "TileScatter": lambda: load_store_cases(False, TileScatter),
    "MaskedCopyLibraryNode": masked_copy_cases,
}

#: Node types split into this many shards, one digest each, so no single test runs long enough to hit the CI
#: timeout under coverage (TileBinop alone lowers 9600 configurations). Shard ``i`` takes every ``n``-th case from ``i``.
SHARDS = {"TileBinop": 16, "TileUnop": 8}


class ShardKey(NamedTuple):
    """One digest entry: a node type, or one shard of it."""

    node_type: str
    index: int
    count: int

    def __str__(self) -> str:
        return self.node_type if self.count == 1 else f"{self.node_type}[{self.index}/{self.count}]"


def shard_keys() -> list[ShardKey]:
    """Every digest entry, in digest-file order."""
    return [
        ShardKey(node_type, index, SHARDS.get(node_type, 1))
        for node_type in sorted(CASES)
        for index in range(SHARDS.get(node_type, 1))
    ]


def lowerings(key: ShardKey) -> dict[str, dict[str, object]]:
    """Every recorded lowering of one digest entry, keyed by configuration."""
    results = {}
    with (
        mock.patch.object(dispatch, "host_supported_isas", lambda: HOST_ISAS),
        mock.patch.object(dispatch, "detect_host_isa", lambda: ISA.AVX512),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore")
        for name, builder in itertools.islice(CASES[key.node_type](), key.index, None, key.count):
            results[name] = lower_everywhere(builder)
    return results


def digest(results: dict[str, dict[str, object]]) -> dict[str, object]:
    payload = json.dumps(results, sort_keys=True, default=str).encode()
    return {"configurations": len(results), "sha256": hashlib.sha256(payload).hexdigest()}


def digests() -> dict[str, dict[str, object]]:
    return {str(key): digest(lowerings(key)) for key in shard_keys()}


def dump(path: str) -> None:
    with open(path, "w") as out:
        json.dump({str(key): lowerings(key) for key in shard_keys()}, out, sort_keys=True, indent=1, default=str)


def update() -> None:
    with open(DIGEST_FILE, "w") as out:
        json.dump(digests(), out, sort_keys=True, indent=1)
        out.write("\n")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "dump":
        dump(sys.argv[2])
    elif len(sys.argv) == 2 and sys.argv[1] == "update":
        update()
    else:
        sys.exit("usage: golden_lowering.py dump <file> | update")
