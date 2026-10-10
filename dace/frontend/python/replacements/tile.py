# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``dace.tile``: the tile library nodes as calls of a ``@dace.program``.

A *window* is an array or a slice of one, ``A[i:i + 8]``, whose extents other than 1 are the lanes of the tile it
makes. A *tile* is a register array, ``dace.define_local([8], dace.float64, storage=dace.StorageType.Register)``. A
call takes tiles or windows (a window is copied into a new tile first) and its result is a new tile.

Every call makes a node of the ``BLOCK`` group: on a CPU one core runs the tile, as a loop over its lanes, and in a GPU
kernel the threads of the block share it (:class:`~dace.libraries.tileops.dispatch.TileGroup`).
"""

from dace import data, dtypes, subsets, symbolic
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python.common import DaceSyntaxError
from dace.frontend.python.replacements.utils import ProgramVisitor
from dace.libraries.tileops import (
    MaskedCopyLibraryNode,
    TileBinop,
    TileFMA,
    TileIota,
    TileITE,
    TileMMA,
    TileReduce,
    TileUnop,
)
from dace.libraries.tileops.dispatch import TileGroup
from dace.libraries.tileops.nodes.masked_copy import MASK_CONNECTOR_NAME
from dace.libraries.tileops.nodes.tile_op import TileOp
from dace.memlet import Memlet
from dace.sdfg import SDFG, SDFGState
from dace.sdfg.nodes import AccessNode

__all__ = ()  # the replacements register themselves

Window = tuple[str, subsets.Range]


def is_tile(sdfg: SDFG, window: Window) -> bool:
    """Whether ``window`` is a whole register array that owns its elements, which is a tile."""
    name, subset = window
    desc = sdfg.arrays[name]
    return (
        desc.transient
        and not isinstance(desc, data.View)
        and desc.storage == dtypes.StorageType.Register
        and subset == subsets.Range.from_array(desc)
    )


def as_window(pv: ProgramVisitor, sdfg: SDFG, operand: str | Window, what: str) -> Window:
    """An operand as a window: the result of an earlier call is the name of an array, which the window covers."""
    if isinstance(operand, str) and isinstance(sdfg.arrays.get(operand), data.Array):
        return operand, subsets.Range.from_array(sdfg.arrays[operand])
    if not isinstance(operand, tuple):
        raise DaceSyntaxError(pv, None, f"{what}: expects arrays or slices of arrays, got '{operand}'")
    return operand


def lanes_of(window: Window) -> tuple[int, ...]:
    """The extents of the window other than 1, which are the lanes of the tile it makes."""
    extents = tuple(extent for extent in window[1].size() if extent != 1)
    if not 1 <= len(extents) <= 3 or not all(symbolic.issymbolic(extent) is False for extent in extents):
        raise ValueError(
            f"a tile has one to three dims of a constant extent, got the extents {extents} "
            "(a window of extent 1 in every dim is a scalar)"
        )
    return tuple(int(extent) for extent in extents)


def dtype_of(sdfg: SDFG, window: Window) -> dtypes.typeclass:
    return sdfg.arrays[window[0]].dtype


def full_memlet(sdfg: SDFG, name: str) -> Memlet:
    return Memlet.from_array(name, sdfg.arrays[name])


def new_tile(sdfg: SDFG, lanes: tuple[int, ...], dtype: dtypes.typeclass, hint: str) -> str:
    return sdfg.add_array(hint, lanes, dtype, transient=True, storage=dtypes.StorageType.Register, find_new_name=True)[
        0
    ]


def add_block_node(state: SDFGState, node: TileOp) -> TileOp:
    """Add ``node`` to ``state`` as a node of the ``BLOCK`` group."""
    node.group = TileGroup.BLOCK
    state.add_node(node)
    return node


def copy_window(
    sdfg: SDFG,
    state: SDFGState,
    source: AccessNode,
    destination: AccessNode,
    source_memlet: Memlet,
    destination_memlet: Memlet,
    lanes: tuple[int, ...],
    mask: AccessNode | None,
) -> None:
    """One masked copy node from ``source`` to ``destination``, one of which is a window and the other a tile."""
    node = add_block_node(
        state, MaskedCopyLibraryNode(f"copy_{destination.data}", widths=lanes, has_mask=mask is not None)
    )
    state.add_edge(source, None, node, node.INPUT_CONNECTOR_NAME, source_memlet)
    state.add_edge(node, node.OUTPUT_CONNECTOR_NAME, destination, None, destination_memlet)
    if mask is not None:
        state.add_edge(mask, None, node, MASK_CONNECTOR_NAME, full_memlet(sdfg, mask.data))


def common_lanes(pv: ProgramVisitor, what: str, windows: tuple[Window, ...]) -> tuple[int, ...]:
    """The lanes the windows share, which they must have all in common."""
    try:
        lanes = {lanes_of(window) for window in windows}
    except ValueError as error:
        raise DaceSyntaxError(pv, None, f"{what}: {error}") from error
    if len(lanes) != 1:
        raise DaceSyntaxError(pv, None, f"{what}: the arguments have different lanes, {sorted(lanes)}")
    return lanes.pop()


def as_tile(sdfg: SDFG, state: SDFGState, window: Window, mask: AccessNode | None) -> AccessNode:
    """The access node of the tile ``window`` is; a window that is no tile is copied into a new one."""
    lanes = lanes_of(window)
    name, subset = window
    if is_tile(sdfg, window):
        return state.add_read(name)
    tile = state.add_write(new_tile(sdfg, lanes, dtype_of(sdfg, window), f"{name}_tile"))
    copy_window(
        sdfg,
        state,
        state.add_read(name),
        tile,
        Memlet(data=name, subset=subset),
        full_memlet(sdfg, tile.data),
        lanes,
        mask,
    )
    return tile


def operand_tiles(
    pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, what: str, operands: tuple[str | Window, ...]
) -> tuple[list[AccessNode], tuple[int, ...], dtypes.typeclass]:
    """The tiles of ``operands``, which must share their lanes and type, with those lanes and that type."""
    windows = tuple(as_window(pv, sdfg, operand, what) for operand in operands)
    lanes = common_lanes(pv, what, windows)
    types = {dtype_of(sdfg, window) for window in windows}
    if len(types) != 1:
        raise DaceSyntaxError(pv, None, f"{what}: the operands have different types, {sorted(map(str, types))}")
    return [as_tile(sdfg, state, window, None) for window in windows], lanes, types.pop()


def elementwise(
    pv: ProgramVisitor,
    sdfg: SDFG,
    state: SDFGState,
    what: str,
    node: TileOp,
    connectors: tuple[str, ...],
    output: str,
    operands: tuple[str | Window, ...],
) -> str:
    """Wire ``node`` from the tiles of ``operands`` (to ``connectors``) to a new tile, whose name it returns."""
    tiles, lanes, dtype = operand_tiles(pv, sdfg, state, what, operands)
    node.widths = list(lanes)
    add_block_node(state, node)
    for tile, connector in zip(tiles, connectors, strict=True):
        state.add_edge(tile, None, node, connector, full_memlet(sdfg, tile.data))
    result = state.add_write(new_tile(sdfg, lanes, dtype, what.rsplit(".", 1)[-1]))
    state.add_edge(node, output, result, None, full_memlet(sdfg, result.data))
    return result.data


def register_binop(name: str, op: str) -> None:
    what = f"dace.tile.{name}"

    def binop(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: str | Window, b: str | Window) -> str:
        return elementwise(pv, sdfg, state, what, TileBinop(name, widths=(1,), op=op), ("_a", "_b"), "_c", (a, b))

    binop.__doc__ = f"``{op}`` of two tiles or windows of the same lanes: a new tile."
    oprepo.replaces_windows(what)(binop)


def register_unop(name: str, op: str) -> None:
    what = f"dace.tile.{name}"

    def unop(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: str | Window) -> str:
        return elementwise(pv, sdfg, state, what, TileUnop(name, widths=(1,), op=op), ("_a",), "_c", (a,))

    unop.__doc__ = f"``{op}`` of every lane of a tile or window: a new tile."
    oprepo.replaces_windows(what)(unop)


for _name, _op in (("add", "+"), ("sub", "-"), ("mul", "*"), ("div", "/"), ("minimum", "min"), ("maximum", "max")):
    register_binop(_name, _op)
for _name in ("abs", "exp", "log", "sqrt", "neg", "tanh"):
    register_unop(_name, _name)


@oprepo.replaces_windows("dace.tile.fma")
def fma(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: str | Window, b: str | Window, c: str | Window) -> str:
    """``a * b + c`` over tiles or windows of the same lanes: a new tile."""
    node = TileFMA("fma", widths=(1,))
    return elementwise(pv, sdfg, state, "dace.tile.fma", node, ("_a", "_b", "_c"), "_o", (a, b, c))


@oprepo.replaces_windows("dace.tile.where")
def where(
    pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, mask: str | Window, then: str | Window, otherwise: str | Window
) -> str:
    """``then`` where ``mask`` is set and ``otherwise`` elsewhere: a new tile."""
    what = "dace.tile.where"
    mask_window = as_window(pv, sdfg, mask, what)
    if dtype_of(sdfg, mask_window) != dtypes.bool_:
        raise DaceSyntaxError(pv, None, f"{what}: the mask must be of type bool, got {dtype_of(sdfg, mask_window)}")
    lanes = common_lanes(pv, what, (mask_window, as_window(pv, sdfg, then, what)))
    mask_tile = as_tile(sdfg, state, mask_window, None)
    node = TileITE("where", widths=lanes)
    result = elementwise(pv, sdfg, state, what, node, ("_t", "_e"), "_o", (then, otherwise))
    state.add_edge(mask_tile, None, node, "_mask", full_memlet(sdfg, mask_tile.data))
    return result


@oprepo.replaces_windows("dace.tile.sum")
def tile_sum(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: str | Window) -> str:
    """The sum of every lane of a tile or window: a new one-lane tile."""
    what = "dace.tile.sum"
    tiles, lanes, dtype = operand_tiles(pv, sdfg, state, what, (a,))
    node = add_block_node(state, TileReduce("sum", widths=lanes, op="+"))
    state.add_edge(tiles[0], None, node, "_src", full_memlet(sdfg, tiles[0].data))
    result = state.add_write(new_tile(sdfg, (1,), dtype, "sum"))
    state.add_edge(node, "_dst", result, None, full_memlet(sdfg, result.data))
    return result.data


@oprepo.replaces_windows("dace.tile.load")
def load(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, source: str | Window) -> str:
    """A new tile holding the lanes of the window ``source``."""
    window = as_window(pv, sdfg, source, "dace.tile.load")
    try:
        lanes_of(window)
    except ValueError as error:
        raise DaceSyntaxError(pv, None, f"dace.tile.load: {error}") from error
    if is_tile(sdfg, window):
        raise DaceSyntaxError(pv, None, "dace.tile.load: the source is a tile already")
    return as_tile(sdfg, state, window, None).data


@oprepo.replaces_windows("dace.tile.store", outputs=(0,))
def store(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, destination: str | Window, source: str | Window) -> None:
    """Copy the tile or window ``source`` into the window or tile ``destination`` of the same lanes."""
    what = "dace.tile.store"
    destination, source = as_window(pv, sdfg, destination, what), as_window(pv, sdfg, source, what)
    lanes = common_lanes(pv, what, (destination, source))
    if dtype_of(sdfg, destination) != dtype_of(sdfg, source):
        raise DaceSyntaxError(pv, None, f"{what}: the destination and the source have different types")
    (source_name, source_subset), (destination_name, destination_subset) = source, destination
    copy_window(
        sdfg,
        state,
        state.add_read(source_name),
        state.add_write(destination_name),
        Memlet(data=source_name, subset=source_subset),
        Memlet(data=destination_name, subset=destination_subset),
        lanes,
        None,
    )


@oprepo.replaces_windows("dace.tile.fill", outputs=(0,))
def fill(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, destination: str | Window, value: int | float) -> None:
    """Set every lane of the tile ``destination`` to the constant ``value``."""
    what = "dace.tile.fill"
    window = as_window(pv, sdfg, destination, what)
    if not is_tile(sdfg, window):
        raise DaceSyntaxError(pv, None, f"{what}: the destination must be a tile")
    if not isinstance(value, (int, float)):
        raise DaceSyntaxError(pv, None, f"{what}: the value must be a constant number, got '{value}'")
    lanes = lanes_of(window)
    ctype = dtype_of(sdfg, window).ctype
    node = add_block_node(state, TileIota(f"fill_{window[0]}", widths=lanes, expr=f"{ctype}({value!r})"))
    state.add_edge(node, "_dst", state.add_write(window[0]), None, full_memlet(sdfg, window[0]))


@oprepo.replaces_windows("dace.tile.mma", outputs=(2,))
def mma(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: str | Window, b: str | Window, c: str | Window) -> None:
    """Accumulate the matrix product of the ``(M, K)`` tile ``a`` and the ``(K, N)`` tile ``b`` into the ``(M, N)``
    tile ``c``: ``c += a @ b``. Windows ``a`` and ``b`` are copied into tiles first."""
    what = "dace.tile.mma"
    a, b, c = (as_window(pv, sdfg, operand, what) for operand in (a, b, c))
    if not is_tile(sdfg, c):
        raise DaceSyntaxError(pv, None, f"{what}: the accumulator must be a tile")
    try:
        (m, k), (k_b, n), (m_c, n_c) = lanes_of(a), lanes_of(b), lanes_of(c)
    except ValueError as error:
        raise DaceSyntaxError(pv, None, f"{what}: the operands must be two-dimensional tiles, {error}") from error
    if (k, m, n) != (k_b, m_c, n_c):
        raise DaceSyntaxError(
            pv, None, f"{what}: shapes ({m}, {k}) @ ({k_b}, {n}) do not accumulate into ({m_c}, {n_c})"
        )
    if len({dtype_of(sdfg, window) for window in (a, b, c)}) != 1:
        raise DaceSyntaxError(pv, None, f"{what}: the operands have different types")
    left, right = as_tile(sdfg, state, a, None), as_tile(sdfg, state, b, None)
    node = add_block_node(state, TileMMA("mma", widths=(m, k, n), alpha=1, beta=1))
    state.add_edge(left, None, node, "_a", full_memlet(sdfg, left.data))
    state.add_edge(right, None, node, "_b", full_memlet(sdfg, right.data))
    state.add_edge(state.add_read(c[0]), None, node, "_cin", full_memlet(sdfg, c[0]))
    state.add_edge(node, "_c", state.add_write(c[0]), None, full_memlet(sdfg, c[0]))


@oprepo.replaces_windows("dace.tile.masked_copy", outputs=(0,))
def masked_copy(
    pv: ProgramVisitor,
    sdfg: SDFG,
    state: SDFGState,
    destination: str | Window,
    source: str | Window,
    mask: str | Window,
) -> None:
    """Copy ``source`` to ``destination`` where ``mask`` is set.

    A window destination keeps its other lanes, and a tile destination has them zeroed. A window copied to a window
    goes through a tile.
    """
    what = "dace.tile.masked_copy"
    destination, source, mask = (as_window(pv, sdfg, operand, what) for operand in (destination, source, mask))
    lanes = common_lanes(pv, what, (destination, source, mask))
    if dtype_of(sdfg, destination) != dtype_of(sdfg, source):
        raise DaceSyntaxError(pv, None, f"{what}: the destination and the source have different types")
    if dtype_of(sdfg, mask) != dtypes.bool_:
        raise DaceSyntaxError(pv, None, f"{what}: the mask must be of type bool, got {dtype_of(sdfg, mask)}")
    if is_tile(sdfg, destination) and is_tile(sdfg, source):
        raise DaceSyntaxError(
            pv, None, f"{what}: copy between two tiles with an assignment, only a window is copied under a mask"
        )
    lane_mask = as_tile(sdfg, state, mask, None)
    destination_name, destination_subset = destination
    if is_tile(sdfg, destination):
        source_name, source_subset = source
        copy_window(
            sdfg,
            state,
            state.add_read(source_name),
            state.add_write(destination_name),
            Memlet(data=source_name, subset=source_subset),
            full_memlet(sdfg, destination_name),
            lanes,
            lane_mask,
        )
        return
    tile = as_tile(sdfg, state, source, lane_mask)
    copy_window(
        sdfg,
        state,
        tile,
        state.add_write(destination_name),
        full_memlet(sdfg, tile.data),
        Memlet(data=destination_name, subset=destination_subset),
        lanes,
        lane_mask,
    )
