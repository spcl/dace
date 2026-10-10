# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``dace.tile.add`` and ``dace.tile.masked_copy``: the tile library nodes as calls of a ``@dace.program``.

A *window* is an array or a slice of one, ``A[i:i + 8]``, whose extents other than 1 are the lanes of the tile it
makes. A *tile* is a register array, ``dace.define_local([8], dace.float64, storage=dace.StorageType.Register)``. The
calls lower to :class:`~dace.libraries.tileops.nodes.masked_copy.MaskedCopyLibraryNode` and
:class:`~dace.libraries.tileops.nodes.tile_binop.TileBinop`, which expand to a loop over the lanes.
"""

from dace import data, dtypes, subsets, symbolic
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python.common import DaceSyntaxError
from dace.frontend.python.replacements.utils import ProgramVisitor
from dace.libraries.tileops import MaskedCopyLibraryNode, TileBinop
from dace.libraries.tileops.nodes.masked_copy import MASK_CONNECTOR_NAME
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
    node = MaskedCopyLibraryNode(f"copy_{destination.data}", widths=lanes, has_mask=mask is not None)
    state.add_node(node)
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


@oprepo.replaces_windows("dace.tile.add")
def add(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, a: str | Window, b: str | Window) -> str:
    """``a + b`` over two tiles or windows of the same lanes: a new tile."""
    what = "dace.tile.add"
    a, b = as_window(pv, sdfg, a, what), as_window(pv, sdfg, b, what)
    lanes = common_lanes(pv, what, (a, b))
    dtype = dtype_of(sdfg, a)
    if dtype != dtype_of(sdfg, b):
        raise DaceSyntaxError(pv, None, f"{what}: the operands have different types, {dtype} and {dtype_of(sdfg, b)}")
    left, right = as_tile(sdfg, state, a, None), as_tile(sdfg, state, b, None)
    result = state.add_write(new_tile(sdfg, lanes, dtype, "sum"))
    node = TileBinop("add", widths=lanes, op="+")
    state.add_node(node)
    state.add_edge(left, None, node, "_a", full_memlet(sdfg, left.data))
    state.add_edge(right, None, node, "_b", full_memlet(sdfg, right.data))
    state.add_edge(node, "_c", result, None, full_memlet(sdfg, result.data))
    return result.data


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
