# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Runs a kernel map inside a persistent megakernel: every thread walks the flattened iteration space in steps of the
grid size, and dimensions too short to spread over the grid stay a loop in every thread."""

from dace import SDFG, SDFGState, dtypes, subsets, symbolic
from dace.sdfg import nodes
from dace.sdfg.replace import replace_dict
from dace.transformation import helpers as xfh

#: A literal extent below this stays a loop in every thread
SMALL_EXTENT = 64
#: The flattened index of a grid-stride loop
FLAT_INDEX = "__mk_g"


def is_small(extent: symbolic.SymbolicType, small_symbols: set[str]) -> bool:
    """Whether a map dimension of ``extent`` iterations is too short to spread over the grid: a literal below
    :data:`SMALL_EXTENT`, or an expression of ``small_symbols`` only."""
    extent = symbolic.pystr_to_symbolic(extent)
    if extent.is_Number:
        return int(extent) < SMALL_EXTENT
    return bool(small_symbols) and {str(s) for s in extent.free_symbols} <= small_symbols


def flatten_to_grid_stride(state: SDFGState, entry: nodes.MapEntry, tid: str, nthreads: str) -> None:
    """Turns ``entry`` into one sequential loop over its flattened iteration space, from thread ``tid`` in steps of
    ``nthreads``; the body reads every original index back from the flat one (the outermost one needs no modulo)."""
    flat = symbolic.symbol(FLAT_INDEX, dtypes.int64)
    sizes = entry.map.range.size()
    repl, stride = {}, 1
    for dim in reversed(range(len(sizes))):
        begin, _, step = entry.map.range[dim]
        index = flat if stride == 1 else symbolic.int_floor(flat, stride)
        if dim > 0:
            index = index % sizes[dim]
        repl[entry.map.params[dim]] = begin + step * index
        stride *= sizes[dim]
    symrepl = {symbolic.pystr_to_symbolic(name): value for name, value in repl.items()}
    replace_dict(state.scope_subgraph(entry), {name: str(value) for name, value in repl.items()}, symrepl)
    entry.map.params = [FLAT_INDEX]
    entry.map.range = subsets.Range(
        [(symbolic.pystr_to_symbolic(tid), stride - 1, symbolic.pystr_to_symbolic(nthreads))]
    )
    entry.map.schedule = dtypes.ScheduleType.Sequential


def grid_stride(
    sdfg: SDFG, state: SDFGState, entry: nodes.MapEntry, tid: str, nthreads: str, small_symbols: set[str]
) -> nodes.MapEntry:
    """Lowers the kernel map ``entry`` to a grid-stride loop of thread ``tid`` among ``nthreads``. Its small
    dimensions (:func:`is_small`) move into an inner sequential map; with only small dimensions, all are spread.

    :return: The grid-stride loop.
    """
    sizes = entry.map.range.size()
    spread = [d for d, extent in enumerate(sizes) if not is_small(extent, small_symbols)] or list(range(len(sizes)))
    if len(spread) < len(sizes):
        entry, inner = xfh.extract_map_dims(sdfg, entry, spread)
        inner.map.schedule = dtypes.ScheduleType.Sequential
    flatten_to_grid_stride(state, entry, tid, nthreads)
    return entry
