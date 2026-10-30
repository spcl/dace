# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Runs a kernel map inside a persistent megakernel: every thread walks every ``nthreads``-th column in steps of the
grid size, and dimensions too short to spread over the grid stay a loop in every thread."""

from dace import SDFG, SDFGState, dtypes, subsets, symbolic
from dace.sdfg import nodes
from dace.sdfg.replace import replace_dict
from dace.transformation import helpers as xfh

#: A literal extent below this stays a loop in every thread
SMALL_EXTENT = 64
#: The index of a grid-stride loop
FLAT_INDEX = "__mk_g"


def is_small(extent: symbolic.SymbolicType, small_symbols: set[str]) -> bool:
    """Whether a map dimension of ``extent`` iterations is too short to spread over the grid: a literal below
    :data:`SMALL_EXTENT`, or an expression of ``small_symbols`` only."""
    extent = symbolic.pystr_to_symbolic(extent)
    if extent.is_Number:
        return int(extent) < SMALL_EXTENT
    return bool(small_symbols) and {str(s) for s in extent.free_symbols} <= small_symbols


def to_grid_stride(state: SDFGState, entry: nodes.MapEntry, tid: str, nthreads: str) -> None:
    """Turns the one-dimensional ``entry`` into a sequential loop from thread ``tid`` in steps of ``nthreads``; the body
    reads the original index back from the loop's."""
    (begin, _, step), size = entry.map.range[0], entry.map.range.size()[0]
    param, index = entry.map.params[0], begin + step * symbolic.symbol(FLAT_INDEX, dtypes.int64)
    replace_dict(state.scope_subgraph(entry), {param: str(index)}, {symbolic.pystr_to_symbolic(param): index})
    entry.map.params = [FLAT_INDEX]
    entry.map.range = subsets.Range([(symbolic.pystr_to_symbolic(tid), size - 1, symbolic.pystr_to_symbolic(nthreads))])
    entry.map.schedule = dtypes.ScheduleType.Sequential


def grid_stride(
    sdfg: SDFG, state: SDFGState, entry: nodes.MapEntry, tid: str, nthreads: str, small_symbols: set[str]
) -> nodes.MapEntry:
    """Lowers the kernel map ``entry`` to a grid-stride loop of thread ``tid`` among ``nthreads`` over its innermost
    dimension that is not small (:func:`is_small`), or its innermost if all are: on nproma-first data, the columns. The
    other dimensions stay loops, in an inner sequential map, so a thread walks its column.

    :return: The grid-stride loop.
    """
    sizes = entry.map.range.size()
    spread = max((d for d, extent in enumerate(sizes) if not is_small(extent, small_symbols)), default=len(sizes) - 1)
    if len(sizes) > 1:
        entry, inner = xfh.extract_map_dims(sdfg, entry, [spread])
        inner.map.schedule = dtypes.ScheduleType.Sequential
    to_grid_stride(state, entry, tid, nthreads)
    return entry
