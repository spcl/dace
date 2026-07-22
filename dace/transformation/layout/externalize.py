# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Externalize one loop nest into a standalone runnable SDFG (GLOBAL_LAYOUT_DESIGN.md, task A1); a thin wrapper over ``SDFGCutout`` that cuts the nest's scope subgraph out and gives it a stable unique name."""
import math
import re
from typing import Dict, Optional

import numpy

import dace
from dace import SDFG, SDFGState
from dace.sdfg import nodes
from dace.sdfg.analysis.cutout import SDFGCutout


def nest_entries(state: SDFGState):
    """The top-level MapEntry nodes of ``state`` -- one per kernel nest post-canonicalize."""
    return [n for n in state.scope_children()[None] if isinstance(n, nodes.MapEntry)]


def externalize_nest(state: SDFGState, map_entry: Optional[nodes.MapEntry] = None, name: Optional[str] = None) -> SDFG:
    """Cut the loop nest under ``map_entry`` out of ``state`` into a standalone runnable SDFG; promotes boundary transients to non-transient inputs/outputs."""
    if map_entry is None:
        entries = nest_entries(state)
        if len(entries) != 1:
            raise ValueError(f"externalize_nest: state '{state.label}' holds {len(entries)} top-level "
                             f"nests; pass map_entry to pick one")
        map_entry = entries[0]
    scope_nodes = state.scope_subgraph(map_entry).nodes()
    cut = SDFGCutout.singlestate_cutout(state,
                                        *scope_nodes,
                                        make_copy=True,
                                        make_side_effects_global=True,
                                        use_alibi_nodes=False)
    cut.name = re.sub(r"\W", "_", name or f"{state.sdfg.name}__{map_entry.map.label}")
    cut.validate()
    return cut


def written_array_names(ext: SDFG):
    """Names of non-transient arrays the externalized nest writes (its real outputs)."""
    written = set()
    for state in ext.states():
        for node in state.data_nodes():
            if state.in_degree(node) > 0 and not ext.arrays[node.data].transient:
                written.add(node.data)
    return written


def constant_or_none(expr, symbols: Dict[str, int]) -> Optional[int]:
    """``expr`` as an int, or None when ``symbols`` does not pin it down (e.g. a map parameter)."""
    try:
        return int(dace.symbolic.evaluate(expr, symbols))
    except (TypeError, ValueError, KeyError):
        return None


def indexed_extent_bound(ext: SDFG, symbols: Dict[str, int]) -> Optional[int]:
    """Smallest extent of any array an index array could subscript -- the tightest in-bounds cap for an
    integer fill; None when the nest has no such access.

    Two signals are needed because the frontend lowers indirection differently per source form:
      * a DYNAMIC (data-dependent) memlet -- e.g. a masked gather, where the condition propagates it;
      * a WHOLE-array read handed to the map BODY -- a plain ``data[idx[i]]`` becomes a nested SDFG
        receiving all of ``data`` through a STATIC full-range memlet, so the dynamic flag never gets set.
    An ordinary elementwise or stencil read enters the body with a point/partial subset and trips neither.
    """
    extents = []
    for state in ext.states():
        for edge in state.edges():
            name = edge.data.data
            if name is None or name not in ext.arrays or ext.arrays[name].transient:
                continue
            shape = [constant_or_none(s, symbols) for s in ext.arrays[name].shape]
            # An extent that `symbols` does not fix (a map parameter, in a triangular nest) leaves the cap
            # unconstrained rather than failing -- the cap is advisory. A 1-element descriptor is a scalar
            # operand, never an index target, and would otherwise clamp every integer fill to zero.
            if None in shape or math.prod(shape) <= 1:
                continue
            elements = constant_or_none(edge.data.subset.num_elements(), symbols) if edge.data.subset else None
            whole = isinstance(edge.src, nodes.MapEntry) and elements == math.prod(shape)
            if edge.data.dynamic or whole:
                extents += shape
    return min(extents) if extents else None


def nest_arguments(ext: SDFG,
                   symbols: Dict[str, int],
                   provided: Optional[Dict[str, numpy.ndarray]] = None,
                   seed: int = 0) -> Dict[str, numpy.ndarray]:
    """Deterministic argument buffers for an externalized nest. ``provided`` arrays are copied verbatim; everything else gets a deterministic fill from ``seed`` (sorted name order)."""
    provided = provided or {}
    rng = numpy.random.default_rng(seed)
    bound = indexed_extent_bound(ext, symbols)
    integer_high = 8 if bound is None else max(1, min(8, bound))
    args: Dict[str, numpy.ndarray] = {}
    for aname in sorted(ext.arrays):
        desc = ext.arrays[aname]
        if desc.transient:
            continue
        if aname in provided:
            args[aname] = provided[aname].copy()
            continue
        shape = tuple(int(dace.symbolic.evaluate(s, symbols)) for s in desc.shape)
        dtype = desc.dtype.as_numpy_dtype()
        if numpy.issubdtype(dtype, numpy.complexfloating):
            args[aname] = (rng.random(shape) + 1j * rng.random(shape)).astype(dtype)
        elif numpy.issubdtype(dtype, numpy.floating):
            args[aname] = rng.random(shape).astype(dtype)
        elif numpy.issubdtype(dtype, numpy.integer):
            # this array may BE the index of an indirect access; capping the fill at the smallest indexed
            # extent keeps it in bounds whichever axis it subscripts. Verification alone cannot catch an
            # out-of-bounds index -- reference and candidates read the same wrong slot and agree.
            args[aname] = rng.integers(0, integer_high, size=shape, dtype=dtype)
        else:
            raise NotImplementedError(f"nest_arguments: no deterministic fill for dtype {dtype} of '{aname}'")
    return args
