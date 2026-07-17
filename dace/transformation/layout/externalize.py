# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Externalize one loop nest into a standalone runnable SDFG (GLOBAL_LAYOUT_DESIGN.md, task A1).

The global-layout machinery scores and times nests INDIVIDUALLY: candidate schedules are derived on
a copy (the whole-SDFG schedule passes must never run on the multi-nest program), eval mode compiles
and times one nest per candidate layout, and per-nest correctness is checked against a per-nest
oracle. ``dace.sdfg.analysis.cutout.SDFGCutout`` already owns the hard part -- deep-copying a state
subgraph into a fresh SDFG with cloned descriptors, free symbols, and boundary transients promoted
to globals -- so externalization is a thin wrapper: pick the nest's scope subgraph, cut it out, give
the result a stable unique name (unique names give disjoint build folders, the same-name build-cache
hazard the test suite already hit once).

nest-forge is deliberately NOT involved: its emit/arena machinery solves cross-compiler lanes; one
map scope to a runnable SDFG is a cutout plus naming.
"""
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


def externalize_nest(state: SDFGState,
                     map_entry: Optional[nodes.MapEntry] = None,
                     name: Optional[str] = None) -> SDFG:
    """Cut the loop nest under ``map_entry`` out of ``state`` into a standalone runnable SDFG.

    The cutout deep-copies the nest, clones the descriptors and free symbols it references, and
    promotes boundary transients (produced by an earlier nest / consumed by a later one) to
    non-transient inputs/outputs -- so the result runs on caller-provided buffers.

    :param state: the state holding the nest.
    :param map_entry: the nest's top-level map entry; ``None`` picks the state's single nest and
                      refuses a state with several (run kernel_per_state first, or pass it).
    :param name: the new SDFG's name; default ``{sdfg.name}__{map label}``, sanitized. Callers
                 building candidate variants must pass a per-candidate unique name.
    :return: the externalized SDFG (an ``SDFGCutout``).
    """
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
    """The names of non-transient arrays the externalized nest writes (its real outputs).
    Transient writes are internal staging (e.g. canonicalize's per-element slice buffers) unless the
    cutout promoted them to globals -- promotion is exactly the boundary-crossing test."""
    written = set()
    for state in ext.states():
        for node in state.data_nodes():
            if state.in_degree(node) > 0 and not ext.arrays[node.data].transient:
                written.add(node.data)
    return written


def nest_arguments(ext: SDFG,
                   symbols: Dict[str, int],
                   provided: Optional[Dict[str, numpy.ndarray]] = None,
                   seed: int = 0) -> Dict[str, numpy.ndarray]:
    """Deterministic argument buffers for an externalized nest.

    Config-``provided`` arrays are taken verbatim (copied, so a run never mutates the caller's
    reference data); every other non-transient array gets deterministic values from ``seed`` --
    iteration is over sorted names, so the same (nest, seed) always sees the same bytes.

    :param ext: the externalized SDFG.
    :param symbols: concrete sizes for the free symbols (concrete shapes are the v1 contract).
    :param provided: arrays supplied by the caller (program inputs), by name.
    :param seed: seed for the deterministic fill of everything not provided.
    """
    provided = provided or {}
    rng = numpy.random.default_rng(seed)
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
        if numpy.issubdtype(dtype, numpy.floating):
            args[aname] = rng.random(shape).astype(dtype)
        elif numpy.issubdtype(dtype, numpy.integer):
            args[aname] = rng.integers(0, 8, size=shape, dtype=dtype)
        else:
            raise NotImplementedError(f"nest_arguments: no deterministic fill for dtype {dtype} of '{aname}'")
    return args
