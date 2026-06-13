# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Invariant checker functions for K-dim vectorization passes.

Per user direction 2026-06-12: ``We should have pre condition and post
condition checks for all passes and by default and always and always run
them.`` + ``Does pre condition really need to pass? Can't be just a
function that checks the condition holds true in a graph / subgraph etc?``

This module exposes plain functions:

* Each checker takes an SDFG (and optionally other args) and returns
  ``None`` if the invariant holds, or a string describing the violation.
* :func:`assert_invariant` raises ``AssertionError`` on violation with a
  formatted message including the pass name + checker description +
  offending node / edge / state.

Each pass calls the checkers directly from its ``apply_pass``:

.. code-block:: python

    def apply_pass(self, sdfg, _):
        result = self._do_work(sdfg)
        assert_invariant(no_memlet_dim_mismatch(sdfg),
                         "WidenAccesses", "memlet dim consistent")
        return result

No mixin, no inheritance, no env gate. Always runs.
"""
from typing import Callable, Optional, Tuple

import dace
from dace.sdfg import SDFG, SDFGState
from dace.sdfg.nodes import AccessNode, MapEntry, NestedSDFG, Tasklet


def assert_invariant(violation: Optional[str], pass_name: str, description: str) -> None:
    """Raise :class:`AssertionError` if ``violation`` is non-None.

    :param violation: The checker's return value (``None`` on success or
        the offending-node description).
    :param pass_name: The pass name to include in the error message.
    :param description: One-line description of the invariant.
    """
    if violation is None:
        return
    raise AssertionError(f"{pass_name}: invariant violated -- {description}: {violation}")


# ---------------------------------------------------------------------------
# Generic structural invariants (work at SDFG or per-state level).
# ---------------------------------------------------------------------------


def no_memlet_dim_mismatch(scope) -> Optional[str]:
    """For memlets connecting a tasklet / lib-node / NSDFG connector to
    an AccessNode (or two such connectors), ``subset`` and ``other_subset``
    must have matching rank. Accepts an SDFG OR a single :class:`SDFGState`.
    Returns ``None`` on success or a description.

    AN -> AN memlets are exempt: they describe pure copies between two
    descriptors that may have different ranks (e.g. a 4D slice copied
    into a 1D flat buffer) so the two subsets are intentionally
    different-rank when the descriptors are.
    """
    states = _iter_states(scope)
    for sd, state in states:
        for edge in state.edges():
            mem = edge.data
            if mem is None or mem.subset is None or mem.other_subset is None:
                continue
            # Pure AN -> AN copies are allowed to carry different-rank
            # subsets when the two descriptors have different ranks.
            if isinstance(edge.src, AccessNode) and isinstance(edge.dst, AccessNode):
                continue
            if len(mem.subset.size()) != len(mem.other_subset.size()):
                return (f"{sd.name}.{state.label}: memlet ``{mem.data}`` subset dim={len(mem.subset.size())} "
                        f"!= other_subset dim={len(mem.other_subset.size())}")
    return None


def no_isolated_access_nodes(scope) -> Optional[str]:
    """No AccessNode may have zero in-edges AND zero out-edges. Accepts
    an SDFG or a single state.
    """
    for sd, state in _iter_states(scope):
        for node in state.nodes():
            if not isinstance(node, AccessNode):
                continue
            if state.in_degree(node) == 0 and state.out_degree(node) == 0:
                return f"{sd.name}.{state.label}: isolated AccessNode ``{node.data}``"
    return None


def no_duplicate_connector_edges(scope) -> Optional[str]:
    """Every NSDFG / Tasklet / lib-node connector has <=1 edge per direction.

    :class:`~dace.sdfg.nodes.MapEntry` and :class:`~dace.sdfg.nodes.MapExit`
    are skipped: their pass-through connectors are designed to fan-out
    (entry's ``OUT_X``) and fan-in (exit's ``IN_X``).
    """
    from dace.sdfg.nodes import MapEntry, MapExit
    for sd, state in _iter_states(scope):
        for node in state.nodes():
            if isinstance(node, (MapEntry, MapExit)):
                continue
            in_counts = {}
            for e in state.in_edges(node):
                if e.dst_conn is None:
                    continue
                in_counts.setdefault(e.dst_conn, 0)
                in_counts[e.dst_conn] += 1
            for conn, count in in_counts.items():
                if count > 1:
                    return (f"{sd.name}.{state.label}: {type(node).__name__} ``{getattr(node, 'label', node)}`` "
                            f"in-connector ``{conn}`` has {count} edges (max 1)")
            out_counts = {}
            for e in state.out_edges(node):
                if e.src_conn is None:
                    continue
                out_counts.setdefault(e.src_conn, 0)
                out_counts[e.src_conn] += 1
            for conn, count in out_counts.items():
                if count > 1:
                    return (f"{sd.name}.{state.label}: {type(node).__name__} ``{getattr(node, 'label', node)}`` "
                            f"out-connector ``{conn}`` has {count} edges (max 1)")
    return None


def mask_connectors_are_bool(scope) -> Optional[str]:
    """Every edge feeding a tile lib-node ``_mask`` connector must be sourced
    from a ``bool`` array. A mask selects per-lane, so a non-bool mask (e.g. a
    ``double`` 1.0/0.0 value mask) is invalid — comparison ops and lifted
    if-conditions produce ``bool``, and every mask consumer (TileBinop /
    TileUnop / TileITE ``_mask``) is defined over a boolean tile.

    Accepts an SDFG or a single :class:`SDFGState`. Returns ``None`` on
    success or a description identifying the first non-bool mask source.
    """
    import dace.dtypes as _dt
    for sd, state in _iter_states(scope):
        for edge in state.edges():
            if edge.dst_conn != "_mask":
                continue
            mem = edge.data
            if mem is None or mem.data is None:
                continue
            desc = sd.arrays.get(mem.data)
            if desc is None:
                continue
            if desc.dtype != _dt.bool_:
                return (f"{sd.name}.{state.label}: ``_mask`` connector on "
                        f"{type(edge.dst).__name__} ``{getattr(edge.dst, 'label', edge.dst)}`` is fed by "
                        f"``{mem.data}`` of dtype {desc.dtype} (must be bool)")
    return None


def memlet_subset_matches_descriptor(scope) -> Optional[str]:
    """Every memlet's ``subset`` rank must match the rank of the descriptor it
    accesses (``len(sdfg.arrays[memlet.data].shape)``). A memlet that reads a
    ``(1,)`` scalar bridge with a 2-D ``[0:8, 0:8]`` tile subset (or vice
    versa) is invalid — the descriptor and the access disagree on rank, which
    ``sdfg.validate()`` later rejects. Surfacing it as a pass post-condition
    localizes which pass widened the memlet without widening the descriptor (or
    staged a too-narrow bridge under a widened consumer).

    Accepts an SDFG or a single :class:`SDFGState`. Returns ``None`` on success
    or a description of the first offender.
    """
    for sd, state in _iter_states(scope):
        for edge in state.edges():
            mem = edge.data
            if mem is None or mem.data is None or mem.subset is None:
                continue
            desc = sd.arrays.get(mem.data)
            if desc is None:
                continue
            if len(mem.subset.size()) != len(desc.shape):
                src = getattr(edge.src, "label", type(edge.src).__name__)
                dst = getattr(edge.dst, "label", type(edge.dst).__name__)
                return (f"{sd.name}.{state.label}: memlet ``{mem.data}`` subset rank "
                        f"{len(mem.subset.size())} != descriptor rank {len(desc.shape)} "
                        f"(shape {tuple(desc.shape)}) on edge {src} -> {dst}")
    return None


def logical_binops_are_bool(scope) -> Optional[str]:
    """Every ``TileBinop`` with a logical op (``&&`` / ``||``) must have two
    ``bool`` inputs (``_a``, ``_b``) and a ``bool`` output (``_c``). A logical
    op over non-bool operands is invalid — the operands are predicates / masks
    and the result is a predicate.

    Accepts an SDFG or a single :class:`SDFGState`. Returns ``None`` on
    success or a description of the first offender.
    """
    import dace.dtypes as _dt
    from dace.libraries.tileops import TileBinop
    for sd, state in _iter_states(scope):
        for node in state.nodes():
            if not isinstance(node, TileBinop) or node.op not in ("&&", "||"):
                continue
            for conn in ("_a", "_b", "_c"):
                edges = ([e for e in state.in_edges(node) if e.dst_conn == conn]
                         + [e for e in state.out_edges(node) if e.src_conn == conn])
                for e in edges:
                    if e.data is None or e.data.data is None:
                        continue
                    desc = sd.arrays.get(e.data.data)
                    if desc is not None and desc.dtype != _dt.bool_:
                        return (f"{sd.name}.{state.label}: logical TileBinop ``{node.label}`` (op {node.op}) "
                                f"connector ``{conn}`` is ``{e.data.data}`` of dtype {desc.dtype} (must be bool)")
    return None


def sdfg_validates(sdfg: SDFG) -> Optional[str]:
    """``sdfg.validate()`` succeeds."""
    try:
        sdfg.validate()
        return None
    except Exception as e:  # noqa: BLE001
        return f"SDFG validation failed: {type(e).__name__}: {str(e)[:200]}"


# ---------------------------------------------------------------------------
# K-dim pipeline invariants (require widths / K context).
# ---------------------------------------------------------------------------


def lane_dep_transients_widened(sdfg: SDFG, K: int, widths: Tuple[int, ...]) -> Optional[str]:
    """Every lane-dependent transient in a tile-tagged body NSDFG is at
    the tile shape ``widths`` OR is an exempt bridge name (gather idx
    tile / ITE materialised tile / cond broadcast tile / Scalar bridge).

    Per user example 2026-06-12: ``all non-scalar non-gather dims are
    widened``.
    """
    import dace.data as _dd
    for _state, nsdfg_node, _map_entry in _tile_tagged_bodies(sdfg, K):
        inner_sdfg = nsdfg_node.sdfg
        for name, desc in inner_sdfg.arrays.items():
            if not desc.transient:
                continue
            if name.startswith("_idx_") or name.startswith("_ite_sym_tile") or name.startswith("_cond_bcast"):
                continue
            if isinstance(desc, _dd.Scalar):
                continue
            if not isinstance(desc, _dd.Array):
                continue
            shape = tuple(desc.shape)
            if shape == tuple(widths):
                continue
            try:
                if all(bool(dace.symbolic.simplify(s - 1) == 0) for s in shape):
                    continue
            except Exception:  # noqa: BLE001
                pass
            return (f"{inner_sdfg.name}: lane-dep transient ``{name}`` has shape {shape} "
                    f"!= widths {tuple(widths)} (expected widened or Scalar bridge)")
    return None


def innermost_map_has_body_nsdfg(sdfg: SDFG, K: int) -> Optional[str]:
    """Every tile-tagged innermost map's scope contains exactly one NestedSDFG."""
    from dace.transformation.passes.vectorization.split_map_for_tile_remainder import (SCALAR_TAIL_MARKER,
                                                                                        TILE_K1_TAIL_MARKER)
    from dace.transformation.passes.vectorization.utils.map_predicates import is_innermost_map
    for sd in sdfg.all_sdfgs_recursive():
        for state in sd.states():
            for node in state.nodes():
                if not isinstance(node, MapEntry):
                    continue
                try:
                    if not is_innermost_map(state, node):
                        continue
                except (StopIteration, ValueError):
                    continue
                if len(node.map.params) < K:
                    continue
                if node.map.label.endswith(SCALAR_TAIL_MARKER) or node.map.label.endswith(TILE_K1_TAIL_MARKER):
                    continue
                try:
                    scope = state.scope_subgraph(node, include_entry=False, include_exit=False).nodes()
                except (StopIteration, ValueError):
                    continue
                nsdfgs = [n for n in scope if isinstance(n, NestedSDFG)]
                if len(nsdfgs) != 1:
                    return (f"{sd.name}.{state.label}: tile-tagged innermost map ``{node.map.label}`` "
                            f"has {len(nsdfgs)} NestedSDFGs in scope (expected 1)")
    return None


def tile_main_map_step_is_widths(sdfg: SDFG, K: int, widths: Tuple[int, ...]) -> Optional[str]:
    """Every TILE_MAIN map has its last-K dim steps == ``widths``."""
    from dace.transformation.passes.vectorization.split_map_for_tile_remainder import TILE_MAIN_MARKER
    for sd in sdfg.all_sdfgs_recursive():
        for state in sd.states():
            for node in state.nodes():
                if not isinstance(node, MapEntry):
                    continue
                if not node.map.label.endswith(TILE_MAIN_MARKER):
                    continue
                if len(node.map.range) < K:
                    continue
                tail_steps = tuple(node.map.range[-K + d][2] for d in range(K))
                if tuple(str(s) for s in tail_steps) != tuple(str(s) for s in widths):
                    return (f"{sd.name}.{state.label}: TILE_MAIN map ``{node.map.label}`` last-K steps "
                            f"{tail_steps} != expected widths {tuple(widths)}")
    return None


# ---------------------------------------------------------------------------
# Helpers (private).
# ---------------------------------------------------------------------------


def _iter_states(scope):
    """Yield ``(sub_sdfg, state)`` for an SDFG (every state recursively)
    OR a single state passed directly.
    """
    if isinstance(scope, SDFGState):
        yield scope.sdfg, scope
        return
    if isinstance(scope, SDFG):
        for sd in scope.all_sdfgs_recursive():
            for state in sd.states():
                yield sd, state
        return
    raise TypeError(f"Invariant scope must be SDFG or SDFGState, got {type(scope).__name__}")


def _tile_tagged_bodies(sdfg: SDFG, K: int):
    """Yield ``(state, nsdfg_node, map_entry)`` for every tile-tagged body NSDFG."""
    from dace.transformation.passes.vectorization.split_map_for_tile_remainder import (SCALAR_TAIL_MARKER,
                                                                                        TILE_K1_TAIL_MARKER)
    from dace.transformation.passes.vectorization.utils.map_predicates import is_innermost_map
    for sd in sdfg.all_sdfgs_recursive():
        for state in sd.states():
            for node in state.nodes():
                if not isinstance(node, MapEntry):
                    continue
                try:
                    if not is_innermost_map(state, node):
                        continue
                except (StopIteration, ValueError):
                    continue
                if len(node.map.params) < K:
                    continue
                if node.map.label.endswith(SCALAR_TAIL_MARKER) or node.map.label.endswith(TILE_K1_TAIL_MARKER):
                    continue
                try:
                    scope = state.scope_subgraph(node, include_entry=False, include_exit=False).nodes()
                except (StopIteration, ValueError):
                    continue
                nsdfgs = [n for n in scope if isinstance(n, NestedSDFG)]
                if len(nsdfgs) != 1:
                    continue
                yield state, nsdfgs[0], node
