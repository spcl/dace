"""
hoist: optimization that lifts allocations out of LoopRegions.

If a transient array is allocated (and freed) inside a ``LoopRegion``, every
loop iteration pays the full allocation/deallocation cost.  This pass moves
the allocation to before the loop and the deallocation to after the loop,
so each happens only once.

Arrays that already carry ``AllocationLifetime.Explicit`` with ``alloc``/``free``
annotations on loop-internal edges are hoisted directly.

Arrays that do not yet use explicit allocation are first converted via
``make_explicit`` (which scopes the alloc/free to the loop's internal edges
for ``Scope``/``State``-lifetime arrays) and then hoisted.  Arrays whose
declared lifetime is ``SDFG`` or ``Global`` are already allocated outside
the loop after ``make_explicit`` and are silently skipped.

Only ``LoopRegion`` blocks are accepted.  Passing a Map scope raises a
``TypeError`` immediately: Maps are parallel — hoisting an allocation out of a
Map would change the memory layout visible to every parallel thread.

Usage::

    from dace.libraries.allocation.hoist import hoist_alloc_out_of_loop
    # already-explicit arrays
    hoist_alloc_out_of_loop(loop_region, ['dx', 'dy', 'dz'])
    # automatic-lifetime arrays (make_explicit + hoist in one call)
    hoist_alloc_out_of_loop(loop_region, ['tmp'])
"""

from typing import List, Tuple

import dace
from dace import dtypes
from dace.sdfg.state import LoopRegion, ControlFlowRegion

from .make_explicit import make_explicit, _alloc_on_incoming_edges, _free_on_outgoing_edges


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _collect_alloc_free_edges(
    region: ControlFlowRegion,
    name: str,
) -> Tuple[list, list]:
    """Recursively collect every edge inside *region* that references *name*.

    Returns ``(alloc_edges, free_edges)`` — both are flat lists of
    ``dace.sdfg.graph.Edge`` objects whose ``data`` is an
    ``InterstateEdge``.

    The search recurses into nested ``ControlFlowRegion`` blocks (e.g. a
    ``LoopRegion`` nested inside another loop, or a ``ConditionalBlock``).
    """
    alloc_edges: list = []
    free_edges:  list = []

    for edge in region.edges():
        if name in edge.data.alloc:
            alloc_edges.append(edge)
        if name in edge.data.free:
            free_edges.append(edge)

    for node in region.nodes():
        if isinstance(node, ControlFlowRegion):
            sub_alloc, sub_free = _collect_alloc_free_edges(node, name)
            alloc_edges.extend(sub_alloc)
            free_edges.extend(sub_free)

    return alloc_edges, free_edges


def _root_sdfg(region: ControlFlowRegion) -> "dace.SDFG":
    """Walk the parent_graph chain upward until we reach the root SDFG."""
    node = region
    while not isinstance(node, dace.SDFG):
        node = node.parent_graph
    return node


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def hoist_alloc_out_of_loop(loop: LoopRegion, array_names: List[str]) -> None:
    """Hoist explicit allocations of *array_names* from inside a loop to its
    surrounding CFG.

    For each array the function:

    1. Validates that *loop* is a ``LoopRegion`` (raises ``TypeError``
       otherwise — Maps are parallel and must not be used here).
    2. Collects every edge inside *loop* (recursively) that has *name* in its
       ``alloc`` or ``free`` list.
    3. Raises ``ValueError`` if no ``alloc`` edge for *name* is found inside
       the loop — the array must actually be allocated inside the loop.
    4. Removes *name* from the ``alloc`` list of every internal edge found.
    5. Removes *name* from the ``free``  list of every internal edge found.
    6. Adds *name* to the ``alloc`` list of every incoming edge of *loop* in
       its parent CFG (inserting a thin predecessor state when the loop is
       the SDFG start and has no incoming edges).
    7. If internal ``free`` edges were found, adds *name* to the ``free``
       list of every outgoing edge of *loop* in its parent CFG (inserting a
       thin successor state when the loop is a CFG sink).

    :param loop:         The ``LoopRegion`` to hoist allocations out of.
    :param array_names:  Names of transient arrays to hoist.
    :raises TypeError:   If *loop* is not a ``LoopRegion``.
    :raises ValueError:  If an array name is unknown, not transient, or has no
                         ``alloc`` annotation on any edge inside *loop*.
    """
    if not isinstance(loop, LoopRegion):
        raise TypeError(
            f"hoist_alloc_out_of_loop expects a LoopRegion, got "
            f"'{type(loop).__name__}'. "
            "Maps are parallel — hoisting allocations out of a Map would "
            "change per-iteration memory layout visible to every parallel "
            "thread and is therefore not supported."
        )

    parent = loop.parent_graph
    sdfg   = _root_sdfg(loop)

    for name in array_names:
        # --- validate ---
        if name not in sdfg.arrays:
            raise ValueError(f"'{name}' is not in the SDFG's array descriptor table.")
        desc = sdfg.arrays[name]
        if not desc.transient:
            raise ValueError(
                f"'{name}' is not a transient data container. "
                "Only transients can be explicitly allocated."
            )

        # --- make explicit if needed ---
        # For Scope/State-lifetime arrays this places alloc/free on edges
        # inside the loop (at the LCA of uses), so the hoist logic below
        # can find and move them.  For SDFG/Global-lifetime arrays
        # make_explicit places alloc/free at the SDFG level already, so
        # there will be nothing to hoist — we skip those silently.
        if desc.lifetime != dtypes.AllocationLifetime.Explicit:
            make_explicit(sdfg, [name])

        # --- find alloc / free edges inside the loop ---
        alloc_edges, free_edges = _collect_alloc_free_edges(loop, name)

        if not alloc_edges:
            # Either the array was SDFG/Global-lifetime (already outside the
            # loop after make_explicit) or it genuinely has no uses inside
            # this loop — nothing to do.
            continue

        # --- remove from internal edges ---
        for edge in alloc_edges:
            edge.data.alloc.remove(name)
        for edge in free_edges:
            edge.data.free.remove(name)

        # --- place alloc before the loop ---
        _alloc_on_incoming_edges(parent, loop, name)

        # --- place free after the loop (only if there were internal frees) ---
        if free_edges:
            _free_on_outgoing_edges(parent, loop, name)
