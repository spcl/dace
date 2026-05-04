"""
make_explicit: utility to convert transient data containers from automatic
(scope-based lifetime) allocation to explicit allocation via the
AllocationLifetime.Explicit / interstate-edge alloc/free mechanism.

For each array the function:
  1. Collects all SDFGStates that contain an access node for the array.
  2. Finds the lowest common ancestor (LCA) ControlFlowRegion of those states.
     This preserves scope semantics: an array used only inside a LoopRegion
     gets alloc/free on edges *within* that loop, not on edges surrounding it
     in the parent CFG.
  3. Within the LCA region, finds the first and last direct child blocks (in
     topological execution order) that contain a use of the array.
  4. Adds the array name to the ``alloc`` list of every incoming interstate
     edge of the first-use block inside the LCA.  A new thin state is inserted
     only when the first-use block has no incoming edges in the LCA.
  5. Adds the array name to the ``free`` list of every outgoing interstate
     edge of the last-use block inside the LCA.  A new thin successor state is
     inserted only when the last-use block has no outgoing edges in the LCA.
  6. Sets the array's ``lifetime`` to ``AllocationLifetime.Explicit`` so that
     DaCe's normal codegen skips the automatic ``new``/``delete[]`` and the
     control-flow code-generator emits them on the edge traversal instead.

Note: Map-scoped arrays (AllocationLifetime.Scope where the innermost scope is
a Map node within a state) cannot be expressed via interstate edge annotations,
since Maps have no CFG-level boundaries.  make_explicit does not special-case
this; callers should not pass such arrays.

Usage::

    from dace.libraries.allocation.make_explicit import make_explicit
    make_explicit(sdfg, ['dx', 'dy', 'dz'])
"""

from typing import List

from dace import dtypes
from dace.sdfg import SDFG, SDFGState, InterstateEdge
from dace.sdfg.state import ControlFlowRegion
from dace.sdfg.analysis import cfg as cfg_analysis


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ancestor_chain(state, sdfg: SDFG) -> list:
    """Return [sdfg, ..., direct_parent_of_state] — ancestor regions from root
    down to (but not including) *state* itself."""
    chain = []
    node = state.parent_graph
    while node is not sdfg:
        chain.append(node)
        node = node.parent_graph
    chain.append(sdfg)
    return list(reversed(chain))  # sdfg first, deepest parent last


def _lca_region(states: list, sdfg: SDFG) -> ControlFlowRegion:
    """Return the lowest common ancestor ControlFlowRegion of *states*.

    The LCA is the innermost region that contains every state in *states*.
    For states all inside the same LoopRegion this returns that LoopRegion;
    for states spread across the SDFG it returns the SDFG itself.
    """
    if not states:
        return sdfg

    chains = [_ancestor_chain(s, sdfg) for s in states]
    lca = sdfg
    for regions in zip(*chains):
        if len({id(r) for r in regions}) == 1:
            lca = regions[0]
        else:
            break
    return lca


def _top_level_block_in(region: ControlFlowRegion, node):
    """Walk up the parent_graph chain to find the direct child of *region*
    that contains *node*.  *node* may already be a direct child."""
    while node.parent_graph is not region:
        node = node.parent_graph
    return node


def _blocks_using_in(region: ControlFlowRegion, name: str) -> list:
    """Return the direct children of *region* (states or nested regions) that
    contain an access node for *name*, in topological execution order
    (duplicates removed)."""
    seen = set()
    result: list = []
    for state in cfg_analysis.blockorder_topological_sort(region, ignore_nonstate_blocks=True):
        if any(n.data == name for n in state.data_nodes()):
            top = _top_level_block_in(region, state)
            if top not in seen:
                result.append(top)
                seen.add(top)
    return result


def _alloc_on_incoming_edges(region: ControlFlowRegion, block, name: str) -> None:
    """Add *name* to the ``alloc`` list of every incoming edge of *block*
    within *region*.

    If *block* has no incoming edges (i.e. it is the region's start block), a
    new predecessor state is inserted and the alloc is placed on the edge
    from the new state to *block*.
    """
    in_edges = region.in_edges(block)
    if in_edges:
        for edge in in_edges:
            if name not in edge.data.alloc:
                edge.data.alloc.append(name)
    else:
        # block is the start — insert a thin predecessor to carry the edge
        new_pre = region.add_state_before(block, label=f'_alloc_pre_{name}', is_start_block=True)
        alloc_edge = region.edges_between(new_pre, block)[0]
        alloc_edge.data.alloc.append(name)


def _free_on_outgoing_edges(region: ControlFlowRegion, block, name: str) -> None:
    """Add *name* to the ``free`` list of every outgoing edge of *block*
    within *region*.

    If *block* has no outgoing edges (i.e. it is a sink), a new successor
    state is inserted and the free is placed on the edge from *block* to the
    new state.
    """
    out_edges = region.out_edges(block)
    if out_edges:
        for edge in out_edges:
            if name not in edge.data.free:
                edge.data.free.append(name)
    else:
        # block is a sink — insert a thin successor to carry the edge
        new_suc = region.add_state_after(block, label=f'_free_suc_{name}')
        free_edge = region.edges_between(block, new_suc)[0]
        free_edge.data.free.append(name)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_explicit(sdfg: SDFG, array_names: List[str]) -> None:
    """Convert transient data containers to explicit interstate-edge allocation.

    Allocation and deallocation are placed at the tightest scope that covers
    all uses of the array — for example, inside a LoopRegion if the array is
    only used there, or at the SDFG level if uses span multiple top-level
    blocks.

    **Limitation — size-1 arrays**: DaCe's state-struct codegen represents
    single-element transients as plain scalar fields (e.g. ``double x;``)
    rather than pointer fields (``double *x;``).  Placing them on alloc/free
    edges would make the C++ emitter emit ``__state->x = new double[1];``
    against a scalar declaration, which is a compile-time type error.
    ``make_explicit`` therefore raises :class:`ValueError` for any array
    whose total element count is 1.  Callers that discover such an array
    should either skip it or use a different strategy (e.g. scalar alias
    rather than pointer reuse).

    :param sdfg: The SDFG to modify in-place.
    :param array_names: Names of transient arrays to make explicitly allocated.
    :raises ValueError: If a name is not a transient array in *sdfg*, or if
        the array has exactly one element (size-1 scalar limitation).
    """
    # Validate ALL arrays before modifying anything so that a late failure
    # cannot leave the SDFG in a partially-modified state.
    for name in array_names:
        if name not in sdfg.arrays:
            raise ValueError(f"'{name}' not found in sdfg.arrays")
        desc = sdfg.arrays[name]
        if not desc.transient:
            raise ValueError(f"'{name}' is not a transient data container")
        if desc.total_size == 1:
            raise ValueError(
                f"'{name}' has total_size=1.  DaCe stores single-element "
                f"transients as scalar state-struct fields, not pointers; "
                f"explicit heap allocation is not supported for them."
            )

        # Remember the original lifetime before overwriting — used below to
        # decide the target scope region.
        original_lifetime = desc.lifetime

        # Switch lifetime so validation passes when we later call validate()
        desc.lifetime = dtypes.AllocationLifetime.Explicit

        # Collect every state that contains an access node for this array
        all_states = [
            s for s in cfg_analysis.blockorder_topological_sort(sdfg, ignore_nonstate_blocks=True)
            if any(n.data == name for n in s.data_nodes())
        ]
        if not all_states:
            # Array is unused — lifetime is flipped but no edges to annotate
            continue

        # Determine the target scope region.
        # SDFG/Global lifetime: always allocate at the SDFG level regardless
        # of where the actual uses are.
        # State/Scope/other: use the LCA of actual uses to preserve the
        # per-iteration or per-state semantics of the original lifetime.
        if original_lifetime in (dtypes.AllocationLifetime.SDFG,
                                  dtypes.AllocationLifetime.Global):
            lca: ControlFlowRegion = sdfg
        else:
            lca = _lca_region(all_states, sdfg)

        # First/last direct children of lca that contain uses
        using = _blocks_using_in(lca, name)
        if not using:
            continue

        _alloc_on_incoming_edges(lca, using[0], name)
        _free_on_outgoing_edges(lca, using[-1], name)
