# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Shared helpers for CopyLibraryNode and MemsetLibraryNode expansions.
"""
from typing import Callable, List, Tuple

import dace
from dace.sdfg import nodes

# The ambient GPU stream symbol libnode CUDA expansions reference. There is
# no pre-existing global constant for it -- the legacy codegen hardcodes the
# bare literal -- so this is the canonical declaration. The name keeps the
# same expanded IR valid under both the legacy codegen (which declares
# ``__dace_current_stream``) and the experimental codegen (whose type-based
# prelude binds it once the stream scheduler wires the connector).
CURRENT_STREAM_NAME = "__dace_current_stream"


def collapse_shape_and_strides(
        subset: dace.subsets.Range,
        strides: List[dace.symbolic.SymExpr]) -> Tuple[List[dace.symbolic.SymExpr], List[dace.symbolic.SymExpr]]:
    """Drop length-1 dimensions from a (subset, strides) pair.

    Surviving strides are scaled by the subset step (``stride * s``) so they describe the access
    pattern as a view into the parent array -- a no-op for unit-step subsets, and the effective
    per-element distance for strided ones.

    :param subset: The access range, one ``(begin, end, step)`` per dimension.
    :param strides: The parent array strides, aligned with ``subset``.
    :returns: ``(collapsed_shape, collapsed_strides)`` with singletons removed.
    """
    collapsed_shape = []
    collapsed_strides = []
    for (b, e, s), stride in zip(subset, strides):
        length = (e + 1 - b) // s
        if length != 1:
            collapsed_shape.append(length)
            collapsed_strides.append(stride * s)
    return collapsed_shape, collapsed_strides


def collapsed_map_lengths(subset: dace.subsets.Range) -> List[dace.symbolic.SymExpr]:
    """Per-dim element counts of ``subset`` with length-1 dims removed.

    Mirrors :func:`collapse_shape_and_strides` on the shape side -- expansions
    use this to size the mapped loop without carrying strides through.
    """
    return [ml for ml in ((e + 1 - b) // s for (b, e, s) in subset) if ml != 1]


def auto_dispatch(node: nodes.LibraryNode, parent_state: dace.SDFGState, parent_sdfg: dace.SDFG,
                  select_fn: Callable[[nodes.LibraryNode, dace.SDFGState, dace.SDFG], str], library_cls: type):
    """Dispatch a library node's ``'Auto'`` implementation to the one picked by ``select_fn``.

    Sets ``node.implementation`` to the resolved name so introspection
    (debug output, downstream passes) reflects what was actually picked.

    :param node: the library node being expanded.
    :param parent_state: state containing ``node``.
    :param parent_sdfg: SDFG containing ``parent_state``.
    :param select_fn: callable returning a concrete implementation name (not ``'Auto'``).
    :param library_cls: the library node class with the ``implementations`` dict.
    :returns: whatever the resolved expansion returns.
    """
    impl_name = select_fn(node, parent_state, parent_sdfg)
    assert impl_name != 'Auto', f"{select_fn.__name__} must not return 'Auto'."
    node.implementation = impl_name
    return library_cls.implementations[impl_name].expansion(node, parent_state, parent_sdfg)
