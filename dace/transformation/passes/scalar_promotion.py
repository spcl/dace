# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The mechanism for replacing a ``Scalar`` descriptor with a length-1 ``Array``.

Mechanism only: nothing here decides WHICH scalars to promote. Two passes supply that separately
and for unrelated reasons -- ``PromoteGPUScalarsToArrays`` because device memory has no scalar form,
``PromoteOutputScalarsToArrays`` because a written signature scalar is passed by value (entry point)
or by C++ reference (nested SDFG connector), and neither returns a result a C caller can read.

Keeping the rewrite in one place is what stops the two from drifting. The subtle parts -- walking the
hierarchy top-down so a parent's promotion is already visible when its children are visited, pushing
the change through NestedSDFG connectors, and rewriting the state-machine slots that name a
descriptor as text rather than through a memlet -- are identical whatever the criteria are.
"""
from typing import Callable, Optional, Set

from dace import data, dtypes
from dace.sdfg import SDFG, SDFGState, nodes
from dace.transformation.passes.length_one_array_scalar_conversion import (rewrite_code_slots, rewrite_refs_to_element)

#: Decides whether ``sdfg.arrays[name]`` is promoted.
PromotionRule = Callable[[SDFG, str], bool]

#: Picks the storage of the array replacing ``sdfg.arrays[name]``; ``None`` keeps the scalar's own.
StorageRule = Callable[[SDFG, str], Optional[dtypes.StorageType]]


def invalidate_array_connectors(sdfg: SDFG):
    """Reset NestedSDFG connectors whose inner descriptor is an ``Array`` to
    ``typeclass(None)`` so a follow-up ``infer_connector_types`` re-derives
    them as pointer-typed.

    A connector typed at construction time as a scalar dtype against an
    ``Array`` inner descriptor produces a wrapper signature ``T name`` that the
    body indexes ``name[0]`` (compile error). Common cause: cuBLAS expansion's
    ``gpu_streams`` connector.
    """
    uninferred = dtypes.typeclass(None)
    for nsdfg in sdfg.all_sdfgs_recursive():
        for state in nsdfg.states():
            for node in state.nodes():
                if not isinstance(node, nodes.NestedSDFG):
                    continue
                for cname in list(node.in_connectors):
                    if cname in node.sdfg.arrays and isinstance(node.sdfg.arrays[cname], data.Array):
                        node.in_connectors[cname] = uninferred
                for cname in list(node.out_connectors):
                    if cname in node.sdfg.arrays and isinstance(node.sdfg.arrays[cname], data.Array):
                        node.out_connectors[cname] = uninferred


def promote_scalar_to_array(sdfg: SDFG, name: str, storage: Optional[dtypes.StorageType] = None) -> None:
    """Replace the ``Scalar`` ``name`` with a length-1 ``Array``, in place, across the hierarchy.

    The descriptor is swapped, bare textual references become ``name[0]``, and every nested SDFG
    that re-declares the same name as a ``Scalar`` behind a connector is promoted with it. Memlets
    are left alone: a ``Scalar`` access already carries subset ``[0]``, identical to a length-1
    array's.

    Module-level rather than a method because the MECHANISM is the same wherever a scalar has to
    become addressable, and only the CRITERIA differ. GPU promotion (below) needs it because device
    memory has no scalar form; MPR's C rendering needs it because a written scalar reaches a nested
    function as a C++ reference, which C cannot spell.

    :param sdfg: the SDFG declaring ``name``.
    :param name: the ``Scalar`` descriptor to promote.
    :param storage: storage for the new array; ``None`` keeps the scalar's own.
    """
    scalar_desc: data.Scalar = sdfg.arrays[name]
    array_desc = data.Array(
        dtype=scalar_desc.dtype,
        shape=(1, ),
        transient=scalar_desc.transient,
        storage=scalar_desc.storage if storage is None else storage,
        location=scalar_desc.location,
        strides=(1, ),
        lifetime=scalar_desc.lifetime,
        allow_conflicts=scalar_desc.allow_conflicts,
        debuginfo=scalar_desc.debuginfo,
    )

    sdfg.remove_data(name, validate=False)
    sdfg.add_datadesc(name, array_desc)

    # Shared with the general Scalar <-> length-1 Array passes: the state-machine slots that name a
    # descriptor textually, and the bare-reference -> ``name[0]`` transform, are the same rewrite.
    rewrite_code_slots(sdfg, lambda text: rewrite_refs_to_element(text, {name: name}))
    for state in sdfg.states():
        push_promotion_into_nested(state, name, storage)


def push_promotion_into_nested(state: SDFGState, name: str, storage: Optional[dtypes.StorageType] = None) -> None:
    """Promote the inner descriptor of every NestedSDFG in ``state`` that binds ``name`` as a Scalar.

    ``symbol_mapping`` values are handled by the shared ``rewrite_code_slots`` walk in
    :func:`promote_scalar_to_array`, not here.

    :param state: the state whose NestedSDFG nodes are visited.
    :param name: the OUTER descriptor name that was promoted.
    :param storage: storage for the new inner array; ``None`` keeps the inner scalar's own.
    """
    for node in state.nodes():
        if not isinstance(node, nodes.NestedSDFG):
            continue

        handled_inner_names: Set[str] = set()  # If data is referenced as input and output.
        for iedge in state.in_edges(node):
            if iedge.data.is_empty():
                continue
            inner_name = iedge.dst_conn
            if iedge.data.data == name and isinstance(node.sdfg.arrays[inner_name], data.Scalar):
                assert inner_name not in handled_inner_names  # Can only appear once.
                promote_scalar_to_array(node.sdfg, inner_name, storage)
                handled_inner_names.add(inner_name)

        for oedge in state.out_edges(node):
            if oedge.data.is_empty():
                continue
            inner_name = oedge.src_conn
            if oedge.data.data == name and inner_name not in handled_inner_names and isinstance(
                    node.sdfg.arrays[inner_name], data.Scalar):
                promote_scalar_to_array(node.sdfg, inner_name, storage)


def promote_matching_scalars(sdfg: SDFG,
                             needs_promotion: PromotionRule,
                             storage_for: Optional[StorageRule] = None) -> int:
    """Promote every scalar in ``sdfg``'s hierarchy that ``needs_promotion`` accepts.

    Does NOT call :func:`invalidate_array_connectors`; whether a pass needs it unconditionally or
    only when it promoted something is a property of that pass, not of the walk.

    :param sdfg: the outermost SDFG, modified in place.
    :param needs_promotion: the criteria.
    :param storage_for: storage for each new array; ``None`` keeps every scalar's own.
    :returns: how many descriptors were promoted.
    """
    promoted = 0
    # Top-down, so a parent's promotion is already visible at the child's matching descriptor and a
    # connector carried along with its parent is not reached again as an independent candidate.
    for nsdfg in list(sdfg.all_sdfgs_recursive()):
        for name in list(nsdfg.arrays):
            if not needs_promotion(nsdfg, name):
                continue
            promote_scalar_to_array(nsdfg, name, None if storage_for is None else storage_for(nsdfg, name))
            promoted += 1
    return promoted
