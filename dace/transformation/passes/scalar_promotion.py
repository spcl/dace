# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``PromoteScalarOutputsToArrays`` -- replace a written ``Scalar`` descriptor with a length-1 ``Array``.

Two callers want this for unrelated reasons, and only their CRITERIA differ:

* host (the default): a written non-transient scalar has no addressable spelling in either signature
  it can appear in. On an entry point ``Scalar.as_arg`` emits ``T name`` -- by value, so the result
  is computed into the callee's copy and discarded. On a nested SDFG ``cpp.emit_memlet_reference``
  binds it as ``T &name``, which C cannot spell at all.
* ``gpu=True``: device memory has no scalar form, so a GPU-storage scalar or a GPU kernel output
  must be widened, and the latter forced to ``GPU_Global``.

One pass with a flag rather than two, so the rewrite cannot drift between them. The subtle parts --
walking the hierarchy top-down so a parent's promotion is already visible at its children, pushing
the change through NestedSDFG connectors, and rewriting the state-machine slots that name a
descriptor as text rather than through a memlet -- are identical whatever the criteria are.
"""
from typing import Any, Callable, Dict, Optional, Set

from dace import data, dtypes, properties
from dace.sdfg import SDFG, SDFGState, infer_types, nodes
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.length_one_array_scalar_conversion import (descriptor_is_written, rewrite_code_slots,
                                                                           rewrite_refs_to_element)

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


def written_by_gpu_map_exit(sdfg: SDFG, name: str) -> bool:
    """Whether ``name`` is written across a GPU-scheduled map's ``MapExit``, i.e. is a kernel output.

    :param sdfg: SDFG to search.
    :param name: Descriptor name.
    :returns: True if some access node of ``name`` is written by a GPU-scheduled ``MapExit``.
    """
    for state in sdfg.states():
        for node in state.nodes():
            if not (isinstance(node, nodes.AccessNode) and node.data == name):
                continue
            for in_edge in state.in_edges(node):
                entry = state.entry_node(in_edge.src) if isinstance(in_edge.src, nodes.ExitNode) else None
                if entry is not None and entry.map.schedule in dtypes.GPU_SCHEDULES:
                    return True
    return False


@properties.make_properties
@transformation.explicit_cf_compatible
class PromoteScalarOutputsToArrays(ppl.Pass):
    """Replace every written ``Scalar`` the criteria accept with a length-1 ``Array``."""

    gpu = properties.Property(dtype=bool,
                              default=False,
                              desc="Use the GPU criteria: promote a GPU-storage scalar (keeping its storage) or a "
                              "scalar written by a GPU map exit (forcing GPU_Global), rather than a written "
                              "non-transient scalar with its storage left alone.")
    non_transient_only = properties.Property(dtype=bool,
                                             default=True,
                                             desc="GPU only: the kernel-output rule promotes non-transient scalars "
                                             "only. A transient scalar written by a GPU map exit stays a Scalar -- "
                                             "the host never observes the value, so it can live in registers / "
                                             "per-thread stack. Disable to promote every kernel-output scalar.")

    #: GPU only. Register-storage scalars are thread-local; widening would force a per-thread
    #: ``cudaMalloc`` inside the kernel body.
    _RULE2_EXEMPT_STORAGES = frozenset({dtypes.StorageType.Register})

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # A library expansion can introduce a fresh scalar connector, which re-arms the pass.
        return bool(modified & (ppl.Modifies.Descriptors | ppl.Modifies.Nodes))

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Promote every matching scalar across the SDFG hierarchy.

        :param sdfg: the outermost SDFG, modified in place.
        :param pipeline_results: unused.
        :returns: how many descriptors were promoted, or ``None`` if none were.
        """
        if self.gpu:
            # The GPU criteria read the final storage decision, which is only settled after inference.
            infer_types.set_default_schedule_and_storage_types(sdfg, None)
        promoted = promote_matching_scalars(sdfg, self.needs_promotion, self.storage_for)
        # Under the GPU criteria this is unconditional: a connector can be mistyped against an Array
        # inner descriptor this pass did not create (cuBLAS expansion's ``gpu_streams``), so the
        # reset is still needed when nothing was promoted here.
        if self.gpu or promoted:
            invalidate_array_connectors(sdfg)
        return promoted or None

    def needs_promotion(self, sdfg: SDFG, name: str) -> bool:
        """Whether ``name`` is a scalar its target cannot use as-is."""
        desc = sdfg.arrays[name]
        if not isinstance(desc, data.Scalar):
            return False
        if not self.gpu:
            return not desc.transient and descriptor_is_written(sdfg, name)

        # Rule 1: GPU storage is incompatible with Scalar.
        if desc.storage in (dtypes.StorageType.GPU_Global, dtypes.StorageType.GPU_Shared):
            return True

        # Rule 2: kernel output -- written by a GPU map's ``MapExit``.
        if desc.storage in self._RULE2_EXEMPT_STORAGES:
            return False
        if self.non_transient_only and desc.transient:
            return False
        return written_by_gpu_map_exit(sdfg, name)

    def storage_for(self, sdfg: SDFG, name: str) -> Optional[dtypes.StorageType]:
        """Storage for the new array: the scalar's own, or real device memory for a GPU kernel write."""
        if not self.gpu:
            return None
        storage = sdfg.arrays[name].storage
        if storage not in (dtypes.StorageType.GPU_Global, dtypes.StorageType.GPU_Shared):
            storage = dtypes.StorageType.GPU_Global
        return storage
