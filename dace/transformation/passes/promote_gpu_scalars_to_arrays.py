# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``PromoteGPUScalarsToArrays`` -- replace GPU-incompatible ``Scalar``
descriptors with length-1 ``Array`` descriptors (after storage/schedule
inference; depends on ``InferDefaultSchedulesAndStorages``).

Two rules: (1) a ``Scalar`` with ``GPU_Global``/``GPU_Shared`` storage keeps
its storage and is widened to length-1; (2) a ``Scalar`` written inside a
``GPU_Device`` kernel is widened and forced to ``GPU_Global`` (``Register``
is exempt -- thread-local stack). Memlets are rewritten via
``Memlet.from_array``, bare-identifier interstate assignments get a ``[0]``
subscript, and nested SDFGs re-declaring the name are promoted recursively.
"""
from typing import Any, Dict, Optional

from dace import data, dtypes, properties
from dace.memlet import Memlet
from dace.sdfg import SDFG, infer_types, nodes
from dace.sdfg.scope import is_devicelevel_gpu
from dace.transformation import pass_pipeline as ppl, transformation


def invalidate_array_connectors(sdfg: SDFG):
    """Reset NestedSDFG connectors whose inner descriptor is an ``Array`` to
    ``typeclass(None)`` so a follow-up ``infer_connector_types`` re-derives
    them as pointer-typed. Needed because a connector typed at construction
    time as a scalar dtype against an Array inner descriptor produces a
    wrapper signature ``T name`` that the body indexes ``name[0]`` (compile
    error). Common cause: cuBLAS expansion's ``gpu_streams`` connector."""
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


@properties.make_properties
@transformation.explicit_cf_compatible
class InferDefaultSchedulesAndStorages(ppl.Pass):
    """Pipeline-shaped wrapper around
    :func:`dace.sdfg.infer_types.set_default_schedule_and_storage_types`.

    The function itself is the actual implementation -- this class exists
    so the call can participate in a ``Pipeline`` with a real
    ``depends_on`` edge from later passes. ``PromoteGPUScalarsToArrays``
    in particular relies on every descriptor having a final, non-default
    storage decision, which is exactly what this pass establishes.
    """

    def modifies(self) -> ppl.Modifies:
        # Storage and schedule attributes live on descriptors and on
        # ``Map`` instances respectively; both are reachable through
        # ``Modifies.Descriptors | Modifies.Nodes``.
        return ppl.Modifies.Descriptors | ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        infer_types.set_default_schedule_and_storage_types(sdfg, None)
        return None


@properties.make_properties
@transformation.explicit_cf_compatible
class PromoteGPUScalarsToArrays(ppl.Pass):
    """Replace GPU-incompatible ``Scalar`` descriptors with length-1 Arrays."""

    # Register-storage scalars are thread-local; widening would force
    # per-thread ``cudaMalloc`` inside the kernel body.
    _RULE2_EXEMPT_STORAGES = frozenset({dtypes.StorageType.Register})

    def depends_on(self):
        return {InferDefaultSchedulesAndStorages}

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # Adding new GPU-storage Scalars (e.g. via library expansion) re-arms
        # the pass; harmless when nothing matches.
        return bool(modified & (ppl.Modifies.Descriptors | ppl.Modifies.Nodes))

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Return the number of scalars promoted across the SDFG hierarchy,
        or ``None`` if nothing changed."""
        promoted = 0
        # Top-down so a parent's promotion is visible when we visit the
        # child's matching descriptor (children inherit the parent's choice
        # -- see ``_promote_one`` for the recursion into nested SDFGs).
        for nsdfg in list(sdfg.all_sdfgs_recursive()):
            for name in list(nsdfg.arrays):
                if not self._needs_promotion(nsdfg, name):
                    continue
                self._promote_one(nsdfg, name)
                promoted += 1

        # Reset NestedSDFG connectors whose inner descriptor became an Array
        # so ``infer_connector_types`` re-derives them as pointer-typed.
        invalidate_array_connectors(sdfg)

        return promoted if promoted > 0 else None

    def _needs_promotion(self, sdfg: SDFG, name: str) -> bool:
        desc = sdfg.arrays[name]
        if not isinstance(desc, data.Scalar):
            return False

        # Rule 1: GPU storage is incompatible with Scalar.
        if desc.storage in (dtypes.StorageType.GPU_Global, dtypes.StorageType.GPU_Shared):
            return True

        # Rule 2: written-to from inside a GPU_Device kernel scope.
        if desc.storage in self._RULE2_EXEMPT_STORAGES:
            return False
        for state in sdfg.states():
            for node in state.nodes():
                if not (isinstance(node, nodes.AccessNode) and node.data == name):
                    continue
                if state.in_degree(node) == 0:
                    continue  # not a write target
                if is_devicelevel_gpu(sdfg, state, node):
                    return True
        return False

    def _promote_one(self, sdfg: SDFG, name: str):
        """Replace ``sdfg.arrays[name]`` (a Scalar) with a length-1 Array,
        rewrite memlets referencing it, and recurse into nested SDFGs that
        re-declare the same name as a Scalar."""
        scalar_desc: data.Scalar = sdfg.arrays[name]

        # Rule 2 promotes Default / CPU-side scalars to GPU_Global because
        # the kernel write needs real device memory; rule 1 keeps the
        # pre-existing GPU storage.
        target_storage = scalar_desc.storage
        if target_storage not in (dtypes.StorageType.GPU_Global, dtypes.StorageType.GPU_Shared):
            target_storage = dtypes.StorageType.GPU_Global

        array_desc = data.Array(
            dtype=scalar_desc.dtype,
            shape=(1, ),
            transient=scalar_desc.transient,
            storage=target_storage,
            location=scalar_desc.location,
            strides=(1, ),
            lifetime=scalar_desc.lifetime,
            allow_conflicts=scalar_desc.allow_conflicts,
            debuginfo=scalar_desc.debuginfo,
        )

        sdfg.remove_data(name, validate=False)
        sdfg.add_datadesc(name, array_desc)

        for state in sdfg.states():
            for edge in state.edges():
                if edge.data is not None and edge.data.data == name:
                    new_memlet = Memlet.from_array(dataname=name, datadesc=array_desc)
                    new_memlet.dynamic = edge.data.dynamic
                    new_memlet.wcr = edge.data.wcr
                    edge.data = new_memlet

        # Interstate edge assignments referencing the promoted name as a
        # bare identifier (e.g. the frontend's ``__sym_X = X`` symbol-promotion
        # assignment for indirect indexing) must be rewritten to subscript
        # the new length-1 array (``__sym_X = X[0]``) -- otherwise the codegen
        # emits ``int = const int*``.
        self._rewrite_interstate_assignments(sdfg, name)

        # Recurse into nested SDFGs that share the name as a Scalar.
        # Connector invalidation happens once at the end of ``apply_pass``
        # over the full hierarchy.
        for state in sdfg.states():
            for node in state.nodes():
                if (isinstance(node, nodes.NestedSDFG) and name in node.sdfg.arrays
                        and isinstance(node.sdfg.arrays[name], data.Scalar)):
                    self._promote_one(node.sdfg, name)

    @staticmethod
    def _rewrite_interstate_assignments(sdfg: SDFG, name: str):
        """Replace bare-identifier references to ``name`` in this SDFG's
        interstate-edge assignment expressions with ``name[0]`` so that
        post-promotion code reads the length-1 Array element rather than
        treating the array pointer as a scalar value."""
        import re as _re
        # Word-boundary regex; subscripted (``name[``) and dotted (``.name``)
        # references are intentionally skipped.
        pattern = _re.compile(rf'(?<![\w.])({_re.escape(name)})(?!\s*\[)\b')
        for cfg in sdfg.all_control_flow_regions():
            for edge in cfg.edges():
                ise = edge.data
                if ise is None or not getattr(ise, 'assignments', None):
                    continue
                for k, v in list(ise.assignments.items()):
                    if not isinstance(v, str):
                        continue
                    new_v = pattern.sub(rf'\1[0]', v)
                    if new_v != v:
                        ise.assignments[k] = new_v
