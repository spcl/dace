# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``PromoteOutputScalarsToArrays`` -- make every WRITTEN signature scalar addressable.

A ``Scalar`` that is not transient is part of a signature: the entry point's, or a NestedSDFG's
connector list. DaCe spells both by value or by C++ reference, and neither survives a translation
unit that has to be plain C:

* on the entry point, ``Scalar.as_arg`` emits ``T name`` -- a by-value parameter, so a written
  result is computed into the callee's own copy and the caller reads back what it passed in,
* on a nested SDFG, ``cpp.emit_memlet_reference`` binds a written scalar connector as ``T &name``,
  which C cannot spell at all.

Promoting the descriptor to a length-1 ``Array`` fixes both at once: the parameter becomes ``T *``
and the body indexes ``[0]``. This is the same rewrite ``PromoteGPUScalarsToArrays`` performs for
device memory, and it shares its machinery
(:mod:`dace.transformation.passes.scalar_promotion`); only the criteria differ, and unlike the GPU
rule the storage is left alone.

Read-only signature scalars are NOT promoted. By value is the right spelling for them, it costs no
indirection, and changing it would churn the signature of every SDFG that takes a scalar parameter.
"""
from typing import Any, Dict, Optional

from dace import data, properties
from dace.sdfg import SDFG
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.length_one_array_scalar_conversion import descriptor_is_written
from dace.transformation.passes.scalar_promotion import invalidate_array_connectors, promote_matching_scalars


@properties.make_properties
@transformation.explicit_cf_compatible
class PromoteOutputScalarsToArrays(ppl.Pass):
    """Replace every written non-transient ``Scalar`` with a length-1 ``Array``, storage preserved."""

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # A library expansion can introduce a fresh scalar connector, which re-arms the pass.
        return bool(modified & (ppl.Modifies.Descriptors | ppl.Modifies.Nodes))

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Promote every written signature scalar across the SDFG hierarchy.

        :param sdfg: the outermost SDFG, modified in place.
        :param pipeline_results: unused.
        :returns: number of scalars promoted, or ``None`` if nothing changed.
        """
        promoted = promote_matching_scalars(sdfg, self.needs_promotion)
        if promoted:
            invalidate_array_connectors(sdfg)
        return promoted or None

    def needs_promotion(self, sdfg: SDFG, name: str) -> bool:
        """Whether ``name`` is a written signature scalar. Storage is left alone."""
        desc = sdfg.arrays[name]
        if not isinstance(desc, data.Scalar) or desc.transient:
            return False
        return descriptor_is_written(sdfg, name)
