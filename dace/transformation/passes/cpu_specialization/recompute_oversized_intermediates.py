# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Recompute a producer map inside its consumer when the intermediate outgrows the cache.

:func:`~dace.transformation.passes.canonicalize.finalize.recompute_fuse_for_gpu` collapses
producer->consumer map chains with ``OTFMapFusion`` and deletes the intermediate, and states why
the CPU deliberately does not: there the intermediates are cache-resident and shared across the
consumer maps, so materializing once and reading beats recomputing.

That argument holds exactly while the intermediate IS cache-resident. A 3-D stencil half-step
buffer -- heat3d's ``B``, jacobi's, the split-statement temporaries a slice-vectorized stencil
leaves behind -- is the size of the grid, so materializing it costs a full DRAM write plus a full
DRAM read that recomputation never pays: ``2 * |T|`` of traffic on a sweep that is bandwidth-bound
to begin with. The producer's own inputs are read either way, and the consumer streams them
anyway, so recomputing spends ALU, not bandwidth.

The CPU rule is therefore the size test the GPU never needs -- fuse when the intermediate provably
does not fit in the host's last-level cache, keep the materialized form when it does. Both halves
of the gate stay conservative:

* a symbolic size counts as oversized (house rule: an unknown extent is assumed big), because the
  materialized form is the one that cannot be recovered once it reaches codegen;
* only a SINGLE-consumer intermediate is fused. Two consumer maps reading one buffer is precisely
  the sharing a cache is good at, and fusing into both re-reads the producer's inputs once per
  consumer -- a traffic question this pass has no model for, so it declines rather than guesses.

A DEVICE SPECIALIZATION, not a canonical form: which side of the recompute/materialize trade wins
is a property of the host's cache, so this runs in the CPU lowering band and the canonical graph
keeps its materialized intermediates.
"""
from typing import Any, Dict, Optional

from dace import SDFG, data, properties, symbolic
from dace.sdfg import nodes
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation.dataflow import OTFMapFusion
from dace.transformation.passes.cpu_specialization.machine import topology


def intermediate_outgrows_cache(sdfg: SDFG, name: str) -> bool:
    """Whether materializing ``name`` provably costs DRAM traffic on this host.

    :param sdfg: The SDFG owning the descriptor.
    :param name: The intermediate transient's name.
    :returns: True when the array is larger than the host's last level cache, or symbolically sized.
    """
    desc = sdfg.arrays[name]
    if not isinstance(desc, data.Array):
        return False
    size = desc.total_size * desc.dtype.bytes
    if symbolic.issymbolic(size):
        return True
    return int(size) > topology().llc_bytes


@properties.make_properties
class OversizedIntermediateOTFFusion(OTFMapFusion):
    """``OTFMapFusion`` restricted to a single-consumer intermediate that outgrows the host's LLC."""

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        if not super().can_be_applied(graph, expr_index, sdfg, permissive):
            return False
        # Consumers, not edges: a stencil reads one intermediate at several offsets through
        # several edges into the SAME map entry, and that is still one consumer.
        if len({e.dst for e in graph.out_edges(self.array)}) != 1:
            return False
        # Counted over the WHOLE SDFG, not this state. ``graph`` is one state, and an intermediate
        # read from several states -- stencil_3d's ``padded``, read at six offsets per tap inside
        # the radius loop -- looks single-consumer in every one of them. Fusing then folds the
        # producer into one reader and leaves the others on a buffer nothing writes: measured on
        # stencil_3d, the surviving access node came out ``in=0, out=6`` and the kernel returned
        # uninitialized memory. The SDFG still validates, so nothing downstream catches it.
        readers = sum(
            1 for state in sdfg.states() for node in state.nodes()
            if isinstance(node, nodes.AccessNode) and node.data == self.array.data and state.out_degree(node) > 0)
        if readers != 1:
            return False
        return intermediate_outgrows_cache(sdfg, self.array.data)

    def apply(self, graph: SDFGState, sdfg: SDFG):
        """Fuse, then drop the intermediate's descriptor once nothing reads it.

        ``OTFMapFusion`` removes the intermediate's access node but leaves its descriptor in
        ``sdfg.arrays``, which is the right split for a dataflow transformation and the wrong end
        state for this one: an array no node touches is still allocated at codegen, so the
        materialization traffic this pass exists to avoid would be paid in full anyway.
        """
        name = self.array.data
        super().apply(graph, sdfg)
        if any(node.data == name for state in sdfg.states() for node in state.nodes()
               if isinstance(node, nodes.AccessNode)):
            return
        sdfg.remove_data(name, validate=False)


@properties.make_properties
class RecomputeOversizedIntermediates(ppl.Pass):
    """Fuse every producer map whose intermediate does not fit in the host's last-level cache."""

    CATEGORY: str = 'CPU Specialization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Memlets | ppl.Modifies.Descriptors

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Apply to fixpoint; returns how many chains were collapsed, or ``None``."""
        applied = sdfg.apply_transformations_repeated(OversizedIntermediateOTFFusion, validate_all=False)
        return applied or None
