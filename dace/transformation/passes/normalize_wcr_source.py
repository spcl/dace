# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Pass that ensures every WCR-bearing edge sources from an :class:`AccessNode`.

The DaCe CPU codegen's WCR-resolution path emits the atomic reduction inline only
for *scalar-typed* output connectors (the typical Tasklet output shape). A
``NestedSDFG`` output is a *pointer*-typed connector, so the WCR if-branch in
:func:`dace.codegen.targets.cpu.CPUCodeGen.process_out_memlets` falls through and
emits nothing: the parallel result is wrong (approximately the last thread's
value, not the running reduction). The OMP ``reduction(...)`` clause analyser
(:func:`_collect_omp_reductions`) similarly assumes a canonical
``AccessNode -[wcr]-> MapExit`` shape upstream of the exit.

The cleanest fix without touching core codegen is to maintain the invariant that
WCR edges *always* have an :class:`AccessNode` source. This pass walks every
state in the SDFG (and every nested SDFG, recursively), finds WCR edges whose
source is a :class:`Tasklet` or :class:`NestedSDFG`, and rewrites them by
inserting a per-iteration private transient :class:`AccessNode` between the
producer and the consumer. The original WCR moves onto the new
``AccessNode -> consumer`` edge; the producer-to-AccessNode edge becomes a plain
write with no WCR.

After the pass:

* Producer (Tasklet / NestedSDFG) ``-[no wcr, memlet=_priv]->`` ``AccessNode(_priv)``
* ``AccessNode(_priv) -[wcr=op, memlet=target]->`` consumer (typically a
  :class:`MapExit`)

The downstream codegen recognises the canonical ``AccessNode``-sourced shape and
emits the correct reduction/atomic.

The intermediate is allocated as a Scope-lifetime transient (per-Map-iteration)
matching the producer's output descriptor (dtype, shape, dtype-class). For
producers whose output connector is rank-0 / length-1, the transient is a
:class:`~dace.data.Scalar`; otherwise it is a same-shape :class:`~dace.data.Array`.

The pass is idempotent: re-running it finds no WCR edges sourced from CodeNodes
(only AccessNode-sourced WCR survives), so a fixed-point pipeline terminates
after a single iteration.
"""
import copy
from typing import Any, Dict, Optional, Set

from dace import SDFG, SDFGState, data, dtypes
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.transformation import pass_pipeline as ppl, transformation


@transformation.explicit_cf_compatible
class NormalizeWCRSource(ppl.Pass):
    """Insert an intermediate :class:`AccessNode` between every WCR-source CodeNode and its
    consumer so that all WCR-bearing edges originate at an AccessNode.
    """

    CATEGORY: str = "Simplification"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Memlets | ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & (ppl.Modifies.Nodes | ppl.Modifies.Memlets))

    def depends_on(self) -> Set:
        return set()

    def _output_descriptor(self, src: nodes.CodeNode, src_conn: str,
                           target_desc: Optional[data.Data]) -> Optional[data.Data]:
        """Return the data descriptor for the new private transient.

        For a :class:`NestedSDFG`, this is the inner array bound to the output connector
        (its shape + dtype are already determined by the body's IR). For a :class:`Tasklet`,
        the connector's typeclass is intentionally not consulted -- per the project rule
        "out connectors stay uninferred" -- so the descriptor is a :class:`Scalar` of the
        WCR target's element type. The value flowing through the WCR is dtype-equal to
        the accumulator slot, so this is correct for any single-element WCR write.
        """
        if isinstance(src, nodes.NestedSDFG):
            return src.sdfg.arrays.get(src_conn)
        if isinstance(src, nodes.Tasklet) and target_desc is not None:
            return data.Scalar(target_desc.dtype)
        return None

    def _make_private_desc(self, inner: data.Data) -> data.Data:
        """Build a Scope-lifetime transient descriptor matching ``inner`` element-wise.

        Length-1 / rank-0 producers get a :class:`Scalar`; multi-element producers get a
        same-shape :class:`Array`. Either way, the descriptor is fresh (deep-copied) and
        forced to ``transient=True`` + Scope lifetime so the codegen allocates it inside
        the Map body.
        """
        is_scalar = (isinstance(inner, data.Scalar) or (isinstance(inner, data.Array) and tuple(inner.shape) == (1, )))
        if is_scalar:
            return data.Scalar(inner.dtype,
                               transient=True,
                               storage=dtypes.StorageType.Default,
                               lifetime=dtypes.AllocationLifetime.Scope)
        new = copy.deepcopy(inner)
        new.transient = True
        new.storage = dtypes.StorageType.Default
        new.lifetime = dtypes.AllocationLifetime.Scope
        return new

    def _priv_subset(self, desc: data.Data) -> str:
        """Memlet subset string covering the entire private buffer."""
        if isinstance(desc, data.Scalar):
            return '0'
        return ', '.join(f'0:{s}' for s in desc.shape)

    def _rewrite_state(self, sdfg: SDFG, state: SDFGState) -> int:
        """Rewrite WCR edges whose source is a Tasklet or NestedSDFG; returns count."""
        rewritten = 0
        # Snapshot first; we mutate the edge set inside the loop.
        targets = [
            e for e in state.edges()
            if e.data is not None and e.data.wcr is not None and isinstance(e.src, (nodes.Tasklet, nodes.NestedSDFG))
        ]
        for e in targets:
            src = e.src
            src_conn = e.src_conn
            if src_conn is None:
                continue
            # InOut connector on a NestedSDFG: splitting only the OUT side onto
            # ``_wcr_priv_<src>_<conn>`` would break the InOut invariant
            # (validation rejects an in/out pair on the same connector name
            # pointing at two different external arrays). Skip the rewrite for
            # such edges -- the WCR stays on the direct NestedSDFG-output edge
            # and codegen falls back to its atomic-add path.
            if isinstance(src, nodes.NestedSDFG) and src_conn in src.in_connectors:
                continue
            target_desc = sdfg.arrays.get(e.data.data) if e.data.data else None
            inner = self._output_descriptor(src, src_conn, target_desc)
            if inner is None:
                continue
            priv_desc = self._make_private_desc(inner)
            priv_name = sdfg.add_datadesc(f'_wcr_priv_{src.label}_{src_conn}', priv_desc, find_new_name=True)
            priv_node = state.add_access(priv_name)

            state.add_edge(src, src_conn, priv_node, None, Memlet(data=priv_name, subset=self._priv_subset(priv_desc)))
            state.add_edge(priv_node, None, e.dst, e.dst_conn, copy.deepcopy(e.data))
            state.remove_edge(e)
            rewritten += 1
        return rewritten

    def _apply(self, sdfg: SDFG) -> int:
        total = 0
        for sd in sdfg.all_sdfgs_recursive():
            for state in sd.all_states():
                total += self._rewrite_state(sd, state)
        return total

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Set[str]]]:
        """Rewrite every WCR-bearing edge so its source is an :class:`AccessNode`.

        :param sdfg: The SDFG to normalize.
        :param pipeline_results: Results of prior passes in the pipeline (unused).
        :returns: ``None`` if no edges were rewritten; otherwise a single-entry dict
                  with the rewritten count under key ``normalized_wcr_edges``.
        """
        n = self._apply(sdfg)
        if n == 0:
            return None
        sdfg.validate()
        return {'normalized_wcr_edges': {str(n)}}
