# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``NormalizeWCRSource`` — interpose a private scalar between a CodeNode and a WCR sink.

DaCe codegen emits a WCR write only when its *source* is an ``AccessNode``: at
:meth:`dace.codegen.targets.cpu.CPUCodeGen._dispatch_node` the WCR branch
fires for ``defined_type == DefinedType.Scalar`` / ``Pointer``, both of which
imply an AccessNode source. A WCR whose source is a :class:`~dace.nodes.NestedSDFG`
(or :class:`~dace.nodes.Tasklet`) output connector falls through the WCR
switch and emits a plain copy, silently dropping the reduction.

This pass walks every WCR edge in the SDFG and, whenever the source is a
CodeNode, inserts a private transient ``Scalar`` AccessNode between them:

``CodeNode -[wcr]-> Sink``  becomes  ``CodeNode -> scalar -[wcr]-> Sink``.

The plain edge into the scalar carries the value out of the CodeNode (no WCR),
and the new edge from the scalar to the original sink carries the WCR. The
sink can be a downstream :class:`~dace.nodes.AccessNode`, a
:class:`~dace.nodes.MapExit`, or anything else — only the WCR semantics need
to land on an AccessNode source for codegen to recognise it.

This is the foundation for vectorising reductions: with the source side
normalised, the per-lane reduction inside the NSDFG can lower to a tile
horizontal-reduce that writes the private scalar, and the outer
``scalar -[wcr]-> MapExit`` edge then lowers to a standard OpenMP reduction.
"""
from typing import Optional

import dace
from dace import properties
from dace.sdfg import nodes
from dace.transformation import pass_pipeline as ppl


@properties.make_properties
class NormalizeWCRSource(ppl.Pass):
    """Interpose a private scalar between every CodeNode source and its WCR sink.

    Idempotent: re-running the pass on an already-normalised SDFG is a no-op
    (every WCR source is already an AccessNode).
    """

    CATEGORY: str = "Vectorization"

    def modifies(self) -> ppl.Modifies:
        """Mark the modifications this pass makes.

        :returns: The set of SDFG facets this pass may modify.
        """
        return ppl.Modifies.States | ppl.Modifies.AccessNodes | ppl.Modifies.Memlets | ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        """Re-run only when nothing changed (the pass is idempotent).

        :param modified: Modifications reported by the prior pipeline pass.
        :returns: ``False`` — never re-run.
        """
        return False

    def apply_pass(self, sdfg: dace.SDFG, _) -> Optional[int]:
        """Walk every WCR edge and interpose a private scalar when its source
        is a CodeNode (``NestedSDFG`` / ``Tasklet``).

        :param sdfg: SDFG to transform in place.
        :param _: Unused pipeline results.
        :returns: Number of edges rewritten, or ``None`` if none.
        """
        rewritten = 0
        for state in list(sdfg.all_states()):
            for edge in list(state.edges()):
                memlet = edge.data
                if memlet is None or memlet.wcr is None:
                    continue
                if not isinstance(edge.src, nodes.CodeNode):
                    continue
                # The sink array determines the scalar's dtype.
                sink_arr = state.sdfg.arrays.get(memlet.data)
                if sink_arr is None:
                    continue
                dtype = sink_arr.dtype
                local_sdfg = state.sdfg
                scalar_name, _ = local_sdfg.add_scalar(
                    f"_wcr_src_{edge.src.label if hasattr(edge.src, 'label') else 'codenode'}",
                    dtype,
                    transient=True,
                    find_new_name=True,
                )
                bridge = state.add_access(scalar_name)
                # Edge CodeNode -> bridge: plain copy of the produced value.
                state.add_edge(edge.src, edge.src_conn, bridge, None,
                               dace.Memlet(data=scalar_name, subset="0"))
                # Edge bridge -> original sink: WCR survives here, source is now
                # an AccessNode so cpu codegen's WCR switch fires.
                wcr_memlet = dace.Memlet(data=memlet.data, subset=memlet.subset, wcr=memlet.wcr)
                if memlet.other_subset is not None:
                    wcr_memlet.other_subset = memlet.other_subset
                state.add_edge(bridge, None, edge.dst, edge.dst_conn, wcr_memlet)
                state.remove_edge(edge)
                rewritten += 1
        return rewritten or None
