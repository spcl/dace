# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Bypass trivial ``AN -> [_out=_in] -> AN`` assign tasklets.

A trivial assign tasklet whose body is exactly ``_out = _in`` (one input
connector, one output connector) and whose only incoming / outgoing edge
each connects to an :class:`~dace.sdfg.nodes.AccessNode` is a pure copy.
DaCe's tasklet codegen for a Python-language ``_out = _in`` body with
tile-pointer connectors emits ``_out = _in;`` — a *pointer* reassignment
of the local variable; the destination transient is never actually
written. This pass exposes the rewrite as a standalone pipeline step so the
multi-dim K=1 / K=2 paths can call it directly.

Two rewrites, in order:

1. **Dedup** — when several trivial assign tasklets carry the same
   ``(src.data, dst.data)`` pair (e.g. ``fp_factor`` branch lowering
   emitting one cond-to-merge chain per arm side-by-side), collapse
   them to ONE. Without this step the source's ``out_degree`` would
   exceed 1 and the bypass below would refuse it.
2. **Bypass** — when at least one side is a transient AND
   ``out_degree(src) == 1`` AND ``in_degree(dst) == 1``, drop the
   tasklet and route the producer / consumer of the transient side
   directly. The single-consumer / single-producer guard keeps
   SSA-like reassignment chains intact (``c1 = c[i]; ...; c1 =
   c1*d1*e1 + ...`` -- bypassing would fold two assignments onto one
   AccessNode and pick up the wrong value).

The pass is body-NSDFG-scoped: the outer SDFG's ``AN -> AN`` edges may
be scatter / gather staging, so they stay untouched. Mirrors
:class:`EliminateDeadCopies`'s scoping.
"""
from typing import Any, Dict, Optional, Tuple

import dace
from dace import subsets
from dace.sdfg import SDFG
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.vectorization.utils.pass_invariants import (assert_invariant,
                                                                            no_duplicate_connector_edges,
                                                                            no_memlet_dim_mismatch)
# _is_assign_tasklet was previously imported from emit_tile_ops (deleted in the walker-primary
# migration). The matcher is inlined below.


def _is_assign_tasklet(t) -> bool:
    """True iff ``t`` is a tasklet with a single in / out connector and a body
    of the form ``<out_conn> = <in_conn>`` (no arithmetic, no calls).
    """
    if not hasattr(t, "code") or not hasattr(t, "in_connectors") or not hasattr(t, "out_connectors"):
        return False
    if len(t.in_connectors) != 1 or len(t.out_connectors) != 1:
        return False
    in_conn = next(iter(t.in_connectors))
    out_conn = next(iter(t.out_connectors))
    body = (t.code.as_string if hasattr(t.code, "as_string") else str(t.code)).strip().rstrip(";")
    return body == f"{out_conn} = {in_conn}"


def _assign_triple(istate: SDFGState, t: dace.nodes.Tasklet) -> Optional[Tuple]:
    """Return ``(in_edge, out_edge)`` iff ``t`` is the trivial
    ``AN -> [_out=_in] -> AN`` triple. ``None`` otherwise.

    Shared gate for the dedup and bypass passes -- collapses six
    repeated checks (assign body, single in/out edge, both endpoints
    AccessNodes) into one helper.
    """
    if not _is_assign_tasklet(t):
        return None
    in_es = istate.in_edges(t)
    out_es = istate.out_edges(t)
    if len(in_es) != 1 or len(out_es) != 1:
        return None
    in_e, out_e = in_es[0], out_es[0]
    if not (isinstance(in_e.src, dace.nodes.AccessNode) and isinstance(out_e.dst, dace.nodes.AccessNode)):
        return None
    return in_e, out_e


def _accessed_in_other_states(inner_sdfg: SDFG, data_name: str, current_state: SDFGState) -> bool:
    """True iff ``data_name`` has an AccessNode in some state OTHER than ``current_state``.

    A transient that is also accessed in another state is a **cross-state value**:
    its producer (or consumer) lives in a different state and the data flows
    through the persistent transient, NOT through an edge in ``current_state``.
    Its in/out degree *within ``current_state``* is therefore misleading -- a
    write whose only reader is in the NEXT state shows ``out_degree == 0`` here,
    yet it is NOT dead. The state-local bypass / dedup rewrites must leave such
    triples alone, else they delete the sole write (or read) and either produce
    an isolated node (invalid SDFG) or silently break the cross-state data flow.
    (cloudsc_one: ``zqx[z1,i,j]`` staged in ``assign_42_12`` and read by the
    cond1 guard in the next state ``slice_zqx_43_0``.)
    """
    for st in inner_sdfg.states():
        if st is current_state:
            continue
        for n in st.data_nodes():
            if n.data == data_name:
                return True
    return False


@transformation.explicit_cf_compatible
class BypassTrivialAssignTasklets(ppl.Pass):
    """Dedup + bypass ``AN -> [_out=_in] -> AN`` triples in body NSDFGs."""

    CATEGORY: str = "Vectorization"

    def modifies(self) -> ppl.Modifies:
        """Drops tasklets / access nodes, rewires memlets."""
        return ppl.Modifies.AccessNodes | ppl.Modifies.Memlets | ppl.Modifies.Tasklets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        """Single fixed-point sweep is enough."""
        return False

    def depends_on(self):
        """Standalone pass."""
        return set()

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Sweep every body NSDFG and apply dedup + bypass.

        :param sdfg: Top-level SDFG.
        :param pipeline_results: Unused.
        :returns: Number of tasklets removed across the SDFG, or ``None`` if zero.
        """
        total = 0
        for nsdfg in sdfg.all_sdfgs_recursive():
            if nsdfg is sdfg:
                continue
            for state in list(nsdfg.states()):
                total += self._dedup_identity_assigns(state)
                total += self._bypass_transient_assigns(state)
        assert_invariant(no_memlet_dim_mismatch(sdfg), "BypassTrivialAssignTasklets",
                         "memlet subset and other_subset have matching dimensionality")
        assert_invariant(no_duplicate_connector_edges(sdfg), "BypassTrivialAssignTasklets",
                         "every connector has <=1 edge per direction")
        return total if total > 0 else None

    @staticmethod
    def _dedup_identity_assigns(istate: SDFGState) -> int:
        """Collapse duplicate ``AN(src) -> [_out=_in] -> AN(dst)`` triples.

        FP-factor branch lowering can leave two arms emitting the same
        ``cond -> float_factor`` assign chain side-by-side -- the cond
        compute writes ONE tile (``tmp_condition_symbol_to_scalar_4``)
        and BOTH per-arm assigns route it to ``float___tmp0``. The
        duplicate out-edges from ``src`` push its out-degree above 1,
        which would trip :meth:`_bypass_transient_assigns`'s safety
        guard (intended to keep SSA-like reassignment chains intact).
        Keep ONE assign per unique ``(src.data, dst.data)`` pair so the
        bypass can proceed; the other copies route into the same
        canonical ``dst`` AccessNode (or are removed when both endpoints
        are the same node) and the cond tile no longer fans out per arm.

        :param istate: Inner state being rewritten.
        :returns: Number of duplicate tasklets removed.
        """
        seen: Dict = {}
        removed = 0
        for t in [n for n in istate.nodes() if isinstance(n, dace.nodes.Tasklet)]:
            triple = _assign_triple(istate, t)
            if triple is None:
                continue
            in_e, out_e = triple
            key = (in_e.src.data, out_e.dst.data)
            keep = seen.setdefault(key, (t, in_e.src, out_e.dst))
            if keep[0] is t:
                continue
            # Duplicate: rewire any other in/out edges of THIS tasklet's
            # endpoints onto the kept tasklet's endpoints, then drop t.
            kept_src, kept_dst = keep[1], keep[2]
            cur_src, cur_dst = in_e.src, out_e.dst
            if cur_dst is not kept_dst:
                for de in list(istate.out_edges(cur_dst)):
                    istate.add_edge(kept_dst, de.src_conn, de.dst, de.dst_conn,
                                    dace.Memlet.from_memlet(de.data) if de.data is not None else dace.Memlet())
                    istate.remove_edge(de)
            for te in list(istate.in_edges(t)) + list(istate.out_edges(t)):
                istate.remove_edge(te)
            istate.remove_node(t)
            removed += 1
            if cur_src is not kept_src and istate.degree(cur_src) == 0:
                istate.remove_node(cur_src)
            if cur_dst is not kept_dst and istate.degree(cur_dst) == 0:
                istate.remove_node(cur_dst)
        return removed

    @staticmethod
    def _bypass_transient_assigns(istate: SDFGState) -> int:
        """Bypass ``AN(src) -> [_out=_in] -> AN(dst)`` when one side is transient.

        Only fires when:

        * the tasklet body is the trivial ``_out = _in`` form;
        * the in-edge ``src`` and out-edge ``dst`` are both AccessNodes;
        * at least one of ``src`` / ``dst`` is a transient (collapsing a
          connector -> connector assign would lose the boundary edge);
        * ``out_degree(src) == 1`` AND ``in_degree(dst) == 1`` (multi-
          consumer / multi-producer patterns include SSA-like
          reassignment chains where the bypass would silently fold
          separate assignments onto one AccessNode and pick the wrong
          value -- leave those alone).

        Preference: route the source's producer to write into the
        destination AccessNode (preserving downstream consumers of
        ``dst``); fall back to the other direction when the source is
        the transient.

        :param istate: Inner state being rewritten.
        :returns: Number of bypassed tasklets dropped.
        """
        inner = istate.sdfg
        removed = 0
        for t in [n for n in istate.nodes() if isinstance(n, dace.nodes.Tasklet)]:
            triple = _assign_triple(istate, t)
            if triple is None:
                continue
            in_e, out_e = triple
            src_an, dst_an = in_e.src, out_e.dst
            src_desc = inner.arrays.get(src_an.data)
            dst_desc = inner.arrays.get(dst_an.data)
            if src_desc is None or dst_desc is None:
                continue
            if not (src_desc.transient or dst_desc.transient):
                continue
            if istate.out_degree(src_an) > 1 or istate.in_degree(dst_an) > 1:
                continue
            # Cross-state transient guard: the side we would collapse must not be
            # accessed in another state, else its real consumer/producer lives
            # elsewhere and the state-local degrees here are misleading (removing
            # the staging would orphan a node / break the cross-state flow). See
            # :func:`_accessed_in_other_states`.
            src_xstate = src_desc.transient and _accessed_in_other_states(inner, src_an.data, istate)
            dst_xstate = dst_desc.transient and _accessed_in_other_states(inner, dst_an.data, istate)
            # Map-scope-boundary guard: the bypass splices the transient's producer
            # (src branch) / consumer (dst branch) directly onto the other endpoint
            # with a single new edge. When that neighbour is a MapEntry / MapExit,
            # the splice would rename only ONE side of the scope's ``IN_x``/``OUT_x``
            # passthrough connector, leaving the two sides naming different data --
            # an invalid SDFG. (spmv: the per-row accumulator ``tmp`` is fed by the
            # idx-map's MapExit; bypassing ``tmp -> __tmp_w`` renamed ``OUT_tmp`` but
            # left ``IN_tmp`` as ``tmp``.) Collapsing across a scope boundary needs a
            # consistent passthrough rename, which a single-edge splice cannot do, so
            # leave such copies in place -- a plain copy the rest of the pipeline handles.
            src_at_scope = any(
                isinstance(pe.src, (dace.nodes.MapEntry, dace.nodes.MapExit)) for pe in istate.in_edges(src_an))
            dst_at_scope = any(
                isinstance(de.dst, (dace.nodes.MapEntry, dace.nodes.MapExit)) for de in istate.out_edges(dst_an))
            if src_desc.transient and istate.in_edges(src_an) and not src_xstate and not src_at_scope:
                # P -> AN(src) -> [_out=_in] -> AN(dst) becomes P -> AN(dst).
                # Carry BOTH sides of the bypassed chain on the new memlet so
                # ``an_side_subset`` can return the lane-dep subset for the
                # source AN downstream (instead of falling back to the full
                # descriptor shape). The memlet's ``data`` must reference one
                # of the actual endpoints (DaCe validates this); pick the
                # endpoint that has an AccessNode side -- for Tasklet -> AN
                # the only AccessNode endpoint is ``dst_an``, so use that.
                for pe in list(istate.in_edges(src_an)):
                    pe_subset = subsets.Range(list(pe.data.subset.ranges)) if pe.data.subset is not None else None
                    out_subset = subsets.Range(list(
                        out_e.data.subset.ranges)) if out_e.data.subset is not None else None
                    if isinstance(pe.src, dace.nodes.AccessNode):
                        new_memlet = dace.Memlet(data=pe.src.data, subset=pe_subset, other_subset=out_subset)
                    else:
                        # Tasklet -> AN: ``data`` must be the AN side (no other_subset).
                        new_memlet = dace.Memlet(data=dst_an.data, subset=out_subset)
                    istate.add_edge(pe.src, pe.src_conn, dst_an, out_e.dst_conn, new_memlet)
                    istate.remove_edge(pe)
                for te in list(istate.in_edges(t)) + list(istate.out_edges(t)):
                    istate.remove_edge(te)
                istate.remove_node(t)
                removed += 1
                if istate.degree(src_an) == 0:
                    istate.remove_node(src_an)
            elif dst_desc.transient and not dst_xstate and not dst_at_scope:
                # AN(src) -> [_out=_in] -> AN(dst) -> C becomes AN(src) -> C.
                # Symmetric to the src-transient branch -- pick ``data``
                # endpoint based on whether C is an AccessNode or a Tasklet.
                consumers = list(istate.out_edges(dst_an))
                if not consumers:
                    # No downstream consumer to reroute the source into. Dropping the
                    # tasklet here would strand ``src_an`` as an ISOLATED node (invalid
                    # SDFG) -- the tasklet's incoming edge was ``src_an``'s only edge.
                    # Resolve the trivial assign to a DIRECT AN -> AN copy instead so
                    # ``src_an``'s value still reaches ``dst_an`` and neither node is
                    # orphaned; a later dead-copy elimination drops it if ``dst_an`` is
                    # genuinely unused. If input and output resolve to the SAME access,
                    # the assign is a self-copy: just drop the tasklet (and the node if
                    # it is then dead) rather than emit a nonsensical self-loop edge.
                    for te in list(istate.in_edges(t)) + list(istate.out_edges(t)):
                        istate.remove_edge(te)
                    istate.remove_node(t)
                    removed += 1
                    if src_an is dst_an:
                        if istate.degree(dst_an) == 0:
                            istate.remove_node(dst_an)
                        continue
                    in_subset = subsets.Range(list(in_e.data.subset.ranges)) if in_e.data.subset is not None else None
                    out_subset = subsets.Range(list(
                        out_e.data.subset.ranges)) if out_e.data.subset is not None else None
                    copy_memlet = dace.Memlet(data=src_an.data, subset=in_subset, other_subset=out_subset)
                    _wcr = out_e.data.wcr if out_e.data.wcr is not None else in_e.data.wcr
                    if _wcr is not None:
                        copy_memlet.wcr = _wcr
                        copy_memlet.wcr_nonatomic = bool(out_e.data.wcr_nonatomic or in_e.data.wcr_nonatomic)
                    istate.add_edge(src_an, in_e.src_conn, dst_an, out_e.dst_conn, copy_memlet)
                    continue
                for de in consumers:
                    in_subset = subsets.Range(list(in_e.data.subset.ranges)) if in_e.data.subset is not None else None
                    de_subset = subsets.Range(list(de.data.subset.ranges)) if de.data.subset is not None else None
                    if isinstance(de.dst, dace.nodes.AccessNode):
                        new_memlet = dace.Memlet(data=src_an.data, subset=in_subset, other_subset=de_subset)
                    else:
                        # AN -> Tasklet: ``data`` must be the AN side (no other_subset).
                        new_memlet = dace.Memlet(data=src_an.data, subset=in_subset)
                    # Preserve a reduction across the bypass: an in-place ``a[i] += b[i]``
                    # canonicalises to ``b -> [_out=_in] -> _wcr_priv -(+=)-> a``; dropping
                    # the WCR here would silently degrade it to ``a = b``. The following
                    # WCRToAugAssign pass converts the surviving WCR into an explicit RMW
                    # tasklet so no WCR is left inside the body NSDFG before tiling.
                    _wcr = de.data.wcr if de.data.wcr is not None else in_e.data.wcr
                    if _wcr is not None:
                        new_memlet.wcr = _wcr
                        new_memlet.wcr_nonatomic = bool(de.data.wcr_nonatomic or in_e.data.wcr_nonatomic)
                    istate.add_edge(src_an, in_e.src_conn, de.dst, de.dst_conn, new_memlet)
                    istate.remove_edge(de)
                for te in list(istate.in_edges(t)) + list(istate.out_edges(t)):
                    istate.remove_edge(te)
                istate.remove_node(t)
                removed += 1
                if istate.degree(dst_an) == 0:
                    istate.remove_node(dst_an)
        return removed
