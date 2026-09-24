# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Bypass trivial ``AN -> [_out=_in] -> AN`` assign tasklets.

A trivial assign tasklet whose body is exactly ``_out = _in`` (one input
connector, one output connector) and whose only incoming / outgoing edge
each connects to an :class:`~dace.sdfg.nodes.AccessNode` is a pure copy.
DaCe's tasklet codegen for a Python-language ``_out = _in`` body with
tile-pointer connectors emits ``_out = _in;`` -- a *pointer* reassignment
of the local variable; the destination transient is never actually
written. This pass exposes the rewrite as a standalone pipeline step so the
multi-dim K=1 / K=2 paths can call it directly.

Two rewrites, in order:

1. **Dedup** -- when several trivial assign tasklets copy the same
   source element into the same destination element (e.g. ``fp_factor``
   branch lowering emitting one cond-to-merge chain per arm side-by-side),
   collapse them to ONE. Without this step the source's ``out_degree`` would
   exceed 1 and the bypass below would refuse it.
2. **Bypass** -- when at least one side is a transient AND
   ``out_degree(src) == 1`` AND ``in_degree(dst) == 1``, drop the
   tasklet and route the producer / consumer of the transient side
   directly. The single-consumer / single-producer guard keeps
   SSA-like reassignment chains intact (``c1 = c[i]; ...; c1 =
   c1*d1*e1 + ...`` -- bypassing would fold two assignments onto one
   AccessNode and pick up the wrong value). A producer that ACCUMULATES
   into the transient is refused outright: its seed is a separate store
   the bypass cannot carry along.

The pass is body-NSDFG-scoped: the outer SDFG's ``AN -> AN`` edges may
be scatter / gather staging, so they stay untouched. Mirrors
:class:`EliminateDeadCopies`'s scoping.
"""
import copy
from typing import Any

import dace
from dace import subsets
from dace.memlet import Memlet
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg import SDFG
from dace.sdfg.state import SDFGState
from dace.libraries.standard.nodes.reduce import Reduce
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.passes.vectorization.utils.pass_invariants import (assert_invariant,
                                                                            no_duplicate_connector_edges,
                                                                            no_memlet_dim_mismatch)
# _is_assign_tasklet was previously imported from emit_tile_ops (deleted in the walker-primary
# migration). The matcher is inlined below.


def _is_assign_tasklet(t: dace.nodes.Node) -> bool:
    # True iff ``t`` is a tasklet with a single in / out connector and a body of the form ``<out_conn> = <in_conn>`` (no
    # arithmetic, no calls).
    if not isinstance(t, dace.nodes.Tasklet):
        return False
    if len(t.in_connectors) != 1 or len(t.out_connectors) != 1:
        return False
    in_conn = next(iter(t.in_connectors))
    out_conn = next(iter(t.out_connectors))
    body = t.code.as_string.strip().rstrip(";")
    return body == f"{out_conn} = {in_conn}"


def _assign_triple(istate: SDFGState,
                   t: dace.nodes.Tasklet) -> tuple[MultiConnectorEdge[Memlet], MultiConnectorEdge[Memlet]] | None:
    # Return ``(in_edge, out_edge)`` iff ``t`` is the trivial ``AN -> [_out=_in] -> AN`` triple.
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
    # True iff ``data_name`` has an AccessNode in some state OTHER than ``current_state``.
    for st in inner_sdfg.states():
        if st is current_state:
            continue
        for n in st.data_nodes():
            if n.data == data_name:
                return True
    return False


def _accumulates_into_destination(pe: MultiConnectorEdge[Memlet]) -> bool:
    # True iff producer edge ``pe`` folds into what its destination ALREADY holds.
    if pe.data is not None and pe.data.wcr is not None:
        return True
    return isinstance(pe.src, Reduce) and pe.src.identity is None


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

    def depends_on(self) -> set[type[ppl.Pass] | ppl.Pass]:
        """Standalone pass."""
        return set()

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> int | None:
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
        # Collapse duplicate ``AN(src) -> [_out=_in] -> AN(dst)`` triples.
        seen: dict = {}
        removed = 0
        for t in [n for n in istate.nodes() if isinstance(n, dace.nodes.Tasklet)]:
            triple = _assign_triple(istate, t)
            if triple is None:
                continue
            in_e, out_e = triple
            # Copies into different elements are separate writes: ``c[0] = z; c[1] = z`` keeps both. Key on the
            # source node: another node of ``x`` may follow a write to ``x``.
            key = (in_e.src, str(in_e.data.subset), out_e.dst.data, str(out_e.data.subset))
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
        # Bypass ``AN(src) -> [_out=_in] -> AN(dst)`` when one side is transient.
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
            # The collapsed side must not be accessed in another state (see :func:`_accessed_in_other_states`).
            src_xstate = src_desc.transient and _accessed_in_other_states(inner, src_an.data, istate)
            dst_xstate = dst_desc.transient and _accessed_in_other_states(inner, dst_an.data, istate)
            # Do not splice onto a MapEntry / MapExit: renaming one side of the ``IN_x`` / ``OUT_x`` passthrough
            # leaves an invalid SDFG (spmv ``tmp``).
            src_at_scope = any(
                isinstance(pe.src, (dace.nodes.MapEntry, dace.nodes.MapExit)) for pe in istate.in_edges(src_an))
            dst_at_scope = any(
                isinstance(de.dst, (dace.nodes.MapEntry, dace.nodes.MapExit)) for de in istate.out_edges(dst_an))
            # The src splice is value-preserving only if the producer defines the value; an accumulator seeded
            # on the transient would strand its seed (tsvc_2_5 reduce_inner_carry).
            src_accumulated = any(_accumulates_into_destination(pe) for pe in istate.in_edges(src_an))
            # Ordering-edge guard: an empty memlet into ``src_an`` sequences the producer's write after
            # another node. Splicing the producer onto ``dst_an`` would leave that ordering on a node
            # nothing writes, and the write would run unordered.
            src_ordered = any(pe.data is None or pe.data.is_empty() for pe in istate.in_edges(src_an))
            # Splice only a sole producer that writes exactly the element the copy reads.
            src_in = istate.in_edges(src_an)
            src_sole = len(src_in) == 1 and (src_in[0].data.get_dst_subset(
                src_in[0], istate) or subsets.Range.from_array(src_desc)) == in_e.data.get_src_subset(in_e, istate)
            if (src_desc.transient and src_sole and not src_xstate and not src_at_scope and not src_accumulated
                    and not src_ordered):
                # P -> AN(src) -> [_out=_in] -> AN(dst) becomes P -> AN(dst), carrying both subsets so
                # ``an_side_subset`` sees the lane-dep source subset; ``data`` names the AccessNode endpoint.
                for pe in list(istate.in_edges(src_an)):
                    # LEAVE dependency edges alone -- an empty memlet only orders two nodes, so
                    # rewriting it as dataflow would invent a copy the program never had.
                    if pe.data is None or pe.data.is_empty():
                        continue
                    pe_subset = copy.deepcopy(pe.data.get_src_subset(pe, istate))
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
                # AN(src) -> [_out=_in] -> AN(dst) -> C becomes AN(src) -> C. Ordering-only out-edges are not
                # consumers; counting them would skip the direct-copy fallback and isolate ``src_an``.
                consumers = [e for e in istate.out_edges(dst_an) if not e.data.is_empty()]
                # An ordering edge out of ``dst_an`` sequences WHEN the copy's value is taken -- CloudSC's
                # ``ztold = ztp1`` before ``ztp1`` is overwritten. Rerouting the consumers onto ``src_an``
                # strands that ordering on a transient nothing writes, and the consumers then read ``src``
                # after the overwrite. Keep ``dst_an`` written through the direct copy below instead.
                dst_ordered = any(e.data is None or e.data.is_empty() for e in istate.out_edges(dst_an))
                if not consumers or dst_ordered:
                    # No consumer: make a direct AN -> AN copy so ``src_an`` is not isolated; a self-copy just drops
                    # the tasklet.
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
                    # Leave empty (ordering) memlets alone: rebuilding one as a data memlet invents a copy, possibly a
                    # write to a read-only input (TSVC s471).
                    if de.data is None or de.data.is_empty():
                        continue
                    in_subset = subsets.Range(list(in_e.data.subset.ranges)) if in_e.data.subset is not None else None
                    de_subset = copy.deepcopy(de.data.get_dst_subset(de, istate))
                    if isinstance(de.dst, dace.nodes.AccessNode):
                        new_memlet = dace.Memlet(data=src_an.data, subset=in_subset, other_subset=de_subset)
                    else:
                        # AN -> Tasklet: ``data`` must be the AN side (no other_subset).
                        new_memlet = dace.Memlet(data=src_an.data, subset=in_subset)
                    # Keep the WCR of an in-place ``a[i] += b[i]``; WCRToAugAssign later turns it into an explicit RMW.
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
