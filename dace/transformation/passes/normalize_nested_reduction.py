# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Pass that rewrites a masked in-nsdfg WCR into a write-only output connector.

The DaCe frontend can emit a masked/nested scalar reduction as a WCR edge that
lands *inside* a :class:`~dace.sdfg.nodes.NestedSDFG` body, writing a
**write-only** output connector, while the ``NestedSDFG -> MapExit`` edge for that
connector is a *plain* copy (no WCR). Codegen lowers the in-body WCR into an
``reduce_atomic`` through the write-only output pointer -- correct on its own, but
downstream canonicalization passes (``WCRToAugAssign`` severs the atomic;
``MapFusionVertical`` double-counts it) mangle this shape.

This pass rewrites it into the shape the DaCe frontend already emits natively for
the equivalent polybench reduction (e.g. ``symm``): the accumulation runs on a
seeded body-local transient, and the cross-iteration reduction is a WCR on the
``NestedSDFG -> MapExit -> accumulator`` edge chain.

For each map-body :class:`~dace.sdfg.nodes.NestedSDFG` with a write-only output
connector ``oc`` whose body contains a *scalar* WCR edge into a sink
:class:`~dace.sdfg.nodes.AccessNode` named ``oc``, and whose ``NestedSDFG ->
MapExit`` edge for ``oc`` is plain, the pass:

1. Introduces a body-local transient ``_nnr_priv`` matching ``oc``'s descriptor,
   seeded to the reduction op's identity in a state that dominates the WCR write,
   redirects the WCR edge to accumulate into ``_nnr_priv``, then copies
   ``_nnr_priv -> oc`` in a state that post-dominates the write. The connector now
   carries a masked *addend* (the op identity where the guard is false).
2. Places the reduction op as a WCR on the accumulator edge chain, sourced from a
   per-iteration private :class:`~dace.sdfg.nodes.AccessNode` inserted between the
   NestedSDFG output and the MapExit (a WCR left directly on the pointer-typed
   ``NestedSDFG -> MapExit`` connector is dropped by the CPU codegen; see
   :mod:`dace.transformation.passes.normalize_wcr_source`). The outer accumulator is
   already identity-seeded for the ``+`` kernels; for ``min`` / ``max`` a seed is
   inserted only when one is missing.

The pass is idempotent: after rewriting, the body WCR writes a transient (not an
output connector) and the ``NestedSDFG -> MapExit`` edge carries a WCR (and is
AccessNode-sourced), so neither detection condition matches on a second run.
"""
import ast
import copy
from typing import Any, Dict, Optional, Set

import numpy

from dace import SDFG, SDFGState, data, dtypes
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.transformation import pass_pipeline as ppl, transformation

#: Reduction op -> the augassign op it normalizes to (``-`` accumulates like ``+``).
_WCR_OP = {'+': '+', '-': '+', '*': '*', 'min': 'min', 'max': 'max'}


def _op_from_wcr(wcr: str) -> Optional[str]:
    """Return the reduction op (``+``/``*``/``min``/``max``) for a WCR lambda string.

    The WCR body is either a binary op (``x + y``) or a call (``min(x, y)``); anything
    else (a non-associative or unrecognized reducer) yields ``None`` so the edge is
    left untouched.
    """
    try:
        tree = ast.parse(wcr.strip(), mode='eval').body
    except SyntaxError:
        return None
    if not isinstance(tree, ast.Lambda):
        return None
    body = tree.body
    if isinstance(body, ast.BinOp):
        opmap = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*'}
        return _WCR_OP.get(opmap.get(type(body.op)))
    if isinstance(body, ast.Call) and isinstance(body.func, ast.Name) and body.func.id in ('min', 'max'):
        return _WCR_OP.get(body.func.id)
    return None


def _identity_value(op: str, dtype: dtypes.typeclass):
    """Return the SDFG-level seed value for ``op``'s identity at ``dtype``.

    Integer dtypes use exact ``iinfo`` bounds for ``min`` / ``max`` (a float ``inf``
    seed would be silently wrong once truncated to an integer accumulator).
    """
    is_int = numpy.issubdtype(dtype.type, numpy.integer)
    if op == '+':
        return 0 if is_int else 0.0
    if op == '*':
        return 1 if is_int else 1.0
    if op == 'min':
        return int(numpy.iinfo(dtype.type).max) if is_int else float('inf')
    if op == 'max':
        return int(numpy.iinfo(dtype.type).min) if is_int else float('-inf')
    raise ValueError(f'unknown op {op!r}')


@transformation.explicit_cf_compatible
class NormalizeNestedReduction(ppl.Pass):
    """Rewrite masked in-nsdfg write-only WCR reductions into the seeded-local +
    map-exit-WCR shape.
    """

    CATEGORY: str = 'Simplification'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Memlets | ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> Set:
        return set()

    def _seed_desc(self, oc_desc: data.Data) -> data.Data:
        """Descriptor for the body-local accumulator mirroring ``oc`` (transient)."""
        new = copy.deepcopy(oc_desc)
        new.transient = True
        new.storage = dtypes.StorageType.Default
        new.lifetime = dtypes.AllocationLifetime.Scope
        return new

    def _full_subset(self, desc: data.Data) -> str:
        if isinstance(desc, data.Scalar):
            return '0'
        return ', '.join(f'0:{s}' for s in desc.shape)

    def _seed_state(self, nsdfg: SDFG, priv: str, desc: data.Data, op: str) -> None:
        """Seed ``priv`` to ``op``'s identity in a fresh start state of ``nsdfg``."""
        val = _identity_value(op, desc.dtype)
        st = nsdfg.add_state_before(nsdfg.start_block, label='_nnr_seed', is_start_block=True)
        w = st.add_write(priv)
        if isinstance(desc, data.Scalar) or tuple(desc.shape) == (1, ):
            t = st.add_tasklet('_nnr_seed', {}, {'__out'}, f'__out = {val}')
            st.add_edge(t, '__out', w, None, Memlet(data=priv, subset='0'))
            return
        # Array accumulator (scatter reduction): fill every element with the identity.
        me, mx = st.add_map('_nnr_seed', {f'_nnr_i{d}': f'0:{s}' for d, s in enumerate(desc.shape)})
        t = st.add_tasklet('_nnr_seed', {}, {'__out'}, f'__out = {val}')
        idx = ', '.join(f'_nnr_i{d}' for d in range(len(desc.shape)))
        st.add_edge(me, None, t, None, Memlet())
        st.add_memlet_path(t, mx, w, src_conn='__out', memlet=Memlet(data=priv, subset=idx))

    def _copyback_state(self, nsdfg: SDFG, priv: str, oc: str, desc: data.Data) -> None:
        """Copy ``priv -> oc`` (plain) in a fresh state after every current sink block."""
        sink = nsdfg.sink_nodes()
        sub = self._full_subset(desc)
        for term in sink:
            st = nsdfg.add_state_after(term, label='_nnr_copyback')
            r = st.add_read(priv)
            w = st.add_write(oc)
            st.add_nedge(r, w, Memlet(data=priv, subset=sub, other_subset=sub))

    def _rewrite_nsdfg_output(self, state: SDFGState, nsdfg: nodes.NestedSDFG, oc: str) -> bool:
        """Normalize a single write-only reduction output ``oc``. Returns True on rewrite."""
        inner = nsdfg.sdfg
        oc_desc = inner.arrays.get(oc)
        if oc_desc is None:
            return False

        # (a) A scalar WCR edge inside the body writing `oc` into a pure-sink AccessNode.
        wcr_edge = None
        wcr_state = None
        for ist in inner.all_states():
            for e in ist.edges():
                if (e.data is not None and e.data.wcr is not None and e.data.data == oc and e.data.subset is not None
                        and e.data.subset.num_elements() == 1 and isinstance(e.dst, nodes.AccessNode)
                        and e.dst.data == oc and ist.out_degree(e.dst) == 0):
                    if wcr_edge is not None:
                        return False  # more than one write-only reduction sink -> not this shape
                    wcr_edge, wcr_state = e, ist
        if wcr_edge is None:
            return False

        op = _op_from_wcr(wcr_edge.data.wcr)
        if op is None:
            return False

        # (b) The NestedSDFG -> MapExit edge for `oc` must be plain (no WCR yet).
        out_edge = next((oe for oe in state.out_edges(nsdfg) if oe.src_conn == oc), None)
        if out_edge is None or out_edge.data.wcr is not None or not isinstance(out_edge.dst, nodes.MapExit):
            return False

        # (c) The map-level private buffer mirrors `oc`'s inner descriptor. If that
        # descriptor's shape references symbols defined only inside the NestedSDFG (e.g.
        # a body-local symbol aliased to an outer one via `symbol_mapping`), materializing
        # the buffer at the outer scope would leak them as free symbols. Leave the shape
        # untouched in that case (the baseline reduction stays correct as-is).
        if {str(s) for s in oc_desc.free_symbols} - set(state.sdfg.symbols.keys()):
            return False

        # --- Rewrite the body: accumulate into a seeded private transient. ---
        priv_desc = self._seed_desc(oc_desc)
        priv = inner.add_datadesc(f'_nnr_priv_{oc}', priv_desc, find_new_name=True)
        priv_node = wcr_state.add_access(priv)
        new_data = copy.deepcopy(wcr_edge.data)
        new_data.data = priv
        wcr_state.add_edge(wcr_edge.src, wcr_edge.src_conn, priv_node, None, new_data)
        old_sink = wcr_edge.dst
        wcr_state.remove_edge(wcr_edge)
        if wcr_state.degree(old_sink) == 0:
            wcr_state.remove_node(old_sink)

        wcr_str = wcr_edge.data.wcr
        self._seed_state(inner, priv, priv_desc, op)
        self._copyback_state(inner, priv, oc, oc_desc)
        inner.reset_cfg_list()

        # --- Rewrite the map level: put the reduction on the accumulator edge chain. ---
        # The WCR must source from an AccessNode: a WCR left on the NestedSDFG->MapExit
        # edge is a *pointer*-typed connector, which the CPU codegen's WCR path drops
        # (see NormalizeWCRSource). Insert a per-iteration private AccessNode between the
        # NestedSDFG output and the MapExit so the reduction is AccessNode-sourced.
        out_priv_desc = self._seed_desc(oc_desc)
        out_priv = state.sdfg.add_datadesc(f'_nnr_out_{oc}', out_priv_desc, find_new_name=True)
        out_priv_node = state.add_access(out_priv)
        state.add_edge(nsdfg, oc, out_priv_node, None, Memlet(data=out_priv, subset=self._full_subset(out_priv_desc)))
        acc_node = None
        for e in state.memlet_path(out_edge):
            e.data.wcr = wcr_str
            if isinstance(e.dst, nodes.AccessNode):
                acc_node = e.dst
        state.add_edge(out_priv_node, None, out_edge.dst, out_edge.dst_conn, copy.deepcopy(out_edge.data))
        state.remove_edge(out_edge)
        if op in ('min', 'max') and acc_node is not None:
            self._ensure_outer_seed(state.sdfg, acc_node.data, op)
        return True

    def _ensure_outer_seed(self, sdfg: SDFG, acc: str, op: str) -> None:
        """Insert an identity seed of ``acc`` before its reduction, only if none exists.

        A seed is any plain (non-WCR) write of ``acc`` reaching the accumulator; if the
        accumulator is only ever WCR-written, a ``min`` / ``max`` reduction would start
        from an uninitialized slot, so an identity-writing state is prepended.
        """
        desc = sdfg.arrays.get(acc)
        if desc is None:
            return
        for st in sdfg.all_states():
            for e in st.edges():
                if (e.data is not None and e.data.data == acc and e.data.wcr is None
                        and isinstance(e.dst, nodes.AccessNode) and e.dst.data == acc):
                    return
        val = _identity_value(op, desc.dtype)
        seed = sdfg.add_state_before(sdfg.start_block, label='_nnr_outer_seed', is_start_block=True)
        w = seed.add_write(acc)
        if isinstance(desc, data.Scalar) or tuple(desc.shape) == (1, ):
            t = seed.add_tasklet('_nnr_outer_seed', {}, {'__out'}, f'__out = {val}')
            seed.add_edge(t, '__out', w, None, Memlet(data=acc, subset='0'))
        else:
            me, mx = seed.add_map('_nnr_outer_seed', {f'_nnr_i{d}': f'0:{s}' for d, s in enumerate(desc.shape)})
            t = seed.add_tasklet('_nnr_outer_seed', {}, {'__out'}, f'__out = {val}')
            idx = ', '.join(f'_nnr_i{d}' for d in range(len(desc.shape)))
            seed.add_edge(me, None, t, None, Memlet())
            seed.add_memlet_path(t, mx, w, src_conn='__out', memlet=Memlet(data=acc, subset=idx))

    def _apply(self, sdfg: SDFG) -> int:
        total = 0
        for sd in sdfg.all_sdfgs_recursive():
            for state in sd.all_states():
                for nsdfg in [n for n in state.nodes() if isinstance(n, nodes.NestedSDFG)]:
                    if not isinstance(state.entry_node(nsdfg), nodes.MapEntry):
                        continue
                    write_only = [oc for oc in nsdfg.out_connectors if oc not in nsdfg.in_connectors]
                    for oc in write_only:
                        if self._rewrite_nsdfg_output(state, nsdfg, oc):
                            total += 1
        return total

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Set[str]]]:
        """Normalize every masked in-nsdfg write-only WCR reduction under a Map.

        :param sdfg: The SDFG to normalize.
        :param pipeline_results: Results of prior passes in the pipeline (unused).
        :returns: ``None`` if nothing was rewritten; otherwise a single-entry dict with
                  the rewritten-output count under key ``normalized_nested_reductions``.
        """
        n = self._apply(sdfg)
        if n == 0:
            return None
        sdfg.reset_cfg_list()
        sdfg.validate()
        return {'normalized_nested_reductions': {str(n)}}
