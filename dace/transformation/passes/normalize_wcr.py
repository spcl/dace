# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Pass that rewrites non-canonical WCR shapes inside map-body NestedSDFGs.

Two shapes are handled:

* **Write-only scalar reduction.** The DaCe frontend emits a masked/nested scalar
  reduction as a WCR edge that lands *inside* a :class:`~dace.sdfg.nodes.NestedSDFG`
  body, writing a **write-only** output connector, while the
  ``NestedSDFG -> MapExit`` edge for that connector is a *plain* copy (no WCR).
  Codegen lowers the in-body WCR into an ``reduce_atomic`` through the write-only
  output pointer -- correct on its own, but downstream canonicalization passes
  (``WCRToAugAssign`` severs the atomic; ``MapFusionVertical`` double-counts it)
  mangle this shape.

* **Read-write redundant interior WCR (TSVC s212).** An in-place aug-assign such as
  ``b[i] += data[i]`` can be encoded with ``oc`` as both an input and output
  connector, a scalar WCR edge into ``AccessNode(oc)`` inside the body, *and* a
  redundant WCR on the ``NestedSDFG -> MapExit`` edge. Dropping the interior WCR
  to plain loses the incoming ``b`` value; instead the pass rewrites the body to an
  explicit plain read-modify-write ``oc_out = oc_in OP addend`` and removes the WCR
  from the boundary.

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
   :meth:`NormalizeWCR.source_from_access_nodes`). The outer accumulator is
   already identity-seeded for the ``+`` kernels; for ``min`` / ``max`` a seed is
   inserted only when one is missing.

The read-write rewrite creates a fresh output connector ``_nnr_rwout_<oc>``,
re-routes the addend through a body-local scratch, and inserts a plain tasklet that
computes ``oc_out = oc_in OP addend``. The new output is write-only and the boundary
edge becomes a plain copy, so the pass is idempotent.

Then every WCR edge a NestedSDFG sources, or a Tasklet sources into a map exit, is
sourced from an AccessNode instead
(:meth:`NormalizeWCR.source_from_access_nodes`): a per-iteration private ``_wcr_priv``
transient takes the producer's plain write and carries the WCR on. The CPU codegen
drops a WCR on a NestedSDFG's pointer-typed output connector, and the reduction
lowerings (OpenMP clause, tree reductions) read the ``AccessNode -[wcr]-> MapExit``
partial. A Tasklet WCR into an access node is left alone: codegen resolves it on its
scalar connector, and a private access node would only turn it into an accumulating copy.
The same rewrite identity-seeds a fresh accumulator that is only ever WCR-written
(:meth:`NormalizeWCR.seed_accumulator`).
"""

import copy
from typing import Any

import numpy

from dace import SDFG, SDFGState, data, dtypes, subsets, symbolic
from dace.frontend.operations import detect_reduction_type
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.symbolic import symbol
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.dataflow.wcr_conversion import wcr_operator
from dace.transformation.passes.analysis import scopes
from dace.transformation.passes.privatize_scatter_reduction import is_data_dependent_scatter_sink

#: Reduction op -> the augassign op it normalizes to (``-`` accumulates like ``+``).
_WCR_OP = {"+": "+", "-": "+", "*": "*", "min": "min", "max": "max"}


def _op_from_wcr(wcr: str) -> str | None:
    """The accumulation the WCR lambda performs (``+``/``*``/``min``/``max``; ``-`` accumulates like ``+``), or
    ``None`` for any other reducer."""
    return _WCR_OP.get(wcr_operator(wcr))


def _binop_expr_from_wcr(wcr: str) -> str | None:
    """The WCR lambda's operator, keeping ``-`` as subtraction (the aug-assign operator); ``None`` for any other
    reducer than ``+ - * min max``."""
    op = wcr_operator(wcr)
    return op if op in _WCR_OP else None


def _rmw_tasklet_code(op: str) -> str:
    """Return a plain tasklet expression that computes ``old OP addend``."""
    if op in ("+", "-", "*"):
        return f"__out = __old {op} __addend"
    if op == "min":
        return "__out = min(__old, __addend)"
    if op == "max":
        return "__out = max(__old, __addend)"
    raise ValueError(f"unsupported read-modify-write op {op!r}")


def _identity_value(op: str, dtype: dtypes.typeclass):
    """Return the SDFG-level seed value for ``op``'s identity at ``dtype``.

    Integer dtypes use exact ``iinfo`` bounds for ``min`` / ``max`` (a float ``inf``
    seed would be silently wrong once truncated to an integer accumulator).
    """
    is_int = numpy.issubdtype(dtype.type, numpy.integer)
    if op == "+":
        return 0 if is_int else 0.0
    if op == "*":
        return 1 if is_int else 1.0
    if op == "min":
        return int(numpy.iinfo(dtype.type).max) if is_int else float("inf")
    if op == "max":
        return int(numpy.iinfo(dtype.type).min) if is_int else float("-inf")
    raise ValueError(f"unknown op {op!r}")


def accumulates_into(sd: SDFG, targets: set[str]) -> bool:
    """True if ``sd`` -- or a NestedSDFG below it, reached by following the connector binding -- folds into one of
    ``targets`` instead of plain-writing it: either a WCR edge accumulating onto the name, or any read of it (an
    in-place read-modify-write). Membership-only: the answer is an order-independent OR over the subtree.
    """
    for ist in sd.states():
        for e in ist.edges():
            if e.data is None or e.data.is_empty():
                continue
            if e.data.wcr is not None and (
                e.data.data in targets or (isinstance(e.dst, nodes.AccessNode) and e.dst.data in targets)
            ):
                return True
            if isinstance(e.src, nodes.AccessNode) and e.src.data in targets:
                return True
        for node in ist.nodes():
            if not isinstance(node, nodes.NestedSDFG):
                continue
            inner: set[str] = set()
            for oe in ist.out_edges(node):
                if (
                    oe.src_conn
                    and oe.data is not None
                    and (oe.data.data in targets or (isinstance(oe.dst, nodes.AccessNode) and oe.dst.data in targets))
                ):
                    inner.add(oe.src_conn)
            for ie in ist.in_edges(node):
                if (
                    ie.dst_conn
                    and ie.data is not None
                    and (ie.data.data in targets or (isinstance(ie.src, nodes.AccessNode) and ie.src.data in targets))
                ):
                    inner.add(ie.dst_conn)
            if inner and accumulates_into(node.sdfg, inner):
                return True
    return False


@transformation.explicit_cf_compatible
class NormalizeWCR(ppl.Pass):
    """Normalize WCR into the codegen-supported shapes.

    First the map-body NestedSDFG reductions (:meth:`normalize_nested_reductions`):
    - Scalar reduction (``num_elements()==1``): masked in-nsdfg write-only WCR ->
      seeded-body-local transient + WCR on the ``NestedSDFG -> MapExit ->
      accumulator`` edge chain (see module docstring).
    - Slice reduction (``num_elements()>1``, e.g. symm ``C[0:i,j] +=``): the nsdfg
      traps a per-element scatter map whose ``X[k]`` single-element WCR only surfaces
      as a slice at the boundary (the nsdfg is multi-state -> not inlinable).
      ``_extract_slice_wcr`` clones that scatter map to the OUTER scope writing the
      destination ``dest[k,...]`` single-element WCR directly (inputs recomposed
      through the nsdfg boundary in the container's coordinates); the trapped nsdfg output
      is redirected to a dead transient (DCE prunes). Yields the vectorizable
      single-element form; a slice WCR is neither omp- nor tblock-reducible.

    ``extract_slice_wcr`` gates the slice rewrite. It assumes the post-``LoopToMap``
    structure (parallel outer map, per-element inner scatter). Run before ``LoopToMap``
    -- as the current canonicalize ``normalize_reduction`` stage does -- it fires on a
    partially-canonicalized body and is unsound, so canonicalize keeps it off until the
    pipeline is reordered to run this pass after ``LoopToMap`` (per the WCR design). The
    scalar rewrite is always on.

    Then every WCR edge gets an AccessNode source (:meth:`source_from_access_nodes`).
    """

    #: Validate the SDFG after a rewrite. Canonicalization turns it off and validates its own result once, when asked.
    validate: bool = True

    CATEGORY: str = "Simplification"

    def __init__(self, extract_slice_wcr: bool = False):
        super().__init__()
        self.extract_slice_wcr = extract_slice_wcr

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Memlets | ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> list[type[ppl.Pass] | ppl.Pass]:
        return []

    def _seed_desc(self, oc_desc: data.Data) -> data.Data:
        """Descriptor for the body-local accumulator mirroring ``oc`` (transient)."""
        new = copy.deepcopy(oc_desc)
        new.transient = True
        new.storage = dtypes.StorageType.Default
        new.lifetime = dtypes.AllocationLifetime.Scope
        return new

    def _full_subset(self, desc: data.Data) -> str:
        if isinstance(desc, data.Scalar):
            return "0"
        return ", ".join(f"0:{s}" for s in desc.shape)

    def _seed_state(self, nsdfg: SDFG, priv: str, desc: data.Data, op: str) -> None:
        """Seed ``priv`` to ``op``'s identity in a fresh start state of ``nsdfg``."""
        val = _identity_value(op, desc.dtype)
        st = nsdfg.add_state_before(nsdfg.start_block, label="_nnr_seed", is_start_block=True)
        w = st.add_write(priv)
        if isinstance(desc, data.Scalar) or tuple(desc.shape) == (1,):
            t = st.add_tasklet("_nnr_seed", {}, {"__out"}, f"__out = {val}")
            st.add_edge(t, "__out", w, None, Memlet(data=priv, subset="0"))
            return
        # Array accumulator (scatter reduction): fill every element with the identity.
        me, mx = st.add_map("_nnr_seed", {f"_nnr_i{d}": f"0:{s}" for d, s in enumerate(desc.shape)})
        t = st.add_tasklet("_nnr_seed", {}, {"__out"}, f"__out = {val}")
        idx = ", ".join(f"_nnr_i{d}" for d in range(len(desc.shape)))
        st.add_edge(me, None, t, None, Memlet())
        st.add_memlet_path(t, mx, w, src_conn="__out", memlet=Memlet(data=priv, subset=idx))

    def _copyback_state(self, nsdfg: SDFG, priv: str, oc: str, desc: data.Data) -> None:
        """Copy ``priv -> oc`` (plain) in a fresh state after every current sink block."""
        sink = nsdfg.sink_nodes()
        sub = self._full_subset(desc)
        for term in sink:
            st = nsdfg.add_state_after(term, label="_nnr_copyback")
            r = st.add_read(priv)
            w = st.add_write(oc)
            st.add_nedge(r, w, Memlet(data=priv, subset=sub, other_subset=sub))

    def _rewrite_nsdfg_output(self, state: SDFGState, nsdfg: nodes.NestedSDFG, oc: str) -> bool:
        """Normalize a single write-only reduction output ``oc``. Returns True on rewrite."""
        inner = nsdfg.sdfg
        oc_desc = inner.arrays.get(oc)
        if oc_desc is None:
            return False

        # Fail-safe refuse: never apply the drop-WCR / whole-buffer ``_nnr_out`` rewrite to a
        # DATA-DEPENDENT MULTI-ELEMENT scatter sink (``hist[bin[i]] (op)= w``). That rewrite
        # assumes the write covers all of ``oc`` and a write-only ``oc`` starts at the op
        # identity -- true for a scalar accumulator, but a scatter writes exactly ONE element
        # of a bounded ``oc`` per iteration, so materializing a per-iteration whole-array
        # buffer reads its other n-1 elements back uninitialised (garbage histogram). This
        # holds REGARDLESS of op (``+ - * / min max`` ...) and device: an un-privatised
        # scatter -- GPU (where ``privatize_scatter_reductions`` is off), or a non-reducible
        # op like ``-`` that ``PrivatizeScatterReduction`` refuses -- falls back to the
        # correct per-element ``reduce_atomic`` / atomicAdd instead of a broken whole-buffer
        # reduction. On CPU with a reducible op, ``PrivatizeScatterReduction`` runs first and
        # surfaces the WCR (making the outer edge non-plain), so this method already no-ops
        # there; the guard is the backstop for every path that pass does not cover.
        if is_data_dependent_scatter_sink(nsdfg, oc):
            return False

        # (a) A scalar WCR edge inside the body writing `oc` into a pure-sink AccessNode.
        wcr_edge = None
        wcr_state = None
        for ist in inner.states():
            for e in ist.edges():
                if (
                    e.data is not None
                    and e.data.wcr is not None
                    and e.data.data == oc
                    and e.data.subset is not None
                    and e.data.subset.num_elements() == 1
                    and isinstance(e.dst, nodes.AccessNode)
                    and e.dst.data == oc
                    and ist.out_degree(e.dst) == 0
                ):
                    if wcr_edge is not None:
                        return False  # more than one write-only reduction sink -> not this shape
                    wcr_edge, wcr_state = e, ist
        if wcr_edge is None:
            return False

        op = _op_from_wcr(wcr_edge.data.wcr)
        if op is None:
            return False

        # (b) The NestedSDFG -> MapExit edge for `oc` is plain, or carries the body's own WCR, which the frontend
        # surfaces onto the boundary of a nested SDFG writing its connector through a WCR.
        out_edge = next((oe for oe in state.out_edges(nsdfg) if oe.src_conn == oc), None)
        if out_edge is None or not isinstance(out_edge.dst, nodes.MapExit):
            return False
        if out_edge.data.wcr is not None and _op_from_wcr(out_edge.data.wcr) != op:
            return False
        # The rewrite buffers the whole connector per iteration; a frontend connector is the outer container, so
        # only a single-element one is the accumulator itself.
        if out_edge.data.wcr is not None and symbolic.equal(oc_desc.total_size, 1) is not True:
            return False

        # (c) The map-level private buffer mirrors `oc`'s inner descriptor. If that
        # descriptor's shape references symbols defined only inside the NestedSDFG (e.g.
        # a body-local symbol aliased to an outer one via `symbol_mapping`), materializing
        # the buffer at the outer scope would leak them as free symbols. Leave the shape
        # untouched in that case (the baseline reduction stays correct as-is).
        if {str(s) for s in oc_desc.free_symbols} - set(state.sdfg.symbols.keys()):
            return False

        wcr_str = wcr_edge.data.wcr
        # Rewrite the body: surface the reduction to the boundary edge chain.
        # Two body shapes, chosen by whether the WCR write runs UNCONDITIONALLY exactly
        # once per NestedSDFG invocation (:meth:`_write_runs_unconditionally_once`):
        #
        # * Unconditional write-once (post-``PredicateMaskedReduction``: the mask is folded
        #   into the addend as ``ITE(mask, val, identity)``): just DROP the WCR on the body
        #   edge -- ``oc (CR)= addend`` becomes ``oc = addend``, exact because a write-only
        #   ``oc`` starts at the op's identity by WCR first-write semantics. This keeps the
        #   body a SINGLE-STATE dataflow chain the tile vectorizer can widen to K lanes; the
        #   seeded ``_nnr_priv`` form below is a 3-state sub-CFG (seed -> accumulate ->
        #   copyback) the widener cannot widen, which strands K-1 lanes of the reduction
        #   buffer uninitialised (all-zero output).
        # * Otherwise (a ConditionalBlock survived -> STILL masked, or an inner loop ->
        #   multiple writes per invocation): keep the seeded body-local ``_nnr_priv``
        #   accumulator. Never surface a still-masked write as unconditional -- that would
        #   run the masked body on every lane.
        if self._write_runs_unconditionally_once(inner, wcr_state, oc, wcr_edge):
            plain = copy.deepcopy(wcr_edge.data)
            plain.wcr = None
            wcr_state.add_edge(wcr_edge.src, wcr_edge.src_conn, wcr_edge.dst, wcr_edge.dst_conn, plain)
            wcr_state.remove_edge(wcr_edge)
        else:
            priv_desc = self._seed_desc(oc_desc)
            priv = inner.add_datadesc(f"_nnr_priv_{oc}", priv_desc, find_new_name=True)
            priv_node = wcr_state.add_access(priv)
            # ``_nnr_priv`` sums the addends a subtraction takes away: the boundary's ``a - b`` subtracts that sum once.
            inner_wcr = "lambda a, b: a + b" if _binop_expr_from_wcr(wcr_str) == "-" else wcr_str
            # The whole memlet path into the accumulator names it, through every scope the write leaves.
            for edge in wcr_state.memlet_path(wcr_edge):
                if edge is not wcr_edge and edge.data.data == oc:
                    edge.data.data = priv
                    if edge.data.wcr is not None:
                        edge.data.wcr = inner_wcr
            new_data = copy.deepcopy(wcr_edge.data)
            new_data.data = priv
            new_data.wcr = inner_wcr
            wcr_state.add_edge(wcr_edge.src, wcr_edge.src_conn, priv_node, None, new_data)
            old_sink = wcr_edge.dst
            wcr_state.remove_edge(wcr_edge)
            if wcr_state.degree(old_sink) == 0:
                wcr_state.remove_node(old_sink)
            self._seed_state(inner, priv, priv_desc, op)
            self._copyback_state(inner, priv, oc, oc_desc)

        # Rewrite the map level: put the reduction on the accumulator edge chain.
        # The WCR must source from an AccessNode: a WCR left on the NestedSDFG->MapExit
        # edge is a *pointer*-typed connector, which the CPU codegen's WCR path drops
        # (see ``source_from_access_nodes``). Insert a per-iteration private AccessNode between the
        # NestedSDFG output and the MapExit so the reduction is AccessNode-sourced.
        out_priv_desc = self._seed_desc(oc_desc)
        out_priv = state.sdfg.add_datadesc(f"_nnr_out_{oc}", out_priv_desc, find_new_name=True)
        out_priv_node = state.add_access(out_priv)
        state.add_edge(nsdfg, oc, out_priv_node, None, Memlet(data=out_priv, subset=self._full_subset(out_priv_desc)))
        acc_node = None
        for e in state.memlet_path(out_edge):
            e.data.wcr = wcr_str
            if isinstance(e.dst, nodes.AccessNode):
                acc_node = e.dst
        state.add_edge(out_priv_node, None, out_edge.dst, out_edge.dst_conn, copy.deepcopy(out_edge.data))
        state.remove_edge(out_edge)
        if op in ("min", "max") and acc_node is not None:
            self._ensure_outer_seed(state.sdfg, acc_node.data, op)
        return True

    def _write_runs_unconditionally_once(self, inner: SDFG, wcr_state: SDFGState, oc: str, wcr_edge) -> bool:
        """True if the body WCR write of ``oc`` executes exactly once, unconditionally,
        per NestedSDFG invocation -- so its accumulation can be surfaced to the boundary
        WITHOUT a seeded body-local ``_nnr_priv`` (see :meth:`_rewrite_nsdfg_output`).

        Requires (both, else the seeded path is the sound choice):

        * No enclosing :class:`ConditionalBlock` or :class:`LoopRegion` between the write
          state and the NestedSDFG. A surviving ConditionalBlock means the reduction is
          STILL masked (``PredicateMaskedReduction`` did not fold the guard into the
          addend); an enclosing loop means the write fires several times per invocation.
          In either case ``oc = addend`` (drop-WCR) would be wrong -- a masked lane would
          execute unconditionally, or only the last loop write would survive.
        * ``oc`` has no OTHER writer in the body: the single WCR write is its sole
          definition, so dropping the WCR leaves exactly one plain write of ``oc``.
        """
        block = wcr_state.parent_graph
        while block is not None and not isinstance(block, SDFG):
            if isinstance(block, (ConditionalBlock, LoopRegion)):
                return False
            block = block.parent_graph
        for ist in inner.states():
            for e in ist.edges():
                if e is wcr_edge:
                    continue
                if (
                    e.data is not None
                    and e.data.data == oc
                    and isinstance(e.dst, nodes.AccessNode)
                    and e.dst.data == oc
                ):
                    return False
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
        for st in sdfg.states():
            for e in st.edges():
                # A copy into the accumulator seeds it too, though its memlet may name the source
                # (``table -> _priv_table``, a privatized reduction seeded from the array it updates).
                if (
                    e.data is not None
                    and not e.data.is_empty()
                    and e.data.wcr is None
                    and isinstance(e.dst, nodes.AccessNode)
                    and e.dst.data == acc
                ):
                    return
        val = _identity_value(op, desc.dtype)
        seed = sdfg.add_state_before(sdfg.start_block, label="_nnr_outer_seed", is_start_block=True)
        w = seed.add_write(acc)
        if isinstance(desc, data.Scalar) or tuple(desc.shape) == (1,):
            t = seed.add_tasklet("_nnr_outer_seed", {}, {"__out"}, f"__out = {val}")
            seed.add_edge(t, "__out", w, None, Memlet(data=acc, subset="0"))
        else:
            me, mx = seed.add_map("_nnr_outer_seed", {f"_nnr_i{d}": f"0:{s}" for d, s in enumerate(desc.shape)})
            t = seed.add_tasklet("_nnr_outer_seed", {}, {"__out"}, f"__out = {val}")
            idx = ", ".join(f"_nnr_i{d}" for d in range(len(desc.shape)))
            seed.add_edge(me, None, t, None, Memlet())
            seed.add_memlet_path(t, mx, w, src_conn="__out", memlet=Memlet(data=acc, subset=idx))

    def _fresh_symbol(self, sdfg: SDFG, base: str) -> str:
        """A map-parameter name unused as a symbol/free-symbol in ``sdfg``."""
        used = set(sdfg.symbols.keys()) | {str(s) for s in sdfg.free_symbols} | set(scopes.scoped_names(sdfg))
        name, i = base, 0
        while name in used:
            i += 1
            name = f"{base}_{i}"
        return name

    def _find_inner_scatter(self, inner: SDFG, oc: str) -> tuple | None:
        """The inner ``(state, MapEntry, tasklet, out_conn, param)`` whose single-element
        ``oc[param]`` WCR write aggregates to the slice output ``oc``. ``None`` unless the
        producer is exactly one single-param per-element scatter map.
        """
        for ist in inner.states():
            for tasklet in [n for n in ist.nodes() if isinstance(n, nodes.Tasklet)]:
                for oe in ist.out_edges(tasklet):
                    if (
                        oe.data is not None
                        and oe.data.data == oc
                        and oe.data.wcr is not None
                        and oe.data.subset is not None
                        and oe.data.subset.num_elements() == 1
                        and isinstance(oe.dst, nodes.MapExit)
                    ):
                        ime = ist.entry_node(oe.dst)
                        if len(ime.map.params) != 1:
                            return None
                        return ist, ime, tasklet, oe.src_conn, ime.map.params[0]
        return None

    def _extract_slice_wcr(self, state: SDFGState, nsdfg: nodes.NestedSDFG, oc: str) -> bool:
        """Extract a trapped per-element scatter (slice-WCR output ``oc``) to the outer
        scope as a single-element ``dest[...]`` WCR. Returns True on rewrite.

        Detection is complete before any mutation: an unhandled shape returns False with
        the graph untouched.
        """
        out_edge = next((oe for oe in state.out_edges(nsdfg) if oe.src_conn == oc), None)
        if (
            out_edge is None
            or out_edge.data.wcr is None
            or out_edge.data.subset is None
            or out_edge.data.subset.num_elements() == 1
            or not isinstance(out_edge.dst, nodes.MapExit)
        ):
            return False
        if _op_from_wcr(out_edge.data.wcr) is None:
            return False
        prod = self._find_inner_scatter(nsdfg.sdfg, oc)
        if prod is None:
            return False
        istate, ime, tasklet, tconn, iparam = prod
        outer_me = state.entry_node(nsdfg)
        outer_mx = state.exit_node(outer_me)
        dest = out_edge.data.data
        dest_an = next(
            (
                oe.dst
                for oe in state.out_edges(outer_mx)
                if oe.data is not None and oe.data.data == dest and isinstance(oe.dst, nodes.AccessNode)
            ),
            None,
        )
        if dest_an is None:
            return False
        # Every tasklet input must trace to a direct nsdfg boundary connector fed by a
        # top-level source at the outer map. Resolve the whole plan before mutating.
        boundary_in = {oe.dst_conn: oe for oe in state.in_edges(nsdfg)}
        top_in: dict[str, Any] = {}
        for ie in state.in_edges(outer_me):
            if ie.data is not None:
                top_in.setdefault(ie.data.data, ie.src)
        plan = []
        for ie in istate.in_edges(tasklet):
            ext = boundary_in.get(ie.data.data)
            if ext is None or ext.data.data not in top_in:
                return False
            plan.append((ie, ext, top_in[ext.data.data]))

        # mutate: clone the scatter map to the outer scope
        ksym, nksym = symbol(iparam), symbol(self._fresh_symbol(state.sdfg, "_wcr_" + iparam))
        new_me, new_mx = state.add_map("extract_" + oc, {str(nksym): str(ime.map.range)})
        new_t = state.add_tasklet(
            "extract_" + oc, tasklet.in_connectors.keys(), tasklet.out_connectors.keys(), tasklet.code.as_string
        )
        # The connectors are the containers (the nested SDFG contract): an inner memlet already indexes the outer
        # container and only takes its name.
        for ie, ext, top_src in plan:
            ml = copy.deepcopy(ie.data)
            ml.data = ext.data.data
            ml.subset.replace({ksym: nksym})
            state.add_memlet_path(top_src, outer_me, new_me, new_t, dst_conn=ie.dst_conn, memlet=ml)
        for oe in istate.out_edges(tasklet):
            if oe.src_conn == tconn:
                ml = copy.deepcopy(oe.data)
                ml.data = dest
                ml.subset.replace({ksym: nksym})
                ml.wcr = None
                state.add_memlet_path(new_t, new_mx, outer_mx, dest_an, src_conn=oe.src_conn, memlet=ml)
                # WCR lives ONLY on the innermost single-element edge; the re-widen edges
                # up to the accumulator stay plain (the shape LoopToMap emits natively).
                inner_edge = next(e for e in state.out_edges(new_t) if e.src_conn == oe.src_conn)
                for pe in state.memlet_path(inner_edge):
                    pe.data.wcr = out_edge.data.wcr if pe is inner_edge else None
            else:
                # Non-scatter co-output (e.g. a scalar reduction sharing the tasklet) is
                # recomputed into a dead scalar; DCE prunes it.
                inner_desc = nsdfg.sdfg.arrays[oe.data.data]
                dname, _ = state.sdfg.add_scalar(
                    "_wcrdead_" + oe.src_conn, inner_desc.dtype, transient=True, find_new_name=True
                )
                state.add_memlet_path(
                    new_t, new_mx, state.add_access(dname), src_conn=oe.src_conn, memlet=Memlet(data=dname, subset="0")
                )

        # Redirect the trapped nsdfg slice output to a dead transient (mirror dest's
        # descriptor so no inner-only symbol leaks into the shape); drop the slice WCR.
        dead_desc = copy.deepcopy(state.sdfg.arrays[dest])
        dead_desc.transient = True
        dead = state.sdfg.add_datadesc("_wcrdead_" + dest, dead_desc, find_new_name=True)
        redir = copy.deepcopy(out_edge.data)
        redir.data = dead
        redir.wcr = None
        dst_conn = out_edge.dst_conn
        state.remove_edge(out_edge)
        state.add_edge(nsdfg, oc, state.add_access(dead), None, redir)
        # If the slice used a dedicated MapExit connector (now unfed), drop it + its pair.
        if dst_conn is not None and not any(e.dst_conn == dst_conn for e in state.in_edges(outer_mx)):
            out_conn = "OUT_" + dst_conn[3:] if dst_conn.startswith("IN_") else None
            for oe in list(state.out_edges(outer_mx)):
                if oe.src_conn == out_conn:
                    state.remove_edge(oe)
            outer_mx.remove_in_connector(dst_conn)
            if out_conn is not None and out_conn in outer_mx.out_connectors:
                outer_mx.remove_out_connector(out_conn)
        return True

    def _rewrite_readwrite_output(self, state: SDFGState, nsdfg: nodes.NestedSDFG, oc: str) -> bool:
        """Rewrite a read-write output connector whose in-body scalar WCR is redundant
        with a WCR on the ``NestedSDFG -> MapExit`` edge.

        The shape is the one the frontend emits for an in-place aug-assign such as
        ``b[i] += data[i]`` inside a map-body NestedSDFG: ``oc`` is both an input and an
        output connector, the body WCRs into ``AccessNode(oc)``, and the boundary edge
        also carries the same WCR. That double accumulation is wrong, so we:

        1. Re-route the addend produced by the original source tasklet into a fresh
           body-local transient ``_nnr_addend_<oc>``.
        2. Read the incoming ``oc`` value and the addend into a new tasklet that emits
           ``oc_out = oc OP addend`` as a *plain* write.
        3. Write the result into a fresh output connector ``_nnr_rwout_<oc>`` and remove
           the WCR from the boundary edge, so each iteration writes its fully-updated
           element back.

        The rewrite is only applied when the interior WCR is scalar, sourced from a
        Tasklet, and uses a supported associative aug-assign op (``+``, ``-``, ``*``,
        ``min``, ``max``). It is a no-op on a second run because the new output is
        write-only and has no interior WCR.
        """
        if oc not in nsdfg.in_connectors or oc not in nsdfg.out_connectors:
            return False

        inner = nsdfg.sdfg
        oc_desc = inner.arrays.get(oc)
        if oc_desc is None:
            return False

        # (a) The body writes the connector array `oc` exactly once, through a scalar WCR edge. Any other
        # write would lose its output connector once `oc` becomes input-only.
        writes = [
            (e, ist)
            for ist in inner.states()
            for e in ist.edges()
            if isinstance(e.dst, nodes.AccessNode) and e.dst.data == oc and not e.data.is_empty()
        ]
        if len(writes) != 1:
            return False
        wcr_edge, wcr_state = writes[0]
        if (
            wcr_edge.data.wcr is None
            or wcr_edge.data.data != oc
            or wcr_edge.data.subset is None
            or wcr_edge.data.subset.num_elements() != 1
            or wcr_state.out_degree(wcr_edge.dst) != 0
        ):
            return False

        op = _binop_expr_from_wcr(wcr_edge.data.wcr)
        if op is None:
            return False

        src = wcr_edge.src
        src_conn = wcr_edge.src_conn
        if src_conn is None or not isinstance(src, nodes.Tasklet):
            return False

        # (b) The boundary edge for `oc` must carry the WCR (redundant with the body WCR).
        out_edge = next((oe for oe in state.out_edges(nsdfg) if oe.src_conn == oc), None)
        if out_edge is None or out_edge.data.wcr is None or not isinstance(out_edge.dst, nodes.MapExit):
            return False
        if out_edge.data.subset is None or out_edge.data.subset.num_elements() != 1:
            return False

        # Do not leak symbols that are only defined inside the NestedSDFG.
        if {str(s) for s in oc_desc.free_symbols} - set(state.sdfg.symbols.keys()):
            return False

        # A plain read-modify-write is only race-free when each map iteration updates its own element. An
        # element no map parameter selects is a reduction across the iterations, and keeps its WCR.
        params = set(state.entry_node(nsdfg).map.params)
        selecting = {
            inner
            for inner, outer in nsdfg.symbol_mapping.items()
            if {str(s) for s in symbolic.pystr_to_symbolic(str(outer)).free_symbols} & params
        }
        if not {str(s) for s in wcr_edge.data.subset.free_symbols} & selecting:
            return False

        # Rewrite the body: explicit plain read-modify-write of the one element the WCR updates. The connector
        # is the container, so that element is the WCR edge's subset, not element 0.
        element = wcr_edge.data.subset
        addend, _ = inner.add_scalar(f"_nnr_addend_{oc}", oc_desc.dtype, transient=True, find_new_name=True)

        # Fresh output connector array (non-transient, because it is a connector).
        out_desc = copy.deepcopy(oc_desc)
        out_desc.transient = False
        out_conn_name = inner.add_datadesc(f"_nnr_rwout_{oc}", out_desc, find_new_name=True)
        nsdfg.remove_out_connector(oc)
        nsdfg.add_out_connector(out_conn_name, out_desc.dtype)

        # Access nodes for the explicit RMW.
        old_node = wcr_state.add_read(oc)
        addend_node = wcr_state.add_access(addend)
        out_node = wcr_state.add_write(out_conn_name)

        # Reroute the addend source onto the scratch.
        wcr_state.add_edge(src, src_conn, addend_node, None, Memlet(data=addend, subset="0"))

        # New tasklet: __out = __old OP __addend (plain).
        dtype = oc_desc.dtype
        rmw = wcr_state.add_tasklet(
            "_nnr_rmw", {"__old": dtype, "__addend": dtype}, {"__out": dtype}, _rmw_tasklet_code(op)
        )
        wcr_state.add_edge(old_node, None, rmw, "__old", Memlet(data=oc, subset=copy.deepcopy(element)))
        wcr_state.add_edge(addend_node, None, rmw, "__addend", Memlet(data=addend, subset="0"))
        wcr_state.add_edge(rmw, "__out", out_node, None, Memlet(data=out_conn_name, subset=copy.deepcopy(element)))

        # Remove the old redundant WCR edge and its sink if orphaned.
        old_sink = wcr_edge.dst
        wcr_state.remove_edge(wcr_edge)
        if wcr_state.degree(old_sink) == 0:
            wcr_state.remove_node(old_sink)

        # Rewrite the boundary: plain copy from the new output connector.
        path = list(state.memlet_path(out_edge))
        for e in path:
            e.data.wcr = None
        state.remove_edge(out_edge)
        state.add_edge(nsdfg, out_conn_name, out_edge.dst, out_edge.dst_conn, copy.deepcopy(out_edge.data))
        return True

    def normalize_nested_reductions(self, sdfg: SDFG) -> int:
        """Rewrite every map-body NestedSDFG reduction output (the write-only and read-write shapes).

        :returns: The number of rewritten outputs.
        """
        total = 0
        for sd in sdfg.all_sdfgs_recursive():
            for state in sd.states():
                for nsdfg in [n for n in state.nodes() if isinstance(n, nodes.NestedSDFG)]:
                    total += self.normalize_nested_node(state, nsdfg)
        return total

    def normalize_nested_node(self, state: SDFGState, nsdfg: nodes.NestedSDFG) -> int:
        """Rewrite the reduction outputs of the map-body NestedSDFG ``nsdfg``; a pass nesting a map body after this
        one ran calls it on the nest it makes.

        :returns: The number of rewritten outputs.
        """
        if not isinstance(state.entry_node(nsdfg), nodes.MapEntry):
            return 0
        total = 0
        write_only = [oc for oc in nsdfg.out_connectors if oc not in nsdfg.in_connectors]
        for oc in write_only:
            # Slice WCR (num_elements > 1) -> extract; scalar WCR -> seed-local rewrite.
            if self.extract_slice_wcr and self._extract_slice_wcr(state, nsdfg, oc):
                total += 1
            elif self._rewrite_nsdfg_output(state, nsdfg, oc):
                total += 1
        readwrite = [oc for oc in nsdfg.out_connectors if oc in nsdfg.in_connectors]
        for oc in readwrite:
            if self._rewrite_readwrite_output(state, nsdfg, oc):
                total += 1
        return total

    @staticmethod
    def partial_source_descriptor(
        src: nodes.CodeNode, src_conn: str, target_desc: data.Data | None
    ) -> data.Data | None:
        """The descriptor the private partial mirrors: a NestedSDFG's inner connector array, or a Scalar of the WCR
        target's element type for a Tasklet (its out connectors stay uninferred)."""
        if isinstance(src, nodes.NestedSDFG):
            return src.sdfg.arrays.get(src_conn)
        if isinstance(src, nodes.Tasklet) and target_desc is not None:
            return data.Scalar(target_desc.dtype)
        return None

    @staticmethod
    def partial_descriptor(inner: data.Data) -> data.Data:
        """A Scope-lifetime transient matching ``inner`` element-wise: a Scalar for one element, else a copy."""
        if isinstance(inner, data.Scalar) or (isinstance(inner, data.Array) and tuple(inner.shape) == (1,)):
            return data.Scalar(
                inner.dtype,
                transient=True,
                storage=dtypes.StorageType.Default,
                lifetime=dtypes.AllocationLifetime.Scope,
            )
        new = copy.deepcopy(inner)
        new.transient = True
        new.storage = dtypes.StorageType.Default
        new.lifetime = dtypes.AllocationLifetime.Scope
        return new

    @staticmethod
    def enclosing_map_params(state: SDFGState, e, scope: dict) -> set[str]:
        """Parameters of every map enclosing ``e``'s source, plus the destination map's when ``e`` enters its exit.
        ``scope`` is the state's pre-rewrite ``scope_dict()``."""
        params: set[str] = set()
        node = scope.get(e.src)
        while node is not None:
            if isinstance(node, nodes.MapEntry):
                params.update(node.map.params)
            node = scope.get(node)
        if isinstance(e.dst, nodes.MapExit):
            params.update(state.entry_node(e.dst).map.params)
        return params

    def is_write_once_wcr(self, state: SDFGState, e, scope: dict) -> bool:
        """True if the WCR writes a slot that varies with an enclosing (or destination) map parameter, not a fold into
        one constant slot. Only such a fresh accumulator is identity-seeded: a constant-slot fold either has its own
        init or continues a live value threaded in under another name (nussinov's ``_priv_table``)."""
        if e.data.subset is None:
            return False
        subsyms = {str(x) for x in e.data.subset.free_symbols}
        if not subsyms:
            return False
        return bool(self.enclosing_map_params(state, e, scope) & subsyms)

    @staticmethod
    def folds_into_multi_element_connector(src: nodes.NestedSDFG, src_conn: str, priv_desc: data.Data) -> bool:
        """True if ``src``'s body folds into (WCR-writes or reads) a multi-element output connector ``src_conn``.

        Such a connector is itself an accumulator the body only partly updates; a whole-buffer partial copies back
        slots the iteration never wrote (unseeded). A single-element connector the body folds into is covered whole.
        """
        if isinstance(priv_desc, data.Scalar):
            return False
        return accumulates_into(src.sdfg, {src_conn})

    def rewrite_wcr_sources(self, sdfg: SDFG, state: SDFGState, requests: dict[str, list]) -> int:
        """Source every WCR edge of ``state`` a NestedSDFG writes, or a Tasklet writes into a map exit, from a fresh
        private AccessNode; record each WCR destination in ``requests`` (``name -> [wcr, all write-once, writing
        states]``) for seeding. A Tasklet WCR into an access node stays: codegen resolves it on its scalar
        connector, and a private access node would only turn it into an accumulating copy.

        :returns: The number of rewritten edges.
        """
        targets = [
            e
            for e in state.edges()
            if e.data is not None
            and e.data.wcr is not None
            and (
                isinstance(e.src, nodes.NestedSDFG)
                or (isinstance(e.src, nodes.Tasklet) and isinstance(e.dst, nodes.MapExit))
            )
        ]
        if not targets:
            return 0
        # The write-once test reads the pre-rewrite topology; the inserted access nodes invalidate the scope cache.
        scope = state.scope_dict()
        write_once = {e: self.is_write_once_wcr(state, e, scope) for e in targets}
        rewritten = 0
        for e in targets:
            src, src_conn = e.src, e.src_conn
            if src_conn is None:
                continue
            # An in/out connector split on its out side only would bind one name to two outer arrays.
            if isinstance(src, nodes.NestedSDFG) and src_conn in src.in_connectors:
                continue
            # A data-dependent scatter accumulates on its inner per-element WCR; a whole-array partial reads back the
            # elements the iteration never wrote (``PrivatizeScatterReduction`` owns the reducible ones).
            if isinstance(src, nodes.NestedSDFG) and is_data_dependent_scatter_sink(src, src_conn):
                continue
            target_desc = sdfg.arrays.get(e.data.data) if e.data.data else None
            inner = self.partial_source_descriptor(src, src_conn, target_desc)
            if inner is None:
                continue
            priv_desc = self.partial_descriptor(inner)
            if isinstance(src, nodes.NestedSDFG) and self.folds_into_multi_element_connector(src, src_conn, priv_desc):
                continue
            # A partial sized by a symbol this SDFG does not define (a triangular ``i + 1`` extent) would leak it.
            if {str(sym) for sym in priv_desc.free_symbols} - set(sdfg.symbols.keys()):
                continue
            priv_name = sdfg.add_datadesc(f"_wcr_priv_{src.label}_{src_conn}", priv_desc, find_new_name=True)
            priv_node = state.add_access(priv_name)
            state.add_edge(src, src_conn, priv_node, None, Memlet(data=priv_name, subset=self._full_subset(priv_desc)))
            state.add_edge(priv_node, None, e.dst, e.dst_conn, copy.deepcopy(e.data))
            state.remove_edge(e)
            rewritten += 1
            if e.data.data:
                entry = requests.setdefault(e.data.data, [e.data.wcr, True, set()])
                entry[1] = entry[1] and write_once[e]
                entry[2].add(state)
        return rewritten

    @staticmethod
    def plain_written(sd: SDFG) -> set[str]:
        """Every name in ``sd`` an access node receives a plain (non-WCR) write into, a copy whose memlet names its
        source included."""
        out: set[str] = set()
        for st in sd.states():
            for e in st.edges():
                if (
                    e.data is not None
                    and e.data.wcr is None
                    and not e.data.is_empty()
                    and isinstance(e.dst, nodes.AccessNode)
                ):
                    out.add(e.dst.data)
        return out

    @staticmethod
    def is_fresh_storage(sd: SDFG, name: str, desc: data.Data) -> bool:
        """``name`` is DaCe-owned scratch safe to seed: a transient, or an out-only nested connector every caller binds
        to a transient, and no enclosing map iteration folds into the same caller slots (an out-only connector may
        alias a live caller array: gramschmidt's ``__tmp_78 -> &A[j]``)."""
        if desc.transient:
            return True
        parent = sd.parent_nsdfg_node
        if parent is None:
            return False
        if name in parent.in_connectors or name not in parent.out_connectors:
            return False
        parent_state, parent_sdfg = sd.parent, sd.parent_sdfg
        if parent_state is None or parent_sdfg is None:
            return False
        bound = [oe for oe in parent_state.out_edges(parent) if oe.src_conn == name]
        if not bound:
            return False
        params: set[str] = set()
        scope = parent_state.entry_node(parent)
        while scope is not None:
            params |= set(scope.map.params)
            scope = parent_state.entry_node(scope)
        for oe in bound:
            cdesc = parent_sdfg.arrays.get(oe.data.data)
            if cdesc is None or not cdesc.transient:
                return False
            if oe.data.wcr is not None and params and not {str(sym) for sym in oe.data.subset.free_symbols} & params:
                return False
        return True

    def seedable_accumulator(self, sd: SDFG, name: str, plain_written: set[str]) -> bool:
        """True if ``name`` is fresh storage with no plain writer and no shape symbol ``sd`` does not define."""
        desc = sd.arrays.get(name)
        if desc is None or name in plain_written:
            return False
        if {str(sym) for sym in desc.free_symbols} - set(sd.symbols.keys()):
            return False
        return self.is_fresh_storage(sd, name, desc)

    @staticmethod
    def enclosing_loops(block) -> list[LoopRegion]:
        """The LoopRegion ancestors of ``block``, innermost first."""
        loops: list[LoopRegion] = []
        cur = block.parent_graph
        while cur is not None:
            if isinstance(cur, LoopRegion):
                loops.append(cur)
            cur = cur.parent_graph if isinstance(cur, ControlFlowRegion) else None
        return loops

    def seed_placement(self, name: str, states: set[SDFGState]):
        """The block the seed goes before: the innermost enclosing loop summed over (its variable does not index
        ``name``), else the reducing state; ``None`` when ambiguous (several states, several summed loops)."""
        if len(states) != 1:
            return None
        state = next(iter(states))
        idx_syms: set[str] = set()
        for e in state.edges():
            if (
                e.data is not None
                and e.data.data == name
                and e.data.wcr is not None
                and isinstance(e.dst, nodes.AccessNode)
                and e.dst.data == name
            ):
                sub = e.data.get_dst_subset(e, state) or e.data.subset
                if sub is not None:
                    idx_syms |= {str(sym) for sym in sub.free_symbols}
        summed = [
            ln for ln in self.enclosing_loops(state) if ln.loop_variable and str(ln.loop_variable) not in idx_syms
        ]
        if not summed:
            return state
        return summed[0] if len(summed) == 1 else None

    @staticmethod
    def identity_literal(val: Any, dtype: dtypes.typeclass) -> str:
        """A codegen-safe literal for the reduction identity at ``dtype`` (a Python scalar, never ``np.float64(...)``)."""
        nptype = dtype.type
        if numpy.issubdtype(nptype, numpy.bool_):
            return "True" if val else "False"
        if numpy.issubdtype(nptype, numpy.complexfloating):
            c = complex(val)
            return f"complex({c.real!r}, {c.imag!r})"
        if numpy.issubdtype(nptype, numpy.integer):
            return str(int(val))
        return repr(float(val))

    @staticmethod
    def written_window(
        sd: SDFG, name: str, state: SDFGState, desc: data.Data, defined: set[str]
    ) -> subsets.Range | None:
        """The bounding box of the reduction's WCR writes into ``name`` in ``state``; ``None`` when a write has no
        subset or a bound names a symbol neither ``sd`` nor ``defined`` defines."""
        window = None
        for e in state.edges():
            if not (isinstance(e.dst, nodes.AccessNode) and e.dst.data == name and e.data.wcr is not None):
                continue
            sub = e.data.get_dst_subset(e, state) or e.data.subset
            if sub is None or len(sub) != len(desc.shape):
                return None
            window = copy.deepcopy(sub) if window is None else subsets.bounding_box_union(window, sub, sd.facts())
        if window is None or {str(sym) for sym in window.free_symbols} - set(sd.symbols.keys()) - defined:
            return None
        return window

    def seed_accumulator(self, sd: SDFG, name: str, wcr: str, states: set[SDFGState]) -> bool:
        """Prepend a state writing the reduction identity into the fresh accumulator ``name`` (a tasklet, not
        ``setzero``: ``*``, ``min`` and ``max`` have other identities).

        :returns: Whether a seed was emitted.
        """
        desc = sd.arrays.get(name)
        if desc is None:
            return False
        red = detect_reduction_type(wcr)
        if red is None:
            return False
        try:
            val = dtypes.reduction_identity(desc.dtype, red)
        except (TypeError, ValueError):
            return False
        if val is None:
            return False
        before = self.seed_placement(name, states)
        if before is None:
            return False
        window = None
        if not (isinstance(desc, data.Scalar) or tuple(desc.shape) == (1,)):
            # A seed inside loops runs every iteration: it may only reset the window those iterations write.
            loop_vars = {str(loop.loop_variable) for loop in self.enclosing_loops(before) if loop.loop_variable}
            window = self.written_window(sd, name, next(iter(states)), desc, loop_vars)
            if window is None:
                if loop_vars:
                    return False
                window = subsets.Range([(0, extent - 1, 1) for extent in desc.shape])
        parent: ControlFlowRegion = before.parent_graph
        seed = parent.add_state_before(before, label=name + "_wcr_seed", is_start_block=parent.start_block is before)
        lit = self.identity_literal(val, desc.dtype)
        w = seed.add_write(name)
        if isinstance(desc, data.Scalar) or tuple(desc.shape) == (1,):
            t = seed.add_tasklet(name + "_wcr_seed", {}, {"__out"}, f"__out = {lit}")
            seed.add_edge(t, "__out", w, None, Memlet(data=name, subset="0"))
        else:
            me, mx = seed.add_map(name + "_wcr_seed", {f"_wcrseed_i{d}": f"0:{s}" for d, s in enumerate(desc.shape)})
            me.map.range = window
            t = seed.add_tasklet(name + "_wcr_seed", {}, {"__out"}, f"__out = {lit}")
            idx = ", ".join(f"_wcrseed_i{d}" for d in range(len(desc.shape)))
            seed.add_edge(me, None, t, None, Memlet())
            seed.add_memlet_path(t, mx, w, src_conn="__out", memlet=Memlet(data=name, subset=idx))
        return True

    def source_from_access_nodes(self, sdfg: SDFG) -> int:
        """Source every WCR edge a NestedSDFG writes, or a Tasklet writes into a map exit, from a private AccessNode (:meth:`rewrite_wcr_sources`), then identity-seed each fresh accumulator they reduce into when every
        write into it is write-once (:meth:`seed_accumulator`).

        :returns: The number of rewritten edges.
        """
        total = 0
        for sd in list(sdfg.all_sdfgs_recursive()):
            requests: dict[str, list] = {}
            for state in list(sd.states()):
                total += self.rewrite_wcr_sources(sd, state, requests)
            if not requests:
                continue
            plain = self.plain_written(sd)
            for name, (wcr, all_write_once, states) in requests.items():
                if all_write_once and self.seedable_accumulator(sd, name, plain):
                    self.seed_accumulator(sd, name, wcr, states)
        return total

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> dict[str, set[str]] | None:
        """Normalize the map-body nested reductions, then source the WCR edges from access nodes.

        :param sdfg: The SDFG to normalize.
        :param pipeline_results: Results of prior passes in the pipeline (unused).
        :returns: ``None`` if nothing was rewritten; otherwise the counts under ``normalized_nested_reductions`` and
                  ``normalized_wcr_edges``.
        """
        nested = self.normalize_nested_reductions(sdfg)
        sourced = self.source_from_access_nodes(sdfg)
        if nested == 0 and sourced == 0:
            return None
        if self.validate:
            sdfg.validate()
        return {"normalized_nested_reductions": {str(nested)}, "normalized_wcr_edges": {str(sourced)}}
