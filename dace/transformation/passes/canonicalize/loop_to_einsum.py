# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lift a contraction / transpose loop nest to a single library node.

A sibling of :class:`~dace.transformation.passes.loop_to_reduce.LoopToReduce`: a
loop nest that computes one tensor contraction is replaced by a state holding a
single :class:`~dace.libraries.blas.nodes.einsum.Einsum` node, and a pure matrix
transpose by a :class:`~dace.libraries.linalg.nodes.transpose.Transpose` node.

Recognised contraction shapes (all as ``@dace.program`` loop nests)::

    for i:                for i: for j:              for i: for j: for k:
      for j:                y[j] += A[i,j]*x[i]        C[i,j] += A[i,k]*B[k,j]
        y[i] += A[i,j]*x[j] (transposed matvec:       (matmul: 'ij,jk->ik')
    (matvec: 'ij,j->i')     'ij,i->j')

and the pure transpose ``for i: for j: B[i,j] = A[j,i]`` (2-D, cross-array).

Direct structural matching is brittle: the Python frontend stages every indexed
read through a transient scalar, wraps loop bodies in connective states, and
encodes the ``+=`` accumulation as a read/compute/write chain rather than a WCR.
So the pass **probes** instead of matching directly:

1. Copy the loop into a throwaway SDFG whose boundary arrays are full-size and
   non-transient (so they look like arguments -- surviving simplification and
   giving the contraction the caller-provides-prior ``beta`` LiftEinsum expects).
2. Run a small ``parallelize + lift`` pipeline on the copy (Simplify, WCR
   conversion, tasklet fusion, LoopToMap, inline, MapCollapse, a redundant-scalar
   splice, then ``LiftEinsum``).
3. If the copy collapsed to exactly one ``Einsum`` (or one transpose-shaped map),
   read that node's spec -- its einsum string, alpha/beta, and the operand /
   output array names + memlet subsets -- and rebuild an equivalent node in the
   ORIGINAL SDFG, wired 1-to-1 to the original arrays by NAME (the copy preserves
   names). The loop nest is spliced out the way ``LoopToReduce._lift`` does.
4. Otherwise do nothing for this loop. The transform is opt-in and never a crash:
   the whole probe is wrapped in ``try/except`` so a probe failure is a clean
   no-op.

NOTE ON THE PIPELINE: the ``+=`` accumulation reaches LiftEinsum's canonical
``map -> single mul-tasklet -> WCR-sum`` shape only after (a) ``TaskletFusion``
folds the ``mul`` + accumulate tasklets into one, (b) ``AugAssignToWCR`` turns the
resulting ``y = y + a*b`` into a WCR write so ``LoopToMap`` will parallelise the
reduction axis, and (c) the frontend's per-read staging scalars sitting inside the
map scope are spliced out (no stock transform removes a redundant scalar between a
MapEntry and a tasklet). The (a)-before-(b) order is load-bearing: ``AugAssignToWCR``
only matches an accumulator whose tasklet inputs are all AccessNodes, and once
canonicalize has de-WCR'd a ``@dace.map`` accumulation (polybench atax's ``compute_y``,
a transposed matvec reducing over the OUTER axis onto a reset-then-accumulated ``y``)
the product feeds the add via a direct tasklet->tasklet edge -- so fusing must happen
first. ``MapFission`` -- part of the naive sequence -- is deliberately omitted: it
miscompiles this staged shape into an invalid (rank-mismatched) SDFG. Because the
pipeline runs on a disposable copy, an over-eager step can only cost a missed lift,
never corrupt the real SDFG.
"""
import copy
from typing import List, NamedTuple, Optional, Set, Tuple

from dace import SDFG, data, subsets
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.state import ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation.transformation import explicit_cf_compatible


class EinsumSpec(NamedTuple):
    """Everything needed to rebuild an ``Einsum`` node on the original arrays."""
    einsum_str: str
    alpha: object
    beta: object
    inputs: List[Tuple[str, str, subsets.Subset, object]]  # (connector, array, subset, dtype)
    output: Tuple[str, str, subsets.Subset, object]  # (connector, array, subset, dtype)


class TransposeSpec(NamedTuple):
    """Everything needed to rebuild a ``Transpose`` node on the original arrays."""
    src: str
    dst: str
    dtype: object
    src_subset: subsets.Subset
    dst_subset: subsets.Subset


def _is_single_element(desc) -> bool:
    """A ``Scalar`` or a length-1 ``Array`` -- the frontend's per-read staging shape."""
    return isinstance(desc, data.Scalar) or (isinstance(desc, data.Array) and all(str(s) == '1' for s in desc.shape))


def _referenced_arrays(loop: LoopRegion) -> Set[str]:
    """Every array name read or written anywhere inside ``loop``."""
    referenced: Set[str] = set()
    for state in loop.all_states():
        for dn in state.data_nodes():
            referenced.add(dn.data)
        for e in state.edges():
            if e.data is not None and e.data.data is not None:
                referenced.add(e.data.data)
    return referenced


def _written_arrays(loop: LoopRegion) -> Set[str]:
    """Array names written (have an in-edge to an AccessNode) inside ``loop``."""
    written: Set[str] = set()
    for state in loop.all_states():
        for dn in state.data_nodes():
            if state.in_degree(dn) > 0:
                written.add(dn.data)
    return written


def _live_outside(loop: LoopRegion, root: SDFG, referenced: Set[str]) -> Set[str]:
    """Referenced arrays that are also touched by data nodes OUTSIDE ``loop`` --
    i.e. cross the loop boundary and so must stay visible (non-transient) in the
    probe, unlike the loop's purely-internal staging scratch."""
    inside = set(id(s) for s in loop.all_states())
    live: Set[str] = set()
    for state in root.all_states():
        if id(state) in inside:
            continue
        for dn in state.data_nodes():
            if dn.data in referenced:
                live.add(dn.data)
    return live


def _has_loop_ancestor(loop: LoopRegion) -> bool:
    """True iff ``loop`` is nested inside another ``LoopRegion`` -- we probe only
    the OUTERMOST loop of a nest (probing it lifts, or no-ops on, the whole nest;
    an inner loop alone is a 1-D reduction, LoopToReduce's domain, not an einsum)."""
    p = loop.parent_graph
    while p is not None and not isinstance(p, SDFG):
        if isinstance(p, LoopRegion):
            return True
        p = p.parent_graph
    return False


def _build_probe(loop: LoopRegion, root: SDFG) -> Optional[SDFG]:
    """A throwaway SDFG wrapping a deep copy of ``loop``. Boundary arrays (live-out
    or non-transient in the original) become full-size non-transient descriptors;
    purely-internal staging scratch keeps its transient flag so simplification /
    the scalar splice can remove it."""
    referenced = _referenced_arrays(loop)
    if not referenced:
        return None
    live = _live_outside(loop, root, referenced)

    probe = SDFG('probe_' + loop.label)
    # sorted(): this fixes the probe's ARRAY INSERTION ORDER, which the probe pipeline (SimplifyPass +
    # apply_transformations_repeated + LiftEinsum) walks when enumerating matches. The lift test is
    # all-or-nothing, so a different order flips lift <-> no-lift for a whole nest -- a nest either collapses
    # into a library node or stays a loop that later becomes a Map. Iterating the raw set made that a
    # PYTHONHASHSEED coin-flip. Names are unique strings, so a stable sort is the canonical order.
    for name in sorted(referenced):
        if name not in root.arrays:
            return None
        desc = copy.deepcopy(root.arrays[name])
        if not desc.transient or name in live:
            desc.transient = False
        probe.add_datadesc(name, desc)

    # Carry over the symbols the loop and the descriptors depend on.
    needed: Set[str] = set(loop.used_symbols(all_symbols=True))
    for name in sorted(referenced):
        needed |= set(map(str, root.arrays[name].free_symbols))
    for sym in sorted(needed):  # symbol insertion order feeds free_symbols / arglist order in the probe
        if sym in root.symbols and sym not in probe.symbols and sym not in probe.arrays:
            probe.add_symbol(sym, root.symbols[sym])

    probe.add_node(copy.deepcopy(loop), is_start_block=True)
    return probe


def _splice_scope_scalars(probe: SDFG) -> None:
    """Splice out redundant single-element transient scalars that sit on the
    memlet path between a producer and a single tasklet consumer, connecting the
    producer straight to the tasklet. This clears the frontend's per-read staging
    (``MapEntry --A[i,j]--> A_index --> tasklet``) that no stock transform removes
    inside a map scope, leaving the scope holding exactly the compute tasklet --
    the shape ``LiftEinsum`` matches. Safe: the scalar is a pure pass-through."""
    changed = True
    while changed:
        changed = False
        for state in probe.states():
            for n in list(state.nodes()):
                if not isinstance(n, nodes.AccessNode):
                    continue
                desc = probe.arrays.get(n.data)
                if desc is None or not desc.transient or not _is_single_element(desc):
                    continue
                in_edges = state.in_edges(n)
                out_edges = state.out_edges(n)
                if len(in_edges) != 1 or len(out_edges) != 1:
                    continue
                e_in, e_out = in_edges[0], out_edges[0]
                if not isinstance(e_out.dst, nodes.Tasklet):
                    continue
                if e_in.data is None or e_in.data.is_empty():
                    continue
                state.add_edge(e_in.src, e_in.src_conn, e_out.dst, e_out.dst_conn, copy.deepcopy(e_in.data))
                state.remove_edge(e_in)
                state.remove_edge(e_out)
                state.remove_node(n)
                changed = True


def _run_probe_pipeline(probe: SDFG) -> None:
    """Parallelize + lift the disposable copy so a contraction collapses to one
    ``Einsum`` and a transpose to one clean 2-D copy map. See the module docstring
    for why each step is present (and why ``MapFission`` is not)."""
    from dace.transformation.passes.simplify import SimplifyPass
    from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated
    from dace.transformation.dataflow.lift_einsum import LiftEinsum
    from dace.transformation.dataflow.map_collapse import MapCollapse
    from dace.transformation.dataflow.tasklet_fusion import TaskletFusion
    from dace.transformation.dataflow.trivial_tasklet_elimination import TrivialTaskletElimination
    from dace.transformation.dataflow.wcr_conversion import AugAssignToWCR
    from dace.transformation.interstate.loop_to_map import LoopToMap
    from dace.transformation.interstate.sdfg_nesting import InlineSDFG
    from dace.transformation.interstate.multistate_inline import InlineMultistateSDFG

    SimplifyPass().apply_pass(probe, {})
    # Reduction-body clean-up (state level, where dataflow transforms reach).
    # ORDER MATTERS: fuse tasklets BEFORE ``AugAssignToWCR``. ``AugAssignToWCR``
    # only matches an accumulator whose tasklet inputs ALL come from AccessNodes
    # (its expr-0 fission guard). A fresh frontend nest materializes the product
    # in a scratch AccessNode, so it would match either way -- but once
    # canonicalize's ``WCRToAugAssign`` has de-WCR'd a ``@dace.map`` accumulation
    # (polybench atax ``compute_y``: ``y[j] += A[i,j]*tmp[i]``, reduction over the
    # OUTER axis), the multiply feeds the add via a DIRECT tasklet->tasklet edge,
    # AugAssignToWCR refuses, the outer reduction axis never becomes a WCR map, and
    # the contraction never lifts. Fusing first collapses ``prod = a*b; y = y +
    # prod`` into one ``y = y + a*b`` tasklet whose inputs are all AccessNodes,
    # which AugAssignToWCR then WCR-ifies; the second fusion mops up any tasklet
    # pair the WCR rewrite newly exposed.
    PatternMatchAndApplyRepeated([TrivialTaskletElimination()]).apply_pass(probe, {})
    probe.apply_transformations_repeated(TaskletFusion, validate=False, validate_all=False)
    probe.apply_transformations_repeated(AugAssignToWCR, validate=False, validate_all=False, permissive=False)
    probe.apply_transformations_repeated(TaskletFusion, validate=False, validate_all=False)
    PatternMatchAndApplyRepeated([TrivialTaskletElimination()]).apply_pass(probe, {})
    SimplifyPass().apply_pass(probe, {})
    # Parallelize: every loop (including the WCR reduction axis) becomes a map;
    # flatten the resulting nested SDFGs and merge the perfect nest into one map.
    probe.apply_transformations_repeated(LoopToMap, validate=False, validate_all=False)
    probe.apply_transformations_repeated([InlineSDFG, InlineMultistateSDFG], validate=False, validate_all=False)
    SimplifyPass().apply_pass(probe, {})
    probe.apply_transformations_repeated(MapCollapse, validate=False, validate_all=False)
    _splice_scope_scalars(probe)
    PatternMatchAndApplyRepeated([LiftEinsum()]).apply_pass(probe, {})


def _probe_compute_nodes(probe: SDFG):
    """(einsum_nodes, tasklets, map_entries, other_libnodes, nested) across the probe."""
    from dace.libraries.blas.nodes.einsum import Einsum
    einsums, tasklets, maps, others, nested = [], [], [], [], []
    for n, _ in probe.all_nodes_recursive():
        if isinstance(n, Einsum):
            einsums.append(n)
        elif isinstance(n, nodes.Tasklet):
            tasklets.append(n)
        elif isinstance(n, nodes.MapEntry):
            maps.append(n)
        elif isinstance(n, nodes.NestedSDFG):
            nested.append(n)
        elif isinstance(n, nodes.LibraryNode):
            others.append(n)
    loops = [r for r in probe.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion) and r.loop_variable]
    return einsums, tasklets, maps, others, nested, loops


def _extract_einsum(probe: SDFG, written: Set[str]) -> Optional[EinsumSpec]:
    """If the probe collapsed to exactly one ``Einsum`` producing one of the loop's
    output arrays (and no other compute), return its rebuildable spec."""
    einsums, tasklets, maps, others, nested, loops = _probe_compute_nodes(probe)
    if len(einsums) != 1 or tasklets or maps or others or nested or loops:
        return None
    node = einsums[0]
    # A scalar-output Einsum -- no free output indices, e.g. ``i,i->`` (a dot product) or
    # ``ij,ij->`` -- is a pure REDUCTION, not an array-producing contraction. Per this
    # pass's own design (see ``_has_loop_ancestor``) a bare 1-D/0-D reduction is
    # LoopToReduce's domain, not an einsum. ``_has_loop_ancestor`` only excludes nests
    # inside an enclosing *LoopRegion*, so an inner reduction nested in outer *map* scopes
    # (covariance's per-``(i,j)`` k-dot ``sum_k data[k,i]*data[k,j]``) slips through and is
    # lifted to a standalone scalar-output Einsum whose result never re-embeds into the
    # per-iteration ``cov[i,j]`` -- corrupting the kernel. Refuse it; LoopToReduce lifts it.
    if node.einsum_str.rstrip().endswith('->'):
        return None
    host = next((st for st in probe.states() if node in st.nodes()), None)
    if host is None:
        return None

    inputs: List[Tuple[str, str, subsets.Subset, object]] = []
    for e in host.in_edges(node):
        if e.data is None or e.data.is_empty() or e.data.data is None:
            return None
        inputs.append((e.dst_conn, e.data.data, copy.deepcopy(e.data.subset), node.in_connectors[e.dst_conn]))
    out_edges = host.out_edges(node)
    if len(out_edges) != 1:
        return None
    oe = out_edges[0]
    if oe.data is None or oe.data.data is None or oe.data.data not in written:
        return None
    output = (oe.src_conn, oe.data.data, copy.deepcopy(oe.data.subset), node.out_connectors[oe.src_conn])
    if not inputs:
        return None
    return EinsumSpec(node.einsum_str, node.alpha, node.beta, inputs, output)


def _axis_order(subset: subsets.Subset) -> Optional[List[str]]:
    """The ordered list of single-point index expressions of ``subset`` (one per
    axis), or ``None`` if any axis is not a single point."""
    order = []
    for rb, re_, _ in subset.ndrange():
        if rb != re_:
            return None
        order.append(str(rb).strip())
    return order


def _extract_transpose(probe: SDFG, written: Set[str]) -> Optional[TransposeSpec]:
    """If the probe collapsed to exactly one 2-D full-range map that is a pure
    cross-array transposed copy ``dst[p,q] = src[q,p]``, return its spec. Pure
    copy = the map scope has no arithmetic (only transient staging AccessNodes
    and/or ``__out = __inp`` copy tasklets)."""
    einsums, tasklets, maps, others, nested, loops = _probe_compute_nodes(probe)
    if einsums or others or nested or loops or len(maps) != 1:
        return None
    map_entry = maps[0]
    if len(map_entry.map.params) != 2:
        return None
    for rng in map_entry.map.range:
        if str(rng[0]) != '0' or str(rng[2]) != '1':
            return None  # partial / strided range -- not a full transpose

    host = next((st for st in probe.states() if map_entry in st.nodes()), None)
    if host is None:
        return None
    map_exit = host.exit_node(map_entry)

    # Only pure copies allowed in the scope: transient AccessNodes + ``__out=__inp``.
    for n in host.all_nodes_between(map_entry, map_exit):
        if isinstance(n, nodes.AccessNode):
            d = probe.arrays.get(n.data)
            if d is None or not d.transient:
                return None
        elif isinstance(n, nodes.Tasklet):
            if not _is_copy_tasklet(n):
                return None
        else:
            return None

    # One boundary input array (into the scope) and one boundary output array.
    read = _boundary_axis_order(host.out_edges(map_entry), probe, transient_ok=False)
    write = _boundary_axis_order(host.in_edges(map_exit), probe, transient_ok=False)
    if read is None or write is None:
        return None
    src, read_order, src_subset = read
    dst, write_order, dst_subset = write
    if src == dst:
        return None  # in-place symmetrization is LoopToSymmetrize's domain
    params = set(map_entry.map.params)
    if set(read_order) != params or set(write_order) != params:
        return None
    if read_order != list(reversed(write_order)):
        return None
    sdesc, ddesc = probe.arrays.get(src), probe.arrays.get(dst)
    if sdesc is None or ddesc is None or len(sdesc.shape) != 2 or len(ddesc.shape) != 2:
        return None
    if dst not in written or sdesc.dtype != ddesc.dtype:
        return None
    return TransposeSpec(src, dst, sdesc.dtype, src_subset, dst_subset)


def _is_copy_tasklet(node: nodes.Tasklet) -> bool:
    """A single-input single-output pure copy ``__out = __inp``."""
    code = node.code.as_string.strip()
    if code.count('=') != 1:
        return False
    lhs, rhs = (s.strip() for s in code.split('=', 1))
    return len(node.in_connectors) == 1 and len(node.out_connectors) == 1 and rhs in node.in_connectors and \
        lhs in node.out_connectors


def _boundary_axis_order(edges, probe: SDFG, transient_ok: bool):
    """From a MapEntry's scope-side out-edges (or a MapExit's scope-side in-edges),
    the single non-transient boundary array touched, its per-axis index order, and
    a full-array subset. ``None`` unless exactly one such array is found."""
    found = None
    for e in edges:
        if e.data is None or e.data.is_empty() or e.data.data is None:
            continue
        desc = probe.arrays.get(e.data.data)
        if desc is None or (desc.transient and not transient_ok):
            continue
        order = _axis_order(e.data.subset)
        if order is None:
            return None
        if found is not None:
            return None  # ambiguous -- more than one boundary array
        full = subsets.Range([(0, s - 1, 1) for s in desc.shape])
        found = (e.data.data, order, full)
    return found


@explicit_cf_compatible
class LoopToEinsum(ppl.Pass):
    """Lift a contraction loop nest to an ``Einsum`` node, or a transpose nest to a
    ``Transpose`` node, via a probe-and-map-back strategy (see module docstring)."""

    CATEGORY: str = "Canonicalization"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Nodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & ppl.Modifies.CFG)

    def apply_pass(self, sdfg: SDFG, _) -> Optional[int]:
        # Only the OUTERMOST loop of each nest is a candidate: probing it lifts (or
        # no-ops on) the whole nest. Snapshot upfront -- a lift removes the nest.
        candidates: List[LoopRegion] = []
        for sd in sdfg.all_sdfgs_recursive():
            for region in sd.all_control_flow_regions(recursive=True):
                if isinstance(region, LoopRegion) and region.loop_variable and not _has_loop_ancestor(region):
                    candidates.append(region)

        count = 0
        for loop in candidates:
            parent = loop.parent_graph
            if not isinstance(parent, ControlFlowRegion) or loop not in parent.nodes():
                continue  # already spliced out (defensive)
            root = parent
            while not isinstance(root, SDFG):
                root = root.parent_graph
            spec = self._probe(loop, root)
            if spec is None:
                continue
            if isinstance(spec, EinsumSpec):
                self._replace_with_einsum(parent, loop, spec)
            else:
                self._replace_with_transpose(parent, loop, spec)
            count += 1
        return count or None

    def _probe(self, loop: LoopRegion, root: SDFG):
        """Copy the loop, run the lift pipeline on the copy, and return an
        ``EinsumSpec`` / ``TransposeSpec`` if it cleanly collapsed, else ``None``.
        Any probe failure is swallowed -- the lift is strictly opt-in."""
        try:
            probe = _build_probe(loop, root)
            if probe is None:
                return None
            written = _written_arrays(loop)
            _run_probe_pipeline(probe)
            spec = _extract_einsum(probe, written)
            if spec is not None:
                return spec
            return _extract_transpose(probe, written)
        except Exception:
            return None

    def _replace_with_einsum(self, parent: ControlFlowRegion, loop: LoopRegion, spec: EinsumSpec) -> None:
        from dace.libraries.blas.nodes.einsum import Einsum
        state = self._replace_loop_with_state(parent, loop, loop.label + '_einsum')
        node = Einsum(loop.label + '_einsum')
        node.einsum_str = spec.einsum_str
        node.alpha = spec.alpha
        node.beta = spec.beta
        state.add_node(node)
        for conn, array, subset, dtype in spec.inputs:
            node.add_in_connector(conn, dtype)
            state.add_edge(state.add_read(array), None, node, conn, Memlet(data=array, subset=copy.deepcopy(subset)))
        out_conn, out_array, out_subset, out_dtype = spec.output
        node.add_out_connector(out_conn, out_dtype)
        state.add_edge(node, out_conn, state.add_write(out_array), None,
                       Memlet(data=out_array, subset=copy.deepcopy(out_subset)))

    def _replace_with_transpose(self, parent: ControlFlowRegion, loop: LoopRegion, spec: TransposeSpec) -> None:
        from dace.libraries.linalg.nodes.transpose import Transpose
        state = self._replace_loop_with_state(parent, loop, loop.label + '_transpose')
        node = Transpose(loop.label + '_transpose', dtype=spec.dtype)
        state.add_node(node)
        state.add_edge(state.add_read(spec.src), None, node, '_inp',
                       Memlet(data=spec.src, subset=copy.deepcopy(spec.src_subset)))
        state.add_edge(node, '_out', state.add_write(spec.dst), None,
                       Memlet(data=spec.dst, subset=copy.deepcopy(spec.dst_subset)))

    def _replace_loop_with_state(self, parent: ControlFlowRegion, loop: LoopRegion, label: str) -> SDFGState:
        """Splice ``loop`` out of ``parent``, replacing it with a fresh (returned)
        state that inherits the loop's in/out interstate edges. Mirrors
        ``LoopToReduce._lift``'s CFG surgery."""
        import dace
        was_start = parent.start_block is loop
        in_edges = list(parent.in_edges(loop))
        out_edges = list(parent.out_edges(loop))
        state = parent.add_state(label, is_start_block=was_start)
        for e in in_edges:
            parent.add_edge(e.src, state, e.data)
        for e in out_edges:
            cond = e.data.condition.as_string if e.data.condition is not None else "1"
            parent.add_edge(state, e.dst, dace.InterstateEdge(condition=cond,
                                                              assignments=dict(e.data.assignments or {})))
        parent.remove_node(loop)
        return state


__all__ = ["LoopToEinsum"]
