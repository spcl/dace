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

MATCHING happens in two tiers.

**Tier 1 -- the direct matcher** (:func:`_match_nest`, the normal path). It reads the
canonical loop form in place: no copy, no pipeline. Measured on the real form (see
below) the nest that reaches this pass is

* a CHAIN of nested single-block ``LoopRegion``s -- one iteration axis each -- and/or
  ``MapEntry`` scopes (a map carries several params at once), innermost body one
  ``SDFGState``;
* whose body accumulates via an explicit read/add/write (``C[i,j] = C[i,j] + prod``,
  what the frontend emits -- NOT a WCR) or via a WCR-sum edge (what a ``@dace.map``
  accumulation emits). Both are accepted directly;
* whose product is spread over several tasklets joined by single-element transient
  staging scalars (``A --A[i,k]--> A_index --> mul``), which the matcher walks through.

The einsum is then read straight off the index expressions: every axis that indexes
an operand but not the output is contracted. Acceptance mirrors ``LiftEinsum``'s
``can_be_applied`` (>=2 tensor operands, at most one scalar coefficient, indices that
are exactly axis parameters or ``0``, rectangular full ranges, non-scalar output).

**Tier 2 -- the probe** (:meth:`LoopToEinsum._probe`, fallback only). Trial
normalization: deep-copy the loop into a throwaway SDFG, run a small canonicalization
pipeline on the copy and see whether one ``Einsum`` falls out. It is kept so a shape
the direct matcher has not been taught cannot silently stop lifting; ``FALLBACK_LIFTS``
counts how often it was the one that fired. It costs SECONDS per candidate, which is
why it is no longer the primary matcher.

The probe works as follows:

0. Screen the loop against cheap NECESSARY conditions first
   (:func:`_plausible_contraction`) -- the probe below costs seconds on a big nest
   and a weather/physics kernel's outer loop can never lift.
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
import ast
import copy
from typing import Dict, List, NamedTuple, Optional, Set, Tuple, Union

import sympy

from dace import SDFG, data, dtypes, subsets
from dace.frontend.operations import detect_reduction_type
from dace.frontend.python import astutils
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.state import ControlFlowRegion, LoopRegion, SDFGState
from dace.symbolic import pystr_to_symbolic
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.analysis import loop_analysis
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


def _data_state_counts(cfg) -> Dict[str, int]:
    """Array name -> number of states in ``cfg`` holding an AccessNode for it."""
    counts: Dict[str, int] = {}
    for state in cfg.all_states():
        for name in set(dn.data for dn in state.data_nodes()):
            counts[name] = counts.get(name, 0) + 1
    return counts


def _live_outside(loop: LoopRegion, root_counts: Dict[str, int], referenced: Set[str]) -> Set[str]:
    """Referenced arrays that are also touched by data nodes OUTSIDE ``loop`` --
    i.e. cross the loop boundary and so must stay visible (non-transient) in the
    probe, unlike the loop's purely-internal staging scratch. ``root_counts`` is the
    enclosing SDFG's :func:`_data_state_counts`, computed ONCE per pass run: the
    inside states are a subset of the root's, so "touched outside" is a subtraction
    rather than a fresh whole-SDFG walk per candidate."""
    inside = _data_state_counts(loop)
    return set(name for name in referenced if root_counts.get(name, 0) > inside.get(name, 0))


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


def _region_in_loop(region) -> bool:
    """True iff ``region`` sits inside a ``LoopRegion`` (so its states' maps are already
    covered by that loop's candidate)."""
    while region is not None and not isinstance(region, SDFG):
        if isinstance(region, LoopRegion):
            return True
        region = region.parent_graph
    return False


class _NestShape(NamedTuple):
    """Cheap structural summary of a loop nest (one walk, no copying)."""
    n_dims: int  # upper bound on the iteration dimensions the probe can collapse into one map
    has_mul: bool  # some tasklet's code holds a '*'
    all_copy_tasklets: bool  # every tasklet is a pure ``__out = __inp``
    has_nested_sdfg: bool  # body the probe inlines -- this summary cannot see into it


def _nest_shape(loop: LoopRegion) -> _NestShape:
    """Summarize ``loop``'s own states in one walk (see :func:`_plausible_contraction`)."""
    n_dims = 1  # ``loop`` itself
    has_mul, all_copy, has_nested = False, True, False
    for region in loop.all_control_flow_regions(recursive=True):
        if region is not loop and isinstance(region, LoopRegion) and region.loop_variable:
            n_dims += 1
    for state in loop.all_states():
        for n in state.nodes():
            if isinstance(n, nodes.MapEntry):
                n_dims += len(n.map.params)
            elif isinstance(n, nodes.NestedSDFG):
                has_nested = True
            elif isinstance(n, nodes.Tasklet):
                has_mul = has_mul or '*' in n.code.as_string
                all_copy = all_copy and _is_copy_tasklet(n)
    return _NestShape(n_dims, has_mul, all_copy, has_nested)


def _plausible_contraction(loop: LoopRegion, root: SDFG, written: Set[str], live: Set[str]) -> bool:
    """Cheap NECESSARY conditions for the probe to collapse to one ``Einsum`` /
    ``Transpose``. The probe pipeline (two ``SimplifyPass`` runs, ``LoopToMap``,
    inlining, ...) costs seconds on a large nest and is pure waste on a nest that
    cannot possibly lift, so screen the candidate structurally first:

    1. Exactly ONE written array survives the probe as a boundary (non-transient)
       descriptor. The lifted node writes exactly one output; any second visible
       write would need a tasklet / map / loop, all of which the acceptance test in
       :func:`_extract_einsum` / :func:`_extract_transpose` refuses. (Writes to
       purely-internal transients do not count -- simplification removes them.)
    2. At least TWO iteration dimensions. ``LiftEinsum`` needs a free output index
       plus a second (contracted or outer-product) index, and the transpose shape
       needs exactly two map parameters -- neither is reachable from a single axis.
    3. A multiplication somewhere, OR a nest of pure copies. ``LiftEinsum`` requires
       the fused tasklet's expression to be the PRODUCT of its input connectors, and
       no pipeline step synthesizes a ``*``; the transpose shape instead needs the
       map scope to hold nothing but ``__out = __inp`` copies.

    (2) and (3) read only this nest's own states, so a nest holding a ``NestedSDFG``
    -- whose body the probe inlines and this summary cannot see -- skips them."""
    boundary: Set[str] = set()
    for name in written:
        desc = root.arrays.get(name)
        if desc is None or name in live or not desc.transient:
            boundary.add(name)
    if len(boundary) != 1:
        return False
    shape = _nest_shape(loop)
    if shape.has_nested_sdfg:
        return True
    return shape.n_dims >= 2 and (shape.has_mul or shape.all_copy_tasklets)


def _build_probe(loop: LoopRegion, root: SDFG, referenced: Set[str], live: Set[str]) -> Optional[SDFG]:
    """A throwaway SDFG wrapping a deep copy of ``loop``. Boundary arrays (live-out
    or non-transient in the original) become full-size non-transient descriptors;
    purely-internal staging scratch keeps its transient flag so simplification /
    the scalar splice can remove it."""
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


# ---------------------------------------------------------------------------
# Tier 1: the direct, loop-native matcher (no copy, no probe pipeline).
# ---------------------------------------------------------------------------

#: Index letters, in the order ``LiftEinsum`` hands them out (kept identical so a
#: directly-matched einsum string is spelled the way the post-LoopToMap lift spells it).
_EINSUM_CHARS = 'ijklmnopqrstuvwxyzabcdefgh'


class _Axis(NamedTuple):
    """One iteration dimension of a nest: ``param`` runs ``0 .. end`` inclusive, step 1."""
    param: str
    end: object


class _Nest(NamedTuple):
    """A matched iteration nest: its axes and the innermost scope holding the compute.

    ``entry`` is the innermost ``MapEntry`` (``None`` when the whole nest is
    ``LoopRegion``s and the body is the state's top level)."""
    axes: List[_Axis]
    state: SDFGState
    entry: Optional[nodes.MapEntry]


class _Leaf(NamedTuple):
    """An array read at a single point: ``array`` indexed by one expression per axis."""
    array: str
    idx: List[str]


class _Product(NamedTuple):
    """A tasklet-level product: ``scale`` (a number) times every term."""
    scale: object
    terms: List[Union[_Leaf, '_Product']]


def _plain_edges(region: ControlFlowRegion) -> bool:
    """True iff every interstate edge inside ``region`` is an unconditional plain link.
    A condition or an assignment means the body is more than straight-line dataflow."""
    for e in region.edges():
        if e.data.assignments:
            return False
        if e.data.condition is not None and e.data.condition.as_string.strip() not in ('1', 'True'):
            return False
    return True


def _single_body_block(region: ControlFlowRegion):
    """``region``'s one meaningful child block, or ``None``. Empty scaffold states --
    the connective ``..._for_inline_pre_state`` blocks the map->loop lowering leaves
    behind -- are ignored; anything else must be alone."""
    if not _plain_edges(region):
        return None
    blocks = [b for b in region.nodes() if not (isinstance(b, SDFGState) and b.number_of_nodes() == 0)]
    return blocks[0] if len(blocks) == 1 else None


def _loop_axis(loop: LoopRegion) -> Optional[_Axis]:
    """``loop`` as one ``0 .. end`` unit-stride axis, or ``None`` if it is not one.
    A non-zero start or a non-unit stride is refused for the same reason
    ``LiftEinsum._partial_range`` refuses it: the contraction spans the operands' full
    extent, so a partial sweep would compute the wrong (or a clobbering) result."""
    if loop.inverted or not loop.loop_variable:
        return None
    init = loop_analysis.get_init_assignment(loop)
    end = loop_analysis.get_loop_end(loop)
    stride = loop_analysis.get_loop_stride(loop)
    if init is None or end is None or stride is None:
        return None
    if init != 0 or stride != 1:
        return None
    return _Axis(loop.loop_variable, end)


def _map_axes(entry: nodes.MapEntry) -> Optional[List[_Axis]]:
    """A map's params as axes -- one MapEntry carries SEVERAL, unlike a LoopRegion."""
    axes = []
    for param, rng in zip(entry.map.params, entry.map.range):
        if rng[0] != 0 or rng[2] != 1:
            return None
        axes.append(_Axis(param, rng[1]))
    return axes


def _descend_maps(state: SDFGState, axes: List[_Axis], entry: Optional[nodes.MapEntry]) -> Optional[_Nest]:
    """Follow one chain of nested map scopes down from ``entry`` (or from ``state``'s
    top level when it is ``None``), appending each map's params to ``axes``. Refuses a
    level that holds compute of its own or forks into several maps -- the lifted node
    replaces the WHOLE scope, so nothing may be left behind."""
    children = state.scope_children()
    while True:
        if entry is not None:
            more = _map_axes(entry)
            if more is None:
                return None
            axes.extend(more)
        level = [n for n in children[entry] if not isinstance(n, nodes.MapExit)]
        inner = [n for n in level if isinstance(n, nodes.MapEntry)]
        if not inner:
            return _Nest(axes, state, entry)
        if len(inner) != 1 or any(isinstance(n, (nodes.Tasklet, nodes.NestedSDFG, nodes.LibraryNode)) for n in level):
            return None
        entry = inner[0]


def _nest_of(root: LoopRegion) -> Optional[_Nest]:
    """The iteration nest rooted at a ``LoopRegion``. Loops contribute ONE axis each and
    must form a perfect chain; the innermost body may then descend through map scopes
    (a mixed ``for i: for j,k in dace.map[...]`` nest is one nest)."""
    axes: List[_Axis] = []
    block = root
    while isinstance(block, LoopRegion):
        axis = _loop_axis(block)
        if axis is None:
            return None
        axes.append(axis)
        block = _single_body_block(block)
        if block is None:
            return None
    if not isinstance(block, SDFGState):
        return None
    return _descend_maps(block, axes, None)


def _nest_of_map(state: SDFGState, entry: nodes.MapEntry) -> Optional[_Nest]:
    """The iteration nest of a map-form contraction rooted at ``entry``."""
    return _descend_maps(state, [], entry)


def _point_index(memlet: Memlet, expect_data: Optional[str] = None) -> Optional[Tuple[str, List[str]]]:
    """``(array, one index expression per axis)`` for a single-element point access,
    else ``None``. Mirrors ``LiftEinsum``'s per-edge screen (no dynamic memlets, exactly
    one element)."""
    if memlet is None or memlet.is_empty() or memlet.data is None or memlet.dynamic:
        return None
    if expect_data is not None and memlet.data != expect_data:
        return None
    subset = memlet.subset
    if subset is None or subset.num_elements() != 1:
        return None
    idx = []
    for rb, re_, _ in subset.ndrange():
        if rb != re_:
            return None
        idx.append(str(rb).strip())
    return memlet.data, idx


def _tasklet_expr(tasklet: nodes.Tasklet):
    """The tasklet's single assigned expression as a sympy expression, or ``None``."""
    if tasklet.code.language != dtypes.Language.Python:
        return None
    code = tasklet.code.code
    if not isinstance(code, list) or len(code) != 1:
        return None
    stmt = code[0]
    if not isinstance(stmt, (ast.Assign, ast.AnnAssign)) or stmt.value is None:
        return None
    if len(tasklet.out_connectors) != 1:
        return None
    try:
        expr = pystr_to_symbolic(astutils.unparse(stmt.value))
    except (TypeError, ValueError, AttributeError, sympy.SympifyError):
        return None
    # A comparison / logical tasklet (``__out = __in1 > __in2``) parses to a sympy
    # RELATIONAL or BooleanFunction, which cannot be divided or subtracted -- and is
    # never a contraction anyway. Reject it here rather than let the arithmetic raise.
    # NOT ``isinstance(expr, Boolean)``: sympy's ``Symbol`` derives from ``Boolean``, so
    # that would also reject the bare-symbol copy tasklet ``__out = __inp`` (the pure
    # transpose shape).
    if expr.is_Relational or isinstance(expr, sympy.logic.boolalg.BooleanFunction):
        return None
    return expr


def _product_scale(tasklet: nodes.Tasklet):
    """The numeric factor of a tasklet that multiplies ALL of its inputs together
    (``__out = 2.5 * __in2`` -> ``2.5``; ``__out = __in1 * __in2`` -> ``1``), else
    ``None``. Same test ``LiftEinsum`` applies to the fused tasklet, here applied per
    tasklet so a product split across several of them still matches."""
    expr = _tasklet_expr(tasklet)
    if expr is None or not tasklet.in_connectors:
        return None
    expected = 1
    for conn in tasklet.in_connectors:
        expected *= pystr_to_symbolic(conn)
    try:
        ratio = sympy.simplify(expr / expected)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return ratio if ratio.is_Number else None


def _is_sum_tasklet(tasklet: nodes.Tasklet) -> bool:
    """``__out = __in1 + __in2`` -- the read/add/write half of an aug-assign. Only a
    plain SUM: a ``-=`` accumulation is not a ``ReductionType.Sum`` contraction."""
    conns = list(tasklet.in_connectors)
    if len(conns) != 2:
        return False
    expr = _tasklet_expr(tasklet)
    if expr is None:
        return False
    try:
        return sympy.simplify(expr - pystr_to_symbolic(conns[0]) - pystr_to_symbolic(conns[1])) == 0
    except (TypeError, ValueError):
        return False


def _resolve(state: SDFGState, sdfg: SDFG, edge, visited: Set) -> Optional[Union[_Leaf, _Product]]:
    """The value flowing along ``edge``, chasing back through single-element transient
    staging scalars and product tasklets to the leaf array reads. ``visited`` collects
    every node consumed, so the caller can verify the body holds nothing else."""
    src = edge.src
    if isinstance(src, nodes.MapEntry):  # scope boundary: the inner memlet is the point read
        origin = state.memlet_path(edge)[0].src
        if not isinstance(origin, nodes.AccessNode):
            return None
        got = _point_index(edge.data, origin.data)
        return None if got is None else _Leaf(*got)
    if isinstance(src, nodes.AccessNode):
        desc = sdfg.arrays.get(src.data)
        if desc is None:
            return None
        if desc.transient and _is_single_element(desc):  # staging scalar: look through it
            if state.in_degree(src) != 1 or state.out_degree(src) != 1:
                return None
            visited.add(src)
            return _resolve(state, sdfg, state.in_edges(src)[0], visited)
        if state.in_degree(src) != 0:
            return None  # not a pure input to this nest
        visited.add(src)
        got = _point_index(edge.data, src.data)
        return None if got is None else _Leaf(*got)
    if isinstance(src, nodes.Tasklet):
        return _resolve_tasklet(state, sdfg, src, visited)
    return None


def _resolve_tasklet(state: SDFGState, sdfg: SDFG, tasklet: nodes.Tasklet, visited: Set) -> Optional[_Product]:
    """``tasklet`` as a product of its resolved inputs, or ``None``. Single-consumer
    only: a re-used intermediate is not a tree and would be counted twice."""
    if tasklet in visited or state.out_degree(tasklet) != 1:
        return None
    scale = _product_scale(tasklet)
    if scale is None:
        return None
    visited.add(tasklet)
    terms: List[Union[_Leaf, _Product]] = []
    for e in state.in_edges(tasklet):
        if e.data is None or e.data.is_empty():
            return None
        term = _resolve(state, sdfg, e, visited)
        if term is None:
            return None
        terms.append(term)
    return _Product(scale, terms) if terms else None


def _flatten(term: Union[_Leaf, _Product]) -> Tuple[object, List[_Leaf]]:
    """``(numeric coefficient, operand leaves)`` of a resolved product tree."""
    if isinstance(term, _Leaf):
        return sympy.Integer(1), [term]
    scale, leaves = term.scale, []
    for sub in term.terms:
        sub_scale, sub_leaves = _flatten(sub)
        scale = scale * sub_scale
        leaves.extend(sub_leaves)
    return scale, leaves


class _BodyValue(NamedTuple):
    """What the nest's innermost body computes: ``coeff * prod(leaves)`` accumulated
    (or assigned, when ``accumulates`` is False) into ``array`` at ``idx``."""
    array: str
    idx: List[str]
    coeff: object
    leaves: List[_Leaf]
    accumulates: bool
    write_node: Optional[nodes.AccessNode]


def _body_value(nest: _Nest, sdfg: SDFG) -> Optional[_BodyValue]:
    """Read the nest's body as one accumulate-or-assign of a product, or ``None``."""
    state, entry = nest.state, nest.entry
    body = [n for n in state.scope_children()[entry] if not isinstance(n, (nodes.MapEntry, nodes.MapExit))]

    # Locate the single write leaving the scope.
    write_node = None
    if entry is not None:
        out_edges = [e for e in state.in_edges(state.exit_node(entry)) if e.data is not None and not e.data.is_empty()]
        if len(out_edges) != 1:
            return None
        out_edge = out_edges[0]
        tail = state.memlet_path(out_edge)[-1].dst
        write_node = tail if isinstance(tail, nodes.AccessNode) else None
    else:
        written = [
            n for n in body if isinstance(n, nodes.AccessNode) and state.in_degree(n) > 0
            and not (sdfg.arrays[n.data].transient and _is_single_element(sdfg.arrays[n.data]))
        ]
        if len(written) != 1 or state.out_degree(written[0]) != 0 or state.in_degree(written[0]) != 1:
            return None
        write_node = written[0]
        out_edge = state.in_edges(write_node)[0]
    got = _point_index(out_edge.data, None if write_node is None else write_node.data)
    if got is None or not isinstance(out_edge.src, nodes.Tasklet):
        return None
    out_array, out_idx = got

    visited: Set = set()
    if out_edge.data.wcr is not None:
        # WCR encoding: the producing tasklet IS the product, the sum lives on the edge.
        if detect_reduction_type(out_edge.data.wcr) != dtypes.ReductionType.Sum:
            return None
        term = _resolve_tasklet(state, sdfg, out_edge.src, visited)
        if term is None:
            return None
        accumulates = True
    else:
        adder = out_edge.src
        in_edges = [e for e in state.in_edges(adder) if e.data is not None and not e.data.is_empty()]
        if _is_sum_tasklet(adder) and len(in_edges) == 2:
            # Aug-assign encoding: one input re-reads the output slot, the other is the product.
            visited.add(adder)
            resolved = [_resolve(state, sdfg, e, visited) for e in in_edges]
            if any(r is None for r in resolved):
                return None
            prior = [i for i, r in enumerate(resolved) if isinstance(r, _Leaf) and list(r) == [out_array, out_idx]]
            if len(prior) != 1:
                return None
            term, accumulates = resolved[1 - prior[0]], True
        else:
            # Plain assignment (the pure-copy / transpose shape).
            term = _resolve_tasklet(state, sdfg, adder, visited)
            if term is None:
                return None
            accumulates = False

    leftover = set(body) - visited - ({write_node} if entry is None else set())
    if leftover:
        return None  # the body computes something the lifted node would drop
    coeff, leaves = _flatten(term)
    return _BodyValue(out_array, out_idx, coeff, leaves, accumulates, write_node)


def _axis_subset(idx: List[str], axes: Dict[str, _Axis]) -> subsets.Range:
    """The whole-operand range a point access ``idx`` sweeps over the nest -- what
    memlet propagation out of the equivalent map produces, and what ``LiftEinsum``
    hands the ``Einsum`` node."""
    return subsets.Range([(0, 0, 1) if i == '0' else (0, axes[i].end, 1) for i in idx])


def _direct_einsum(nest: _Nest, sdfg: SDFG, value: _BodyValue) -> Optional[EinsumSpec]:
    """The ``Einsum`` spec for an accumulated contraction, or ``None``. The acceptance
    rule mirrors ``LiftEinsum.can_be_applied`` on the map this nest would become."""
    if not value.accumulates:
        return None
    axes = {a.param: a for a in nest.axes}
    if len(axes) != len(nest.axes):
        return None  # a repeated parameter -- indices would be ambiguous
    # Rectangular iteration space: a parameter-dependent bound (a triangular syrk-style
    # nest) contracts over less than the full product the einsum expansion assumes.
    for axis in nest.axes:
        if set(map(str, pystr_to_symbolic(axis.end).free_symbols)) & set(axes):
            return None

    tensors, coeffs = [], []
    for leaf in value.leaves:
        if any(i != '0' and i not in axes for i in leaf.idx):
            return None  # an index that is not an axis parameter
        (tensors if set(leaf.idx) - {'0'} else coeffs).append(leaf)
    # >=2 tensor operands: a single one is a copy / transpose / reduction, not a matmul.
    # At most one runtime scalar: the node wires exactly one ``_alpha`` connector.
    if len(tensors) < 2 or len(coeffs) > 1:
        return None
    if any(i != '0' and i not in axes for i in value.idx):
        return None
    if any(t.array == value.array for t in tensors):
        return None  # the output feeds its own contraction -- a real loop-carried dependence
    out_chars = set(value.idx) - {'0'}
    in_chars = set().union(*[set(t.idx) - {'0'} for t in tensors])
    if not out_chars:
        return None  # scalar output: a dot / full reduction, LoopToReduce's domain
    if not out_chars <= in_chars:
        return None  # an output index no operand carries -- a broadcast, not a contraction
    if not in_chars - out_chars:
        return None  # no contracted axis: elementwise or an outer product
    if in_chars != set(axes):
        return None  # an axis indexes nothing -- lifting would silently drop its repetition

    # Canonical operand order. Any order is CORRECT (the einsum terms are emitted in the
    # same order as the connectors that carry them, which is the pairing the expansion's
    # ``*sorted(inputs)`` uses), but it must not depend on the frontend's edge insertion
    # order -- otherwise the same contraction gets a different einsum string depending on
    # how it was spelled (the probe emits ``ij,jk->ik`` for a matmul but ``jk,ij->ik`` for
    # the alpha-scaled one). Sorting by operand identity pins it.
    tensors.sort(key=lambda t: (t.array, tuple(t.idx)))

    mapping: Dict[str, str] = {}

    def term(idx: List[str]) -> str:
        out = ''
        for i in idx:
            if i == '0':
                continue
            if i not in mapping:
                mapping[i] = _EINSUM_CHARS[len(mapping)]
            out += mapping[i]
        return out

    terms = [term(t.idx) for t in tensors]
    einsum_str = f"{','.join(terms)}->{term(value.idx)}"
    # Connector names are chosen so SORTED order is operand order: the ``Einsum``
    # expansion feeds ``*sorted(inputs)`` to the contraction, so the two must agree.
    inputs = [(f'_ein{i:02d}', t.array, _axis_subset(t.idx, axes), None) for i, t in enumerate(tensors)]
    if coeffs:
        inputs.append(('_alpha', coeffs[0].array, _axis_subset(coeffs[0].idx, axes), None))
    beta = _fold_coefficient(sdfg, value.array, value.write_node)
    return EinsumSpec(einsum_str, value.coeff, beta, inputs, ('_out', value.array, _axis_subset(value.idx, axes), None))


def _fold_coefficient(sdfg: SDFG, array: str, write_node: Optional[nodes.AccessNode]) -> float:
    """The ``Einsum``'s ``beta``: ``out = alpha*contraction + beta*out_prior``. The SAME
    three signals ``LiftEinsum.apply`` uses, in the same order -- folding onto undefined
    memory and discarding a meaningful prior both miscompile:

    1. The accumulator carries a zero identity (``setzero``) -> it starts at 0, so
       OVERWRITE (beta=0).
    2. Else there is another in-SDFG writer of the array (a reset / pre-scale loop, e.g.
       atax's ``y[:] = 0`` before the matvec) -> FOLD onto it (beta=1).
    3. Else a non-transient output: the CALLER supplies the prior value that ``+=`` reads
       -> FOLD (beta=1).
    4. Else a fresh, never-written transient (k3mm's ``E = define_local(...)`` accumulator)
       -> OVERWRITE (beta=0) rather than fold uninitialized garbage.
    """
    if write_node is not None and write_node.setzero:
        return 0.0
    # Scoped to THIS sdfg so a same-named array in an unrelated nested SDFG cannot
    # false-positive a prior writer.
    has_prior_writer = any(n.data == array and st.in_degree(n) > 0 and n is not write_node for st in sdfg.states()
                           for n in st.data_nodes())
    return 1.0 if (has_prior_writer or not sdfg.arrays[array].transient) else 0.0


def _direct_transpose(nest: _Nest, sdfg: SDFG, value: _BodyValue) -> Optional[TransposeSpec]:
    """The ``Transpose`` spec for a pure 2-D cross-array transposed copy, or ``None``."""
    if value.accumulates or len(nest.axes) != 2 or len(value.leaves) != 1 or value.coeff != 1:
        return None
    params = set(a.param for a in nest.axes)
    read = value.leaves[0]
    if set(read.idx) != params or set(value.idx) != params:
        return None
    if read.idx != list(reversed(value.idx)):
        return None
    src, dst = read.array, value.array
    if src == dst:
        return None  # in-place symmetrization is LoopToSymmetrize's domain
    sdesc, ddesc = sdfg.arrays.get(src), sdfg.arrays.get(dst)
    if sdesc is None or ddesc is None or len(sdesc.shape) != 2 or len(ddesc.shape) != 2:
        return None
    if sdesc.dtype != ddesc.dtype:
        return None
    full = lambda desc: subsets.Range([(0, s - 1, 1) for s in desc.shape])
    return TransposeSpec(src, dst, sdesc.dtype, full(sdesc), full(ddesc))


def _may_hold_map_contraction(state: SDFGState) -> bool:
    """Cheap NECESSARY condition for a map-form contraction, in ONE node walk: the state
    must hold a map AND a multiplication. Building a state's scope tree is not free, and a
    big weather/physics kernel is mostly maps that multiply nothing."""
    has_map, has_mul = False, False
    for node in state.nodes():
        if isinstance(node, nodes.MapEntry):
            has_map = True
        elif isinstance(node, nodes.Tasklet) and '*' in node.code.as_string:
            has_mul = True
        if has_map and has_mul:
            return True
    return False


def _match_nest(nest: Optional[_Nest], sdfg: SDFG):
    """An ``EinsumSpec`` / ``TransposeSpec`` read directly off ``nest``, or ``None``."""
    if nest is None or len(nest.axes) < 2:
        return None
    value = _body_value(nest, sdfg)
    if value is None:
        return None
    return _direct_einsum(nest, sdfg, value) or _direct_transpose(nest, sdfg, value)


@explicit_cf_compatible
class LoopToEinsum(ppl.Pass):
    """Lift a contraction nest to an ``Einsum`` node, or a transpose nest to a
    ``Transpose`` node. The nest may be spelled as ``LoopRegion``s, as map scopes or as
    a mix of both; matching is direct, with the legacy probe as a fallback (see the
    module docstring)."""

    CATEGORY: str = "Canonicalization"

    #: How often the direct matcher declined but the fallback probe still lifted. Any
    #: non-zero count is a shape the direct matcher has not been taught -- see the
    #: module docstring. Cumulative across runs, for auditing.
    FALLBACK_LIFTS: int = 0

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Nodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & ppl.Modifies.CFG)

    def apply_pass(self, sdfg: SDFG, _) -> Optional[int]:
        count = self._lift_loops(sdfg)
        count += self._lift_maps(sdfg)
        return count or None

    def _lift_loops(self, sdfg: SDFG) -> int:
        """Lift nests rooted at a ``LoopRegion``: the whole loop is spliced out."""
        # Only the OUTERMOST loop of each nest is a candidate: matching it lifts (or
        # no-ops on) the whole nest. Snapshot upfront -- a lift removes the nest.
        candidates: List[Tuple[LoopRegion, SDFG]] = []
        for sd in sdfg.all_sdfgs_recursive():
            for region in sd.all_control_flow_regions(recursive=True):
                if isinstance(region, LoopRegion) and region.loop_variable and not _has_loop_ancestor(region):
                    candidates.append((region, sd))

        # Per-SDFG data-node census, shared by every candidate of that SDFG (see
        # ``_live_outside``). A lift rewires the CFG, so drop the census after one.
        counts_cache: Dict[int, Dict[str, int]] = {}
        count = 0
        for loop, root in candidates:
            parent = loop.parent_graph
            if not isinstance(parent, ControlFlowRegion) or loop not in parent.nodes():
                continue  # already spliced out (defensive)
            counts = counts_cache.get(id(root))
            if counts is None:
                counts = counts_cache[id(root)] = _data_state_counts(root)
            spec = self._match(loop, root, counts)
            if spec is None:
                continue
            if isinstance(spec, EinsumSpec):
                self._replace_with_einsum(parent, loop, spec)
            else:
                self._replace_with_transpose(parent, loop, spec)
            counts_cache.clear()
            count += 1
        return count

    def _lift_maps(self, sdfg: SDFG) -> int:
        """Lift nests rooted at a top-level ``MapEntry``: the map scope is replaced in
        place, leaving the rest of its state alone. Maps under a ``LoopRegion`` are
        skipped -- the enclosing loop is the candidate that covers them."""
        candidates: List[Tuple[SDFGState, nodes.MapEntry, SDFG]] = []
        for sd in sdfg.all_sdfgs_recursive():
            for state in sd.states():
                if not _may_hold_map_contraction(state):
                    continue  # cheap screen: no map, or no product to contract (one node walk)
                parent = state.parent_graph
                if isinstance(parent, LoopRegion) or (parent is not sd and _region_in_loop(parent)):
                    continue
                for node in state.scope_children()[None]:
                    if isinstance(node, nodes.MapEntry):
                        candidates.append((state, node, sd))

        count = 0
        for state, entry, root in candidates:
            if entry not in state.nodes():
                continue  # swallowed by an earlier lift in the same state (defensive)
            spec = _match_nest(_nest_of_map(state, entry), root)
            if not isinstance(spec, EinsumSpec):
                continue  # map-form transposes stay maps: no loop scaffold to remove
            self._replace_map_with_einsum(state, entry, spec)
            count += 1
        return count

    def _match(self, loop: LoopRegion, root: SDFG, root_counts: Dict[str, int]):
        """Match ``loop`` directly; on a decline fall back to the probe (and count it)."""
        referenced = _referenced_arrays(loop)
        if not referenced:
            return None
        live = _live_outside(loop, root_counts, referenced)
        written = _written_arrays(loop)
        if not _plausible_contraction(loop, root, written, live):
            return None
        spec = _match_nest(_nest_of(loop), root)
        if spec is not None:
            return spec
        spec = self._probe(loop, root, referenced, live, written)
        if spec is not None:
            LoopToEinsum.FALLBACK_LIFTS += 1
        return spec

    def _probe(self, loop: LoopRegion, root: SDFG, referenced: Set[str], live: Set[str], written: Set[str]):
        """Copy the loop, run the lift pipeline on the copy, and return an
        ``EinsumSpec`` / ``TransposeSpec`` if it cleanly collapsed, else ``None``.
        Any probe failure is swallowed -- the lift is strictly opt-in."""
        try:
            probe = _build_probe(loop, root, referenced, live)
            if probe is None:
                return None
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

    def _replace_map_with_einsum(self, state: SDFGState, entry: nodes.MapEntry, spec: EinsumSpec) -> None:
        """Swap a map scope for an ``Einsum`` node in place, reusing the AccessNodes
        the scope already reads from / writes to (the rest of the state is untouched)."""
        from dace.libraries.blas.nodes.einsum import Einsum
        scope = state.scope_subgraph(entry, include_entry=True, include_exit=True)
        node = Einsum(entry.map.label + '_einsum')
        node.einsum_str = spec.einsum_str
        node.alpha = spec.alpha
        node.beta = spec.beta
        state.add_node(node)
        # The operand order the spec fixes is authoritative; find each operand's outside
        # AccessNode by name among the scope's sources (they are unique per array here).
        sources = {e.src.data: e.src for e in state.in_edges(entry) if isinstance(e.src, nodes.AccessNode)}
        for conn, array, subset, dtype in spec.inputs:
            node.add_in_connector(conn, dtype)
            src = sources.get(array) or state.add_read(array)
            state.add_edge(src, None, node, conn, Memlet(data=array, subset=copy.deepcopy(subset)))
        out_conn, out_array, out_subset, out_dtype = spec.output
        node.add_out_connector(out_conn, out_dtype)
        exit_node = state.exit_node(entry)
        dst = next((e.dst for e in state.out_edges(exit_node) if isinstance(e.dst, nodes.AccessNode)), None)
        state.add_edge(node, out_conn, dst if dst is not None else state.add_write(out_array), None,
                       Memlet(data=out_array, subset=copy.deepcopy(out_subset)))
        state.remove_nodes_from(scope.nodes())

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
