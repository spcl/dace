# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Privatize data-dependent scatter REDUCTIONS to remove atomic contention.

The *azimint histogram* pattern is a data-dependent accumulate under a parallel
map::

    for i in dace.map[0:N]:
        hist[bin[i]] += w[i]        # bin[i] is read from data -> a runtime index

The write slot ``bin[i]`` is not an affine function of the map parameter -- it is
an integer read from an array -- so it is neither a plain per-element write (each
iteration a distinct slot) nor a provable permutation (the
:class:`~dace.transformation.passes.scatter_to_guarded_maps.ScatterToGuardedMaps`
guard does not apply: histogram bins repeat). It is a genuine *reduction*: many
iterations fold into the same slot. The DaCe CPU codegen lowers the per-iteration
write to a ``reduce_atomic`` on ``hist[bin[i]]`` -- correct but, with ``T`` threads
hammering a small ``hist``, cache-line contention makes it ~200x slower than numpy.

**Strategy A -- OpenMP array-section reduction (implemented here).** When the
accumulator ``hist`` is a plain, 0-offset, C-contiguous array (a histogram: its
size is the bin count, independent of ``N``) and the reduction operator is one
OpenMP's ``reduction`` clause supports (``+`` / ``*`` / ``min`` / ``max``), the
whole buffer is marked as the map's reduction target so the CPU codegen emits
``#pragma omp parallel for reduction(op:hist[0:nbins])`` (see
:func:`dace.codegen.targets.cpu.CPUCodeGen._collect_omp_reductions`, gated on
``sdfg.openmp_array_reductions`` which ``canonicalize`` turns on). OpenMP then
gives each thread a private, identity-initialised copy of ``hist``, the body
accumulates into that private copy with no cross-thread contention, and the runtime
tree-reduces the copies at the region barrier -- the "thread-private array + merge"
the reduction is asking for, done by the runtime.

Mechanically this pass only has to *surface* the reduction: the frontend already
emits the scatter as a map-body :class:`~dace.sdfg.nodes.NestedSDFG` whose body
holds the single-element data-dependent WCR write ``oc[bin] (+)= w`` into a
write-only output connector ``oc``, while the ``NestedSDFG -> MapExit ->
accumulator`` edge chain is *plain*. The pass copies the reduction operator onto
that outer edge chain (keeping the inner WCR untouched and inserting NO
intermediate buffer). Codegen then (a) recognises the whole-buffer WCR at the map
exit and emits the ``reduction(...)`` clause, and (b) accumulates via the inner
single-element WCR straight into the per-thread private buffer -- an *uncontended*
atomic on a thread-local cache line, not a shared one.

Surfacing the WCR here (before
:class:`~dace.transformation.passes.normalize_wcr.NormalizeWCR`) also *prevents*
that pass's drop-WCR shortcut, which is unsound for a scatter: it rewrites
``oc[bin] (+)= w`` into a plain ``oc[bin] = w`` on the assumption that a write-only
``oc`` starts at the reduction identity and the write covers ``oc`` -- true for a
scalar accumulator, but a scatter writes only ONE element of a bounded ``oc``,
leaving the rest of a per-iteration whole-array private buffer uninitialised (read
back into the sum -> garbage). Because this pass makes the outer edge non-plain,
``NormalizeWCR`` and ``NormalizeWCRSource`` both skip the scatter (the latter via
:func:`scatter_reduction_wcr_edge`, so its per-iteration whole-array buffer -- the
same latent bug -- is never inserted either).

**Correctness.** The rewrite is value-preserving. A ``+`` (or ``*``) reduction
reassociates the fold across threads, so a floating-point result is
``np.allclose``-equal to the sequential order, NOT bit-identical -- expected and
sound. An integer / count histogram is bit-exact (integer ``+`` is associative).
``min`` / ``max`` are order-independent (bit-exact). A non-commutative /
non-associative accumulate (e.g. ``-`` / ``/`` / a custom lambda) is REFUSED and
left as the correct, contended atomic; likewise a self-referential accumulate (the
map also READS the accumulator array, directly or through a View / alias) is refused
so the whole-buffer privatisation -- which would make those reads see the private
identity copy -- never fires.

**Only parallel scatters are privatised.** The contention this pass removes only
exists when several threads accumulate concurrently, so it fires only for a PARALLEL
map (:func:`map_is_parallel`): a top-level map whose schedule is not ``Sequential``. A
sequential scatter -- an inner map (single-threaded per outer iteration) or an
explicitly ``Sequential`` map -- is left as a plain serial WCR accumulate, which is
already optimal (no atomic contention single-threaded, and an OpenMP ``reduction(...)``
clause on a non-parallel loop is a no-op). Independently, the whole-buffer refuse-guard
in :class:`~dace.transformation.passes.normalize_wcr.NormalizeWCR` protects EVERY
data-dependent scatter sink (parallel or sequential, any op) from the unsound drop-WCR
rewrite.

**Strategy B -- explicit per-thread buffer + merge (deferred).** The general
fallback for a non-OpenMP-reducible operator or a very large accumulator is a
``[num_threads, *hist.shape]`` private buffer indexed by ``omp_get_thread_num()``
plus a second parallel merge map over the accumulator axis. Strategy A already
covers every OpenMP-reducible operator (which is the entire azimint family and the
common ``+`` / ``*`` / ``min`` / ``max`` cases) with a correct graceful fallback
for the rest, so Strategy B is documented but intentionally not built here (YAGNI);
the refuse path keeps the non-reducible cases correct.
"""
import ast
from typing import Optional, Set

from dace import SDFG, data, dtypes, properties
from dace.sdfg import SDFGState, nodes
from dace.sdfg.utils import get_last_view_node
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf

#: Reduction operators OpenMP's ``reduction`` clause supports (Strategy A). ``-`` /
#: ``/`` and custom lambdas are refused (left as the correct contended atomic).
SCATTER_REDUCIBLE_OPS = frozenset({'+', '*', 'min', 'max'})


def scatter_wcr_op(wcr: str) -> Optional[str]:
    """The reduction operator (``+`` / ``*`` / ``min`` / ``max``) of a WCR lambda
    string, or ``None`` when the reducer is not exactly one of those over its two
    arguments.

    Strict: the body must be ``p0 (+|*) p1`` or ``min|max(p0, p1)`` where ``p0`` /
    ``p1`` are the lambda's two parameters -- an associative, commutative fold. Anything
    else (``p0 - p1``, ``2*p0 + p1``, a custom lambda) returns ``None`` so the scatter is
    left as the correct, contended atomic rather than mis-privatised (a whole-buffer
    reduction reassociates the fold, which is only value-preserving for such an op).
    """
    try:
        tree = ast.parse(wcr.strip(), mode='eval').body
    except (SyntaxError, ValueError, TypeError):
        return None
    if not isinstance(tree, ast.Lambda) or len(tree.args.args) != 2:
        return None
    params = {a.arg for a in tree.args.args}
    body = tree.body

    def _is_two_params(*operands) -> bool:
        names = [o.id for o in operands if isinstance(o, ast.Name)]
        return len(names) == 2 and set(names) == params

    if isinstance(body, ast.BinOp) and _is_two_params(body.left, body.right):
        return {ast.Add: '+', ast.Mult: '*'}.get(type(body.op))
    if (isinstance(body, ast.Call) and isinstance(body.func, ast.Name) and body.func.id in ('min', 'max')
            and len(body.args) == 2 and _is_two_params(*body.args)):
        return body.func.id
    return None


def _is_single_element(desc: data.Data) -> bool:
    """True if ``desc`` holds at most one element (a scalar accumulator -- that is
    :class:`~dace.transformation.passes.normalize_wcr.NormalizeWCR`'s job, not a
    scatter into a bounded buffer)."""
    if isinstance(desc, data.Scalar):
        return True
    try:
        return int(desc.total_size) <= 1
    except (TypeError, ValueError):
        return False  # symbolic size -> a genuine bounded array (e.g. ``bins``)


def _data_dependent_index(inner: SDFG, subset, in_connectors: Set[str]) -> bool:
    """True if ``subset``'s index is a runtime value derived from data -- a
    data-dependent scatter index -- rather than a constant slot or a map-parameter
    affine offset.

    The scatter index is a symbol that is computed INSIDE the map body: either a
    nested INPUT connector used directly (``oc[__idx_r]``, ``bin`` read straight from
    an array) or a symbol bound within the body from such a read (``bin :=
    min(compute_bin_ret[0], bins - 1)`` -- azimint's index). Both are distinguished
    from the two non-scatter cases by NOT being passed in from outside the body:

    * A constant slot ``oc[0]`` has no free symbol at all.
    * A loop-invariant program parameter (``bins``, ``N``) and the enclosing map
      iterator (threaded in through ``symbol_mapping``) are both *free symbols* of the
      nested SDFG -- bound by the caller, not the body. A data-derived index symbol is
      bound by the body's own dataflow / interstate edges, so it is NOT free.

    Hence the index is data-dependent iff the subset references an input connector, or
    a symbol that the nested SDFG does not receive from its caller.
    """
    subsyms = {str(s) for s in subset.free_symbols}
    if not subsyms:
        return False  # constant slot -> a same-slot fold, not a scatter
    # An input connector used directly as the scatter-index symbol.
    if subsyms & in_connectors:
        return True
    # A symbol bound INSIDE the body (not received from the caller) -> data-derived.
    inner_free = {str(s) for s in inner.free_symbols}
    return bool(subsyms - inner_free - in_connectors)


def data_dependent_scatter_wcr_edge(nsdfg: nodes.NestedSDFG, oc: str):
    """Return the inner single-element data-dependent WCR edge accumulating into a
    write-only bounded output ``oc`` of ``nsdfg`` -- the scatter-reduction signature --
    REGARDLESS of the reduction operator, or ``None``.

    This is the operator-agnostic core of the scatter detection. A match requires: ``oc``
    is a write-only output (not also an input -- an aliased live value) holding more than
    one element; and its body contains exactly one WCR edge into a pure-sink
    ``AccessNode`` named ``oc`` whose write subset is a single, *data-dependent* element
    (a runtime index read from data, not a constant slot or a map-parameter affine
    offset). The reducer is *not* consulted -- ``+`` / ``-`` / ``*`` / ``/`` / ``min`` /
    ``max`` / a custom lambda all match.

    It is the fail-safe predicate the WCR-normalization passes
    (:class:`~dace.transformation.passes.normalize_wcr.NormalizeWCR`,
    :class:`~dace.transformation.passes.normalize_wcr_source.NormalizeWCRSource`) key their
    refusal on: the drop-WCR / whole-buffer ``_nnr_out`` / ``_wcr_priv`` rewrites are
    UNSOUND for such a sink -- only one element of a per-iteration whole-array buffer is
    written, the rest read back uninitialised -- so both refuse it (any op, any device)
    and it falls back to the correct per-element atomic.
    """
    if oc in nsdfg.in_connectors or oc not in nsdfg.out_connectors:
        return None  # in/out (aliased live value) or not an output -> not a fresh scatter accumulator
    inner = nsdfg.sdfg
    oc_desc = inner.arrays.get(oc)
    if oc_desc is None or _is_single_element(oc_desc):
        return None
    in_conns = set(nsdfg.in_connectors.keys())
    found = None
    for ist in inner.all_states():
        for e in ist.edges():
            if (e.data is not None and e.data.wcr is not None and e.data.data == oc and e.data.subset is not None
                    and e.data.subset.num_elements() == 1 and isinstance(e.dst, nodes.AccessNode) and e.dst.data == oc
                    and ist.out_degree(e.dst) == 0):
                if not _data_dependent_index(inner, e.data.subset, in_conns):
                    return None  # constant slot / affine per-element -> not a scatter reduction
                if found is not None:
                    return None  # more than one write-only reduction sink -> not this shape
                found = e
    return found


def is_data_dependent_scatter_sink(nsdfg: nodes.NestedSDFG, oc: str) -> bool:
    """True if ``oc`` is a data-dependent multi-element scatter WCR sink of ``nsdfg`` for
    ANY reduction operator -- the fail-safe refuse predicate shared by the WCR-normalization
    passes (see :func:`data_dependent_scatter_wcr_edge`). Distinct from
    :func:`scatter_reduction_wcr_edge`, which additionally requires an OpenMP-reducible
    operator (the subset this pass can privatise)."""
    return data_dependent_scatter_wcr_edge(nsdfg, oc) is not None


def scatter_reduction_wcr_edge(nsdfg: nodes.NestedSDFG, oc: str):
    """Return the inner single-element data-dependent WCR edge that accumulates into a
    write-only bounded output ``oc`` of ``nsdfg`` (the scatter-reduction signature) when
    the reducer is one Strategy A supports (``+`` / ``*`` / ``min`` / ``max``), else
    ``None``.

    This is :func:`data_dependent_scatter_wcr_edge` narrowed to the OpenMP-reducible
    operators -- the exact set :class:`PrivatizeScatterReduction` can surface into an
    array-section ``reduction(...)`` clause. A non-reducible op (``-`` / ``/`` / a custom
    lambda) is still a scatter sink (:func:`is_data_dependent_scatter_sink` fires) but is
    left to the correct contended atomic here.
    """
    found = data_dependent_scatter_wcr_edge(nsdfg, oc)
    if found is None:
        return None
    return found if scatter_wcr_op(found.data.wcr) in SCATTER_REDUCIBLE_OPS else None


@properties.make_properties
@xf.explicit_cf_compatible
class PrivatizeScatterReduction(ppl.Pass):
    """Surface data-dependent scatter reductions to a whole-buffer map WCR so codegen
    privatises the accumulator with an OpenMP array-section ``reduction(...)`` clause
    instead of a contended per-element atomic. See the module docstring.
    """

    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Memlets

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> Set:
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results) -> Optional[int]:
        """Surface every eligible scatter reduction in ``sdfg`` (and nested SDFGs).

        :returns: the number of scatter reductions surfaced, or ``None`` if none.
        """
        count = 0
        for sd in sdfg.all_sdfgs_recursive():
            for state in sd.all_states():
                for nsdfg in [n for n in state.nodes() if isinstance(n, nodes.NestedSDFG)]:
                    if not isinstance(state.entry_node(nsdfg), nodes.MapEntry):
                        continue
                    for oc in list(nsdfg.out_connectors.keys()):
                        if surface_scatter_reduction(state, nsdfg, oc):
                            count += 1
        return count or None


def map_is_parallel(state: SDFGState, map_entry: nodes.MapEntry) -> bool:
    """True if ``map_entry`` runs across multiple threads -- the only case a data-dependent
    scatter reduction benefits from OpenMP array-section privatisation.

    Privatising the accumulator (a per-thread private copy + runtime tree-merge) removes
    the cache-line contention of ``T`` threads atomically hammering a shared ``hist``. A
    SEQUENTIAL scatter has no such contention: a single thread accumulates in place, so a
    plain serial WCR is already optimal, and an OpenMP ``reduction(...)`` clause on a
    non-parallel loop is a no-op at best. A map is parallel iff it is NOT nested inside
    another map (an inner map runs single-threaded per outer iteration) AND its schedule is
    not explicitly ``Sequential`` (a top-level map with an unset / ``Default`` schedule
    still lowers to a parallel ``#pragma omp parallel for``, so it counts as parallel)."""
    if state.entry_node(map_entry) is not None:
        return False  # nested under an outer map -> single-threaded per outer iteration
    return map_entry.map.schedule != dtypes.ScheduleType.Sequential


def resolve_root_data(state: SDFGState, node: nodes.AccessNode) -> str:
    """The underlying array name ``node`` ultimately accesses: follow a View chain to its
    root viewed AccessNode (:func:`~dace.sdfg.utils.get_last_view_node`), else ``node``'s
    own ``data``. A plain array resolves to itself; a View (or a chain of Views) resolves
    to the concrete array it aliases, so a self-referential accumulator read *through a
    View* is not missed by a name-only comparison."""
    root = get_last_view_node(state, node)
    return root.data if root is not None else node.data


def surface_scatter_reduction(state: SDFGState, nsdfg: nodes.NestedSDFG, oc: str) -> bool:
    """Surface the scatter reduction on output ``oc`` of ``nsdfg`` onto the
    ``NestedSDFG -> MapExit -> accumulator`` edge chain. Returns ``True`` on rewrite.

    Detection is complete before any mutation; an ineligible shape returns ``False``
    with the graph untouched.
    """
    wcr_edge = scatter_reduction_wcr_edge(nsdfg, oc)
    if wcr_edge is None:
        return False
    # The outer edge to the MapExit must still be plain (idempotence: a surfaced
    # scatter has a WCR here already, so a re-run no-ops).
    out_edge = next((oe for oe in state.out_edges(nsdfg) if oe.src_conn == oc), None)
    if out_edge is None or out_edge.data.wcr is not None or not isinstance(out_edge.dst, nodes.MapExit):
        return False
    # Resolve the accumulator AccessNode at the end of the memlet path.
    path = list(state.memlet_path(out_edge))
    accumulator = path[-1].dst
    if not isinstance(accumulator, nodes.AccessNode):
        return False
    # Refuse a self-referential / read-then-accumulate accumulator: if the map also
    # READS the accumulator array, a whole-buffer privatisation would make those reads
    # see the private identity copy (the live contribution is silently dropped). Compare
    # on the RESOLVED root array, not the AccessNode name: a read through a View (or a
    # differently-named alias) of the accumulator would slip past a name-only check.
    map_entry = state.entry_node(nsdfg)
    # Only a PARALLEL scatter benefits from OpenMP array-section privatisation. A sequential
    # map (explicitly ``Sequential``, or nested inside an outer map so it runs single-threaded
    # per outer iteration) has no cross-thread contention -- a plain serial WCR accumulate is
    # already optimal, and a ``reduction(...)`` clause on a non-parallel loop buys nothing --
    # so leave it untouched (the NormalizeWCR whole-buffer refuse-guard still protects it).
    if not map_is_parallel(state, map_entry):
        return False
    acc_root = resolve_root_data(state, accumulator)
    if any(
            isinstance(ie.src, nodes.AccessNode) and resolve_root_data(state, ie.src) == acc_root
            for ie in state.in_edges(map_entry)):
        return False
    # Surface: copy the reduction operator onto every edge of the accumulator path.
    for e in path:
        e.data.wcr = wcr_edge.data.wcr
    return True


__all__ = [
    'PrivatizeScatterReduction', 'surface_scatter_reduction', 'scatter_reduction_wcr_edge',
    'data_dependent_scatter_wcr_edge', 'is_data_dependent_scatter_sink', 'map_is_parallel', 'resolve_root_data',
    'scatter_wcr_op', 'SCATTER_REDUCIBLE_OPS'
]
