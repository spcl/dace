# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Expose wavefront parallelism in 2-D loop nests by loop skewing (ISL-backed).

A 2-D nest of the form ::

    for u in range(u_lo, u_hi):
        for v in range(v_lo, v_hi):
            arr[f(u), g(v)] = h(arr[..], arr[..], ...)

carries dependences whose distance vectors are all lexicographically backward,
so neither loop parallelises on its own -- yet the *anti-diagonal* is parallel.
The classical unimodular skew ``(u, v) -> (t = a*u + b*v, p = v)`` makes one such
diagonal the sequential ``t`` axis and leaves ``p`` free, after which the inner
``p``-loop lifts to a parallel Map.

This pass generalises the textbook single-tasklet rectangular case (TSVC
``s2111``) to the full affine family that shows up in practice:

* **Affine, non-identity write index** -- ``table[N-1-u, v]`` (the shape a
  negative-stride normalisation leaves behind, as in polybench ``nussinov``).
  The dependence distances are computed in *iteration* space by inverting the
  write's affine index map, not by subtracting the loop variable.
* **Imperfect, multi-state bodies** -- guards, several writing states, and a
  nested reduction loop. Reads are collected from the whole inner-loop body;
  a read inside an enclosing reduction loop contributes a *parametric* distance
  bounded by that loop's range.
* **Triangular domains** -- ``for v in range(N-u, N)``. The skewed loop bounds
  come from an exact ISL projection of the (triangular) iteration polyhedron,
  which folds a bound like ``v >= N-u`` into the ``t``-range constraint
  ``t >= N`` automatically.

Legality is the classical Lamport / Feautrier test: a skew ``tau`` is legal iff
``tau . delta < 0`` for every dependence ``delta`` over the whole domain. It is
decided exactly (integer emptiness) by :mod:`~dace.transformation.passes.
canonicalize.wavefront_polyhedron`. Crucially the pass **only** skews a genuine
wavefront -- a nest where *neither* axis-aligned schedule is already legal
(``tau=(1,0)`` = inner-parallel, ``tau=(0,1)`` = outer-parallel). A stencil that
is already a parallel-map-over-sequential-scan (or scan-over-map) keeps its
clean structure; the skew never clobbers it.

``islpy`` is an optional dependency. Without it the pass is a no-op -- loops stay
sequential and the ``pinned_sequential`` safety net preserves the
never-slower-than-``auto_optimize`` guarantee.

References:

- Lamport, *"The parallel execution of DO loops"* (CACM '74) -- the hyperplane /
  wavefront method and its legality condition.
- Wolf & Lam, *"A loop transformation theory ..."* (IEEE TPDS '91).
- Bondhugula et al., *"A practical automatic polyhedral parallelizer ..."*
  (PLDI '08) -- Pluto, the affine-schedule generalisation.
"""
import copy
from typing import Dict, List, Optional, Tuple

import dace
from dace import SDFG, properties, symbolic
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.canonicalize import wavefront_polyhedron as poly

#: Prefix for the synthesised skewed iterators.
_SKEW_T_PREFIX = '_skew_t_'
_SKEW_P_PREFIX = '_skew_p_'

#: Suffix ``SplitStatements`` gives the per-iteration anti-dependence snapshot it
#: inserts (``arr`` -> ``arr_split_snap``). Recognising it lets the skew absorb
#: the snapshot back into the live array (see :func:`absorb_split_snapshots`).
_SPLIT_SNAP_SUFFIX = '_split_snap'

#: Candidate diagonal skews, in preference order. ``tau = (a, b)``; the skew is
#: unimodular when ``|a| == 1`` (``p = v``, ``u = a*(t - b*p)``) or ``|b| == 1``
#: (``p = u``, ``v = b*(t - a*p)``). The shallow 45-degree family comes first:
#: ``(1, 1)`` sum diagonal (heat-flux / Smith-Waterman / nussinov), ``(1, -1)``
#: difference diagonal. The steeper diagonals follow -- ``(2, +-1)`` is the
#: Gauss-Seidel case whose stored deps ``{(0,-1),(-1,0),(-1,-1),(-1,1)}`` need
#: ``a > b > 0`` (seidel_2d), the ``(1, +-2)`` transposes cover the reflected
#: nests. ``b = 0`` / ``a = 0`` is not a skew (that is the axis-aligned schedule
#: tested for refusal).
_SKEW_CANDIDATES: Tuple[Tuple[int, int], ...] = ((1, 1), (1, -1), (2, 1), (2, -1), (1, 2), (1, -2))


def sym(name: str):
    """A globally-registered DaCe symbol for ``name`` (never a raw sympy symbol,
    so assumptions / registration match the symbols already in the SDFG)."""
    return symbolic.pystr_to_symbolic(name)


class WriteMap:
    """The separable unit-affine index map of the carrier's write:
    ``row = c0 + c1*U``, ``col = d0 + d2*V`` (or the transpose, ``U`` and ``V``
    swapped between the axes). ``c1, d2 in {1, -1}`` so it inverts exactly."""

    def __init__(self, u: str, v: str, c0, c1, d0, d2, transposed: bool):
        self.u = u
        self.v = v
        self.c0, self.c1, self.d0, self.d2 = c0, c1, d0, d2
        self.transposed = transposed

    def invert(self, row_expr, col_expr) -> Tuple[object, object]:
        """Iteration coordinates ``(u_r, v_r)`` that write array cell
        ``(row_expr, col_expr)``. Exact because ``c1, d2 in {1, -1}``."""
        if not self.transposed:
            u_r = (row_expr - self.c0) * self.c1     # c1 in {1,-1} => 1/c1 == c1
            v_r = (col_expr - self.d0) * self.d2
        else:
            u_r = (col_expr - self.d0) * self.d2
            v_r = (row_expr - self.c0) * self.c1
        return symbolic.simplify(u_r), symbolic.simplify(v_r)


def split_var(expr, name: str) -> Tuple[object, object]:
    """``(coeff, remainder)`` splitting the named symbol out of an affine expr;
    ``remainder`` no longer contains that symbol. Matches by name."""
    e = symbolic.simplify(expr)
    for s in e.free_symbols:
        if s.name == name:
            c = e.coeff(s, 1)
            return c, symbolic.simplify(e - c * s)
    return symbolic.pystr_to_symbolic(0), e


def parse_write_map(row_expr, col_expr, u: str, v: str) -> Optional[WriteMap]:
    """Recognise ``row/col`` as a separable unit-affine map of ``(u, v)``.
    Returns a :class:`WriteMap` or ``None`` if not separable-unit."""

    def axis(expr):
        cu, rem_u = split_var(expr, u)
        cv, _ = split_var(expr, v)
        has_u = cu != 0
        has_v = cv != 0
        if has_u and not has_v and cu in (1, -1):
            return ('u', cu, rem_u)
        if has_v and not has_u and cv in (1, -1):
            _, rem_v = split_var(expr, v)
            return ('v', cv, rem_v)
        return None

    ra, rb = axis(row_expr), axis(col_expr)
    if ra is None or rb is None:
        return None
    # remainders must be free of u, v (pure offset / parameter).
    for _, _, rem in (ra, rb):
        names = {s.name for s in symbolic.simplify(rem).free_symbols}
        if u in names or v in names:
            return None
    if ra[0] == 'u' and rb[0] == 'v':          # row carries u, col carries v
        return WriteMap(u, v, ra[2], ra[1], rb[2], rb[1], transposed=False)
    if ra[0] == 'v' and rb[0] == 'u':          # transpose: row carries v, col carries u
        return WriteMap(u, v, ra[2], ra[1], rb[2], rb[1], transposed=True)
    return None


def unit_positive_stride(loop: LoopRegion) -> bool:
    s = loop_analysis.get_loop_stride(loop)
    try:
        return s is not None and int(symbolic.simplify(s)) == 1
    except (TypeError, ValueError):
        return False


def is_split_snapshot_state(state: SDFGState) -> bool:
    """``state`` is a pure anti-dependence snapshot ``arr_split_snap = arr`` -- one
    copy edge ``AccessNode(arr) -> AccessNode(arr_split_snap)`` and nothing else.
    ``SplitStatements`` inserts exactly this before a loop to break a
    per-iteration anti-dependence; the wavefront can absorb it (the diagonal
    schedule already reads the old value before it is overwritten)."""
    ns = list(state.nodes())
    if len(ns) != 2 or not all(isinstance(n, nodes.AccessNode) for n in ns):
        return False
    edges = list(state.edges())
    if len(edges) != 1:
        return False
    e = edges[0]
    if not (isinstance(e.src, nodes.AccessNode) and isinstance(e.dst, nodes.AccessNode)):
        return False
    return e.dst.data == f'{e.src.data}{_SPLIT_SNAP_SUFFIX}'


def extract_two_level_nest(outer: LoopRegion) -> Optional[LoopRegion]:
    """The single inner :class:`LoopRegion` perfectly nested in ``outer`` (its
    body may itself be imperfect); ``None`` if ``outer`` holds anything else with
    data alongside the one inner loop. A pure ``arr_split_snap = arr`` snapshot
    state is tolerated -- :func:`absorb_split_snapshots` folds it away before the
    skew reasons about dependences."""
    blocks = list(outer.nodes())
    inner = [b for b in blocks if isinstance(b, LoopRegion)]
    if len(inner) != 1:
        return None
    for b in blocks:
        if b is inner[0]:
            continue
        if isinstance(b, SDFGState) and len(list(b.nodes())) > 0 and not is_split_snapshot_state(b):
            return None
    return inner[0]


def lex_sign(offset: List[object]) -> Optional[int]:
    """Sign of the first non-zero component of a distance vector: ``+1`` forward,
    ``-1`` backward, ``0`` for the zero vector, or ``None`` when that first
    non-zero component is not a compile-time constant (undecidable)."""
    for c in offset:
        cs = symbolic.simplify(c)
        if cs == 0:
            continue
        if cs.is_number:
            return 1 if cs > 0 else -1
        return None
    return 0


def live_reader(state: SDFGState, name: str) -> nodes.AccessNode:
    """A read-only ``AccessNode(name)`` in ``state`` (``in_degree == 0``), reusing
    one if present so redirected reads coalesce on the existing input node."""
    for n in state.data_nodes():
        if n.data == name and state.in_degree(n) == 0:
            return n
    return state.add_access(name)


def absorb_split_snapshots(outer: LoopRegion, inner: LoopRegion, sdfg: SDFG) -> bool:
    """Redirect every inner-body read of an ``arr_split_snap`` snapshot back to the
    live array ``arr`` and drop the snapshot copy, so the skew reasons about the
    real (anti) dependence directly instead of the redundant snapshot.

    Sound because the snapshot only ever stands in for a *forward* read -- an
    element a later iteration overwrites (``a[i, j+1]`` / ``a[i+1, j]`` in an
    in-place Gauss-Seidel). Its snapshotted value is the not-yet-overwritten old
    element, which both the current sequential order *and* the diagonal wavefront
    preserve (the writer runs on a strictly later diagonal). Refuses -- leaving the
    SDFG untouched -- if any snapshot read is same-row backward (redirecting would
    change its value) or its offset from the write is undecidable.

    Returns ``True`` when there is nothing to absorb (already a perfect nest) or
    the absorb succeeded; ``False`` to abort the skew with no mutation."""
    copy_states = [
        b for b in outer.nodes()
        if isinstance(b, SDFGState) and b is not inner and is_split_snapshot_state(b)
    ]
    if not copy_states:
        return True

    snap_src: Dict[str, str] = {}  # snapshot array -> live source array
    for st in copy_states:
        e = list(st.edges())[0]
        snap_src[e.dst.data] = e.src.data
    live_names = set(snap_src.values())

    write_idx: Dict[str, List[List[object]]] = {}  # live array -> point write indices
    snap_reads: List[Tuple[SDFGState, nodes.AccessNode, object, List[object], str]] = []
    for state in inner.all_states():
        for node in state.data_nodes():
            if node.data in snap_src:
                for e in state.out_edges(node):
                    if e.data is None or e.data.subset is None:
                        return False
                    ridx = point_index(e.data.subset)
                    if ridx is None:
                        return False
                    snap_reads.append((state, node, e, ridx, snap_src[node.data]))
            if node.data in live_names:
                for e in state.in_edges(node):
                    if e.data is None or e.data.subset is None:
                        continue
                    widx = point_index(e.data.subset)
                    if widx is None:
                        return False
                    write_idx.setdefault(node.data, []).append(widx)

    # Every snapshot read must be lexicographically forward of (or equal to) the
    # live write it shadows -- only then does the old snapshot value match the
    # live element under both the sequential and the diagonal schedule.
    for (_st, _node, _e, ridx, src_name) in snap_reads:
        writes = write_idx.get(src_name)
        if not writes:
            return False
        for widx in writes:
            if len(widx) != len(ridx):
                return False
            if lex_sign([r - w for r, w in zip(ridx, widx)]) not in (0, 1):
                return False

    # Commit: rewire snapshot reads onto the live array, drop the orphaned
    # snapshot readers, then empty the copy states (structural cleanup removes
    # the now-empty state and eliminates the dead ``arr_split_snap`` array).
    for (state, _snap_node, e, _ridx, src_name) in snap_reads:
        reader = live_reader(state, src_name)
        redirected = copy.deepcopy(e.data)
        redirected.data = src_name
        state.add_edge(reader, None, e.dst, e.dst_conn, redirected)
        state.remove_edge(e)
    for (state, snap_node, _e, _ridx, _src) in snap_reads:
        if snap_node in state.nodes() and state.degree(snap_node) == 0:
            state.remove_node(snap_node)
    for st in copy_states:
        for n in list(st.nodes()):
            st.remove_node(n)
    return True


def point_index(subset) -> Optional[List[object]]:
    """The per-dimension index of a *point* subset (``start == end`` on every
    axis); ``None`` if any axis is a range."""
    idx = []
    for (start, end, _step) in subset.ndrange():
        if start != end:
            return None
        idx.append(symbolic.pystr_to_symbolic(start))
    return idx


def loop_bounds(loop: LoopRegion) -> Optional[Tuple[object, object]]:
    lo = loop_analysis.get_init_assignment(loop)
    hi = loop_analysis.get_loop_end(loop)
    if lo is None or hi is None:
        return None
    return symbolic.pystr_to_symbolic(lo), symbolic.pystr_to_symbolic(hi)


def nested_loop_context(state: SDFGState, inner: LoopRegion) -> Optional[List[Tuple[str, object, object]]]:
    """The reduction loops strictly between ``state`` and ``inner`` (inclusive of
    neither), innermost first: ``[(var, lo, hi), ...]``. ``None`` if a non-unit
    stride loop is on the path (its range does not translate to a clean
    interval)."""
    ctx: List[Tuple[str, object, object]] = []
    g = state.parent_graph
    while g is not None and g is not inner:
        if isinstance(g, LoopRegion) and g.loop_variable:
            if not unit_positive_stride(g):
                return None
            b = loop_bounds(g)
            if b is None:
                return None
            ctx.append((g.loop_variable, b[0], b[1]))
        g = g.parent_graph if g is not None else None
    return ctx


class Dependence:
    """One carried dependence: distance ``(du, dv) = writer_iteration - current``
    (possibly parametric in the enclosing reduction-loop vars ``nested``, each
    with an interval), plus its ``kind``.

    ``kind`` is ``'flow'`` when the current iteration reads a value the sweep
    already produced (the writer is lexicographically *before* the current
    iteration -- a backward distance), and ``'anti'`` when it reads a value that
    a *later* iteration overwrites (the writer is lexicographically *after* --
    a forward distance, as in the in-place Gauss-Seidel reads ``A[i, j+1]`` /
    ``A[i+1, j]`` of the old value). The two impose opposite-signed legality
    constraints (see :func:`schedule_legal`): flow needs ``tau . delta < 0``
    (producer before consumer), anti needs ``tau . delta > 0`` (read before the
    overwrite). Treating an anti dependence as flow -- the pre-fix behaviour --
    makes both the backward and forward reads demand contradictory signs, so no
    skew is ever found for a symmetric stencil."""

    def __init__(self, du, dv, nested: List[Tuple[str, object, object]], kind: str = 'flow'):
        self.du = symbolic.simplify(du)
        self.dv = symbolic.simplify(dv)
        self.nested = nested
        self.kind = kind


def dependence_kind(du, dv) -> str:
    """Classify a distance ``(du, dv) = writer - current`` as ``'flow'`` or
    ``'anti'`` by the lexicographic sign of its first non-zero component (a
    positive first component means the writer runs after the current iteration,
    so the read sees the soon-to-be-overwritten old value -- an anti dependence).

    Only *numeric* (constant) distances are classified as ``'anti'``; a distance
    with a symbolic component keeps the conservative ``'flow'`` treatment. This is
    sound: a symbolic forward read handled as flow can only cause the skew to be
    *refused* (its flow constraint is unsatisfiable under the symbol-positivity
    assumption), never mis-scheduled -- and every symbolic case in practice is a
    backward (flow) read anyway."""
    du_s, dv_s = symbolic.simplify(du), symbolic.simplify(dv)
    if not (du_s.is_number and dv_s.is_number):
        return 'flow'
    if du_s > 0 or (du_s == 0 and dv_s > 0):
        return 'anti'
    return 'flow'


def collect_carrier(inner: LoopRegion, sdfg: SDFG,
                    u: str, v: str) -> Optional[Tuple[str, WriteMap, List[Dependence]]]:
    """Find the unique carrier array (written *and* self-read with a non-zero
    distance) in ``inner``'s body, its write map, and its dependences. ``None``
    if there is no clean single carrier (refuse)."""
    writes: Dict[str, List[List[object]]] = {}
    reads: Dict[str, List[Tuple[List[object], List[Tuple[str, object, object]]]]] = {}
    for state in inner.all_states():
        ctx = None
        for node in state.data_nodes():
            desc = sdfg.arrays.get(node.data)
            if desc is None or len(desc.shape) != 2:
                continue
            if ctx is None:
                ctx = nested_loop_context(state, inner)
                if ctx is None:
                    return None
            for e in state.in_edges(node):
                if e.data is None or e.data.subset is None:
                    continue
                idx = point_index(e.data.subset)
                if idx is None:
                    return None            # non-point write -> refuse
                writes.setdefault(node.data, []).append(idx)
            for e in state.out_edges(node):
                if e.data is None or e.data.subset is None:
                    continue
                idx = point_index(e.data.subset)
                if idx is None:
                    return None            # non-point read of a 2-D carrier -> refuse
                reads.setdefault(node.data, []).append((idx, ctx))

    carriers: List[Tuple[str, WriteMap, List[Dependence]]] = []
    for arr, wsubs in writes.items():
        wmap = consistent_write_map(wsubs, u, v)
        if wmap is None:
            return None                    # written but not separable-unit affine -> refuse
        deps: List[Dependence] = []
        for (idx, ctx) in reads.get(arr, []):
            u_r, v_r = wmap.invert(idx[0], idx[1])
            du = symbolic.simplify(u_r - sym(u))
            dv = symbolic.simplify(v_r - sym(v))
            if du == 0 and dv == 0:
                continue                   # in-place self-read, not a dependence
            deps.append(Dependence(du, dv, ctx, dependence_kind(du, dv)))
        if deps:
            carriers.append((arr, wmap, deps))
    if len(carriers) != 1:
        return None                        # zero or several carriers -> refuse
    return carriers[0]


def consistent_write_map(write_subs: List[List[object]], u: str, v: str) -> Optional[WriteMap]:
    """A single :class:`WriteMap` agreeing with *every* write subset, else ``None``."""
    wmap = None
    for idx in write_subs:
        wm = parse_write_map(idx[0], idx[1], u, v)
        if wm is None:
            return None
        if wmap is None:
            wmap = wm
        elif (wm.transposed, wm.c0, wm.c1, wm.d0, wm.d2) != (wmap.transposed, wmap.c0, wmap.c1, wmap.d0, wmap.d2):
            return None
    return wmap


def domain_constraints(u: str, v: str, ub: Tuple[object, object],
                       vb: Tuple[object, object]) -> List[object]:
    """The 2-D iteration polyhedron as exprs, each ``>= 0``."""
    U, V = sym(u), sym(v)
    return [U - ub[0], ub[1] - U, V - vb[0], vb[1] - V]


def tau_dot(tau: Tuple[int, int], dep: Dependence):
    a, b = tau
    return symbolic.simplify(a * dep.du + b * dep.dv)


def dep_dims_and_cons(dep: Dependence, u: str, v: str, domain: List[object],
                      assume: List[object]) -> Tuple[List[str], List[object]]:
    """Dims + full constraint list (domain + this dep's nested ranges + assumptions)."""
    dims = [u, v] + [nm for (nm, _, _) in dep.nested]
    cons = list(domain)
    for (nm, lo, hi) in dep.nested:
        S = sym(nm)
        cons += [S - lo, hi - S]
    cons += list(assume)
    return dims, cons


def params_of(cons: List[object], dims: List[str]) -> List[str]:
    names = set()
    for c in cons:
        names |= {s.name for s in symbolic.simplify(c).free_symbols}
    return sorted(names - set(dims))


def schedule_legal(tau: Tuple[int, int], deps: List[Dependence], u: str, v: str,
                   domain: List[object], assume: List[object]) -> bool:
    """``tau`` is legal iff every dependence is strictly ordered on the sequential
    ``t`` axis. For a **flow** dependence the producer must precede the consumer
    (``tau.delta < 0``, i.e. no domain point with ``tau.delta >= 0``); for an
    **anti** dependence the read must precede the overwrite (``tau.delta > 0``,
    i.e. no domain point with ``tau.delta <= 0``). ``delta`` is the stored
    ``(du, dv) = writer - current``."""
    for dep in deps:
        dims, cons = dep_dims_and_cons(dep, u, v, domain, assume)
        # Add the constraint whose satisfiable region is the *illegal* one: for
        # flow that is ``tau.delta >= 0``, for anti ``tau.delta <= 0`` (rendered
        # as ``-tau.delta >= 0``). ``tau`` is legal for this dep iff that region
        # is empty over the domain.
        td = tau_dot(tau, dep)
        cons = cons + [td if dep.kind == 'flow' else symbolic.simplify(-td)]
        try:
            empty = poly.is_domain_empty(dims, params_of(cons, dims), cons)
        except ValueError:
            return False       # non-affine / unmapped -> cannot prove -> illegal
        if not empty:
            return False
    return True


def offset_symbols(deps: List[Dependence], dims: List[str]) -> List[object]:
    """Distinct parameter symbols appearing in any distance component."""
    nested_names = {nm for d in deps for (nm, _, _) in d.nested}
    syms = {}
    for dep in deps:
        for comp in (dep.du, dep.dv):
            for s in symbolic.simplify(comp).free_symbols:
                if s.name not in dims and s.name not in nested_names:
                    syms[s.name] = s
    return list(syms.values())


@properties.make_properties
@xf.explicit_cf_compatible
class WavefrontSkew(ppl.Pass):
    """Skew genuine 2-D wavefront nests so the inner loop lifts to a parallel Map.

    Backed by an exact ISL legality + bound projection; refuses anything that is
    not a genuine wavefront (already-parallel axis, non-affine carrier, several
    carriers, non-unit strides)."""

    CATEGORY: str = 'Canonicalization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.Symbols

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _) -> Optional[int]:
        """Skew every eligible 2-D nest. Returns the count or ``None`` on no match
        (also ``None`` when ``islpy`` is unavailable -- the pass degrades to a
        no-op and the loops stay sequential)."""
        if not poly.HAVE_ISL:
            return None
        skewed = 0
        for sd in sdfg.all_sdfgs_recursive():
            for cfg in list(sd.all_control_flow_regions()):
                if not (isinstance(cfg, LoopRegion) and cfg.loop_variable):
                    continue
                parent = cfg.parent_graph
                if parent is None or cfg not in parent.nodes():
                    continue           # stale snapshot: a prior skew removed this node
                if self._try_skew(cfg, sd):
                    skewed += 1
        return skewed or None

    def _try_skew(self, outer: LoopRegion, sdfg: SDFG) -> bool:
        if not unit_positive_stride(outer):
            return False
        inner = extract_two_level_nest(outer)
        if inner is None or not unit_positive_stride(inner):
            return False
        ub = loop_bounds(outer)
        vb = loop_bounds(inner)
        if ub is None or vb is None:
            return False
        u, v = outer.loop_variable, inner.loop_variable
        # The inner bound must not leak the inner var (malformed); the outer var
        # in the inner bound is fine -- that is the triangular case ISL handles.
        vsym = sym(v)
        if vsym in symbolic.simplify(vb[0]).free_symbols or vsym in symbolic.simplify(vb[1]).free_symbols:
            return False

        # Fold away any per-iteration anti-dependence snapshot SplitStatements
        # left in the outer body (the imperfect-nest cause) so the dependence is
        # carried by the diagonal schedule instead. Correctness-preserving on its
        # own, so a later skew refusal leaves a valid (still sequential) nest.
        if not absorb_split_snapshots(outer, inner, sdfg):
            return False

        carrier = collect_carrier(inner, sdfg, u, v)
        if carrier is None:
            return False
        _arr, _wmap, deps = carrier

        domain = domain_constraints(u, v, ub, vb)
        dims = [u, v]

        # --- Genuine-wavefront guard: refuse if an axis is already parallel. ---
        # tau=(1,0): inner v parallel (map-in-inner / column-independent stencil).
        # tau=(0,1): outer u parallel (map-of-scans / row-independent stencil).
        # Either legal -> a plain LoopToMap yields the parallel axis; do NOT skew.
        if schedule_legal((1, 0), deps, u, v, domain, []):
            return False
        if schedule_legal((0, 1), deps, u, v, domain, []):
            return False

        # --- Pick a legal diagonal skew, using symbol positivity if declared. ---
        off_syms = offset_symbols(deps, dims)
        assume_annotated = [s - 1 for s in off_syms if s.is_positive]
        tau = None
        guard_syms: List[object] = []
        for cand in _SKEW_CANDIDATES:
            if schedule_legal(cand, deps, u, v, domain, assume_annotated):
                tau = cand
                break
        if tau is None:
            # Optimistic retry: also assume the unannotated offset symbols are
            # positive, and plant a runtime guard for them.
            assume_all = [s - 1 for s in off_syms]
            for cand in _SKEW_CANDIDATES:
                if schedule_legal(cand, deps, u, v, domain, assume_all):
                    tau = cand
                    guard_syms = [s for s in off_syms if not s.is_positive]
                    break
        if tau is None:
            return False

        probe_t = f'{_SKEW_T_PREFIX}probe'
        probe_p = f'{_SKEW_P_PREFIX}probe'
        bounds = poly.skew_bounds((u, v), params_of(domain, dims), domain, tau, probe_t, probe_p)
        if bounds is None:
            return False

        if guard_syms:
            self._emit_positive_guard(outer, deps, guard_syms)

        self._rewrite(outer, inner, sdfg, u, v, tau, bounds)
        return True

    def _rewrite(self, outer: LoopRegion, inner: LoopRegion, sdfg: SDFG, u: str, v: str,
                 tau: Tuple[int, int], bounds) -> None:
        """Relabel ``outer -> t`` and ``inner -> p`` with the projected bounds, then
        substitute the original iterators in terms of ``(t, p)`` in the inner body
        and lift it to a parallel Map. The substitution matches the unimodular
        family ``skew_bounds`` used: ``p = v`` when ``|a| == 1``, else ``p = u``."""
        a, b = tau
        nid = _next_id(sdfg)
        t_var = f"{_SKEW_T_PREFIX}{nid}"
        p_var = f"{_SKEW_P_PREFIX}{nid}"
        sdfg.add_symbol(t_var, dace.int64)
        sdfg.add_symbol(p_var, dace.int64)
        subs = {f'{_SKEW_T_PREFIX}probe': t_var, f'{_SKEW_P_PREFIX}probe': p_var}

        t_lo = bound_expr(bounds.t_lo_terms, subs, 'max')
        t_hi = bound_expr(bounds.t_hi_terms, subs, 'min')
        p_lo = bound_expr(bounds.p_lo_terms, subs, 'max')
        p_hi = bound_expr(bounds.p_hi_terms, subs, 'min')

        outer.loop_variable = t_var
        outer.init_statement = properties.CodeBlock(f"{t_var} = ({t_lo})")
        outer.loop_condition = properties.CodeBlock(f"{t_var} <= ({t_hi})")
        outer.update_statement = properties.CodeBlock(f"{t_var} = {t_var} + 1")
        # The diagonal ``t`` axis carries every wavefront dependence by
        # construction (that is why ``p`` is free); pin it so a downstream
        # LoopToMap / LoopToReduce never races it into a parallel map.
        outer.pinned_sequential = True

        inner.loop_variable = p_var
        inner.init_statement = properties.CodeBlock(f"{p_var} = ({p_lo})")
        inner.loop_condition = properties.CodeBlock(f"{p_var} <= ({p_hi})")
        inner.update_statement = properties.CodeBlock(f"{p_var} = {p_var} + 1")

        # Express the original iterators in (t, p). The parallel axis p is v when
        # |a| == 1 (u = a*(t - b*p)); it is u when the steep skew forces |b| == 1
        # (v = b*(t - a*p)). Both keep the (u, v) -> (t, p) map unimodular.
        if abs(a) == 1:
            u_expr = symbolic.symstr(a * (sym(t_var) - b * sym(p_var)))
            inner.replace_dict({u: u_expr, v: p_var})
        else:
            v_expr = symbolic.symstr(b * (sym(t_var) - a * sym(p_var)))
            inner.replace_dict({u: p_var, v: v_expr})

        self._convert_inner_to_map(outer, inner, sdfg)

    def _convert_inner_to_map(self, outer: LoopRegion, inner: LoopRegion, sdfg: SDFG) -> None:
        """Lift the skewed inner ``p``-loop to a Map via ``LoopToMap.apply``,
        bypassing ``can_be_applied``: independence of the ``p``-iterations at
        fixed ``t`` is guaranteed by the legality proof (``tau.delta < 0`` for
        every dependence => no intra-``t`` dependence). An exception here signals
        a real upstream bug and is intentionally not swallowed."""
        from dace.transformation.interstate.loop_to_map import LoopToMap
        instance = LoopToMap()
        instance.loop = inner
        instance.apply(outer, sdfg)

    def _emit_positive_guard(self, outer: LoopRegion, deps: List[Dependence],
                             guard_syms: List[object]) -> None:
        """Plant a ``__builtin_trap`` before ``outer`` that fires if any distance
        component carrying an unannotated symbol is positive at runtime (soundness
        needs it ``<= 0``). Mirrors ``BreakAntiDependence``'s positive guard."""
        gset = {s.name for s in guard_syms}
        exprs = []
        seen = set()
        for dep in deps:
            for comp in (dep.du, dep.dv):
                cs = symbolic.simplify(comp)
                names = {s.name for s in cs.free_symbols}
                if names & gset and not cs.is_number:
                    key = str(cs)
                    if key not in seen:
                        seen.add(key)
                        exprs.append(cs)
        if not exprs:
            return
        parts = ' || '.join(f'(({symbolic.symstr(e)}) > 0)' for e in exprs)
        tag = abs(hash(parts)) & 0xfffffff
        code = f'if ({parts}) {{ __builtin_trap(); }}'
        pre = outer.parent_graph.add_state_before(outer, label=f'_skew_guard_{tag:x}')
        guard = pre.add_tasklet(name=f'_skew_guard_{tag:x}', inputs={}, outputs={},
                                code=code, language=dace.dtypes.Language.CPP)
        guard.side_effects = True


def bound_expr(terms: List[object], subs: Dict[str, str], fn: str) -> str:
    """Render bound terms into a loop-bound string: a single term verbatim, or
    ``max(...)`` / ``min(...)`` (``fn``) of several. ``subs`` renames the probe
    iterators to the real ``t`` / ``p`` symbol names."""
    rendered = [symbolic.symstr(rename_symbols(t, subs)) for t in terms]
    if len(rendered) == 1:
        return rendered[0]
    return f"{fn}(" + ", ".join(rendered) + ")"


def rename_symbols(expr, subs: Dict[str, str]):
    """Rename symbols in ``expr`` by name."""
    e = symbolic.simplify(expr)
    mp = {}
    for s in e.free_symbols:
        if s.name in subs:
            mp[s] = sym(subs[s.name])
    return e.subs(mp)


def _next_id(sdfg: SDFG) -> int:
    """Lowest ``<N>`` no existing ``_skew_(t|p)_<N>`` symbol uses."""
    used = set()
    for sd in sdfg.all_sdfgs_recursive():
        for s in list(sd.symbols.keys()):
            for pre in (_SKEW_T_PREFIX, _SKEW_P_PREFIX):
                if s.startswith(pre) and s[len(pre):].isdigit():
                    used.add(int(s[len(pre):]))
        for cfg in sd.all_control_flow_regions():
            if isinstance(cfg, LoopRegion) and cfg.loop_variable:
                for pre in (_SKEW_T_PREFIX, _SKEW_P_PREFIX):
                    lv = cfg.loop_variable
                    if lv.startswith(pre) and lv[len(pre):].isdigit():
                        used.add(int(lv[len(pre):]))
    n = 0
    while n in used:
        n += 1
    return n


__all__ = ['WavefrontSkew']
