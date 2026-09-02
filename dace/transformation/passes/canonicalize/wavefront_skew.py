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
  write's affine index map, not by subtracting the loop variable. The map need
  not be axis-separable: rebasing a triangular inner loop to a 0-based origin
  folds its old begin into the subscripts (``table[N-u-1, N-u+v]``), and any
  *unimodular* 2x2 integer map inverts exactly (see :class:`WriteMap`).
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
wavefront -- a nest where *neither* axis is already a parallel map in the current
loop order (inner-parallel = ``tau=(1,0)`` legal, outer-parallel =
:func:`outer_axis_parallel`). A stencil that is already a parallel-map-over-
sequential-scan (or scan-over-map) keeps its clean structure; the skew never
clobbers it.

The diagonal is lowered as a skewed **tiling**, not element by element: a
sequential tile-diagonal loop over a parallel tile-column Map over the two
sequential intra-tile loops, the innermost of which keeps unit stride. The
element-granularity diagonal walks memory at stride ``N`` and forks a parallel
region per diagonal -- 0.17-0.18x the plain sequential nest at ``N = 768`` on 4
threads, where the tiled form measures 2.04-2.16x. The tile-index domain is the
same polyhedron under ``u -> Bi*I``, ``v -> Bj*J``, so the very same ISL
projection produces its bounds. Tiling is only taken where it is provably order-
preserving (:func:`tiling_legal`); everything else keeps the untiled lowering.

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
import zlib
from typing import Dict, List, Optional, Tuple

import dace
from dace import SDFG, properties, subsets, symbolic
from dace.sdfg import nodes
from dace.sdfg.analysis import cfg
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.canonicalize import wavefront_polyhedron as poly
from dace.transformation.passes.canonicalize.fuse_consecutive_loops import (commit_guarded_fusion, plan_guarded_fusion)

#: Prefix for the synthesised skewed iterators.
_SKEW_T_PREFIX = '_skew_t_'
_SKEW_P_PREFIX = '_skew_p_'

#: Default extent of a skewed tile on each axis. 64x64 doubles on the paper shapes
#: at N=768: measured 2.04-2.16x over the sequential nest on 4 threads, against
#: 0.17-0.18x for the element-granularity diagonal it replaces. Bigger ``Bj`` is
#: NOT better -- 256 leaves 3 tile columns at N=768, fewer than the thread count.
DEFAULT_TILE_SIZE = 64

#: Default extent of a skewed tile on each axis for the GPU lowering. A tile diagonal replaces one
#: kernel launch per ELEMENT anti-diagonal with one per TILE anti-diagonal: at N = 16746 that is
#: 523 launches instead of 33,491, and it restores unit stride to the innermost access, which the
#: element diagonal walks N-1 elements (131 KB) apart -- a separate memory transaction per lane.
#: 128 makes the intra-tile anti-diagonal at most 128 wide -- two CDNA wavefronts, the low end of
#: the 2-4 warps a thread block wants. One wavefront per block leaves nothing for the scheduler to
#: overlap against a memory stall; more than four buys nothing here, because the anti-diagonal
#: ramps and the extra lanes are masked off for most of the tile.
#:
#: What the size actually trades, at N = 16746 (the campaign size):
#:
#:   B     launches   peak blocks   block width   CUs touched
#:   64         523           262            64           262
#:  128         261           131           128           131
#:  256         131            66           256            66
#:
#: ``peak blocks * block width`` is ~N for every row, and so is the total barrier count: the
#: dependence caps instantaneous parallelism at N however the nest is tiled, and tiling only
#: changes the KIND of barrier. A bigger tile converts kernel launches into ``__syncthreads``,
#: which are far cheaper; it also concentrates the work on fewer CUs, and this is a memory-bound
#: stencil, so aggregate bandwidth pulls the other way. 128 sits between the two.
DEFAULT_GPU_TILE_SIZE = 64

#: Dim names for the tile-index polyhedron handed to ``poly.skew_bounds``, and the
#: PARAMETER names standing for its two tile counts. Handing ISL the counts as opaque
#: parameters (rather than ``int_ceil(N - 1, B)`` inline) keeps the projected diagonal
#: bound affine; an inline integer division comes back as an ISL existential that
#: ``pwaff_bound`` cannot render, and the whole tiling would be refused.
_TILE_I_PROBE = '_skew_ti_probe'
_TILE_J_PROBE = '_skew_tj_probe'
_TILE_NI_PROBE = '_skew_ni_probe'
_TILE_NJ_PROBE = '_skew_nj_probe'

#: Suffix ``BreakAntiDependence`` gives the per-iteration anti-dependence snapshot it
#: inserts (``arr`` -> ``arr_split_snap``). Recognising it lets the skew absorb
#: the snapshot back into the live array (see :func:`commit_split_snapshots`).
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
    """The pass's own DaCe symbol for ``name`` (never a raw sympy symbol). It carries no
    assumptions -- the registry does not hand those back -- so it is only ever the ITERATOR
    spelling, and every expression the pass takes out of the SDFG is re-keyed onto it by
    :func:`canonical_iterators`."""
    return symbolic.pystr_to_symbolic(name)


def canonical_iterators(expr, iters: tuple[str, ...]):
    """``expr`` with every free symbol NAMED in ``iters`` re-keyed onto :func:`sym`'s object.

    The Python frontend mints a loop iterator with explicit ``nonnegative=None`` /
    ``positive=None`` assumptions (``frontend/python/newast.py``), so the ``i`` inside a memlet
    subset is a DIFFERENT sympy symbol from ``sym('i')`` and ``i_subset - sym('i')`` stays an
    uncancelled ``-i + i``. That residual is never structurally zero and never a number, so
    every distance looked non-trivial: forward reads degraded to ``'flow'`` (no skew for the
    5-point Gauss-Seidel), and the loop variables leaked into the emitted runtime guard, where
    they became program arguments no caller can supply.

    Only the ITERATORS are re-keyed. An offset symbol keeps the object carrying its declared
    ``positive=True``, which :func:`offset_symbols` reads to decide whether a guard is needed.
    """
    e = symbolic.pystr_to_symbolic(expr)
    mp = {}
    # sorted: the map is built from a sympy set, whose iteration order is not stable. Every entry is
    # independent so the result cannot differ, but sorting makes that provable rather than incidental.
    for s in sorted(e.free_symbols, key=str):
        name = str(s)
        if name in iters:
            mp[s] = sym(name)
    return e.xreplace(mp) if mp else e


class WriteMap:
    """The carrier write's integer affine index map ``(row, col) = M.(u, v) + c``
    with ``M = ((m00, m01), (m10, m11))`` stored flat and ``c = (c_row, c_col)``
    free of ``u, v``.

    ``M`` must be **unimodular** (``|det M| == 1``): only then is the map a
    bijection of the integer lattice, so :meth:`invert` names the one iteration
    that wrote a given array cell, exactly and over the integers. The axis-
    separable shapes ``row = c0 + c1*u, col = d0 + d2*v`` (and the transpose) are
    the diagonal / anti-diagonal ``M``; the general form is what a triangular
    inner loop leaves behind once its origin is rebased and the old begin folds
    into the subscripts (``col = N - u + v``)."""

    def __init__(self, u: str, v: str, m: Tuple[int, int, int, int], c: Tuple[object, object]) -> None:
        self.u = u
        self.v = v
        self.m = m
        self.c = c
        self.det = m[0] * m[3] - m[1] * m[2]
        if abs(self.det) != 1:
            raise ValueError(f'write map {m} is not unimodular (det={self.det}); it has no integer inverse')

    def invert(self, row_expr, col_expr) -> Tuple[object, object]:
        """Iteration coordinates ``(u_r, v_r)`` that write array cell
        ``(row_expr, col_expr)``. Exact because ``det in {1, -1}``, so the adjugate
        already IS the inverse up to the factor ``1/det == det``."""
        m00, m01, m10, m11 = self.m
        dr = row_expr - self.c[0]
        dc = col_expr - self.c[1]
        u_r = self.det * (m11 * dr - m01 * dc)
        v_r = self.det * (m00 * dc - m10 * dr)
        return symbolic.simplify(u_r), symbolic.simplify(v_r)


def split_var(expr, name: str) -> Tuple[object, object]:
    """``(coeff, remainder)`` splitting the named symbol out of an affine expr;
    ``remainder`` no longer contains that symbol. Matches by name."""
    e = symbolic.simplify(expr)
    # sorted: sympy allows two distinct Symbol objects with the SAME name but different assumptions in one
    # expression, and this returns on the first name match -- which one we split on would otherwise be
    # hash-order dependent, changing the skew decision.
    for s in sorted(e.free_symbols, key=lambda s: (s.name, str(s.assumptions0))):
        if s.name == name:
            c = e.coeff(s, 1)
            return c, symbolic.simplify(e - c * s)
    return symbolic.pystr_to_symbolic(0), e


def integer_value(expr) -> Optional[int]:
    """``expr`` as a Python ``int`` when it is an integer literal, else ``None``."""
    e = symbolic.simplify(expr)
    if e.is_Integer:
        return int(e)
    return None


def affine_coeffs(expr, u: str, v: str) -> Optional[Tuple[int, int, object]]:
    """``(cu, cv, rest)`` for ``expr == cu*u + cv*v + rest`` with INTEGER ``cu, cv``
    and ``rest`` free of ``u, v``; ``None`` when ``expr`` is not affine in ``(u, v)``
    over the integers.

    Both gates are load-bearing. A non-integer coefficient (``N*u``, or the ``v``
    that a product ``u*v`` leaves behind) has no integer lattice inverse, and a
    ``rest`` that still mentions ``u`` or ``v`` means the leftover is non-linear
    (``u**2``, ``int_floor(u, 2)``) rather than an offset."""
    cu, rem = split_var(expr, u)
    cv, rest = split_var(rem, v)
    iu, iv = integer_value(cu), integer_value(cv)
    if iu is None or iv is None:
        return None
    rest = symbolic.simplify(rest)
    names = dict.fromkeys(s.name for s in rest.free_symbols)
    if u in names or v in names:
        return None
    return iu, iv, rest


def parse_write_map(row_expr, col_expr, u: str, v: str) -> Optional[WriteMap]:
    """Recognise ``row/col`` as a UNIMODULAR integer affine map of ``(u, v)``.
    Returns a :class:`WriteMap`, or ``None`` when the subscripts are not integer
    affine or the map is not unimodular.

    The determinant gate is the legality hinge, not a convenience: the pass turns
    a read's array cell back into the iteration that wrote it by inverting this
    map. ``det == 0`` means the map is singular -- a whole line of iterations
    writes the same cell, so there is no unique writer to name. ``|det| > 1``
    means the image is a strict sublattice, so the preimage of a read cell is
    rational and every dependence distance derived from it would be wrong. Only
    ``|det| == 1`` gives an integer inverse, hence exact distances.

    Nothing downstream is relaxed: the map only widens what can be PARSED. The
    axis-separable family this used to be limited to is unchanged -- a separable
    integer map is diagonal or anti-diagonal, and such a matrix has ``|det| == 1``
    exactly when both non-zero entries are ``+-1``, which is the old
    ``c1, d2 in {1, -1}`` condition verbatim."""
    ra = affine_coeffs(row_expr, u, v)
    rb = affine_coeffs(col_expr, u, v)
    if ra is None or rb is None:
        return None
    m = (ra[0], ra[1], rb[0], rb[1])
    if abs(m[0] * m[3] - m[1] * m[2]) != 1:
        return None
    return WriteMap(u, v, m, (ra[2], rb[2]))


def unit_positive_stride(loop: LoopRegion) -> bool:
    s = loop_analysis.get_loop_stride(loop)
    try:
        return s is not None and int(symbolic.simplify(s)) == 1
    except (TypeError, ValueError):
        return False


def split_snapshot_window(state: SDFGState) -> Optional[subsets.Range]:
    """The array window a pure anti-dependence snapshot copy state covers (one
    ``AccessNode(arr) -> AccessNode(arr_split_snap)`` edge, nothing else), or ``None`` if
    ``state`` isn't one. The window need not span the whole array -- callers must check
    containment of each read separately (:func:`snapshot_reads_in_window`) -- but the index
    mapping must be identity (equal subsets, unit stride) or a redirected read would move.
    """
    ns = list(state.nodes())
    if len(ns) != 2 or not all(isinstance(n, nodes.AccessNode) for n in ns):
        return None
    edges = list(state.edges())
    if len(edges) != 1:
        return None
    e = edges[0]
    if not (isinstance(e.src, nodes.AccessNode) and isinstance(e.dst, nodes.AccessNode)):
        return None
    if e.dst.data != f'{e.src.data}{_SPLIT_SNAP_SUFFIX}':
        return None
    src_desc = state.sdfg.arrays.get(e.src.data)
    dst_desc = state.sdfg.arrays.get(e.dst.data)
    if src_desc is None or dst_desc is None or e.data is None:
        return None
    if len(src_desc.shape) != len(dst_desc.shape):
        return None
    if any(symbolic.simplify(a - b) != 0 for a, b in zip(src_desc.shape, dst_desc.shape)):
        return None
    src_sub = e.data.get_src_subset(e, state)
    dst_sub = e.data.get_dst_subset(e, state)
    if src_sub is None or (dst_sub is not None and dst_sub != src_sub):
        return None
    if any(symbolic.simplify(step - 1) != 0 for (_lo, _hi, step) in src_sub.ndrange()):
        return None
    return src_sub


def is_split_snapshot_state(state: SDFGState) -> bool:
    """``state`` is a snapshot copy the wavefront can absorb."""
    return split_snapshot_window(state) is not None


def extract_two_level_nest(outer: LoopRegion) -> Optional[LoopRegion]:
    """The single inner :class:`LoopRegion` perfectly nested in ``outer`` (its
    body may itself be imperfect); ``None`` if ``outer`` holds anything else with
    data alongside the one inner loop. A pure ``arr_split_snap = arr`` snapshot
    state is tolerated -- :func:`commit_split_snapshots` folds it away before the
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


def live_reader(state: SDFGState, name: str) -> nodes.AccessNode:
    """A read-only ``AccessNode(name)`` in ``state`` (``in_degree == 0``), reusing
    one if present so redirected reads coalesce on the existing input node."""
    for n in state.data_nodes():
        if n.data == name and state.in_degree(n) == 0:
            return n
    return state.add_access(name)


#: A single snapshot read to redirect:
#: ``(state, snap_node, edge, read_index, live_array, copied_window)``.
SnapRead = Tuple[SDFGState, nodes.AccessNode, object, List[object], str, subsets.Range]
#: A planned absorb: ``(snap_src map, reads to redirect, copy states to drop)``.
SnapPlan = Tuple[Dict[str, str], List[SnapRead], List[SDFGState]]


def plan_split_snapshots(outer: LoopRegion, inner: LoopRegion, sdfg: SDFG) -> Optional[SnapPlan]:
    """Gather the ``arr_split_snap = arr`` snapshots ``BreakAntiDependence`` left beside
    ``inner`` and the inner-body reads that would redirect onto the live array,
    WITHOUT mutating. The plan is only committed (:func:`commit_split_snapshots`)
    once a legal skew is found, so a refused skew leaves the snapshot -- and the
    inner-axis parallelism it enables -- untouched.

    Returns ``({}, [], [])`` when there is no snapshot (proceed unchanged); a
    ``(snap_src, snap_reads, copy_states)`` plan; or ``None`` to refuse the skew.
    Refuses if a snapshot array is used anywhere other than its copy-state write
    and the inner-body reads -- an external reader would be left consuming a dead
    transient once the copy is dropped -- or if any inner read index is not a
    single point."""
    copy_states: List[SDFGState] = []
    snap_src: Dict[str, str] = {}  # snapshot array -> live source array
    snap_win: Dict[str, subsets.Range] = {}  # snapshot array -> window the copy covers
    for b in outer.nodes():
        if not isinstance(b, SDFGState) or b is inner:
            continue
        window = split_snapshot_window(b)
        if window is None:
            continue
        copy_states.append(b)
        e = list(b.edges())[0]
        snap_src[e.dst.data] = e.src.data
        snap_win[e.dst.data] = window
    if not copy_states:
        return {}, [], []
    snap_names = dict.fromkeys(snap_src)

    iters = (outer.loop_variable, inner.loop_variable)
    inner_states = dict.fromkeys(inner.all_states())
    copy_set = dict.fromkeys(copy_states)
    snap_reads: List[SnapRead] = []
    for state in sdfg.all_states():
        for node in state.data_nodes():
            if node.data not in snap_names:
                continue
            # A snapshot array may ONLY be written in its copy state and read in the
            # inner body; any other use dangles once the copy is dropped.
            if state.in_degree(node) > 0 and state not in copy_set:
                return None
            if state.out_degree(node) > 0 and state not in inner_states:
                return None
            if state in inner_states:
                for e in state.out_edges(node):
                    if e.data is None or e.data.subset is None:
                        return None
                    ridx = point_index(e.data.subset, iters)
                    if ridx is None:
                        return None
                    snap_reads.append((state, node, e, ridx, snap_src[node.data], snap_win[node.data]))
    return snap_src, snap_reads, copy_states


def snapshot_reads_forward(snap_reads: List[SnapRead], carrier: Tuple[str, 'WriteMap', List['Dependence']], u: str,
                           v: str) -> bool:
    """Every snapshot read must be a FORWARD (anti) dependence in ITERATION space:
    the writer of the cell it reads runs strictly later, so its value is the
    not-yet-overwritten old element the snapshot captured -- which the diagonal
    schedule preserves. Computed by inverting the carrier's write map (the same
    machinery :func:`collect_carrier` uses for its dependence distances), so a
    reflected / negative-stride write map is classified in iteration space, not by
    a raw array-index offset. Refuses if any snapshot read is a backward (flow)
    dependence, sits on a non-carrier array, or has an undecidable distance."""
    if not snap_reads:
        return True
    arr, wmap, _deps = carrier
    if wmap is None:
        return False  # a reduced (one-axis) carrier does not name the writing iteration
    for (_st, _node, _e, ridx, src_name, _win) in snap_reads:
        if src_name != arr or len(ridx) != 2:
            return False  # snapshot not on the 2-D carrier -> cannot reason
        u_r, v_r = wmap.invert(ridx[0], ridx[1])
        du = symbolic.simplify(u_r - sym(u))
        dv = symbolic.simplify(v_r - sym(v))
        if du == 0 and dv == 0:
            continue  # reads the very cell being written (old value)
        if dependence_kind(du, dv) != 'anti':
            return False  # backward (flow) or undecidable -> unsafe to redirect
    return True


def snapshot_reads_in_window(snap_reads: List[SnapRead], u: str, v: str, domain: List[object]) -> bool:
    """Every redirected read must index a cell the snapshot window actually covers; refuses
    whatever ISL cannot decide."""
    dims = [u, v]
    iters = (u, v)
    for (_st, _node, _e, ridx, _src, window) in snap_reads:
        rng = window.ndrange()
        if len(rng) != len(ridx):
            return False
        for idx, (lo, hi, _step) in zip(ridx, rng):
            below = symbolic.simplify(canonical_iterators(lo, iters) - idx - 1)
            above = symbolic.simplify(idx - canonical_iterators(hi, iters) - 1)
            for outside in (below, above):
                cons = list(domain) + [outside]
                try:
                    if not poly.is_domain_empty(dims, params_of(cons, dims), cons):
                        return False
                except ValueError:
                    return False
    return True


def commit_split_snapshots(snap_reads: List[SnapRead], copy_states: List[SDFGState]) -> None:
    """Rewire the planned snapshot reads onto the live array and drop the copies.
    Called only after a legal skew is confirmed. Structural cleanup then removes
    the emptied copy states and eliminates the dead ``arr_split_snap`` arrays."""
    for (state, _snap_node, e, _ridx, src_name, _win) in snap_reads:
        reader = live_reader(state, src_name)
        redirected = copy.deepcopy(e.data)
        redirected.data = src_name
        state.add_edge(reader, None, e.dst, e.dst_conn, redirected)
        state.remove_edge(e)
    for (state, snap_node, _e, _ridx, _src, _win) in snap_reads:
        if snap_node in state.nodes() and state.degree(snap_node) == 0:
            state.remove_node(snap_node)
    for st in copy_states:
        for n in list(st.nodes()):
            st.remove_node(n)


def point_index(subset, iters: tuple[str, ...]) -> Optional[List[object]]:
    """The per-dimension index of a *point* subset (``start == end`` on every
    axis); ``None`` if any axis is a range.

    This is the single boundary at which memlet subsets enter the pass, so it is where the
    iterators named in ``iters`` are re-keyed (:func:`canonical_iterators`) -- everything
    downstream then works in one symbol spelling."""
    idx = []
    for (start, end, _step) in subset.ndrange():
        if start != end:
            return None
        idx.append(canonical_iterators(start, iters))
    return idx


def access_extent(subset, iters: tuple[str, ...]) -> Optional[List[Tuple[object, object]]]:
    """The per-dimension ``(start, end)`` of a subset, keyed on the nest's iterators.

    The range-tolerant form of :func:`point_index`: a whole-row update ``A[i, 1:N-1]`` is a point
    on the axis that carries the dependence and a fixed span on the one that does not, and only
    :func:`uniform_axes` can tell which is which -- that needs every access of the array at once,
    so the decision cannot be made here. ``None`` for a strided axis, whose two endpoints do not
    describe the elements it touches."""
    idx = []
    for (start, end, step) in subset.ndrange():
        if step != 1:
            return None
        idx.append((canonical_iterators(start, iters), canonical_iterators(end, iters)))
    return idx


def uniform_axes(extents: List[List[Tuple[object, object]]], iters: tuple[str, ...]) -> List[int]:
    """Axes that every access of one array spans identically and independently of the nest.

    Such an axis carries no dependence: any two iterations that touch a common cell agree on it
    by construction, so it can be dropped and the distance read off what is left. A row sweep
    ``A[i, 1:N-1] = f(A[i-1, 1:N-1], A[i, 1:N-1], A[i+1, 1:N-1])`` is the shape this exists for --
    every access spans the same columns, and the wavefront lives entirely in the row index.

    An axis is kept whenever that does not hold, including when the accesses disagree on it: a
    range there is a real span of distances, and :func:`reduced_points` is what decides whether it
    can be enumerated or the nest must be refused.
    """
    if not extents:
        return []
    width = len(extents[0])
    if any(len(e) != width for e in extents):
        return []
    nest = set(iters)
    drop: List[int] = []
    for axis in range(width):
        spans = [e[axis] for e in extents]
        first = spans[0]
        if first[0] == first[1]:
            continue  # a point on the first access -> not a span every access shares
        if any(lo != first[0] or hi != first[1] for lo, hi in spans):
            continue  # the accesses disagree -> a distance can live here
        if any(nest & {str(x) for x in bound.free_symbols} for bound in first):
            continue  # a range that moves with the nest carries its own dependence
        drop.append(axis)
    return drop


#: Widest constant-width range read expanded into its individual points. A stencil reads a
#: handful of neighbours; anything wider is a sweep whose distances are not a small fixed set.
MAX_RANGE_READ_WIDTH = 8


def reduced_points(extent: List[Tuple[object, object]], drop: List[int]) -> Optional[List[List[object]]]:
    """``extent`` with the wildcard axes removed, as the list of point indices it covers.

    A kept axis that is a range of CONSTANT width is the union of that many point accesses --
    memlet consolidation turns the three neighbour reads of a row sweep into the single
    ``A[i-1 : i+2, ...]``, and expanding it back is what recovers the three distances. A
    symbolic or over-wide range has no such finite reading, so it refuses (``None``) unless
    ``endpoints_only`` says the caller only needs the extremes.

    A wide range reaching here is a genuine sweep whose distances are not a small fixed set --
    the reduction WINDOW of a lifted WCR map does not, because :func:`scan_state_accesses` reads
    that map's in-scope point subsets instead of the widened memlet at its entry.
    """
    axes: List[List[object]] = []
    for axis, (lo, hi) in enumerate(extent):
        if axis in drop:
            continue
        if lo == hi:
            axes.append([lo])
            continue
        width = symbolic.simplify(hi - lo)
        if not width.is_number or width < 0 or width > MAX_RANGE_READ_WIDTH:
            return None
        axes.append([symbolic.simplify(lo + k) for k in range(int(width) + 1)])
    if not axes:
        return None
    points = [[]]
    for values in axes:
        points = [pt + [val] for pt in points for val in values]
    return points


def axis_write_map(write_idxs: List[List[object]], u: str, v: str) -> Optional[Tuple[object, object]]:
    """``(coeff, const)`` for a carrier reduced to ONE axis and written at ``coeff * v + const``,
    with ``coeff`` in ``{1, -1}`` and ``const`` free of both iterators; ``None`` otherwise.

    Only a write independent of the OUTER iterator is admitted. It is the shape that makes a
    reduced carrier interesting -- the row sweep writes row ``i`` on every time step -- and it is
    the one whose reader can still be named: the location does not say which ``u`` wrote it, but
    within one ``u`` the rows are written in increasing ``v``, so the last write before a read is
    a lexicographic predecessor (:func:`axis_distance`). A write independent of the INNER
    iterator instead has its last write at the inner loop's final iteration, a bound rather than
    a distance, so it gets no uniform delta and is refused here."""
    found = None
    for idx in write_idxs:
        if len(idx) != 1:
            return None
        cu, rest = split_var(idx[0], u)
        if cu != 0:
            return None
        cv, const = split_var(rest, v)
        if cv not in (1, -1):
            return None
        if found is None:
            found = (cv, const)
        elif found != (cv, const):
            return None
    return found


def axis_distance(idx: object, coeff, const, u: str, v: str) -> Optional[Tuple[object, object]]:
    """``(du, dv) = writer - current`` for a read of the reduced carrier at ``idx``.

    ``coeff`` is its own inverse, so the writing ``v`` is ``coeff * (idx - const)``. Which ``u``
    wrote it is not in the location, so it is decided by order: a row BELOW the current one was
    written earlier in this same sweep (``du = 0``), while the current row and any row above it
    are written at or after this iteration, so the value being read is the previous sweep's
    (``du = -1``). ``None`` when the offset's sign is not decidable -- an undecided sign would
    have to pick one of the two answers, and picking wrong mis-orders the schedule."""
    v_w = symbolic.simplify(coeff * (idx - const))
    d = symbolic.simplify(v_w - sym(v))
    if not d.is_number:
        return None
    zero, back = symbolic.pystr_to_symbolic('0'), symbolic.pystr_to_symbolic('-1')
    return (zero, d) if d < 0 else (back, d)


def read_guard(state: SDFGState, inner: LoopRegion, iters: tuple[str, ...]) -> List[object]:
    """The branch conditions ``state`` executes under, as constraints ``>= 0`` keyed on the
    nest's iterator symbols; empty when nothing guards it or the guard has no convex form.

    The walk stops at ``inner`` so only conditions written inside the nest are collected -- one
    further out can be phrased in symbols an interstate edge reassigns on the way in, and would
    then constrain the wrong values."""
    cons = poly.constraints_from_condition(cfg.collect_enclosing_conditions(state, stop=inner))
    if not cons:
        return []
    return [canonical_iterators(c, iters) for c in cons]


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
    skew is ever found for a symmetric stencil.

    ``guard`` holds the branch conditions the READ executes under, as expressions meant
    ``>= 0`` in the current iteration's coordinates. A guarded read carries a dependence only
    where its guard holds, so the constraints join the iteration domain in
    :func:`dep_dims_and_cons`. Only the read's guard is collected, never the write's: assuming
    the write always happens can add dependences that do not exist but can never drop one."""

    def __init__(self,
                 du,
                 dv,
                 nested: List[Tuple[str, object, object]],
                 kind: str = 'flow',
                 guard: Optional[List[object]] = None):
        self.du = symbolic.simplify(du)
        self.dv = symbolic.simplify(dv)
        self.nested = nested
        self.kind = kind
        self.guard = guard or []


def dependence_kind(du, dv) -> str:
    """Classify a distance ``(du, dv) = writer - current`` as ``'flow'`` or
    ``'anti'`` by the lexicographic sign of its first non-zero component (a
    positive first component means the writer runs after the current iteration,
    so the read sees the soon-to-be-overwritten old value -- an anti dependence).

    A *symbolic* leading component is classified ``'anti'`` when its forward
    (positive) sign is PROVABLE -- e.g. a read ``arr[i, j + K]`` with ``K`` a
    declared-positive symbol is a genuine forward anti-dependence. Classifying it
    as flow (the pre-fix blanket rule) is UNSOUND: the pass then finds the
    difference-diagonal skew ``tau = (1, -1)`` legal under the (wrong) flow
    constraint ``-K < 0`` and schedules the overwrite *before* the read, silently
    miscompiling. A distance whose forward sign is not provable -- an unannotated
    or a backward symbol (``arr[i, j - K]``) -- stays conservatively ``'flow'``:
    it is a genuine backward recurrence, or an unannotated offset the optimistic
    retry pins with a runtime positive-guard, so a flow treatment can only refuse
    or trap, never mis-order."""
    du_s, dv_s = symbolic.simplify(du), symbolic.simplify(dv)
    if du_s.is_number and dv_s.is_number:
        if du_s > 0 or (du_s == 0 and dv_s > 0):
            return 'anti'
        return 'flow'
    lead = du_s if du_s != 0 else dv_s  # lexicographically leading (first non-zero) component
    if lead.is_positive:
        return 'anti'
    return 'flow'


def map_scope_context(state: SDFGState, node: nodes.Node) -> Optional[List[Tuple[str, object, object]]]:
    """The map scopes enclosing ``node`` in ``state``, innermost first, as ``[(param, lo, hi), ...]``.

    A reduction lifted to a WCR map is the same shape as the sequential reduction loop it replaced
    (:func:`nested_loop_context`), one level further in: its parameter reaches the distances and its
    range bounds them. Reading the widened memlet at the map's entry instead would say the body
    touches the whole array and refuse every schedule. ``None`` when a range is not a unit-stride
    interval, which has no such reading.
    """
    ctx: List[Tuple[str, object, object]] = []
    scope = state.scope_dict()
    cur = scope[node]
    while cur is not None:
        if not isinstance(cur, nodes.MapEntry):
            return None
        for param, (lo, hi, step) in zip(cur.map.params, cur.map.range):
            if symbolic.simplify(step) != 1:
                return None
            ctx.append((param, lo, hi))
        cur = scope[cur]
    return ctx


def widened_beyond_reading(subset) -> bool:
    """``subset`` has an axis whose width is symbolic or wider than a stencil's neighbourhood.

    That is precisely what :func:`reduced_points` cannot enumerate, and precisely what memlet
    propagation produces at a map scope boundary when the body's window is symbolic.
    """
    for (lo, hi, step) in subset.ndrange():
        if symbolic.simplify(step) != 1:
            return True
        width = symbolic.simplify(hi - lo)
        if not width.is_number or width > MAX_RANGE_READ_WIDTH:
            return True
    return False


def descend_into_map_scopes(state: SDFGState, data_name: str) -> bool:
    """Whether ``data_name``'s accesses in ``state`` should be read at the map-scope leaves.

    Only when a READ was widened past reading by propagation at a map entry -- the reduction window
    of a lifted WCR map -- AND no write to the array is widened. All of an array's accesses have to
    be described at the SAME level: :func:`uniform_axes` drops an axis only when every access spans
    it identically, so pairing a leaf-level read against a range-level write hides the very axis a
    row sweep depends on dropping. Asked per ARRAY, not per AccessNode: one state routinely holds
    several nodes for the same array, and the widened write that pins the level can sit on a
    different node than the widened read.
    """
    descend = False
    for node in state.data_nodes():
        if node.data != data_name:
            continue
        for e in state.in_edges(node):
            if e.data is not None and e.data.subset is not None and widened_beyond_reading(e.data.subset):
                return False
        for e in state.out_edges(node):
            if e.data is None or e.data.subset is None:
                continue
            if widened_beyond_reading(e.data.subset) and isinstance(e.dst, nodes.MapEntry):
                descend = True
    return descend


def in_scope_accesses(state: SDFGState, edge, is_read: bool) -> Optional[List[Tuple[object, List[Tuple]]]]:
    """``[(subset, map_ctx), ...]`` for the innermost edges ``edge`` stands for.

    Normally the edge's own subset, which is what every previously-analysable shape uses: a
    consolidated neighbourhood read (``A[i-1 : i+2, 1:N-1]``) says more about distances than its
    per-element leaves do, because the column axis it spans uniformly is exactly what
    :func:`uniform_axes` drops.

    The exception is a subset propagation has WIDENED past reading -- ``A[0:N, 0:N]`` at the entry
    of a map reducing over a symbolic window, once the ``k`` reduction is a WCR map instead of a
    sequential loop. That says the body touches everything and refuses every schedule, while the
    memlet tree's leaves carry the per-iteration subsets it actually reads (``A[i, _k]``) with
    :func:`map_scope_context` supplying the ``_k`` range that bounds them.
    """
    other = edge.dst if is_read else edge.src
    if not isinstance(other, (nodes.MapEntry, nodes.MapExit)) or not widened_beyond_reading(edge.data.subset):
        return [(edge.data.subset, [])]

    out: List[Tuple[object, List[Tuple]]] = []
    for leaf in state.memlet_tree(edge).leaves():
        endpoint = leaf.dst if is_read else leaf.src
        if isinstance(endpoint, (nodes.MapEntry, nodes.MapExit)):
            return None  # a scope that is not a plain map nest -> refuse
        ctx = map_scope_context(state, endpoint)
        if ctx is None or leaf.data is None or leaf.data.subset is None:
            return None
        out.append((leaf.data.subset, ctx))
    return out or None


def scan_state_accesses(state: SDFGState, inner: LoopRegion, sdfg: SDFG, u: str, v: str, v_local: str,
                        snap_src: Dict[str, str], sibling_cons: List[object], writes: Dict[str, List[List[object]]],
                        reads: Dict[str, List[Tuple]]) -> bool:
    """Record ``state``'s point accesses to 2-D arrays into ``writes`` / ``reads``.

    ``False`` means refuse: a non-point subset, or an enclosing loop whose range is not a clean
    interval. Each read is stored with the constraints it executes under -- its sibling's slice
    of the iteration space plus its own branch guards.

    ``v_local`` is the iterator THIS body is written in. Sibling loops are normalized apart, so
    each carries a private ``_loop_it_N``; every index and guard taken from this body is renamed
    onto the shared ``v`` here, where the subsets enter, rather than by mutating the graph before
    the skew is known to be legal."""
    ctx = None
    rename = {sym(v_local): sym(v)} if v_local != v else {}
    guard = sibling_cons + [g.xreplace(rename) for g in read_guard(state, inner, (u, v_local))]
    iters: tuple[str, ...] = (u, v_local)
    for node in state.data_nodes():
        data_name = snap_src.get(node.data, node.data)
        desc = sdfg.arrays.get(data_name)
        if desc is None or len(desc.shape) != 2:
            continue
        if ctx is None:
            ctx = nested_loop_context(state, inner)
            if ctx is None:
                return False
            # An enclosing reduction loop's iterator reaches the distances too, so it needs
            # the same single spelling as ``u`` / ``v``.
            iters = (u, v, *(nm for (nm, _, _) in ctx))
        descend = descend_into_map_scopes(state, data_name)
        for e in state.in_edges(node):
            if e.data is None or e.data.subset is None:
                continue
            accesses = in_scope_accesses(state, e, is_read=False) if descend else [(e.data.subset, [])]
            if accesses is None:
                return False
            for subset, map_ctx in accesses:
                if map_ctx:
                    return False  # a write parameterized by a map index names no single writer
                idx = access_extent(subset, iters)
                if idx is None:
                    return False  # strided write -> refuse
                writes.setdefault(data_name, []).append([(lo.xreplace(rename), hi.xreplace(rename)) for lo, hi in idx])
        for e in state.out_edges(node):
            if e.data is None or e.data.subset is None:
                continue
            accesses = in_scope_accesses(state, e, is_read=True) if descend else [(e.data.subset, [])]
            if accesses is None:
                return False
            for subset, map_ctx in accesses:
                # A map index reaching the read is one more parametric dimension of the distance,
                # exactly like an enclosing reduction loop's iterator, so it joins ``iters`` and
                # travels with the read as part of its context.
                read_iters = (*iters, *(nm for (nm, _, _) in map_ctx))
                idx = access_extent(subset, read_iters)
                if idx is None:
                    return False  # strided read of a 2-D carrier -> refuse
                reads.setdefault(data_name, []).append(
                    ([(lo.xreplace(rename), hi.xreplace(rename)) for lo, hi in idx], [*ctx, *map_ctx], guard))
    return True


def collect_carrier(inners: List[Tuple[LoopRegion, str, List[object]]],
                    sdfg: SDFG,
                    u: str,
                    v: str,
                    snap_src: Optional[Dict[str, str]] = None) -> Optional[Tuple[str, WriteMap, List[Dependence]]]:
    """Find the unique carrier array (written *and* self-read with a non-zero
    distance) across ``inners``' bodies, its write map, and its dependences. ``None``
    if there is no clean single carrier (refuse).

    ``inners`` is normally the one inner loop, paired with an empty guard. A chain of
    adjacent-range SIBLING loops is passed as several entries, each paired with the constraints
    that hold inside it (``plan_guarded_fusion``'s ``guards``): the siblings partition one
    iteration space, and a read in one of them carries a dependence only over that sibling's
    slice of it. Analysing them together is what lets a nest whose two halves are written as
    separate loops -- polybench ``lu`` and ``ludcmp`` -- be seen as the single 2-D space it is.

    ``snap_src`` maps each not-yet-committed ``arr_split_snap`` snapshot array to
    its live source; a read of a snapshot array is attributed to the live array so
    the carrier's dependence set is COMPLETE before the snapshot is folded away
    (the actual redirect is deferred until a legal skew is confirmed). Without it,
    the snapshotted read would be invisible and the skew decided on a partial
    dependence set."""
    snap_src = snap_src or {}
    writes: Dict[str, List[List[Tuple[object, object]]]] = {}
    reads: Dict[str, List[Tuple[List[Tuple[object, object]], List[Tuple[str, object, object]], List[object]]]] = {}
    for inner, v_local, sibling_guard in inners:
        sibling_cons = [canonical_iterators(g, (u, v)) for g in sibling_guard]
        for state in inner.all_states():
            if not scan_state_accesses(state, inner, sdfg, u, v, v_local, snap_src, sibling_cons, writes, reads):
                return None

    carriers: List[Tuple[str, Optional[WriteMap], List[Dependence]]] = []
    for arr, wextents in writes.items():
        rextents = reads.get(arr, [])
        drop = uniform_axes(wextents + [r[0] for r in rextents], (u, v))
        wsubs: List[List[object]] = []
        rsubs: List[Tuple[List[object], object, List[object]]] = []
        for extent in wextents:
            points = reduced_points(extent, drop)
            if points is None:
                return None  # a kept axis is a range with no finite reading -> refuse
            wsubs.extend(points)
        for (extent, ctx, guard) in rextents:
            points = reduced_points(extent, drop)
            if points is None:
                return None
            rsubs.extend((pt, ctx, guard) for pt in points)
        deps = carrier_dependences(wsubs, rsubs, u, v)
        if deps is None:
            return None
        wmap, dep_list = deps
        if dep_list:
            carriers.append((arr, wmap, dep_list))
    if len(carriers) != 1:
        return None  # zero or several carriers -> refuse
    return carriers[0]


def carrier_dependences(wsubs: List[List[object]], rsubs: List[Tuple[List[object], object, List[object]]], u: str,
                        v: str) -> Optional[Tuple[Optional[WriteMap], List[Dependence]]]:
    """``(write map, dependences)`` for one array, or ``None`` to refuse the whole nest.

    Two carrier shapes, by how many axes survived :func:`uniform_axes`:

    * TWO -- the write is a unimodular 2-D map, so a location names the single iteration that
      wrote it and every distance is exact.
    * ONE -- the write is independent of the outer iterator, so a location is written once per
      outer iteration and names no single writer. The distances come from program order instead
      (:func:`axis_distance`), and the repeated writes are themselves an output dependence: the
      previous sweep's write to the same location must stay ordered before this one, which is
      what stops the outer axis from being handed out in parallel.
    """
    if all(len(idx) == 2 for idx in wsubs):
        wmap = consistent_write_map(wsubs, u, v)
        if wmap is None:
            return None  # written by a non-affine or non-unimodular map -> refuse
        deps: List[Dependence] = []
        for (idx, ctx, guard) in rsubs:
            if len(idx) != 2:
                return None
            u_r, v_r = wmap.invert(idx[0], idx[1])
            du = symbolic.simplify(u_r - sym(u))
            dv = symbolic.simplify(v_r - sym(v))
            if du == 0 and dv == 0:
                continue  # in-place self-read, not a dependence
            deps.append(Dependence(du, dv, ctx, dependence_kind(du, dv), guard))
        return wmap, deps

    amap = axis_write_map(wsubs, u, v)
    if amap is None:
        return None
    coeff, const = amap
    deps = []
    for (idx, ctx, guard) in rsubs:
        if len(idx) != 1:
            return None
        d = axis_distance(idx[0], coeff, const, u, v)
        if d is None:
            return None  # undecidable offset sign -> refuse
        deps.append(Dependence(d[0], d[1], ctx, dependence_kind(d[0], d[1]), guard))
    if deps:
        # Output dependence: the same location is rewritten every outer iteration, so consecutive
        # sweeps must not land on one diagonal. Unguarded on purpose -- assuming a write always
        # happens can only add order, never drop it.
        deps.append(Dependence(symbolic.pystr_to_symbolic('-1'), symbolic.pystr_to_symbolic('0'), [], 'flow'))
    return None, deps


def consistent_write_map(write_subs: List[List[object]], u: str, v: str) -> Optional[WriteMap]:
    """A single :class:`WriteMap` agreeing with *every* write subset, else ``None``."""
    wmap = None
    for idx in write_subs:
        wm = parse_write_map(idx[0], idx[1], u, v)
        if wm is None:
            return None
        if wmap is None:
            wmap = wm
        elif (wm.m, wm.c) != (wmap.m, wmap.c):
            return None
    return wmap


def domain_constraints(u: str, v: str, ub: Tuple[object, object], vb: Tuple[object, object]) -> List[object]:
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
    cons += list(dep.guard)
    cons += list(assume)
    return dims, cons


def params_of(cons: List[object], dims: List[str]) -> List[str]:
    names: Dict[str, None] = {}
    for c in cons:
        names.update(dict.fromkeys(s.name for s in symbolic.simplify(c).free_symbols))
    return sorted(n for n in names if n not in dims)


def outer_axis_parallel(deps: List[Dependence], u: str, v: str, domain: List[object]) -> bool:
    """The OUTER ``u`` loop is ALREADY a parallel map in the current loop order:
    no dependence crosses ``u`` anywhere in the domain (``du == 0``), so a plain
    ``LoopToMap`` lifts it and the skew must not clobber the nest.

    Strictly stronger than ``schedule_legal((0, 1), ...)``, which only says the
    ``v`` *axis* carries every dependence. That also holds for a nest whose
    dependences still cross ``u`` (``du = -1``) -- there ``u`` is parallel only
    after an INTERCHANGE, and no pass in the pipeline performs one, so refusing on
    it trades a legal wavefront for a fully sequential nest. Origin-sensitive, and
    exactly the shape a triangular-inner-loop rebase produces: rebasing shears
    ``dv`` by ``du`` (``(-1, 0) -> (-1, -1)``), which flips ``tau = (0, 1)`` from
    illegal to legal without changing a thing about the nest's parallelism."""
    for dep in deps:
        dims, cons = dep_dims_and_cons(dep, u, v, domain, [])
        # Does the domain hold a point with du >= 1, or one with du <= -1?
        for crossing in (dep.du - 1, symbolic.simplify(-dep.du - 1)):
            probe = cons + [crossing]
            try:
                if not poly.is_domain_empty(dims, params_of(probe, dims), probe):
                    return False
            except ValueError:
                return False  # non-affine / unmapped -> not provably parallel
    return True


def schedule_legal(tau: Tuple[int, int], deps: List[Dependence], u: str, v: str, domain: List[object],
                   assume: List[object]) -> bool:
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
            return False  # non-affine / unmapped -> cannot prove -> illegal
        if not empty:
            return False
    return True


def tile_signs(d: int) -> Tuple[int, ...]:
    """Tile-index distances a single-axis element distance ``d`` can induce once
    ``|d|`` is known not to exceed the tile extent: the two iterations either sit
    in the same tile or straddle exactly one boundary, in the direction of ``d``."""
    if d == 0:
        return (0, )
    return (0, 1) if d > 0 else (-1, 0)


def tiling_legal(deps: List[Dependence], tau: Tuple[int, int], bi: int, bj: int) -> bool:
    """``tau`` still orders every dependence strictly once the nest is tiled ``bi x bj``.

    Two conditions, both necessary.

    1. Every distance component must be a known integer no larger in magnitude than
       its tile extent. Only then is the tile-index distance of a dependent pair the
       CLAMPED SIGN of the element distance, i.e. one of :func:`tile_signs`. A
       parametric distance -- a read under an enclosing reduction loop, as in
       polybench ``nussinov`` -- carries no such bound and counts as exceeding.
    2. For every tile-index distance the pair can actually take, ``tau`` must impose
       the same strict order it was proved to impose element-wise. ``(0, 0)`` is
       exempt: both iterations land in one tile, which runs its points in the
       original sequential ``(u, v)`` order.

    Condition 2 does NOT follow from condition 1. The Gauss-Seidel skew
    ``tau = (2, 1)`` with the flow distance ``(-1, +1)`` is legal element-wise
    (``tau . delta = -1 < 0``), yet the pair can straddle the ``v`` boundary alone,
    giving the tile distance ``(0, +1)`` and ``tau . (0, 1) = +1 > 0`` -- the
    producing tile would run one diagonal AFTER the consuming one, a miscompile.
    Such a nest keeps the untiled lowering."""
    a, b = tau
    for dep in deps:
        du, dv = integer_value(dep.du), integer_value(dep.dv)
        if du is None or dv is None:
            return False
        if abs(du) > bi or abs(dv) > bj:
            return False
        for di in tile_signs(du):
            for dj in tile_signs(dv):
                if di == 0 and dj == 0:
                    continue  # same tile -> original sequential order
                dot = a * di + b * dj
                if dot >= 0 if dep.kind == 'flow' else dot <= 0:
                    return False
    return True


def domain_bbox(u: str, v: str, params: List[str], domain: List[object]) -> Optional[List[object]]:
    """``[u_lo, u_hi, v_lo, v_hi]`` -- the exact integer bounding box of the 2-D
    iteration domain -- or ``None`` when a side is not a single readable bound.

    The tile grid is anchored at that box corner and stays RECTANGULAR even for a
    triangular domain: tiles below the diagonal simply run empty, and the clip back
    to the real bounds lives in the intra-tile loop bounds (``max(j0, i)``). The box
    is an ISL projection, so a ``u`` value whose ``v`` range is empty is already
    outside it and costs no tile."""
    s, nmap = poly.make_set((u, v), params, domain)
    inv = {safe: orig for orig, safe in nmap.items()}
    box: List[object] = []
    for keep in (0, 1):
        proj = s.project_out(poly.isl.dim_type.set, 1 - keep, 1).coalesce()
        for pw in (proj.dim_min(0), proj.dim_max(0)):
            box.append(poly.pwaff_bound(pw, inv))
    if any(o is None for o in box):
        return None
    return box


class TilePlan:
    """The skewed TILE-index bounds plus the grid origin and extents
    :meth:`WavefrontSkew._rewrite_tiled` emits from."""

    def __init__(self, bounds, u_lo, v_lo, n_i, n_j, bi: int, bj: int) -> None:
        self.bounds = bounds
        self.u_lo = u_lo
        self.v_lo = v_lo
        self.n_i = n_i
        self.n_j = n_j
        self.bi = bi
        self.bj = bj


def offset_symbols(deps: List[Dependence], dims: List[str]) -> List[object]:
    """Distinct parameter symbols appearing in any distance component."""
    nested_names = dict.fromkeys(nm for d in deps for (nm, _, _) in d.nested)
    syms = {}
    for dep in deps:
        for comp in (dep.du, dep.dv):
            for s in sorted(symbolic.simplify(comp).free_symbols, key=lambda s: s.name):
                if s.name not in dims and s.name not in nested_names:
                    syms[s.name] = s
    # sorted by name: the returned order reaches the ISL constraint text. ISL emptiness is order-independent
    # today, so this is inert -- but it makes that provable rather than incidental.
    return [syms[k] for k in sorted(syms)]


@properties.make_properties
@xf.explicit_cf_compatible
class WavefrontSkew(ppl.Pass):
    """Skew genuine 2-D wavefront nests so the inner loop lifts to a parallel Map.

    Backed by an exact ISL legality + bound projection; refuses anything that is
    not a genuine wavefront (already-parallel axis, non-affine carrier, several
    carriers, non-unit strides).

    The lowering is a skewed TILING (:meth:`_rewrite_tiled`) wherever
    :func:`tiling_legal` holds AND the target asks for blocking, and the
    element-granularity diagonal (:meth:`_rewrite`) otherwise. Only ``target='cpu'``
    asks: blocking is locality tuning for a cache, and a GPU is better served by the
    diagonal's wider parallelism."""

    CATEGORY: str = 'Canonicalization'

    target = properties.Property(dtype=str,
                                 default='cpu',
                                 choices=['cpu', 'gpu'],
                                 desc="Target policy: 'cpu' blocks the wavefront into skewed tiles wherever that "
                                 "is legal; 'gpu' always takes the element-granularity diagonal.")

    tile_i = properties.Property(dtype=int,
                                 default=DEFAULT_TILE_SIZE,
                                 desc='Skewed-tile extent on the outer (u) axis. A dependence reaching further than '
                                 'this on that axis falls back to the untiled lowering. CPU only.')
    tile_j = properties.Property(dtype=int,
                                 default=DEFAULT_TILE_SIZE,
                                 desc='Skewed-tile extent on the inner (v) axis; the innermost emitted loop runs one '
                                 'tile row of it at unit stride. CPU only.')

    gpu_tile_i = properties.Property(dtype=int,
                                     default=DEFAULT_GPU_TILE_SIZE,
                                     desc='Skewed-tile extent on the outer (u) axis for the GPU lowering. Separate '
                                     'from tile_i because the two targets are sized against different things: a CPU '
                                     'tile is sized to a cache, a GPU tile trades grid width (tiles per diagonal, '
                                     'which must cover the CUs) against block width (the intra-tile anti-diagonal, '
                                     'which must cover a wavefront).')
    gpu_tile_j = properties.Property(dtype=int,
                                     default=DEFAULT_GPU_TILE_SIZE,
                                     desc='Skewed-tile extent on the inner (v) axis for the GPU lowering.')

    def __init__(self, target: str = 'cpu'):
        super().__init__()
        self.target = target

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
                    continue  # stale snapshot: a prior skew removed this node
                if self._try_skew(cfg, sd):
                    skewed += 1
        return skewed or None

    def _try_skew(self, outer: LoopRegion, sdfg: SDFG) -> bool:
        if not unit_positive_stride(outer):
            return False
        ub = loop_bounds(outer)
        if ub is None:
            return False
        # One inner loop is the common shape. A chain of adjacent-range SIBLING loops -- lu's
        # ``j < i`` column update followed by its ``j >= i`` row update -- is the SAME 2-D space
        # spelled as two loops, so it is analysed jointly (each sibling's range becomes a guard on
        # its own reads) and fused into one guarded loop only once a legal skew is confirmed.
        inner = extract_two_level_nest(outer)
        fusion = None
        if inner is None:
            fusion = plan_guarded_fusion(outer)
            if fusion is None:
                return False
            inners = list(zip(fusion.loops, fusion.loop_vars, fusion.guards))
            v, vb = fusion.var, (fusion.lo, fusion.hi)
            snap_src, snap_reads, copy_states = {}, [], []
        else:
            if not unit_positive_stride(inner):
                return False
            vb = loop_bounds(inner)
            if vb is None:
                return False
            inners = [(inner, inner.loop_variable, [])]
            v = inner.loop_variable
        u = outer.loop_variable
        # The inner bound must not leak the inner var (malformed); the outer var
        # in the inner bound is fine -- that is the triangular case ISL handles.
        vsym = sym(v)
        if vsym in symbolic.simplify(vb[0]).free_symbols or vsym in symbolic.simplify(vb[1]).free_symbols:
            return False

        # Plan (do not yet apply) the absorb of any per-iteration anti-dependence
        # snapshot BreakAntiDependence left in the outer body -- the imperfect-nest
        # cause. Refuse now if a snapshot is used outside the nest; the reads are
        # only redirected onto the live array once a legal skew is confirmed, so a
        # refusal below leaves the snapshot (and the parallelism it enables) intact.
        if fusion is None:
            plan = plan_split_snapshots(outer, inner, sdfg)
            if plan is None:
                return False
            snap_src, snap_reads, copy_states = plan

        # Analyse against the live array (snapshot reads attributed virtually), so
        # the dependence set is complete without mutating.
        carrier = collect_carrier(inners, sdfg, u, v, snap_src=snap_src)
        if carrier is None:
            return False
        _arr, _wmap, deps = carrier

        # Redirecting a snapshot read is only value-preserving when it is a FORWARD
        # (anti) dependence in iteration space -- the writer runs on a strictly
        # later diagonal, so the diagonal schedule reproduces the snapshot's old
        # value. A backward read would change value, so refuse (no mutation).
        if not snapshot_reads_forward(snap_reads, carrier, u, v):
            return False

        domain = domain_constraints(u, v, ub, vb)
        dims = [u, v]

        # Needs the domain, so checked here, still before any mutation.
        if not snapshot_reads_in_window(snap_reads, u, v, domain):
            return False

        # --- Genuine-wavefront guard: refuse if an axis is already a parallel map
        # IN THE CURRENT LOOP ORDER, so a plain LoopToMap already reaches it. ---
        # Inner v parallel (map-in-inner / column-independent stencil) <=> every
        # dependence is carried by u, which is exactly tau=(1,0) legality.
        if schedule_legal((1, 0), deps, u, v, domain, []):
            return False
        # Outer u parallel (map-of-scans / row-independent stencil) <=> nothing
        # crosses u. NOT tau=(0,1) legality -- see :func:`outer_axis_parallel`.
        if outer_axis_parallel(deps, u, v, domain):
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

        # Prefer the tiled lowering; decided BEFORE any mutation so a refusal here
        # only picks the untiled path, never leaves the nest half-rewritten.
        tiles = self._plan_tiles(deps, u, v, domain, dims, tau)

        # Legal skew confirmed -- now (and only now) commit the snapshot absorb,
        # so every earlier refusal left the snapshot intact. Reads are redirected
        # onto the live array before the rewrite skews the inner body over them.
        commit_split_snapshots(snap_reads, copy_states)

        if fusion is not None:
            inner = commit_guarded_fusion(fusion, outer)

        if guard_syms:
            self._emit_positive_guard(outer, deps, guard_syms)

        if tiles is None:
            self._rewrite(outer, inner, sdfg, u, v, tau, bounds)
        else:
            self._rewrite_tiled(outer, inner, sdfg, u, ub, vb, tau, tiles)
        return True

    def _plan_tiles(self, deps: List[Dependence], u: str, v: str, domain: List[object], dims: List[str],
                    tau: Tuple[int, int]) -> Optional[TilePlan]:
        """The tile-index skew for a ``tile_i x tile_j`` blocking of this nest, or
        ``None`` to keep the element-granularity lowering.

        The tile-index domain is the SAME polyhedron under ``u -> bi*I``,
        ``v -> bj*J``, so ``poly.skew_bounds`` projects it with no changes: it is
        handed a rectangle over ``(I, J)`` and returns the diagonal ``T`` range plus
        the parametric tile-column ``P`` range at fixed ``T``."""
        # Both targets block, for different reasons and at different sizes (see the class
        # docstring): a CPU tile is cache blocking, a GPU tile is what turns one kernel launch per
        # element anti-diagonal into one per tile anti-diagonal.
        if self.target == 'cpu':
            bi, bj = int(self.tile_i), int(self.tile_j)
        elif self.target == 'gpu':
            bi, bj = int(self.gpu_tile_i), int(self.gpu_tile_j)
        else:
            return None
        if bi < 1 or bj < 1 or not tiling_legal(deps, tau, bi, bj):
            return None
        ti, tj = sym(_TILE_I_PROBE), sym(_TILE_J_PROBE)
        tdims = [_TILE_I_PROBE, _TILE_J_PROBE]
        # I in [0, NI - 1], J in [0, NJ - 1] over the OPAQUE counts (see the probe names).
        tile_domain = [ti, sym(_TILE_NI_PROBE) - 1 - ti, tj, sym(_TILE_NJ_PROBE) - 1 - tj]
        try:
            box = domain_bbox(u, v, params_of(domain, dims), domain)
            if box is None:
                return None
            bounds = poly.skew_bounds(tuple(tdims), params_of(tile_domain, tdims), tile_domain, tau,
                                      f'{_SKEW_T_PREFIX}probe', f'{_SKEW_P_PREFIX}probe')
        except ValueError:
            return None  # not renderable for ISL -> keep the untiled lowering
        if bounds is None:
            return None
        u_lo, u_hi, v_lo, v_hi = box
        n_i = symbolic.int_ceil(symbolic.simplify(u_hi - u_lo + 1), bi)
        n_j = symbolic.int_ceil(symbolic.simplify(v_hi - v_lo + 1), bj)
        return TilePlan(bounds, u_lo, v_lo, n_i, n_j, bi, bj)

    def _rewrite(self, outer: LoopRegion, inner: LoopRegion, sdfg: SDFG, u: str, v: str, tau: Tuple[int, int],
                 bounds) -> None:
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
        subs = {f'{_SKEW_T_PREFIX}probe': sym(t_var), f'{_SKEW_P_PREFIX}probe': sym(p_var)}

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

    def _rewrite_tiled(self, outer: LoopRegion, inner: LoopRegion, sdfg: SDFG, u: str, ub: Tuple[object, object],
                       vb: Tuple[object, object], tau: Tuple[int, int], plan: TilePlan) -> None:
        """Lower the wavefront as a skewed TILING -- four loops instead of two::

            for T in [t_lo .. t_hi]:              # tile diagonal, pinned sequential
              parallel for P in [p_lo .. p_hi]:   # tile column, lifted to a Map
                for u in [u0 + I*Bi .. min(u_hi, u0 + I*Bi + Bi - 1)]:
                  for v in [max(v_lo, v0 + J*Bj) .. min(v_hi, v0 + J*Bj + Bj - 1)]:
                      <original body>

        with ``(I, J)`` the tile indices read back from ``(T, P)`` through the same
        unimodular complement ``skew_bounds`` used (``I = a*(T - b*P), J = P`` when
        ``|a| == 1``; ``I = P, J = b*(T - a*P)`` when ``|b| == 1``). The triangular
        clip folds into the ``v`` lower bound exactly as the ISL projection does
        untiled. ``u`` and ``v`` keep their original names, so the body is reused
        verbatim -- no substitution, no memlet rewrite.

        Why this and not the element-granularity diagonal: the untiled form walks a
        stride-``N`` anti-diagonal and forks a parallel region per diagonal, which
        measured 0.17-0.18x of the plain sequential nest at N=768 on 4 threads. The
        tiled form gives the innermost loop unit stride and cuts the number of
        parallel regions by the tile area; the same shapes measure 2.04-2.16x.

        Bit-exactness is unchanged by the tiling. Each cell is still written exactly
        once; :func:`tiling_legal` proves every summand a cell reads is the final
        value of a cell from a strictly earlier tile diagonal (or from the same tile,
        where the original sequential ``(u, v)`` order is preserved verbatim); and no
        statement's own evaluation order is touched. So every read sees the very
        value the sequential nest gave it, in the same order -- no reassociation, no
        renormalisation, bit-for-bit."""
        a, b = tau
        v = inner.loop_variable
        nid = _next_id(sdfg)
        t_var = f"{_SKEW_T_PREFIX}{nid}"
        p_var = f"{_SKEW_P_PREFIX}{nid}"
        sdfg.add_symbol(t_var, dace.int64)
        sdfg.add_symbol(p_var, dace.int64)
        subs = {
            f'{_SKEW_T_PREFIX}probe': sym(t_var),
            f'{_SKEW_P_PREFIX}probe': sym(p_var),
            _TILE_NI_PROBE: plan.n_i,
            _TILE_NJ_PROBE: plan.n_j,
        }
        bounds = plan.bounds

        outer.loop_variable = t_var
        outer.init_statement = properties.CodeBlock(f"{t_var} = ({bound_expr(bounds.t_lo_terms, subs, 'max')})")
        outer.loop_condition = properties.CodeBlock(f"{t_var} <= ({bound_expr(bounds.t_hi_terms, subs, 'min')})")
        outer.update_statement = properties.CodeBlock(f"{t_var} = {t_var} + 1")
        # The tile diagonal carries every wavefront dependence by construction; pin it
        # so a downstream LoopToMap / LoopToReduce never races it into a parallel map.
        outer.pinned_sequential = True

        t_sym, p_sym = sym(t_var), sym(p_var)
        if abs(a) == 1:
            i_tile, j_tile = a * (t_sym - b * p_sym), p_sym
        else:
            i_tile, j_tile = p_sym, b * (t_sym - a * p_sym)
        i_lo = symbolic.simplify(plan.u_lo + plan.bi * i_tile)
        j_lo = symbolic.simplify(plan.v_lo + plan.bj * j_tile)

        p_loop = LoopRegion(f'{outer.label}_tile_diag', f"{p_var} <= ({bound_expr(bounds.p_hi_terms, subs, 'min')})",
                            p_var, f"{p_var} = ({bound_expr(bounds.p_lo_terms, subs, 'max')})",
                            f"{p_var} = {p_var} + 1")
        i_loop = LoopRegion(f'{outer.label}_tile_row',
                            f"{u} <= (min({symbolic.symstr(ub[1])}, {symbolic.symstr(i_lo + plan.bi - 1)}))", u,
                            f"{u} = ({symbolic.symstr(i_lo)})", f"{u} = {u} + 1")
        # The intra-tile loops carry the dependences the diagonal spreads apart, so
        # they stay sequential for the same reason the diagonal does.
        i_loop.pinned_sequential = True

        # Re-parent: ``inner`` becomes the innermost unit-stride ``v`` loop, wrapped by
        # the new row loop, wrapped by the tile-column loop that replaces it in ``outer``.
        in_edges = list(outer.in_edges(inner))
        out_edges = list(outer.out_edges(inner))
        was_start = outer.start_block is inner
        outer.remove_node(inner)
        i_loop.add_node(inner, is_start_block=True)
        p_loop.add_node(i_loop, is_start_block=True)
        outer.add_node(p_loop, is_start_block=was_start, ensure_unique_name=True)
        for e in in_edges:
            outer.add_edge(e.src, p_loop, e.data)
        for e in out_edges:
            outer.add_edge(p_loop, e.dst, e.data)

        inner.init_statement = properties.CodeBlock(f"{v} = (max({symbolic.symstr(vb[0])}, {symbolic.symstr(j_lo)}))")
        inner.loop_condition = properties.CodeBlock(
            f"{v} <= (min({symbolic.symstr(vb[1])}, {symbolic.symstr(j_lo + plan.bj - 1)}))")
        inner.update_statement = properties.CodeBlock(f"{v} = {v} + 1")
        inner.pinned_sequential = True

        # On a GPU the tile INTERIOR is the thread block, so it is skewed too (see
        # :meth:`_skew_within_tile`). Done before the tile-column lift because both steps run
        # LoopToMap, and it is far simpler to rewrite loops while they are still loops.
        if self.target == 'gpu':
            self._skew_within_tile(i_loop, inner, sdfg, u, v, tau, plan, i_lo, j_lo, ub, vb)

        self._convert_inner_to_map(outer, p_loop, sdfg)

    def _skew_within_tile(self, i_loop: LoopRegion, inner: LoopRegion, sdfg: SDFG, u: str, v: str, tau: Tuple[int, int],
                          plan: TilePlan, i_lo, j_lo, ub: Tuple[object, object], vb: Tuple[object, object]) -> None:
        """Turn a tile's two sequential interior loops into a diagonal over a PARALLEL Map.

        The tile interior is the same shape as the nest that contains it -- a 2-D nest whose
        dependences the same ``tau`` orders -- so this is :meth:`_rewrite` applied a second time,
        one level down, and nothing new has to be proven. Legality carries over for free: ISL
        decided ``tau`` over the WHOLE domain, and the tile interior is a subset of it, so a
        schedule with no violating pair in the whole domain has none in a part of it.

        What it buys is the reason the GPU lowering tiles at all. The tile-column Map is the grid,
        one block per tile; without this the block would run its tile on a single thread. With it,
        the block's threads walk the intra-tile anti-diagonal together, and that anti-diagonal is
        at most ``min(Bi, Bj)`` wide -- one wavefront at the default 64. The Map carries
        ``is_warp_tile``, which is a REQUEST, not a schedule: the device offload assigns every
        nested scope ``Sequential`` (correctly -- a kernel launch inside a kernel is not
        expressible) and ``PromoteWarpTiles`` reads the tag afterwards to give it
        ``GPU_ThreadBlock``.

        The clip is what makes this ISL's job rather than arithmetic. A boundary tile is partial,
        and a triangular domain clips ``v`` against ``u`` itself, so the anti-diagonal's extent is
        not ``min(Bi, Bj)`` at the edges. Both clips go in as plain affine constraints and the
        projection returns exact bounds, which is why no guard is emitted inside the body.

        Falls back to leaving the interior sequential -- correct, just narrow -- whenever the
        projection is not renderable.
        """
        u_sym, v_sym = sym(u), sym(v)
        # Every ``min``/``max`` in the interior bounds is a CONJUNCTION of affine constraints, which
        # is the form ISL wants. Each entry is constrained non-negative.
        interior = [
            u_sym - i_lo,
            symbolic.simplify(ub[1] - u_sym),
            symbolic.simplify(i_lo + plan.bi - 1 - u_sym),
            symbolic.simplify(v_sym - vb[0]),
            v_sym - j_lo,
            symbolic.simplify(vb[1] - v_sym),
            symbolic.simplify(j_lo + plan.bj - 1 - v_sym),
        ]
        dims = (u, v)
        # The diagonal runs over the enclosing RECTANGLE, not the clipped region: its exact extent
        # is piecewise (interior tile / last row / last column / corner) and renders as no single
        # loop bound. A superset is free here because the parallel range at each diagonal is read
        # exactly, so a diagonal past the real edge runs an empty Map. It also makes the count a
        # compile-time constant, ``bi + bj - 1``.
        a, b = tau
        corner = a * i_lo + b * j_lo
        d_lo = corner + min(a * (plan.bi - 1), 0) + min(b * (plan.bj - 1), 0)
        d_hi = corner + max(a * (plan.bi - 1), 0) + max(b * (plan.bj - 1), 0)
        try:
            bounds = poly.skew_bounds(dims,
                                      params_of(interior, list(dims)),
                                      interior,
                                      tau,
                                      f'{_SKEW_T_PREFIX}probe',
                                      f'{_SKEW_P_PREFIX}probe',
                                      t_range=(symbolic.simplify(d_lo), symbolic.simplify(d_hi)))
        except ValueError:
            return  # not renderable for ISL -> the interior stays sequential
        if bounds is None:
            return
        self._emit_tile_interior(i_loop, inner, sdfg, u, v, tau, plan, i_lo, j_lo, bounds, (d_lo, d_hi))

    def _emit_tile_interior(self, i_loop: LoopRegion, inner: LoopRegion, sdfg: SDFG, u: str, v: str,
                            tau: Tuple[int, int], plan: TilePlan, i_lo, j_lo, bounds, d_range) -> None:
        """Diagonal over a CONSTANT-width parallel Map, with the real extent as a guard.

        This is :meth:`_rewrite` with one difference, and the difference is forced by the hardware.
        A thread block's width is fixed for the whole launch, but the intra-tile anti-diagonal
        RAMPS -- one point at the tile's corner, ``min(Bi, Bj)`` across its middle, back to one.
        Giving the Map the exact projected extent therefore emits a block dimension built from
        ``_skew_p_0`` and the intra-tile diagonal, both of which exist only INSIDE the kernel:

            dim3(-Max(64*_skew_p_0 + 1, ...) + Min(N - 1, ...) + 1, 1, 1)   // will not compile

        So the Map runs the full tile width, a compile-time constant, and the projected bounds
        become a predicate on the body. Lanes outside the current anti-diagonal are masked off,
        which is what a fixed block over a ramping wavefront costs and the only way to pay it.
        """
        a, b = tau
        nid = _next_id(sdfg)
        d_var = f'{_SKEW_T_PREFIX}{nid}'
        k_var = f'{_SKEW_P_PREFIX}{nid}'
        sdfg.add_symbol(d_var, dace.int64)
        sdfg.add_symbol(k_var, dace.int64)
        subs = {f'{_SKEW_T_PREFIX}probe': sym(d_var), f'{_SKEW_P_PREFIX}probe': sym(k_var)}
        p_lo = bound_expr(bounds.p_lo_terms, subs, 'max')
        p_hi = bound_expr(bounds.p_hi_terms, subs, 'min')

        i_loop.loop_variable = d_var
        i_loop.init_statement = properties.CodeBlock(f'{d_var} = ({symbolic.symstr(d_range[0])})')
        i_loop.loop_condition = properties.CodeBlock(f'{d_var} <= ({symbolic.symstr(d_range[1])})')
        i_loop.update_statement = properties.CodeBlock(f'{d_var} = {d_var} + 1')
        i_loop.pinned_sequential = True

        # ``k`` indexes the block's threads: 0 .. width-1, the tile extent on the parallel axis.
        width = plan.bj if abs(a) == 1 else plan.bi
        inner.loop_variable = k_var
        inner.init_statement = properties.CodeBlock(f'{k_var} = 0')
        inner.loop_condition = properties.CodeBlock(f'{k_var} <= {width - 1}')
        inner.update_statement = properties.CodeBlock(f'{k_var} = {k_var} + 1')

        # The projected axis is an absolute coordinate; a thread holds it at ``origin + k``.
        origin = j_lo if abs(a) == 1 else i_lo
        coord = symbolic.simplify(origin + sym(k_var))
        if abs(a) == 1:
            inner.replace_dict({u: symbolic.symstr(a * (sym(d_var) - b * coord)), v: symbolic.symstr(coord)})
        else:
            inner.replace_dict({u: symbolic.symstr(coord), v: symbolic.symstr(b * (sym(d_var) - a * coord))})

        here = symbolic.symstr(coord)
        self._guard_body(inner, sdfg, f'({here}) >= ({p_lo}) and ({here}) <= ({p_hi})')
        self._convert_inner_to_map(i_loop, inner, sdfg)
        for node, _ in i_loop.all_nodes_recursive():
            if isinstance(node, nodes.MapEntry):
                node.map.is_warp_tile = True

    def _guard_body(self, loop: LoopRegion, sdfg: SDFG, condition: str) -> None:
        """Move ``loop``'s body under a single-branch ConditionalBlock testing ``condition``."""
        branch = ControlFlowRegion(f'{loop.label}_active', sdfg=sdfg)
        # Snapshot before moving: ``add_node`` re-homes each block's ``parent_graph``, so the edge
        # list has to be read off the loop while it still owns them.
        start, blocks, edges = loop.start_block, list(loop.nodes()), list(loop.edges())
        for blk in blocks:
            branch.add_node(blk, is_start_block=blk is start)
        for e in edges:
            branch.add_edge(e.src, e.dst, e.data)
        for blk in blocks:
            loop.remove_node(blk)
        guard = ConditionalBlock(f'{loop.label}_lane', sdfg=sdfg)
        guard.add_branch(condition, branch)
        loop.add_node(guard, is_start_block=True)

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

    def _emit_positive_guard(self, outer: LoopRegion, deps: List[Dependence], guard_syms: List[object]) -> None:
        """Plant a ``std::abort`` before ``outer`` that fires if any distance
        component carrying an unannotated symbol is positive at runtime (soundness
        needs it ``<= 0``). Mirrors ``BreakAntiDependence``'s positive guard."""
        gset = dict.fromkeys(s.name for s in guard_syms)
        exprs = []
        seen: Dict = {}
        for dep in deps:
            for comp in (dep.du, dep.dv):
                cs = symbolic.simplify(comp)
                names = dict.fromkeys(s.name for s in cs.free_symbols)
                if any(n in gset for n in names) and not cs.is_number:
                    key = str(cs)
                    if key not in seen:
                        seen[key] = None
                        exprs.append(cs)
        if not exprs:
            return
        parts = ' || '.join(f'(({symbolic.symstr(e)}) > 0)' for e in exprs)
        # crc32, NOT hash(): ``hash()`` of a str is randomized per process by PYTHONHASHSEED, so the guard
        # state and tasklet got a different label on every run of the SAME input -- different emitted C
        # symbols and a different build hash. crc32 is a stable digest of the same text.
        tag = zlib.crc32(parts.encode()) & 0xfffffff
        code = f'if ({parts}) {{ std::abort(); }}'
        pre = outer.parent_graph.add_state_before(outer, label=f'_skew_guard_{tag:x}')
        guard = pre.add_tasklet(name=f'_skew_guard_{tag:x}',
                                inputs={},
                                outputs={},
                                code=code,
                                language=dace.dtypes.Language.CPP)
        guard.side_effects = True


def bound_expr(terms: List[object], subs: Dict[str, object], fn: str) -> str:
    """Render bound terms into a loop-bound string: a single term verbatim, or
    ``max(...)`` / ``min(...)`` (``fn``) of several. ``subs`` resolves the probe
    names -- the real ``t`` / ``p`` symbols, and for the tiled lowering the two tile
    counts the tile-index polyhedron carried as opaque parameters."""
    rendered = [symbolic.symstr(substitute_by_name(t, subs)) for t in terms]
    if len(rendered) == 1:
        return rendered[0]
    return f"{fn}(" + ", ".join(rendered) + ")"


def substitute_by_name(expr, subs: Dict[str, object]):
    """Substitute symbols in ``expr`` by NAME, so a probe symbol is matched whatever
    assumptions its object carries."""
    e = symbolic.simplify(expr)
    mp = {}
    for s in e.free_symbols:
        if s.name in subs:
            mp[s] = subs[s.name]
    return e.subs(mp)


def _next_id(sdfg: SDFG) -> int:
    """Lowest ``<N>`` no existing ``_skew_(t|p)_<N>`` symbol uses."""
    used: Dict[int, None] = {}
    for sd in sdfg.all_sdfgs_recursive():
        for s in list(sd.symbols.keys()):
            for pre in (_SKEW_T_PREFIX, _SKEW_P_PREFIX):
                if s.startswith(pre) and s[len(pre):].isdigit():
                    used[int(s[len(pre):])] = None
        for cfg in sd.all_control_flow_regions():
            if isinstance(cfg, LoopRegion) and cfg.loop_variable:
                for pre in (_SKEW_T_PREFIX, _SKEW_P_PREFIX):
                    lv = cfg.loop_variable
                    if lv.startswith(pre) and lv[len(pre):].isdigit():
                        used[int(lv[len(pre):])] = None
    n = 0
    while n in used:
        n += 1
    return n


__all__ = ['WavefrontSkew']
