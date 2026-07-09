# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""K-dim tile access classifier for the vectorization tile pipeline.

:func:`classify_tile_access`: given a memlet subset, the K tile iter-vars, optionally the inner
SDFG, how does access vary as the K-dim tile iterates its (W_0, ..., W_{K-1}) lanes?

Target-isa-agnostic, no SDFG mutation. Sole output = :class:`TileAccess` record; emitter consumes
it to pick a tile lib node (``TileLoad`` / ``TileLoadStrided`` / ``TileLoad`` (``gather_dims``) /
``TileStore`` / ``TileStore`` (``gather_dims``) / ``TileBroadcast``). Arch intrinsics live in
lib-node expansions.

Per-dim classification
----------------------

Each subset dim tagged independently as a :class:`PerDimKind`:

* **BROADCAST** -- no tile iter-var in dim expr (loop-invariant; one element splats to all lanes).
* **STRUCTURED_1** -- one tile iter-var as direct top-level symbol, identity coeff
  (``iter_var + c``): lane ``l`` reads ``addr + l * elem_size``.
* **AFFINE** -- iter-var(s) as direct symbols, non-unit coeff (``2*i + 3``) or combined (``i + j``).
  Emitter picks strided-load / loop-of-loads / GATHER fallback per arch.
* **GATHER** -- tile iter-var inside a :class:`Subscript` (``arr[idx[i]]``). Data-dependent; always
  lowers to TileLoad (gather_dims).

Whole-subset composition (DIAGONAL / TRANSPOSE flags)
-----------------------------------------------------

Whole-subset kind = strongest per-dim kind, order ``GATHER > AFFINE > STRUCTURED > BROADCAST``.
Two composition flags:

* **diagonal** -- a STRUCTURED_1 iter-var is direct symbol in multiple dims (``arr[i, i]``).
* **transpose** -- STRUCTURED_1 iter-vars in non-canonical permutation of ``spec.iter_vars``.

Both informational; per locked design both fold into ``TileLoad`` (``gather_dims``) today
(diagonal → 1-D index tile; transpose → permuted index tile or transpose intrinsic).

dim_strides convention
----------------------

Non-GATHER dims expose the per-dim coefficient as ``dim_strides`` for load/store lib nodes:

* ``0`` -- BROADCAST (dim invariant per lane; splat).
* ``1`` -- STRUCTURED_1 (lane ``l`` reads index ``l`` more).
* ``c > 1`` -- AFFINE with coefficient ``c``.

Mixed-rank broadcasts (K=0->K=2 full splat; K=1->K=2 one-dim broadcast; ...) drop out
automatically: BROADCAST dims get ``dim_stride = 0``, expansion replicates across lanes.

Gather-side composition rule
----------------------------

ANY GATHER dim → whole-subset kind GATHER. Non-GATHER dims fold into the gather's index expr inline
(structured dims contribute affine sub-exprs; no separate index tile).
"""
import enum
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import sympy

from dace import symbolic
from dace.sdfg import SDFG, nodes
from dace.subsets import Range


class PerDimKind(enum.Enum):
    """Per-dim classification of a memlet subset for tile lowering.

    See TILIFICATION_TRANSFORMATION_DESIGN.md section 4 for the lattice.
    """

    #: No tile iter-var -- loop-invariant. All W lanes share one source element (codegen splats).
    CONSTANT = "constant"

    #: ``iter_var + c`` -- stride 1, constant offset. Lane ``l`` reads index ``l`` more.
    LINEAR = "linear"

    #: ``int_floor(c * iter_var + c0, k)`` / ``int_ceil(...)`` with ``1 < k < W`` -- group-broadcast
    #: within the dim, factor ``k`` (codegen loads ``W/k``, replicates each ``k`` times).
    REPLICATE = "replicate"

    #: ``s * iter_var + c``, ``s`` outer-scope constant, ``s >= 1``. Strided load. Multiple iter-vars
    #: sharing a dim → GATHER.
    AFFINE = "affine"

    #: ``(c * iter_var + c0) % N``, ``N`` outer-scope constant. Cyclic wrap. When ``N | c * W_p``
    #: (tile-aligned), reduces to LINEAR with a per-tile constant offset.
    MODULAR = "modular"

    #: Data-dependent / unsupported. Tile-dependent symbols force GATHER (section 4.2 join rule).
    GATHER = "gather"

    # Backwards-compat aliases -- removed once the rename propagates through every consumer.
    BROADCAST = "constant"
    STRUCTURED_1 = "linear"


class TileAccessKind(enum.Enum):
    """Whole-subset kind: strongest per-dim kind wins, order ``GATHER > AFFINE > STRUCTURED >
    BROADCAST``. DIAGONAL / TRANSPOSE compositions tagged on :class:`TileAccess` for emitter info."""

    BROADCAST = "broadcast"
    STRUCTURED = "structured"
    AFFINE = "affine"
    GATHER = "gather"


@dataclass(frozen=True)
class TileAccess:
    """Result of :func:`classify_tile_access` (data only). Emitter reads it + surrounding scope
    (mask availability, target_isa) to pick a lib node and wire its properties."""

    #: Whole-subset kind (strongest per-dim kind).
    kind: TileAccessKind

    #: Per-dim classification, one entry per subset dim (in subset order).
    per_dim_kind: Tuple[PerDimKind, ...]

    #: Per-dim stride. ``0`` = broadcast; ``1`` = identity; ``c > 1`` = affine coeff. ``None`` for
    #: GATHER dims (stride lives in the gather-index expr). Same length as ``per_dim_kind``.
    dim_strides: Tuple[Optional[int], ...]

    #: STRUCTURED_1 / AFFINE dims: which tile iter-var drives the dim's per-lane stride. ``None`` for
    #: BROADCAST / GATHER dims. Same length as ``per_dim_kind``.
    dim_iter_var: Tuple[Optional[str], ...]

    #: GATHER dims: the in-scope :class:`AccessNode` of the index array, when resolvable. ``None``
    #: for non-GATHER dims OR when the index source isn't a direct array read. Same length as
    #: ``per_dim_kind``.
    gather_index_per_dim: Tuple[Optional[nodes.AccessNode], ...]

    #: STRUCTURED_1 / AFFINE dims: the constant offset (``c`` in ``iter_var + c``). ``None`` for
    #: BROADCAST / GATHER dims.
    dim_offset: Tuple[Optional[sympy.Expr], ...]

    #: Per-dim replicate factor (lanes-per-distinct-value within the dim); unifies BROADCAST /
    #: STRUCTURED_1 / REPLICATE. Same length as :attr:`per_dim_kind`:
    #:
    #: * ``None`` BROADCAST -- one element splat to dim's full W lanes (factor = W, but W invisible
    #:   to classifier → ``None`` = "all lanes" sentinel).
    #: * ``1`` STRUCTURED_1 -- each lane reads a distinct consecutive element, no replication.
    #: * ``k > 1`` REPLICATE (``int_floor`` / ``int_ceil`` of affine arg) -- ``k`` consecutive lanes
    #:   share each element; codegen loads ``W / k``, group-broadcasts each ``k`` times.
    #: * ``None`` AFFINE / GATHER -- N/A.
    replicate_factor_per_dim: Tuple[Optional[int], ...]

    #: Diagonal composition: tile iter-var name → tuple of subset dims it appears in directly. Only
    #: populated when one iter-var spans multiple dims. Empty dict when no diagonal.
    diagonal: Dict[str, Tuple[int, ...]] = field(default_factory=dict)

    #: Transpose composition: permutation of tile iter-vars across subset dims as
    #: ``(perm_0, ..., perm_{K-1})``, ``perm_d`` = iter-var INDEX in ``spec.iter_vars`` driving dim
    #: ``d``. ``None`` when the dim order is canonical or the subset isn't a pure permutation.
    transpose: Optional[Tuple[int, ...]] = None


# ----- internal helpers --------------------------------------------------


def _safe_sympify(expr) -> Optional[sympy.Expr]:
    """Best-effort sympify. ``None`` on failure (caller treats unparseable exprs as opaque --
    typically GATHER or refused)."""
    try:
        return symbolic.pystr_to_symbolic(str(expr))
    except Exception:
        return None


def _direct_symbols(expr: sympy.Expr) -> Set[str]:
    """Symbol names appearing OUTSIDE any gather Subscript in ``expr``. Math functions on iter-vars
    (``floor(i / 4)``, ``exp(i)``) still expose ``i`` as direct (iter-var in address arithmetic, no
    memory load; affine-coeff analysis catches non-affine cases). :class:`~dace.symbolic.Subscript`
    = only stop boundary (explicit data-dependent access)."""
    if expr is None:
        return set()
    if isinstance(expr, sympy.Symbol):
        return {str(expr)}
    # Subscript = gather stop boundary: inner symbols are gather-index inputs, not address arithmetic
    if isinstance(expr, symbolic.Subscript):
        return set()
    args = getattr(expr, 'args', None)
    if not args:
        return set()
    result: Set[str] = set()
    for arg in args:
        result |= _direct_symbols(arg)
    return result


#: DaCe / numpy dtype names appearing as cast *functions* in tasklet bodies (``int64(i)``,
#: ``float64(x)``). After sympify these are :class:`sympy.Function` nodes; :func:`_strip_casts`
#: collapses them to their argument.
_CAST_NAMES = frozenset({
    'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'float16', 'float32', 'float64', 'bool',
    'bool_'
})

#: Strip ``dace.`` / ``np.`` / ``numpy.`` prefix before a cast name so the remainder parses
#: (``dace.int64(`` -> ``int64(``). Attribute form ``dace.int64`` is what
#: :func:`dace.symbolic.pystr_to_symbolic` can't parse; this minimal strip = only text fixup.
_CAST_PREFIX_RE = re.compile(r'\b(?:dace|np|numpy)\.(?=(?:u?int(?:8|16|32|64)|float(?:16|32|64)|bool_?)\b)')


def _strip_casts(expr: sympy.Expr) -> sympy.Expr:
    """Collapse dtype-cast Function nodes (``int64(i)`` -> ``i``) in ``expr``.

    Cast = no-op for index arithmetic but hides affine structure (``int64(i) + c`` not recognised as
    ``i + c``). Done at sympy level per the "parse with dace.symbolic" convention.
    """
    if expr is None:
        return expr
    try:
        return expr.replace(lambda e: isinstance(e, sympy.Function) and e.func.__name__ in _CAST_NAMES,
                            lambda e: e.args[0])
    except Exception:  # noqa: BLE001
        return expr


def _sympify_tasklet_rhs(text: str) -> Optional[sympy.Expr]:
    """Parse a tasklet/interstate RHS to a sympy expr, casts collapsed.

    Strip only the unparseable ``dace.``/``np.`` prefix textually, then defer to
    :func:`dace.symbolic.pystr_to_symbolic` + sympy-level cast collapse.
    """
    return _strip_casts(_safe_sympify(_CAST_PREFIX_RE.sub('', text)))


def _reaching_ise_assignment(state, symbol: str, inner_sdfg: Optional[SDFG] = None) -> Optional[str]:
    """Backward-walk the (flat) state graph from ``state`` for the nearest interstate-edge
    assignment of ``symbol`` (its reaching definition).

    Frontend emits one ``__sym_i_plus_offset1 = <local>`` per program point (read-b, read-a,
    write-a), so symbol is multiply-assigned across the body; only the assignment on the path into
    ``state`` is right.

    Flat-states pre-condition: after if-condition mask lowering the body CFG is one level of plain
    states, so a one-level BFS over ``state``'s region edges is a complete reaching-def (no
    parent-CFG ascent).

    :param state: The access state (an :class:`SDFGState`).
    :param symbol: Symbol whose reaching definition is sought.
    :param inner_sdfg: Unused (kept for call-site symmetry / future use).
    :returns: RHS expression string, or ``None`` when no assignment reaches.
    """
    region = getattr(state, "parent_graph", None)
    if region is None:
        return None
    visited = set()
    frontier = [state]
    while frontier:
        nxt = []
        for blk in frontier:
            for e in region.in_edges(blk):
                assigns = e.data.assignments if e.data is not None else {}
                if symbol in assigns:
                    return str(assigns[symbol])
                if id(e.src) not in visited:
                    visited.add(id(e.src))
                    nxt.append(e.src)
        frontier = nxt
    return None


def _build_symbol_definition_map(inner_sdfg: Optional[SDFG], state=None) -> Dict[str, sympy.Expr]:
    """Map ``symbol_name -> defining sympy expression`` for symbols resolvable within ``inner_sdfg``
    (optionally reaching-def-disambiguated at ``state``).

    Two definition sources (frontend promotes a computed index like ``i + offset1`` through BOTH into
    a memlet subset):

    1. **Interstate-edge assignments** ``sym = rhs``. Uniquely-assigned → recorded directly.
       Multiply-assigned (one ``__sym_i_plus_offset1`` per program point) → disambiguated by reaching
       def at ``state`` when given, else omitted (unresolved).
    2. **Scalar AccessNodes written by a single tasklet** ``__out = <body>``. Body input connectors
       rewritten to source data names (``__in2`` -> ``offset1``) via :meth:`subs`, casts collapsed,
       so recorded expr is in terms of source arrays/scalars + directly-read symbols (map iter-var
       ``i`` appears verbatim).

    Ambiguity resolved toward omission: unresolvable symbol absent from map, so
    :func:`resolve_index_expr` leaves it as-is (correctness-preserving fallback).

    :param inner_sdfg: Body SDFG to scan; ``None`` yields an empty map.
    :param state: Optional access state for reaching-def disambiguation of multiply-assigned
        interstate symbols.
    :returns: ``{name: sympy.Expr}`` of resolvable symbols.
    """
    if inner_sdfg is None:
        return {}

    # --- source 1: interstate-edge symbol assignments ---
    ise_rhs: Dict[str, Set[str]] = {}
    for edge in inner_sdfg.all_interstate_edges():
        assigns = edge.data.assignments if edge.data is not None else {}
        for k, v in assigns.items():
            ise_rhs.setdefault(k, set()).add(str(v))

    defs: Dict[str, sympy.Expr] = {}
    for k, rhs_set in ise_rhs.items():
        if len(rhs_set) == 1:
            chosen = next(iter(rhs_set))
        elif state is not None:
            chosen = _reaching_ise_assignment(state, k, inner_sdfg)  # reaching def disambiguates
            if chosen is None:
                continue
        else:
            continue  # ambiguous, no state -> unresolved
        expr = _safe_sympify(chosen)
        if expr is not None:
            defs[k] = expr

    # --- source 2: scalars written by a single tasklet ``__out = <body>`` ---
    # name -> set of resolved-expr strings; keep only unambiguous singletons.
    scalar_defs: Dict[str, Set[str]] = {}
    for sd in inner_sdfg.all_sdfgs_recursive():
        for state in sd.states():
            for node in state.nodes():
                if not isinstance(node, nodes.AccessNode):
                    continue
                in_edges = state.in_edges(node)
                if len(in_edges) != 1:
                    continue
                producer = in_edges[0].src
                if not isinstance(producer, nodes.Tasklet) or len(producer.out_connectors) != 1:
                    continue
                out_conn = next(iter(producer.out_connectors))
                body = producer.code.as_string if producer.code is not None else ""
                body = body.strip().rstrip(";").strip()
                prefix = f"{out_conn} = "
                if not body.startswith(prefix):
                    continue
                rhs_expr = _sympify_tasklet_rhs(body[len(prefix):].strip())
                if rhs_expr is None:
                    continue
                # Rewrite input connectors -> source data names
                rename = {}
                for ie in state.in_edges(producer):
                    if ie.dst_conn and ie.data is not None and ie.data.data is not None:
                        rename[symbolic.pystr_to_symbolic(ie.dst_conn)] = symbolic.pystr_to_symbolic(ie.data.data)
                if rename:
                    rhs_expr = rhs_expr.subs(rename)
                scalar_defs.setdefault(node.data, set()).add(str(rhs_expr))

    for name, rhs_set in scalar_defs.items():
        if name in defs or len(rhs_set) != 1:
            continue  # ISE def wins / ambiguous scalar def -> skip
        expr = _safe_sympify(next(iter(rhs_set)))
        if expr is not None:
            defs[name] = expr
    # A symbol whose own definition references itself (``j = j + 1``) is a loop-carried RECURRENCE:
    # its value changes between program points. Such a loop is never a tiled parallel map (LoopToMap
    # refuses recurrences) → access stays in scalar control flow, so leave the symbol UNRESOLVED,
    # preserving the subset (``a[j]``) verbatim. Substituting would run ``resolve_index_expr``'s
    # fixpoint to the cap (``j`` -> ``j + _max_depth``) and corrupt the subset (TSVC s123). The same
    # instability propagates transitively: an index DEFINED from a recurrence symbol (``k = j + 1``,
    # snapshotted at loop-top while ``j`` is bumped before the ``b[k]``/``c[k]`` use -- TSVC s128) is
    # equally unstable, because the substituted ``j`` would read its post-update value at the use
    # site. Collect every recurrence symbol from the raw assignment RHSs and drop both the
    # self-referential defs AND any def whose RHS depends on one. (Tiled access carrying such an
    # index is refused downstream by the tile-index builder, not rewritten -- see InsertTileLoadStore.)
    recurrence_syms: Set[str] = set()
    for sym, rhs_set in ise_rhs.items():
        for rhs in rhs_set:
            rexpr = _safe_sympify(rhs)
            if rexpr is not None and sym in {str(s) for s in rexpr.free_symbols}:
                recurrence_syms.add(sym)
                break
    for name, rhs_set in scalar_defs.items():
        for rhs in rhs_set:
            rexpr = _safe_sympify(rhs)
            if rexpr is not None and name in {str(s) for s in rexpr.free_symbols}:
                recurrence_syms.add(name)
                break
    # Taint every def transitively reaching a recurrence symbol, to a fixpoint. A def that
    # references a recurrence symbol is itself unstable (``LEN_1D_minus_k = LEN_1D - k``, ``k``
    # carried; ``k = j + 1``, ``j`` carried), and so is any def that references such a tainted mint
    # in turn (``__sym_LEN_1D_minus_k = LEN_1D_minus_k`` -- the frontend's promoted subset symbol).
    # Propagating the taint UP the whole chain is essential: dropping only the leaf ``LEN_1D - k``
    # link while keeping ``__sym_LEN_1D_minus_k -> LEN_1D_minus_k`` leaves resolution stranded at the
    # dropped intermediate ``LEN_1D_minus_k`` -- a scalar the frontend already promoted away and no
    # longer declares in the tiled scope, so the subset ``b[LEN_1D_minus_k]`` compiles to an
    # undeclared reference (TSVC s122). Dropping the whole chain instead leaves the ORIGINAL promoted
    # subset symbol unresolved (matching s128's fully-unresolved ``b[k]``), which the downstream
    # tile-index builder refuses cleanly.
    tainted: Set[str] = set(recurrence_syms)
    changed = True
    while changed:
        changed = False
        for k, v in defs.items():
            if k in tainted:
                continue
            v_syms = {str(s) for s in v.free_symbols}
            if k in v_syms or (v_syms & tainted):  # self-ref, or reaches a tainted symbol
                tainted.add(k)
                changed = True
    filtered = {k: v for k, v in defs.items() if k not in tainted}
    return filtered


def resolve_index_expr(expr: sympy.Expr,
                       inner_sdfg: Optional[SDFG],
                       _defs: Optional[Dict[str, sympy.Expr]] = None,
                       _max_depth: int = 16) -> sympy.Expr:
    """Resolve promoted index symbols in ``expr`` back to their defining arithmetic so iter-var
    dependence is visible to the classifier.

    Frontend promotes a computed index ``i + offset1`` to a scalar then to a symbol
    ``__sym_i_plus_offset1`` in the memlet subset; classifier would else see that opaque symbol as
    loop-invariant. Substitutes each resolvable free symbol (see
    :func:`_build_symbol_definition_map`) with its definition, recursively, to a fixpoint or
    ``_max_depth``. Cycle/ambiguity safe: unresolvable symbols untouched.

    :param expr: The (sympified) index expression to resolve.
    :param inner_sdfg: Body SDFG carrying the definitions.
    :param _defs: Precomputed definition map (internal; built once per subset).
    :param _max_depth: Substitution-iteration cap (cycle guard).
    :returns: Resolved expression (or ``expr`` unchanged if nothing resolves).
    """
    if expr is None:
        return expr
    defs = _build_symbol_definition_map(inner_sdfg) if _defs is None else _defs
    if not defs:
        return expr
    cur = expr
    for _ in range(_max_depth):
        free = {str(s) for s in cur.free_symbols}
        applicable = {s: defs[s] for s in free if s in defs}
        if not applicable:
            break
        subs = {symbolic.pystr_to_symbolic(s): rhs for s, rhs in applicable.items()}
        nxt = cur.subs(subs)
        if nxt == cur:
            break  # fixpoint (or self-referential def)
        cur = nxt
    return cur


def _scalar_loaded_from_array(sdfg: SDFG, name: str) -> bool:
    """True if ``name`` is a transient Scalar whose value is loaded from a (non-Scalar) Array -- a
    gather-index scalar (``N__slice = Xiv[j]``, written by a memlet COPY). The frontend promotes such
    a scalar to a subset symbol (``__sym_N__slice = N__slice``); ``_build_symbol_definition_map``
    source 2 only rewrites TASKLET-defined scalars to their source array, so a COPY-defined one is
    missed and the array name never surfaces. The scalar is state-local, so inlining it into a later
    state's subset references it out of scope (undeclared-identifier compile error) -- keep the
    promoted symbol instead.
    """
    import dace.data as _dd
    desc = sdfg.arrays.get(name)
    if not (isinstance(desc, _dd.Scalar) and desc.transient):
        return False
    for state in sdfg.states():
        for node in state.nodes():
            if not (isinstance(node, nodes.AccessNode) and node.data == name):
                continue
            for edge in state.in_edges(node):
                src = edge.src
                if isinstance(src, nodes.AccessNode):
                    sources = [src.data]
                elif isinstance(src, nodes.Tasklet):
                    sources = [e.data.data for e in state.in_edges(src) if e.data is not None and e.data.data is not None]
                else:
                    sources = []
                for sname in sources:
                    sdesc = sdfg.arrays.get(sname)
                    if isinstance(sdesc, _dd.Array) and not isinstance(sdesc, _dd.Scalar):
                        return True
    return False


def expr_is_data_dependent(expr: sympy.Expr, sdfg: SDFG) -> bool:
    """True if ``expr`` is a data-dependent index -- reads an array value (a gather like ``idx[i]``),
    so must NOT be inlined into a memlet subset (stays gather form for the gather machinery).

    Detected three ways: a :class:`~dace.symbolic.Subscript` node anywhere, a free symbol naming a
    non-Scalar :class:`~dace.data.Array` descriptor (resolver rewrites a gather scalar's defining
    tasklet to read the source array name, so ``idx`` shows up as a free symbol), or a transient
    Scalar loaded from an Array (a copy-defined gather-index scalar the tasklet-only rewrite misses).
    """
    if expr is None:
        return False
    import dace.data as _dd
    try:
        if expr.atoms(symbolic.Subscript):
            return True
    except Exception:  # noqa: BLE001
        pass
    for s in expr.free_symbols:
        desc = sdfg.arrays.get(str(s))
        if isinstance(desc, _dd.Array) and not isinstance(desc, _dd.Scalar):
            return True
        if _scalar_loaded_from_array(sdfg, str(s)):  # gather-index scalar (copy-defined)
            return True
    return False


def propagate_subset(subset, inner_sdfg: Optional[SDFG], state=None):
    """Rewrite a memlet ``subset`` by inlining promoted index symbols back to their original
    arithmetic (``A[__sym]`` / ``A[i_plus_offset]`` -> ``A[i+offset]``) so access is direct, widens
    to a dense load.

    Each range bound resolved via :func:`resolve_index_expr` (crossing interstate-edge assignments +
    scalar-defining tasklets, reaching-def via ``state``). Bound left untouched when its resolved
    form is data-dependent (:func:`expr_is_data_dependent`) -- a genuine gather index that must keep
    its symbol/Subscript for the gather machinery.

    :param subset: The memlet :class:`~dace.subsets.Range` to rewrite.
    :param inner_sdfg: Body SDFG carrying the symbol/scalar definitions.
    :param state: Access state for reaching-def disambiguation.
    :returns: A new :class:`~dace.subsets.Range` if anything changed, else ``None``.
    """
    if inner_sdfg is None or subset is None or not hasattr(subset, "ranges"):
        return None
    defs = _build_symbol_definition_map(inner_sdfg, state)
    if not defs:
        return None

    def _rewrite(bound):
        e = _safe_sympify(bound)
        if e is None:
            return bound, False
        resolved = resolve_index_expr(e, inner_sdfg, _defs=defs)
        if resolved == e:
            return bound, False
        if expr_is_data_dependent(resolved, inner_sdfg):
            return bound, False  # gather index -> keep
        return resolved, True

    new_ranges = []
    changed = False
    for (lo, hi, step) in subset.ranges:
        nlo, c1 = _rewrite(lo)
        nhi, c2 = _rewrite(hi)
        new_ranges.append((nlo, nhi, step))
        changed = changed or c1 or c2
    if not changed:
        return None
    from dace.subsets import Range as _Range
    return _Range(new_ranges)


def _is_tile_dependent(symbol: str,
                       iter_vars: Set[str],
                       inner_sdfg: Optional[SDFG],
                       memo: Optional[Dict[str, bool]] = None) -> bool:
    """True iff ``symbol`` transitively depends on a tile iter-var (section 4.2 join rule).

    Depends via interstate-edge assignments in ``inner_sdfg``. Same relation codegen uses for
    per-lane materialisation, reused as ground truth for GATHER fallback. Walks
    ``inner_sdfg.all_interstate_edges()`` collecting ``edge.data.assignments``, follows RHS free
    symbols transitively. ``memo`` caches per-symbol results.

    :param symbol: Symbol name to test.
    :param iter_vars: The K tile iter-var names.
    :param inner_sdfg: Body NSDFG (assignments live on its interstate edges). ``None`` → only the
        direct ``symbol in iter_vars`` check fires.
    :param memo: Optional shared memo across calls.
    :returns: ``True`` if ``symbol`` is (transitively) tile-dependent.
    """
    if symbol in iter_vars:
        return True
    if inner_sdfg is None:
        return False
    if memo is None:
        memo = {}
    if symbol in memo:
        return memo[symbol]
    memo[symbol] = False  # tentative no -- cycle guard
    for edge in inner_sdfg.all_interstate_edges():
        assigns = edge.data.assignments if edge.data is not None else {}
        if symbol not in assigns:
            continue
        rhs_expr = _safe_sympify(assigns[symbol])
        if rhs_expr is None:
            continue
        for fs in rhs_expr.free_symbols:
            if _is_tile_dependent(str(fs), iter_vars, inner_sdfg, memo):
                memo[symbol] = True
                return True
    return memo[symbol]


def classify_symbols(expr: sympy.Expr, iter_vars: Sequence[str], inner_sdfg: Optional[SDFG]) -> Dict[str, bool]:
    """Per-symbol tile-dependence map for every symbol in ``expr``.

    :param expr: The expression to walk.
    :param iter_vars: Tile iter-var names.
    :param inner_sdfg: Body NSDFG carrying interstate assignments.
    :returns: ``{symbol_name: is_tile_dependent}`` for every free symbol in ``expr``. Tile
        iter-vars appear in the map (always ``True``).
    """
    if expr is None:
        return {}
    iter_var_set = set(iter_vars)
    memo: Dict[str, bool] = {}
    out: Dict[str, bool] = {}
    for fs in expr.free_symbols:
        name = str(fs)
        out[name] = _is_tile_dependent(name, iter_var_set, inner_sdfg, memo)
    return out


def compute_per_iter_var_dep_mask(gather_expr: str, iter_vars: Sequence[str],
                                  inner_sdfg: Optional[SDFG]) -> Tuple[bool, ...]:
    """Per-iter-var dependency mask for a gather expression.

    Per tile iter-var, ``True`` iff gather expr transitively depends on it (via interstate-edge
    assignments in ``inner_sdfg``). Walks back through per-lane symbols introduced by
    :class:`BypassTrivialAssignTasklets` -- e.g. ``__sym_<>`` from ``__sym_<> = idx[i]`` tracked as
    dep on ``i``, not ``j``.

    Post-Bypass-aware version of the materialiser's direct ``free_symbols`` check. Walker calls this
    BEFORE :func:`materialise_per_lane_index_tile` so per-dim ONE-marker emission has the correct dep
    mask.

    :param gather_expr: Gather expr string (e.g. ``"__sym_x"`` or ``"idx[i]"``).
    :param iter_vars: Tile iter-var names in order.
    :param inner_sdfg: Body NSDFG carrying interstate assignments. ``None`` → direct iter-var
        membership check (no transitive walk).
    :returns: A tuple ``(dep_i, dep_j, ...)`` of length ``len(iter_vars)``.
    """
    expr = _safe_sympify(gather_expr)
    if expr is None:
        return tuple(True for _ in iter_vars)
    mask: List[bool] = []
    for v in iter_vars:
        any_dep = False
        memo: Dict[str, bool] = {}
        for fs in expr.free_symbols:
            if _is_tile_dependent(str(fs), {v}, inner_sdfg, memo):
                any_dep = True
                break
        mask.append(any_dep)
    return tuple(mask)


def _gather_subscripts(expr: sympy.Expr) -> List[symbolic.Subscript]:
    """Every :class:`Subscript` node anywhere in ``expr``. Detects data-dependent indices
    (``arr[idx[i]]``)."""
    if expr is None:
        return []
    result: List[symbolic.Subscript] = []
    if isinstance(expr, symbolic.Subscript):
        result.append(expr)
    args = getattr(expr, 'args', None)
    if args:
        for arg in args:
            result.extend(_gather_subscripts(arg))
    return result


def _find_named_symbol(expr: sympy.Expr, var_name: str) -> Optional[sympy.Symbol]:
    """Actual :class:`sympy.Symbol` instance in ``expr`` named ``var_name``. DaCe's ``symbol``
    subclasses :class:`sympy.Symbol` but doesn't compare equal to a fresh ``sympy.Symbol``;
    :func:`sympy.Poly` then treats the mismatched generator as opaque (degree 0). Picking the real
    symbol out of ``expr.free_symbols`` lets Poly see the variable."""
    if expr is None:
        return None
    for s in expr.free_symbols:
        if str(s) == var_name:
            return s
    return None


def _affine_coeff_for(expr: sympy.Expr, var_name: str) -> Optional[sympy.Expr]:
    """Coefficient of ``var_name`` in ``expr`` if ``expr`` is affine in it (``expr = coeff * var +
    rest``, ``rest`` free of ``var_name``). ``None`` if non-affine or unresolvable."""
    if expr is None:
        return None
    sym = _find_named_symbol(expr, var_name) or symbolic.pystr_to_symbolic(var_name)
    try:
        poly = sympy.Poly(expr, sym)
    except (sympy.PolynomialError, sympy.GeneratorsError, TypeError):
        return None
    if poly.degree() > 1:
        return None
    if poly.degree() == 0:
        return sympy.Integer(0)
    return poly.coeff_monomial(sym)  # degree 1


def _affine_offset_for(expr: sympy.Expr, var_name: str) -> Optional[sympy.Expr]:
    """Constant term of ``expr`` w.r.t. ``var_name`` (the ``c`` in ``coeff * var + c``). ``None`` if
    non-affine."""
    if expr is None:
        return None
    sym = _find_named_symbol(expr, var_name) or symbolic.pystr_to_symbolic(var_name)
    try:
        poly = sympy.Poly(expr, sym)
    except (sympy.PolynomialError, sympy.GeneratorsError, TypeError):
        return None
    if poly.degree() > 1:
        return None
    return poly.nth(0)  # constant term


def _detect_replicate_factor(expr: sympy.Expr, var_name: str) -> Optional[int]:
    """Detect ``int_floor(affine_in_var, k)`` / ``int_ceil(...)`` at the top of ``expr``; return
    integer divisor ``k`` when the inner arg is affine in ``var_name``.

    ``None`` when: not an ``int_floor`` / ``int_ceil`` call, divisor not a concrete positive integer,
    or dividend not affine in ``var_name``.

    Within-dim replicate: dim still walks the source array (not a full-dim broadcast), but every
    ``k`` consecutive lanes share one source element. Codegen reads a ``W / k``-element contracted
    box, group-broadcasts.
    """
    if expr is None:
        return None
    fname = type(expr).__name__
    # Two equivalent names: ``int_floor`` / ``int_ceil`` (user-facing) and ``__int_floor`` /
    # ``__int_ceil`` (Python-operator-derived, from ``i // 4``). Same operation.
    if fname not in ("int_floor", "int_ceil", "__int_floor", "__int_ceil"):
        return None
    if len(expr.args) != 2:
        return None
    dividend, divisor = expr.args
    # Divisor: positive integer OR symbolic (e.g. ``DV`` in ``i // DV``). Per user direction
    # 2026-06-10 (tile dim must be a multiple of replicate factor):
    # * Static: TileLoad construction validates ``W % k == 0``, ValueError on violation.
    # * Symbolic: not statically verifiable; codegen emits ``__l / DV``, CORRECT only when
    #   ``W % DV == 0`` at runtime (user responsibility, documented on TileLoad property).
    #   Non-dividing symbolic divisors (e.g. ``test_div_index_symbol[3]``: DV=3, W=8) give wrong
    #   results -- by-design refusal, not a codegen bug.
    # Refuse floats: access exprs integer-valued; float divisor = upstream pass leaked a numeric
    # type. Fall to AFFINE/GATHER rather than truncating to int.
    if isinstance(divisor, (sympy.Float, float)):
        return None
    try:
        k = int(divisor)
        if k <= 1:
            return None
    except (TypeError, ValueError):
        if divisor is None:
            return None
        k = divisor  # symbolic -- runtime check (W % k == 0) at codegen per 2c7b88e26
    # Dividend must be affine in ``var_name`` (regular replication -- ``int_floor(idx[i], 2)`` is
    # data-dependent → GATHER, not REPLICATE).
    coeff = _affine_coeff_for(dividend, var_name)
    if coeff is None:
        return None
    return k


def _detect_modular_factor(expr: sympy.Expr, var_name: str) -> Optional[int]:
    """Detect ``(c * var + c0) % N`` -- the MODULAR per-dim pattern.

    Returns ``N`` (positive integer) when ``expr`` is a Mod / mod call with positive-integer RHS and
    LHS affine in ``var_name``; else ``None``. Tile-aligned cases (``N | c * W_p``) detected later by
    the classifier so MODULAR can reduce to LINEAR.
    """
    if expr is None:
        return None
    fname = type(expr).__name__
    # SymPy modulo is ``Mod`` (also from ``a % b``); DaCe's ``__mod__`` overload may yield ``mod``
    # or ``Mod`` depending on construction.
    if fname not in ("Mod", "mod", "__mod__"):
        return None
    if len(expr.args) != 2:
        return None
    dividend, divisor = expr.args
    try:
        N = int(divisor)
    except (TypeError, ValueError):
        return None
    if N <= 1:
        return None
    if _affine_coeff_for(dividend, var_name) is None:
        return None
    return N


def _resolve_gather_index_an(inner_sdfg: Optional[SDFG], expr: sympy.Expr) -> Optional[nodes.AccessNode]:
    """If ``expr`` has exactly one Subscript whose base is an array name in ``inner_sdfg.arrays``,
    return an :class:`AccessNode` for that array from any state. ``None`` when there are zero /
    multiple subscripts, the base isn't an array, or no AccessNode exists."""
    if inner_sdfg is None or expr is None:
        return None
    subs = _gather_subscripts(expr)
    if len(subs) != 1:
        return None
    base = subs[0].args[0]
    base_name = str(base)
    if base_name not in inner_sdfg.arrays:
        return None
    for st in inner_sdfg.all_states():
        for n in st.nodes():
            if isinstance(n, nodes.AccessNode) and n.data == base_name:
                return n
    return None


# ----- public API --------------------------------------------------------


def classify_tile_access(subset: Range,
                         iter_vars: Sequence[str],
                         inner_sdfg: Optional[SDFG] = None,
                         state=None) -> TileAccess:
    """Classify a memlet subset for tile lib-node dispatch.

    :param subset: The :class:`Range` to classify (typically a memlet's ``subset``).
    :param iter_vars: The K tile iter-var names, in tile-lane order (innermost-last by convention).
    :param inner_sdfg: Optional inner SDFG to resolve GATHER index AccessNodes and promoted index
        symbols. ``None`` outside the body context (gather-index field left empty).
    :param state: Optional access state; disambiguates multiply-assigned promoted index symbols by
        reaching definition (one ``__sym_i_plus_offset1`` per program point).
    :returns: A :class:`TileAccess` record. Always returns, never raises. Unrecognisable patterns
        degrade to GATHER (correctness fallback).
    """
    iter_var_set: Set[str] = set(iter_vars)
    n_dims = len(subset.ranges)

    per_dim_kind: List[PerDimKind] = []
    dim_strides: List[Optional[int]] = []
    dim_iter_var: List[Optional[str]] = []
    gather_index_per_dim: List[Optional[nodes.AccessNode]] = []
    dim_offset: List[Optional[sympy.Expr]] = []
    replicate_factor_per_dim: List[Optional[int]] = []

    # Per-iter-var tracking for diagonal / transpose detection.
    iter_var_in_dim: Dict[str, List[int]] = {v: [] for v in iter_vars}
    dim_to_canonical_iter_var: List[Optional[int]] = []

    # Resolve promoted index symbols (``__sym_i_plus_offset1`` -> ``i + offset1``) once per subset so
    # each dim's iter-var dependence is visible. Empty/unresolvable leaves exprs untouched. ``state``
    # disambiguates multiply-assigned interstate symbols by reaching def.
    _sym_defs = _build_symbol_definition_map(inner_sdfg, state)

    for d, (lo, _hi, _stp) in enumerate(subset.ranges):
        lo_sym = _safe_sympify(lo)
        if lo_sym is not None and _sym_defs:
            lo_sym = resolve_index_expr(lo_sym, inner_sdfg, _defs=_sym_defs)
        direct = _direct_symbols(lo_sym) if lo_sym is not None else set()
        direct_tile_vars = direct & iter_var_set

        # Stop 1: any tile iter-var inside a Subscript -> GATHER dim.
        gather_dim = False
        if lo_sym is not None:
            for sub in _gather_subscripts(lo_sym):
                # Subscript args = (container, idx0, idx1, ...); iter-var nested if in any idx arg's
                # free symbols.
                for sub_arg in sub.args[1:]:
                    fs = {str(s) for s in sub_arg.free_symbols}
                    if fs & iter_var_set:
                        gather_dim = True
                        break
                if gather_dim:
                    break
        if gather_dim:
            per_dim_kind.append(PerDimKind.GATHER)
            dim_strides.append(None)
            dim_iter_var.append(None)
            gather_index_per_dim.append(_resolve_gather_index_an(inner_sdfg, lo_sym))
            dim_offset.append(None)
            replicate_factor_per_dim.append(None)
            dim_to_canonical_iter_var.append(None)
            continue

        # Section 4.2 join rule: any non-iter-var symbol that is transitively tile-dependent
        # (defined by an interstate edge whose RHS touches a tile iter-var) forces GATHER. Catches
        # ``a[2*sym + 1]`` (sym <- i + 3), which would else look CONSTANT to the direct-symbol check.
        if lo_sym is not None and inner_sdfg is not None:
            non_tile_syms = direct - iter_var_set
            if any(_is_tile_dependent(s, iter_var_set, inner_sdfg) for s in non_tile_syms):
                per_dim_kind.append(PerDimKind.GATHER)
                dim_strides.append(None)
                dim_iter_var.append(None)
                gather_index_per_dim.append(None)
                dim_offset.append(None)
                replicate_factor_per_dim.append(None)
                dim_to_canonical_iter_var.append(None)
                continue

        # Stop 2: no tile iter-var as direct symbol -> BROADCAST dim (replicate_factor = None: all W
        # lanes share one element).
        if not direct_tile_vars:
            per_dim_kind.append(PerDimKind.BROADCAST)
            dim_strides.append(0)
            dim_iter_var.append(None)
            gather_index_per_dim.append(None)
            dim_offset.append(_safe_sympify(lo))
            replicate_factor_per_dim.append(None)
            dim_to_canonical_iter_var.append(None)
            continue

        # Stop 3: exactly one tile iter-var directly -> STRUCTURED_1, REPLICATE, or AFFINE.
        if len(direct_tile_vars) == 1:
            tvar = next(iter(direct_tile_vars))
            # Stop 3a: ``int_floor(c*tvar + c0, k)`` / ``int_ceil(...)`` -> REPLICATE factor k.
            # Before the affine check so the function call doesn't fall through to AFFINE.
            replicate_k = _detect_replicate_factor(lo_sym, tvar)
            if replicate_k is not None:
                per_dim_kind.append(PerDimKind.REPLICATE)
                dim_strides.append(1)  # contracted-box stride
                dim_iter_var.append(tvar)
                gather_index_per_dim.append(None)
                dim_offset.append(None)
                replicate_factor_per_dim.append(replicate_k)
                iter_var_in_dim[tvar].append(d)
                dim_to_canonical_iter_var.append(list(iter_vars).index(tvar))
                continue
            # Stop 3b: ``(c * tvar + c0) % N`` -> MODULAR. Codegen falls back to GATHER for the
            # general case; tile-aligned reduction to LINEAR is future work (design section 4.2).
            modular_N = _detect_modular_factor(lo_sym, tvar)
            if modular_N is not None:
                per_dim_kind.append(PerDimKind.MODULAR)
                dim_strides.append(None)
                dim_iter_var.append(tvar)
                gather_index_per_dim.append(None)
                dim_offset.append(None)
                replicate_factor_per_dim.append(None)
                iter_var_in_dim[tvar].append(d)
                dim_to_canonical_iter_var.append(list(iter_vars).index(tvar))
                continue
            coeff = _affine_coeff_for(lo_sym, tvar)
            offset = _affine_offset_for(lo_sym, tvar)
            if coeff is not None and coeff == 1:
                per_dim_kind.append(PerDimKind.STRUCTURED_1)
                dim_strides.append(1)
                dim_iter_var.append(tvar)
                gather_index_per_dim.append(None)
                dim_offset.append(offset)
                replicate_factor_per_dim.append(1)
                iter_var_in_dim[tvar].append(d)
                dim_to_canonical_iter_var.append(list(iter_vars).index(tvar))
                continue
            if coeff is not None:
                # G6 (design section 4.2 join rule): AFFINE coefficient must be tile-independent.
                # ``a[N*i]`` with N outer-constant = AFFINE stride N; with N tile-dependent (defined
                # by an interstate edge whose RHS touches a tile iter-var) → GATHER. Forces GATHER for
                # symbolic coefficients whose tile-independence we can't prove.
                tile_dep_coeff = False
                if inner_sdfg is not None:
                    coeff_syms = {str(s) for s in coeff.free_symbols} if hasattr(coeff, "free_symbols") else set()
                    if any(_is_tile_dependent(s, iter_var_set, inner_sdfg) for s in coeff_syms):
                        tile_dep_coeff = True
                if tile_dep_coeff:
                    per_dim_kind.append(PerDimKind.GATHER)
                    dim_strides.append(None)
                    dim_iter_var.append(None)
                    gather_index_per_dim.append(None)
                    dim_offset.append(None)
                    replicate_factor_per_dim.append(None)
                    dim_to_canonical_iter_var.append(None)
                    continue
                # Affine in one iter-var. Coerce to int when possible (emitter prefers concrete
                # strides). Symbolic tile-independent coefficient: keep the sympy expr so the lib
                # node's ListProperty (element_type=pystr_to_symbolic) serialises it and codegen
                # inlines it as a C++ variable.
                try:
                    stride_value = int(coeff)
                except (TypeError, ValueError):
                    stride_value = coeff
                per_dim_kind.append(PerDimKind.AFFINE)
                dim_strides.append(stride_value)
                dim_iter_var.append(tvar)
                gather_index_per_dim.append(None)
                dim_offset.append(offset)
                replicate_factor_per_dim.append(None)
                iter_var_in_dim[tvar].append(d)
                dim_to_canonical_iter_var.append(list(iter_vars).index(tvar))
                continue
            # Non-affine, not int_floor/int_ceil (e.g. ``i**2``): AFFINE without an int stride;
            # emitter degrades to GATHER with the per-lane expression.
            per_dim_kind.append(PerDimKind.AFFINE)
            dim_strides.append(None)
            dim_iter_var.append(tvar)
            gather_index_per_dim.append(None)
            dim_offset.append(None)
            replicate_factor_per_dim.append(None)
            iter_var_in_dim[tvar].append(d)
            dim_to_canonical_iter_var.append(list(iter_vars).index(tvar))
            continue

        # Stop 4: multiple tile iter-vars in the dim. AFFINE only if JOINTLY affine -- each tile
        # var's coefficient must be tile-independent (e.g. ``i + j``, a diagonal). Cross-term like
        # ``i*j`` (or resolved ``syma*i`` with ``syma <- j``) makes one var's coefficient depend on
        # another tile var -> non-linear in tile vars -> GATHER (G6 join rule, section 4.2). Mirrors
        # the single-var check at Stop 3.
        multi_var_gather = False
        for tv in direct_tile_vars:
            c = _affine_coeff_for(lo_sym, tv)
            if c is None:  # non-affine in tv (e.g. i**2)
                multi_var_gather = True
                break
            c_syms = {str(s) for s in c.free_symbols} if hasattr(c, "free_symbols") else set()
            if (c_syms & iter_var_set) or (inner_sdfg is not None
                                           and any(_is_tile_dependent(s, iter_var_set, inner_sdfg) for s in c_syms)):
                multi_var_gather = True
                break
        if multi_var_gather:
            per_dim_kind.append(PerDimKind.GATHER)
            dim_strides.append(None)
            dim_iter_var.append(None)
            gather_index_per_dim.append(None)
            dim_offset.append(None)
            replicate_factor_per_dim.append(None)
            dim_to_canonical_iter_var.append(None)
            continue
        # Jointly affine in multiple tile vars (e.g. ``i + j``) -> AFFINE (multi-var).
        per_dim_kind.append(PerDimKind.AFFINE)
        dim_strides.append(None)
        # No single representative iter-var; report the first alphabetically for consistency.
        rep = sorted(direct_tile_vars)[0]
        dim_iter_var.append(rep)
        gather_index_per_dim.append(None)
        dim_offset.append(None)
        replicate_factor_per_dim.append(None)
        for tv in direct_tile_vars:
            iter_var_in_dim[tv].append(d)
        dim_to_canonical_iter_var.append(list(iter_vars).index(rep))

    # Whole-subset kind: strongest per-dim kind. MODULAR / REPLICATE share the STRUCTURED bucket
    # (both perfectly regular; codegen picks the intrinsic from the per-dim records).
    kinds = set(per_dim_kind)
    if PerDimKind.GATHER in kinds or PerDimKind.MODULAR in kinds:
        # MODULAR routes to GATHER until the tile-aligned reduction lands (section 4.2 future work).
        kind = TileAccessKind.GATHER
    elif PerDimKind.AFFINE in kinds:
        kind = TileAccessKind.AFFINE
    elif kinds & {PerDimKind.LINEAR, PerDimKind.REPLICATE}:
        kind = TileAccessKind.STRUCTURED
    else:
        kind = TileAccessKind.BROADCAST

    # Diagonal: any iter-var spans >= 2 dims.
    diagonal = {v: tuple(dims) for v, dims in iter_var_in_dim.items() if len(dims) >= 2}

    # Transpose: STRUCTURED with iter-vars in non-canonical order (canonical: dim ``d`` carries
    # ``iter_vars[d]``).
    transpose: Optional[Tuple[int, ...]] = None
    if kind == TileAccessKind.STRUCTURED and len(per_dim_kind) == len(iter_vars):
        perm = tuple(dim_to_canonical_iter_var)  # type: ignore[arg-type]
        if all(p is not None for p in perm) and tuple(perm) != tuple(range(len(iter_vars))):
            transpose = perm  # type: ignore[assignment]

    return TileAccess(
        kind=kind,
        per_dim_kind=tuple(per_dim_kind),
        dim_strides=tuple(dim_strides),
        dim_iter_var=tuple(dim_iter_var),
        gather_index_per_dim=tuple(gather_index_per_dim),
        dim_offset=tuple(dim_offset),
        replicate_factor_per_dim=tuple(replicate_factor_per_dim),
        diagonal=diagonal,
        transpose=transpose,
    )
