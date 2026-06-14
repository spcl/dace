# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""K-dim tile access classifier for the vectorization tile pipeline.

One primitive -- :func:`classify_tile_access` -- answers: *given a memlet
subset, the K tile iter-vars, and (optionally) the inner SDFG, how does
the access vary as the K-dim tile iterates over its (W_0, ..., W_{K-1})
lanes?*

The classifier is **target-isa-agnostic** and does **no SDFG mutation**:
its sole output is a :class:`TileAccess` record. The emitter consumes
that record to pick a tile lib node (``TileLoad`` / ``TileLoadStrided``
/ ``TileLoad`` (with ``gather_dims``) / ``TileStore`` / ``TileStore`` (with ``gather_dims``) / ``TileBroadcast``)
and arch-specific intrinsics live in the lib-node expansions.

Per-dim classification
----------------------

Every dim of the subset is tagged independently as one of
:class:`PerDimKind`:

* **BROADCAST** -- no tile iter-var anywhere in the dim's expression
  (loop-invariant; one element splats to all lanes of this dim).
* **STRUCTURED_1** -- one tile iter-var as a *direct top-level symbol*
  with identity coefficient (``iter_var + c``). The cleanest case: lane
  ``l`` reads ``addr + l * elem_size``.
* **AFFINE** -- one or more tile iter-vars present as direct symbols
  but with non-unit coefficient (``2*i + 3``) or multiple iter-vars
  combined (``i + j``). Emitter chooses strided-load intrinsic vs.
  loop-of-loads vs. fallback to GATHER per arch.
* **GATHER** -- a tile iter-var appears inside a :class:`Subscript`
  (``arr[idx[i]]``). Data-dependent access; always lowers to TileLoad (gather_dims).

Whole-subset composition (DIAGONAL / TRANSPOSE flags)
-----------------------------------------------------

The whole-subset kind is the strongest per-dim kind, in order
``GATHER > AFFINE > STRUCTURED > BROADCAST``. Two composition flags are
also captured:

* **diagonal** -- a STRUCTURED_1 iter-var appears as the direct symbol
  in *multiple* dims (``arr[i, i]``). Per the locked design, lowers as
  GATHER with a 1-D index tile.
* **transpose** -- STRUCTURED_1 iter-vars appear in a non-canonical
  permutation of ``spec.iter_vars``. Lowers as GATHER with a permuted
  index tile (or a transpose intrinsic when the emitter can use one).

The classifier captures these as informational fields; the emitter
decides the actual lib-node target. Per the user's locked direction,
DIAGONAL and TRANSPOSE both fold into ``TileLoad`` (with ``gather_dims``) today.

dim_strides convention
----------------------

For non-GATHER classifications the per-dim coefficient is exposed as
``dim_strides`` so the load/store lib nodes can directly consume it:

* ``0`` -- BROADCAST dim (this dim doesn't vary per lane; splat).
* ``1`` -- STRUCTURED_1 dim (lane ``l`` reads index ``l`` more).
* ``c > 1`` -- AFFINE dim with coefficient ``c``.

Mixed-rank broadcasts (K=0 -> K=2 full splat; K=1 -> K=2 broadcast on
one dim; etc.) drop out automatically: the BROADCAST dims get
``dim_stride = 0`` and the load lib node's expansion replicates that
value across the dim's lanes.

Gather-side composition rule
----------------------------

When ANY dim is GATHER, the whole-subset kind is GATHER. Non-GATHER
dims fold into the gather's index expression (the structured dims
contribute affine sub-expressions inline; no separate index tile
materialisation is needed for them).
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

    #: No tile iter-var -- loop-invariant. All W lanes share one source
    #: element (codegen splats).
    CONSTANT = "constant"

    #: Exactly ``iter_var + c`` -- stride 1, constant offset. Lane ``l``
    #: reads index ``l`` more.
    LINEAR = "linear"

    #: ``int_floor(c * iter_var + c0, k)`` / ``int_ceil(...)`` with
    #: ``1 < k < W`` -- group-broadcast within the dim with factor ``k``
    #: (codegen loads ``W/k`` and replicates each ``k`` times).
    REPLICATE = "replicate"

    #: ``s * iter_var + c`` with ``s`` outer-scope constant, ``s >= 1``.
    #: Strided load with constant stride. Multiple iter-vars sharing a
    #: dim is GATHER.
    AFFINE = "affine"

    #: ``(c * iter_var + c0) % N`` with ``N`` outer-scope constant.
    #: Cyclic wrap. When ``N | c * W_p`` (tile-aligned), reduces to
    #: LINEAR with a per-tile constant offset.
    MODULAR = "modular"

    #: Data-dependent / unsupported. Tile-dependent symbols in the
    #: expression force GATHER (see section 4.2 join rule).
    GATHER = "gather"

    # Backwards-compat aliases -- removed after the rename has
    # propagated through every consumer.
    BROADCAST = "constant"
    STRUCTURED_1 = "linear"


class TileAccessKind(enum.Enum):
    """Whole-subset kind. The strongest per-dim kind wins, in the order
    ``GATHER > AFFINE > STRUCTURED > BROADCAST``. Two specialised
    compositions (DIAGONAL / TRANSPOSE) of STRUCTURED_1 dims are tagged
    on the :class:`TileAccess` record for emitter information only --
    both fold into ``TileLoad`` (with ``gather_dims``) per the locked design."""

    BROADCAST = "broadcast"
    STRUCTURED = "structured"
    AFFINE = "affine"
    GATHER = "gather"


@dataclass(frozen=True)
class TileAccess:
    """Result of :func:`classify_tile_access` (data only, no behaviour).

    The emitter reads this record + the surrounding scope (mask
    availability, target_isa) to pick a lib node and wire its
    properties.
    """

    #: Whole-subset kind. Determined by the strongest per-dim kind.
    kind: TileAccessKind

    #: Per-dim classification, one entry per subset dim (in subset order).
    per_dim_kind: Tuple[PerDimKind, ...]

    #: Per-dim stride. ``0`` = broadcast dim; ``1`` = identity stride;
    #: ``c > 1`` = affine coefficient. ``None`` for GATHER dims (the
    #: stride lives in the gather-index expression). Same length as
    #: ``per_dim_kind``.
    dim_strides: Tuple[Optional[int], ...]

    #: For dims where the per-dim kind is STRUCTURED_1 or AFFINE: which
    #: tile iter-var is the dim's per-lane stride driver. ``None`` for
    #: BROADCAST and GATHER dims. Same length as ``per_dim_kind``.
    dim_iter_var: Tuple[Optional[str], ...]

    #: For GATHER dims: the in-scope :class:`AccessNode` of the index
    #: array, when resolvable. ``None`` for non-GATHER dims OR when the
    #: index source isn't a direct array read. Same length as
    #: ``per_dim_kind``.
    gather_index_per_dim: Tuple[Optional[nodes.AccessNode], ...]

    #: For dims where the per-dim kind is STRUCTURED_1 or AFFINE: the
    #: constant offset (the ``c`` in ``iter_var + c``). ``None`` for
    #: BROADCAST and GATHER dims.
    dim_offset: Tuple[Optional[sympy.Expr], ...]

    #: Per-dim replicate factor (lanes-per-distinct-value within the dim).
    #: The unifying field across BROADCAST / STRUCTURED_1 / REPLICATE
    #: regimes:
    #:
    #: * ``None`` for BROADCAST dims -- the codegen reads one element and
    #:   splats it to the dim's full W lanes (equivalently, factor = W
    #:   but the dim's W isn't visible to the classifier, so ``None`` is
    #:   the sentinel for "all lanes").
    #: * ``1`` for STRUCTURED_1 dims -- each lane reads a distinct
    #:   consecutive element, no replication.
    #: * ``k > 1`` for REPLICATE dims (``int_floor`` / ``int_ceil`` of an
    #:   affine arg) -- ``k`` consecutive lanes share each source
    #:   element; the codegen loads ``W / k`` elements and group-broadcasts
    #:   each ``k`` times.
    #: * ``None`` for AFFINE and GATHER dims (the concept doesn't apply).
    #:
    #: Same length as :attr:`per_dim_kind`.
    replicate_factor_per_dim: Tuple[Optional[int], ...]

    #: Diagonal composition: maps a tile iter-var name to the tuple of
    #: subset dims it appears in directly. Only populated when one
    #: iter-var spans multiple dims. Empty dict when no diagonal pattern.
    diagonal: Dict[str, Tuple[int, ...]] = field(default_factory=dict)

    #: Transpose composition: when present, the permutation of tile
    #: iter-vars across the subset dims, as a tuple ``(perm_0, ..., perm_{K-1})``
    #: with ``perm_d`` the iter-var INDEX in ``spec.iter_vars`` that
    #: drives dim ``d``. ``None`` when the dim order is canonical or
    #: the subset isn't a pure permutation.
    transpose: Optional[Tuple[int, ...]] = None


# ----- internal helpers --------------------------------------------------


def _safe_sympify(expr) -> Optional[sympy.Expr]:
    """Best-effort sympify. Returns ``None`` on failure (the caller treats
    unparseable expressions as opaque -- typically classified as GATHER
    or refused)."""
    try:
        return symbolic.pystr_to_symbolic(str(expr))
    except Exception:
        return None


def _direct_symbols(expr: sympy.Expr) -> Set[str]:
    """Set of symbol names that appear **outside any gather Subscript**
    in ``expr``. Math functions on iter-vars (``floor(i / 4)``,
    ``exp(i)``) still expose ``i`` as direct -- the iter-var participates
    in the address arithmetic without going through a memory load; the
    affine-coefficient analysis then catches the non-affine cases.
    :class:`~dace.symbolic.Subscript` is the only stop boundary because
    it explicitly represents a data-dependent access."""
    if expr is None:
        return set()
    if isinstance(expr, sympy.Symbol):
        return {str(expr)}
    # Subscript is the gather stop boundary: its inner symbols are
    # gather-index inputs, not address-arithmetic contributors.
    if isinstance(expr, symbolic.Subscript):
        return set()
    args = getattr(expr, 'args', None)
    if not args:
        return set()
    result: Set[str] = set()
    for arg in args:
        result |= _direct_symbols(arg)
    return result


#: DaCe / numpy dtype names that appear as cast *functions* in tasklet bodies
#: (``int64(i)``, ``float64(x)``). After sympify these are :class:`sympy.Function`
#: nodes; :func:`_strip_casts` collapses them to their argument.
_CAST_NAMES = frozenset({
    'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'float16', 'float32', 'float64', 'bool',
    'bool_'
})

#: Strips the ``dace.`` / ``np.`` / ``numpy.`` module prefix in front of a cast
#: name so the remainder parses (``dace.int64(`` -> ``int64(``; the attribute
#: form ``dace.int64`` is what :func:`dace.symbolic.pystr_to_symbolic` cannot
#: parse, so this minimal prefix strip is the only text-level fixup needed).
_CAST_PREFIX_RE = re.compile(r'\b(?:dace|np|numpy)\.(?=(?:u?int(?:8|16|32|64)|float(?:16|32|64)|bool_?)\b)')


def _strip_casts(expr: sympy.Expr) -> sympy.Expr:
    """Collapse dtype-cast Function nodes (``int64(i)`` -> ``i``) in ``expr``.

    The cast is a no-op for index arithmetic and would otherwise hide the
    affine structure (``int64(i) + c`` is not recognised as ``i + c``). Done
    at the sympy level per the "parse with dace.symbolic" convention.
    """
    if expr is None:
        return expr
    try:
        return expr.replace(lambda e: isinstance(e, sympy.Function) and e.func.__name__ in _CAST_NAMES,
                            lambda e: e.args[0])
    except Exception:  # noqa: BLE001
        return expr


def _sympify_tasklet_rhs(text: str) -> Optional[sympy.Expr]:
    """Parse a tasklet/interstate RHS to a sympy expr with casts collapsed.

    Strips only the unparseable ``dace.``/``np.`` module prefix textually, then
    defers entirely to :func:`dace.symbolic.pystr_to_symbolic` and the
    sympy-level cast collapse.
    """
    return _strip_casts(_safe_sympify(_CAST_PREFIX_RE.sub('', text)))


def _reaching_ise_assignment(state, symbol: str, inner_sdfg: Optional[SDFG] = None) -> Optional[str]:
    """Backward-walk the (flat) state graph from ``state`` to find the nearest
    interstate-edge assignment of ``symbol`` (its reaching definition).

    The frontend emits one ``__sym_i_plus_offset1 = <local>`` per program point
    (read-b, read-a, write-a), so the symbol is multiply-assigned across the
    body; only the assignment on the path into ``state`` is the right one.

    Relies on the flat-states pre-condition: after if-condition mask lowering
    the body CFG is a single level of plain states, so a one-level BFS over
    ``state``'s region edges is a complete reaching-def — no parent-CFG ascent
    is needed.

    :param state: The access state (an :class:`SDFGState`).
    :param symbol: The symbol whose reaching definition is sought.
    :param inner_sdfg: Unused (kept for call-site symmetry / future use).
    :returns: The RHS expression string, or ``None`` when no assignment reaches.
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
    """Map ``symbol_name -> defining sympy expression`` for symbols resolvable
    within ``inner_sdfg`` (optionally reaching-def-disambiguated at ``state``).

    Two definition sources are collected (the frontend promotes a computed
    index like ``i + offset1`` through *both* on its way into a memlet subset):

    1. **Interstate-edge assignments** ``sym = rhs``. A uniquely-assigned symbol
       is recorded directly. A multiply-assigned symbol (the frontend emits one
       ``__sym_i_plus_offset1`` per program point) is disambiguated by the
       reaching definition at ``state`` when ``state`` is given; otherwise it is
       omitted (left unresolved).
    2. **Scalar AccessNodes written by a single tasklet** ``__out = <body>``.
       The body's input connectors are rewritten to their source data names
       (``__in2`` -> ``offset1``) via :meth:`subs` and dtype casts collapsed, so
       the recorded expression is in terms of source arrays/scalars + directly-
       read symbols (the map iter-var ``i`` appears verbatim in the body).

    Ambiguity is resolved conservatively toward *omission*: an unresolvable
    symbol is simply not in the map, so :func:`resolve_index_expr` leaves it
    as-is and downstream classification falls back to its current
    (correctness-preserving) behavior.

    :param inner_sdfg: Body SDFG to scan; ``None`` yields an empty map.
    :param state: Optional access state for reaching-def disambiguation of
        multiply-assigned interstate symbols.
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
            continue  # ambiguous, no state -> leave unresolved
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
                # Rewrite input connectors -> their source data names via subs.
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
    return defs


def resolve_index_expr(expr: sympy.Expr,
                       inner_sdfg: Optional[SDFG],
                       _defs: Optional[Dict[str, sympy.Expr]] = None,
                       _max_depth: int = 16) -> sympy.Expr:
    """Resolve promoted index symbols in ``expr`` back to their defining
    arithmetic so iter-var dependence is visible to the classifier.

    The frontend promotes a computed index ``i + offset1`` to a scalar then to
    a symbol ``__sym_i_plus_offset1`` used in the memlet subset; the classifier
    would otherwise see that opaque symbol as loop-invariant. This substitutes
    each resolvable free symbol (see :func:`_build_symbol_definition_map`) with
    its definition, recursively, until a fixpoint or ``_max_depth`` is reached.
    Cycle/ambiguity safe: unresolvable symbols are left untouched.

    :param expr: The (sympified) index expression to resolve.
    :param inner_sdfg: Body SDFG carrying the definitions.
    :param _defs: Precomputed definition map (internal; built once per subset).
    :param _max_depth: Substitution-iteration cap (cycle guard).
    :returns: The resolved expression (or ``expr`` unchanged if nothing
        resolves).
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
            break  # fixpoint (or self-referential def) -> stop
        cur = nxt
    return cur


def expr_is_data_dependent(expr: sympy.Expr, sdfg: SDFG) -> bool:
    """Whether ``expr`` is a *data-dependent* index — it reads an array value
    (a gather like ``idx[i]``), so it must NOT be inlined into a memlet subset
    (it stays the gather form and flows to the gather machinery).

    Detected two ways: a :class:`~dace.symbolic.Subscript` node anywhere in the
    expression, or a free symbol that names a non-Scalar :class:`~dace.data.Array`
    descriptor (the resolver rewrites a gather scalar's defining tasklet to read
    the source array name, so ``idx`` shows up as a free symbol).
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
    return False


def propagate_subset(subset, inner_sdfg: Optional[SDFG], state=None):
    """Rewrite a memlet ``subset`` by inlining promoted index symbols back to
    their original arithmetic (``A[__sym]`` / ``A[i_plus_offset]`` -> ``A[i+offset]``)
    so the access pattern is direct and widens to a dense load.

    Each range bound is resolved via :func:`resolve_index_expr` (crossing
    interstate-edge assignments + scalar-defining tasklets, reaching-def via
    ``state``). A bound is left untouched when its resolved form is
    **data-dependent** (:func:`expr_is_data_dependent`) — that is a genuine
    gather index and must keep its symbol/Subscript for the gather machinery.

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
            return bound, False  # gather index -> keep original
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
    """True iff ``symbol`` transitively depends on a tile iter-var.

    Implements the section 4.2 join rule: a symbol is tile-dependent
    iff it (or any symbol it transitively depends on via interstate-
    edge assignments inside ``inner_sdfg``) is a tile iter-var. This
    is the same dependency relation the codegen uses to decide whether
    a symbol requires per-lane materialisation, so the classifier
    reuses it as the ground truth for GATHER fallback.

    Walks ``inner_sdfg.all_interstate_edges()`` collecting
    ``edge.data.assignments`` and follows the RHS free symbols
    transitively. ``memo`` caches per-symbol results.

    :param symbol: Symbol name to test.
    :param iter_vars: The K tile iter-var names.
    :param inner_sdfg: The body NSDFG (assignments live on its
        interstate edges). When ``None`` only the direct check
        ``symbol in iter_vars`` fires.
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
    memo[symbol] = False  # tentatively assume no -- guards against cycles
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
    :returns: ``{symbol_name: is_tile_dependent}`` for every free
        symbol in ``expr``. Tile iter-vars appear in the map (always
        ``True``).
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

    For each tile iter-var, returns ``True`` iff the gather expression
    transitively depends on it (via interstate-edge assignments inside
    ``inner_sdfg``). Walks back through per-lane symbols introduced by
    :class:`BypassTrivialAssignTasklets` -- e.g. ``__sym_<>`` defined by
    ``__sym_<> = idx[i]`` is tracked as dep on ``i``, not on ``j``.

    This is the post-Bypass-aware version of the direct ``free_symbols``
    check the materialiser does on its own. The walker calls this helper
    BEFORE invoking :func:`materialise_per_lane_index_tile` so the per-dim
    ONE-marker emission has the correct dep mask.

    :param gather_expr: The gather expression string (e.g. ``"__sym_x"``
        or ``"idx[i]"``).
    :param iter_vars: Tile iter-var names in order.
    :param inner_sdfg: Body NSDFG carrying interstate assignments. When
        ``None`` the function falls back to the direct iter-var membership
        check (no transitive walk).
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
    """Every :class:`Subscript` node anywhere in ``expr``. Used to
    detect data-dependent indices (``arr[idx[i]]``)."""
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
    """Return the actual :class:`sympy.Symbol` instance in ``expr`` whose
    name is ``var_name``. DaCe's ``symbol`` subclasses :class:`sympy.Symbol`
    but does not compare equal to a freshly-constructed ``sympy.Symbol``;
    :func:`sympy.Poly` treats the mismatched generator as opaque (degree
    0). Picking the real symbol out of ``expr.free_symbols`` lets the
    Poly call see the variable correctly."""
    if expr is None:
        return None
    for s in expr.free_symbols:
        if str(s) == var_name:
            return s
    return None


def _affine_coeff_for(expr: sympy.Expr, var_name: str) -> Optional[sympy.Expr]:
    """Return the coefficient of ``var_name`` in ``expr`` if ``expr`` is
    affine in ``var_name`` (i.e. ``expr = coeff * var + rest`` with
    ``rest`` having no ``var_name``). ``None`` if non-affine or
    unresolvable."""
    if expr is None:
        return None
    sym = _find_named_symbol(expr, var_name) or sympy.Symbol(var_name)
    try:
        poly = sympy.Poly(expr, sym)
    except (sympy.PolynomialError, sympy.GeneratorsError, TypeError):
        return None
    if poly.degree() > 1:
        return None
    if poly.degree() == 0:
        return sympy.Integer(0)
    # Degree 1: coefficient of the variable.
    return poly.coeff_monomial(sym)


def _affine_offset_for(expr: sympy.Expr, var_name: str) -> Optional[sympy.Expr]:
    """Constant term of ``expr`` w.r.t. ``var_name``, i.e. the ``c`` in
    ``coeff * var + c``. ``None`` if non-affine."""
    if expr is None:
        return None
    sym = _find_named_symbol(expr, var_name) or sympy.Symbol(var_name)
    try:
        poly = sympy.Poly(expr, sym)
    except (sympy.PolynomialError, sympy.GeneratorsError, TypeError):
        return None
    if poly.degree() > 1:
        return None
    # ``nth(0)`` returns the constant term.
    return poly.nth(0)


def _detect_replicate_factor(expr: sympy.Expr, var_name: str) -> Optional[int]:
    """Detect the ``int_floor(affine_in_var, k)`` / ``int_ceil(...)``
    pattern at the top of ``expr`` and return the integer divisor ``k``
    when the inner argument is affine in ``var_name``.

    Returns ``None`` when:

    * ``expr`` is not an ``int_floor`` / ``int_ceil`` call,
    * the divisor isn't a concrete positive integer,
    * the dividend isn't affine in ``var_name``.

    This is the within-dim replicate detection: the dim still walks the
    source array (so it's not a full-dim broadcast), but every ``k``
    consecutive lanes share the same source element. The codegen reads
    a ``W / k``-element contracted box and group-broadcasts.
    """
    if expr is None:
        return None
    fname = type(expr).__name__
    # DaCe has two equivalent function names for these patterns:
    # ``int_floor`` / ``int_ceil`` are the user-facing forms, and
    # ``__int_floor`` / ``__int_ceil`` are the Python-operator-derived
    # forms (produced when ``i // 4`` is parsed). Both lower to the
    # same operation.
    if fname not in ("int_floor", "int_ceil", "__int_floor", "__int_ceil"):
        return None
    if len(expr.args) != 2:
        return None
    dividend, divisor = expr.args
    # Divisor must be a positive integer OR a symbolic expression (e.g.
    # ``DV`` in ``i // DV``). Per user direction 2026-06-10: ``We can
    # enforce tile dim needs to be multiple of replicate factor``.
    # * Static factor: TileLoad construction validates ``W % k == 0`` and
    #   refuses with ValueError when violated.
    # * Symbolic factor: cannot be statically verified; the codegen emits
    #   ``__l / DV`` which is CORRECT only when ``W % DV == 0`` at runtime.
    #   User responsibility (documented in the TileLoad property): pass a
    #   divisor that divides ``W``. Tests with non-dividing symbolic divisors
    #   (e.g. ``test_div_index_symbol[3]`` with DV=3, W=8) produce incorrect
    #   results -- this is by-design refusal, not a codegen bug.
    # Refuse floats outright -- access expressions are integer-valued by
    # definition; a float divisor means an upstream pass leaked a numeric
    # type. Fall to AFFINE/GATHER rather than silently truncating to int.
    if isinstance(divisor, (sympy.Float, float)):
        return None
    try:
        k = int(divisor)
        if k <= 1:
            return None
    except (TypeError, ValueError):
        if divisor is None:
            return None
        k = divisor  # symbolic -- runtime check (W % k == 0) at codegen per 2c7b88e26.
    # Dividend must be affine in ``var_name`` (so the replication is
    # regular -- ``int_floor(idx[i], 2)`` is data-dependent and lowers
    # as GATHER, not REPLICATE).
    coeff = _affine_coeff_for(dividend, var_name)
    if coeff is None:
        return None
    return k


def _detect_modular_factor(expr: sympy.Expr, var_name: str) -> Optional[int]:
    """Detect ``(c * var + c0) % N`` -- the MODULAR per-dim pattern.

    Returns ``N`` (positive integer) when ``expr`` is a Mod / mod call
    whose RHS is a positive integer constant and whose LHS is affine
    in ``var_name``. Returns ``None`` otherwise. Tile-aligned cases
    (``N | c * W_p``) are detected later by the classifier so MODULAR
    can reduce to LINEAR.
    """
    if expr is None:
        return None
    fname = type(expr).__name__
    # SymPy's modulo is ``Mod`` (also produced by ``a % b``). DaCe's
    # ``__mod__`` overload may yield ``mod`` or ``Mod`` depending on
    # how the expression was constructed.
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
    """If ``expr`` contains exactly one Subscript whose base is an array
    name in ``inner_sdfg.arrays``, find an :class:`AccessNode` for that
    array in any state and return it. Returns ``None`` when there are
    zero or multiple subscripts, the base isn't an array, or no
    AccessNode exists."""
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

    :param subset: The :class:`Range` to classify (typically a memlet's
        ``subset`` field).
    :param iter_vars: The K tile iter-var names, in tile-lane order
        (innermost-last by convention).
    :param inner_sdfg: Optional inner SDFG used to resolve GATHER index
        AccessNodes and promoted index symbols. Pass ``None`` when the
        analysis runs outside the body context (the gather-index field is
        left empty).
    :param state: Optional access state; disambiguates multiply-assigned
        promoted index symbols by reaching definition (the frontend emits
        one ``__sym_i_plus_offset1`` per program point).
    :returns: A :class:`TileAccess` record. Always returns; never
        raises. Unrecognisable patterns degrade to GATHER (correctness
        fallback).
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

    # Resolve promoted index symbols (``__sym_i_plus_offset1`` -> ``i + offset1``)
    # once per subset so each dim's iter-var dependence is visible. Built lazily;
    # empty/unresolvable leaves the original expressions untouched. The access
    # ``state`` disambiguates multiply-assigned interstate symbols by reaching def.
    _sym_defs = _build_symbol_definition_map(inner_sdfg, state)

    for d, (lo, _hi, _stp) in enumerate(subset.ranges):
        lo_sym = _safe_sympify(lo)
        if lo_sym is not None and _sym_defs:
            lo_sym = resolve_index_expr(lo_sym, inner_sdfg, _defs=_sym_defs)
        direct = _direct_symbols(lo_sym) if lo_sym is not None else set()
        direct_tile_vars = direct & iter_var_set

        # Stop 1: any tile iter-var inside a Subscript -> GATHER dim.
        # Recompute the symbols nested inside Subscripts to detect this.
        gather_dim = False
        if lo_sym is not None:
            for sub in _gather_subscripts(lo_sym):
                # A Subscript stores (container, idx0, idx1, ...) in args.
                # Iter-var appears nested if it shows up in any subscript
                # argument's free symbols.
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

        # Section 4.2 join rule: any non-iter-var symbol in the expression that is
        # transitively tile-dependent (defined by an interstate edge whose RHS touches
        # a tile iter-var) forces GATHER. Catches the `a[2*sym + 1]` (sym <- i + 3)
        # pattern that would otherwise look CONSTANT to the direct-symbol check.
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

        # Stop 2: no tile iter-var as direct symbol -> BROADCAST dim
        # (replicate_factor = None, meaning all W lanes of the dim
        # share one element).
        if not direct_tile_vars:
            per_dim_kind.append(PerDimKind.BROADCAST)
            dim_strides.append(0)
            dim_iter_var.append(None)
            gather_index_per_dim.append(None)
            dim_offset.append(_safe_sympify(lo))
            replicate_factor_per_dim.append(None)
            dim_to_canonical_iter_var.append(None)
            continue

        # Stop 3: exactly one tile iter-var directly -> STRUCTURED_1,
        # REPLICATE, or AFFINE.
        if len(direct_tile_vars) == 1:
            tvar = next(iter(direct_tile_vars))
            # Stop 3a: ``int_floor(c*tvar + c0, k)`` / ``int_ceil(...)``
            # -> REPLICATE with factor k. Detected BEFORE the affine
            # check so the function call doesn't fall through to AFFINE.
            replicate_k = _detect_replicate_factor(lo_sym, tvar)
            if replicate_k is not None:
                per_dim_kind.append(PerDimKind.REPLICATE)
                dim_strides.append(1)  # contracted-box stride = 1
                dim_iter_var.append(tvar)
                gather_index_per_dim.append(None)
                dim_offset.append(None)
                replicate_factor_per_dim.append(replicate_k)
                iter_var_in_dim[tvar].append(d)
                dim_to_canonical_iter_var.append(list(iter_vars).index(tvar))
                continue
            # Stop 3b: ``(c * tvar + c0) % N`` -> MODULAR. The codegen
            # falls back to GATHER for the general case; the tile-
            # aligned reduction to LINEAR is a future optimisation
            # (TILIFICATION_TRANSFORMATION_DESIGN.md section 4.2).
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
                # G6 (design section 4.2 join rule): the AFFINE coefficient must be tile-
                # independent. `a[N*i]` with N outer-constant is AFFINE stride N; with N tile-
                # dependent (defined by an interstate edge whose RHS touches a tile iter-var)
                # the access is GATHER. Forces a GATHER fallback for unrecognised symbolic
                # coefficients whose tile-independence we cannot prove.
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
                # Affine in one iter-var. Coerce to int when possible
                # (the emitter prefers concrete strides). When the coefficient
                # is symbolic and tile-independent (the ``tile_dep_coeff``
                # check above ruled out tile-dep), keep the sympy expression
                # so the lib node's ListProperty (element_type=pystr_to_symbolic)
                # serialises it and the codegen inlines it as a C++ variable.
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
            # Non-affine and not int_floor/int_ceil (e.g. ``i**2``):
            # fall through to AFFINE without an int stride; the emitter
            # degrades to GATHER with the per-lane expression.
            per_dim_kind.append(PerDimKind.AFFINE)
            dim_strides.append(None)
            dim_iter_var.append(tvar)
            gather_index_per_dim.append(None)
            dim_offset.append(None)
            replicate_factor_per_dim.append(None)
            iter_var_in_dim[tvar].append(d)
            dim_to_canonical_iter_var.append(list(iter_vars).index(tvar))
            continue

        # Stop 4: multiple tile iter-vars in the dim. AFFINE only if the expression
        # is JOINTLY affine -- each tile var's coefficient must be tile-independent
        # (e.g. ``i + j``, a diagonal access). A cross-term like ``i*j`` (or a
        # resolved ``syma*i`` with ``syma <- j``) makes one var's coefficient depend
        # on another tile var -> non-linear in the tile vars -> GATHER (G6 join rule,
        # design section 4.2). Mirrors the single-var coefficient check at Stop 3.
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
        # No single representative iter-var for the dim; report the
        # first one alphabetically for consistency.
        rep = sorted(direct_tile_vars)[0]
        dim_iter_var.append(rep)
        gather_index_per_dim.append(None)
        dim_offset.append(None)
        replicate_factor_per_dim.append(None)
        for tv in direct_tile_vars:
            iter_var_in_dim[tv].append(d)
        dim_to_canonical_iter_var.append(list(iter_vars).index(rep))

    # Whole-subset kind: strongest per-dim kind. MODULAR / REPLICATE
    # share the STRUCTURED bucket (both perfectly regular; codegen
    # picks the right intrinsic from the per-dim records).
    kinds = set(per_dim_kind)
    if PerDimKind.GATHER in kinds or PerDimKind.MODULAR in kinds:
        # MODULAR currently routes to GATHER until the tile-aligned
        # reduction lands (section 4.2 future work).
        kind = TileAccessKind.GATHER
    elif PerDimKind.AFFINE in kinds:
        kind = TileAccessKind.AFFINE
    elif kinds & {PerDimKind.LINEAR, PerDimKind.REPLICATE}:
        kind = TileAccessKind.STRUCTURED
    else:
        kind = TileAccessKind.BROADCAST

    # Diagonal: any iter-var spans >= 2 dims.
    diagonal = {v: tuple(dims) for v, dims in iter_var_in_dim.items() if len(dims) >= 2}

    # Transpose: STRUCTURED with iter-vars in non-canonical order.
    # Canonical order: dim ``d`` carries ``iter_vars[d]``.
    transpose: Optional[Tuple[int, ...]] = None
    if kind == TileAccessKind.STRUCTURED and len(per_dim_kind) == len(iter_vars):
        # Build the permutation from each dim's iter-var index.
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
