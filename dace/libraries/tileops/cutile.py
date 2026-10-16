# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared cuTile dtype and launch-grid expressions."""

from typing import List, Sequence


def ct_dtype_name(dtype) -> str:
    """``cuda.tile`` dtype attribute name for a DaCe typeclass.

    ``ct`` exposes numpy-style names (``ct.float64``, ``ct.int32``, ...);
    only ``bool`` is spelled ``bool_``.

    :param dtype: A :class:`dace.dtypes.typeclass`.
    :returns: The attribute name on the ``cuda.tile`` module.
    """
    name = dtype.as_numpy_dtype().name
    return "bool_" if name == "bool" else name


def _enclosing_cutile_map(node, parent_state, parent_sdfg):
    """Return the ``MapEntry`` of the enclosing ``CuTile``-scheduled map.

    Walks the in-state scope chain of ``node`` and, when ``node`` lives in a
    NestedSDFG body, continues up through ``parent_nsdfg_node`` to the owning
    state.

    :param node: The tile-op library node being expanded.
    :param parent_state: The state that owns ``node``.
    :param parent_sdfg: The SDFG that owns ``parent_state``.
    :returns: The enclosing CuTile ``MapEntry``, or ``None`` if there is none.
    """
    from dace import dtypes as _dtypes
    from dace.sdfg import nodes as _nodes

    cur_node = node
    cur_state = parent_state
    cur_sdfg = parent_sdfg
    while cur_state is not None:
        scope = cur_state.scope_dict()
        m = scope.get(cur_node)
        while m is not None:
            if isinstance(m, _nodes.MapEntry) and m.map.schedule == _dtypes.ScheduleType.CuTile:
                return m
            m = scope.get(m)
        nsdfg_node = cur_sdfg.parent_nsdfg_node if cur_sdfg is not None else None
        if nsdfg_node is None or cur_sdfg.parent is None:
            break
        cur_node = nsdfg_node
        cur_state = cur_sdfg.parent
        cur_sdfg = cur_state.sdfg
    return None


def cutile_grid_size_exprs(cutile_map_entry) -> List[str]:
    """Per-dimension grid size (tile count) expressions for a CuTile map.

    Each dimension's grid size is the number of tiles
    ``int_ceil(extent, step)`` where ``extent = end - start + 1`` (inclusive
    map-range end). Emitted as a structural integer ceil-division to keep both
    operands integral for the Python/cuTile backend (mirrors
    ``cutile_target._grid_exprs_from_map_entry``).

    :param cutile_map_entry: The enclosing CuTile ``MapEntry``.
    :returns: List of grid-size expression strings, one per map dimension.
    """
    from dace.symbolic import symstr

    exprs: List[str] = []
    for start, end, step in cutile_map_entry.map.range:
        extent_s = symstr(end - start + 1)
        step_s = symstr(step)
        exprs.append(extent_s if step_s == "1" else f"int_ceil({extent_s}, {step_s})")
    return exprs


def cutile_launch_grid_dims(grid_size_exprs: Sequence[str]) -> List[str]:
    """Fold per-dimension grid sizes onto the <=3-axis cuTile launch grid.

    :param grid_size_exprs: Per-dimension grid-size expressions in map order
        (dimension 0 outermost), length ``M``.
    :returns: List of at most three launch-grid dimension expressions passed to
        ``ct.launch``. For ``M <= 3`` the sizes are padded with ``"1"`` to a
        3-tuple; for ``M > 3`` the leading ``M-2`` dimensions are multiplied
        onto axis 0.
    """
    num_dims = len(grid_size_exprs)
    if num_dims <= 3:
        return (list(grid_size_exprs) + ["1", "1", "1"])[:3]
    folded = grid_size_exprs[: num_dims - 2]
    axis0 = " * ".join(f"({g})" for g in folded)
    return [axis0, grid_size_exprs[num_dims - 2], grid_size_exprs[num_dims - 1]]


def cutile_bid_expr(map_axis: int, num_map_dims: int, grid_size_exprs: Sequence[str]) -> str:
    """Block-ID expression for a single CuTile map grid axis (folding-aware).

    Returns the Python expression that evaluates, inside a cuTile kernel, to
    the block index of map dimension ``map_axis``. For ``num_map_dims <= 3``
    this is simply ``ct.bid(map_axis)``. For ``num_map_dims > 3`` the leading
    ``M-2`` dimensions are recovered from ``ct.bid(0)`` via div/mod and the two
    innermost dimensions read ``ct.bid(1)`` / ``ct.bid(2)`` (see the module
    header for the canonical layout).

    :param map_axis: The map dimension index (grid axis) to recover.
    :param num_map_dims: Total number of CuTile map dimensions (``M``).
    :param grid_size_exprs: Per-dimension grid-size expressions (length ``M``).
    :returns: A Python expression string yielding the block ID for ``map_axis``.
    """
    if num_map_dims <= 3:
        return f"ct.bid({map_axis})"
    if map_axis == num_map_dims - 2:
        return "ct.bid(1)"
    if map_axis == num_map_dims - 1:
        return "ct.bid(2)"
    # Folded leading dimension (0 <= map_axis <= M-3), recovered from ct.bid(0).
    g_axis = grid_size_exprs[map_axis]
    inner = grid_size_exprs[map_axis + 1 : num_map_dims - 2]  # dims strictly inside, up to M-3
    if not inner:  # innermost folded dim (map_axis == M-3)
        return f"(ct.bid(0) % ({g_axis}))"
    divisor = " * ".join(f"({g})" for g in inner)
    if map_axis == 0:  # outermost folded dim: quotient is already < g_axis
        return f"(ct.bid(0) // ({divisor}))"
    return f"((ct.bid(0) // ({divisor})) % ({g_axis}))"


def cutile_bid_lines(node, parent_state, parent_sdfg, axes: Sequence[int]) -> List[str]:
    """Emit ``__pid{k} = <block-id>`` bindings for a tile op's grid axes.

    Resolves each tile-op tile dim's block ID against the enclosing CuTile map,
    applying the >3-D grid fold (see the module header) so the axis indices
    stay within the runtime's 3-axis cap. When there is no enclosing CuTile map
    or it has at most three dimensions, the direct ``ct.bid(axis)`` form is used
    (identical to the pre-folding behavior).

    :param node: The tile-op library node being expanded.
    :param parent_state: The state that owns ``node``.
    :param parent_sdfg: The SDFG that owns ``parent_state``.
    :param axes: Per-tile-dim grid axis (enclosing-map dimension index), or
        ``None`` for a dim no grid axis drives (its base is fully carried by
        the recovered offset; see :func:`cutile_tile_dim_bids`) — bound as
        ``__pid{k} = 0``.
    :returns: List of ``__pid{k} = ...`` statement strings, one per tile dim.
    """
    m = _enclosing_cutile_map(node, parent_state, parent_sdfg)
    if m is None or len(m.map.range) <= 3:
        return [f"__pid{k} = {'0' if axes[k] is None else f'ct.bid({axes[k]})'}" for k in range(len(axes))]
    num_map_dims = len(m.map.range)
    grid = cutile_grid_size_exprs(m)
    return [
        f"__pid{k} = {'0' if axes[k] is None else cutile_bid_expr(axes[k], num_map_dims, grid)}"
        for k in range(len(axes))
    ]


def cutile_grid_dim_offset(node, parent_state, parent_sdfg, K: int) -> int:
    """Number of enclosing CuTile-map grid dims that precede the K tile dims.

    The cuTile codegen binds ``__pid{d} = ct.bid(d)`` positionally for each
    dim of the enclosing ``CuTile`` map (``map.range[d]`` -> grid axis ``d``).
    A tile op runs inside the body of the innermost (tiled) loops; when the map
    also carries OUTER, non-tiled point dims (e.g. an ICON block loop ``jb``),
    those are the LEADING grid axes, so a tile op spanning ALL ``K`` tiled dims
    occupies grid axes ``offset .. offset + K - 1`` with
    ``offset = len(map.range) - K``.

    This is correct only for a tile op whose K dims ARE the innermost K (the
    common case: the iteration mask, the main data load/store). For a tile op
    that tiles a STRICT SUBSET of the tiled dims (e.g. a 1-D gather index tile
    that depends on a non-innermost loop) use :func:`cutile_tile_dim_bids`,
    which resolves each tile dim's grid axis individually.

    :param node: The tile-op library node being expanded.
    :param parent_state: The state that owns ``node``.
    :param parent_sdfg: The SDFG that owns ``parent_state``.
    :param K: The tile-op's tile-dim count (``len(widths)``).
    :returns: The grid-dim offset, or 0 when there is no enclosing CuTile map
        or it has exactly K dims (the common fully-tiled case, e.g. cuTile V1).
    """
    m = _enclosing_cutile_map(node, parent_state, parent_sdfg)
    return max(0, len(m.map.range) - K) if m is not None else 0


def cutile_tile_dim_bids(
    node, parent_state, parent_sdfg, used_dimensions: Sequence[int], src_begins: Sequence[str], K: int
) -> list:
    """Resolve the ``ct.bid`` grid axis for each of a load/store's K tile dims.

    The grid axis for tile dim ``d`` is the position, in the enclosing CuTile
    map's ``params``, of the iteration variable that tile dim ``d`` walks. That
    variable is the memlet begin of the source/dest array dim the tile dim maps
    to (``src_begins[used_dimensions[d]]`` -- e.g. ``A[jb, jk:jk+8]`` gives
    begin ``jk`` for the tiled dim). Matching it to ``map.params`` yields the
    exact grid axis even when the tile dim is NOT the innermost loop (the
    positional :func:`cutile_grid_dim_offset` is wrong in that case).

    Falls back to the trailing-K offset for any dim whose variable cannot be
    matched (e.g. a constant/broadcast dim, whose ``__pid`` is unused anyway),
    and entirely when there is no resolvable enclosing CuTile map.

    :param node: The tile-op library node being expanded.
    :param parent_state: The state that owns ``node``.
    :param parent_sdfg: The SDFG that owns ``parent_state``.
    :param used_dimensions: Per-tile-dim source/dest array dim index.
    :param src_begins: Per-source-dim memlet begin expression (as strings).
    :param K: The tile-op's tile-dim count.
    :returns: List of ``K`` entries: the grid-axis index, or ``None`` for a
        dim whose begin is driven by a non-grid variable (e.g. a sequential
        inner tile loop) — no ``__pid*W`` term applies there.
    """
    import dace.symbolic as _sym

    m = _enclosing_cutile_map(node, parent_state, parent_sdfg)
    offset = max(0, len(m.map.range) - K) if m is not None else 0
    fallback = [offset + d for d in range(K)]
    if m is None:
        return fallback
    params = [str(p) for p in m.map.params]
    bids = []
    for d in range(K):
        sd = used_dimensions[d] if d < len(used_dimensions) else None
        pos = None
        has_symbols = False
        if sd is not None and sd < len(src_begins):
            try:
                syms = {str(s) for s in _sym.pystr_to_symbolic(str(src_begins[sd])).free_symbols}
            except Exception:  # noqa: BLE001 - non-symbolic begin -> use the raw string
                syms = {str(src_begins[sd])}
            has_symbols = any(not s.lstrip("-").isdigit() for s in syms)
            for i, p in enumerate(params):
                if p in syms:
                    pos = i
                    break
        if pos is None and has_symbols:
            # The begin is driven by a variable that is NOT a grid param
            # (e.g. a tile-strided SEQUENTIAL inner map inside the kernel,
            # from a dependent range like ``j = i+1 : N : W``).  No block id
            # advances this dim -- the full base is carried by the recovered
            # offset (:func:`cutile_tile_dim_offsets` leaves non-grid
            # variables in place) -- so the ``__pid*W`` term must vanish:
            # signal "no grid axis" and let :func:`cutile_bid_lines` bind
            # ``__pid{d} = 0``.  The positional fallback here would read a
            # WRONG grid axis (an outer point dim) and silently miscompile.
            bids.append(None)
        else:
            bids.append(pos if pos is not None else fallback[d])
    return bids


def cutile_tile_dim_offsets(
    node, parent_state, parent_sdfg, used_dimensions: Sequence[int], begins: Sequence[str], K: int
) -> list:
    """Return the constant per-tile-dim element offset carried by the memlet begin.

    A tiled load/store's per-dim memlet begin has the form ``<iter-var> + c`` --
    e.g. ``A[1:-1]`` yields begin ``__i0 + 1`` for the tiled dim, where ``__i0``
    is the block's element start (``__pid_k * W``) and ``c = 1`` is the constant
    offset of the slice. The block-aligned ``ct.load`` / ``ct.arange + __pid*W``
    index only reconstructs ``__i0`` and drops ``c``, so an offset slice
    (``A[1:-1]``, ``A[2:]``) reads/writes one (or more) elements too low. This
    helper recovers ``c`` for each tile dim by substituting every enclosing
    CuTile-map iteration variable in the begin with its range START -- what
    remains is the part of the address the block id does NOT advance and must
    be added back to the per-lane element index. For canonical 0-start maps the
    start substitution degenerates to the plain constant slice offset; for a
    nonzero-start map (e.g. the vectorizer's remainder map over
    ``4*int_floor(N,4):N``) the start itself becomes part of the offset, since
    ``__pid * W`` counts blocks relative to the range start.

    Map parameters whose range provably has a single iteration (e.g. a
    single-tile inner loop ``0:4:4``) are substituted with their constant
    value first — that is exact, not a guess. After that the begin must be
    affine in AT MOST ONE enclosing-map parameter (the dim's own iteration
    variable, the same one :func:`cutile_tile_dim_bids` resolves the block id
    from). A begin coupling several multi-iteration map parameters (e.g.
    ``A[__i0 + __i1]``) cannot be reconstructed from a single block id and
    raises ``NotImplementedError`` instead of silently producing a wrong
    offset.

    :param node: The tile-op library node being expanded.
    :param parent_state: The state that owns ``node``.
    :param parent_sdfg: The SDFG that owns ``parent_state``.
    :param used_dimensions: Per-tile-dim source/dest array dim index.
    :param begins: Per-source/dest-dim memlet begin expression (as strings).
    :param K: The tile-op's tile-dim count.
    :returns: List of ``K`` symbolic offsets (usually integers; ``0`` when the
        slice is anchored at the block-aligned start).
    :raises NotImplementedError: If a begin cannot be parsed or couples more
        than one enclosing-map parameter.
    """
    import dace.symbolic as _sym

    m = _enclosing_cutile_map(node, parent_state, parent_sdfg)
    params = []
    starts = []
    single_iter_subs = {}
    if m is not None:
        for p, rng in zip(m.map.params, m.map.range):
            try:
                p_sym = _sym.pystr_to_symbolic(str(p))
            except Exception:  # noqa: BLE001 - skip unparseable params
                params.append(None)
                starts.append(rng[0])
                continue
            params.append(p_sym)
            start, end, step = rng[0], rng[1], rng[2]
            starts.append(start)
            # A range with provably one iteration (extent <= step) pins the
            # parameter to its start value; substituting it is exact.
            try:
                if bool(_sym.simplify(end - start + 1 - step) <= 0):
                    single_iter_subs[p_sym] = start
            except (TypeError, ValueError):
                pass  # symbolic, not provably single-iteration
    param_set = {p for p in params if p is not None}
    offsets = []
    for d in range(K):
        sd = used_dimensions[d] if d < len(used_dimensions) else None
        if sd is None or sd >= len(begins):
            offsets.append(0)
            continue
        try:
            begin = _sym.pystr_to_symbolic(str(begins[sd]))
        except Exception as ex:  # noqa: BLE001 - unusable begin must fail loudly
            raise NotImplementedError(
                f"{node.label}: cannot parse memlet begin {begins[sd]!r} for tile dim {d} "
                f"(array dim {sd}); refusing to guess the element offset."
            ) from ex
        if single_iter_subs:
            begin = begin.subs(single_iter_subs)
        own = sorted((p for p in param_set if p in begin.free_symbols), key=str)
        if len(own) > 1:
            raise NotImplementedError(
                f"{node.label}: memlet begin {begins[sd]!r} for tile dim {d} (array dim {sd}) couples "
                f"multiple enclosing CuTile-map parameters {[str(p) for p in own]}; the block-aligned "
                f"index reconstruction supports at most one iteration variable per dim. Such accesses "
                f"should have been routed to the gather/scatter path by the detection passes."
            )
        if own:
            own_p = own[0]
            # ``__pid * W`` counts blocks relative to the range start, so the
            # begin evaluated AT the start is exactly the part the block id
            # does not advance (0-start maps degenerate to the slice constant).
            start = starts[params.index(own_p)]
            begin = begin.subs(own_p, start)
        remaining = _sym.simplify(begin)
        foreign = param_set & set(remaining.free_symbols)
        if foreign:
            raise NotImplementedError(
                f"{node.label}: element offset {remaining} for tile dim {d} (array dim {sd}) still "
                f"contains enclosing CuTile-map parameter(s) {[str(p) for p in sorted(foreign, key=str)]} "
                f"after removing the dim's own iteration variable; refusing to silently zero them."
            )
        offsets.append(remaining)
    return offsets


def cutile_offset_is_nonzero(offset) -> bool:
    """Whether a tile-dim element offset is provably non-zero.

    :param offset: A symbolic / numeric offset from :func:`cutile_tile_dim_offsets`.
    :returns: ``True`` if the offset does not simplify to ``0``.
    """
    import dace.symbolic as _sym

    try:
        return bool(_sym.simplify(offset) != 0)
    except Exception:  # noqa: BLE001 - be conservative: treat as offset present
        return offset != 0


def cutile_offset_block_shift(offset, width: int):
    """Block-index shift for a tile-dim element offset, if provably block-aligned.

    When the element offset is a provably nonnegative multiple of the tile
    width, an aligned ``ct.load`` / ``ct.store`` stays valid with the block
    index shifted by ``offset // width`` (fast path instead of per-element
    gather/scatter). Nonnegativity guards the undefined-behavior case of a
    tile lying entirely before the array start.

    :param offset: A symbolic / numeric offset from :func:`cutile_tile_dim_offsets`.
    :param width: The tile width of that dim.
    :returns: The symbolic block shift ``offset // width``, or ``None`` when
        divisibility / nonnegativity is not provable (caller must take the
        gather/scatter path).
    """
    import dace.symbolic as _sym

    try:
        q = _sym.simplify(_sym.pystr_to_symbolic(str(offset)) / int(width))
    except Exception:  # noqa: BLE001 - unprovable -> no fast path
        return None
    if getattr(q, "is_integer", None) and getattr(q, "is_nonnegative", None):
        return q
    return None
