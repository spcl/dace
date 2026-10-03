# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The alignment proof behind the widened half-precision loads and stores of the CUDA backend.

Two fp16 elements make one 32-bit word, the minimum vector and the minimum aligned load, so an access the proof cannot
place on an even element has nothing between ``half2`` and a per-element ``LDG.E.U16``. The proof reads the memlet
offset of an access, the ranges of the enclosing maps and the divisibility guarantees the vectorizer records, and
yields the alignment and the shift of the access from the aligned window the widened word covers.
"""
import warnings
from collections.abc import Iterable

import dace
from dace.data import Data
from dace.memlet import Memlet
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.nodes import Node, Tasklet
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import SDFGState
from dace.symbolic import SymbolicType
from dace.optionals import required
from dace.sdfg.narrowing import as_basic, as_expr, as_map_entry, as_range

#: Suffix of the label of the map a tile-remainder split leaves fully in bounds. The vectorizer writes it and the proof
#: below reads it: a tiled dim of such a map has an extent that is a whole number of tiles.
TILE_MAIN_MARKER = "__tile_main"

#: Label of the state holding the vectorizer's runtime divisibility guards. That state IS the record the proof reads
#: back (:func:`guarded_stride_divisors`), so a fact can never outlive the abort-on-violation check that establishes it.
TILE_GUARD_STATE_LABEL = "tile_even_range_check"

#: Tasklet-label prefix of a guarded stride-parity fact, spelled ``<prefix><symbol>_<modulus>``. The symbol name is an
#: identifier and the modulus an integer, so an ``rpartition('_')`` reads the pair back exactly.
STRIDE_GUARD_PREFIX = "tile_stride_div_"

# Byte alignment the allocator guarantees for the base of an array, per storage class. The GPU
# figure is the gpuMalloc/gpuMallocAsync contract (256 B). Anything else is unknown and gets no
# assumption. Register is NOT here on purpose -- see :func:`base_align_bytes`.
BASE_ALIGN_BYTES = {
    dace.dtypes.StorageType.GPU_Global: 256,
    dace.dtypes.StorageType.CPU_Pinned: 256,
}


def base_align_bytes(arr: Data) -> int:
    """Bytes the allocation guarantees for ``arr``'s base address.

    A Register array is a declared local, so its guarantee is whatever the descriptor asked the
    codegen to emit -- ``alignment == 0`` means no attribute at all, i.e. only the element's natural
    alignment. Reading the descriptor rather than assuming the old blanket ``DACE_ALIGN(64)`` is
    what keeps the widened reinterpret below a PROVEN fact: over-promising here is a misaligned
    device access, which faults, not merely slow code.
    """
    if arr.storage == dace.dtypes.StorageType.Register:
        return arr.alignment if arr.alignment > 0 else arr.dtype.bytes
    return BASE_ALIGN_BYTES.get(arr.storage, 0)


def enclosing_param_ranges(
    node: Node, parent_state: SDFGState, parent_sdfg: SDFG
) -> tuple[dict[str, tuple[SymbolicType, SymbolicType, SymbolicType]], list[tuple[SymbolicType, int]]]:
    """``({param: (start, end, step)}, [(extent, width)])`` for the map scopes enclosing ``node``.

    Walks out through nested-SDFG boundaries, rewriting each level's params through the
    ``symbol_mapping`` that connects them, so a param of an OUTER map is reported under the name
    the INNER offset expression uses. A param whose range cannot be followed is simply absent,
    which leaves it un-substituted and therefore not provably divisible.

    The second list is the extents the vectorizer GUARANTEES divisible: a ``__tile_main`` map's
    tiled dim either had its end tightened to a whole number of tiles (the peeled-remainder path)
    or is the caller's ``assume_even`` promise, which that pass raises on when provably violated
    and otherwise guards with a host-side abort. Nothing here re-derives the guarantee; it reads
    the marker the guarantee is recorded under.
    """
    ranges, even, cur_node, state, sdfg = {}, [], node, parent_state, parent_sdfg
    while state is not None:
        entry = state.entry_node(cur_node)
        while entry is not None:
            for param, rng in zip(as_map_entry(entry).map.params, as_map_entry(entry).map.range):
                ranges.setdefault(str(param), (rng[0], rng[1], rng[2]))
                step = dace.symbolic.simplify(rng[2])
                # Only a TILED dim (step == its width > 1) carries a guarantee; a step-1 dim's
                # extent divides 1 and says nothing.
                if as_map_entry(entry).map.label.endswith(TILE_MAIN_MARKER) and as_basic(step).is_Integer and step > 1:
                    even.append((rng[1] - rng[0] + 1, int(as_expr(step))))
            entry = state.entry_node(entry)
        nsdfg_node = sdfg.parent_nsdfg_node
        if nsdfg_node is None:
            break
        # Names change across the boundary: an inner symbol equal to a bare outer symbol can keep
        # that outer map's range, anything richer is not followed.
        renamed = {}
        for inner, outer in nsdfg_node.symbol_mapping.items():
            outer_sym = str(dace.symbolic.pystr_to_symbolic(outer))
            if outer_sym in ranges and str(inner) != outer_sym:
                renamed[str(inner)] = ranges[outer_sym]
        ranges.update(renamed)
        cur_node, state = nsdfg_node, sdfg.parent
        sdfg = sdfg.parent_sdfg
        if not isinstance(sdfg, SDFG):
            break
    return ranges, even


def even_extent_substitutions(even: list[tuple[SymbolicType, int]]) -> dict[SymbolicType, SymbolicType]:
    """``{symbol: width*t - b}`` for every guaranteed-divisible extent of the form ``symbol + b``.

    This is what makes a SYMBOLIC row stride usable. An ``N``-column array tiled over ``1:N-1``
    has extent ``N - 2``, and a width-2 tile guarantees that is even, so ``N = 2t + 2`` -- which
    makes ``N*i`` provably even and a ``A[i, j+1]`` offset provably ODD instead of unknown. Without
    it every multi-dim access on a symbolic shape stays on the per-element path.

    ``t`` is non-negative because a tiled map runs at least one full tile. Only a bare ``symbol +
    constant`` extent is solved: a richer one (``N*M - 1``) has no single symbol to pin and is left
    alone, which simply leaves the offset undecidable.
    """
    subs = {}
    # Widest first: two tiled dims can constrain the same symbol, and the wider one is stronger.
    for n, (extent, width) in enumerate(sorted(even, key=lambda ew: -ew[1])):
        extent = dace.symbolic.simplify(extent)
        free = list(as_basic(extent).free_symbols)
        if len(free) != 1 or free[0] in subs:
            continue
        sym = free[0]
        rest = dace.symbolic.simplify(extent - sym)
        if not as_basic(rest).is_Integer:
            continue
        t = dace.symbolic.symbol(f"__dace_align_t{n}", nonnegative=True, integer=True)
        subs[sym] = width * t - int(as_expr(rest))
    return subs


def guarded_stride_divisors(sdfg: SDFG) -> dict[str, int]:
    """``{symbol: modulus}`` for every stride-divisibility fact ``sdfg`` CHECKS before it runs.

    Read off the guard tasklets themselves, not off a parallel bookkeeping structure: the fact and the abort that
    enforces it are the same node, so the proof can never widen an access on a promise nothing tests. No guard state
    gives an empty dict, and the proof stays on whatever it can show unaided (a constant stride), which is the
    per-element path for a symbolic one.

    :param sdfg: SDFG to read the guards of (this level only, not nested ones).
    :returns: Symbol name -> the modulus its value is checked to be a nonzero multiple of.
    """
    facts: dict[str, int] = {}
    for state in sdfg.states():
        # ``add_state_before`` uniquifies a duplicate label, hence the prefix test.
        if not state.label.startswith(TILE_GUARD_STATE_LABEL):
            continue
        for node in state.nodes():
            if not isinstance(node, Tasklet) or not node.label.startswith(STRIDE_GUARD_PREFIX):
                continue
            name, separator, modulus = node.label[len(STRIDE_GUARD_PREFIX):].rpartition("_")
            if separator and name and modulus.isdigit():
                facts[name] = int(modulus)
    return facts


def stride_divisor_facts(parent_sdfg: SDFG) -> dict[str, int]:
    """``{symbol: modulus}`` for the stride-parity facts guarded anywhere up ``parent_sdfg``'s chain.

    The facts are read off the guard tasklets ``SplitMapForTileRemainder`` emitted, so every entry
    here is backed by a host-side ``abort`` that runs before the kernel: the proof below widens an
    access only on divisibility the program itself verifies. No guard -> no entry -> the symbolic
    stride stays undecidable and the caller keeps the per-element path.

    Guards live on the SDFG that owns the tiled MAP, which is usually an ancestor of the one that
    owns the tile ops, hence the walk. A name a NestedSDFG boundary REBINDS (``symbol_mapping``
    sends a different outer expression to it) means something else on the two sides, so it is
    refused from there outward rather than translated -- a wrong translation would be exactly the
    unchecked promise this is built to avoid.
    """
    facts, sdfg, rebound = {}, parent_sdfg, set()
    while isinstance(sdfg, SDFG):
        for name, modulus in guarded_stride_divisors(sdfg).items():
            if name not in rebound:
                facts.setdefault(name, modulus)
        nsdfg_node = sdfg.parent_nsdfg_node
        if nsdfg_node is None:
            break
        for inner, outer in nsdfg_node.symbol_mapping.items():
            outer_name = str(dace.symbolic.pystr_to_symbolic(outer))
            if outer_name != str(inner):
                rebound.add(str(inner))
                rebound.add(outer_name)
        sdfg = sdfg.parent_sdfg
    return facts


def base_offset_is_visible(sdfg: SDFG, name: str) -> bool:
    """True if the memlet offset seen INSIDE ``sdfg`` is the whole offset from the allocation.

    A nested SDFG is handed a pointer that its enclosing memlet may already have shifted -- the
    body gets ``&A[N*i]`` and then indexes it by ``j`` alone. Reading only the inner subset would
    then miss the ``N*i`` and claim an alignment the pointer does not have, so every enclosing
    connection for this array has to start at element 0 for the inner view to be the whole story.
    """
    cur_sdfg, cur_name = sdfg, name
    while cur_sdfg is not None:
        nsdfg_node = cur_sdfg.parent_nsdfg_node
        state = cur_sdfg.parent
        if nsdfg_node is None or state is None:
            return True
        edges = ([e for e in state.in_edges(nsdfg_node) if e.dst_conn == cur_name] +
                 [e for e in state.out_edges(nsdfg_node) if e.src_conn == cur_name])
        if not edges:
            return False
        outer_names = set()
        for e in edges:
            if e.data.subset is None:
                return False
            if any(dace.symbolic.simplify(s) != 0 for s in as_range(e.data.subset).min_element()):
                return False
            outer_names.add(e.data.data)
        if len(outer_names) != 1:
            return False
        cur_sdfg, cur_name = cur_sdfg.parent_sdfg, outer_names.pop()
    return True


def linear_base_offset(node: Node, parent_state: SDFGState, parent_sdfg: SDFG,
                       edge: MultiConnectorEdge[Memlet]) -> tuple[SymbolicType, SymbolicType, SymbolicType] | None:
    """``(offset_at_tile_base, offset_at_param_ends, allocated_elements)``, or ``None``.

    The first expression has every enclosing map param replaced by ``start + step*k`` (``k`` a
    fresh non-negative symbol), which is what makes a residue modulo a chunk decidable. The second
    puts every param at the END of its range, which upper-bounds the offset -- used to prove a
    widened window stays inside the allocation. A param whose coefficient is not provably
    non-negative makes the end substitution no bound at all, so the whole thing is refused.

    All three carry the guaranteed-divisible-extent rewrite (:func:`even_extent_substitutions`),
    which is what a symbolic shape needs to decide anything at all -- and which the size has to
    carry too, or the bound is compared against the wrong allocation.
    """
    arr = parent_sdfg.arrays[required(edge.data.data)]
    # A view starts wherever its source says, and a start_offset shifts the base out from under
    # the allocator's guarantee.
    if isinstance(arr, dace.data.View) or arr.start_offset != 0:
        return None
    if not base_offset_is_visible(parent_sdfg, required(edge.data.data)):
        return None
    subset = edge.data.subset
    if subset is None:
        return None
    offset = dace.symbolic.pystr_to_symbolic(sum(s * st for s, st in zip(as_range(subset).min_element(), arr.strides)))
    ranges, even = enclosing_param_ranges(node, parent_state, parent_sdfg)
    at_base, at_end = offset, offset
    # Match params against the symbols the offset ACTUALLY holds, by name. A freshly built
    # ``pystr_to_symbolic(param)`` carries the default int32 dtype while the subset's symbol is
    # int64, and a DaCe symbol hashes on its dtype -- so ``sym in offset.free_symbols`` is silently
    # False and every param stays un-substituted, which leaves the residue undecidable for reasons
    # that have nothing to do with the shape.
    by_name = {str(s): s for s in offset.free_symbols}
    size = dace.symbolic.pystr_to_symbolic(arr.total_size)
    # Built BEFORE the coefficient test, not after: a stride of ``N - 2`` (the shape a transient
    # holding a stencil's interior gets) is not provably non-negative as written, while the very
    # extent fact that makes its parity decidable, ``N = 2t + 2``, also makes it ``2t``. Testing
    # the raw coefficient refused every such transient for a reason the substitution answers.
    subs = even_extent_substitutions(even)
    add_stride_substitutions(subs, stride_divisor_facts(parent_sdfg), (offset, size))
    for idx, (param, (start, end, step)) in enumerate(ranges.items()):
        sym = by_name.get(param)
        if sym is None:
            continue
        coeff = dace.symbolic.simplify(dace.symbolic.equalize_symbol(as_expr(offset).diff(sym).subs(subs)))
        if as_basic(coeff).is_nonnegative is not True:
            return None
        k = dace.symbolic.symbol(f"__dace_align_k{idx}", nonnegative=True, integer=True)
        start, end, step = (dace.symbolic.pystr_to_symbolic(x) for x in (start, end, step))
        at_base = at_base.subs(sym, start + step * k)
        at_end = at_end.subs(sym, end)
    return tuple(e.subs(subs) for e in (at_base, at_end, size))


def add_stride_substitutions(subs: dict[SymbolicType, SymbolicType], facts: dict[str, int],
                             exprs: Iterable[SymbolicType]) -> None:
    """Fold the guarded stride-parity facts into ``subs`` as ``symbol -> modulus * t``.

    Same shape as :func:`even_extent_substitutions` and for the same reason: a residue modulo a
    chunk is only decidable once the symbol carrying the row stride is written as a multiple of it.
    ``N -> 2*t`` is what turns heat3d's ``N**2*i + N*j + k`` from an unknown parity into a KNOWN
    odd one -- which is a widened load with a one-element shift, not a refusal.

    ``t`` is POSITIVE, not merely non-negative, because the guard checks ``stride >= modulus`` as
    well as the divisibility; that is what proves the widened window still ends inside the
    allocation. An extent fact already pinning the symbol wins: it carries an offset too, so it is
    strictly the stronger statement.
    """
    pinned = {str(s) for s in subs}
    by_name = {}
    for expr in exprs:
        for sym in as_basic(expr).free_symbols:
            by_name.setdefault(str(sym), sym)
    for n, (name, modulus) in enumerate(sorted(facts.items())):
        sym = by_name.get(name)
        if sym is None or name in pinned:
            continue
        subs[sym] = modulus * dace.symbolic.symbol(f"__dace_align_s{n}", positive=True, integer=True)


def declined(arr: Data, edge: MultiConnectorEdge[Memlet], elem_bytes: int, allow_shift: bool) -> tuple[int, int]:
    """The per-element result, plus the reason a SUB-32-bit access had to take it.

    Silence here is expensive and invisible: fp16 is a CLIFF, not a slope. Two elements make one
    32-bit word, which is both the minimum vector and the minimum aligned load, so an access the
    proof cannot place on an even element has NOTHING between ``half2`` and a per-element
    ``LDG.E.U16`` -- roughly six scalar loads where one wide one would do. A 4-byte or wider dtype
    merely degrades a step (128 -> 64 -> 32 bit) and is left unreported.

    The actionable half is the shape: every dimension of a sub-32-bit array has to be a multiple of
    the elements-per-word, or the row start's parity alternates and no access off it is decidable.

    Only the SHIFT-eligible side (the loads) reports. A store never widens off a boundary by
    design -- the widened word covers neighbours it must not overwrite -- so warning there would
    fire on every correct program.
    """
    if elem_bytes < 4 and allow_shift:
        warnings.warn(f'"{edge.data.data}" ({arr.dtype}) is loaded/stored per element: its access could not be '
                      f'placed on a {4 // elem_bytes}-element boundary. Sub-32-bit arrays need EVERY dimension '
                      f'to be a multiple of {4 // elem_bytes} for the vectorized path -- pad the shape, or pin '
                      f'the symbolic extents so their parity is decidable.')
    return elem_bytes, 0


def array_align_shift(node: Node, parent_state: SDFGState, parent_sdfg: SDFG, edge: MultiConnectorEdge[Memlet],
                      vlen: int, allow_shift: bool) -> tuple[int, int]:
    """``(alignment bytes of the aligned base, element shift of the access from it)``.

    The tile side of a load/store is always DACE_ALIGN(64); the array side is a base pointer plus
    the memlet's linear element offset, so it is only as aligned as that offset is divisible.

    When the offset IS divisible the access itself is aligned and the shift is 0 -- the case
    ``c13a93c97`` already handles. When it is not, a ``±1`` stencil neighbour being the canonical
    example, the residue is often still a compile-time constant: ``A[i, j, k+1]`` on a 128-column
    array offsets by ``16384*i + 128*j + k + 16513`` with ``k`` even, which is odd for every
    iteration. Reporting that residue lets the caller load the ALIGNED window below the access and
    pick the elements out in registers, instead of declining to widen at all.

    The window reads ``chunk - shift`` elements past the tile, so the access must not be the last
    one in the allocation; ``at_end`` bounds that. It reads nothing BELOW element 0: the aligned
    base is ``offset - shift`` with ``offset >= shift`` by construction of the residue.

    A symbolic row stride is what defeats both: ``A[i, j]`` on an ``N``-column array offsets by
    ``N*i + j``, whose parity is unknown, so neither a divisibility nor a residue is decidable and
    the caller keeps the scalar path. Returns the element size and shift 0 when nothing is provable.
    """
    arr = parent_sdfg.arrays[required(edge.data.data)]
    elem_bytes = arr.dtype.bytes
    base_bytes = base_align_bytes(arr)
    if base_bytes < elem_bytes:
        return declined(arr, edge, elem_bytes, allow_shift)
    offsets = linear_base_offset(node, parent_state, parent_sdfg, edge)
    if offsets is None:
        return declined(arr, edge, elem_bytes, allow_shift)
    at_base, at_end, size = offsets
    for chunk in (8, 4, 2):
        if chunk * elem_bytes > base_bytes:
            continue
        if dace.symbolic.simplify(at_base % chunk) == 0:
            return chunk * elem_bytes, 0
    # One 32-bit word is the granularity the register extraction works at (__byte_perm), so the
    # window is chunked at that width and the shift has to leave the used elements inside two
    # consecutive words -- guaranteed by chunk | vlen.
    chunk = 4 // elem_bytes
    if not allow_shift or chunk < 2 or vlen % chunk != 0 or chunk * elem_bytes > base_bytes:
        return declined(arr, edge, elem_bytes, allow_shift)
    shift = dace.symbolic.simplify(at_base % chunk)
    if not as_basic(shift).is_Integer:
        return declined(arr, edge, elem_bytes, allow_shift)
    if as_basic(dace.symbolic.simplify(at_end + vlen + chunk - int(as_expr(shift)) - size)).is_nonpositive is not True:
        return declined(arr, edge, elem_bytes, allow_shift)
    return chunk * elem_bytes, int(as_expr(shift))


def align_template_arg(node: Node,
                       parent_state: SDFGState,
                       parent_sdfg: SDFG,
                       edge: MultiConnectorEdge[Memlet],
                       backend: str,
                       vlen: int,
                       allow_shift: bool = False) -> str:
    """``", <bytes>"`` or ``", <bytes>, <shift>"`` for a tile_load/tile_store template list.

    Only the CUDA header takes the trailing ``Align`` / ``Shift`` parameters, and only fp16 has a
    widened path behind them, so every other backend and dtype keeps the exact 3-argument call it
    emitted before.

    ``allow_shift`` is the load/store asymmetry: a load may read the aligned window around its
    elements and discard the extras, a store may not write them -- that would clobber the
    neighbours the widened word covers -- so a shifted store keeps the per-element loop.
    """
    if backend != "cuda":
        return ""
    arr = parent_sdfg.arrays[required(edge.data.data)]
    if arr.dtype != dace.float16:
        return ""
    align, shift = array_align_shift(node, parent_state, parent_sdfg, edge, vlen, allow_shift)
    if shift:
        return f", {align}, {shift}"
    return f", {align}" if align > arr.dtype.bytes else ""
