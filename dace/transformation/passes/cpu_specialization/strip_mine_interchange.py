# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Strip-mine the parallel axis of a carried loop nest and move the tiles outside the carry.

A sequential loop wrapped around a parallel map is the recurrence-sweep shape (TSVC ``s231`` /
``s235`` / ``s275``: ``aa[j, i] = aa[j-1, i] + ...``, ``i`` the contiguous parallel column, ``j`` the
carry; CLOUDSC's vertical ``jk`` loop over its ``(nblocks, klon)`` maps).
:class:`~dace.transformation.passes.cpu_specialization.hoist_parallel_region.HoistParallelRegion`
opens the OpenMP team once for such a nest, but the worksharing construct inside stays per trip::

    #pragma omp parallel                       <- one fork/join
    for (j = 1; j < N; ++j) {
        #pragma omp for simd                   <- an implicit barrier, ONCE PER TRIP
        for (i = 0; i < N; ++i) aa[j][i] = aa[j-1][i] + bb[j][i];
    }

At the sizes these kernels run the trip count IS the array extent, so the nest pays ``N-1`` barriers
to hand each thread about ``N/P`` elements of streaming work between them. Measured on ``s231`` at
``LEN_2D=11966``, 24 threads: 11,965 barriers cost 24.5 ms of a 41.0 ms kernel, ~2.05 us each.

This pass cuts the parallel axis into tiles of ``T`` positions, interchanges the tile loop with the
carry and shares the tiles out, so the nest pays ONE barrier::

    #pragma omp parallel for                   <- one barrier, at the very end
    for (t = 0; t < ceil(N / T); ++t)
        for (j = 1; j < N; ++j)
            #pragma omp simd                   <- i still innermost, still contiguous
            for (i = t*T; i < min((t+1)*T, N); ++i) aa[j][i] = aa[j-1][i] + bb[j][i];

``T`` is the one knob (:func:`tile_size`). An axis that is not innermost in every map -- the
monolith's ``nblocks`` over ``klon`` -- is interchanged whole, ``T = 1``: the strides inside do not
change. An innermost axis is strip-mined: ``T`` is at most the largest power of two in ``[16, 256]``
whose working set fills half of one core's L2, so the contiguous remainder stays inside and a tile's carry
stays in cache, and is evened out so every thread runs the same number of tiles. :func:`axis_choices`
orders the candidate axes so.

Why not simply interchange
--------------------------

:class:`~dace.transformation.passes.canonicalize.move_loop_into_map_gated.MoveLoopIntoMapGated`
already turns ``for(seq) { map }`` into ``map { for(seq) }``, and DECLINES it on CPU for exactly
these kernels: hoisting the whole map puts ``j`` innermost at stride ``N``, one cache line per
element. Lifting only a TILE of the parallel axis keeps the contiguous remainder inside. (On GPU
that pass always interchanges, which is this same shape at ``T = 1``: one thread per column.)

Where the cut lands
-------------------

Every worksharing map at the top level of the loop body is a statement, and every statement is
narrowed along the same extent: one parameter of its own nest, or, when its own nest has none, one
parameter in every map nest of the nested SDFG its body is (:func:`targets_for`). The monolith runs
each species loop as a map over five species whose body holds its ``(nblocks, klon)`` plane work;
the tiles narrow those plane maps through the nested SDFG's ``symbol_mapping``.

Legality
--------

A tile runs its whole carry alone -- every trip of ``j`` for the positions it owns, in order, with no
other thread touching them in between, and with no barrier between two maps of one trip either. That
is correct iff no dependence crosses a tile boundary, a LOCAL test on the accesses
(:func:`tile_local`): every dependence must be at distance zero in the cut axis. Each map's
parameters are read as offsets from their range's begin, which is what the tiles slice, so ``t[i]``
over ``0:n`` and ``t[i - k]`` over ``k:k+n`` are one position.

- ``aa[j-1, i]`` against ``aa[j, i]`` -- distance 0 in ``i``. Accepted (``s231``, ``s235``,
  ``s275``).
- ``s119``'s ``aa[i-1, j-1]`` and ``wf_diff_skew``'s ``a[i-1, j+1]`` -- distance one in the map
  parameter, so a tile would read a neighbour's column. Refused.
- ``s115`` reads a scalar every tile needs but one tile writes: its destination names no map
  parameter at all, so it is a location shared across tiles. Refused.
- The wavefront skew's diagonal writes ``A[t - 2*p]`` from a row assembled in a transient, so the
  carried write leaves as a COPY through the map exit rather than as a tasklet store. The boundary
  edges are read as accesses too (:func:`boundary_accesses`); a test that sees tasklet stores alone
  finds no write for ``A`` and approves the nest vacuously.
- A body canonicalize left as a nested SDFG carries the whole-map UNION on those boundary edges --
  CLOUDSC's ``ztp1[0:N, 0:N]``. A union names no map parameter because it is every iteration at once,
  so it is dropped rather than read as ``s115``'s shared location.
- An access through a view is an access of the array it views, at the position the view binds
  (:func:`resolved`); a view that is no slice refuses.
- What every tile runs itself -- a statement beside the maps (:func:`replicable`) and the control
  flow of the body, its branch conditions and interstate assignments -- must read nothing another tile
  writes (:func:`record_control_flow_reads`).

Correctness does NOT depend on how OpenMP distributes the tile loop. A tile's entire carry sits
inside ONE iteration of the outer map, so whichever thread runs tile ``t`` runs all of ``t``'s trips
in order, for any tile count and any schedule. That is what this form has over dropping the barrier
with ``nowait``, which is correct only while consecutive worksharing regions hand the same
iterations to the same thread.

Condition (H) of ``HoistParallelRegion`` is inherited, with one difference: a statement beside the
maps is run by every tile rather than wrapped in a one-iteration worksharing map, which is right only
when it writes nothing but loop-local scalars each tile keeps its own copy of. Condition (T) does
not apply: a tile runs its whole carry on one thread. A loop this pass refuses is left to
``HoistParallelRegion``, which keeps the barrier per trip.
"""

from collections.abc import Iterator
from functools import lru_cache
from typing import Any, NamedTuple

import sympy

from dace import SDFG, data, dtypes, properties, subsets, symbolic
from dace.optionals import required
from dace.ordered import OrderedSet
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.narrowing import as_basic
from dace.sdfg.sdfg import memlets_in_ast
from dace.sdfg.state import AbstractControlFlowRegion, ConditionalBlock, LoopRegion, SDFGState, SymbolResolver
from dace.transformation import transformation
from dace.transformation.passes.canonicalize.supply_num_threads import DTYPE as NUM_THREADS_DTYPE
from dace.transformation.passes.cpu_specialization.hoist_parallel_region import (
    WORKSHARED,
    HoistParallelRegion,
    loop_defined_symbols,
    loop_local_transients,
    outline,
    top_level_loop_locals,
)
from dace.transformation.passes.cpu_specialization.machine import topology

#: Parameter name of the tile loop. One name, so the reshape can find the map it just made.
TILE_PARAM = "__dace_tile"

#: Built ONCE, and built WITH THEIR DTYPE. DaCe folds dtype and assumptions into symbol identity,
#: so a name minted twice at two widths is two symbols that never cancel; declaring these here at
#: the same ``int64`` the SDFG declaration uses keeps one instance of each name in the graph. One
#: instance is also what makes the reuse cheap -- ``pystr_to_symbolic`` runs the sympy parser, far
#: too expensive to repeat per map, per dimension, per candidate loop.
TILE_SYMBOL = symbolic.symbol(TILE_PARAM, NUM_THREADS_DTYPE)
THREADS_SYMBOL = symbolic.symbol(symbolic.NUM_THREADS_SYMBOL, NUM_THREADS_DTYPE)


#: Name prefix of the positional stand-ins :func:`axis_symbol` mints.
AXIS_PREFIX = "__dace_tile_axis"


#: What a symbol a nested SDFG defines itself stands for once its accesses are translated out of it
#: (:func:`descend_into_nested`): an axis that is never the cut one.
INNER_SYMBOL = symbolic.symbol(f"{AXIS_PREFIX}_inner", NUM_THREADS_DTYPE)


@lru_cache(maxsize=None, typed=True)
def axis_symbol(position: int):
    """The stand-in for a tiled map's parameter ``position`` places out from its innermost.

    Every map of one loop body is cut the same way, so its parameter at a given position denotes
    the same tile as another map's at that position. The two maps spell it with different names
    (``_loop_it_1`` against ``_loop_it_4``), and comparing the spellings would read one location as
    two unrelated ones, which is how a cross-map dependence goes unseen.
    """
    return symbolic.symbol(f"{AXIS_PREFIX}{position}", NUM_THREADS_DTYPE)


def index_expressions(subset: subsets.Subset | None) -> list[Any]:
    """The per-dimension index expression of ``subset``, one entry per dimension: where each ``Range`` dimension
    starts."""
    if isinstance(subset, subsets.Range):
        return [begin for begin, _, _ in subset.ranges]
    return []


def names_a_param(expr: Any, params: set[str]) -> bool:
    """Whether ``expr`` mentions any of ``params``, which are symbol NAMES.

    By name, never by symbol object. Two ``dace.symbolic.symbol`` instances that share a name but
    carry different sympy assumptions -- which is what a subset rewritten through a nested SDFG's
    ``symbol_mapping`` ends up holding against a freshly parsed map parameter -- are unequal, so an
    intersection of symbol sets silently finds nothing and every real kernel is refused.
    """
    return bool(symbolic.issymbolic(expr) and ({str(s) for s in expr.free_symbols} & params))


def same_index(left: Any, right: Any) -> bool:
    """Whether two index expressions denote the same position.

    The printed forms are compared first: the accepted shape carries the SAME expression on both
    sides of a cut dimension, and that answers without touching sympy at all.

    Anything else goes to :func:`~dace.symbolic.equal`, which equalizes the two expressions'
    same-named symbols onto one instance before comparing. That step is the whole reason not to
    subtract them directly, and not to reparse them into fresh ones: these subsets come from
    different sources -- one rewritten through a nested SDFG's ``symbol_mapping``, one straight off
    a memlet -- and dace folds dtype and assumptions into symbol identity, so two instances of the
    same NAME do not cancel. ``equal`` answers ``None`` when it cannot decide, which is read here
    as "not the same position": an undecided distance must refuse the tile, never take it.
    """
    if left is right or str(left) == str(right):
        return True
    # is_length=False: an index may be zero or negative (``jk - 1`` at the top of a sweep), so the
    # positive-integer assumptions ``equal`` makes for extents would be wrong here.
    return symbolic.equal(left, right, is_length=False) is True


def boundary_names(state: SDFGState, map_entry: nodes.MapEntry, map_exit: nodes.MapExit) -> set[str]:
    """The arrays that cross ``map_entry``'s scope boundary.

    Only these can carry a value from one trip of the enclosing loop to the next. Everything else
    under the map -- the per-iteration scalars a body is built from, ``aa_index`` and friends --
    is created and consumed inside ONE iteration, so it is privatized by construction and says
    nothing about whether a tile is self-contained. Including them would refuse every real kernel,
    because such a scalar is indexed ``[0]`` and so names no map parameter.
    """
    names = set()
    for edge in state.in_edges(map_entry):
        if edge.data is not None and edge.data.data is not None:
            names.add(edge.data.data)
    for edge in state.out_edges(map_exit):
        if edge.data is not None and edge.data.data is not None:
            names.add(edge.data.data)
    return names


class ViewBinding(NamedTuple):
    """What a view reads or writes through: the array it views and the subset of it the view binds."""

    data: str
    subset: subsets.Range


def view_bindings(state: SDFGState) -> dict[str, ViewBinding] | None:
    """``view -> binding`` for every view of ``state``, or ``None`` if one is not a slice of an array -- a
    reshape, a view of a view -- or one view is bound to two subsets. An access through a view is an access
    of the array, and only :func:`resolved` puts it where the array's own accesses can be compared with it."""
    sdfg = state.sdfg
    bindings: dict[str, ViewBinding] = {}
    for node in state.data_nodes():
        desc = node.desc(sdfg)
        if not isinstance(desc, data.View):
            continue
        edge = sdutil.get_view_edge(state, node)
        viewed = None if edge is None else (edge.src if edge.dst is node else edge.dst)
        if not isinstance(viewed, nodes.AccessNode) or isinstance(viewed.desc(sdfg), data.View):
            return None
        subset = edge.data.subset if edge.data.data == viewed.data else edge.data.other_subset
        if not isinstance(subset, subsets.Range):
            return None
        if [str(size) for size in subset.size() if size != 1] != [str(size) for size in desc.shape if size != 1]:
            return None
        known = bindings.get(node.data)
        if known is not None and (known.data != viewed.data or str(known.subset) != str(subset)):
            return None
        bindings[node.data] = ViewBinding(viewed.data, subset)
    return bindings


def resolved(name: str, indices: list[Any], bindings: dict[str, ViewBinding], sdfg: SDFG) -> tuple[str, list[Any]]:
    """``name[indices]`` as the element of the array it is: through a view, the view's non-unit dimensions
    pair up in order with the binding's, each at its begin plus its step times the index. An index list that
    does not cover the view becomes an index-less access, which matches no write in :func:`tile_local`."""
    binding = bindings.get(name)
    if binding is None:
        return name, indices
    shape = sdfg.arrays[name].shape
    if len(indices) != len(shape):
        return binding.data, []
    through = iter([index for index, size in zip(indices, shape, strict=True) if size != 1])
    composed = [
        begin if size == 1 else begin + step * next(through)
        for (begin, _, step), size in zip(binding.subset.ranges, binding.subset.size(), strict=True)
    ]
    return binding.data, composed


def collect_accesses(
    state: SDFGState,
    scope_node: nodes.MapEntry,
    allowed: set[str],
    reads: dict[str, list[list[Any]]],
    writes: dict[str, list[list[Any]]],
    nested: bool = True,
) -> bool:
    """Gather the PER-ELEMENT accesses under ``scope_node`` into ``reads`` / ``writes``.

    The memlets crossing a map's boundary are the union over the whole map -- for a body that
    canonicalize left as a nested SDFG they read ``ztp1[0:N, 0:N]``, which names no map parameter
    and says nothing about any single iteration. The accesses that DO are the ones adjacent to the
    code nodes, so this walks down to them, following nested SDFGs and rewriting each subset into
    the outer symbols on the way (a nested graph names its own arrays and its own symbols).

    :param state: the state holding ``scope_node``.
    :param scope_node: the map entry whose scope to walk.
    :param allowed: the arrays that cross the scope boundary; anything else is per-iteration.
    :param reads: ``name -> [index expression lists]``, extended in place.
    :param writes: ``name -> [index expression lists]``, extended in place.
    :param nested: descend into nested SDFGs; off when the caller reads their maps itself.
    :returns: ``False`` if something in the scope cannot be analysed, in which case the caller
              must refuse rather than trust a partial picture.
    """
    bindings = view_bindings(state)
    if bindings is None:
        return False
    allowed = allowed | {bindings[name].data for name in allowed if name in bindings}
    for node in state.scope_subgraph(scope_node, include_entry=False, include_exit=False).nodes():
        if isinstance(node, nodes.Tasklet):
            for edge in state.in_edges(node):
                if edge.data.is_empty() or edge.data.subset is None:
                    continue
                name, indices = resolved(edge.data.data, index_expressions(edge.data.subset), bindings, state.sdfg)
                if name in allowed:
                    reads.setdefault(name, []).append(indices)
            for edge in state.out_edges(node):
                if edge.data.is_empty() or edge.data.subset is None:
                    continue
                name, indices = resolved(edge.data.data, index_expressions(edge.data.subset), bindings, state.sdfg)
                if name in allowed:
                    writes.setdefault(name, []).append(indices)
                    if edge.data.wcr is not None:  # an accumulation reads what it updates
                        reads.setdefault(name, []).append(indices)
        elif isinstance(node, nodes.NestedSDFG):
            if not nested:
                continue
            if node.sdfg is None or not descend_into_nested(state, node, allowed, reads, writes):
                return False
        elif isinstance(node, nodes.LibraryNode):
            return False  # expands to whatever it likes; its accesses are not known here
    return True


def descend_into_nested(
    state: SDFGState,
    node: nodes.NestedSDFG,
    allowed: set[str],
    reads: dict[str, list[list[Any]]],
    writes: dict[str, list[list[Any]]],
    enclosing: dict[str, str] | None = None,
    substitution: dict[str, Any] | None = None,
) -> bool:
    """Add the accesses inside ``node``, translated into the names and symbols ``allowed`` is spelled in.

    A nested SDFG inside ``node`` is descended into the same way, one level at a time: the names and symbols
    of each level are translated into those of the level above, and on through ``enclosing`` and
    ``substitution`` into the outermost ones (velocity_tendencies' gathers load each element in a nested
    ``loop_body`` inside the map's own nested body).

    :param state: the state holding ``node``.
    :param node: the nested SDFG to descend into.
    :param allowed: the arrays that cross the outermost scope boundary.
    :param reads: ``name -> [index expression lists]``, extended in place.
    :param writes: ``name -> [index expression lists]``, extended in place.
    :param enclosing: ``array of state's graph -> outermost name``; ``None`` where ``state`` is outermost.
    :param substitution: ``symbol of state's graph -> outermost expression``; ``None`` where ``state`` is
                         outermost.
    :returns: ``False`` if the boundary cannot be translated.
    """
    # A nested graph names its arrays by CONNECTOR, so the edges say which outer array each is.
    outer_name = {}
    for edge in state.in_edges(node):
        if edge.dst_conn is not None and edge.data is not None and edge.data.data is not None:
            outer_name[edge.dst_conn] = edge.data.data
    for edge in state.out_edges(node):
        if edge.src_conn is not None and edge.data is not None and edge.data.data is not None:
            outer_name[edge.src_conn] = edge.data.data
    if enclosing is not None:
        outer_name = {inner: enclosing[outer] for inner, outer in outer_name.items() if outer in enclosing}
    # ... and its own symbols, so an inner subset has to be rewritten before it can be compared
    # against an outer map parameter. ``symbol_mapping`` is keyed by NAME, and the rewrite below
    # keys on the symbol instances the expression actually holds -- minting a key from the name
    # here would produce a symbol whose dtype and assumptions need not match the one in the
    # expression, and ``subs`` matches on identity, so the substitution would quietly do nothing
    # and leave inner names to be compared against outer ones.
    mapping = {
        inner: value if substitution is None else rewritten([value], substitution, INNER_SYMBOL)[0]
        for inner, value in node.symbol_mapping.items()
    }
    # A symbol the nested graph defines itself -- a loop variable, a map parameter, an interstate assignment --
    # is in no ``symbol_mapping``. Its spelling says nothing about its value at one access against another
    # (``s = 1`` before a read of ``a[i + s]``, ``s = 0`` before a write of it), so it becomes an axis of its
    # own, which :func:`tile_local` never takes for the cut position.
    allowed_names = {inner: outer for inner, outer in outer_name.items() if outer in allowed}
    record_control_flow_reads(node.sdfg, allowed_names, reads, mapping, INNER_SYMBOL)
    for inner_state in node.sdfg.states():
        bindings = view_bindings(inner_state)
        if bindings is None:
            return False
        for inner_node in inner_state.nodes():
            if isinstance(inner_node, nodes.NestedSDFG):
                # A view handed down would need its binding composed into every access below; refused instead.
                if any(
                    edge.data.data in bindings and outer_name.get(bindings[edge.data.data].data) in allowed
                    for edge in inner_state.all_edges(inner_node)
                    if not edge.data.is_empty()
                ):
                    return False
                if inner_node.sdfg is None or not descend_into_nested(
                    inner_state, inner_node, allowed, reads, writes, outer_name, mapping
                ):
                    return False
                continue
            if not isinstance(inner_node, nodes.AccessNode):
                continue
            viewed = bindings[inner_node.data].data if inner_node.data in bindings else inner_node.data
            name = outer_name.get(viewed)
            if name is None or name not in allowed:
                continue  # private to the nested graph, so it cannot carry across trips
            binding_edge = sdutil.get_view_edge(inner_state, inner_node) if inner_node.data in bindings else None

            def element(edge: Any, inner_node: nodes.AccessNode = inner_node) -> list[Any]:
                if edge.data.data != inner_node.data:
                    return []  # the subset names the other end; read as index-less
                _, indices = resolved(inner_node.data, index_expressions(edge.data.subset), bindings, node.sdfg)
                return rewritten(indices, mapping, INNER_SYMBOL)

            # An edge into or out of a nested SDFG carries its union; the descent above records its accesses
            # element by element.
            for edge in inner_state.in_edges(inner_node):
                if edge is not binding_edge and edge.data.subset is not None:
                    if isinstance(edge.src, nodes.NestedSDFG):
                        continue
                    writes.setdefault(name, []).append(element(edge))
                    if edge.data.wcr is not None:  # an accumulation reads what it updates
                        reads.setdefault(name, []).append(element(edge))
            for edge in inner_state.out_edges(inner_node):
                if edge is not binding_edge and edge.data.subset is not None:
                    if isinstance(edge.dst, nodes.NestedSDFG):
                        continue
                    reads.setdefault(name, []).append(element(edge))
    return True


def substituted(subset: subsets.Subset | None, substitution: dict[str, Any]) -> list[Any]:
    """``subset``'s index expressions rewritten through ``substitution``, which is keyed by NAME.

    Each expression is replaced against the symbol instances it actually carries, looked up by
    name, so the rewrite cannot be defeated by two same-named symbols that differ in dtype or
    assumptions -- ``subs`` matches on identity, and a key minted from the name need not be the
    instance in the expression.
    """
    return rewritten(index_expressions(subset), substitution)


def rewritten(expressions: list[Any], substitution: dict[str, Any], other: Any = None) -> list[Any]:
    """``expressions`` rewritten through ``substitution``, keyed by NAME; see :func:`substituted`. A symbol
    ``substitution`` does not name becomes ``other``, or stays as it is where ``other`` is ``None``.

    The replacement is simultaneous (``xreplace``): ``subs`` with a dict substitutes one key after another, so
    ``{k: j, j: s}`` would carry ``k`` on into ``s``.
    """
    out = []
    for expr in expressions:
        if not symbolic.issymbolic(expr):
            out.append(expr)
            continue
        replacements = {
            sym: substitution[sym.name] if sym.name in substitution else other
            for sym in expr.free_symbols
            if sym.name in substitution or other is not None
        }
        out.append(expr.xreplace(replacements) if replacements else expr)
    return out


def boundary_accesses(
    state: SDFGState,
    map_entry: nodes.MapEntry,
    map_exit: nodes.MapExit,
    allowed: set[str],
    reads: dict[str, list[list[Any]]],
    writes: dict[str, list[list[Any]]],
    skip_nested: bool = False,
) -> bool:
    """Add the PER-ITERATION accesses the scope BOUNDARY edges carry, whatever produced them.

    :func:`collect_accesses` reads the subsets adjacent to code nodes, so a value assembled in a
    transient and copied out through the map exit is recorded nowhere -- and a missing WRITE makes
    :func:`tile_local` vacuous: it iterates the writes, finds none for the array, and approves a
    nest whose whole carry crosses every tile. The wavefront skew emits exactly that shape -- a
    diagonal's row is built in a transient and copied into ``A`` -- and the tiled form raced at
    four threads. Every value a tile could carry crosses this boundary, and its subset is on the
    boundary edge whatever assembled it, so reading these edges closes the hole for every copy
    shape at once.

    A boundary edge is such an access only while it NAMES A MAP PARAMETER. Where the body is a
    nested SDFG the same edge carries the propagated union over the whole map instead --
    ``ztp1[0:N, 0:N]`` -- which is every iteration at once and so says nothing about any one of
    them. :func:`index_expressions` reduces that union to its begins ``[0, 0]``, and
    :func:`tile_local` then reads a union WRITE as ``s115``'s location shared by every tile and a
    union READ as column zero of a per-column write. Both refuse, and both are wrong: CLOUDSC's
    vertical carry and a gather on a read-only operand are distance-0 in the cut axis and must
    tile. So a union is dropped rather than trusted.

    Dropping it must not restore the vacuum this function exists to close, so a union write whose
    array the interior walk recorded nothing for is an unaccounted write, and refuses.

    :param state: the state holding the map.
    :param map_entry: the worksharing map whose axis would be cut.
    :param map_exit: ``map_entry``'s exit.
    :param allowed: the arrays that cross the scope boundary.
    :param reads: ``name -> [index expression lists]``, extended in place.
    :param writes: ``name -> [index expression lists]``, extended in place.
    :returns: ``False`` if a boundary write is a union no interior access accounts for.
    """
    bindings = view_bindings(state)
    if bindings is None:
        return False
    allowed = allowed | {bindings[name].data for name in allowed if name in bindings}
    params = OrderedSet(map_entry.map.params)
    union_writes: list[str] = []
    directions = ((state.out_edges(map_entry), reads, False), (state.in_edges(map_exit), writes, True))
    for edges, target, written in directions:
        for edge in edges:
            if edge.data is None or edge.data.data not in allowed or edge.data.subset is None:
                continue
            # An edge into or out of a NESTED map carries that map's union; its iterations' own accesses are
            # what the interior walk recorded, element by element.
            inside = edge.dst if not written else edge.src
            if isinstance(inside, (nodes.MapEntry, nodes.MapExit)) or (
                skip_nested and isinstance(inside, nodes.NestedSDFG)
            ):
                continue
            name, indices = resolved(edge.data.data, index_expressions(edge.data.subset), bindings, state.sdfg)
            if any(names_a_param(index, params) for index in indices):
                target.setdefault(name, []).append(indices)
                if written and edge.data.wcr is not None:  # an accumulation reads what it updates
                    reads.setdefault(name, []).append(indices)
            elif written:
                union_writes.append(name)
    # Read AFTER the loop, so an array carrying both a union edge and a per-iteration one is
    # accounted for whichever order the exit edges come in.
    return all(name in writes for name in union_writes)


class Target(NamedTuple):
    """A map whose parameter is cut, and the nested SDFG node it lives in (``None``: the statement's own state)."""

    owner: nodes.MapEntry
    param: str
    nested: nodes.NestedSDFG | None


class Statement(NamedTuple):
    """One worksharing map at the top level of the loop body and the maps whose cut parameter its tiles narrow:
    one map of its own nest, or every map nest inside the nested SDFG its body is (CLOUDSC's monolith wraps
    each species loop's plane work in one)."""

    state: SDFGState
    entry: nodes.MapEntry
    targets: list[Target]


class InnerNest(NamedTuple):
    """A map nest at the top level of a state of a nested SDFG in a statement's scope."""

    nested: nodes.NestedSDFG
    state: SDFGState
    entry: nodes.MapEntry
    axes: list[tuple[nodes.MapEntry, str, Any]]


class Shape(NamedTuple):
    """A statement before an axis is chosen: its own nest's axes and the map nests one nested SDFG down."""

    state: SDFGState
    entry: nodes.MapEntry
    exit: nodes.MapExit
    nest: list[nodes.MapEntry]
    axes: list[tuple[nodes.MapEntry, str, Any]]
    inner: list[InnerNest] | None


def map_nest(entry: nodes.MapEntry, children: dict[Any, list[nodes.Node]]) -> list[nodes.MapEntry]:
    """``entry`` and every map nested in its scope (in this state), outermost first."""
    nest = [entry]
    for node in children[entry]:
        if isinstance(node, nodes.MapEntry):
            nest.extend(map_nest(node, children))
    return nest


def nest_axes(nest: list[nodes.MapEntry]) -> list[tuple[nodes.MapEntry, str, Any]]:
    """``(map entry, parameter, extent)`` of every unit-stride parameter of ``nest``, outermost first, or ``[]``
    if a range is strided or depends on a parameter of the nest (a triangular range has no fixed extent)."""
    params = {p for entry in nest for p in entry.map.params}
    axes = []
    for entry in nest:
        if params & entry.map.range.free_symbols:
            return []
        for param, dimension in zip(entry.map.params, entry.map.range.ranges, strict=True):
            if dimension[2] != 1:
                return []
            axes.append((entry, param, extent(dimension)))
    return axes


def control_flow_reads(region: AbstractControlFlowRegion) -> list[tuple[str, list[Any]]]:
    """``(container, indices)`` for every element the control flow of ``region`` reads: interstate edges,
    loop headers (``region``'s own among them) and branch conditions, of its nested regions too. A container
    named without a subscript the parser can read is listed with no index."""
    arrays = region.sdfg.arrays
    names: set[str] = set()
    found: list[tuple[str, list[Any]]] = []
    for edge in region.all_interstate_edges():
        names |= edge.data.read_symbols()
        try:
            found += [(m.data, index_expressions(m.subset)) for m in edge.data.get_read_memlets(arrays, True)]
        except (ValueError, TypeError, KeyError, IndexError, AttributeError, SyntaxError):
            pass  # an indirect subscript: the name below stands for it, index-less
    for block in [region, *region.all_control_flow_blocks()]:
        if isinstance(block, LoopRegion):
            codes = [block.init_statement, block.loop_condition, block.update_statement]
        elif isinstance(block, ConditionalBlock):
            codes = [condition for condition, _ in block.branches]
        else:
            continue
        for code in codes:
            if code is None:
                continue
            names |= {str(name) for name in code.get_free_symbols()}
            if code.language != dtypes.Language.Python:
                continue
            for statement in code.code:
                try:
                    found += [
                        (m.data, index_expressions(m.subset))
                        for m in memlets_in_ast(statement, arrays, include_scalars=True)
                    ]
                except (ValueError, TypeError, KeyError, IndexError, AttributeError, SyntaxError):
                    pass
    indexed = {name for name, _ in found}
    return found + [(name, []) for name in sorted(names) if name in arrays and name not in indexed]


def record_control_flow_reads(
    region: AbstractControlFlowRegion,
    outer_name: dict[str, str],
    reads: dict[str, list[list[Any]]],
    substitution: dict[str, Any] | None = None,
    other: Any = None,
) -> None:
    """Add what the control flow of ``region`` reads (:func:`control_flow_reads`) to ``reads``, under its
    enclosing name (``outer_name``) and in the enclosing symbols (``substitution``, a nested SDFG's
    ``symbol_mapping``; a symbol it does not name becomes ``other``, see :func:`rewritten`). Every tile takes a
    branch or derives a symbol itself, which is the same as taking it once only while no other tile writes what
    it reads: ``s275``'s ``aa[0, i]`` guard reads its own column; an index-less read matches no write in
    :func:`tile_local`."""
    for name, indices in control_flow_reads(region):
        if name in outer_name:
            reads.setdefault(outer_name[name], []).append(rewritten(indices, substitution or {}, other))


def inner_nests(state: SDFGState, entry: nodes.MapEntry) -> list[InnerNest] | None:
    """The map nests of the nested SDFGs in ``entry``'s scope, or ``None`` if one of them holds something a tile
    cannot narrow: a statement outside every map that writes data from outside the nest, a copy or library node
    at the top level, or a nested SDFG of its own."""
    nests: list[InnerNest] = []
    for node in state.scope_subgraph(entry).nodes():
        if not isinstance(node, nodes.NestedSDFG):
            continue
        outside = set(node.in_connectors) | set(node.out_connectors)
        for inner_state in node.sdfg.states():
            children = inner_state.scope_children()
            for inner in children[None]:
                if isinstance(inner, (nodes.NestedSDFG, nodes.LibraryNode)):
                    return None
                if isinstance(inner, nodes.Tasklet) and any(
                    not e.data.is_empty() and e.data.data in outside for e in inner_state.out_edges(inner)
                ):
                    return None
                if isinstance(inner, nodes.MapEntry):
                    nests.append(InnerNest(node, inner_state, inner, nest_axes(map_nest(inner, children))))
            if any(
                isinstance(e.src, nodes.AccessNode) and isinstance(e.dst, nodes.AccessNode) and not e.data.is_empty()
                for e in inner_state.edges()
            ):
                return None
    return nests


def count_axis(axes: list[tuple[nodes.MapEntry, str, Any]], size: Any) -> int:
    """How many parameters of ``axes`` have extent ``size``."""
    return sum(1 for _, _, other in axes if same_extent(other, size))


def targets_for(shape: Shape, size: Any) -> list[Target] | None:
    """The maps a tile of extent ``size`` narrows in ``shape``: the outermost parameter of its own nest with that
    extent; or, when its own nest has none, the outermost one with it in every inner nest. ``None`` otherwise.

    Outermost, not the only one: a cube ``(N, N, N, B)`` has three axes of extent ``N``, and refusing them all left
    only the short innermost ``B`` to cut -- one or two tiles, the whole volume on one or two threads
    (vector_stencil_4d). Where statements disagree on which axis that is, :func:`tile_local` sees the dependence at
    a nonzero distance and refuses the cut."""
    own = [(owner, param) for owner, param, other in shape.axes if same_extent(other, size)]
    if own:
        return [Target(own[0][0], own[0][1], None)]
    if not shape.inner or any(count_axis(inner.axes, size) == 0 for inner in shape.inner):
        return None
    return [
        Target(owner, param, inner.nested)
        for inner in shape.inner
        for owner, param, other in inner.axes[
            : next(i for i, a in enumerate(inner.axes) if same_extent(a[2], size)) + 1
        ]
        if same_extent(other, size)
    ]


def axis_choices(shapes: list[Shape], facts: symbolic.Facts | None = None) -> list[tuple[Any, bool]]:
    """The extents to try cutting, best first, each with whether it is innermost in every statement.

    A candidate is an extent every statement can be narrowed along (:func:`targets_for`). Extents not
    innermost everywhere come first: moving one outside the carry keeps the innermost, unit-stride axis
    innermost (CLOUDSC's monolith: ``nblocks`` over ``klon``). An innermost one is cut into tiles instead, so
    the contiguous remainder stays inside (the column axis of CLOUDSC); the deepest of those comes first. An
    innermost extent too small for two tiles is no candidate: one tile runs the whole nest on one thread.

    A ``SMALL`` extent (``nblks``: a handful, under ``facts``) is never a candidate: its tiles could not
    occupy the team. Nor does it hide the axis it sits under from being innermost:
    :class:`~dace.transformation.passes.cpu_specialization.sequentialize_small_axes.SequentializeSmallAxes`
    moves ``SMALL`` axes into an inner sequential map, so ``velocity_tendencies``' ``(nproma)`` maps each hold
    an ``(nblks)`` one, and reading ``nblks`` as the innermost axis moved ``nproma`` outside the carry WHOLE
    -- one column per tile, every level of it at stride ``nproma``.
    """
    if not shapes:
        return []

    def small(size: Any) -> bool:
        return facts is not None and symbolic.issymbolic(size) and symbolic.is_small(size, facts)

    first = shapes[0]
    seen = [size for _, _, size in first.axes] + [size for inner in first.inner or [] for _, _, size in inner.axes]
    candidates = []
    for size in seen:
        if small(size) or any(same_extent(size, other) for other in candidates):
            continue
        if all(targets_for(shape, size) is not None for shape in shapes):
            candidates.append(size)

    def innermost(shape: Shape, size: Any) -> bool:
        nests = [shape.axes] if count_axis(shape.axes, size) else [inner.axes for inner in shape.inner or []]
        unsmall = [[axis for axis in axes if not small(axis[2])] for axes in nests]
        return all(axes and same_extent(axes[-1][2], size) for axes in unsmall)

    outer = [(size, False) for size in candidates if not all(innermost(shape, size) for shape in shapes)]
    inner = [
        (size, True)
        for size in reversed(candidates)
        if all(innermost(shape, size) for shape in shapes)
        and not (symbolic.issymbolic(size) is False and size < 2 * MIN_TILE)
    ]
    return outer + inner


def renamed(index_lists: dict[str, list[list[Any]]], renaming: dict[str, Any]) -> dict[str, list[list[Any]]]:
    """``index_lists`` with every expression rewritten through ``renaming`` (by name)."""
    return {name: [rewritten(indices, renaming) for indices in lists] for name, lists in index_lists.items()}


def nest_renaming(
    nest: list[nodes.MapEntry], cut: tuple[nodes.MapEntry, str] | None, first: int
) -> tuple[dict[str, Any], int]:
    """Stand-ins for every parameter of ``nest``: the ``cut`` one to ``axis_symbol(0)``, the others to positions
    from ``first`` on, each plus its range's begin. Returns the renaming and the next free position."""
    renaming: dict[str, Any] = {}
    position = first
    for owner in nest:
        for param, (begin, _, _) in zip(owner.map.params, owner.map.range.ranges, strict=True):
            if cut is not None and owner is cut[0] and param == cut[1]:
                renaming[param] = axis_symbol(0) + begin
            else:
                renaming[param] = axis_symbol(position) + begin
                position += 1
    return renaming, position


def collect_tile_accesses(
    statement: Statement,
    shape: Shape,
    reads: dict[str, list[list[Any]]],
    writes: dict[str, list[list[Any]]],
    every_array: bool = False,
) -> bool:
    """Add ``statement``'s per-element accesses to ``reads`` / ``writes``, on the tiles' own axes.

    Every map parameter is rewritten to a positional stand-in (:func:`axis_symbol`) plus its range's begin: the
    cut one to ``axis_symbol(0)``, the others to ``1, 2, ...``. The stand-ins put the accesses of two DIFFERENT
    maps of one body in the same coordinates, and the OFFSET from the begin is what the tiles slice: a map
    over ``0:n`` reading ``t[i]`` and one over ``k:k+n`` writing ``t[i - k]`` touch the same element in the
    same tile, and only their offsets say so. Where the cut maps sit in a nested SDFG, its maps' accesses are
    read element by element, translated into the enclosing names and symbols; the union a nested SDFG's
    boundary carries says nothing about one tile.

    :param statement: the worksharing map and the maps its tiles narrow.
    :param shape: the statement's nests.
    :param reads: ``name -> [index expression lists]``, extended in place.
    :param writes: ``name -> [index expression lists]``, extended in place.
    :param every_array: record the arrays that never cross the statement's boundary too (its per-iteration
                        scratch), which the legality test leaves out and the slicing of loop locals needs.
    :returns: ``False`` if something in the scope cannot be analysed.
    """
    local_reads: dict[str, list[list[Any]]] = {}
    local_writes: dict[str, list[list[Any]]] = {}
    state, entry = statement.state, statement.entry
    allowed = set(state.sdfg.arrays) if every_array else boundary_names(state, entry, shape.exit)
    inner_cut = statement.targets[0].nested is not None
    if not collect_accesses(state, entry, allowed, local_reads, local_writes, nested=not inner_cut):
        return False
    if not boundary_accesses(state, entry, shape.exit, allowed, local_reads, local_writes, skip_nested=inner_cut):
        return False
    own_cut = None if inner_cut else (statement.targets[0].owner, statement.targets[0].param)
    renaming, position = nest_renaming(shape.nest, own_cut, 1)
    if inner_cut:
        for nested in {id(inner.nested): inner.nested for inner in shape.inner or []}.values():
            if not replicated_reads(state, nested, local_reads):
                return False
        for inner in shape.inner or []:
            target = next(
                t
                for t in statement.targets
                if t.nested is inner.nested and t.owner in map_nest(inner.entry, inner.state.scope_children())
            )
            outer_name = outer_names(state, inner.nested)
            inner_reads: dict[str, list[list[Any]]] = {}
            inner_writes: dict[str, list[list[Any]]] = {}
            if not collect_accesses(inner.state, inner.entry, set(outer_name), inner_reads, inner_writes):
                return False
            inner_renaming, position = nest_renaming(
                map_nest(inner.entry, inner.state.scope_children()), (target.owner, target.param), position
            )
            for source, sink in ((inner_reads, local_reads), (inner_writes, local_writes)):
                for name, lists in renamed(renamed(source, inner_renaming), inner.nested.symbol_mapping).items():
                    sink.setdefault(outer_name[name], []).extend(lists)
    for source, target in ((local_reads, reads), (local_writes, writes)):
        for name, index_lists in renamed(source, renaming).items():
            target.setdefault(name, []).extend(index_lists)
    return True


def outer_names(state: SDFGState, node: nodes.NestedSDFG) -> dict[str, str]:
    """``connector -> array`` of the data ``node`` reads or writes in ``state``."""
    names = {e.dst_conn: e.data.data for e in state.in_edges(node) if e.dst_conn and not e.data.is_empty()}
    names.update({e.src_conn: e.data.data for e in state.out_edges(node) if e.src_conn and not e.data.is_empty()})
    return names


def replicated_reads(state: SDFGState, node: nodes.NestedSDFG, reads: dict[str, list[list[Any]]]) -> bool:
    """Add what every tile runs of ``node`` outside its maps to ``reads``, in the enclosing names and symbols:
    the tasklets beside the maps (:func:`inner_nests` admits them only while they write nothing from
    outside) and its control flow (:func:`control_flow_reads`). What they read must be no other tile's write.

    :returns: ``False`` if a view of ``node`` cannot be resolved (:func:`view_bindings`).
    """
    outer_name = outer_names(state, node)
    record_control_flow_reads(node.sdfg, outer_name, reads, node.symbol_mapping)
    for inner_state in node.sdfg.states():
        bindings = view_bindings(inner_state)
        if bindings is None:
            return False
        scope = inner_state.scope_dict()
        for tasklet in inner_state.nodes():
            if not isinstance(tasklet, nodes.Tasklet) or scope[tasklet] is not None:
                continue
            for edge in inner_state.in_edges(tasklet):
                if edge.data.is_empty():
                    continue
                name, indices = resolved(edge.data.data, index_expressions(edge.data.subset), bindings, node.sdfg)
                if name in outer_name:
                    reads.setdefault(outer_name[name], []).append(rewritten(indices, node.symbol_mapping))
    return True


def tile_local(reads: dict[str, list[list[Any]]], writes: dict[str, list[list[Any]]]) -> bool:
    """Whether every dependence among ``reads`` / ``writes`` stays inside one tile.

    A tile owns a slice of the CUT axis (``axis_symbol(0)``, the innermost parameter of every tiled
    map) and every value of the others, so only the dimensions of a write that name the cut axis
    decide which tile an element belongs to; a read must name the same position there. Those
    dimensions must not mix in another axis either: ``a[i + j]`` is written by many ``i`` for one
    element, and so by more than one tile.

    :param reads: ``name -> [index expression lists]``, on the tile's own axes.
    :param writes: the same for the writes.
    :returns: ``True`` if no dependence crosses a tile boundary.
    """
    cut = {str(axis_symbol(0))}
    for name, write_list in writes.items():
        if name not in reads:
            continue  # written but never read back: no carry to keep inside a tile
        for write_indices in write_list:
            # Which dimensions the tile is cut along is a property of the WRITE alone, so it is
            # decided once per write rather than per read.
            cut_dims = [i for i, expr in enumerate(write_indices) if names_a_param(expr, cut)]
            # A destination naming NO cut position is one location every tile writes -- the
            # ``s115`` shared scalar. Tiling races on it whatever the distances are.
            if not cut_dims:
                return False
            if any(
                {str(s) for s in write_indices[dim].free_symbols if str(s).startswith(AXIS_PREFIX)} != cut
                for dim in cut_dims
            ):
                return False
            for read_indices in reads[name]:
                if len(read_indices) != len(write_indices):
                    return False
                # Only the cut dimensions constrain the tile. A dimension indexed by the loop
                # variable alone IS the carry, and any distance there is fine -- one tile runs
                # those trips in order.
                for dim in cut_dims:
                    if not same_index(read_indices[dim], write_indices[dim]):
                        return False
    return True


def replicable(tasklet: nodes.Tasklet, state: SDFGState, private: set[str]) -> bool:
    """Whether every tile may run the top-level ``tasklet`` itself: it has no side effect and writes only
    loop-local scalars, which each tile keeps its own copy of. What it reads is checked with the maps.

    :param tasklet: a tasklet beside the maps of a loop body.
    :param state: the state holding it.
    :param private: the loop-local transients outlining privatizes.
    :returns: ``True`` if running it once per tile is the same as running it once.
    """
    if tasklet.has_side_effects(state.sdfg):
        return False
    outputs = state.out_edges(tasklet)
    return bool(outputs) and all(
        isinstance(edge.dst, nodes.AccessNode)
        and edge.dst.data in private
        and edge.dst.desc(state.sdfg).total_size == 1
        for edge in outputs
    )


def collect_replicated_reads(
    state: SDFGState, top: list[nodes.Node], private: set[str], reads: dict[str, list[list[Any]]]
) -> bool:
    """Add the reads of the tasklets among ``top`` -- the statements beside the maps -- to ``reads``.

    Such a statement runs in EVERY tile, with no barrier to order it. That is the same as running it once
    only if what it writes is each tile's own (:func:`replicable`) and what it reads no other tile writes
    in the meantime -- ``s115``'s scalar, written by one tile and read by all, is the shape that must
    refuse, and it does so in :func:`tile_local` through the reads recorded here.

    :returns: ``False`` if one of them is not replicable.
    """
    bindings = view_bindings(state)
    if bindings is None:
        return False
    for node in top:
        if not isinstance(node, nodes.Tasklet):
            continue
        if not replicable(node, state, private):
            return False
        for edge in state.in_edges(node):
            if not edge.data.is_empty():
                name, indices = resolved(edge.data.data, index_expressions(edge.data.subset), bindings, state.sdfg)
                reads.setdefault(name, []).append(indices)
    return True


def tile_slice(index_lists: list[list[Any]]) -> bool:
    """Whether every access of ``index_lists`` names one tile's own slice: one dimension carries the SAME cut
    position in all of them (:func:`same_index`), naming the cut axis and no other, so two tiles never touch one
    element."""
    cut = {str(axis_symbol(0))}
    if not index_lists:
        return False
    first = index_lists[0]
    for dim, index in enumerate(first):
        named = {
            str(s)
            for s in (as_basic(index).free_symbols if isinstance(index, (sympy.Basic, symbolic.SymExpr)) else ())
            if str(s).startswith(AXIS_PREFIX)
        }
        if named != cut:
            continue
        if all(len(other) == len(first) and same_index(other[dim], index) for other in index_lists):
            return True
    return False


def within_statements(region: LoopRegion, name: str, cut: "Cut") -> bool:
    """Whether every access node of ``name`` in ``region`` is one a cut statement covers: inside one of its map
    scopes, or beside one with every edge into its entry or out of its exit. What the statements read and write
    there is recorded element by element in ``cut``; an access anywhere else is not."""
    entries = {id(statement.entry) for statement in cut.statements}
    for block in region.all_control_flow_blocks():
        if not isinstance(block, SDFGState):
            continue
        scope = block.scope_dict()
        for node in block.data_nodes():
            if node.data != name:
                continue
            outermost = scope[node]
            while outermost is not None and scope[outermost] is not None:
                outermost = scope[outermost]
            if outermost is not None:
                if id(outermost) not in entries:
                    return False
                continue
            for edge in block.all_edges(node):
                other = edge.dst if edge.src is node else edge.src
                entry = block.entry_node(other) if isinstance(other, nodes.ExitNode) else other
                if id(entry) not in entries:
                    return False
    return True


def sliced_locals(loop: LoopRegion, sdfg: SDFG, cut: "Cut") -> set[str]:
    """The loop-local arrays each tile touches only its own slice of, which may stay outside the tile nest: one
    buffer for every tile instead of a whole-extent copy per tile call.

    Every access of such an array is one the cut was proven on (:func:`within_statements`), nested SDFGs' element
    accesses included, and all of them name one tile's slice (:func:`tile_slice`). A branch or a symbol every tile
    derives itself names no slice.

    :param loop: the loop region about to be tiled.
    :param sdfg: the SDFG owning ``loop``.
    :param cut: the approved cut.
    :returns: their names.
    """
    sliced = set()
    branch_reads = {name for name, _ in control_flow_reads(loop)}
    for name in loop_local_transients(sdfg, loop) - branch_reads:
        accesses = cut.reads.get(name, []) + cut.writes.get(name, [])
        if name in cut.writes and tile_slice(accesses) and within_statements(loop, name, cut):
            sliced.add(name)
    return sliced


def extent(dimension: tuple[Any, Any, Any]) -> Any:
    """The number of positions ``(begin, end, step)`` covers at unit step."""
    begin, end, _ = dimension
    return end - begin + 1


class Cut(NamedTuple):
    """What :func:`tileable_maps` approved: the statements, the extent of the axis they share, and whether that
    axis is innermost (strip-mined into tiles) or not (moved outside the carry whole)."""

    statements: list[Statement]
    size: Any
    innermost: bool
    bytes_per_position: int
    #: Every array's per-element accesses in the statements, ``name -> [index lists]`` on the tiles' axes
    #: (:func:`collect_tile_accesses`), nested SDFGs' included under their enclosing names.
    reads: dict[str, list[list[Any]]] = {}  # noqa: RUF012 -- read only
    writes: dict[str, list[list[Any]]] = {}  # noqa: RUF012 -- read only


def tileable_maps(loop: LoopRegion, sdfg: SDFG) -> Cut | None:
    """The worksharing maps of ``loop`` and the axis they may be tiled along, or ``None`` if there is none.

    :param loop: the candidate loop region, shared out.
    :param sdfg: the SDFG owning ``loop``.
    :returns: the approved cut, or ``None``.
    """
    reads: dict[str, list[list[Any]]] = {}
    record_control_flow_reads(loop, {name: name for name in sdfg.arrays}, reads)
    private = loop_local_transients(sdfg, loop)
    shapes: list[Shape] = []
    for block in loop.all_control_flow_blocks():
        if not isinstance(block, SDFGState):
            continue
        # ONE scope walk for the whole block. ``entry_node`` and ``exit_node`` each rebuild the
        # scope from scratch, so asking them per node turns a linear pass over the state into a
        # quadratic one -- and this runs over every candidate loop in the graph.
        scope = block.scope_dict()
        children = block.scope_children()
        if not collect_replicated_reads(block, children[None], private, reads):
            return None
        for node in block.nodes():
            if not isinstance(node, nodes.MapEntry) or node.map.schedule != WORKSHARED:
                continue
            if scope[node] is not None:
                return None  # a nested worksharing map: the outer one is the statement, not this
            map_exit = next(n for n in children[node] if isinstance(n, nodes.MapExit))
            nest = map_nest(node, children)
            shapes.append(Shape(block, node, map_exit, nest, nest_axes(nest), inner_nests(block, node)))
    defined = loop_defined_symbols(loop)
    for size, innermost in axis_choices(shapes, SymbolResolver().facts_at(loop)):
        # The tile count is the team map's range, evaluated before the loop runs: an extent naming a symbol the
        # loop assigns (a CSR row's bounds, read per trip) has no value there.
        if {str(name) for name in symbolic.free_symbols_and_functions(size)} & defined:
            continue
        cut = cut_along(shapes, size, innermost, reads, sdfg)
        if cut is not None:
            return cut
    return None


def cut_along(
    shapes: list[Shape], size: Any, innermost: bool, replicated: dict[str, list[list[Any]]], sdfg: SDFG
) -> Cut | None:
    """The cut of ``shapes`` along their axis of extent ``size``, or ``None`` if a dependence crosses tiles.

    :param shapes: the statements of the loop body.
    :param size: the extent to cut.
    :param innermost: whether that axis is innermost in every statement.
    :param replicated: the reads of the statements beside the maps (:func:`collect_replicated_reads`).
    :param sdfg: the SDFG owning the loop.
    """
    reads = {name: list(lists) for name, lists in replicated.items()}
    writes: dict[str, list[list[Any]]] = {}
    statements: list[Statement] = []
    for shape in shapes:
        statement = Statement(shape.state, shape.entry, required(targets_for(shape, size)))
        if not collect_tile_accesses(statement, shape, reads, writes):
            return None
        statements.append(statement)
    # Tiling drops the barrier BETWEEN two worksharing maps of the body as well as the one per trip, so
    # what one map writes and the next reads at a neighbouring index is a cross-tile dependence exactly as a
    # loop-carried one is -- and a per-map test never compares two maps against each other. The jacobi
    # stencils are that shape: ``B[i+1] = f(A[i..i+2])`` then ``A[i+1] = f(B[i..i+2])``. Hence ONE test
    # over every map that would be tiled.
    if not tile_local(reads, writes):
        return None
    cut_axis = {str(axis_symbol(0))}
    touched = {
        name
        for accesses in (reads, writes)
        for name, index_lists in accesses.items()
        if any(names_a_param(index, cut_axis) for indices in index_lists for index in indices)
    }
    footprint = sum(sdfg.arrays[name].dtype.bytes for name in touched if name in sdfg.arrays)
    # Every array's element accesses, the statements' own scratch included, for :func:`sliced_locals`; a scope
    # that cannot be read leaves them empty, and so nothing sliced.
    elements: tuple[dict[str, list[list[Any]]], dict[str, list[list[Any]]]] = ({}, {})
    for shape, statement in zip(shapes, statements, strict=True):
        if not collect_tile_accesses(statement, shape, *elements, every_array=True):
            elements = ({}, {})
            break
    return Cut(statements, size, innermost, footprint, *elements)


def same_extent(left: Any, right: Any) -> bool:
    """Whether two extents are provably equal; undecided reads as unequal."""
    return str(left) == str(right) or symbolic.equal(left, right) is True


#: Bounds of a tile of an innermost axis. Powers of two, so every SIMD width up to the lower bound divides a
#: tile and only the last one carries a remainder.
MIN_TILE = 16
MAX_TILE = 256


def tile_size(cut: Cut) -> Any:
    """How many positions of the cut axis one tile owns.

    An axis moved outside the carry whole takes one position per tile: the innermost axis inside is the
    vector, and many tiles per thread balance the load. An innermost axis is strip-mined: the tile is the
    largest power of two in ``[MIN_TILE, MAX_TILE]`` whose working set -- one element per position of every
    container the body touches along the axis -- fills at most half of one core's L2, then evened out so
    every thread gets the same number of tiles: with ``r`` the rounds of ``P`` cache tiles the axis needs, a
    tile is ``ceil(size / (P * r))`` rounded up to a multiple of ``MIN_TILE``, so every tile starts on a
    vector boundary (unaligned, CLOUDSC at M ran 7-11% slower on 8 and 24 threads). One contiguous share per thread (``r = 1`` on a long axis) loses to the
    cache tile at every team size on CLOUDSC at M (4187 ms against 3247 ms on one thread, 345 ms against
    237 ms on 24); the cache tile alone leaves threads idle -- 768 columns are three tiles of 256 for twelve
    threads on ``s231``, twice as slow -- or a round half full: 4096 columns are sixteen for twelve.
    """
    if not cut.innermost:
        return 1
    positions = topology().l2_bytes // (2 * max(cut.bytes_per_position, 1))
    tile = MIN_TILE
    while tile * 2 <= min(positions, MAX_TILE):
        tile *= 2
    rounds = symbolic.int_ceil(cut.size, THREADS_SYMBOL * tile)
    return MIN_TILE * symbolic.int_ceil(symbolic.int_ceil(cut.size, THREADS_SYMBOL * rounds), MIN_TILE)


def cut_into_tiles(statement: Statement, tile: Any, size: Any) -> None:
    """Narrow every map ``statement``'s tiles cut to the ``size`` positions tile ``tile`` owns; the last tile
    stops at the original end. A map in a nested SDFG reads the tile parameter and the symbols of ``size``
    (the thread count) through its ``symbol_mapping``.

    :param statement: the statement whose targets are narrowed in place.
    :param tile: the tile loop's parameter symbol.
    :param size: positions per tile.
    """
    for target in statement.targets:
        owner = target.owner
        ranges = list(owner.map.range.ranges)
        position = owner.map.params.index(target.param)
        begin, end, step = ranges[position]
        last = begin + (tile + 1) * size - 1
        ranges[position] = (begin + tile * size, last if size == 1 else symbolic.sympy.Min(last, end), step)
        owner.map.range = subsets.Range(ranges)
        if target.nested is not None:
            declared = statement.state.sdfg.symbols
            for symbol in [TILE_SYMBOL, *symbolic.symlist(size).values()]:
                target.nested.symbol_mapping[symbol.name] = symbol
                if symbol.name not in target.nested.sdfg.symbols:
                    target.nested.sdfg.add_symbol(symbol.name, declared.get(symbol.name, NUM_THREADS_DTYPE))


@properties.make_properties
@transformation.explicit_cf_compatible
class StripMineInterchange(HoistParallelRegion):
    """Strip-mine the parallel axis of a carried loop nest and share the tiles out around the carry: one
    barrier for the nest, not one per trip.

    The traversal, condition (H) and the shared-out statements are inherited from
    :class:`~dace.transformation.passes.cpu_specialization.hoist_parallel_region.HoistParallelRegion`.
    Condition (T) is not: a tile runs its whole carry on one thread, so a transient handed from one
    map to the next stays with the thread that wrote it. Arrays still stay outside the nest -- shared,
    each tile touching its own slice, allocated once -- and loop-local scalars are privatized, which
    is what lets every tile run the statements beside the maps itself. This subclass adds the tile
    legality test and reshapes what the outlining produced. See the module docstring.
    """

    CATEGORY: str = "Optimization"

    def __init__(self):
        super().__init__()
        #: The maps :meth:`hoistable` approved, kept for the :meth:`hoist` that follows it. The
        #: walk asks the predicate and then immediately rewrites the same loop, so ONE entry is
        #: enough and the analysis is never run twice over a graph.
        self._approved = None

    def hoistable(self, loop: LoopRegion, sdfg: SDFG) -> bool:
        """Tile locality of the replication-free, shared-out ``loop``.

        :param loop: the candidate loop region.
        :param sdfg: the SDFG owning ``loop``.
        :returns: ``True`` if the loop may be tiled.
        """
        targets = tileable_maps(loop, sdfg)
        self._approved = (loop, targets) if targets is not None else None
        return targets is not None

    def hoist(self, loop: LoopRegion, sdfg: SDFG) -> None:
        """Outline ``loop`` as the parent does, then turn the team into the tile loop.

        :param loop: the loop region to tile; must satisfy :meth:`hoistable`.
        :param sdfg: the SDFG owning ``loop``.
        """
        approved_loop, cut = self._approved if self._approved is not None else (None, None)
        if approved_loop is not loop:
            cut = tileable_maps(loop, sdfg)  # asked out of order: recompute rather than trust a stale cut
        self._approved = None
        cut = required(cut)
        size = tile_size(cut)
        for statement in cut.statements:
            # Narrowed BEFORE outlining, while the maps are still reachable from here: the outlining
            # moves these states into a nested SDFG and the node handles would go stale.
            cut_into_tiles(statement, TILE_SYMBOL, size)
            # The tile loop is the worksharing construct now; everything under it is one thread's.
            for entry in map_nest(statement.entry, statement.state.scope_children()):
                entry.map.schedule = dtypes.ScheduleType.Sequential
            for target in statement.targets:
                if target.nested is not None:
                    for inner, _ in target.nested.sdfg.all_nodes_recursive():
                        if isinstance(inner, nodes.MapEntry):
                            inner.map.schedule = dtypes.ScheduleType.Sequential
        # The outlining replaces ``loop`` with a state in the region that held it, so the map it
        # creates is in THAT region -- searching the whole SDFG, twice, would make the pass
        # quadratic in graph size for a node whose location is already known.
        parent = loop.parent_graph
        before = {id(n) for n in map_entries_of(parent)}
        shared = {name for name in top_level_loop_locals(sdfg, loop) if sdfg.arrays[name].total_size != 1}
        # A loop-local array each tile touches only its own slice of stays outside too, allocated once. Outlined,
        # every tile call allocated a private copy of the WHOLE extent: CLOUDSC's (klon)-wide scratch, one set per
        # thread, ran past the kernel's memory cap to use one tile's columns of it.
        defined = loop_defined_symbols(loop)
        shared |= {
            name
            for name in sliced_locals(loop, sdfg, cut)
            if not isinstance(sdfg.arrays[name], data.View)
            and sdfg.arrays[name].total_size != 1
            and not {str(symbol) for symbol in sdfg.arrays[name].free_symbols} & defined
        }
        outline(loop, sdfg, shared)
        team, state = new_team_map(parent, before)
        if team is None:
            # The outlining names the SDFG rather than the region holding the loop, so for a loop
            # nested in another region the new state can land a level up. Rare, and a full walk is
            # affordable once it happens -- unlike doing it unconditionally.
            team, state = new_team_map(sdfg, before, recursive=True)
        assert team is not None, "the outlining produced no team map to widen into tiles"
        # The unit team map becomes the tile loop itself, worksharing, so codegen emits one
        # ``#pragma omp parallel for`` and no inner barrier survives.
        team.map.params = [TILE_PARAM]
        team.map.range = subsets.Range([(0, symbolic.int_ceil(cut.size, size) - 1, 1)])
        team.map.schedule = WORKSHARED
        # The narrowed maps sit inside the nested SDFG the outlining made, so the tile parameter
        # has to be handed across that boundary like any other symbol.
        for node in required(state).scope_subgraph(team).nodes():
            if isinstance(node, nodes.NestedSDFG) and node.sdfg is not None:
                node.symbol_mapping[TILE_PARAM] = TILE_SYMBOL
                if TILE_PARAM not in node.sdfg.symbols:
                    node.sdfg.add_symbol(TILE_PARAM, NUM_THREADS_DTYPE)


def new_team_map(
    region: AbstractControlFlowRegion, before: set[int], recursive: bool = False
) -> tuple[nodes.MapEntry | None, SDFGState | None]:
    """The ``CPU_Persistent`` map the outlining just added, with the state holding it.

    :param region: where to look.
    :param before: ids of the map entries that existed before the outlining.
    :param recursive: descend into nested SDFGs as well.
    :returns: ``(map_entry, state)``, or ``(None, None)``.
    """
    blocks = region.states() if recursive else [b for b in region.nodes() if isinstance(b, SDFGState)]
    for block in blocks:
        for node in block.nodes():
            if (
                isinstance(node, nodes.MapEntry)
                and id(node) not in before
                and node.map.schedule == dtypes.ScheduleType.CPU_Persistent
            ):
                return node, block
    return None, None


def states_and_map_entries(region: AbstractControlFlowRegion) -> Iterator[tuple[SDFGState, nodes.MapEntry]]:
    """``(state, map_entry)`` for every map directly inside ``region``'s own states.

    Deliberately NOT recursive: the caller is looking for a node the outlining just added beside
    the loop it replaced, which is always at this level.
    """
    for block in region.nodes():
        if isinstance(block, SDFGState):
            for node in block.nodes():
                if isinstance(node, nodes.MapEntry):
                    yield block, node


def map_entries_of(region: AbstractControlFlowRegion) -> Iterator[nodes.MapEntry]:
    """Every map entry directly inside ``region``'s own states."""
    for _, node in states_and_map_entries(region):
        yield node
