# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Give each thread a BAND of the parallel axis for the whole carry, not a barrier per trip.

A sequential loop wrapped around a parallel map is the recurrence-sweep shape (TSVC ``s231`` /
``s235`` / ``s275``: ``aa[j, i] = aa[j-1, i] + ...``, ``i`` the contiguous parallel column, ``j`` the
carry). :class:`~dace.transformation.passes.cpu_specialization.hoist_parallel_region.HoistParallelRegion`
opens the OpenMP team once for such a nest, but the worksharing construct inside stays per trip::

    #pragma omp parallel                       <- one fork/join
    for (j = 1; j < N; ++j) {
        #pragma omp for simd                   <- an implicit barrier, ONCE PER TRIP
        for (i = 0; i < N; ++i) aa[j][i] = aa[j-1][i] + bb[j][i];
    }

At the sizes these kernels run the trip count IS the array extent, so the nest pays ``N-1`` barriers
to hand each thread about ``N/P`` elements of streaming work between them. Measured on ``s231`` at
``LEN_2D=11966``, 24 threads: 11,965 barriers cost 24.5 ms of a 41.0 ms kernel, ~2.05 us each.

This pass emits the same computation with ONE barrier, by cutting the parallel axis into bands and
giving a whole band's carry to one thread::

    #pragma omp parallel for                   <- one barrier, at the very end
    for (t = 0; t < __dace_num_threads; ++t)
        for (j = 1; j < N; ++j)
            #pragma omp simd                   <- i still innermost, still contiguous
            for (i = lo(t); i < hi(t); ++i) aa[j][i] = aa[j-1][i] + bb[j][i];

Measured on ``s231`` XL, 24 threads: 41.0 ms -> 16.5 ms.

Not more parallel -- differently synchronised
---------------------------------------------

The parallel DEGREE is unchanged: the same ``N`` independent columns over the same ``P`` threads
doing the same work. What changes is that the parallelism is expressed as ``P`` tasks that
rendezvous once instead of ``N-1`` worksharing regions that rendezvous every trip. Per-trip load
imbalance is likewise paid once instead of ``N-1`` times, which is the second and smaller part.

Why not simply interchange
--------------------------

:class:`~dace.transformation.passes.canonicalize.move_loop_into_map_gated.MoveLoopIntoMapGated`
already turns ``for(seq) { map }`` into ``map { for(seq) }``, and DECLINES it on CPU for exactly
these kernels: hoisting the whole map puts ``j`` innermost at stride ``N``, one cache line per
element. That decline is right about stride and wrong about the alternative, because it picks
between two orders when the useful form is a third -- lift only a TILE of the parallel axis and the
contiguous remainder stays inside. (On GPU that pass always interchanges, which is this same shape
at a band size of one: one thread per column, consecutive threads on consecutive ``i``. Only the
CPU end was missing.)

Legality
--------

A band runs its whole carry alone -- every trip of ``j`` for the columns it owns, in order, with no
other thread touching them in between. That is correct iff no dependence crosses a band boundary,
which is a LOCAL test on the map's memlets: every loop-carried dependence must be at distance zero
in the map's own parameters.

- ``aa[j-1, i]`` against ``aa[j, i]`` -- distance 0 in ``i``. Accepted (``s231``, ``s235``,
  ``s275``).
- ``s119``'s ``aa[i-1, j-1]`` and ``wf_diff_skew``'s ``a[i-1, j+1]`` -- distance one in the map
  parameter, so a band would read a neighbour's column. Refused.
- ``s115`` reads a scalar every band needs but one band writes: its destination names no map
  parameter at all, so it is a location shared across bands. Refused.
- The wavefront skew's diagonal writes ``A[t - 2*p]`` from a row assembled in a transient, so the
  carried write leaves as a COPY through the map exit rather than as a tasklet store. The row one
  band writes at diagonal ``t`` is read by the NEXT band at ``t + 1``, so the nest has to be
  refused -- and it is, but only because the boundary edges are read as accesses too
  (:func:`boundary_accesses`); a test that sees tasklet stores alone finds no write for ``A`` at
  all and approves the nest vacuously.
- A body canonicalize left as a nested SDFG carries the whole-map UNION on those same boundary
  edges -- CLOUDSC's ``ztp1[0:N, 0:N]`` and ``zcovptot[0:N]``, the read-only gather's ``a[0:N,
  0:N]``. A union names no map parameter because it is every iteration at once, so it is evidence
  of nothing and must not be read as a per-iteration access; taken as one it looks like ``s115``'s
  shared location and refuses every nest whose body is nested. Only a boundary edge that DOES name
  a map parameter is an access (:func:`boundary_accesses`).

Correctness does NOT depend on how OpenMP distributes the band loop. A band's entire carry sits
inside ONE iteration of the outer map, so whichever thread runs band ``t`` runs all of ``t``'s trips
in order, for any band count and any schedule. That is what this form has over dropping the barrier
with ``nowait``, which is correct only while consecutive worksharing regions hand the same
iterations to the same thread -- a conditional guarantee whose conditions exclude the
``simd``-associated loops canonicalize emits.

Conditions (H) and (T) of ``HoistParallelRegion`` are inherited unchanged: this pass outlines the
loop the same way and wraps it in a map the same way, so the same replication and privatization
rules apply. Only the wrapping map's extent and the inner map's range differ.
"""
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set

from dace import SDFG, dtypes, properties, subsets, symbolic
from dace.ordered import OrderedSet
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion, SDFGState
from dace.transformation import transformation
from dace.transformation.passes.canonicalize.supply_num_threads import DTYPE as NUM_THREADS_DTYPE
from dace.transformation.passes.cpu_specialization.hoist_parallel_region import (WORKSHARED, HoistParallelRegion)

#: Parameter name of the band loop. One name, so the reshape can find the map it just made.
BAND_PARAM = '__dace_band'

#: Built ONCE, and built WITH THEIR DTYPE. DaCe folds dtype and assumptions into symbol identity,
#: so a name minted twice at two widths is two symbols that never cancel; declaring these here at
#: the same ``int64`` the SDFG declaration uses keeps one instance of each name in the graph. One
#: instance is also what makes the reuse cheap -- ``pystr_to_symbolic`` runs the sympy parser, far
#: too expensive to repeat per map, per dimension, per candidate loop.
BAND_SYMBOL = symbolic.symbol(BAND_PARAM, NUM_THREADS_DTYPE)
THREADS_SYMBOL = symbolic.symbol(symbolic.NUM_THREADS_SYMBOL, NUM_THREADS_DTYPE)


@lru_cache(maxsize=None, typed=True)
def axis_symbol(position: int):
    """The stand-in for a banded map's parameter ``position`` places out from its innermost.

    Every map of one loop body is cut the same way, so its parameter at a given position denotes
    the same band as another map's at that position. The two maps spell it with different names
    (``_loop_it_1`` against ``_loop_it_4``), and comparing the spellings would read one location as
    two unrelated ones, which is how a cross-map dependence goes unseen.
    """
    return symbolic.symbol(f'__dace_band_axis{position}', NUM_THREADS_DTYPE)


def index_expressions(subset) -> List[Any]:
    """The per-dimension index expression of ``subset``, one entry per dimension.

    A :class:`~dace.subsets.Range` carries ``(begin, end, step)`` triples and a
    :class:`~dace.subsets.Indices` bare expressions; the band test reads where a dimension starts
    in either.
    """
    if isinstance(subset, subsets.Indices):
        return list(subset.indices)
    if isinstance(subset, subsets.Range):
        return [begin for begin, _, _ in subset.ranges]
    return []


def names_a_param(expr, params: Set[str]) -> bool:
    """Whether ``expr`` mentions any of ``params``, which are symbol NAMES.

    By name, never by symbol object. Two ``dace.symbolic.symbol`` instances that share a name but
    carry different sympy assumptions -- which is what a subset rewritten through a nested SDFG's
    ``symbol_mapping`` ends up holding against a freshly parsed map parameter -- are unequal, so an
    intersection of symbol sets silently finds nothing and every real kernel is refused.
    """
    return bool(symbolic.issymbolic(expr) and ({str(s) for s in expr.free_symbols} & params))


def same_index(left, right) -> bool:
    """Whether two index expressions denote the same position.

    The printed forms are compared first: the accepted shape carries the SAME expression on both
    sides of a cut dimension, and that answers without touching sympy at all.

    Anything else goes to :func:`~dace.symbolic.equal`, which equalizes the two expressions'
    same-named symbols onto one instance before comparing. That step is the whole reason not to
    subtract them directly, and not to reparse them into fresh ones: these subsets come from
    different sources -- one rewritten through a nested SDFG's ``symbol_mapping``, one straight off
    a memlet -- and dace folds dtype and assumptions into symbol identity, so two instances of the
    same NAME do not cancel. ``equal`` answers ``None`` when it cannot decide, which is read here
    as "not the same position": an undecided distance must refuse the band, never take it.
    """
    if left is right or str(left) == str(right):
        return True
    # is_length=False: an index may be zero or negative (``jk - 1`` at the top of a sweep), so the
    # positive-integer assumptions ``equal`` makes for extents would be wrong here.
    return symbolic.equal(left, right, is_length=False) is True


def boundary_names(state: SDFGState, map_entry: nodes.MapEntry, map_exit: nodes.MapExit) -> Set[str]:
    """The arrays that cross ``map_entry``'s scope boundary.

    Only these can carry a value from one trip of the enclosing loop to the next. Everything else
    under the map -- the per-iteration scalars a body is built from, ``aa_index`` and friends --
    is created and consumed inside ONE iteration, so it is privatized by construction and says
    nothing about whether a band is self-contained. Including them would refuse every real kernel,
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


def collect_accesses(state: SDFGState, scope_node, allowed: Set[str], reads: Dict, writes: Dict) -> bool:
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
    :returns: ``False`` if something in the scope cannot be analysed, in which case the caller
              must refuse rather than trust a partial picture.
    """
    for node in state.scope_subgraph(scope_node, include_entry=False, include_exit=False).nodes():
        if isinstance(node, nodes.Tasklet):
            for edge in state.in_edges(node):
                if edge.data is not None and edge.data.data in allowed and edge.data.subset is not None:
                    reads.setdefault(edge.data.data, []).append(index_expressions(edge.data.subset))
            for edge in state.out_edges(node):
                if edge.data is not None and edge.data.data in allowed and edge.data.subset is not None:
                    writes.setdefault(edge.data.data, []).append(index_expressions(edge.data.subset))
        elif isinstance(node, nodes.NestedSDFG):
            if node.sdfg is None or not descend_into_nested(state, node, allowed, reads, writes):
                return False
        elif isinstance(node, nodes.LibraryNode):
            return False  # expands to whatever it likes; its accesses are not known here
    return True


def descend_into_nested(state: SDFGState, node: nodes.NestedSDFG, allowed: Set[str], reads: Dict, writes: Dict) -> bool:
    """Add the accesses inside ``node``, translated into the enclosing graph's names and symbols.

    :param state: the state holding ``node``.
    :param node: the nested SDFG to descend into.
    :param allowed: the arrays that cross the enclosing scope boundary.
    :param reads: ``name -> [index expression lists]``, extended in place.
    :param writes: ``name -> [index expression lists]``, extended in place.
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
    # ... and its own symbols, so an inner subset has to be rewritten before it can be compared
    # against an outer map parameter. ``symbol_mapping`` is keyed by NAME, and the rewrite below
    # keys on the symbol instances the expression actually holds -- minting a key from the name
    # here would produce a symbol whose dtype and assumptions need not match the one in the
    # expression, and ``subs`` matches on identity, so the substitution would quietly do nothing
    # and leave inner names to be compared against outer ones.
    substitution = node.symbol_mapping
    for inner_state in node.sdfg.states():
        for inner_node in inner_state.nodes():
            if isinstance(inner_node, nodes.NestedSDFG):
                return False  # one level is what the connector translation above covers
            if not isinstance(inner_node, nodes.AccessNode):
                continue
            name = outer_name.get(inner_node.data)
            if name is None or name not in allowed:
                continue  # private to the nested graph, so it cannot carry across trips
            for edge in inner_state.in_edges(inner_node):
                if edge.data is not None and edge.data.subset is not None:
                    writes.setdefault(name, []).append(substituted(edge.data.subset, substitution))
            for edge in inner_state.out_edges(inner_node):
                if edge.data is not None and edge.data.subset is not None:
                    reads.setdefault(name, []).append(substituted(edge.data.subset, substitution))
    return True


def substituted(subset, substitution: Dict[str, Any]) -> List:
    """``subset``'s index expressions rewritten through ``substitution``, which is keyed by NAME.

    Each expression is replaced against the symbol instances it actually carries, looked up by
    name, so the rewrite cannot be defeated by two same-named symbols that differ in dtype or
    assumptions -- ``subs`` matches on identity, and a key minted from the name need not be the
    instance in the expression.
    """
    return rewritten(index_expressions(subset), substitution)


def rewritten(expressions: List, substitution: Dict[str, Any]) -> List:
    """``expressions`` rewritten through ``substitution``, keyed by NAME; see :func:`substituted`."""
    out = []
    for expr in expressions:
        if not symbolic.issymbolic(expr):
            out.append(expr)
            continue
        replacements = {sym: substitution[sym.name] for sym in expr.free_symbols if sym.name in substitution}
        out.append(expr.subs(replacements) if replacements else expr)
    return out


def boundary_accesses(state: SDFGState, map_entry: nodes.MapEntry, map_exit: nodes.MapExit, allowed: Set[str],
                      reads: Dict[str, List[List[Any]]], writes: Dict[str, List[List[Any]]]) -> bool:
    """Add the PER-ITERATION accesses the scope BOUNDARY edges carry, whatever produced them.

    :func:`collect_accesses` reads the subsets adjacent to code nodes, so a value assembled in a
    transient and copied out through the map exit is recorded nowhere -- and a missing WRITE makes
    :func:`band_local` vacuous: it iterates the writes, finds none for the array, and approves a
    nest whose whole carry crosses every band. The wavefront skew emits exactly that shape -- a
    diagonal's row is built in a transient and copied into ``A`` -- and the banded form raced at
    four threads. Every value a band could carry crosses this boundary, and its subset is on the
    boundary edge whatever assembled it, so reading these edges closes the hole for every copy
    shape at once.

    A boundary edge is such an access only while it NAMES A MAP PARAMETER. Where the body is a
    nested SDFG the same edge carries the propagated union over the whole map instead --
    ``ztp1[0:N, 0:N]`` -- which is every iteration at once and so says nothing about any one of
    them. :func:`index_expressions` reduces that union to its begins ``[0, 0]``, and
    :func:`band_local` then reads a union WRITE as ``s115``'s location shared by every band and a
    union READ as column zero of a per-column write. Both refuse, and both are wrong: CLOUDSC's
    vertical carry and a gather on a read-only operand are distance-0 in the cut axis and must
    band. So a union is dropped rather than trusted.

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
    params = OrderedSet(map_entry.map.params)
    union_writes: List[str] = []
    directions = ((state.out_edges(map_entry), reads, False), (state.in_edges(map_exit), writes, True))
    for edges, target, written in directions:
        for edge in edges:
            if edge.data is None or edge.data.data not in allowed or edge.data.subset is None:
                continue
            indices = index_expressions(edge.data.subset)
            if any(names_a_param(index, params) for index in indices):
                target.setdefault(edge.data.data, []).append(indices)
            elif written:
                union_writes.append(edge.data.data)
    # Read AFTER the loop, so an array carrying both a union edge and a per-iteration one is
    # accounted for whichever order the exit edges come in.
    return all(name in writes for name in union_writes)


def collect_band_accesses(state: SDFGState, map_entry: nodes.MapEntry, map_exit: nodes.MapExit, reads: Dict,
                          writes: Dict) -> bool:
    """Add ``map_entry``'s per-element accesses to ``reads`` / ``writes``, on the band's own axes.

    The map's parameters are rewritten to the positional stand-ins (:func:`axis_symbol`) so that the
    accesses of two DIFFERENT maps of one body land in the same coordinates and can be compared.

    :param state: the state holding the map.
    :param map_entry: the worksharing map whose axis would be cut.
    :param map_exit: ``map_entry``'s exit, passed in because resolving it costs a scope walk.
    :param reads: ``name -> [index expression lists]``, extended in place.
    :param writes: ``name -> [index expression lists]``, extended in place.
    :returns: ``False`` if something in the scope cannot be analysed.
    """
    local_reads: Dict[str, List] = {}
    local_writes: Dict[str, List] = {}
    allowed = boundary_names(state, map_entry, map_exit)
    if not collect_accesses(state, map_entry, allowed, local_reads, local_writes):
        return False
    if not boundary_accesses(state, map_entry, map_exit, allowed, local_reads, local_writes):
        return False
    # Counted from the INNERMOST parameter, because that is the one ``cut_into_bands`` slices: two
    # maps of different rank still have to agree on which axis the band owns.
    depth = len(map_entry.map.params)
    renaming = {param: axis_symbol(depth - 1 - position) for position, param in enumerate(map_entry.map.params)}
    for source, target in ((local_reads, reads), (local_writes, writes)):
        for name, index_lists in source.items():
            target.setdefault(name, []).extend(rewritten(indices, renaming) for indices in index_lists)
    return True


def band_local(reads: Dict, writes: Dict, params: Set[str]) -> bool:
    """Whether every dependence among ``reads`` / ``writes`` stays inside one band.

    :param reads: ``name -> [index expression lists]``, on the band's own axes.
    :param writes: the same for the writes.
    :param params: the stand-in axis names the indices may mention.
    :returns: ``True`` if no dependence crosses a band boundary.
    """
    for name, write_list in writes.items():
        if name not in reads:
            continue  # written but never read back: no carry to keep inside a band
        for write_indices in write_list:
            # Which dimensions the band is cut along is a property of the WRITE alone, so it is
            # decided once per write rather than per read.
            cut_dims = [i for i, expr in enumerate(write_indices) if names_a_param(expr, params)]
            # A destination naming NO map parameter is one location every band writes -- the
            # ``s115`` shared scalar. Banding races on it whatever the distances are.
            if not cut_dims:
                return False
            for read_indices in reads[name]:
                if len(read_indices) != len(write_indices):
                    return False
                # Only the cut dimensions constrain the band. A dimension indexed by the loop
                # variable alone IS the carry, and any distance there is fine -- one band runs
                # those trips in order.
                for dim in cut_dims:
                    if not same_index(read_indices[dim], write_indices[dim]):
                        return False
    return True


def bandable_maps(loop: LoopRegion) -> Optional[List]:
    """The worksharing maps of ``loop`` whose axis may be cut, or ``None`` if any may not.

    :param loop: the candidate loop region.
    :returns: ``(state, map_entry)`` pairs to band, or ``None``.
    """
    found = []
    reads: Dict[str, List] = {}
    writes: Dict[str, List] = {}
    params: Set[str] = set()
    cut_ranges: List[Any] = []
    for block in loop.all_control_flow_blocks():
        if not isinstance(block, SDFGState):
            continue
        # ONE scope walk for the whole block. ``entry_node`` and ``exit_node`` each rebuild the
        # scope from scratch, so asking them per node turns a linear pass over the state into a
        # quadratic one -- and this runs over every candidate loop in the graph.
        scope = block.scope_dict()
        children = block.scope_children()
        # A statement beside the map is one the whole team would run. The team hoist repairs that
        # by wrapping it in a one-iteration WORKSHARING map -- run once, with a barrier after --
        # but banding has no barrier to offer it, and its value is one every band goes on to read
        # (``s115``'s scalar). Refuse the shape rather than band around it.
        for node in children[None]:
            if isinstance(node, nodes.Tasklet):
                return None
        for node in block.nodes():
            if not isinstance(node, nodes.MapEntry) or node.map.schedule != WORKSHARED:
                continue
            if scope[node] is not None:
                return None  # a nested worksharing map: the outer one is the axis, not this
            # Unit stride only: the band bounds below divide an extent, and a strided axis would
            # need the division to land on the stride as well. Checked BEFORE the dependence
            # test, which is the expensive one.
            if any(step != 1 for _, _, step in node.map.range.ranges):
                return None
            map_exit = next(n for n in children[node] if isinstance(n, nodes.MapExit))
            if not collect_band_accesses(block, node, map_exit, reads, writes):
                return None
            params |= {str(axis_symbol(position)) for position in range(len(node.map.params))}
            cut_ranges.append(node.map.range.ranges[-1])
            found.append((block, node))
    # Banding drops the barrier BETWEEN two worksharing maps of the body as well as the one per
    # trip, so what one map writes and the next reads at a neighbouring index is a cross-band
    # dependence exactly as a loop-carried one is -- and a per-map test never compares two maps
    # against each other. The jacobi stencils are that shape: ``B[i+1] = f(A[i..i+2])`` then
    # ``A[i+1] = f(B[i..i+2])``. Hence ONE test over every map that would be banded.
    # ``cut_into_bands`` slices each map's OWN cut axis, so two maps own the same elements per band
    # only where that axis carries the same range in both; otherwise alike-printing indices name
    # different bands and the test below would read a real dependence as a local one.
    if any(str(cut) != str(cut_ranges[0]) for cut in cut_ranges[1:]):
        return None
    if not found or not band_local(reads, writes, params):
        return None
    return found


def cut_into_bands(map_entry: nodes.MapEntry, band: Any) -> None:
    """Narrow ``map_entry``'s last dimension to the slice of it that band ``band`` owns.

    The last dimension is the one codegen makes innermost, so cutting there is what keeps each
    band's rows contiguous -- the property the whole rewrite exists to preserve. Bounds mirror
    :class:`~dace.transformation.dataflow.strip_mining.StripMining`'s ``NumberOfTiles`` form; at
    the last band the upper bound lands exactly on the original end, so no ``Min`` is needed.

    :param map_entry: the map to narrow, in place.
    :param band: the band loop's parameter symbol.
    """
    ranges = list(map_entry.map.range.ranges)
    begin, end, step = ranges[-1]
    extent = end - begin + 1
    ranges[-1] = (begin + symbolic.int_floor(extent * band, THREADS_SYMBOL),
                  begin + symbolic.int_floor(extent * (band + 1), THREADS_SYMBOL) - 1, step)
    map_entry.map.range = subsets.Range(ranges)


@properties.make_properties
@transformation.explicit_cf_compatible
class BandCarriedLoops(HoistParallelRegion):
    """Band the parallel axis of a carried loop nest: one barrier for the nest, not one per trip.

    Everything about WHICH loops may be rewritten and HOW they are outlined is inherited from
    :class:`~dace.transformation.passes.cpu_specialization.hoist_parallel_region.HoistParallelRegion`
    -- conditions (H) and (T), the traversal, and the outlining. This subclass adds the band
    legality test and reshapes what the outlining produced. See the module docstring.
    """

    CATEGORY: str = 'Optimization'

    def __init__(self):
        super().__init__()
        #: The maps :meth:`hoistable` approved, kept for the :meth:`hoist` that follows it. The
        #: walk asks the predicate and then immediately rewrites the same loop, so ONE entry is
        #: enough and the analysis is never run twice over a graph.
        self._approved = None

    def hoistable(self, loop: LoopRegion, sdfg: SDFG) -> bool:
        """Conditions (H) and (T), plus band locality.

        :param loop: the candidate loop region.
        :param sdfg: the SDFG owning ``loop``.
        :returns: ``True`` if the loop may be banded.
        """
        # (H) and (T) first: they are node-kind and lifetime checks, where the band test walks
        # memlets and can reach sympy. Cheapest discriminator first.
        if not super().hoistable(loop, sdfg):
            return False
        targets = bandable_maps(loop)
        self._approved = (loop, targets) if targets is not None else None
        return targets is not None

    def hoist(self, loop: LoopRegion, sdfg: SDFG) -> None:
        """Outline ``loop`` as the parent does, then widen the team into bands.

        :param loop: the loop region to band; must satisfy :meth:`hoistable`.
        :param sdfg: the SDFG owning ``loop``.
        """
        approved_loop, targets = self._approved if self._approved is not None else (None, None)
        if approved_loop is not loop:
            targets = bandable_maps(loop)  # asked out of order: recompute rather than trust a stale set
        self._approved = None
        for _, map_entry in targets:
            # Narrowed BEFORE outlining, while the maps are still reachable from here: the parent's
            # hoist moves these states into a nested SDFG and the node handles would go stale.
            cut_into_bands(map_entry, BAND_SYMBOL)
            # The band loop is the worksharing construct now; everything under it is one thread's.
            map_entry.map.schedule = dtypes.ScheduleType.Sequential
        # The outlining replaces ``loop`` with a state in the region that held it, so the map it
        # creates is in THAT region -- searching the whole SDFG, twice, would make the pass
        # quadratic in graph size for a node whose location is already known.
        parent = loop.parent_graph
        before = {id(n) for n in map_entries_of(parent)}
        super().hoist(loop, sdfg)
        team, state = new_team_map(parent, before)
        if team is None:
            # The outlining names the SDFG rather than the region holding the loop, so for a loop
            # nested in another region the new state can land a level up. Rare, and a full walk is
            # affordable once it happens -- unlike doing it unconditionally.
            team, state = new_team_map(sdfg, before, recursive=True)
        assert team is not None, 'the outlining produced no team map to widen into bands'
        # The unit team map becomes the band loop itself: P iterations, worksharing, so codegen
        # emits one ``#pragma omp parallel for`` and no inner barrier survives.
        team.map.params = [BAND_PARAM]
        team.map.range = subsets.Range([(0, THREADS_SYMBOL - 1, 1)])
        team.map.schedule = WORKSHARED
        # The narrowed maps sit inside the nested SDFG the outlining made, so the band parameter
        # has to be handed across that boundary like any other symbol.
        for node in state.scope_subgraph(team).nodes():
            if isinstance(node, nodes.NestedSDFG) and node.sdfg is not None:
                node.symbol_mapping[BAND_PARAM] = BAND_SYMBOL
                if BAND_PARAM not in node.sdfg.symbols:
                    node.sdfg.add_symbol(BAND_PARAM, NUM_THREADS_DTYPE)


def new_team_map(region, before, recursive: bool = False):
    """The ``CPU_Persistent`` map the outlining just added, with the state holding it.

    :param region: where to look.
    :param before: ids of the map entries that existed before the outlining.
    :param recursive: descend into nested SDFGs as well.
    :returns: ``(map_entry, state)``, or ``(None, None)``.
    """
    blocks = (region.all_states() if recursive else [b for b in region.nodes() if isinstance(b, SDFGState)])
    for block in blocks:
        for node in block.nodes():
            if (isinstance(node, nodes.MapEntry) and id(node) not in before
                    and node.map.schedule == dtypes.ScheduleType.CPU_Persistent):
                return node, block
    return None, None


def states_and_map_entries(region):
    """``(state, map_entry)`` for every map directly inside ``region``'s own states.

    Deliberately NOT recursive: the caller is looking for a node the outlining just added beside
    the loop it replaced, which is always at this level.
    """
    for block in region.nodes():
        if isinstance(block, SDFGState):
            for node in block.nodes():
                if isinstance(node, nodes.MapEntry):
                    yield block, node


def map_entries_of(region):
    """Every map entry directly inside ``region``'s own states."""
    for _, node in states_and_map_entries(region):
        yield node
