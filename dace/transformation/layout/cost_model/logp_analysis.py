# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""LogP cost analysis of an SDFG loop nest.

This reads the loop nest (a map scope) off the SDFG directly and assigns it a LogP/LogGP cost from
the measured hardware parameters. The framing (the user's): a contiguous data request is one message;
LOCAL memory -- registers, GPU shared -- is free (its access cost is a later, SRAM task); a GLOBAL
access is a message (a request and its reply, whose whole round trip is ``L``); the bytes that come
back cross the channels at the per-byte gap ``G``.

The nest's total memory time is ONE continuous formula (``loggp.nest_memory_time``):

    T  =  max( total_bytes * G ,  total_messages * L / C )

where ``C`` is the concurrency the nest exposes -- how many independent block requests it keeps in
flight. The latency term is Little's Law (sustained request rate ``C / L``); the bandwidth term is
the channel ceiling. There is NO regime switch: ``C = 1`` recovers the dependent chain (each miss
waits out the full ``L``), large ``C`` saturates the channels, and the crossover is continuous. The
old serialized-vs-overlapped branch is retained only as a diagnostic ceiling -- the MLP sweep
(2026-07-17) refuted it as a predictor by the measured ~8x (rate scales linearly with independent
chains: 1.97x at 2, 3.99x at 4, knee ~8).

The two counts live at DIFFERENT granularities, and that split is where the latency model explains
load/store performance:

  * ``messages`` -- new blocks touched at the REQUEST granularity (``line_bytes``): what pays ``L``.
  * ``bytes_moved`` -- new blocks at the TRANSFER granularity (``sector_bytes``) times the sector
    size: what pays ``G``. On x86 the two coincide; on NVIDIA a 128-byte request carries 32-byte
    sectors, so a scattered access moves 4x fewer bytes than requests*line -- but needs
    ``line/sector`` MORE outstanding requests to saturate the channels (the saturation threshold is
    ``C >= L / (k * sector_bytes * G)`` with ``k`` the sectors used per request).

``messages`` per iteration is the average new blocks touched (``blocks_touched.average_blocks_touched``
-- one input metric among others, NOT the model); a layout transformation changes it, which is how a
layout change reaches the cost.
"""
from dataclasses import dataclass
from typing import Dict, FrozenSet, List

import dace
import sympy as sp

from dace.transformation.layout.cost_model.access_subsets import get_access_subsets
from dace.transformation.layout.cost_model.blocks_touched import average_blocks_touched
from dace.transformation.layout.cost_model.loggp import (LogGP, bandwidth_delay_product, nest_memory_time, regime)

# Storage that counts as LOCAL (free for now): registers and GPU shared memory are "my memory".
LOCAL_STORAGE: FrozenSet[dace.dtypes.StorageType] = frozenset(
    {dace.dtypes.StorageType.Register, dace.dtypes.StorageType.GPU_Shared})

# Schedules whose iterations run concurrently, so the nest exposes memory-level parallelism.
PARALLEL_SCHEDULES: FrozenSet[dace.dtypes.ScheduleType] = frozenset({
    dace.dtypes.ScheduleType.CPU_Multicore,
    dace.dtypes.ScheduleType.GPU_Device,
    dace.dtypes.ScheduleType.GPU_ThreadBlock,
    dace.dtypes.ScheduleType.Default,
})


@dataclass(frozen=True)
class ArrayLogP:
    """The LogP cost a single array contributes per iteration of the nest."""
    array: str
    is_local: bool  # local memory is free (skipped in the latency / bandwidth sums)
    messages_per_iter: sp.Basic  # new blocks at REQUEST granularity (line) -- the latency events
    sectors_per_iter: sp.Basic  # new blocks at TRANSFER granularity (sector) -- the transfer units
    bytes_moved_per_iter: sp.Basic  # bytes crossing the channels = sectors_per_iter * sector_bytes


@dataclass(frozen=True)
class NestCounts:
    """TIER-0 result: the structural block counts of one nest, with NO measured parameters.

    ``line_bytes``/``sector_bytes`` are device-CLASS constants (64/64 x86, 128/32 NVIDIA), not
    performance measurements -- tier 0 is parameter-free in the sense that no benchmark feeds it.
    Totals are symbolic in the nest's loop bounds; the write factor and element spans are already
    folded into ``bytes_moved_per_iter``.
    """
    total_iters: sp.Basic
    arrays: Dict[str, ArrayLogP]
    line_bytes: int
    sector_bytes: int

    def globals_(self) -> List[ArrayLogP]:
        return [a for a in self.arrays.values() if not a.is_local]

    def messages(self) -> sp.Basic:
        """M: total latency events (request-granularity blocks), whole nest."""
        return sp.Add(*[a.messages_per_iter for a in self.globals_()], sp.Integer(0)) * self.total_iters

    def bytes_moved(self) -> sp.Basic:
        """B: total bytes crossing the channels (sector granularity, write factor included)."""
        return sp.Add(*[a.bytes_moved_per_iter for a in self.globals_()], sp.Integer(0)) * self.total_iters


@dataclass(frozen=True)
class LoopNestLogP:
    """The LogP cost of one loop nest, in terms of the measured parameters ``p``."""
    total_iters: sp.Basic
    arrays: Dict[str, ArrayLogP]
    p: LogGP
    concurrency: float  # independent block requests the nest can keep in flight (its exposed MLP)

    def _globals(self) -> List[ArrayLogP]:
        return [a for a in self.arrays.values() if not a.is_local]

    def regime(self) -> str:
        """``"bandwidth"`` or ``"latency"`` -- a DIAGNOSTIC of which term of :meth:`total_time` is
        expected to bind, by comparing the nest's concurrency to the bandwidth-delay product. It no
        longer selects a formula (``total_time`` is one continuous expression); it only reports.
        Coalesced approximation -- the exact per-nest crossover is where :meth:`total_time_latency`
        equals :meth:`total_time_bandwidth`."""
        return regime(self.p, self.concurrency)

    def bandwidth_delay_product(self) -> float:
        """Outstanding requests needed to saturate; the threshold ``concurrency`` is compared to."""
        return bandwidth_delay_product(self.p)

    def total_time(self) -> sp.Basic:
        """THE nest time -- ``max(total_bytes * G, total_messages * L / concurrency)``, one
        continuous formula (``loggp.nest_memory_time``). ``concurrency = 1`` recovers the dependent
        chain; a saturating nest lands on the bandwidth term; the crossover is continuous. There is
        no regime branch to pick."""
        return nest_memory_time(self.p, self.total_bytes(), self.total_messages(), self.concurrency)

    def total_time_bandwidth(self) -> sp.Basic:
        """The bandwidth term alone: ``total_bytes * G`` -- the channel ceiling; binds at saturation."""
        return self.total_bytes() * self.p.G

    def total_time_latency(self) -> sp.Basic:
        """The latency term alone: ``total_messages * L / concurrency`` -- Little's Law (sustained
        request rate ``concurrency / L``); binds when the nest cannot keep the channels full. This is
        where LogP explains load/store performance: the term quantifies exactly how much exposed
        latency the nest's MLP fails to hide."""
        return self.total_messages() * self.p.L / self.concurrency

    def latency_per_iter(self) -> sp.Basic:
        """Latency events per iteration priced at full ``L`` -- one round trip per new block message,
        summed over the global arrays. Local arrays contribute nothing. NOTE: this is the C=1 price;
        the nest-level latency term divides by the exposed concurrency (:meth:`total_time_latency`)."""
        return sp.Add(*[a.messages_per_iter * self.p.L for a in self._globals()]) if self._globals() else sp.Integer(0)

    def bandwidth_per_iter(self) -> sp.Basic:
        """Bandwidth term per iteration: the bytes that cross the channels times the per-byte gap."""
        return sp.Add(*[a.bytes_moved_per_iter * self.p.G for a in self._globals()]) if self._globals() else sp.Integer(0)

    def time_per_iter(self) -> sp.Basic:
        """The serialised per-iteration LogGP cost ``L*messages + G*bytes`` -- latency plus bandwidth.
        A DIAGNOSTIC ceiling: it charges every message its full latency with no overlap. Refuted as a
        predictor by the MLP sweep (~8x high on a Zen 4 core); see :meth:`total_time`."""
        return self.latency_per_iter() + self.bandwidth_per_iter()

    def total_messages(self) -> sp.Basic:
        """Total latency events: new blocks touched at REQUEST (line) granularity, whole nest."""
        return sp.Add(*[a.messages_per_iter for a in self._globals()], sp.Integer(0)) * self.total_iters

    def total_sectors(self) -> sp.Basic:
        """Total transfer units: new blocks touched at TRANSFER (sector) granularity, whole nest."""
        return sp.Add(*[a.sectors_per_iter for a in self._globals()], sp.Integer(0)) * self.total_iters

    def total_bytes(self) -> sp.Basic:
        """Bytes crossing the channels for the whole nest: ``total_sectors * sector_bytes``."""
        return sp.Add(*[a.bytes_moved_per_iter for a in self._globals()], sp.Integer(0)) * self.total_iters

    def total_time_serialized(self) -> sp.Basic:
        """Every message pays its full latency in series: ``L*M + G*B``. A DIAGNOSTIC upper bound
        (the zero-overlap ceiling), NOT a predictor -- the MLP sweep measured rate scaling linearly
        with independent chains (1.97x at 2, 3.99x at 4), refuting this sum by the core's MLP (~8x).
        Kept because the ceiling is still informative: ``total_time_serialized / total_time`` is the
        overlap the model credits, and a measurement ABOVE the ceiling means the parameters are wrong."""
        return self.time_per_iter() * self.total_iters


def _loop_ranges(state: dace.SDFGState, map_entry: dace.nodes.MapEntry) -> List[Dict[str, dace.subsets.Range]]:
    """The nest's ``{param: range}`` maps, outer-to-inner, over every map in the scope subtree."""
    scope_dict = state.scope_dict()

    def depth(entry):
        d, cur = 0, scope_dict[entry]
        while cur is not None:
            d, cur = d + 1, scope_dict[cur]
        return d

    entries = [map_entry]
    for node, _ in state.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry) and node is not map_entry and scope_dict.get(node) is not None:
            # keep only maps nested under map_entry
            cur = scope_dict[node]
            while cur is not None and cur is not map_entry:
                cur = scope_dict[cur]
            if cur is map_entry:
                entries.append(node)
    entries.sort(key=depth)
    return [{p: r for p, r in zip(e.map.params, e.map.range)} for e in entries]


def exposed_concurrency(state: dace.SDFGState, map_entry: dace.nodes.MapEntry, p: LogGP,
                        n_cores: int = None) -> float:
    """Estimate how many INDEPENDENT block requests the nest can keep in flight -- its exposed MLP.

    Heuristic from the schedule, refined by how iterations DISTRIBUTE onto cores:

      * Any PARALLEL map in the nest -> the iterations spread over the cores in contiguous chunks.
        (DaCe emits ``#pragma omp parallel for`` with no schedule clause; the libgomp/LLVM default is
        static contiguous chunking, so the premise holds in practice -- but a map with
        ``omp_schedule`` dynamic/guided breaks it, inflating per-core block counts toward the
        interleaved bound. Each core streams its own chunk, preserving per-chunk spatial locality;
        the chunk boundaries add one shared line per edge, O(cores), negligible.) Aggregate concurrency is
        ``n_cores * p.core_mlp`` when the caller says how many cores populate the nest; without a
        core count it is ``inf`` -- the saturated assumption, consistent with ``G`` being the
        all-core bandwidth. Passing ``n_cores`` is more honest: 16 cores x 8 outstanding = 128 on
        this box, BELOW the ~148-request bandwidth-delay product -- even all cores pinned do not
        quite saturate, which blanket ``inf`` hides.
      * Every map ``Sequential`` -> ``p.core_stream_mlp``, ONE core's PREFETCH-INCLUSIVE streaming
        concurrency (``bw_core * L / line``, Little's Law on the measured single-core bandwidth) --
        NOT the demand-miss knee, and NOT 1. Sequential does not mean one request at a time: an
        affine access's addresses are computable ahead of the loads, and for the contiguous/strided
        patterns the affine analysis scores, the PREFETCHER keeps line fills in flight beyond the
        demand-miss queue -- the chase knee (``core_mlp``, ~8) deliberately defeats the prefetcher
        and therefore under-credits a streaming nest severalfold (the box's own single-core triad
        implies ~50 outstanding lines).

    The demand-miss budget ``p.core_mlp`` is the right ``C`` for PREFETCH-HOSTILE patterns --
    scattered or replayed-indirect accesses -- which is a LAYOUT statement: scattering an access
    demotes its concurrency as well as its block count (the second penalty). Callers costing such a
    nest pass ``concurrency = n_units * p.core_mlp`` explicitly. ``C = 1`` is DATA-DEPENDENT
    addressing (the next address is the previous load's value); pass ``concurrency=1``. A caller who
    knows the true device MLP overrides this estimate entirely.
    """
    scope_dict = state.scope_dict()
    schedules = [map_entry.map.schedule]
    for node, _ in state.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry) and node is not map_entry:
            cur = scope_dict.get(node)
            while cur is not None and cur is not map_entry:
                cur = scope_dict.get(cur)
            if cur is map_entry:
                schedules.append(node.map.schedule)
    parallel = any(s in PARALLEL_SCHEDULES for s in schedules)
    # GPUs have no prefetcher worth the name: a warp's latency hiding IS its demand budget, so GPU
    # units use the demand-miss ``core_mlp`` (per-warp outstanding loads). CPU units get the
    # prefetch-inclusive ``core_stream_mlp`` -- the affine analysis only ever scores prefetch-
    # friendly patterns (scattered/indirect ones must come through replayed_counts, and their
    # callers pass ``concurrency = n_units * p.core_mlp`` explicitly).
    gpu = any(s in (dace.dtypes.ScheduleType.GPU_Device, dace.dtypes.ScheduleType.GPU_ThreadBlock)
              for s in schedules)
    unit_mlp = p.core_mlp if gpu else p.core_stream_mlp
    if not parallel:
        return unit_mlp
    return n_cores * unit_mlp if n_cores is not None else float("inf")


def written_arrays(state: dace.SDFGState, map_entry: dace.nodes.MapEntry) -> FrozenSet[str]:
    """Arrays the nest WRITES: memlets flowing toward the exit (tasklet -> MapExit) or into an
    in-scope access node. A written block moves twice on a write-allocate cache (fetch + writeback),
    so writes are priced differently from reads."""
    exit_node = state.exit_node(map_entry)
    names = set()
    for e in state.scope_subgraph(map_entry).edges():
        if e.data.data is None:
            continue
        if e.dst is exit_node or isinstance(e.dst, (dace.nodes.MapExit, dace.nodes.AccessNode)):
            names.add(e.data.data)
    return frozenset(names)


def dynamic_memlet_arrays(state: dace.SDFGState, map_entry: dace.nodes.MapEntry) -> FrozenSet[str]:
    """Arrays reached through a DYNAMIC memlet inside the scope -- the frontend's shape for a
    data-dependent (gather/scatter) access, whose declared subset is a whole-array over-approximation
    that must not be priced as an invariant read."""
    names = set()
    for e in state.scope_subgraph(map_entry).edges():
        if e.data.data is not None and e.data.dynamic:
            names.add(e.data.data)
    return frozenset(names)


def analyze_loop_nest(state: dace.SDFGState,
                      map_entry: dace.nodes.MapEntry,
                      p: LogGP,
                      block_bytes: int = None,
                      local_arrays: FrozenSet[str] = frozenset(),
                      concurrency: float = None,
                      n_cores: int = None,
                      replayed_counts: Dict[str, tuple] = None) -> LoopNestLogP:
    """LogP cost of the perfectly-nested map scope rooted at ``map_entry``.

    :param p: the measured LogGP parameters (L, G, ...) for the memory level.
    :param block_bytes: ONE granularity for both counts, overriding ``p``. ``None`` (the default)
                        takes both from ``p`` -- messages at ``p.line_bytes`` (the request
                        granularity that pays ``L``), bytes at ``p.sector_bytes`` (the transfer
                        granularity that pays ``G``). Passing a value collapses the two, which is
                        only correct where they coincide (x86: line == sector == 64).
    :param local_arrays: array names to force-treat as local (free) beyond the storage-type rule.
    :param concurrency: override the exposed-MLP estimate (the outstanding requests the nest can keep
                        in flight); defaults to :func:`exposed_concurrency`.
    :param n_cores: how many cores populate a parallel nest (threads pinned, iterations in contiguous
                    chunks). Feeds the concurrency estimate: ``n_cores * p.core_stream_mlp`` instead
                    of the saturated ``inf``. Ignored when ``concurrency`` is given.
    :param replayed_counts: per-array ``(messages_per_iter, sectors_per_iter)`` overrides for
                            data-dependent accesses, from ``replayed_blocks_touched`` on the
                            materialized index array. Required for any array reached through a
                            dynamic memlet.
    """
    if concurrency is None:
        concurrency = exposed_concurrency(state, map_entry, p, n_cores)
    line_bytes = block_bytes if block_bytes is not None else p.line_bytes
    sector_bytes = block_bytes if block_bytes is not None else p.sector_bytes
    counts = count_loop_nest(state, map_entry, line_bytes=line_bytes, sector_bytes=sector_bytes,
                             local_arrays=local_arrays, replayed_counts=replayed_counts)
    return LoopNestLogP(total_iters=counts.total_iters, arrays=counts.arrays, p=p, concurrency=concurrency)


def count_loop_nest(state: dace.SDFGState,
                    map_entry: dace.nodes.MapEntry,
                    line_bytes: int = 64,
                    sector_bytes: int = None,
                    local_arrays: FrozenSet[str] = frozenset(),
                    replayed_counts: Dict[str, tuple] = None) -> NestCounts:
    """TIER 0: the structural block counts of the nest -- no LogGP parameters, no benchmark.

    This is the counting core `analyze_loop_nest` layers time on top of. Granularities are device-
    class constants (defaults: x86 line 64, no sectoring). Same contract for indirection: a dynamic
    memlet without a ``replayed_counts`` entry is refused, an unscoreable subset raises.
    """
    sdfg = state.sdfg
    if sector_bytes is None:
        sector_bytes = line_bytes
    loop_ranges = _loop_ranges(state, map_entry)
    subsets = get_access_subsets(state, map_entry)

    # Iteration count of the whole nest.
    total_iters = sp.Integer(1)
    for nest in loop_ranges:
        for begin, end, step in nest.values():
            total_iters *= dace.symbolic.int_floor(
                dace.symbolic.pystr_to_symbolic(end) - dace.symbolic.pystr_to_symbolic(begin),
                dace.symbolic.pystr_to_symbolic(step)) + 1

    dynamic = dynamic_memlet_arrays(state, map_entry)
    written = written_arrays(state, map_entry)
    replayed_counts = replayed_counts or {}

    arrays: Dict[str, ArrayLogP] = {}
    for name, subset in subsets.items():
        if name not in sdfg.arrays:
            continue
        desc = sdfg.arrays[name]
        is_local = desc.storage in LOCAL_STORAGE or name in local_arrays
        if is_local:
            arrays[name] = ArrayLogP(name, True, sp.Integer(0), sp.Integer(0), sp.Integer(0))
            continue

        dtype_bytes = desc.dtype.bytes
        if name in replayed_counts:
            # The static-replay hook (indirect access with a materialized index array): the caller
            # supplies per-iteration (messages, sectors) from replayed_blocks_touched.
            messages, sectors = map(sp.sympify, replayed_counts[name])
        elif name in dynamic:
            # A dynamic memlet is a data-dependent access wearing a whole-array subset: scoring the
            # SUBSET would price a gather as one invariant read -- silently and enormously wrong.
            # Refuse loudly; the honest route is a replayed count.
            raise ValueError(
                f"array {name!r} is accessed through a dynamic (data-dependent) memlet; its block "
                f"counts are not statically derivable. Replay the materialized index array with "
                f"blocks_touched.replayed_blocks_touched and pass the chosen bound via "
                f"replayed_counts={{'{name}': (messages_per_iter, sectors_per_iter)}}.")
        else:
            # An element wider than the granularity spans several blocks; blocks_touched counts at
            # >= one-element granularity, so multiply the span back in (e.g. a 64-byte vector dtype
            # on 32-byte sectors moves 2 sectors per element).
            line_elems = max(1, line_bytes // dtype_bytes)
            sector_elems = max(1, sector_bytes // dtype_bytes)
            line_span = max(1, -(-dtype_bytes // line_bytes))
            sector_span = max(1, -(-dtype_bytes // sector_bytes))
            messages = average_blocks_touched(state, loop_ranges, {name: subset}, line_elems).get(name)
            if messages is None:
                raise ValueError(f"array {name!r}: blocks_touched cannot reduce its access subset "
                                 f"{subset}; refusing to silently drop its cost from the nest")
            messages = messages * line_span
            if sector_elems == line_elems and sector_span == line_span:
                sectors = messages
            else:
                sectors = average_blocks_touched(state, loop_ranges, {name: subset}, sector_elems)[name]
                sectors = sectors * sector_span
        # A written block moves TWICE on a write-allocate cache: fetched (read-for-ownership) and
        # written back. Non-temporal stores would skip the fetch, but compilers rarely emit them --
        # the triad's own accounting ("4 streams, not 3") is this factor measured. Messages stay 1x:
        # the RFO fetch is the round trip; the writeback is posted, no reply awaited.
        write_factor = 2 if name in written else 1
        bytes_moved = sectors * sector_bytes * write_factor
        arrays[name] = ArrayLogP(name, False, messages, sectors, bytes_moved)

    return NestCounts(total_iters=sp.simplify(total_iters), arrays=arrays, line_bytes=line_bytes,
                      sector_bytes=sector_bytes)


def sign_of(expr: sp.Basic) -> str:
    """``"neg" | "zero" | "pos" | "unknown"`` for a (possibly symbolic) difference, using the
    nonnegative-symbol assumptions DaCe symbols carry. ``unknown`` is an honest verdict, not an
    error -- the caller escalates (substitute concrete sizes, or use tier 2)."""
    d = sp.simplify(expr)
    if d.is_zero:
        return "zero"
    if d.is_positive:
        return "pos"
    if d.is_negative:
        return "neg"
    if d.is_nonpositive:
        return "neg"  # <= 0 and not provably zero: rank as not-worse; ties collapse upstream
    if d.is_nonnegative:
        return "pos"
    return "unknown"


def dominance_verdict(a: NestCounts, b: NestCounts, subs: Dict = None) -> str:
    """TIER-0 layout comparison by the dominance lemma -- NO parameters.

    ``"first"``: a's counts are <= b's on BOTH axes (M and B), so a is at least as fast for EVERY
    ``L, G, C > 0`` (each term of ``max(B*G, M*L/C)`` is monotone in its count). ``"second"``: the
    reverse. ``"tie"``: equal counts. ``"undecided"``: the counts DISAGREE (fewer requests vs fewer
    bytes -- the ranking is genuinely concurrency-dependent, flip at ``C_flip = M2*L/(B1*G)``) or a
    symbolic sign could not be determined; either way tier 0 cannot settle it -- substitute concrete
    sizes via ``subs`` or escalate to tier 2.
    """
    dm = a.messages() - b.messages()
    db = a.bytes_moved() - b.bytes_moved()
    if subs:
        dm, db = dm.subs(subs), db.subs(subs)
    sm, sb = sign_of(dm), sign_of(db)
    if sm == "unknown" or sb == "unknown":
        return "undecided"
    if sm == "zero" and sb == "zero":
        return "tie"
    if sm in ("neg", "zero") and sb in ("neg", "zero"):
        return "first"
    if sm in ("pos", "zero") and sb in ("pos", "zero"):
        return "second"
    return "undecided"  # genuine disagreement: fewer requests vs fewer bytes


def pareto_front(candidates: Dict[str, NestCounts], subs: Dict = None) -> List[str]:
    """TIER-0 pruning for a layout sweep: the candidates NOT dominated by any other, in input order.

    Every returned layout is optimal for SOME ``(L, G, C)``; every dropped one is at least as slow
    as some survivor for ALL parameters -- so dropping it before compiling/timing costs nothing.
    ``undecided`` comparisons prune nothing (conservative: never drop on uncertainty). Ties keep the
    first-listed candidate."""
    names = list(candidates)
    survivors = []
    for i, name in enumerate(names):
        beaten = False
        for j, other in enumerate(names):
            if i == j:
                continue
            v = dominance_verdict(candidates[other], candidates[name], subs)
            if v == "first" or (v == "tie" and j < i):
                beaten = True
                break
        if not beaten:
            survivors.append(name)
    return survivors
