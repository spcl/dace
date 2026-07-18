# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""LogP/LogGP cost analysis of an SDFG loop nest: time = max(total_bytes * G, total_messages * L / concurrency); see loggp.nest_memory_time."""
from dataclasses import dataclass
from typing import Dict, FrozenSet, List

import dace
import sympy as sp

from dace.transformation.layout.cost_model.access_subsets import get_access_subsets
from dace.transformation.layout.cost_model.blocks_touched import average_blocks_touched
from dace.transformation.layout.cost_model.loggp import (LogGP, bandwidth_delay_product, nest_memory_time, regime)

# LOCAL (free-for-now) storage: registers + GPU shared memory.
LOCAL_STORAGE: FrozenSet[dace.dtypes.StorageType] = frozenset(
    {dace.dtypes.StorageType.Register, dace.dtypes.StorageType.GPU_Shared})

# Schedules whose iterations run concurrently (nest exposes MLP).
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
    is_local: bool  # free: skipped in latency/bandwidth sums
    messages_per_iter: sp.Basic  # new blocks at REQUEST granularity (line): latency events
    sectors_per_iter: sp.Basic  # new blocks at TRANSFER granularity (sector)
    bytes_moved_per_iter: sp.Basic  # bytes crossing channels = sectors_per_iter * sector_bytes


@dataclass(frozen=True)
class NestCounts:
    """TIER-0 result: structural block counts of one nest, no measured parameters."""
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
    concurrency: float  # exposed MLP: independent block requests in flight

    def _globals(self) -> List[ArrayLogP]:
        return [a for a in self.arrays.values() if not a.is_local]

    def regime(self) -> str:
        """Diagnostic only: which term of :meth:`total_time` is expected to bind."""
        return regime(self.p, self.concurrency)

    def bandwidth_delay_product(self) -> float:
        """Outstanding requests needed to saturate; the threshold ``concurrency`` is compared to."""
        return bandwidth_delay_product(self.p)

    def total_time(self) -> sp.Basic:
        """Nest time: max(total_bytes * G, total_messages * L / concurrency)."""
        return nest_memory_time(self.p, self.total_bytes(), self.total_messages(), self.concurrency)

    def total_time_bandwidth(self) -> sp.Basic:
        """Bandwidth term alone: total_bytes * G; binds at saturation."""
        return self.total_bytes() * self.p.G

    def total_time_latency(self) -> sp.Basic:
        """Latency term alone: total_messages * L / concurrency (Little's Law)."""
        return self.total_messages() * self.p.L / self.concurrency

    def latency_per_iter(self) -> sp.Basic:
        """Latency per iteration at C=1: full L per message, global arrays only."""
        return sp.Add(*[a.messages_per_iter * self.p.L for a in self._globals()]) if self._globals() else sp.Integer(0)

    def bandwidth_per_iter(self) -> sp.Basic:
        """Bandwidth term per iteration: the bytes that cross the channels times the per-byte gap."""
        return sp.Add(*[a.bytes_moved_per_iter * self.p.G for a in self._globals()]) if self._globals() else sp.Integer(0)

    def time_per_iter(self) -> sp.Basic:
        """Serialised per-iteration cost L*messages + G*bytes; diagnostic ceiling, not a predictor."""
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
        """Zero-overlap upper bound L*M + G*B; ratio to total_time is the credited overlap."""
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
    """Estimate independent block requests the nest can keep in flight (exposed MLP), from schedule and n_cores."""
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
    # GPU: demand-miss core_mlp (per-warp). CPU: prefetch-inclusive core_stream_mlp.
    gpu = any(s in (dace.dtypes.ScheduleType.GPU_Device, dace.dtypes.ScheduleType.GPU_ThreadBlock)
              for s in schedules)
    unit_mlp = p.core_mlp if gpu else p.core_stream_mlp
    if not parallel:
        return unit_mlp
    return n_cores * unit_mlp if n_cores is not None else float("inf")


def written_arrays(state: dace.SDFGState, map_entry: dace.nodes.MapEntry) -> FrozenSet[str]:
    """Arrays the nest writes: memlets flowing to MapExit or an in-scope access node."""
    exit_node = state.exit_node(map_entry)
    names = set()
    for e in state.scope_subgraph(map_entry).edges():
        if e.data.data is None:
            continue
        if e.dst is exit_node or isinstance(e.dst, (dace.nodes.MapExit, dace.nodes.AccessNode)):
            names.add(e.data.data)
    return frozenset(names)


def dynamic_memlet_arrays(state: dace.SDFGState, map_entry: dace.nodes.MapEntry) -> FrozenSet[str]:
    """Arrays reached through a dynamic (data-dependent) memlet inside the scope."""
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
    """LogP cost of the perfectly-nested map scope at ``map_entry``; ``replayed_counts`` required for arrays behind a dynamic memlet."""
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
    """TIER-0 structural block counts of the nest, no LogGP parameters; requires ``replayed_counts`` for dynamic memlets."""
    sdfg = state.sdfg
    if sector_bytes is None:
        sector_bytes = line_bytes
    loop_ranges = _loop_ranges(state, map_entry)
    subsets = get_access_subsets(state, map_entry)

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
            # static-replay hook: per-iteration (messages, sectors) from replayed_blocks_touched
            messages, sectors = map(sp.sympify, replayed_counts[name])
        elif name in dynamic:
            # dynamic memlet subset is whole-array; scoring it would mis-price a gather as one read
            raise ValueError(
                f"array {name!r} is accessed through a dynamic (data-dependent) memlet; its block "
                f"counts are not statically derivable. Replay the materialized index array with "
                f"blocks_touched.replayed_blocks_touched and pass the chosen bound via "
                f"replayed_counts={{'{name}': (messages_per_iter, sectors_per_iter)}}.")
        else:
            # element wider than granularity spans multiple blocks; multiply span back in
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
        # written block moves twice (fetch + writeback); messages stay 1x (writeback posted, no reply)
        write_factor = 2 if name in written else 1
        bytes_moved = sectors * sector_bytes * write_factor
        arrays[name] = ArrayLogP(name, False, messages, sectors, bytes_moved)

    return NestCounts(total_iters=sp.simplify(total_iters), arrays=arrays, line_bytes=line_bytes,
                      sector_bytes=sector_bytes)


def sign_of(expr: sp.Basic) -> str:
    """Sign of a (possibly symbolic) difference: "neg"|"zero"|"pos"|"unknown"."""
    d = sp.simplify(expr)
    if d.is_zero:
        return "zero"
    if d.is_positive:
        return "pos"
    if d.is_negative:
        return "neg"
    if d.is_nonpositive:
        return "neg"  # not provably zero: rank as not-worse
    if d.is_nonnegative:
        return "pos"
    return "unknown"


def dominance_verdict(a: NestCounts, b: NestCounts, subs: Dict = None) -> str:
    """TIER-0 dominance-lemma comparison: "first"|"second"|"tie"|"undecided" (undecided if counts disagree or sign unknown)."""
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
    """TIER-0 pruning: candidates not dominated by any other, in input order. Ties keep the first-listed candidate."""
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
