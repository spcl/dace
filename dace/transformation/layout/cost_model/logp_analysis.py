# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""LogP cost analysis of an SDFG loop nest.

This reads the loop nest (a map scope) off the SDFG directly and assigns it a LogP/LogGP cost from
the measured hardware parameters. The framing (the user's): a contiguous data request is one message;
LOCAL memory -- registers, GPU shared -- is free (its access cost is a later, SRAM task); a GLOBAL
access is a message (a request and its reply, whose whole round trip is ``L``); the bytes that come
back cross the channels at the per-byte gap ``G``.

Per iteration the nest therefore pays, for each global array it touches:

  * a LATENCY term ``messages * L`` -- one round trip per new block message, and
  * a BANDWIDTH term ``bytes_moved * G`` -- the whole blocks that move, at the channel gap.

``messages`` per iteration is the average new blocks touched (``blocks_touched.average_blocks_touched``
-- one input metric among others, NOT the model); a layout transformation changes it, which is how a
layout change reaches the cost. ``bytes_moved`` is ``messages * block_bytes``, because memory moves
whole blocks: a contiguous request amortises the latency over the block, a scattered one pays a whole
block per element.

The per-iteration latency and bandwidth are exposed separately so the caller can combine them as the
regime demands: summed (``L + n*G``, a serialised upper bound) for a latency-exposed nest, or fed
through the overlap model (:func:`~dace.transformation.layout.cost_model.loggp.achievable_rate`) when
the channels saturate.
"""
from dataclasses import dataclass
from typing import Dict, FrozenSet, List

import dace
import sympy as sp

from dace.transformation.layout.cost_model.access_subsets import get_access_subsets
from dace.transformation.layout.cost_model.blocks_touched import average_blocks_touched
from dace.transformation.layout.cost_model.loggp import LogGP, achievable_rate, regime, bandwidth_delay_product

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
    messages_per_iter: sp.Basic  # new block messages -- the latency events (from blocks_touched)
    bytes_moved_per_iter: sp.Basic  # whole-block bytes that cross the channels = messages * block_bytes


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
        """``"bandwidth"`` or ``"latency"`` -- which bound the nest hits, by comparing the concurrency
        it exposes against the bandwidth-delay product. This is how you SEE whether a nest is latency-
        or bandwidth-bound: a parallel nest keeps enough requests in flight to fill the channels
        (bandwidth-bound), a dependency-serialized loop keeps ~1 and cannot hide the latency."""
        return regime(self.p, self.concurrency)

    def bandwidth_delay_product(self) -> float:
        """Outstanding requests needed to saturate; the threshold ``concurrency`` is compared to."""
        return bandwidth_delay_product(self.p)

    def total_time(self) -> sp.Basic:
        """The regime-appropriate total: the overlapped (bandwidth) time when the nest saturates the
        channels, the serialized (latency) time when it cannot hide the latency."""
        return self.total_time_overlapped() if self.regime() == "bandwidth" else self.total_time_serialized()

    def latency_per_iter(self) -> sp.Basic:
        """Latency term per iteration: one round trip ``L`` per new block message, summed over the
        global arrays. Local arrays contribute nothing."""
        return sp.Add(*[a.messages_per_iter * self.p.L for a in self._globals()]) if self._globals() else sp.Integer(0)

    def bandwidth_per_iter(self) -> sp.Basic:
        """Bandwidth term per iteration: the whole blocks moved times the per-byte gap ``G``."""
        return sp.Add(*[a.bytes_moved_per_iter * self.p.G for a in self._globals()]) if self._globals() else sp.Integer(0)

    def time_per_iter(self) -> sp.Basic:
        """The serialised per-iteration LogGP cost ``L*messages + G*bytes`` -- latency plus bandwidth.
        An upper bound: it charges every message its full latency, with no overlap between them."""
        return self.latency_per_iter() + self.bandwidth_per_iter()

    def total_messages(self) -> sp.Basic:
        return sp.Add(*[a.messages_per_iter for a in self._globals()], sp.Integer(0)) * self.total_iters

    def total_bytes(self) -> sp.Basic:
        return sp.Add(*[a.bytes_moved_per_iter for a in self._globals()], sp.Integer(0)) * self.total_iters

    def total_time_serialized(self) -> sp.Basic:
        """Every message pays its latency in series -- the latency-exposed upper bound."""
        return self.time_per_iter() * self.total_iters

    def total_time_overlapped(self) -> sp.Basic:
        """The realistic time once requests overlap: the block traffic over the sustained rate. This
        is the branch that matters at saturation; ``total_time_serialized`` is ~40x higher because it
        forbids overlap (see loggp.achievable_rate)."""
        return self.total_bytes() / achievable_rate(self.p)


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


def exposed_concurrency(state: dace.SDFGState, map_entry: dace.nodes.MapEntry, p: LogGP) -> float:
    """Estimate how many INDEPENDENT block requests the nest can keep in flight -- its exposed MLP,
    which decides the regime.

    Heuristic from the schedule: if any map in the nest runs its iterations concurrently (a parallel
    schedule), the nest exposes iteration-level parallelism, and under the model's assumption of many
    cores / many warps it reaches saturation -- returned as ``inf``, so the regime is bandwidth-bound.
    If EVERY map is ``Sequential``, the frontend did not parallelize it -- usually a loop-carried
    dependency -- so it is serialized to one request at a time, like a pointer chase, and is
    latency-bound.

    This is deliberately coarse: ``G`` is already the SATURATED, all-core bandwidth, so a parallel
    nest running at that rate is by construction at or above the bandwidth-delay product. The nuances
    it skips (a single core tracks only ~24 outstanding misses and CANNOT saturate DRAM alone; a tiny
    parallel nest has too few iterations to fill the pipe) are reached by passing ``concurrency``
    explicitly -- e.g. ``p.concurrency`` for the single-core view, or a measured device MLP.
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
    return float("inf") if parallel else 1.0


def analyze_loop_nest(state: dace.SDFGState,
                      map_entry: dace.nodes.MapEntry,
                      p: LogGP,
                      block_bytes: int,
                      local_arrays: FrozenSet[str] = frozenset(),
                      concurrency: float = None) -> LoopNestLogP:
    """LogP cost of the perfectly-nested map scope rooted at ``map_entry``.

    :param p: the measured LogGP parameters (L, G, ...) for the memory level.
    :param block_bytes: the transfer granularity in BYTES (64 for a CPU line, 32 for a GPU sector).
    :param local_arrays: array names to force-treat as local (free) beyond the storage-type rule.
    :param concurrency: override the exposed-MLP estimate (the outstanding requests the nest can keep
                        in flight); defaults to :func:`exposed_concurrency`.
    """
    sdfg = state.sdfg
    if concurrency is None:
        concurrency = exposed_concurrency(state, map_entry, p)
    loop_ranges = _loop_ranges(state, map_entry)
    subsets = get_access_subsets(state, map_entry)

    # Iteration count of the whole nest.
    total_iters = sp.Integer(1)
    for nest in loop_ranges:
        for begin, end, step in nest.values():
            total_iters *= dace.symbolic.int_floor(
                dace.symbolic.pystr_to_symbolic(end) - dace.symbolic.pystr_to_symbolic(begin),
                dace.symbolic.pystr_to_symbolic(step)) + 1

    arrays: Dict[str, ArrayLogP] = {}
    for name, subset in subsets.items():
        if name not in sdfg.arrays:
            continue
        desc = sdfg.arrays[name]
        is_local = desc.storage in LOCAL_STORAGE or name in local_arrays
        if is_local:
            arrays[name] = ArrayLogP(name, True, sp.Integer(0), sp.Integer(0))
            continue

        dtype_bytes = desc.dtype.bytes
        block_size_elems = max(1, block_bytes // dtype_bytes)
        messages = average_blocks_touched(state, loop_ranges, {name: subset}, block_size_elems).get(name)
        if messages is None:
            continue
        bytes_moved = messages * block_bytes  # whole blocks cross the channels
        arrays[name] = ArrayLogP(name, False, messages, bytes_moved)

    return LoopNestLogP(total_iters=sp.simplify(total_iters), arrays=arrays, p=p, concurrency=concurrency)
