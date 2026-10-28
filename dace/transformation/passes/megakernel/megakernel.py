# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``MegaKernelOptimization``: run each maximal hierarchical linear chain of GPU kernels as one persistent,
cooperative kernel.

Preparation (``GatherKernelStates``) first gathers the kernels of consecutive states into one state. A chain moves into
the body of one ``GPU_Device`` map that launches as many thread blocks as the device holds at once
(``ComputePersistentKernelRange``). In the body, every kernel becomes a grid-stride loop (``grid_stride``) and grid
barriers separate the dependent ones (``InsertGridSyncs``). Where scheduling stages by thread ownership saves barriers,
a stage runs each iteration in the thread that owns the elements it writes instead (``ownership``).
"""

import copy

from dace import SDFG, SDFGState, dtypes, properties, symbolic
from dace.libraries.standard.nodes import ComputePersistentKernelRange
from dace.libraries.standard.nodes.compute_persistent_kernel_range import NBLK_CONNECTOR
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.graph import SubgraphView
from dace.sdfg.state import ConditionalBlock, ControlFlowBlock, ControlFlowRegion
from dace.transformation import helpers as xfh
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.passes.megakernel.discovery import (
    TARGETS,
    MegakernelChain,
    Target,
    bound_symbols,
    find_chains,
    is_kernel,
    kernels_in,
)
from dace.transformation.passes.megakernel.gather_kernels import GatherKernelStates
from dace.transformation.passes.megakernel.grid_stride import grid_stride
from dace.transformation.passes.megakernel.grid_sync import InsertGridSyncs
from dace.transformation.passes.megakernel.ownership import own_stride

#: The thread index of the megakernel map, and the number of its threads, inside the body
THREAD_INDEX = "__mk_tid"
THREAD_COUNT = "__mk_nthreads"


def nested_blocks(blocks: list[ControlFlowBlock]) -> list[ControlFlowBlock]:
    """``blocks`` and every block inside them, conditional branches included."""
    inner = [
        b
        for block in blocks
        if isinstance(block, (ControlFlowRegion, ConditionalBlock))
        for b in block.all_control_flow_blocks()
    ]
    return blocks + inner


def assigned_symbols(chain: MegakernelChain) -> set[str]:
    """The symbols ``chain`` assigns: on the edges between its blocks, and the loop variables and assignments inside
    them (conditional branches included)."""
    between = [e for b in chain.blocks[:-1] for e in chain.region.out_edges(b)]
    names = set().union(*(e.data.assignments.keys() for e in between))
    for block in chain.blocks:
        if isinstance(block, ControlFlowRegion):
            names |= bound_symbols(block)
        elif isinstance(block, ConditionalBlock):
            names = names.union(*(bound_symbols(branch) for cond, branch in block.branches))
    return names


def used_outside_chain(chain: MegakernelChain) -> set[str]:
    """The names that blocks and interstate edges outside ``chain`` use; a region outside counts only its own
    conditions."""
    inside = set(nested_blocks(chain.blocks))
    names = set()
    for block in chain.region.sdfg.all_control_flow_blocks():
        if block not in inside:
            names |= block.used_symbols(all_symbols=True, with_contents=isinstance(block, SDFGState))
    for edge in chain.region.sdfg.all_interstate_edges():
        if edge.src not in inside or edge.dst not in inside:
            names |= edge.data.read_symbols()
    return names


def keep_shared_memory_outside(host: SDFGState, node: nodes.NestedSDFG, target: Target) -> None:
    """Allocates in the parent of ``node`` the transients its stages share: a GPU kernel cannot allocate global
    memory, and every thread of a CPU team would allocate its own copy. Containers only one kernel's scope uses
    stay inside."""
    sdfg, body = host.sdfg, node.sdfg
    between_stages = {n.data for state in body.states() for n in state.data_nodes() if state.entry_node(n) is None}
    for name, desc in list(body.arrays.items()):
        if desc.transient and desc.storage == dtypes.StorageType.GPU_Global and name not in between_stages:
            # Only map scopes touch it: data of one iteration, which hoisted would be one buffer all threads race on
            desc.storage = dtypes.StorageType.Register
            continue
        stage_shared = name in between_stages and target.shares(desc, scoped=False)
        if desc.transient and (stage_shared or desc.storage == dtypes.StorageType.GPU_Global):
            sdfg.add_datadesc(name, copy.deepcopy(desc))
            desc.transient = False
            node.add_out_connector(name)
            host.add_edge(node, name, host.add_write(name), None, Memlet.from_array(name, desc))


def nest_chain(chain: MegakernelChain, target: Target) -> tuple[SDFGState, nodes.NestedSDFG] | None:
    """Moves the blocks of ``chain`` into a nested SDFG, which global memory and the symbols the code after it reads
    stay outside of.

    :return: The state calling the nested SDFG and its node, or ``None`` (nothing changed) when code outside the chain
             reads a symbol the chain assigns: a kernel cannot hand it out.
    """
    # Leading blocks without kernels that assign what code outside reads (bounds computed in a conditional) stay on
    # the host, which passes the values in
    while assigned_symbols(chain) & used_outside_chain(chain) and not kernels_in(chain.blocks[0], target):
        chain.blocks.pop(0)
    region, blocks = chain.region, chain.blocks
    sdfg = region.sdfg
    if assigned_symbols(chain) & used_outside_chain(chain):
        return None
    if len(blocks) == 1 and isinstance(blocks[0], SDFGState):
        host = blocks[0]
        node = xfh.nest_state_subgraph(sdfg, host, SubgraphView(host, host.nodes()))
    else:
        host = xfh.nest_sdfg_subgraph(sdfg, SubgraphView(region, blocks))
        node = next(n for n in host.nodes() if isinstance(n, nodes.NestedSDFG))
    keep_shared_memory_outside(host, node, target)
    return host, node


def launch_cooperatively(host: SDFGState, node: nodes.NestedSDFG, handle: str, block_size: int) -> None:
    """Wraps ``node`` into a cooperative ``GPU_Device`` map of ``block_size``-thread blocks, sized by a
    ``ComputePersistentKernelRange`` in a state before ``host``."""
    region, sdfg = host.parent_graph, host.sdfg
    nblk = f"__{handle}_nblk"
    nblk_data, _ = sdfg.add_scalar(f"{nblk}_value", dtypes.int32, transient=True, find_new_name=True)
    sdfg.add_symbol(nblk, dtypes.int32)
    size = region.add_state_before(host, f"{handle}_size", is_start_block=region.start_block is host)
    sizing = ComputePersistentKernelRange(f"{handle}_size", kernel_handle=handle)
    size.add_node(sizing)
    size.add_edge(sizing, NBLK_CONNECTOR, size.add_write(nblk_data), None, Memlet(f"{nblk_data}[0]"))
    region.out_edges(size)[0].data.assignments[nblk] = nblk_data

    entry, exit_ = host.add_map(
        handle, {THREAD_INDEX: f"0:{nblk}*{block_size}"}, schedule=dtypes.ScheduleType.GPU_Device
    )
    entry.map.gpu_block_size = [block_size, 1, 1]
    entry.map.cooperative = entry.map.explicit_sync = True
    entry.map.kernel_handle = handle
    for sym in (THREAD_INDEX, THREAD_COUNT):
        node.sdfg.add_symbol(sym, dtypes.int64)
    node.symbol_mapping[THREAD_INDEX] = THREAD_INDEX
    node.symbol_mapping[THREAD_COUNT] = f"{nblk} * {block_size}"
    wrap(host, node, entry, exit_)


def launch_team(host: SDFGState, node: nodes.NestedSDFG, handle: str, num_threads: int) -> None:
    """Wraps ``node`` into a ``CPU_Persistent`` map: one OpenMP team (``omp parallel``) runs every stage. The map's
    parameter is the thread index; its range is not read. The team has ``num_threads`` threads, or OpenMP's default
    for 0; every thread reads the team size into ``THREAD_COUNT`` when the body starts."""
    entry, exit_ = host.add_map(handle, {THREAD_INDEX: "0:1"}, schedule=dtypes.ScheduleType.CPU_Persistent)
    entry.map.omp_num_threads = num_threads
    body = node.sdfg
    body.add_symbol(THREAD_INDEX, dtypes.int64)
    node.symbol_mapping[THREAD_INDEX] = THREAD_INDEX
    # Every thread reads its own copy: no barrier orders it
    size = body.add_scalar(
        "__mk_team_size", dtypes.int64, transient=True, storage=dtypes.StorageType.CPU_ThreadLocal, find_new_name=True
    )[0]
    state = body.add_state_before(body.start_block, f"{handle}_team_size", is_start_block=True)
    tasklet = state.add_tasklet(
        "team_size",
        {},
        {"size"},
        "size = omp_get_num_threads();",
        language=dtypes.Language.CPP,
        code_global="#include <omp.h>",
    )
    state.add_edge(tasklet, "size", state.add_write(size), None, Memlet(f"{size}[0]"))
    body.out_edges(state)[0].data.assignments[THREAD_COUNT] = size
    wrap(host, node, entry, exit_)


def wrap(host: SDFGState, node: nodes.NestedSDFG, entry: nodes.MapEntry, exit_: nodes.MapExit) -> None:
    """Routes the connectors of ``node`` through the map ``entry`` / ``exit_``."""
    for edge in host.in_edges(node):
        host.remove_edge(edge)
        host.add_memlet_path(edge.src, entry, node, dst_conn=edge.dst_conn, memlet=edge.data)
    for edge in host.out_edges(node):
        host.remove_edge(edge)
        host.add_memlet_path(node, exit_, edge.dst, src_conn=edge.src_conn, memlet=edge.data)
    if host.in_degree(node) == 0:
        host.add_nedge(entry, node, Memlet())
    if host.out_degree(node) == 0:
        host.add_nedge(node, exit_, Memlet())


@properties.make_properties
@transformation.explicit_cf_compatible
class MegaKernelOptimization(ppl.Pass):
    """Fuses the kernels of every maximal hierarchical linear chain in ``region`` (default: the whole SDFG) into one
    persistent kernel: on the GPU, a cooperative grid-stride kernel; on the CPU, one OpenMP team whose stages are
    ``omp for nowait`` loops. Both separate dependent stages with grid barriers.

    ``small_symbols`` names the symbols whose extents are too short to spread over the GPU grid: a map dimension
    whose extent names only them stays a loop in every thread. The pass reads no symbol facts of the SDFG.
    """

    CATEGORY: str = "Optimization Preparation"

    small_symbols = properties.SetProperty(
        element_type=str, default=set(), desc="Symbols whose extents stay a loop in every thread"
    )
    block_size = properties.Property(dtype=int, default=256, desc="Threads per block of a GPU megakernel")
    device = properties.EnumProperty(
        dtype=dtypes.DeviceType, default=dtypes.DeviceType.GPU, desc="The device whose kernels are fused"
    )
    num_threads = properties.Property(dtype=int, default=0, desc="Threads of a CPU team; 0 is OpenMP's default")
    sync_all = properties.Property(dtype=bool, default=True, desc="Grid barrier between every two stages")

    def __init__(
        self,
        region: ControlFlowRegion | None = None,
        small_symbols: set[str] | None = None,
        device: dtypes.DeviceType = dtypes.DeviceType.GPU,
    ) -> None:
        super().__init__()
        self._region = region
        self.small_symbols = set(small_symbols or ())
        self.device = device

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict) -> list[MegakernelChain] | None:
        """:return: The chains fused into a megakernel each."""
        target = TARGETS[self.device]
        GatherKernelStates(self.device).apply_pass(sdfg, {})
        fused = []
        for chain in find_chains(self._region if self._region is not None else sdfg, sdfg, target):
            nested = nest_chain(chain, target)
            if nested is None:
                continue
            host, node = nested
            handle = f"{chain.region.sdfg.name}_megakernel_{len(fused)}"
            node.sdfg.name = f"{handle}_body"
            if self.device == dtypes.DeviceType.GPU:
                launch_cooperatively(host, node, handle, self.block_size)
            else:
                launch_team(host, node, handle, self.num_threads)
            self.lower_body(node.sdfg, target)
            fused.append(chain)
        return fused or None

    def lower_body(self, body: SDFG, target: Target) -> None:
        syncs = InsertGridSyncs(self.device, self.sync_all, small_symbols=self.small_symbols)
        syncs.apply_pass(body, {})
        plan = syncs.plan
        gpu = self.device == dtypes.DeviceType.GPU
        if plan is not None and plan.owners:
            # GPU: neighboring threads take neighboring iterations (coalesced). CPU: every thread one contiguous block
            # of the fastest key dimension (no false sharing)
            block = 1 if gpu else symbolic.int_ceil(plan.extent(), symbolic.pystr_to_symbolic(THREAD_COUNT))
        for state in body.states():
            for entry in [n for n in state.nodes() if is_kernel(n, state, target)]:
                if plan is not None and state in plan.owners:
                    own_stride(body, state, entry, plan.owners[state], plan, THREAD_INDEX, THREAD_COUNT, block)
                elif gpu:
                    grid_stride(body, state, entry, THREAD_INDEX, THREAD_COUNT, self.small_symbols)
                else:  # Work-shared by the team, which the stage's barrier (if any) waits for
                    entry.map.omp_nowait = True
                    entry.map.omp_schedule = dtypes.OMPScheduleType.Static
