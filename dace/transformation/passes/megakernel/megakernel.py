# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``MegaKernelOptimization``: run each maximal hierarchical linear chain of GPU kernels as one persistent,
cooperative kernel.

A chain moves into the body of one ``GPU_Device`` map that launches as many thread blocks as the device holds at once
(``ComputePersistentKernelRange``). In the body, every kernel becomes a grid-stride loop (``grid_stride``) and grid
barriers separate the dependent ones (``InsertGridSyncs``).
"""

import copy

from dace import SDFG, SDFGState, dtypes, properties
from dace.libraries.standard.nodes import ComputePersistentKernelRange
from dace.libraries.standard.nodes.compute_persistent_kernel_range import NBLK_CONNECTOR
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.state import ControlFlowBlock, ControlFlowRegion, LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.passes.megakernel.discovery import (
    MegakernelChain,
    code_names,
    find_chains,
    is_kernel,
    states_in,
)
from dace.transformation.passes.megakernel.grid_stride import grid_stride
from dace.transformation.passes.megakernel.grid_sync import InsertGridSyncs

#: The thread index of the megakernel map, and the number of its threads, inside the body
THREAD_INDEX = "__mk_tid"
THREAD_COUNT = "__mk_nthreads"


def chain_names(blocks: list[ControlFlowBlock], edges: list) -> tuple[set[str], set[str], set[str]]:
    """The containers ``blocks`` read and write, and the symbols the chain assigns (interstate edges, loop
    variables)."""
    reads, writes, assigned = set(), set(), set()
    sdfg = blocks[0].sdfg
    for state in (s for b in blocks for s in states_in(b)):
        state_reads, state_writes = state.read_and_write_sets()
        reads |= state_reads
        writes |= state_writes
    regions = [b for b in blocks if isinstance(b, ControlFlowRegion)]
    regions += [r for b in regions for r in b.all_control_flow_regions() if r is not b]
    for region in regions:
        edges = edges + list(region.edges())
        if isinstance(region, LoopRegion):
            assigned.add(region.loop_variable)
            codes = (region.loop_condition, region.init_statement, region.update_statement)
            reads |= {n for code in codes if code is not None for n in code_names(code)} & sdfg.arrays.keys()
    for edge in edges:
        assigned |= edge.data.assignments.keys()
        reads |= edge.data.read_symbols() & sdfg.arrays.keys()
    return reads, writes, assigned


def used_outside(sdfg: SDFG, inside: set[int]) -> tuple[set[str], set[str]]:
    """The containers and symbols that the states and interstate edges outside the blocks with ids ``inside`` use."""
    data, symbols = set(), set()
    for state in sdfg.states():
        if id(state) not in inside:
            data |= {n.data for n in state.data_nodes()}
            symbols |= state.free_symbols
    for edge in sdfg.all_interstate_edges():
        if id(edge.src) not in inside:
            symbols |= edge.data.free_symbols
    return data, symbols


def move_blocks(region: ControlFlowRegion, blocks: list[ControlFlowBlock], host: SDFGState, name: str) -> SDFG:
    """Moves the consecutive ``blocks`` of ``region``, with the edges between them, into a new SDFG; ``host`` takes
    their place in ``region``."""
    inner_edges = [e for b in blocks[:-1] for e in region.out_edges(b)]
    for edge in region.in_edges(blocks[0]):
        region.add_edge(edge.src, host, edge.data)
    for edge in region.out_edges(blocks[-1]):
        region.add_edge(host, edge.dst, edge.data)
    was_start = region.start_block is blocks[0]
    body = SDFG(name)
    for block in blocks:
        region.remove_node(block)
        body.add_node(block, is_start_block=block is blocks[0])
    if was_start:
        region.start_block = region.node_id(host)
    for edge in inner_edges:
        body.add_edge(edge.src, edge.dst, edge.data)
    body.reset_cfg_list()
    return body


def declare_body_data(sdfg: SDFG, body: SDFG, names: set[str], outside_data: set[str]) -> set[str]:
    """Declares the containers ``names`` and the free symbols in ``body``. Thread-private transients that only the
    body uses move into it; everything else, global memory included, stays allocated outside the kernel.

    :return: The transients only the body uses: their values never enter or leave it.
    """
    private = set()
    for data_name in sorted(names):
        desc = sdfg.arrays[data_name]
        if desc.transient and data_name not in outside_data:
            private.add(data_name)
            if desc.storage != dtypes.StorageType.GPU_Global:
                sdfg.remove_data(data_name, validate=False)
                body.add_datadesc(data_name, desc)
                continue
        desc = copy.deepcopy(desc)
        desc.transient = False
        body.add_datadesc(data_name, desc)
    for sym in sorted(body.free_symbols - body.symbols.keys() - body.arrays.keys() - {THREAD_INDEX, THREAD_COUNT}):
        body.add_symbol(sym, sdfg.symbols.get(sym, dtypes.int64))
    for sym in (THREAD_INDEX, THREAD_COUNT):
        body.add_symbol(sym, dtypes.int64)
    return private


def nest_chain(chain: MegakernelChain, host: SDFGState, name: str) -> nodes.NestedSDFG | None:
    """Moves the blocks of ``chain`` into a new SDFG, called from ``host``. The loop variables stay inside, unlike
    ``nest_sdfg_subgraph``, which hands them out.

    :return: The nested SDFG, or ``None`` (nothing changed) when code after the chain reads a symbol it assigns.
    """
    region, blocks = chain.region, chain.blocks
    sdfg = region.sdfg
    reads, writes, assigned = chain_names(blocks, [e for b in blocks[:-1] for e in region.out_edges(b)])
    inside = {id(b) for b in blocks} | {id(s) for b in blocks for s in states_in(b)}
    outside_data, outside_symbols = used_outside(sdfg, inside)
    if assigned & outside_symbols:
        return None
    body = move_blocks(region, blocks, host, name)
    private = declare_body_data(sdfg, body, reads | writes, outside_data)

    ins = dict.fromkeys(sorted(reads - private))
    outs = dict.fromkeys(n for n in sorted(writes) if not body.arrays[n].transient)
    node = host.add_nested_sdfg(body, ins, outs, {s: s for s in body.symbols})
    for n in ins:
        host.add_edge(host.add_read(n), None, node, n, Memlet.from_array(n, sdfg.arrays[n]))
    for n in outs:
        host.add_edge(node, n, host.add_write(n), None, Memlet.from_array(n, sdfg.arrays[n]))
    return node


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
    node.symbol_mapping[THREAD_COUNT] = f"{nblk} * {block_size}"
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
    """Fuses the GPU kernels of every maximal hierarchical linear chain in ``region`` (default: the whole SDFG) into
    one persistent kernel.

    ``small_symbols`` names the symbols whose extents are too short to spread over the grid: a map dimension whose
    extent names only them stays a loop in every thread. The pass reads no symbol facts of the SDFG.
    """

    CATEGORY: str = "Optimization Preparation"

    small_symbols = properties.SetProperty(
        element_type=str, default=set(), desc="Symbols whose extents stay a loop in every thread"
    )
    block_size = properties.Property(dtype=int, default=256, desc="Threads per block of the megakernel")

    def __init__(self, region: ControlFlowRegion | None = None, small_symbols: set[str] | None = None) -> None:
        super().__init__()
        self._region = region
        self.small_symbols = set(small_symbols or ())

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict) -> list[MegakernelChain] | None:
        """:return: The chains fused into a megakernel each."""
        fused = []
        for chain in find_chains(self._region if self._region is not None else sdfg, sdfg):
            handle = f"{chain.region.sdfg.name}_megakernel_{len(fused)}"
            host = chain.region.add_state(handle)
            node = nest_chain(chain, host, f"{handle}_body")
            if node is None:
                chain.region.remove_node(host)
                continue
            launch_cooperatively(host, node, handle, self.block_size)
            self.lower_body(node.sdfg)
            fused.append(chain)
        return fused or None

    def lower_body(self, body: SDFG) -> None:
        InsertGridSyncs().apply_pass(body, {})
        for state in body.states():
            for entry in [n for n in state.nodes() if is_kernel(n, state)]:
                grid_stride(body, state, entry, THREAD_INDEX, THREAD_COUNT, self.small_symbols)
