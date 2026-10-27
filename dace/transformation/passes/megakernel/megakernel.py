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
from dace.sdfg.graph import SubgraphView
from dace.sdfg.state import ControlFlowBlock, ControlFlowRegion, LoopRegion
from dace.transformation import helpers as xfh
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.passes.megakernel.discovery import MegakernelChain, find_chains, is_kernel
from dace.transformation.passes.megakernel.grid_stride import grid_stride
from dace.transformation.passes.megakernel.grid_sync import InsertGridSyncs

#: The thread index of the megakernel map, and the number of its threads, inside the body
THREAD_INDEX = "__mk_tid"
THREAD_COUNT = "__mk_nthreads"


def nested_blocks(blocks: list[ControlFlowBlock]) -> list[ControlFlowBlock]:
    """``blocks`` and every block inside them."""
    inner = [b for block in blocks if isinstance(block, ControlFlowRegion) for b in block.all_control_flow_blocks()]
    return blocks + inner


def assigned_symbols(chain: MegakernelChain) -> set[str]:
    """The symbols ``chain`` assigns: its loop variables and the assignments on its interstate edges."""
    blocks = chain.blocks
    edges = [e for b in blocks[:-1] for e in chain.region.out_edges(b)]
    regions = [b for b in nested_blocks(blocks) if isinstance(b, ControlFlowRegion)]
    edges += [e for region in regions for e in region.edges()]
    assigned = {r.loop_variable for r in regions if isinstance(r, LoopRegion) and r.loop_variable}
    return assigned.union(*(e.data.assignments.keys() for e in edges))


def keep_global_memory_outside(host: SDFGState, node: nodes.NestedSDFG) -> None:
    """Allocates the global-memory transients of ``node`` in its parent instead: a kernel cannot allocate them."""
    sdfg, body = host.sdfg, node.sdfg
    for name, desc in list(body.arrays.items()):
        if desc.transient and desc.storage == dtypes.StorageType.GPU_Global:
            sdfg.add_datadesc(name, copy.deepcopy(desc))
            desc.transient = False
            node.add_out_connector(name)
            host.add_edge(node, name, host.add_write(name), None, Memlet.from_array(name, desc))


def nest_chain(chain: MegakernelChain) -> tuple[SDFGState, nodes.NestedSDFG] | None:
    """Moves the blocks of ``chain`` into a nested SDFG, which global memory and the symbols the code after it reads
    stay outside of.

    :return: The state calling the nested SDFG and its node, or ``None`` (nothing changed) when code outside the chain
             reads a symbol the chain assigns: a kernel cannot hand it out.
    """
    region, blocks = chain.region, chain.blocks
    sdfg = region.sdfg
    if assigned_symbols(chain) & xfh.symbols_read_outside(sdfg, set(nested_blocks(blocks))):
        return None
    if len(blocks) == 1 and isinstance(blocks[0], SDFGState):
        host = blocks[0]
        node = xfh.nest_state_subgraph(sdfg, host, SubgraphView(host, host.nodes()))
    else:
        host = xfh.nest_sdfg_subgraph(sdfg, SubgraphView(region, blocks))
        node = next(n for n in host.nodes() if isinstance(n, nodes.NestedSDFG))
    keep_global_memory_outside(host, node)
    for sym in (THREAD_INDEX, THREAD_COUNT):
        node.sdfg.add_symbol(sym, dtypes.int64)
    node.symbol_mapping[THREAD_INDEX] = THREAD_INDEX
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
            nested = nest_chain(chain)
            if nested is None:
                continue
            host, node = nested
            launch_cooperatively(host, node, f"{chain.region.sdfg.name}_megakernel_{len(fused)}", self.block_size)
            self.lower_body(node.sdfg)
            fused.append(chain)
        return fused or None

    def lower_body(self, body: SDFG) -> None:
        InsertGridSyncs().apply_pass(body, {})
        for state in body.states():
            for entry in [n for n in state.nodes() if is_kernel(n, state)]:
                grid_stride(body, state, entry, THREAD_INDEX, THREAD_COUNT, self.small_symbols)
