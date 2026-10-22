# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lowering of dynamic thread-block maps into thread-block maps that drive the ``dace::DynamicMap`` work queue."""

import copy

from ordered_set import OrderedSet

import dace
from dace import SDFG, config, dtypes, properties, symbolic
from dace.codegen import common
from dace.libraries.standard.nodes.sync import Sync, SyncScope
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg import nodes, propagation
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import SubgraphView
from dace.sdfg.state import (
    ConditionalBlock,
    ControlFlowBlock,
    ControlFlowRegion,
    LoopRegion,
    SDFGState,
    StateSubgraphView,
)
from dace.transformation import gpu_helpers, helpers, transformation
from dace.transformation import pass_pipeline as ppl
from dace.transformation.dataflow.add_threadblock_map import AddThreadBlockMap

#: Prefix of the containers and symbols the lowering adds
PREFIX = "__dace_dynmap"
#: The shared state of the queue of a kernel
QUEUE_STATE = f"{PREFIX}_state"


def queue_state_elements(fine_grained: bool, block_size: int) -> int:
    """The indices in the shared state of the queue (``dace::DynamicMap::shared_type``): the range of a block round
    and, fine-grained on CUDA, two lists of ``warp size`` squared indices per warp."""
    if not fine_grained or common.get_gpu_backend() == "hip":
        return 4
    warp_size = common.gpu_warp_size()
    return max(4, 2 * (block_size // warp_size) * warp_size * warp_size)


def is_dynamic_map(node: nodes.Node) -> bool:
    return isinstance(node, nodes.MapEntry) and node.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock_Dynamic


def has_dynamic_map(block: ControlFlowBlock) -> bool:
    """Whether a dynamic thread-block map is in ``block``, also through nested regions and SDFGs."""
    if isinstance(block, ConditionalBlock):
        return any(has_dynamic_map(b) for _, branch in block.branches for b in branch.nodes())
    if not isinstance(block, SDFGState):
        return any(has_dynamic_map(b) for b in block.nodes())
    return any(
        is_dynamic_map(node)
        or (isinstance(node, nodes.NestedSDFG) and any(has_dynamic_map(b) for b in node.sdfg.nodes()))
        for node in block.nodes()
    )


def add_sync(region: ControlFlowRegion, label: str, scope: SyncScope) -> SDFGState:
    """A state of ``region`` that holds only a barrier of ``scope``."""
    state = region.add_state(label)
    state.add_node(Sync(label, scope=scope))
    return state


def guard_block(region: ControlFlowRegion, block: ControlFlowBlock, condition: str) -> ConditionalBlock:
    """Runs ``block`` only where ``condition`` holds, by moving it into a conditional block that takes its place."""
    guard = ConditionalBlock(f"{block.label}_guard", sdfg=region.sdfg, parent=region)
    region.add_node(guard, is_start_block=region.start_block is block)
    for edge in region.in_edges(block):
        region.add_edge(edge.src, guard, edge.data)
    for edge in region.out_edges(block):
        region.add_edge(guard, edge.dst, edge.data)
    region.remove_node(block)
    branch = ControlFlowRegion(f"{block.label}_guarded", sdfg=region.sdfg, parent=guard)
    branch.add_node(block, is_start_block=True)
    guard.add_branch(CodeBlock(condition), branch)
    return guard


def forward_symbols(state: SDFGState, node: nodes.NestedSDFG, names: set[str]) -> None:
    """Makes the symbols ``names`` of ``state`` mean the same inside ``node``."""
    defined = state.symbols_defined_at(node)
    for name in sorted(names):
        mapped = node.symbol_mapping.get(name)
        if mapped is not None and str(mapped) != name:
            raise NotImplementedError(f"{node.label} maps {name} to {mapped}, so a guard on {name} cannot follow it")
        if name not in node.sdfg.symbols:
            node.sdfg.add_symbol(name, defined[name])
        node.symbol_mapping[name] = name


@properties.make_properties
@transformation.explicit_cf_compatible
class LowerDynamicThreadBlockMaps(ppl.Pass):
    """Lower ``GPU_ThreadBlock_Dynamic`` maps into thread-block maps that share their iterations through a work queue.

    The kernel that contains dynamic maps gets an explicit ``GPU_ThreadBlock`` map of
    ``compiler.cuda.dynamic_map_block_size`` threads, one per iteration of the kernel's last parameter, so that every
    thread of a block reaches the queue; what the thread runs outside of a dynamic map is guarded by the bound of its
    iteration. Each dynamic map becomes a nested SDFG in which every thread opens its range of iterations
    (``dace::DynamicMap::begin``, an empty range out of bounds) and then runs the batches the queue hands it in block,
    warp and thread rounds, each through a sequential copy of the map, with the parameter of the thread that owns the
    batch. The barriers between the steps of the queue are :class:`~dace.libraries.standard.nodes.sync.Sync` nodes.
    The queue's state is one ``GPU_Shared`` array per kernel, which its dynamic maps share one after the other, the
    per-thread cursor an opaque register. In a ``GPU_Persistent`` kernel, the ``GPU_Device`` maps nested in it that
    contain dynamic maps are lowered as kernels.
    """

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, object]) -> int | None:
        """Lowers the dynamic maps of every GPU kernel; returns how many, or ``None`` if there were none.

        :raises NotImplementedError: A dynamic map or its kernel has a shape the lowering does not support.
        """
        lowered = 0
        for kernel_sdfg, state, kernel in gpu_helpers.gpu_kernels(sdfg):
            if any(
                is_dynamic_map(n)
                or (isinstance(n, nodes.NestedSDFG) and any(has_dynamic_map(b) for b in n.sdfg.nodes()))
                for n in state.scope_subgraph(kernel).nodes()
            ):
                lowered += self.lower_kernel(kernel_sdfg, state, kernel)
        if lowered:
            sdfg.reset_cfg_list()
        return lowered or None

    def lower_kernel(self, sdfg: SDFG, state: SDFGState, kernel: nodes.MapEntry) -> int:
        """Gives the kernel a thread-block map and lowers its dynamic maps; returns how many."""
        if kernel.map.schedule == dtypes.ScheduleType.GPU_Persistent:
            # Each block runs the nested device maps in a grid-stride loop of its own, which a thread-block map spreads
            # over its threads like the blocks of a kernel
            devices = [
                n
                for n in state.scope_children()[kernel]
                if isinstance(n, nodes.MapEntry) and n.map.schedule == dtypes.ScheduleType.GPU_Device
            ]
            lowered = 0
            for device in devices:
                lowered += self.lower_device_map(sdfg, state, device)
            if any(
                is_dynamic_map(n)
                or (isinstance(n, nodes.NestedSDFG) and any(has_dynamic_map(b) for b in n.sdfg.nodes()))
                for n in state.scope_subgraph(kernel).nodes()
            ):
                raise NotImplementedError(
                    f"Dynamic thread-block maps in {kernel.map.label} are not nested in one of its device maps"
                )
            return lowered
        if kernel.map.schedule != dtypes.ScheduleType.GPU_Device:
            raise NotImplementedError(f"Dynamic thread-block maps in {kernel.map.schedule} kernels are unsupported")
        return self.lower_device_map(sdfg, state, kernel)

    def lower_device_map(self, sdfg: SDFG, state: SDFGState, kernel: nodes.MapEntry) -> int:
        """Gives the device map ``kernel`` a thread-block map and lowers its dynamic maps; returns how many."""
        if not any(
            is_dynamic_map(n) or (isinstance(n, nodes.NestedSDFG) and any(has_dynamic_map(b) for b in n.sdfg.nodes()))
            for n in state.scope_subgraph(kernel).nodes()
        ):
            return 0
        block_dims = gpu_helpers.dynamic_map_block_dims()
        if any(dim != 1 for dim in block_dims[1:]):
            raise NotImplementedError("Dynamic thread-block maps only support one-dimensional thread blocks")
        block_size = block_dims[0]
        kernel_exit = state.exit_node(kernel)
        body = list(state.all_nodes_between(kernel, kernel_exit))
        if any(isinstance(n, nodes.MapEntry) and n.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock for n in body):
            raise NotImplementedError("GPU_ThreadBlock and GPU_ThreadBlock_Dynamic maps cannot share a kernel")

        index_type = common.gpu_dynamic_map_index_type(sdfg, state, kernel)
        fine_grained = config.Config.get_bool("compiler", "cuda", "dynamic_map_fine_grained")
        queue = (
            f"dace::DynamicMap<{'true' if fine_grained else 'false'}, {block_size}, {common.gpu_warp_size()}, "
            f"{index_type.ctype}>"
        )

        # One thread per iteration of the last parameter, in blocks of ``block_size``: the thread-block map covers the
        # whole block, also past the end of the range, so that every thread of the block reaches the queue
        end = kernel.map.range[-1][1]
        kernel.map.gpu_block_size = [block_size, 1, 1]
        AddThreadBlockMap.apply_to(sdfg, verify=False, map_entry=kernel)
        thread_block = kernel
        param = thread_block.map.params[-1]
        first, _, stride = thread_block.map.range[-1]
        thread_block.map.range[-1] = (first, first + stride * (block_size - 1), stride)
        thread_block_exit = state.exit_node(thread_block)
        threads = helpers.nest_state_subgraph(
            sdfg,
            state,
            StateSubgraphView(state, list(state.all_nodes_between(thread_block, thread_block_exit))),
            name=f"{thread_block.map.label}_threads",
        )

        # Out of bounds, a thread only takes part in the queue
        guard = f"{param} <= {symbolic.symstr(end, cpp_mode=False)}"
        defined = state.symbols_defined_at(threads)
        for name in sorted({param} | {str(s) for s in symbolic.pystr_to_symbolic(end).free_symbols}):
            if name not in threads.sdfg.symbols:
                threads.sdfg.add_symbol(name, defined[name])
            threads.symbol_mapping.setdefault(name, name)
        # The queue's state, which the dynamic maps of the kernel share one after the other
        threads.sdfg.add_array(
            QUEUE_STATE,
            [queue_state_elements(fine_grained, block_size)],
            index_type,
            storage=dtypes.StorageType.GPU_Shared,
            transient=True,
        )
        listed = fine_grained and common.get_gpu_backend() != "hip"
        lowering = _KernelLowering(guard, param, queue, listed, index_type)
        lowering.lower_region(threads.sdfg)
        propagation.propagate_memlets_state(sdfg, state)
        return lowering.lowered


class _KernelLowering:
    """Lowers the dynamic maps below the thread-block map of one kernel."""

    def __init__(self, guard: str, param: str, queue: str, listed: bool, index_type: dtypes.typeclass) -> None:
        #: The condition selecting the threads of in-bounds iterations, over ``param`` and symbols of the kernel
        self.guard = guard
        self.guard_names = {str(s) for s in symbolic.pystr_to_symbolic(guard).free_symbols}
        #: The kernel parameter that identifies the iteration of a thread
        self.param = param
        #: The ``dace::DynamicMap`` instantiation
        self.queue = queue
        #: Whether the warps list their remaining items in the shared state (fine-grained on CUDA)
        self.listed = listed
        self.index_type = index_type
        self.lowered = 0

    def lower_region(self, region: ControlFlowRegion) -> None:
        """Guards the blocks of ``region`` that hold no dynamic map and lowers the dynamic maps of the others."""
        arrays = set(region.sdfg.arrays)
        for edge in region.edges():
            if edge.data.read_symbols() & arrays:
                raise NotImplementedError(
                    f"The interstate edge {edge.src.label} -> {edge.dst.label} reads data, which out-of-bounds "
                    "threads would read too"
                )
        for block in list(region.nodes()):
            if not has_dynamic_map(block):
                guard_block(region, block, self.guard)
            elif isinstance(block, SDFGState):
                self.lower_state(block)
            else:
                raise NotImplementedError(
                    f"A dynamic map in {block.label} runs under control flow the threads of a block may not agree on"
                )

    def lower_state(self, state: SDFGState) -> None:
        """Lowers the dynamic maps of ``state``, then splits it so that what the dynamic maps need runs guarded."""
        collective: list[nodes.NestedSDFG] = []
        for node in list(sdutil.dfs_topological_sort(state)):
            if is_dynamic_map(node):
                if state.entry_node(node) is not None:
                    raise NotImplementedError(f"The dynamic map {node.map.label} is nested in another scope")
                collective.append(self.lower_dynamic_map(state, node))
            elif isinstance(node, nodes.NestedSDFG) and any(has_dynamic_map(b) for b in node.sdfg.nodes()):
                if state.entry_node(node) is not None:
                    raise NotImplementedError(f"The dynamic maps in {node.label} are nested in another scope")
                forward_symbols(state, node, self.guard_names)
                desc = copy.deepcopy(state.sdfg.arrays[QUEUE_STATE])
                desc.transient = False
                node.sdfg.add_datadesc(QUEUE_STATE, desc)
                self.lower_region(node.sdfg)
                collective.append(node)
        # The collective nodes take turns on the queue's state
        previous = None
        for node in collective:
            node.add_in_connector(QUEUE_STATE, force=True)
            node.add_out_connector(QUEUE_STATE, force=True)
            memlet = Memlet.from_array(QUEUE_STATE, state.sdfg.arrays[QUEUE_STATE])
            state.add_edge(previous or state.add_access(QUEUE_STATE), None, node, QUEUE_STATE, copy.deepcopy(memlet))
            previous = state.add_access(QUEUE_STATE)
            state.add_edge(node, QUEUE_STATE, previous, None, memlet)
        self.split_state(state, collective)

    def split_state(self, state: SDFGState, collective: list[nodes.NestedSDFG]) -> None:
        """Splits ``state`` into states that either hold only what every thread runs, or only what runs guarded."""
        top_level = [n for n in state.nodes() if state.entry_node(n) is None and not isinstance(n, nodes.ExitNode)]
        guarded = [n for n in top_level if not isinstance(n, nodes.AccessNode) and n not in collective]
        if not guarded:
            return
        # How many collective nodes run before each node: a node runs in the layer after them
        after: dict[nodes.Node, int] = {}
        for node in sdutil.dfs_topological_sort(state):
            after[node] = max((after[p] + (p in collective) for p in state.predecessors(node)), default=0)
        layer = {n: 2 * after[n] + (n in collective) for n in top_level if not isinstance(n, nodes.AccessNode)}
        layers = sorted(set(layer.values()))
        parts: list[tuple[SDFGState, bool]] = []
        for current in layers[:-1]:
            first = [
                inner
                for n, at in layer.items()
                if at == current
                for inner in (state.scope_subgraph(n).nodes() if isinstance(n, nodes.EntryNode) else [n])
            ]
            parts.append((helpers.state_fission(SubgraphView(state, first)), current % 2 == 1))
        parts.append((state, layers[-1] % 2 == 1))
        region = state.parent_graph
        for part, is_collective in parts:
            mixed = [
                n
                for n in part.nodes()
                if not isinstance(n, nodes.AccessNode)
                and part.entry_node(n) is None
                and (n in collective) != is_collective
            ]
            if mixed:
                raise NotImplementedError(f"Cannot separate {mixed} from the dynamic maps of {state.label}")
            if not is_collective:
                guard_block(region, part, self.guard)

    def lower_dynamic_map(self, state: SDFGState, entry: nodes.MapEntry) -> nodes.NestedSDFG:
        """Replaces the dynamic map ``entry`` with a nested SDFG that runs its iterations through the queue."""
        sdfg = state.sdfg
        exit_ = state.exit_node(entry)
        if len(entry.map.params) != 1:
            raise NotImplementedError(f"The dynamic map {entry.map.label} is not one-dimensional")
        map_param = entry.map.params[0]
        begin, end, step = entry.map.range[0]
        defined = state.symbols_defined_at(entry)
        map_types = entry.new_symbols(sdfg, state, defined)
        body = list(state.all_nodes_between(entry, exit_))
        dynamic_inputs = [e for e in state.in_edges(entry) if e.dst_conn and not e.dst_conn.startswith("IN_")]
        inputs = [e for e in state.in_edges(entry) if e not in dynamic_inputs and not e.data.is_empty()]
        outputs = [e for e in state.out_edges(exit_) if not e.data.is_empty()]
        for edge in dynamic_inputs + inputs + outputs:
            if not isinstance(edge.dst if edge in outputs else edge.src, nodes.AccessNode):
                raise NotImplementedError(f"The dynamic map {entry.map.label} is connected to a non-access node")

        lowered = SDFG(f"{entry.map.label}_lowered")
        # The map's data, whole, under the same names: its subsets keep their meaning
        outer_names = OrderedSet(e.data.data for e in dynamic_inputs + inputs + outputs)
        for name in outer_names:
            desc = copy.deepcopy(sdfg.arrays[name])
            desc.transient = False
            lowered.add_datadesc(name, desc)
        # The map's own transients move along
        moved = OrderedSet(n.data for n in body if isinstance(n, nodes.AccessNode) and n.data not in lowered.arrays)
        for name in moved:
            lowered.add_datadesc(name, copy.deepcopy(sdfg.arrays[name]))

        names = {key: f"{PREFIX}_{key}" for key in ("lo", "hi", "cursor", "src", "first", "count", "stride")}
        names["state"] = QUEUE_STATE
        symbols = {key: f"{PREFIX}_{key}_sym" for key in ("more", "first", "count", "stride")}
        owner = f"{PREFIX}_owner"
        register = dtypes.StorageType.Register
        for key in ("lo", "hi", "src", "first", "count", "stride"):
            lowered.add_scalar(names[key], self.index_type, storage=register, transient=True)
        names["more"] = f"{PREFIX}_more"
        lowered.add_scalar(names["more"], dtypes.bool_, storage=register, transient=True)
        lowered.add_scalar(
            names["cursor"], dtypes.opaque(f"{self.queue}::cursor_type"), storage=register, transient=True
        )
        queue_desc = copy.deepcopy(sdfg.arrays[QUEUE_STATE])
        queue_desc.transient = False
        lowered.add_datadesc(QUEUE_STATE, queue_desc)
        queue_state = Memlet.from_array(names["state"], lowered.arrays[names["state"]])
        shared = f"*reinterpret_cast<{self.queue}::shared_type *>(__state)"

        # The thread's range of the map's (normalized) iterations, empty out of bounds
        init = lowered.add_state("empty_range", is_start_block=True)
        tasklet = init.add_tasklet("empty_range", {}, {"__lo", "__hi"}, "__lo = 0\n__hi = 0")
        init.add_edge(tasklet, "__lo", init.add_write(names["lo"]), None, Memlet(f"{names['lo']}[0]"))
        init.add_edge(tasklet, "__hi", init.add_write(names["hi"]), None, Memlet(f"{names['hi']}[0]"))
        bounds_region = ControlFlowRegion("bounds_region", sdfg=lowered)
        bounds = bounds_region.add_state("bounds", is_start_block=True)
        if step == 1:
            code = f"__lo = {symbolic.symstr(begin, cpp_mode=False)}\n__hi = {symbolic.symstr(end + 1, cpp_mode=False)}"
        else:
            code = f"__lo = 0\n__hi = {symbolic.symstr(symbolic.int_ceil(end + 1 - begin, step), cpp_mode=False)}"
        tasklet = bounds.add_tasklet("bounds", {e.dst_conn for e in dynamic_inputs}, {"__lo", "__hi"}, code)
        self.read_dynamic_inputs(bounds, dynamic_inputs, tasklet)
        bounds.add_edge(tasklet, "__lo", bounds.add_write(names["lo"]), None, Memlet(f"{names['lo']}[0]"))
        bounds.add_edge(tasklet, "__hi", bounds.add_write(names["hi"]), None, Memlet(f"{names['hi']}[0]"))
        bounds_guard = ConditionalBlock("bounds_guard", sdfg=lowered, parent=lowered)
        lowered.add_node(bounds_guard)
        bounds_guard.add_branch(CodeBlock(self.guard), bounds_region)
        lowered.add_edge(init, bounds_guard, dace.InterstateEdge())

        # Every thread of the block opens its range
        opening = lowered.add_state("begin")
        tasklet = opening.add_tasklet(
            "begin",
            {"__lo", "__hi", "__state"},
            {"__cursor"},
            f"__cursor = {self.queue}::open(__lo, __hi, {self.param});\n{self.queue}::reset({shared});",
            language=dtypes.Language.CPP,
            side_effects=True,
        )
        opening.add_edge(opening.add_read(names["lo"]), None, tasklet, "__lo", Memlet(f"{names['lo']}[0]"))
        opening.add_edge(opening.add_read(names["hi"]), None, tasklet, "__hi", Memlet(f"{names['hi']}[0]"))
        opening.add_edge(opening.add_access(names["state"]), None, tasklet, "__state", copy.deepcopy(queue_state))
        opening.add_edge(tasklet, "__cursor", opening.add_write(names["cursor"]), None, Memlet(f"{names['cursor']}[0]"))
        lowered.add_edge(bounds_guard, opening, dace.InterstateEdge())
        previous = add_sync(lowered, "begin_sync", SyncScope.BLOCK)
        lowered.add_edge(opening, previous, dace.InterstateEdge())

        # The rounds of each phase of the queue: in each, every thread receives a strided batch of the iterations of
        # one owner, which the map runs sequentially, with the parameter and dynamic inputs of that owner. Each phase
        # has its own loop and copy of the map, so that the compiler sees the owner the same across a warp.
        first, count, stride = (symbolic.pystr_to_symbolic(symbols[key]) for key in ("first", "count", "stride"))
        last = first + (count - 1) * stride
        entry.map.range = (
            [(first, last, stride)] if step == 1 else [(begin + step * first, begin + step * last, step * stride)]
        )
        entry.map.schedule = dtypes.ScheduleType.Sequential
        batch = dace.SDFGState("batch", sdfg=lowered)
        self.move_scope(state, entry, exit_, batch)
        batch.replace(self.param, owner)
        batch_symbols = {symbols[key]: names[key] for key in ("first", "count", "stride")}
        batch_symbols[owner] = names["src"]

        def step_state(region: ControlFlowRegion, label: str, code: str, outputs: tuple[str, ...]) -> SDFGState:
            """A state of ``region`` that runs ``code`` of the queue on the cursor and the shared state."""
            step_state = region.add_state(label)
            tasklet = step_state.add_tasklet(
                label,
                {"__cursor_in", "__state"},
                {f"__{key}" for key in outputs},
                code,
                language=dtypes.Language.CPP,
                side_effects=True,
            )
            step_state.add_edge(
                step_state.add_access(names["state"]), None, tasklet, "__state", copy.deepcopy(queue_state)
            )
            step_state.add_edge(
                step_state.add_read(names["cursor"]), None, tasklet, "__cursor_in", Memlet(f"{names['cursor']}[0]")
            )
            for key in outputs:
                step_state.add_edge(
                    tasklet, f"__{key}", step_state.add_write(names[key]), None, Memlet(f"{names[key]}[0]")
                )
            return step_state

        def run_batch(region: ControlFlowRegion, phase: str, copied: bool) -> ConditionalBlock:
            """A conditional block of ``region`` that runs the batch of the thread, if it has one."""
            work_region = ControlFlowRegion(f"{phase}_batch_region", sdfg=lowered)
            work = ConditionalBlock(f"{phase}_batch", sdfg=lowered, parent=region)
            work.add_branch(CodeBlock(f"{symbols['count']} > 0"), work_region)
            region.add_node(work)
            phase_batch = copy.deepcopy(batch) if copied else batch
            phase_batch.label = f"{phase}_batch"
            work_region.add_node(phase_batch, is_start_block=True)
            propagation.propagate_memlets_state(lowered, phase_batch)
            return work

        def add_loop(label: str) -> LoopRegion:
            nonlocal previous
            loop = LoopRegion(f"{entry.map.label}_{label}", condition_expr=symbols["more"], inverted=True, sdfg=lowered)
            lowered.add_node(loop)
            lowered.add_edge(previous, loop, dace.InterstateEdge())
            previous = loop
            return loop

        def add_lowered_sync(label: str, scope: SyncScope) -> None:
            nonlocal previous
            sync = add_sync(lowered, label, scope)
            lowered.add_edge(previous, sync, dace.InterstateEdge())
            previous = sync

        # Block rounds: the threads with a range of a block elect one, which publishes it to the block
        loop = add_loop("block_rounds")
        elect = step_state(loop, "block_elect", f"{self.queue}::block_elect({shared}, __cursor_in);", ())
        loop.start_block = loop.node_id(elect)
        elect_sync = add_sync(loop, "block_elect_sync", SyncScope.BLOCK)
        loop.add_edge(elect, elect_sync, dace.InterstateEdge())
        claim = step_state(
            loop,
            "block_claim",
            f"__cursor = __cursor_in;\n__more = {self.queue}::block_claim({shared}, __cursor);",
            ("cursor", "more"),
        )
        loop.add_edge(elect_sync, claim, dace.InterstateEdge())
        taking = ControlFlowRegion("block_take_region", sdfg=lowered)
        take_block = ConditionalBlock("block_take", sdfg=lowered, parent=loop)
        take_block.add_branch(CodeBlock(symbols["more"]), taking)
        loop.add_node(take_block)
        loop.add_edge(claim, take_block, dace.InterstateEdge(assignments={symbols["more"]: names["more"]}))
        claim_sync = add_sync(taking, "block_claim_sync", SyncScope.BLOCK)
        taking.start_block = taking.node_id(claim_sync)
        take = step_state(
            taking,
            "block_take",
            f"__cursor = __cursor_in;\n{self.queue}::block_take({shared}, __cursor, __src, __first, __count, __stride);",
            ("cursor", "src", "first", "count", "stride"),
        )
        taking.add_edge(claim_sync, take, dace.InterstateEdge())
        take_sync = add_sync(taking, "block_take_sync", SyncScope.BLOCK)
        taking.add_edge(take, take_sync, dace.InterstateEdge(assignments=dict(batch_symbols)))
        taking.add_edge(take_sync, run_batch(taking, "block", copied=True), dace.InterstateEdge())
        if self.listed:
            # The lists of the warps overlap the range of the block rounds
            add_lowered_sync("block_rounds_sync", SyncScope.BLOCK)

        # Warp rounds, then the remaining iterations, in thread rounds
        rounds = (
            ("warp", f"__more = {self.queue}::warp_round({shared}, __cursor, __src, __first, __count, __stride);"),
            (
                "thread",
                f"__more = {self.queue}::thread_round({shared}, __cursor, __src, __first, __count);\n__stride = 1;",
            ),
        )
        for phase, call in rounds:
            loop = add_loop(f"{phase}_rounds")
            round_state = step_state(
                loop,
                f"{phase}_round",
                f"__cursor = __cursor_in;\n{call}",
                ("cursor", "src", "first", "count", "stride", "more"),
            )
            loop.start_block = loop.node_id(round_state)
            assignments = dict(batch_symbols)
            assignments[symbols["more"]] = names["more"]
            loop.add_edge(
                round_state,
                run_batch(loop, phase, copied=phase != "thread"),
                dace.InterstateEdge(assignments=assignments),
            )
            if self.listed and phase == "warp":
                # The lists of the warp are written before the thread rounds read them
                add_lowered_sync("warp_rounds_sync", SyncScope.WARP)
        if self.listed:
            # The next dynamic map of the kernel resets the state, which the lists overlap
            add_lowered_sync("thread_rounds_sync", SyncScope.BLOCK)
        # The copies of the map hold copies of its nested SDFGs
        sdutil.set_nested_sdfg_parent_references(lowered)

        # Replace the map scope with the nested SDFG
        input_names = OrderedSet(e.data.data for e in dynamic_inputs + inputs) | {QUEUE_STATE}
        output_names = OrderedSet(e.data.data for e in outputs) | {QUEUE_STATE}
        symbol_mapping = {}
        for name in sorted(lowered.free_symbols | {self.param}):
            if name in defined:
                if name not in lowered.symbols:
                    lowered.add_symbol(name, defined[name])
                symbol_mapping[name] = name
        for name in (owner, symbols["first"], symbols["count"], symbols["stride"]):
            lowered.add_symbol(name, self.index_type)
        lowered.add_symbol(symbols["more"], dtypes.bool_)
        sources = {}
        for edge in dynamic_inputs + inputs:
            sources.setdefault(edge.data.data, edge.src)
        destinations = {}
        for edge in outputs:
            destinations.setdefault(edge.data.data, (edge.dst, edge.data.wcr))
        node = state.add_nested_sdfg(lowered, input_names, output_names, symbol_mapping=symbol_mapping)
        for name, src in sources.items():
            state.add_edge(src, None, node, name, Memlet.from_array(name, sdfg.arrays[name]))
        for name, (dst, wcr) in destinations.items():
            memlet = Memlet.from_array(name, sdfg.arrays[name])
            memlet.wcr = wcr
            state.add_edge(node, name, dst, None, memlet)
        # Another access node of the same data that only connected to the map
        for edge in dynamic_inputs + inputs + outputs:
            for access in (edge.src, edge.dst):
                if access in state.nodes() and isinstance(access, nodes.AccessNode) and state.degree(access) == 0:
                    state.remove_node(access)
        used = {n.data for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.AccessNode)}
        for name in moved:
            if name not in used:
                sdfg.remove_data(name, validate=False)
        self.lowered += 1
        return node

    @staticmethod
    def read_dynamic_inputs(state: SDFGState, dynamic_inputs: list, tasklet: nodes.Tasklet) -> None:
        """Reads the dynamic inputs of a map into the connectors of ``tasklet`` of the same name."""
        reads = {}
        for edge in dynamic_inputs:
            if edge.data.data not in reads:
                reads[edge.data.data] = state.add_read(edge.data.data)
            state.add_edge(reads[edge.data.data], None, tasklet, edge.dst_conn, copy.deepcopy(edge.data))

    @staticmethod
    def move_scope(state: SDFGState, entry: nodes.MapEntry, exit_: nodes.MapExit, target: SDFGState) -> None:
        """Moves the map scope into ``target``, where its data is read and written through access nodes of its own."""
        scope = state.scope_subgraph(entry).nodes()
        inside = set(scope)
        edges = [e for e in state.edges() if e.src in inside or e.dst in inside]
        reads: dict[str, nodes.AccessNode] = {}
        writes: dict[str, nodes.AccessNode] = {}
        for node in scope:
            target.add_node(node)
        for edge in edges:
            if edge.src in inside and edge.dst in inside:
                target.add_edge(edge.src, edge.src_conn, edge.dst, edge.dst_conn, edge.data)
            elif edge.data.is_empty():
                continue
            elif edge.dst in inside:
                if edge.data.data not in reads:
                    reads[edge.data.data] = target.add_read(edge.data.data)
                target.add_edge(reads[edge.data.data], None, edge.dst, edge.dst_conn, copy.deepcopy(edge.data))
            else:
                if edge.data.data not in writes:
                    writes[edge.data.data] = target.add_write(edge.data.data)
                target.add_edge(edge.src, edge.src_conn, writes[edge.data.data], None, copy.deepcopy(edge.data))
        for node in scope:
            state.remove_node(node)
