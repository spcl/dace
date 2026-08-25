# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Decide what runs on the accelerator, and place the host/device copies that follow from it.

The pass reads the SDFG once into a small control-flow IR (:class:`OffloadingIRNode`), records per
block which arrays are wanted on the CPU and which on the GPU, propagates those sets along the
graph, and only then materializes copies -- so a copy is emitted where the location actually
changes rather than around every kernel.
"""
from copy import deepcopy
from typing import Any, Set

from ordered_set import OrderedSet

from dace import dtypes, properties, data, Memlet, subsets
from dace.config import Config
from dace.sdfg import nodes, SDFG, InterstateEdge
from dace.sdfg.state import (SDFGState, ConditionalBlock, ControlFlowRegion, LoopRegion, ReturnBlock, ContinueBlock,
                             BreakBlock, ControlFlowBlock, AbstractControlFlowRegion)
from dace.sdfg.utils import get_last_view_node
from dace.transformation import pass_pipeline as ppl
from dace.transformation.transformation import explicit_cf_compatible
from dace.transformation.passes import FullMapFusion
from dace.transformation.passes.length_one_array_scalar_conversion import (ConvertLengthOneArraysToScalars,
                                                                           ConvertScalarsToLengthOneArrays)
from dace.transformation.passes.offloading.taskloop import taskloop_maps

PRINT_NAMES = 500

# scope dictionary: cache it
# replace_dict: always bacth as much as possible
# no has_attr, get_attr


class OffloadingIRNode:
    # INVARIANT: IR-trees are always DAGs
    STATE = -1
    OPEN = 0
    CLOSE = 1
    OPEN_LOOP = 2
    OPEN_COND = 3
    EDGE = 4  # interstate edge

    def __init__(self, type: int, block: ControlFlowBlock, cpu_set: set, gpu_set: set, next: list, close):
        assert block is None or isinstance(block, ControlFlowBlock), f"{block}, {block.__class__.__name__}"
        self.type = type
        self.block: ControlFlowBlock = block
        self.cpu_set: set[str] = cpu_set
        self.gpu_set: set[str] = gpu_set
        self.next: list[OffloadingIRNode] = next
        self.close = close

        self.open = None
        self.debug_name = "Cpt. Nemo"
        self.copy_successor = False

        # there should be a reference to the corresponding close node IFF the current node is an open node
        assert (
            self.close
            is not None) == self.is_open_node(), f"node {self.debug_name} of type {self.type} has close {self.close}"

    def __repr__(self):
        return self._get_str(set(), -4)

    def __str__(self):
        return self.__repr__()

    def _get_str(self, visited_set, len_before):
        s = f"{self.debug_name}:"
        spaces = 40 - (len_before + len(s))
        cpu = sorted(name for name in self.cpu_set if len(name) <= PRINT_NAMES)
        gpu = sorted(name for name in self.gpu_set if len(name) <= PRINT_NAMES)
        s += spaces * " " + f"cpu = {cpu}, gpu = {gpu}\n"

        if self in visited_set:
            return s
        visited_set.add(self)

        next_list = sorted(self.next, key=lambda x: x.debug_name)
        for next in next_list:
            s += f"{self.debug_name} => {next._get_str(visited_set, len(self.debug_name))}"
        return s

    # utility functions
    def is_empty(self):
        return not self.cpu_set and not self.gpu_set

    def is_open_node(self):
        return self.type in [OffloadingIRNode.OPEN, OffloadingIRNode.OPEN_LOOP, OffloadingIRNode.OPEN_COND]

    def is_close_node(self):
        return self.type in [OffloadingIRNode.CLOSE]

    def append_node(self, node):
        self.next.append(node)

    def get_all_tails(self):
        assert self.is_open_node()

        def recursion(node, result: list):
            for next in node.next:
                if next == self.close:  # definition of a tail: a node that points at this section's end (close-node)
                    result.append(node)
                    return
                recursion(next, result)

        result = []
        recursion(self, result)
        return result

    # static makers
    def new_open_node(block: ControlFlowBlock):
        close = OffloadingIRNode(OffloadingIRNode.CLOSE, None, set(), set(), [], None)
        close.debug_name = f"_close_{block.label}"

        type: int
        if isinstance(block, LoopRegion):
            type = OffloadingIRNode.OPEN_LOOP
        elif isinstance(block, ConditionalBlock):
            type = OffloadingIRNode.OPEN_COND
        else:
            type = OffloadingIRNode.OPEN

        open = OffloadingIRNode(type, block, set(), set(), [], close)
        open.debug_name = f"_{OffloadingIRNode.get_type_as_str(type)}_{block.label}"
        close.open = open

        return open

    def new_state_node(block: ControlFlowBlock, cpu_set: set, gpu_set: set):
        state = OffloadingIRNode(OffloadingIRNode.STATE, block, cpu_set, gpu_set, [], None)
        state.debug_name = f"_state_{block.label}"
        return state

    def new_edge_node(edge: InterstateEdge, cpu_set: set):
        edge_node = OffloadingIRNode(OffloadingIRNode.EDGE, edge, cpu_set, set(), [], None)
        edge_node.debug_name = f"_edge_{edge.label}"
        return edge_node

    def get_type_as_str(type: int):
        match type:
            case OffloadingIRNode.STATE:
                return "state"
            case OffloadingIRNode.OPEN:
                return "open"
            case OffloadingIRNode.OPEN:
                return "close"
            case OffloadingIRNode.OPEN_LOOP:
                return "loop"
            case OffloadingIRNode.OPEN_COND:
                return "cond"
        raise ValueError(f"Invalid IR type to convert to string: {type}")


@properties.make_properties
@explicit_cf_compatible
class OffloadToAccelerator(ppl.Pass):
    """Move the work an accelerator can take to it, and insert the copies that decision implies.

    Replaces :class:`~dace.transformation.interstate.gpu_transform_sdfg.GPUTransformSDFG` under
    ``optimizer.new_gpu_offloading_pass``. The difference is where the copies land: the old
    transformation copies in and out around the kernels it makes, while this one propagates the
    wanted location of every array through the control flow first, so an array that stays on the
    device across a whole loop is copied once rather than per iteration, and a host-only branch
    pays for its copies inside that branch.
    """

    CATEGORY: str = 'Offload To Accelerator'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    #def depends_on(self) -> Set[Union[Type['Pass'], 'Pass']]:
    #    return set()

    #def report(self, pass_retval: Any) -> Optional[str]:
    #    """
    #    Returns a user-readable string report based on the results of this pass.
    #
    #    :param pass_retval: The return value from applying this pass.
    #    :return: A string with the user-readable report, or None if nothing to report.
    #    """
    #    return None

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> Any | None:
        """
        Applies the pass to the given SDFG.

        :param sdfg: The SDFG to apply the pass to.
        :param pipeline_results: If in the context of a ``Pipeline``, a dictionary that is populated with prior Pass
                                 results as ``{Pass subclass name: returned object from pass}``. If not run in a
                                 pipeline, an empty dictionary is expected.
        :return: Some object if pass was applied, or None if nothing changed.
        """

        self.taskloop_heuristics = Config.get_bool('optimizer', 'gpu_taskloop_heuristics')
        self.cache_scopes(sdfg)

        # step 1: set schedule of maps and library nodes -> heuristic only!
        self.find_taskloops(sdfg)
        self.assign_schedules(sdfg)

        self.place_and_copy(sdfg)
        self.offload_taskloop_bodies(sdfg)

    def place_and_copy(self, sdfg: SDFG) -> None:
        """Place every array across ONE host level and copy accordingly; taskloop bodies come later."""
        # step 2:
        # Names already put through a conversion. A SIGNATURE descriptor is not rewritten in place:
        # ``preserve_abi`` stages a transient beside it and copies, so the array itself is still an
        # array on the next scan and would be requested again for ever (TSVC s332's ``result``).
        attempted: Set[str] = set()

        for _ in range(3):
            # step 2: copy analysis -> IR stores analysis results
            self.hybrid_states = set()
            sdfgIR = self.sdfg_to_IR(sdfg)

            # step 3: resolve hybrid states
            if self.hybrid_states:
                new_maps = set()
                for state in self.hybrid_states:
                    new_maps |= self.make_size1_map_wrappers(sdfg, state)

                mapfusion_pass = FullMapFusion(
                    strict_dataflow=True,
                    perform_vertical_map_fusion=True,
                    perform_horizontal_map_fusion=True,
                )
                mapfusion_pipeline = ppl.Pipeline([mapfusion_pass])
                mapfusion_pipeline.apply_pass(sdfg, {})

            # step 4: assign scalars / len1-arrays correctly
            all_scalars: set[str] = {data_name for data_name in sdfg.arrays if self._is_scalar(data_name, sdfg)}
            all_len1arrays: set[str] = {
                data_name
                for data_name in sdfg.arrays if self._is_length1_array(data_name, sdfg)
            }
            gpu_written = set()
            for state in sdfg.states():
                for node in state.nodes():
                    if isinstance(node, (nodes.MapExit, nodes.LibraryNode)) and self.has_GPU_schedule(node):
                        gpu_written |= self.get_data_used_by_outgoing_access_nodes(sdfg,
                                                                                   state,
                                                                                   node,
                                                                                   include_scalars=True)

            to_len1_arrays = (all_scalars & gpu_written) - attempted
            # ``__return`` stays by reference: the caller reads the result back through it.
            to_scalars = {
                name
                for name in (all_len1arrays - gpu_written) - attempted if not name.startswith("__return")
            }
            attempted |= to_len1_arrays | to_scalars
            # What the conversions REWROTE, not what they were asked to: both decline a descriptor
            # they cannot express (a View, an opaque handle, a signature array they may not stage),
            # and a request that is declined every round is a fixed point, not progress. Reading the
            # request instead spins until the retry budget runs out and then raises on a graph that
            # had already settled -- TSVC s332, whose non-transient ``result`` is refused each time.
            rewritten: Set[str] = set()
            if to_len1_arrays:
                rewritten |= ConvertScalarsToLengthOneArrays(
                    recursive=True,
                    preserve_abi=True,
                    filter=to_len1_arrays,
                ).apply_pass(sdfg, {}) or set()

            if to_scalars:
                rewritten |= ConvertLengthOneArraysToScalars(
                    recursive=True,
                    preserve_abi=True,
                    filter=to_scalars,
                ).apply_pass(sdfg, {}) or set()

            # rerun IR if anything has changed
            if rewritten or self.hybrid_states:  # sdfg has been changed
                self.cache_scopes(sdfg)
                continue  # repeat phases 2 - 4

            break  # else IR is correct for current sdfg, go on to next step

        else:
            raise RuntimeError("Offloading did not settle: the copy analysis, the hybrid-state "
                               "resolution and the scalar/len-1 assignment kept changing the graph "
                               "for 3 rounds.")

        # TODO: remove eventually
        def assert_no_scalars(node: OffloadingIRNode):
            scalars = {data_name for data_name in node.gpu_set | node.cpu_set if self._is_scalar(data_name, sdfg)}
            assert not scalars, (f"scalars {scalars} found in {node.debug_name}\n"
                                 f"\tgpu: {node.gpu_set}\n\tcpu: {node.cpu_set}")

        self.__traverse_IR(sdfgIR, assert_no_scalars)

        # step 4: insert copies based on IR
        self.eval_IR(sdfg, sdfgIR)

    def offload_taskloop_bodies(self, sdfg: SDFG) -> None:
        """Place again inside every nested SDFG a taskloop launches: the body is its own host level.

        Its connector-bound descriptors already carry this level's storage, so the body seeds from
        that and copies only what its own control flow needs. Schedules were assigned tree-wide.
        """
        if not self.taskloop_heuristics:
            return

        # Copy insertion added states; the cached scopes predate them.
        self.cache_scopes(sdfg)
        for state in sdfg.states():
            for entry in self.cached_scope_children[state].get(None, ()):
                if not isinstance(entry, nodes.MapEntry) or entry not in self.taskloops:
                    continue
                for node in self.host_level_nested_sdfgs(state, entry):
                    self.inherit_binding_storage(sdfg, state, node)
                    body = type(self)()
                    body.taskloop_heuristics = True
                    body.taskloops = self.taskloops
                    body.cache_scopes(node.sdfg)
                    body.place_and_copy(node.sdfg)
                    body.offload_taskloop_bodies(node.sdfg)

    def host_level_nested_sdfgs(self, state: SDFGState, entry: nodes.MapEntry):
        """Nested SDFGs under ``entry`` reached through taskloops only -- one below a kernel is device code."""
        for node in self.cached_scope_children[state].get(entry, ()):
            if isinstance(node, nodes.NestedSDFG):
                yield node
            elif isinstance(node, nodes.MapEntry) and node in self.taskloops:
                yield from self.host_level_nested_sdfgs(state, node)

    def inherit_binding_storage(self, sdfg: SDFG, state: SDFGState, nsdfg_node: nodes.NestedSDFG) -> None:
        """Bind inner descriptors to their outer storage: the body reads these as starting locations."""
        for edge in state.in_edges(nsdfg_node) + state.out_edges(nsdfg_node):
            if edge.data is None or edge.data.is_empty():
                continue
            connector = edge.dst_conn if edge.dst is nsdfg_node else edge.src_conn
            if connector is None or connector not in nsdfg_node.sdfg.arrays:
                continue
            outer = sdfg.arrays[edge.data.data]
            nsdfg_node.sdfg.arrays[connector].storage = outer.storage

    def cache_scopes(self, sdfg):
        # Nested SDFGs too: a taskloop's body is a host level, read by the walk and the analysis.
        self.cached_scopes = {}
        self.cached_scope_children = {}
        for nested in sdfg.all_sdfgs_recursive():
            for state in nested.states():
                self.cached_scopes[state] = state.scope_dict()
                self.cached_scope_children[state] = state.scope_children()

    ### STEP 1 ###
    def find_taskloops(self, sdfg: SDFG) -> None:
        """Record which maps only launch work; with the heuristics off, none do."""
        self.taskloops = taskloop_maps(sdfg) if self.taskloop_heuristics else OrderedSet()

    def assign_schedules(self, sdfg: SDFG, host_level: bool = True) -> None:
        """``GPU_Device`` at a host level, ``Sequential`` below one; a taskloop keeps its body host-level.

        With no taskloops this is the old rule: top-level maps are the kernels.
        """

        def walk(state: SDFGState, entry, host_level: bool) -> None:
            for node in self.cached_scope_children[state].get(entry, ()):
                if isinstance(node, nodes.MapEntry):
                    is_kernel = host_level and node not in self.taskloops
                    self.set_schedule(node,
                                      dtypes.ScheduleType.GPU_Device if is_kernel else dtypes.ScheduleType.Sequential)
                    walk(state, node, host_level and not is_kernel)

                elif isinstance(node, nodes.LibraryNode):
                    self.set_schedule(node,
                                      dtypes.ScheduleType.GPU_Device if host_level else dtypes.ScheduleType.Sequential)

                elif isinstance(node, nodes.NestedSDFG):
                    # A host level only under a taskloop; otherwise its maps stay sequential.
                    self.assign_schedules(node.sdfg, host_level and self.taskloop_heuristics)

        for state in sdfg.states():
            walk(state, None, host_level)

    def set_schedule(self, node, schedule: dtypes.ScheduleType) -> None:
        # Sequential specifically: Default can be lowered to CUDA in the wrong places.
        if schedule is dtypes.ScheduleType.Sequential and self.has_GPU_schedule(node):
            raise RuntimeError("Invalid SDFG for OffloadToAccelerator pass. All maps must have default or CPU "
                               f"schedule before pass. Node {node} has schedule type {self.get_schedule(node)}")
        node.schedule = schedule

    ### generic HELPERS ###

    def get_schedule(self, node):
        if isinstance(node, (nodes.MapEntry, nodes.MapExit)):
            return node.map.schedule
        if isinstance(node, nodes.LibraryNode):
            return node.schedule
        raise TypeError(f'node {node} of type {type(node).__name__} carries no schedule')

    def has_GPU_schedule(self, node):
        return self.get_schedule(node) in dtypes.GPU_SCHEDULES

    def get_children(self, state, node):
        return {e.dst for e in state.out_edges(node)}

    def get_predecessors(self, state, node):
        return {e.src for e in state.in_edges(node)}

    ### STEP 2: copy analysis ###

    ### Helpers to get the set of arrays accessed by specific nodes or edges ###

    def get_data_used_by_incoming_access_nodes(self,
                                               sdfg: SDFG,
                                               state: SDFGState,
                                               node: nodes.Node,
                                               include_scalars: bool = False) -> set[str]:

        def recursion(node: nodes.Node, visited_set: set[nodes.Node]):
            # the visited set is necessary for edge cases, e.g. an access node A whose predecessor B is a view node
            # refering back to A
            if node in visited_set:
                return set()
            visited_set.add(node)

            # find accessed arrays
            arrays: set[str] = set()
            if isinstance(node, nodes.AccessNode):
                data_name = node.data
                if self._is_array(data_name, sdfg):
                    arrays.add(data_name)

                elif self._is_view(data_name, sdfg):  # trace it if it is a view
                    original = get_last_view_node(
                        state, node
                    )  # once the view access node is known, its original access node can be found and it's data added
                    arrays |= recursion(original, visited_set)

                elif include_scalars and self._is_scalar(data_name, sdfg):
                    arrays.add(data_name)

            # check if more access nodes UPstream
            for n in self.get_predecessors(state, node):
                if isinstance(n, nodes.AccessNode):
                    arrays |= recursion(n, visited_set)

            return arrays

        return recursion(node, set())

    def get_data_used_by_outgoing_access_nodes(self,
                                               sdfg: SDFG,
                                               state: SDFGState,
                                               node: nodes.Node,
                                               include_scalars: bool = False) -> set[str]:

        def recursion(node: nodes.Node, visited_set: set[nodes.Node]):
            # the visited set is necessary for edge cases, e.g. an access node A whose successor B is a view node
            # refering back to A
            if node in visited_set:
                return set()
            visited_set.add(node)

            # find accessed arrays
            arrays: set[str] = set()
            if isinstance(node, nodes.AccessNode):
                data_name = node.data

                if self._is_array(data_name, sdfg):
                    arrays.add(data_name)

                elif self._is_view(data_name, sdfg):  # trace it if it is a view
                    original = get_last_view_node(
                        state, node
                    )  # once the view access node is known, its original access node can be found and it's data added
                    arrays |= recursion(original, visited_set)

                elif include_scalars and self._is_scalar(data_name, sdfg):
                    arrays.add(data_name)

            # check if more access nodes DOWNstream
            for n in self.get_children(state, node):
                if isinstance(n, nodes.AccessNode):
                    arrays |= recursion(n, visited_set)

            return arrays

        return recursion(node, set())

    def get_arrays_used_by_edge(self, sdfg: SDFG, state: SDFGState, edge, is_out_edge: bool):
        if edge.data and not edge.data.is_empty():
            data_name = edge.data.data

            if self._is_array(data_name, sdfg):  # array access on edge
                return {data_name}

            elif self._is_view(data_name,
                               sdfg):  # view -> we need to find the corresponding view access node by iteration.
                for n in state.data_nodes():
                    if n.data == data_name:
                        if is_out_edge:
                            return self.get_data_used_by_outgoing_access_nodes(sdfg, state, n)
                        return self.get_data_used_by_incoming_access_nodes(sdfg, state, n)

            elif self._is_scalar(data_name, sdfg):  # might be a scalar access of an array slice
                if is_out_edge:
                    if isinstance(edge.dst, nodes.AccessNode):
                        return self.get_data_used_by_outgoing_access_nodes(sdfg, state, edge.dst)
                else:
                    if isinstance(edge.src, nodes.AccessNode):
                        return self.get_data_used_by_incoming_access_nodes(sdfg, state, edge.src)

            else:
                raise RuntimeError(f"edge {edge} carries {edge.data}, which is neither an array, a scalar nor a view")

        return set()

    def get_arrays_used_by_node(self, sdfg, state, node):
        arrays: set[str] = set()

        # edges
        for e in state.in_edges(node):
            arrays |= self.get_arrays_used_by_edge(sdfg, state, e, False)

        for e in state.out_edges(node):
            arrays |= self.get_arrays_used_by_edge(sdfg, state, e, True)

        # neighbouring access nodes
        arrays |= self.get_data_used_by_incoming_access_nodes(sdfg, state, node)
        arrays |= self.get_data_used_by_outgoing_access_nodes(sdfg, state, node)

        return arrays

    ### Data Analysis: traverse the graph and sort all accessed arrays into gpu and cpu sets ###

    def get_data_locations_of_map(self, sdfg: SDFG, state: SDFGState, map_entry: nodes.MapEntry):
        """
        finds all arrays accessed by a map, i.e. arrays which are
            - part of the read/write set of an enclosed tasklet
            - data of an enclosed access node
            - accessed by a second, enclosed map
            - the original arrays behind an accessed view

        and decides whether their location should be on gpu or a cpu, i.e.
            - gpu if ANY parent map has a gpu schedule (even if the direct parent has cpu-schedule)
            - cpu else

        returns two sets (gpu_set, cpu_set) with the names of the respective arrays
        """

        # helper to validate data and add it to correct set
        def _add_data(data_name: str,
                      gpu_set: set[str],
                      cpu_set: set[str],
                      is_gpu: bool,
                      host_level: bool = False) -> tuple[set[str], set[str]]:
            if data_name in gpu_set:  # has already been accessed on GPU
                if not is_gpu:  # is now accessed on CPU
                    if host_level:
                        # A launcher staging on the host for a kernel is a hybrid state, not an error.
                        cpu_set.add(data_name)
                        return
                    raise RuntimeError("GPU->CPU inside a map: an inner sequential map still runs as a kernel, so data "
                                       "under a GPU map has to stay on the GPU")

            elif data_name in cpu_set:  # has already been accessed on CPU
                if is_gpu:  # is now accessed on GPU
                    gpu_set.add(data_name)
                    #raise RuntimeError("CPU->GPU copy needed within map for " + data_name)

            else:
                assert isinstance(data_name, str), f"{data_name} -> {data_name.__class__.__name__}"
                (gpu_set if is_gpu else cpu_set).add(data_name)

        # main work horse, can recurse to nested maps
        def _recursive_helper(sdfg: SDFG,
                              state: SDFGState,
                              map_entry: nodes.MapEntry,
                              gpu_set: set[str],
                              cpu_set: set[str],
                              is_gpu: bool,
                              host_level: bool = True):
            is_gpu = is_gpu or map_entry.map.schedule in dtypes.GPU_SCHEDULES  # TODO Q: how not to hardcode?
            is_taskloop = map_entry in self.taskloops
            host_level = host_level and is_taskloop

            # get all nodes within this map's scope
            map_nodes = [n for n, parent in self.cached_scopes[state].items() if parent is map_entry]

            # input & output nodes
            input_and_output = self.get_data_used_by_incoming_access_nodes(
                sdfg, state, map_entry) | self.get_data_used_by_outgoing_access_nodes(
                    sdfg, state, state.exit_node(map_entry))
            if is_taskloop:
                pass  # transparent for now, resolved below once the body has spoken
            elif is_gpu:
                gpu_set |= input_and_output
            else:
                cpu_set |= input_and_output

            # internal nodes
            for node in map_nodes:
                if isinstance(node, nodes.MapEntry):  # recurse on inner map
                    _recursive_helper(sdfg, state, node, gpu_set, cpu_set, is_gpu, host_level)

                elif isinstance(node, nodes.AccessNode):  # find accessed arrays -> add
                    # Staging between a launcher and a kernel says nothing about where data belongs.
                    if not host_level:
                        for name in self.get_data_used_by_outgoing_access_nodes(sdfg, state, node):
                            _add_data(name, gpu_set, cpu_set, is_gpu, host_level)

                elif isinstance(node, nodes.Tasklet):  # find accessed arrays -> add
                    for name in self.get_arrays_used_by_node(sdfg, state, node):
                        _add_data(name, gpu_set, cpu_set, is_gpu, host_level)

                elif isinstance(node, (ControlFlowRegion)):
                    g, c = self.get_data_locations_of_cfregion(sdfg, node)
                    if not is_gpu:
                        gpu_set |= g
                        cpu_set |= c
                    else:
                        gpu_set |= g | c

                elif isinstance(node, nodes.LibraryNode):
                    # Launched from the host, reading device memory: the schedule around it says nothing.
                    on_gpu = is_gpu or self.has_GPU_schedule(node)
                    for name in self.get_arrays_used_by_node(sdfg, state, node):
                        _add_data(name, gpu_set, cpu_set, on_gpu, host_level)

                elif isinstance(node, nodes.NestedSDFG):
                    if is_gpu:
                        pass  # inside a kernel everything below is on the device already
                    else:
                        g, c = self.get_data_locations_of_nested_sdfg(sdfg, state, node)
                        gpu_set |= g
                        cpu_set |= c

                elif isinstance(node, nodes.MapExit):
                    pass

                else:
                    raise RuntimeError(f"unhandled node {node.label} of type {type(node).__name__} inside map "
                                       f"{map_entry} in state {state}")

            if is_taskloop:
                # Unclaimed by the body means device data, or every iteration pays for a copy of it.
                gpu_set |= input_and_output - cpu_set

        # function body, calls recursive helper
        gpu_set: set[str] = set()
        cpu_set: set[str] = set()
        _recursive_helper(sdfg, state, map_entry, gpu_set, cpu_set, False)
        return gpu_set, cpu_set

    def get_data_locations_of_nested_sdfg(self, sdfg: SDFG, state: SDFGState,
                                          node: nodes.NestedSDFG) -> tuple[set[str], set[str]]:
        """Where a nested SDFG wants its bound arrays, in the OUTER SDFG's names."""
        # Its hybrid states are resolved when the body is offloaded; wrapping them needs that SDFG.
        outer_hybrid = self.hybrid_states
        self.hybrid_states = set()
        inner_gpu, inner_cpu = self.get_data_locations_of_cfregion(node.sdfg, node.sdfg)
        self.hybrid_states = outer_hybrid

        gpu_set: set[str] = set()
        cpu_set: set[str] = set()
        for edge in state.in_edges(node) + state.out_edges(node):
            if edge.data is None or edge.data.is_empty():
                continue
            connector = edge.dst_conn if edge.dst is node else edge.src_conn
            name = edge.data.data
            if connector is None or name not in sdfg.arrays or not self._is_array(name, sdfg):
                continue
            if connector in inner_gpu:
                gpu_set.add(name)
            elif connector in inner_cpu:
                cpu_set.add(name)
        return gpu_set, cpu_set

    def get_data_locations_of_state(self,
                                    sdfg: SDFG,
                                    state: SDFGState,
                                    recursive_call=False) -> tuple[set[str], set[str]]:
        # iterate through all toplevel nodes of this state
        #  - map entry -> give to get_data_locations_of_map, which handles all nodes inside scope
        #  - control flow (nested) -> recurse
        #  - non-nested toplevel scopes -> add accessed data to cpu set
        gpu_set: set[str] = set()
        cpu_set: set[str] = set()

        top_level_nodes = state.scope_children()[None]
        for node in top_level_nodes:

            g, c = set(), set()

            # process map and all nodes within -> may be on GPU
            if isinstance(node, nodes.MapEntry):
                g, c = self.get_data_locations_of_map(sdfg, state, node)

            elif isinstance(node, nodes.MapExit):
                pass

            # library nodes are usually GPU, can be CPU
            elif isinstance(node, nodes.LibraryNode):
                if self.has_GPU_schedule(node):
                    g = self.get_arrays_used_by_node(sdfg, state, node)
                else:
                    c = self.get_arrays_used_by_node(sdfg, state, node)

            # a nested SDFG at the top of a state is host code holding its own maps
            elif isinstance(node, nodes.NestedSDFG):
                g, c = self.get_data_locations_of_nested_sdfg(sdfg, state, node)

            # recurse if nested
            elif isinstance(node, ControlFlowRegion):
                g, c = self.get_data_locations_of_cfregion(sdfg, node)

            # all else is definitely on CPU
            elif isinstance(node, nodes.Tasklet):  # outside a map scope (else handled by locations_of_map) -> cpu
                c = self.get_arrays_used_by_node(sdfg, state, node)

            elif isinstance(node, nodes.AccessNode):
                pass  # nothing to do; cannot be classified without context

            else:
                raise RuntimeError(f"unhandled node {node} of type {node.__class__.__name__} in state {state}")

            gpu_set |= g
            cpu_set |= c

        # Check for hybrid state configurations, where arrays are accessed on both CPU and GPU
        overlap = gpu_set & cpu_set
        if overlap:
            self.hybrid_states.add(state)
            gpu_set |= cpu_set
            cpu_set = set()

        return gpu_set, cpu_set

    def get_data_locations_of_condblock(self, sdfg: SDFG, block: ConditionalBlock) -> tuple[set[str], set[str]]:
        gpu_set: set[str] = set()
        cpu_set: set[str] = set()

        # get array accesses in condition
        for memlet in block.get_meta_read_memlets():
            if not memlet:
                continue
            data_name = memlet.data
            if memlet.data in sdfg.arrays and self._is_array(data_name, sdfg):
                cpu_set.add(memlet.data)

        # add array accesses in branches
        for _, branch in block.branches:
            g, c = self.get_data_locations_of_cfregion(sdfg, branch)
            gpu_set |= g
            cpu_set |= c

        return gpu_set, cpu_set

    def get_data_locations_of_loop(self, sdfg: SDFG, loop: LoopRegion) -> tuple[set[str], set[str]]:
        # get array accesses in init_statement, update_statement, and loop_condition
        cpu_set: set[str] = set()
        for memlet in loop.get_meta_read_memlets():
            if not memlet:
                continue
            data_name = memlet.data
            if data_name in sdfg.arrays and self._is_array(data_name, sdfg):
                cpu_set.add(data_name)

        # add array accesses in loop body
        gpu_set, c = self.get_data_locations_of_cfregion(sdfg, loop)
        cpu_set |= c

        return gpu_set, cpu_set

    def get_data_locations_of_cfblock(self, sdfg: SDFG, block: ControlFlowBlock) -> tuple[set[str], set[str]]:
        if isinstance(block, SDFGState):
            return self.get_data_locations_of_state(sdfg, block)

        elif isinstance(block, ConditionalBlock):
            return self.get_data_locations_of_condblock(sdfg, block)

        elif isinstance(block, LoopRegion):
            return self.get_data_locations_of_loop(sdfg, block)

        elif isinstance(block, ControlFlowRegion):
            return self.get_data_locations_of_cfregion(sdfg, block)

        elif isinstance(block, (nodes.NestedSDFG, ReturnBlock, ContinueBlock, BreakBlock)):
            return set(), set()  # do nothing

        raise RuntimeError(f"Unknown block type: {block} of type {block.__class__.__name__}")

    def get_data_locations_of_cfregion(self, sdfg: SDFG, cfr: ControlFlowRegion) -> tuple[set[str], set[str]]:
        gpu_set: set[str] = set()
        cpu_set: set[str] = set()

        for block in cfr.bfs_nodes():
            g, c = self.get_data_locations_of_cfblock(sdfg, block)
            gpu_set |= g
            cpu_set |= c

        return gpu_set, cpu_set

    # wrapper
    #def get_data_locations(self, sdfg:SDFG) -> tuple[set[str], set[str]]:
    #    return self.get_data_locations_of_cfregion(sdfg, sdfg)

    ### STEP 3: Intermediate Representation ###
    def is_array_stored_on_GPU(self, sdfg, array_name):
        storage = sdfg.arrays[array_name].storage
        if storage == dtypes.StorageType.GPU_Global or storage in dtypes.GPU_STORAGES:
            return True
        elif storage in {
                dtypes.StorageType.Default, dtypes.StorageType.Register, dtypes.StorageType.CPU_Heap,
                dtypes.StorageType.CPU_Pinned, dtypes.StorageType.CPU_ThreadLocal
        }:
            return False
        else:
            raise NotImplementedError(f"array {array_name!r} lives in {storage}, which this pass does not offload")

    def written_arrays(self, sdfg: SDFG) -> set[str]:
        """Names this SDFG writes: an access node with an incoming edge."""
        written: set[str] = set()
        for state in sdfg.states():
            for node in state.data_nodes():
                if state.in_degree(node) > 0:
                    written.add(node.data)
        return written

    def sdfg_to_IR(self, sdfg: SDFG):

        # remember initial non-transient array locations
        non_transients = {
            name
            for name in sdfg.arrays if not sdfg.arrays[name].transient and not self._is_scalar(name, sdfg)
        }
        initially_on_gpu = set()
        initially_on_cpu = set()

        for array_name in non_transients:
            if self.is_array_stored_on_GPU(sdfg, array_name):
                initially_on_gpu.add(array_name)
            else:
                initially_on_cpu.add(array_name)

        # create inital node (open node)
        IR = OffloadingIRNode.new_open_node(sdfg)
        IR.gpu_set = initially_on_gpu.copy()
        IR.cpu_set = initially_on_cpu.copy()  # no copy -> may cause sideeffects

        # parse entire graph
        end = self._parse_to_IR(sdfg, sdfg, IR)

        # finish graph: tie the final node together with the inital close node
        end.append_node(IR.close)
        # Only what this SDFG WROTE goes back: restoring a read-only input is dead traffic, and
        # inside a nested SDFG it writes an input connector, which is invalid.
        written = self.written_arrays(sdfg)
        IR.close.gpu_set = initially_on_gpu & written
        IR.close.cpu_set = initially_on_cpu & written

        self._propagate_arrays(IR)

        return IR

    def _parse_to_IR(self, sdfg: SDFG, cfr: ControlFlowRegion, curr_node: OffloadingIRNode) -> OffloadingIRNode:
        # NOTE to self: ControlFlowRegion inherits from ControlFlowBlock
        block: ControlFlowBlock
        for block in cfr.bfs_nodes():

            # iterate through all (incoming) interstate edges
            in_edge_arrays = set()
            for edge in cfr.in_edges(block):
                arrays = {
                    data_name
                    for data_name in edge.data.used_arrays(sdfg.arrays) if self._is_array(data_name, sdfg)
                }
                in_edge_arrays |= arrays

            if in_edge_arrays:
                edge_node = OffloadingIRNode.new_edge_node(block, in_edge_arrays)
                curr_node.append_node(edge_node)
                curr_node = edge_node

            # iterate through all nodes
            # non-nested state
            if isinstance(block, SDFGState):
                state: SDFGState = block
                gpu_set, cpu_set = self.get_data_locations_of_state(sdfg,
                                                                    state)  # beating heart of this entire function
                state_node = OffloadingIRNode.new_state_node(state, cpu_set, gpu_set)
                curr_node.append_node(state_node)
                curr_node = state_node

            # do nothing
            elif isinstance(block, (ReturnBlock, ContinueBlock, BreakBlock)):
                pass

            # container node with outer wrapper
            else:
                # outer node
                outer_node = OffloadingIRNode.new_open_node(block)
                curr_node.append_node(outer_node)
                curr_node = outer_node

                # if else
                if isinstance(block, ConditionalBlock):
                    cond_block: ConditionalBlock = block

                    # branch condition
                    meta_data_node: OffloadingIRNode = None
                    meta_data = {
                        memlet.data
                        for memlet in cond_block.get_meta_read_memlets() if memlet.data in sdfg.arrays
                    }
                    if meta_data:
                        meta_data_node = OffloadingIRNode.new_state_node(block, cpu_set=meta_data, gpu_set=set())
                        curr_node.append_node(meta_data_node)
                        curr_node = meta_data_node

                    # parse branches and connect each branch to close node
                    for _, branch in cond_block.branches:
                        branch_end: OffloadingIRNode = self._parse_to_IR(sdfg, branch, curr_node)
                        # TODO: FIND ALL TAILS
                        branch_end.append_node(outer_node.close)

                # loop
                elif isinstance(block, LoopRegion):
                    loop: LoopRegion = block

                    # add meta data node if needed
                    meta_data_node: OffloadingIRNode = None
                    meta_data = {memlet.data for memlet in loop.get_meta_read_memlets() if memlet.data in sdfg.arrays}
                    if meta_data:
                        meta_data_node = OffloadingIRNode.new_state_node(block, cpu_set=meta_data, gpu_set=set())
                        curr_node.append_node(meta_data_node)
                        curr_node = meta_data_node

                    # parse body and connect to loop close node
                    # TODO: FIND ALL TAILS
                    curr_node = self._parse_to_IR(sdfg, loop,
                                                  curr_node)  # linked list representing all internal nodes of loop
                    curr_node.append_node(outer_node.close)

                # nested region -> flatten
                elif isinstance(block, ControlFlowRegion):
                    curr_node = self._parse_to_IR(sdfg, block, curr_node)
                    curr_node.append_node(outer_node.close)

                elif isinstance(block, nodes.NestedSDFG):
                    curr_node = self._parse_to_IR(block.sdfg, block.sdfg, curr_node)
                    curr_node.append_node(outer_node.close)

                else:
                    raise RuntimeError(f"Unknown block type: {block} of type {block.__class__.__name__}")

                # finish container
                self._populate_container_node_sets(outer_node)
                curr_node = outer_node.close

        # TODO: FIND ALL TAILS?
        return curr_node

    def __traverse_IR(self, IR: OffloadingIRNode, method):

        def recursion(node, visited_set):
            if node in visited_set:
                return
            visited_set.add(node)

            method(node)

            for next in node.next:
                recursion(next, visited_set)

        return recursion(IR, set())

    def __traverse_same_level(self, IR: OffloadingIRNode, method):  #DFS
        queue = IR.next.copy()
        while queue:
            curr = queue.pop()
            if curr.type == OffloadingIRNode.STATE or curr.type == OffloadingIRNode.EDGE:  # data node
                method(curr)
                queue += curr.next

            elif curr.is_open_node():
                method(curr)
                queue += curr.close.next

            elif curr.type == OffloadingIRNode.CLOSE:
                break

            else:
                raise ValueError(f'unhandled IR node type {OffloadingIRNode.get_type_as_str(curr.type)}')

    def _populate_container_node_sets(self, IR: OffloadingIRNode):
        self.__populate_open_node_sets(IR)
        self.__populate_close_node_sets(IR)
        # TODO for both: deal with FIND ALL TAILS

    def __populate_open_node_sets(self, IR: OffloadingIRNode):
        assert IR.is_open_node(), str(IR)

        # Behavior 1:
        # if there are no or multiple direct children, leave the sets empty & simply propagate later
        # there is no good heuristic to choose from here, which copies to make and which not
        # (not without significantly more analysis)
        children = IR.next
        if len(children) != 1:
            return

        # Behavior 2:
        # if there is a single direct child, then analyse the section & find first known location of each used array
        # if the graph splits later, the first of all possible paths is chosen for analysis
        # this can lead to unnecessary copies in the other paths
        location_on_gpu = {}

        def gather_data(node: OffloadingIRNode):
            if isinstance(node.block, nodes.NestedSDFG
                          ):  # Nested SDFGs do not share namespace, array names should not leak to outer scope
                return

            for array_name in node.gpu_set:
                if array_name not in location_on_gpu:
                    location_on_gpu[array_name] = True

            for array_name in node.cpu_set:
                if array_name not in location_on_gpu:
                    location_on_gpu[array_name] = False

        # traverse graph
        self.__traverse_same_level(IR, gather_data)

        # populate IR sets
        IR.gpu_set = {array_name for array_name in location_on_gpu if location_on_gpu[array_name]}
        IR.cpu_set = {array_name for array_name in location_on_gpu if not location_on_gpu[array_name]}

    def __populate_close_node_sets(self, IR: OffloadingIRNode):
        assert IR.is_open_node(), str(IR)

        tails = IR.get_all_tails()
        assert tails, f"{IR.debug_name} doesn't have any tails! {IR}"

        # Behavior 1:
        # if there is a single tail node (node that leads to this section's close node),
        # then analyse the section & find last known location of each used array
        if len(tails) == 1:
            # define data gathering function
            location_on_gpu = {}

            def gather_data(node: OffloadingIRNode):
                if isinstance(node.block, nodes.NestedSDFG
                              ):  # Nested SDFGs do not share namespace, array names should not leak to outer scope
                    return

                for array_name in node.gpu_set:
                    location_on_gpu[array_name] = True

                for array_name in node.cpu_set:
                    location_on_gpu[array_name] = False

            # traverse graph
            self.__traverse_same_level(IR, gather_data)

            # populate IR sets
            IR.close.gpu_set = {array_name for array_name in location_on_gpu if location_on_gpu[array_name]}
            IR.close.cpu_set = {array_name for array_name in location_on_gpu if not location_on_gpu[array_name]}

        # Behaviour 2:
        # if there are multiple tail nodes, then mark this node for later.
        # In a second pass, it will assume the gpu&cpu set of its next successor.
        # This means that each branch will have to insert copies individually, usually leading to the least amount of
        # necessary copies.
        # There is however a risk that this introduces copies within a loop unnecessarily.
        else:
            IR.copy_successor = True

    def _propagate_arrays(self, IR: OffloadingIRNode):
        # all arrays which aren't used by this state retain their previous status
        # ASSUMPTION: arrays are either gpu or cpu within a state
        def propagate(node):
            for next in node.next:
                next_arrays = next.cpu_set | next.gpu_set

                for array in node.cpu_set:
                    if array not in next_arrays:
                        next.cpu_set.add(array)
                for array in node.gpu_set:
                    if array not in next_arrays:
                        next.gpu_set.add(array)

        self.__traverse_IR(IR, propagate)

    def _insert_copy_names_in_block(self, sdfg: SDFG, block: ControlFlowBlock, rename_dict: dict):
        if block is None:
            return

        cfr = block.parent_graph
        if cfr and isinstance(cfr, AbstractControlFlowRegion):
            for edge in cfr.in_edges(block):
                relevant_edge_arrays = edge.data.used_arrays(rename_dict)
                # Renaming is decided by where the array BEGINS, not by where this edge reads
                # it: one that begins on the GPU is read through its host copy under the host
                # name, one that begins on the CPU keeps its own name. The copies themselves
                # are inserted later.
                for name in relevant_edge_arrays:
                    if sdfg.arrays[name].storage == dtypes.StorageType.GPU_Global:
                        edge.data.replace(name, self._get_host_name(name))

        if isinstance(block, SDFGState):
            self._insert_copy_names_in_state(block, rename_dict)

        elif isinstance(block, ControlFlowBlock):
            # rename meta accesses (control-flow metadata like loop bounds or conditions)
            block.replace_meta_accesses(rename_dict)
            # NOTE: states / blocks within the current block all have their own IRNodes and don't need to be handled
            # recursively here
        else:
            raise NotImplementedError(
                f"in _correct_names_in_block: IR.block unhandled type: {block} is {block.__class__.__name__}")

    def _insert_copy_names_in_state(self, state: SDFGState, rename_dict: dict):
        # rename access nodes
        for access in state.data_nodes():
            if access.data in rename_dict:
                access.data = rename_dict[access.data]

        # rename edge conditions
        for edge in state.edges():
            for e in state.memlet_tree(edge):
                memlet = e.data
                if memlet is not None and not memlet.is_empty() and memlet.data in rename_dict:
                    memlet.data = rename_dict[memlet.data]

    def _insert_copy_names(self, sdfg: SDFG, IR: OffloadingIRNode):
        # make a rename dict for each IR node, then rename all such arrays in the IR.block
        def _insert_copy_names_in_node(node: OffloadingIRNode):
            rename_dict = {}
            for name in node.gpu_set:
                assert name in sdfg.arrays
                if sdfg.arrays[name].storage == dtypes.StorageType.Default:  # starts on CPU, but this access is on GPU
                    rename_dict[name] = self._get_gpu_name(name)

            for name in node.cpu_set:
                assert name in sdfg.arrays
                if sdfg.arrays[
                        name].storage == dtypes.StorageType.GPU_Global:  # starts on GPU, but this access is on CPU
                    rename_dict[name] = self._get_host_name(name)

            self._insert_copy_names_in_block(sdfg, node.block, rename_dict)

        self.__traverse_IR(IR, _insert_copy_names_in_node)

    def _correct_transient_storage_locations(self, sdfg: SDFG, IR: OffloadingIRNode):
        seen_transients = set()

        def _correct_transients(node: OffloadingIRNode):
            for name in node.gpu_set:
                assert name in sdfg.arrays
                desc = sdfg.arrays[name]
                if desc.transient and name not in seen_transients:
                    desc.storage = dtypes.StorageType.GPU_Global
                    seen_transients.add(name)

            for name in node.cpu_set:
                assert name in sdfg.arrays
                desc = sdfg.arrays[name]
                if desc.transient and name not in seen_transients:
                    desc.storage = dtypes.StorageType.Default
                    seen_transients.add(name)

        self.__traverse_IR(IR, _correct_transients)

    def eval_IR(self, sdfg, IR: OffloadingIRNode):
        # modifies SDFG in place & inserts all necessary copies
        def insert_copies(node, next, node_block, next_block):
            gpu_copies = node.cpu_set & next.gpu_set
            if gpu_copies:
                self.create_interstate_copy(sdfg, node_block, next_block, gpu_copies, to_gpu=True)

            cpu_copies = node.gpu_set & next.cpu_set
            if cpu_copies:
                self.create_interstate_copy(sdfg, node_block, next_block, cpu_copies, to_gpu=False)

        def eval(node: OffloadingIRNode):
            for next in node.next:

                if node.cpu_set & node.gpu_set:
                    raise NotImplementedError(
                        f"state {node.debug_name} uses {node.cpu_set & node.gpu_set} on both the CPU and "
                        f"the GPU; this pass cannot place a copy inside a single state")

                # edge case: if this condition is true, both blocks are None, can't insert
                if node.type == OffloadingIRNode.CLOSE and next.type == OffloadingIRNode.CLOSE:
                    insert_copies(node, next, node.open.block, None)

                elif next.type == OffloadingIRNode.EDGE:  # then I want the copy AFTER the node, not before
                    insert_copies(node, next, node.block, None)

                else:  # the usual: copies between node -> next
                    insert_copies(node, next, node.block, next.block)

            # loop copies if applicable
            if node.type == OffloadingIRNode.OPEN_LOOP:
                top = node  # INV: top.type == OffloadingIRNode.OPEN_LOOP
                bottom = node.close  # INV: bottom.type == OffloadingIRNode.CLOSE
                tails = OffloadingIRNode.get_all_tails(top)  # INV: all are STATE or CLOSE if there's a nested loop

                gpu_copies = bottom.cpu_set & top.gpu_set

                if gpu_copies:
                    for tail in tails:
                        if tail.type == OffloadingIRNode.CLOSE:  # and bottom.type == OffloadingIRNode.CLOSE:
                            self.create_interstate_copy(sdfg, tail.open.block, None, gpu_copies, to_gpu=True)
                        else:
                            self.create_interstate_copy(sdfg, tail.block, None, gpu_copies, to_gpu=True)

                cpu_copies = bottom.gpu_set & top.cpu_set
                if cpu_copies:
                    for tail in tails:
                        if tail.type == OffloadingIRNode.CLOSE:
                            self.create_interstate_copy(sdfg, tail.open.block, None, cpu_copies, to_gpu=False)
                        else:
                            self.create_interstate_copy(sdfg, tail.block, None, cpu_copies, to_gpu=False)

        self._correct_transient_storage_locations(sdfg, IR)
        self._insert_copy_names(sdfg, IR)
        self.__traverse_IR(IR, eval)

    ### Step 4: Copy Insertion ###
    # create ONE copy state for all arrays in array_names

    def create_interstate_copy(self, sdfg, state1, state2, array_names, to_gpu: bool):
        assert state1 is not None or state2 is not None, "invalid: both states are None"

        # 1) insert new state
        copy_state: SDFGState
        joined = '_'.join(sorted(array_names))
        direction = 'to_gpu' if to_gpu else 'to_host'
        label = f"copy_{joined}_{direction}"

        if state2 is not None:
            target_graph = state2.parent_graph
            assert target_graph is not None, "copy insertion requires a parent control-flow graph (s2)"

            copy_state = target_graph.add_state_before(state2, label=label)
            if state2 is target_graph.start_block:
                target_graph.start_block = target_graph.node_id(copy_state)  # copy state becomes new start block

        elif state1 is not None:
            target_graph = state1.parent_graph if state1.parent_graph else state1
            assert target_graph is not None, "copy insertion requires a parent control-flow graph (s1)"

            #copy_state = self.add_state_after(target_graph, state1, label)
            copy_state = target_graph.add_state_after(state1, label=label)

        # 2) create the copy map with correct names
        copy_map = {}
        name: str
        for name in array_names:
            assert name in sdfg.arrays

            if self.is_array_stored_on_GPU(sdfg, name):  # original array is on GPU
                if not to_gpu:  # copy goes to CPU: A -> A_host
                    copy_map[name] = self._get_host_name(name)

                else:  # copy goes to GPU: A_host -> A
                    copy_map[self._get_host_name(name)] = name

            else:  # original array is on CPU
                if to_gpu:  # copy goes to GPU: A -> A_gpu
                    copy_map[name] = self._get_gpu_name(name)

                else:  # copy goes to CPU: A_gpu -> A
                    copy_map[self._get_gpu_name(name)] = name

        # 3) build all the copies inside the new state
        for old_name, new_name in copy_map.items():

            # a) if first copy of this array: register new copy array with sdfg
            if new_name not in sdfg.arrays:
                self._register_new_copy_transient(sdfg, new_name, old_name)
            elif old_name not in sdfg.arrays:
                self._register_new_copy_transient(
                    sdfg, old_name, new_name
                    # in some cases, e.g. loops, a copy-from can be registered before its copy-to, leading to an unknown
                    # "old_name"
                )

            # b) add (Access Node -> Access Node) to state
            copy_in = copy_state.add_access(old_name)
            copy_out = copy_state.add_access(new_name)

            src_desc = sdfg.arrays[old_name]
            dst_desc = sdfg.arrays[new_name]
            src_subset = subsets.Range.from_array(src_desc)
            dst_subset = subsets.Range.from_array(dst_desc)

            copy_memlet = Memlet(
                data=old_name,
                subset=src_subset,
                other_subset=dst_subset,
            )

            copy_state.add_edge(copy_in, None, copy_out, None, copy_memlet)

    def _register_new_copy_transient(self, sdfg: SDFG, unknown_name: str, known_name: str):
        assert known_name in sdfg.arrays
        desc = sdfg.arrays[known_name]

        new_storage = dtypes.StorageType.Default if self.is_array_stored_on_GPU(
            sdfg, known_name) else dtypes.StorageType.GPU_Global
        if isinstance(desc, data.View):
            sdfg.add_view(unknown_name, desc.shape, desc.dtype, storage=new_storage)
        else:
            sdfg.add_array(unknown_name, desc.shape, desc.dtype, storage=new_storage, transient=True)

    def _get_host_name(self, name: str) -> str:
        """Host-side copy of ``name``.

        ``__return`` is special-cased because it names the SDFG's return slot: a copy of it is a
        buffer, not another return value.
        """
        if name.startswith("__return"):
            return f"buffer__return{name[8:]}_host"
        return f"{name}_host"

    def _get_gpu_name(self, name: str) -> str:
        """Device-side copy of ``name``; see :meth:`_get_host_name` for ``__return``."""
        if name.startswith("__return"):
            return f"buffer__return{name[8:]}_gpu"
        return f"{name}_gpu"

##### OPTIMIZATION #####

# heuristic: size1 maps are faster than more CPU-GPU copies

    from collections import deque

    def _get_root_nodes(self, state: SDFGState, bounded_set: set):
        return {node for node in bounded_set if state.in_degree(node) == 0}

    def _get_leaf_nodes(self, state: SDFGState, bounded_set: set):
        return {node for node in bounded_set if state.out_degree(node) == 0}

    def _get_boundary_in_edges(self, state: SDFGState, node, bounded_set: set):
        # A list, not a set: the connector numbering below follows this order.
        return [e for e in state.in_edges(node) if e.src not in bounded_set]

    def _get_boundary_out_edges(self, state: SDFGState, node, bounded_set: set):
        return [e for e in state.out_edges(node) if e.dst not in bounded_set]

    def _get_entry_nodes(self, state: SDFGState, bounded_set: set):
        return {node for node in bounded_set if all(e.src not in bounded_set for e in state.in_edges(node))}

    def _get_exit_nodes(self, state: SDFGState, bounded_set: set):
        return {node for node in bounded_set if all(e.dst not in bounded_set for e in state.out_edges(node))}

    def _wrap_region_in_size1_map(self, state: SDFGState, region_nodes: set) -> tuple[nodes.MapEntry, nodes.MapExit]:
        if not region_nodes:
            return
        map_label, map_param = self._get_new_map_identifiers(state, "size1_wrap_region", "__wrap_i")
        map_entry, map_exit = state.add_map(name=map_label,
                                            ndrange={map_param: '0:1'},
                                            schedule=dtypes.ScheduleType.GPU_Device)

        # MAP ENTRY
        boundary_in_edges = []
        for node in region_nodes:
            boundary_in_edges += self._get_boundary_in_edges(state, node, region_nodes)

        if not boundary_in_edges:  # if there are no boundary in edges, add new dependcy edges
            root_nodes = self._get_root_nodes(state, region_nodes)
            assert root_nodes, f"region: {region_nodes}"
            for root in root_nodes:
                state.add_nedge(map_entry, root, Memlet())

        else:  # if there are, rewire them through the map
            idx = 0
            for edge in boundary_in_edges:
                src, src_conn, dst, dst_conn = edge.src, edge.src_conn, edge.dst, edge.dst_conn
                ext_memlet = deepcopy(edge.data)
                int_memlet = deepcopy(edge.data)
                state.remove_edge(edge)

                # An empty memlet ORDERS; it carries no data and so may not carry a connector.
                if ext_memlet.is_empty():
                    state.add_nedge(src, map_entry, ext_memlet)
                    state.add_nedge(map_entry, dst, int_memlet)
                    continue

                in_conn = f"IN_REGION_IN_{idx}"
                out_conn = f"OUT_REGION_IN_{idx}"
                map_entry.add_in_connector(in_conn)
                map_entry.add_out_connector(out_conn)
                idx += 1

                state.add_edge(src, src_conn, map_entry, in_conn, ext_memlet)
                state.add_edge(map_entry, out_conn, dst, dst_conn, int_memlet)

        # MAP EXIT
        boundary_out_edges = []
        for node in region_nodes:
            boundary_out_edges += self._get_boundary_out_edges(state, node, region_nodes)

        if not boundary_out_edges:  # add new dependency edges
            leaf_nodes = self._get_leaf_nodes(state, region_nodes)
            assert leaf_nodes
            for leaf in leaf_nodes:
                state.add_nedge(leaf, map_exit, Memlet())

        else:  # rewire out edges
            idx = 0
            for edge in boundary_out_edges:
                src, src_conn, dst, dst_conn = edge.src, edge.src_conn, edge.dst, edge.dst_conn
                int_memlet = deepcopy(edge.data)
                ext_memlet = deepcopy(edge.data)
                state.remove_edge(edge)

                if int_memlet.is_empty():  # ordering edge, see above
                    state.add_nedge(src, map_exit, int_memlet)
                    state.add_nedge(map_exit, dst, ext_memlet)
                    continue

                in_conn = f"IN_REGION_OUT_{idx}"
                out_conn = f"OUT_REGION_OUT_{idx}"
                map_exit.add_in_connector(in_conn)
                map_exit.add_out_connector(out_conn)
                idx += 1

                state.add_edge(src, src_conn, map_exit, in_conn, int_memlet)
                state.add_edge(map_exit, out_conn, dst, dst_conn, ext_memlet)

        return map_entry, map_exit

    def _get_new_map_identifiers(self, state: SDFGState, map_label: str, map_param: str):
        existing_labels = {getattr(node, "label", None) for node in state.nodes()}
        existing_params = set()
        for node in state.nodes():
            if isinstance(node, nodes.MapEntry):
                existing_params |= set(node.map.params)

        suffix = 0
        new_label = map_label
        while new_label in existing_labels:
            suffix += 1
            new_label = f"{map_label}_{suffix}"

        suffix = 0
        new_param = map_param
        while new_param in existing_params:
            suffix += 1
            new_param = f"{map_param}_{suffix}"

        return new_label, new_param

    def _subgraphs_after_removing_partition_nodes(self, state: SDFGState,
                                                  partition_nodes: set) -> list[set[nodes.Node]]:
        """
        Returns connected components (as sets of nodes) after removing partition_nodes
        from a SINGLE SDFG state graph.

        Connectivity is treated as undirected (uses both in/out edges).
        """
        visited = set()
        components = []
        remaining_nodes = [n for n in state.scope_children()[None] if n not in partition_nodes]  # top level nodes only

        for start in remaining_nodes:
            if start in visited:
                continue

            comp = set()
            queue = self.deque([start])
            visited.add(start)

            while queue:
                u = queue.popleft()
                comp.add(u)

                neighbors = {e.dst for e in state.out_edges(u)} | {e.src for e in state.in_edges(u)}
                for v in neighbors:
                    if v in partition_nodes or v in visited:
                        continue
                    visited.add(v)
                    queue.append(v)

            components.append(comp)

        return components

    def _remove_all_outer_access_nodes_from_group(self, state: SDFGState, group: set):
        outer_nodes = self._get_entry_nodes(state, group) | self._get_exit_nodes(state, group)
        nodes_to_remove = {node for node in outer_nodes if isinstance(node, nodes.AccessNode)}

        while nodes_to_remove:
            group -= nodes_to_remove
            outer_nodes = self._get_entry_nodes(state, group) | self._get_exit_nodes(state, group)
            nodes_to_remove = {node for node in outer_nodes if isinstance(node, nodes.AccessNode)}

    def _insert_access_between_adjacent_maps(self, state: SDFGState, map_exit: nodes.MapExit) -> None:
        # avoid illegal direct map-to-map connections by routing through an access node.
        for edge in list(state.out_edges(map_exit)):
            if not isinstance(edge.dst, nodes.MapEntry):
                continue
            if edge.data is None or edge.data.is_empty() or edge.data.data is None:
                continue

            src, src_conn, dst, dst_conn = edge.src, edge.src_conn, edge.dst, edge.dst_conn
            access = state.add_access(edge.data.data)
            out_memlet = deepcopy(edge.data)
            in_memlet = deepcopy(edge.data)

            state.remove_edge(edge)
            state.add_edge(src, src_conn, access, None, out_memlet)
            state.add_edge(access, None, dst, dst_conn, in_memlet)

    def _find_last_access_nodes_in_map_bfs(self, state: SDFGState, map_entry: nodes.MapEntry, map_exit: nodes.MapExit,
                                           data_names: set[str]) -> dict[str, nodes.AccessNode]:
        if not data_names:
            return {}
        last_access: dict[str, nodes.AccessNode] = {}
        queue = self.deque([map_entry])
        visited = {map_entry}

        while queue:
            node = queue.popleft()

            if isinstance(node, nodes.AccessNode) and node.data in data_names:
                last_access[node.data] = node

            if node is map_exit:
                continue

            for edge in state.out_edges(node):
                child = edge.dst
                if not child or child in visited:
                    continue
                visited.add(child)
                queue.append(child)

        return last_access

    def _forward_input_only_map_data(self, state: SDFGState, map_entry: nodes.MapEntry,
                                     map_exit: nodes.MapExit) -> None:
        # For map inputs that are not map outputs, route final in-map access through map_exit
        # -> Ensure all map inputs are also outputs to avoid dace erroneusly labeling them as constants

        # get inputs & isolate those without corresponding outputs
        input_memlets = [
            edge.data for edge in state.in_edges(map_entry)
            if edge.data is not None and not edge.data.is_empty() and edge.data.data is not None
        ]
        input_only_data = [
            memlet.data for memlet in input_memlets
            if all(edge.data is None or edge.data.is_empty() or edge.data.data != memlet.data
                   for edge in state.out_edges(map_exit))
        ]
        # find last accesses(ignore data without accesses)
        # INV: dictionary holds ONLY data which goes into the map, is accessed within but does not exit -> if left
        # unchanged this would be detected as a constant and lead to errors
        last_accesses: dict = self._find_last_access_nodes_in_map_bfs(state, map_entry, map_exit, input_only_data)

        # wire the last access through map_exit to a new outside access node
        for input_memlet in input_memlets:
            data_name = input_memlet.data
            if data_name not in last_accesses:
                continue
            last_access = last_accesses[data_name]

            # create unique connectors
            connector_index = 0
            while (f"IN_INPUT_ONLY_{connector_index}" in map_exit.in_connectors
                   or f"OUT_INPUT_ONLY_{connector_index}" in map_exit.out_connectors):
                connector_index += 1
            in_conn = f"IN_INPUT_ONLY_{connector_index}"
            out_conn = f"OUT_INPUT_ONLY_{connector_index}"

            # add new external access node & edges to it
            map_exit.add_in_connector(in_conn)
            map_exit.add_out_connector(out_conn)
            outside_access = state.add_access(data_name)
            internal_memlet = deepcopy(input_memlet)
            external_memlet = deepcopy(input_memlet)
            state.add_edge(last_access, None, map_exit, in_conn, internal_memlet)
            state.add_edge(map_exit, out_conn, outside_access, None, external_memlet)

    def _store_output_scalars_on_GPU(self, sdfg: SDFG, state: SDFGState, map_exit: nodes.MapExit) -> None:
        for edge in state.out_edges(map_exit):
            if not edge or not isinstance(edge.dst, nodes.AccessNode):
                continue
            access = edge.dst
            if access.data and self._is_scalar(access.data, sdfg):
                sdfg.arrays[access.data].storage = dtypes.StorageType.GPU_Global

    def make_size1_map_wrappers(self, sdfg: SDFG, state: SDFGState):
        # top level GPU nodes partition the graph

        lib_nodes = {
            node
            for node in state.scope_children()[None]
            if isinstance(node, (nodes.LibraryNode)) and self.has_GPU_schedule(node)
        }
        map_entries = {
            node
            for node in state.scope_children()[None]
            if isinstance(node, (nodes.MapEntry)) and self.has_GPU_schedule(node)
        }
        map_exits = {state.exit_node(node) for node in map_entries}
        partition_nodes = lib_nodes | map_entries | map_exits

        partitions = self._subgraphs_after_removing_partition_nodes(state, partition_nodes)
        new_maps = set()

        # each partition is wrapped into a map
        ctr = 0
        for partition in partitions:

            # if only scalars are accessed, then no wrap is needed
            array_access = False
            for node in partition:
                if isinstance(node, nodes.AccessNode) and node.data and not self._is_scalar(node.data, sdfg):
                    array_access = True
                    break
            if not array_access:
                continue

            # reduce partition to nodes which need to go into wrap
            self._remove_all_outer_access_nodes_from_group(state, partition)
            ctr += 1

            # if anything is left, wrap it
            if partition:
                map_entry, map_exit = self._wrap_region_in_size1_map(state, partition)
                new_maps.add((map_entry, map_exit))

                # Avoid illegal direct map-to-map connections by routing through an access node.
                self._insert_access_between_adjacent_maps(state, map_exit)

                # Ensure all map inputs are also outputs to avoid dace erroneusly labeling them as constants
                self._forward_input_only_map_data(state, map_entry, map_exit)

                # Avoid illegal copy from gpu_device scheduled map to (by default) cpu_heap stored scalar: make all
                # write scalars gpu_global
                self._store_output_scalars_on_GPU(sdfg, state, map_exit)

        return new_maps


################################################################
## Fix Point Iteration Over Lattice                           ##
# A GPU-scheduled map that writes a variable needs it to be a len-1 ARRAY: a scalar is
# passed by value, so the written value is lost. The rule propagates -- if any input or
# output of a tasklet is GPU-written, every output of that tasklet can be too -- so the
# answer is the fixpoint, compared against the current scalars / len-1 arrays with the
# mismatches converted.

    def _is_scalar(self, data_name: str, sdfg: SDFG):
        assert data_name in sdfg.arrays
        desc = sdfg.arrays[data_name]
        return isinstance(desc, data.Scalar)

    def _is_array(self, data_name: str, sdfg: SDFG):
        assert data_name in sdfg.arrays
        desc = sdfg.arrays[data_name]
        return isinstance(desc, data.Array)

    def _is_view(self, data_name: str, sdfg: SDFG):
        assert data_name in sdfg.arrays
        desc = sdfg.arrays[data_name]
        return isinstance(desc, data.View)

    def _is_length1_array(self, data_name: str, sdfg: SDFG):
        assert data_name in sdfg.arrays
        desc = sdfg.arrays[data_name]
        return isinstance(desc, data.Array) and len(desc.shape) == 1 and desc.shape[0] == 1

    def decide_length1_array_or_scalar_FPI(self, sdfg: SDFG):
        # 1)
        all_scalars: set[str] = {data_name for data_name in sdfg.arrays if self._is_scalar(data_name, sdfg)}
        all_len1arrays: set[str] = {data_name for data_name in sdfg.arrays if self._is_length1_array(data_name, sdfg)}
        vars: set[str] = all_scalars | all_len1arrays

        # 2) with current scheduling heuristic, only toplevel can be GPU
        # 3) tasklets within nested sdfgs are not relevant
        gpu_written: set[str] = set()
        tasklet_dict: dict = {
            # maps tasklet to (inputs, outputs) where both are sets of data names (array & scalar) accessed as
            # input/output
        }
        for state in sdfg.states():
            for node in state.nodes():

                if isinstance(node, (nodes.MapExit, nodes.LibraryNode)) and self.has_GPU_schedule(node):
                    outputs = self.get_data_used_by_outgoing_access_nodes(sdfg, state, node, include_scalars=True)
                    gpu_written |= outputs & vars

                elif isinstance(node, nodes.Tasklet):
                    inputs = self.get_data_used_by_incoming_access_nodes(sdfg, state, node, include_scalars=True)
                    outputs = self.get_data_used_by_outgoing_access_nodes(sdfg, state, node, include_scalars=True)
                    tasklet_dict[node] = (inputs, outputs)

        # 4)

        if gpu_written:
            new_gpu_written = gpu_written.copy()

            while True:
                for inputs, outputs in tasklet_dict.values():

                    # at least one in- or output var is written to by gpu
                    if inputs & gpu_written or outputs & gpu_written:
                        new_gpu_written |= outputs  # add all outputs as being potentially written to by gpu

                if new_gpu_written == gpu_written:  # fixpoint reached
                    break
                gpu_written = new_gpu_written.copy()

        # 5)
        to_len1_arrays = all_scalars & gpu_written
        to_scalars = all_len1arrays - gpu_written
        to_scalars = {name
                      for name in to_scalars if not name.startswith("__return")
                      }  # is usually very inefficient because __return if mostly used at the end of the graph

        if to_len1_arrays:
            ConvertScalarsToLengthOneArrays(
                recursive=True,
                preserve_abi=True,
                filter=to_len1_arrays,
            ).apply_pass(sdfg, {})

        if to_scalars:
            ConvertLengthOneArraysToScalars(
                recursive=True,
                preserve_abi=True,
                filter=to_scalars,
            ).apply_pass(sdfg, {})
