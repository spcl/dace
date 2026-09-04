# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Decide what runs on the accelerator, and place the host/device copies that follow from it.

The pass reads the SDFG once into a small control-flow IR (:class:`OffloadingIRNode`), records per
block which arrays are wanted on the CPU and which on the GPU, propagates those sets along the
graph, and only then materializes copies -- so a copy is emitted where the location actually
changes rather than around every kernel.
"""
from copy import deepcopy
from typing import Any, Optional

from dace.ordered import OrderedSet

from dace import dtypes, properties, data, Memlet, subsets
from dace.config import Config
from dace.libraries.standard.helper import GPU_RESIDENT_STORAGES
from dace.sdfg import nodes, SDFG, InterstateEdge
from dace.sdfg.state import (SDFGState, ConditionalBlock, ControlFlowRegion, LoopRegion, ReturnBlock, ContinueBlock,
                             BreakBlock, ControlFlowBlock, AbstractControlFlowRegion)
from dace.sdfg.scope import is_devicelevel_gpu
from dace.sdfg.utils import get_last_view_node
from dace.transformation import pass_pipeline as ppl
from dace.transformation.transformation import explicit_cf_compatible
from dace.transformation.dataflow import TrivialMapElimination
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
        self.cpu_set: OrderedSet[str] = cpu_set
        self.gpu_set: OrderedSet[str] = gpu_set
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
        return self._get_str(OrderedSet(), -4)

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
        close = OffloadingIRNode(OffloadingIRNode.CLOSE, None, OrderedSet(), OrderedSet(), [], None)
        close.debug_name = f"_close_{block.label}"

        type: int
        if isinstance(block, LoopRegion):
            type = OffloadingIRNode.OPEN_LOOP
        elif isinstance(block, ConditionalBlock):
            type = OffloadingIRNode.OPEN_COND
        else:
            type = OffloadingIRNode.OPEN

        open = OffloadingIRNode(type, block, OrderedSet(), OrderedSet(), [], close)
        open.debug_name = f"_{OffloadingIRNode.get_type_as_str(type)}_{block.label}"
        close.open = open

        return open

    def new_state_node(block: ControlFlowBlock, cpu_set: set, gpu_set: set):
        state = OffloadingIRNode(OffloadingIRNode.STATE, block, cpu_set, gpu_set, [], None)
        state.debug_name = f"_state_{block.label}"
        return state

    def new_edge_node(edge: InterstateEdge, cpu_set: set):
        edge_node = OffloadingIRNode(OffloadingIRNode.EDGE, edge, cpu_set, OrderedSet(), [], None)
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


def in_sequential_specialization_arm(block) -> bool:
    """Whether ``block`` is inside the sequential arm of a guarded specialization.

    Canonicalization emits a loop it can only parallelize under a runtime condition as both arms of
    one ConditionalBlock -- a Map for the parallel case, the original LoopRegion for the fallback.
    That fallback IS host code, and it owns the copies that bring its inputs down; lifting its
    tasklets into size-1 kernels would delete the copies and defeat the arm. The parallel arm has no
    such loop, so a hybrid there is resolved the usual way.

    A sequential loop that is not a specialization arm (npbench nbody's ``for_48``) is ordinary host
    code around device work and is NOT this: the LoopRegion has to be reached before a
    ConditionalBlock for the block to be a fallback.
    """
    seen_loop = False
    current = getattr(block, 'parent_graph', None)
    while current is not None:
        if isinstance(current, LoopRegion):
            seen_loop = True
        elif isinstance(current, ConditionalBlock):
            return seen_loop
        current = getattr(current, 'parent_graph', None)
    return False


@properties.make_properties
@explicit_cf_compatible
class OffloadToAccelerator(ppl.Pass):
    """Move the work an accelerator can take to it, and insert the copies that decision implies.

    The only offloader on this branch, and what ``SDFG.apply_gpu_transformations`` runs. Where the
    transformation it replaced copied in and out around each kernel it made, this one propagates the
    wanted location of every array through the control flow first, so an array that stays on the
    device across a whole loop is copied once rather than per iteration, and a host-only branch pays
    for its copies inside that branch.
    """

    CATEGORY: str = 'Offload To Accelerator'

    taskloop_overrides = properties.DictProperty(
        key_type=str,
        value_type=bool,
        default={},
        desc='Map label -> whether that map is a taskloop, deciding it outright. Final in both '
        'directions and consulted before any heuristic: a caller naming a map has looked at the '
        'kernel, and the rules have not. Maps left unnamed are classified as usual.')

    def __init__(self, taskloop_overrides: dict[str, bool] | None = None):
        self.taskloop_overrides = dict(taskloop_overrides) if taskloop_overrides else {}

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    #def depends_on(self) -> OrderedSet[Union[Type['Pass'], 'Pass']]:
    #    return OrderedSet()

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
        self.hybrid_overlap: dict = {}

        # step 1: set schedule of maps and library nodes -> heuristic only!
        self.find_taskloops(sdfg)
        self.assign_schedules(sdfg)

        self.place_single_sided_data(sdfg)
        self.place_and_copy(sdfg)
        self.offload_host_level_bodies(sdfg)
        self.scalarize_locals_of_removed_trivial_maps(sdfg)
        self.refuse_by_value_scalars_the_device_writes(sdfg)

    def kernel_local_len1_arrays(self, sdfg: SDFG) -> OrderedSet[str]:
        """Length-1 transients written inside a TRIVIAL SEQUENTIAL map that a kernel encloses.

        That map is a loop of one iteration inside device code, and the array is the stack slot its
        body writes through. Both halves of the guard are load-bearing. A map with a real extent
        needs the array, because the iterations are distinct writes. A trivial map at a HOST level,
        or one carrying a GPU schedule, is not device-local at all: the schedule is what makes the
        write a register rather than a kernel argument, and a scalar handed to a kernel goes by value
        and loses the write -- which is why :meth:`place_and_copy` promoted these to arrays to begin
        with.
        """
        found: OrderedSet[str] = OrderedSet()
        for nested in sdfg.all_sdfgs_recursive():
            for state in nested.states():
                scopes = state.scope_dict()
                for entry in state.nodes():
                    if not isinstance(entry, nodes.MapEntry) or entry.map.schedule in dtypes.GPU_SCHEDULES:
                        continue
                    if not all(begin == end for begin, end, _ in entry.map.range):
                        continue
                    if not self.enclosing_kernel(scopes, entry):
                        continue
                    for node in state.scope_subgraph(entry).nodes():
                        if not isinstance(node, nodes.AccessNode) or node.data not in nested.arrays:
                            continue
                        desc = nested.arrays[node.data]
                        if (desc.transient and self._is_length1_array(node.data, nested)
                                and desc.storage not in GPU_RESIDENT_STORAGES):
                            found.add(node.data)
        return found

    def enclosing_kernel(self, scopes: dict, node: nodes.Node) -> Optional[nodes.MapEntry]:
        """The nearest enclosing map with a GPU schedule, or None outside every kernel."""
        scope = scopes[node]
        while scope is not None:
            if isinstance(scope, nodes.MapEntry) and scope.map.schedule in dtypes.GPU_SCHEDULES:
                return scope
            scope = scopes[scope]
        return None

    def data_written_by_device_code(self, sdfg: SDFG) -> OrderedSet[str]:
        """Every descriptor a GPU-scheduled scope writes whose value has to OUTLIVE that scope.

        A kernel writes across TWO boundaries and the analysis used to see only the first:

        * an access node OUTSIDE the scope, reached through the ``MapExit`` -- what
          :meth:`get_data_used_by_outgoing_access_nodes` reports;
        * an access node INSIDE the scope. A size-1 wrapper that pulls a tasklet and the node it
          writes into the same kernel leaves nothing at that exit, so the first test reports
          nothing and the descriptor is never claimed for the device (polybench durbin).

        Whether an inside write counts is decided by SCOPE, not by storage. Storage is still
        ``Default`` when this runs, so durbin's ``alpha`` and the kernel-local
        ``_wcr_priv_set_sum_out_sum`` are indistinguishable by it; what separates them is that
        ``alpha`` is also accessed outside the kernel that writes it. An access under a DIFFERENT
        kernel counts as outside too -- two launches cannot hand a value to each other in a
        register.
        """
        through_the_exit: OrderedSet[str] = OrderedSet()
        written_inside: OrderedSet[str] = OrderedSet()
        kernels_per_data: dict[str, OrderedSet[Optional[nodes.MapEntry]]] = {}
        for state in sdfg.states():
            scopes = state.scope_dict()
            for node in state.nodes():
                if isinstance(node, (nodes.MapExit, nodes.LibraryNode)) and self.has_GPU_schedule(node):
                    through_the_exit |= self.get_data_used_by_outgoing_access_nodes(sdfg,
                                                                                    state,
                                                                                    node,
                                                                                    include_scalars=True)
                if not isinstance(node, nodes.AccessNode) or node.data not in sdfg.arrays:
                    continue
                kernel = self.enclosing_kernel(scopes, node)
                kernels_per_data.setdefault(node.data, OrderedSet()).add(kernel)
                if kernel is not None and state.in_degree(node) > 0:
                    written_inside.add(node.data)
        return through_the_exit | OrderedSet(name for name in written_inside if len(kernels_per_data[name]) > 1)

    def refuse_by_value_scalars_the_device_writes(self, sdfg: SDFG) -> None:
        """Raise if a Scalar a kernel writes would reach that kernel BY VALUE.

        ``Scalar.as_arg`` renders a pointer for ``GPU_Global`` and a plain ``double x`` parameter
        for every other storage, so a kernel handed one writes its own stack and the write is
        discarded. Nothing downstream objects -- the launch succeeds and the numbers are wrong
        (polybench durbin) -- so the placement is CHECKED here rather than trusted. A descriptor
        this pass failed to claim has to stop the compile, not reach a user as a result.
        """
        offenders = [
            name for name in self.data_written_by_device_code(sdfg) if isinstance(sdfg.arrays[name], data.Scalar)
            and sdfg.arrays[name].storage is not dtypes.StorageType.GPU_Global
        ]
        if offenders:
            raise ValueError(f'device code writes {offenders}, still Scalars in host storage. A kernel takes those '
                             f'BY VALUE, so the write would be discarded and the result silently wrong: '
                             f'{[(name, sdfg.arrays[name].storage.name) for name in offenders]}')

    def scalarize_locals_of_removed_trivial_maps(self, sdfg: SDFG) -> None:
        """Drop single-iteration maps, then scalarize the kernel locals that lost theirs.

        ``TrivialMapElimination`` declines a GPU schedule itself, so a kernel is never the map that
        goes; what goes is a one-iteration loop inside one. The filter is the DIFFERENCE across the
        elimination rather than the census before it: a map the transformation declines (a dynamic
        map range keeps one parameter alive) leaves its array under a map still, and converting that
        one would hand a kernel a by-value scalar and lose the write.
        """
        before = self.kernel_local_len1_arrays(sdfg)
        if not sdfg.apply_transformations_repeated(TrivialMapElimination, validate=False, validate_all=False):
            return
        self.cache_scopes(sdfg)
        freed = before - self.kernel_local_len1_arrays(sdfg)
        if freed:
            ConvertLengthOneArraysToScalars(recursive=True, filter=freed).apply_pass(sdfg, {})
            self.cache_scopes(sdfg)

    def touches_device_code(self, scopes: dict, node: nodes.Node) -> bool:
        """Whether ``node`` is device code, or the boundary of a scope that is."""
        if isinstance(node, (nodes.MapEntry, nodes.MapExit)):
            return self.get_schedule(node) in dtypes.GPU_SCHEDULES
        # A nested SDFG holds its own scopes, so answer for it the way that cannot be wrong in the
        # direction that matters: calling it device keeps its containers out of the host staging.
        if isinstance(node, nodes.NestedSDFG):
            return True
        if isinstance(node, nodes.LibraryNode):
            return self.has_GPU_schedule(node)
        return self.enclosing_kernel(scopes, node) is not None

    def data_sides(self, sdfg: SDFG) -> tuple[OrderedSet[str], OrderedSet[str], OrderedSet[str]]:
        """Which side of the machine touches each top-level container, and what is written.

        Read off the graph rather than off the IR: the per-state analysis answers for one state at
        a time and its hybrid resolution rewrites the answer, so by the time the IR exists a
        container that only host code touches already reads as a device one.
        """
        host: OrderedSet[str] = OrderedSet()
        device: OrderedSet[str] = OrderedSet()
        written: OrderedSet[str] = OrderedSet()
        for state in sdfg.states():
            scopes = state.scope_dict()
            # A fallback arm's host reads are conditional, and the arm already copies what it needs
            # inside itself. Counting them here would answer a rarely-taken read with a copy every
            # execution pays for -- the exact cost the arm-local copies exist to avoid.
            fallback = in_sequential_specialization_arm(state)
            # Host means "touched by code that STAYS on the host". A free tasklet sharing its state
            # with device code does not: the hybrid resolution wraps it in a len-1 map, and a
            # container staged on the strength of that use is then written from inside a kernel.
            hybrid = any(self.touches_device_code(scopes, node) for node in state.nodes())
            for node in state.data_nodes():
                if node.data not in sdfg.arrays:
                    continue
                if state.in_degree(node) > 0:
                    written.add(node.data)
                for edge in state.all_edges(node):
                    other = edge.dst if edge.src is node else edge.src
                    if self.touches_device_code(scopes, other):
                        device.add(node.data)
                    elif not fallback and not hybrid:
                        host.add(node.data)
        return host, device, written

    def stage_on_host(self, sdfg: SDFG, name: str, write_back: bool) -> bool:
        """Give ``name`` a host copy, point every use at it, and copy at the SDFG's boundary.

        Declines rather than stage a written container it cannot write back: the host copy would
        hold the answer and the caller's array would not, which is a wrong number rather than a
        broken graph, so it has to be refused where it can still be seen.
        """
        sinks = sdfg.sink_nodes() if write_back else []
        if write_back and not sinks:
            return False

        existing = OrderedSet(sdfg.all_control_flow_blocks())
        self.create_interstate_copy(sdfg, None, sdfg.start_block, OrderedSet([name]), to_gpu=False)

        rename = {name: self._get_host_name(name)}
        for block in existing:
            if isinstance(block, SDFGState):
                self._insert_copy_names_in_state(block, rename)
            else:
                block.replace_meta_accesses(rename)
        for edge in sdfg.all_interstate_edges():
            edge.data.replace(name, self._get_host_name(name))

        # Every exit needs the write-back, not just one: a sink each side of a branch is two ways out.
        for sink in sinks:
            self.create_interstate_copy(sdfg, sink, None, OrderedSet([name]), to_gpu=True)
        return True

    def place_single_sided_data(self, sdfg: SDFG) -> None:
        """Give a container ONE home before the per-state placement runs.

        A signature array is put on the device wholesale so the caller can hand one down, but that
        is an ABI decision, not a placement: a container only host code touches is then device
        memory the host writes, and the per-state analysis cannot undo it because it sees one state
        at a time and its hybrid resolution answers by moving the whole state onto the device --
        which for npbench nbody's ``PE`` means wrapping a scalar accumulation into one kernel launch
        per iteration. Staging it on the host instead is both correct and free.

        Only the single-sided case is decided here. A container both sides WRITE needs coherence and
        stays with the per-state copies; one both sides only read needs none, so a single copy at
        entry serves every host read for the rest of the run.
        """
        host, device, written = self.data_sides(sdfg)
        for name in list(sdfg.arrays):
            desc = sdfg.arrays[name]
            if desc.transient or desc.storage != dtypes.StorageType.GPU_Global or name not in host:
                continue
            if name in device and name in written:
                continue

            if name in device:
                # Read on both sides and written by neither, so the two copies can never disagree:
                # one copy at entry, and the per-state renaming already points the host reads at it.
                self.create_interstate_copy(sdfg, None, sdfg.start_block, OrderedSet([name]), to_gpu=False)
            elif not self.stage_on_host(sdfg, name, write_back=name in written):
                continue

            # The copy states are new blocks; every later step reads scopes from the cache.
            self.cache_scopes(sdfg)

    def place_and_copy(self, sdfg: SDFG) -> None:
        """Place every array across ONE host level and copy accordingly; taskloop bodies come later."""
        # step 2:
        # Names already put through a conversion. A SIGNATURE descriptor is not rewritten in place:
        # ``preserve_abi`` stages a transient beside it and copies, so the array itself is still an
        # array on the next scan and would be requested again for ever (TSVC s332's ``result``).
        attempted: OrderedSet[str] = OrderedSet()

        for _ in range(3):
            # step 2: copy analysis -> IR stores analysis results
            self.hybrid_states = OrderedSet()
            sdfgIR = self.sdfg_to_IR(sdfg)

            # step 3: resolve hybrid states
            new_maps = OrderedSet()
            if self.hybrid_states:
                for state in self.hybrid_states:
                    new_maps |= self.make_size1_map_wrappers(sdfg, state)

            if new_maps:
                mapfusion_pass = FullMapFusion(
                    strict_dataflow=True,
                    perform_vertical_map_fusion=True,
                    perform_horizontal_map_fusion=True,
                )
                mapfusion_pipeline = ppl.Pipeline([mapfusion_pass])
                mapfusion_pipeline.apply_pass(sdfg, {})

            # step 4: assign scalars / len1-arrays correctly
            all_scalars: OrderedSet[str] = OrderedSet(data_name for data_name in sdfg.arrays
                                                      if self._is_scalar(data_name, sdfg))
            all_len1arrays: OrderedSet[str] = {
                data_name
                for data_name in sdfg.arrays if self._is_length1_array(data_name, sdfg)
            }
            gpu_written = self.data_written_by_device_code(sdfg)

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
            rewritten: OrderedSet[str] = OrderedSet()
            if to_len1_arrays:
                rewritten |= ConvertScalarsToLengthOneArrays(
                    recursive=True,
                    preserve_abi=True,
                    filter=to_len1_arrays,
                ).apply_pass(sdfg, {}) or OrderedSet()

            if to_scalars:
                rewritten |= ConvertLengthOneArraysToScalars(
                    recursive=True,
                    preserve_abi=True,
                    filter=to_scalars,
                ).apply_pass(sdfg, {}) or OrderedSet()

            # What the wrappers BUILT, not the states that asked: a partition can be legitimately
            # declined (a lone staging node, or one computing only scalars -- covariance's ``N - 1``),
            # and a request declined every round is a fixed point, not progress.
            if rewritten or new_maps:  # sdfg has been changed
                self.cache_scopes(sdfg)
                continue  # repeat phases 2 - 4

            break  # else IR is correct for current sdfg, go on to next step

        else:
            raise RuntimeError("Offloading did not settle: the copy analysis, the hybrid-state "
                               "resolution and the scalar/len-1 assignment kept changing the graph "
                               "for 3 rounds.")

        # TODO: remove eventually
        def assert_no_scalars(node: OffloadingIRNode):
            scalars = OrderedSet(data_name for data_name in node.gpu_set | node.cpu_set
                                 if self._is_scalar(data_name, sdfg))
            assert not scalars, (f"scalars {scalars} found in {node.debug_name}\n"
                                 f"\tgpu: {node.gpu_set}\n\tcpu: {node.cpu_set}")

        self.__traverse_IR(sdfgIR, assert_no_scalars)

        # step 4: insert copies based on IR
        self.eval_IR(sdfg, sdfgIR)

    def offload_host_level_bodies(self, sdfg: SDFG) -> None:
        """Place again inside every nested SDFG that is still host code: each body is its own level.

        Its connector-bound descriptors already carry this level's storage, so the body seeds from
        that and copies only what its own control flow needs. Schedules were assigned tree-wide.

        Two kinds of body qualify, and both fail the same way when skipped -- their interstate edges
        read device arrays from host code. A taskloop body, whatever the config says, because
        ``find_taskloops`` records a map enclosing a device-wide library node unconditionally
        (npbench spmv's ``start = A_indptr[i]``); and one sitting at a state's own top level, which
        no map encloses at all (npbench scattering_self_energies' ``neigh_idx``). The level's own
        analysis only asks such a body where it wants its bound arrays -- it never places within it.
        """
        # Copy insertion added states; the cached scopes predate them.
        self.cache_scopes(sdfg)
        for state in sdfg.states():
            for node in self.host_level_nested_sdfgs(state, None):
                self.stage_device_scalar_bindings(sdfg, state, node)
                self.inherit_binding_storage(sdfg, state, node)
                body = type(self)()
                body.taskloop_heuristics = self.taskloop_heuristics
                body.taskloops = self.taskloops
                # The body places its own level, so it needs the state ``apply_pass`` seeds.
                body.hybrid_overlap = {}
                body.cache_scopes(node.sdfg)
                body.place_and_copy(node.sdfg)
                body.offload_host_level_bodies(node.sdfg)

    def stage_device_scalar_bindings(self, sdfg: SDFG, state: SDFGState, nsdfg_node: nodes.NestedSDFG) -> None:
        """Materialize on the host a device element bound to a scalar connector host code reads.

        The lowering rule is that a scalar connector names ONE element by reference, so the body
        reads it wherever the binding points -- and type inference propagates the outer storage
        inward, which is right. A body that reads it as host code therefore has no valid binding to
        device memory at all: nothing writes the value, so no copy-back exists to place, and the
        placement machinery works on arrays and never sees a scalar. Bringing the element to the
        host at the binding site is the decision the graph is missing (npbench azimint_hist reads
        ``bin_edges[i]`` in a host subtraction).

        Inputs only, and only when the body reads the value outside a kernel: one used on the device
        alone is already where it belongs.
        """
        for edge in state.in_edges(nsdfg_node):
            if edge.data is None or edge.data.is_empty() or edge.dst_conn is None:
                continue
            body = nsdfg_node.sdfg
            if edge.dst_conn not in body.arrays or not self._is_scalar(edge.dst_conn, body):
                continue
            if sdfg.arrays[edge.data.data].storage not in GPU_RESIDENT_STORAGES:
                continue
            if not self.read_outside_a_kernel(body, edge.dst_conn):
                continue
            desc = body.arrays[edge.dst_conn]
            host_name, _ = sdfg.add_scalar(f"{edge.dst_conn}_host",
                                           desc.dtype,
                                           transient=True,
                                           storage=dtypes.StorageType.Default,
                                           find_new_name=True)
            staged = state.add_access(host_name)
            # Reuse the edge's own source: a second access node for the same data would leave the
            # original isolated once this edge goes, which is not a valid SDFG.
            source = edge.src if isinstance(edge.src, nodes.AccessNode) else state.add_read(edge.data.data)
            state.remove_edge(edge)
            state.add_edge(source, edge.src_conn, staged, None, deepcopy(edge.data))
            state.add_edge(staged, None, nsdfg_node, edge.dst_conn,
                           Memlet.from_array(host_name, sdfg.arrays[host_name]))

    def read_outside_a_kernel(self, sdfg: SDFG, name: str) -> bool:
        """True if ``name`` is read anywhere in ``sdfg`` that a device schedule does not cover."""
        for nested in sdfg.all_sdfgs_recursive():
            for state in nested.states():
                for node in state.data_nodes():
                    if node.data == name and state.out_degree(node) > 0 and not is_devicelevel_gpu(nested, state, node):
                        return True
            for edge in nested.all_interstate_edges():
                if name in edge.data.used_arrays(nested.arrays):
                    return True
        return False

    def host_level_nested_sdfgs(self, state: SDFGState, entry: Optional[nodes.MapEntry]):
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
            inner = nsdfg_node.sdfg.arrays[connector]
            # ARRAYS only. A scalar connector binds ONE element by reference, so it names no memory
            # of its own and the outer storage says nothing about where the body may read it: giving
            # it the device storage makes every host read of the value invalid (npbench azimint_hist
            # subtracts ``bin_edges[i]`` on the host).
            if isinstance(inner, data.Scalar):
                continue
            inner.storage = sdfg.arrays[edge.data.data].storage

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
        """Record which maps belong on the host.

        A map enclosing a device-wide library node is recorded whatever the config says: that call is
        issued by host code, so a kernel cannot contain it. The launch-only rule is the optional half.
        """
        self.taskloops = taskloop_maps(sdfg, launch_only=self.taskloop_heuristics, overrides=self.taskloop_overrides)

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
                    # A nested SDFG at a host level is a host level of its own. Under a kernel
                    # ``host_level`` is already False, so no extra gate belongs here.
                    self.assign_schedules(node.sdfg, host_level)

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
        return OrderedSet(e.dst for e in state.out_edges(node))

    def get_predecessors(self, state, node):
        return OrderedSet(e.src for e in state.in_edges(node))

    ### STEP 2: copy analysis ###

    ### Helpers to get the set of arrays accessed by specific nodes or edges ###

    def get_data_used_by_incoming_access_nodes(self,
                                               sdfg: SDFG,
                                               state: SDFGState,
                                               node: nodes.Node,
                                               include_scalars: bool = False) -> OrderedSet[str]:

        def recursion(node: nodes.Node, visited_set: OrderedSet[nodes.Node]):
            # the visited set is necessary for edge cases, e.g. an access node A whose predecessor B is a view node
            # refering back to A
            if node in visited_set:
                return OrderedSet()
            visited_set.add(node)

            # find accessed arrays
            arrays: OrderedSet[str] = OrderedSet()
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

        return recursion(node, OrderedSet())

    def get_data_used_by_outgoing_access_nodes(self,
                                               sdfg: SDFG,
                                               state: SDFGState,
                                               node: nodes.Node,
                                               include_scalars: bool = False) -> OrderedSet[str]:

        def recursion(node: nodes.Node, visited_set: OrderedSet[nodes.Node]):
            # the visited set is necessary for edge cases, e.g. an access node A whose successor B is a view node
            # refering back to A
            if node in visited_set:
                return OrderedSet()
            visited_set.add(node)

            # find accessed arrays
            arrays: OrderedSet[str] = OrderedSet()
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

        return recursion(node, OrderedSet())

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

        return OrderedSet()

    def host_preferred_arrays(self, sdfg: SDFG, state: SDFGState, node: nodes.LibraryNode) -> OrderedSet[str]:
        """Single-element INPUTS of a device library node, which are cheaper to leave on the host.

        A vendor call reads a coefficient or a seed through a host pointer just as happily as a
        device one, so moving one element to the device buys nothing and costs a transfer before the
        launch. This is a preference, not a pin: a value some kernel already writes on the device
        stays there and the expansion takes the device-pointer path instead. Outputs are excluded --
        the call writes those on the device.
        """
        preferred: OrderedSet[str] = OrderedSet()
        for edge in state.in_edges(node):
            if edge.dst_conn is None or edge.data is None or edge.data.is_empty():
                continue
            name = edge.data.data
            # Length-1 ARRAYS only: a scalar is never placed at all (the pass asserts as much), so
            # naming one here would put it in a set that must not hold it -- tsvc_2_5
            # ext_break_capture's ``__ff_KFIND``.
            if name in sdfg.arrays and self._is_length1_array(name, sdfg):
                preferred.add(name)
        return preferred

    def host_pinned_arrays_in_state(self, sdfg: SDFG, state: SDFGState) -> OrderedSet[str]:
        """Every host-pinned array of the library nodes at this state's own level.

        Only that level: a node under a kernel is device code, and a pin there would name memory
        the host cannot reach anyway.
        """
        pinned: OrderedSet[str] = OrderedSet()
        for node in self.cached_scope_children[state].get(None, ()):
            if isinstance(node, nodes.LibraryNode):
                pinned |= self.host_pinned_arrays(sdfg, state, node)
        return pinned

    def host_pinned_arrays(self, sdfg: SDFG, state: SDFGState, node: nodes.LibraryNode) -> OrderedSet[str]:
        """Arrays this library node reaches through a connector it declares HOST-resident.

        A node whose expansion is a device call can still read part of its interface on the host --
        cuBLAS takes alpha and beta through a host pointer, ``ScatterConflictCheck`` reads its flag
        there -- and says so through ``LibraryNode.host_connectors``. Placing one of those on the
        device gives the expansion a device pointer to dereference in host code.
        """
        if not node.host_connectors:
            return OrderedSet()
        pinned: OrderedSet[str] = OrderedSet()
        for edge in state.in_edges(node):
            if edge.dst_conn in node.host_connectors:
                pinned |= self.get_arrays_used_by_edge(sdfg, state, edge, False)
        for edge in state.out_edges(node):
            if edge.src_conn in node.host_connectors:
                pinned |= self.get_arrays_used_by_edge(sdfg, state, edge, True)
        return pinned

    def get_arrays_used_by_node(self, sdfg, state, node):
        arrays: OrderedSet[str] = OrderedSet()

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
                      gpu_set: OrderedSet[str],
                      cpu_set: OrderedSet[str],
                      is_gpu: bool,
                      host_level: bool = False) -> tuple[OrderedSet[str], OrderedSet[str]]:
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
                              gpu_set: OrderedSet[str],
                              cpu_set: OrderedSet[str],
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
                    # A host pin is about a HOST-issued call taking a value by value; inside a kernel
                    # the expansion is device code and there is no host to read it from.
                    host_side = OrderedSet() if is_gpu else (self.host_pinned_arrays(sdfg, state, node)
                                                             | self.host_preferred_arrays(sdfg, state, node))
                    for name in self.get_arrays_used_by_node(sdfg, state, node):
                        _add_data(name, gpu_set, cpu_set, on_gpu and name not in host_side, host_level)

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
        gpu_set: OrderedSet[str] = OrderedSet()
        cpu_set: OrderedSet[str] = OrderedSet()
        _recursive_helper(sdfg, state, map_entry, gpu_set, cpu_set, False)
        return gpu_set, cpu_set

    def get_data_locations_of_nested_sdfg(self, sdfg: SDFG, state: SDFGState,
                                          node: nodes.NestedSDFG) -> tuple[OrderedSet[str], OrderedSet[str]]:
        """Where a nested SDFG wants its bound arrays, in the OUTER SDFG's names."""
        # Its hybrid states are resolved when the body is offloaded; wrapping them needs that SDFG.
        outer_hybrid = self.hybrid_states
        self.hybrid_states = OrderedSet()
        inner_gpu, inner_cpu = self.get_data_locations_of_cfregion(node.sdfg, node.sdfg)
        self.hybrid_states = outer_hybrid

        gpu_set: OrderedSet[str] = OrderedSet()
        cpu_set: OrderedSet[str] = OrderedSet()
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
                                    recursive_call=False) -> tuple[OrderedSet[str], OrderedSet[str]]:
        # iterate through all toplevel nodes of this state
        #  - map entry -> give to get_data_locations_of_map, which handles all nodes inside scope
        #  - control flow (nested) -> recurse
        #  - non-nested toplevel scopes -> add accessed data to cpu set
        gpu_set: OrderedSet[str] = OrderedSet()
        cpu_set: OrderedSet[str] = OrderedSet()

        # The analysis phase never mutates, so the scope map cached for this round is the current one.
        top_level_nodes = self.cached_scope_children[state][None]
        #: What a bare host tasklet touches, which is the one host use a size-1 map can lift.
        free_tasklet_data: OrderedSet[str] = OrderedSet()

        for node in top_level_nodes:

            g, c = OrderedSet(), OrderedSet()

            # process map and all nodes within -> may be on GPU
            if isinstance(node, nodes.MapEntry):
                g, c = self.get_data_locations_of_map(sdfg, state, node)

            elif isinstance(node, nodes.MapExit):
                pass

            # library nodes are usually GPU, can be CPU
            elif isinstance(node, nodes.LibraryNode):
                host_side = (self.host_pinned_arrays(sdfg, state, node) | self.host_preferred_arrays(sdfg, state, node))
                if self.has_GPU_schedule(node):
                    g = self.get_arrays_used_by_node(sdfg, state, node) - host_side
                    c = host_side
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
                free_tasklet_data |= c

            elif isinstance(node, nodes.AccessNode):
                pass  # nothing to do; cannot be classified without context

            else:
                raise RuntimeError(f"unhandled node {node} of type {node.__class__.__name__} in state {state}")

            gpu_set |= g
            cpu_set |= c

        # A name a host-issued call reads BY VALUE cannot be moved, whatever the state around it
        # does -- ``gpucub::DeviceScan``'s seed (tsvc_2_5 fission_dep_then_indep).
        pinned = self.host_pinned_arrays_in_state(sdfg, state)
        if pinned:
            cpu_set |= gpu_set & pinned
            gpu_set -= pinned

        # Check for hybrid state configurations, where arrays are accessed on both CPU and GPU.
        # A free tasklet over an already device-resident array is the same hybrid wearing a shape
        # the overlap cannot see: it is host code reading device memory, and nothing else in the
        # state marks that array as a device use, so the state goes unreconciled and the graph is
        # invalid (npbench nbody writes PE from one such tasklet and never reads it on the device).
        # The sequential arm of a guarded specialization is the exception -- see
        # :func:`in_sequential_specialization_arm`.
        resident: OrderedSet[str] = OrderedSet()
        if not in_sequential_specialization_arm(state):
            resident = OrderedSet(name for name in free_tasklet_data
                                  if name in sdfg.arrays and sdfg.arrays[name].storage == dtypes.StorageType.GPU_Global)
        overlap = (gpu_set & cpu_set) | (resident - pinned)
        if overlap:
            self.hybrid_states.add(state)
            self.hybrid_overlap[state] = OrderedSet(overlap)
            gpu_set |= cpu_set - pinned
            cpu_set &= pinned

        return gpu_set, cpu_set

    def get_data_locations_of_condblock(self, sdfg: SDFG,
                                        block: ConditionalBlock) -> tuple[OrderedSet[str], OrderedSet[str]]:
        gpu_set: OrderedSet[str] = OrderedSet()
        cpu_set: OrderedSet[str] = OrderedSet()

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

    def get_data_locations_of_loop(self, sdfg: SDFG, loop: LoopRegion) -> tuple[OrderedSet[str], OrderedSet[str]]:
        # get array accesses in init_statement, update_statement, and loop_condition
        cpu_set: OrderedSet[str] = OrderedSet()
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

    def get_data_locations_of_cfblock(self, sdfg: SDFG,
                                      block: ControlFlowBlock) -> tuple[OrderedSet[str], OrderedSet[str]]:
        if isinstance(block, SDFGState):
            return self.get_data_locations_of_state(sdfg, block)

        elif isinstance(block, ConditionalBlock):
            return self.get_data_locations_of_condblock(sdfg, block)

        elif isinstance(block, LoopRegion):
            return self.get_data_locations_of_loop(sdfg, block)

        elif isinstance(block, ControlFlowRegion):
            return self.get_data_locations_of_cfregion(sdfg, block)

        elif isinstance(block, (nodes.NestedSDFG, ReturnBlock, ContinueBlock, BreakBlock)):
            return OrderedSet(), OrderedSet()  # do nothing

        raise RuntimeError(f"Unknown block type: {block} of type {block.__class__.__name__}")

    def get_data_locations_of_cfregion(self, sdfg: SDFG,
                                       cfr: ControlFlowRegion) -> tuple[OrderedSet[str], OrderedSet[str]]:
        gpu_set: OrderedSet[str] = OrderedSet()
        cpu_set: OrderedSet[str] = OrderedSet()

        for block in cfr.bfs_nodes():
            g, c = self.get_data_locations_of_cfblock(sdfg, block)
            gpu_set |= g
            cpu_set |= c

        return gpu_set, cpu_set

    # wrapper
    #def get_data_locations(self, sdfg:SDFG) -> tuple[OrderedSet[str], OrderedSet[str]]:
    #    return self.get_data_locations_of_cfregion(sdfg, sdfg)

    ### STEP 3: Intermediate Representation ###
    def is_array_stored_on_GPU(self, sdfg, array_name):
        storage = sdfg.arrays[array_name].storage
        if storage in GPU_RESIDENT_STORAGES:
            return True
        elif storage in {
                dtypes.StorageType.Default, dtypes.StorageType.Register, dtypes.StorageType.CPU_Heap,
                dtypes.StorageType.CPU_Pinned, dtypes.StorageType.CPU_ThreadLocal
        }:
            return False
        else:
            raise NotImplementedError(f"array {array_name!r} lives in {storage}, which this pass does not offload")

    def written_arrays(self, sdfg: SDFG) -> OrderedSet[str]:
        """Names this SDFG writes: an access node with an incoming edge.

        Not ``SDFG.read_and_write_sets``, which resolves every access node's descriptor: the one
        caller runs between ``_insert_copy_names``, which renames nodes onto the device names, and
        ``create_interstate_copy``, which registers those descriptors -- so the graph names data it
        does not hold yet and resolving raises.
        """
        written: OrderedSet[str] = OrderedSet()
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
        initially_on_gpu = OrderedSet()
        initially_on_cpu = OrderedSet()

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
            in_edge_arrays = OrderedSet()
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
                        meta_data_node = OffloadingIRNode.new_state_node(block, cpu_set=meta_data, gpu_set=OrderedSet())
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
                    meta_data = OrderedSet(memlet.data for memlet in loop.get_meta_read_memlets()
                                           if memlet.data in sdfg.arrays)
                    if meta_data:
                        meta_data_node = OffloadingIRNode.new_state_node(block, cpu_set=meta_data, gpu_set=OrderedSet())
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

        return recursion(IR, OrderedSet())

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
        IR.gpu_set = OrderedSet(array_name for array_name in location_on_gpu if location_on_gpu[array_name])
        IR.cpu_set = OrderedSet(array_name for array_name in location_on_gpu if not location_on_gpu[array_name])

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
            IR.close.gpu_set = OrderedSet(array_name for array_name in location_on_gpu if location_on_gpu[array_name])
            IR.close.cpu_set = OrderedSet(array_name for array_name in location_on_gpu
                                          if not location_on_gpu[array_name])

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

    def _insert_copy_names_in_block(self,
                                    sdfg: SDFG,
                                    block: ControlFlowBlock,
                                    rename_dict: dict,
                                    interstate_only: bool = False):
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

        # An EDGE node stands for the interstate edges REACHING ``block``, and it holds the same block
        # object as the state node that follows it. Its decision is about what those edges read, so
        # letting it fall through here would apply it a second time to dataflow the state node has
        # already decided about -- tsvc s315, where a host tasklet writing ``a`` came out writing
        # ``a_gpu``.
        if interstate_only:
            return

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
            # Which SIDE the descriptor lives on, not which storage names that side. ``Default`` is
            # one host storage of several, and a graph that has been through ``auto_optimize`` for
            # the CPU carries ``CPU_Heap`` instead -- against which an identity test is False, so the
            # device accesses kept the HOST name. The kernel then wrote the host array while the
            # copy-back overwrote it with a device buffer nothing had written: vadv came back exactly
            # as it went in.
            for name in node.gpu_set:
                assert name in sdfg.arrays
                if not self.is_array_stored_on_GPU(sdfg, name):  # starts on CPU, but this access is on GPU
                    rename_dict[name] = self._get_gpu_name(name)

            for name in node.cpu_set:
                assert name in sdfg.arrays
                if self.is_array_stored_on_GPU(sdfg, name):  # starts on GPU, but this access is on CPU
                    rename_dict[name] = self._get_host_name(name)

            self._insert_copy_names_in_block(sdfg, node.block, rename_dict, node.type == OffloadingIRNode.EDGE)

        self.__traverse_IR(IR, _insert_copy_names_in_node)

    def _correct_transient_storage_locations(self, sdfg: SDFG, IR: OffloadingIRNode):
        seen_transients = OrderedSet()

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
        # Filled after the renaming below, where a host-side write takes the host name.
        written: OrderedSet[str] = OrderedSet()

        def insert_copies(node, next, node_block, next_block):
            # Copying BACK to the device is about host-side modifications. A name whose host copy is
            # never written already matches on the device, and when it is a nested SDFG's input
            # connector the copy is not merely wasted -- it writes a container the body may only read
            # (npbench scattering_self_energies' ``neigh_idx``).
            gpu_copies = {
                name
                for name in node.cpu_set & next.gpu_set
                if not self.is_array_stored_on_GPU(sdfg, name) or self._get_host_name(name) in written
            }
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
        written |= self.written_arrays(sdfg)
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

    def _get_boundary_in_edges(self, state: SDFGState, node, bounded_set: set):
        # A list, not a set: the connector numbering below follows this order.
        return [e for e in state.in_edges(node) if e.src not in bounded_set]

    def _get_boundary_out_edges(self, state: SDFGState, node, bounded_set: set):
        return [e for e in state.out_edges(node) if e.dst not in bounded_set]

    def _get_entry_nodes(self, state: SDFGState, bounded_set: set):
        return OrderedSet(node for node in bounded_set if all(e.src not in bounded_set for e in state.in_edges(node)))

    def _get_exit_nodes(self, state: SDFGState, bounded_set: set):
        return OrderedSet(node for node in bounded_set if all(e.dst not in bounded_set for e in state.out_edges(node)))

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

        # A region root the rewiring did not reach reads nothing, so no edge put it under the entry
        # and the scope does not contain it -- while whatever it feeds does, which is the invalid
        # inside-to-outside path. Order it after the entry instead. Rewiring a boundary edge does
        # not make the OTHER roots any less dangling, so this cannot be an else-branch of it: tsvc
        # s252 wraps one tasklet reading two arrays beside one reading none.
        for node in region_nodes:
            if state.in_degree(node) == 0:
                state.add_nedge(map_entry, node, Memlet())

        # MAP EXIT
        boundary_out_edges = []
        for node in region_nodes:
            boundary_out_edges += self._get_boundary_out_edges(state, node, region_nodes)

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

        # The leaf half of the same rule.
        for node in region_nodes:
            if state.out_degree(node) == 0:
                state.add_nedge(node, map_exit, Memlet())

        return map_entry, map_exit

    def _get_new_map_identifiers(self, state: SDFGState, map_label: str, map_param: str):
        existing_labels = OrderedSet(node.label for node in state.nodes())
        existing_params = OrderedSet()
        for node in state.nodes():
            if isinstance(node, nodes.MapEntry):
                existing_params |= OrderedSet(node.map.params)

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

    def _subgraphs_after_removing_partition_nodes(self,
                                                  state: SDFGState,
                                                  partition_nodes: set,
                                                  scope_entry=None,
                                                  scope_children=None) -> list[OrderedSet[nodes.Node]]:
        """
        Returns connected components (as sets of nodes) after removing partition_nodes
        from ONE SCOPE of a SINGLE SDFG state graph.

        ``scope_entry`` is the map whose body is partitioned, or None for the state's top level, and
        ``scope_children`` the caller's already-computed scope map -- recomputing one walks the whole
        state. Connectivity is treated as undirected (uses both in/out edges) but never leaves the
        scope: an edge out of it lands on the enclosing entry or exit, a boundary and not a member.
        """
        visited = OrderedSet()
        components = []
        if scope_children is None:
            scope_children = state.scope_children()
        members = scope_children[scope_entry]
        scope_nodes = OrderedSet(members)
        remaining_nodes = [n for n in members if n not in partition_nodes]

        for start in remaining_nodes:
            if start in visited:
                continue

            comp = OrderedSet()
            queue = self.deque([start])
            visited.add(start)

            while queue:
                u = queue.popleft()
                comp.add(u)

                neighbors = OrderedSet(e.dst for e in state.out_edges(u)) | OrderedSet(e.src for e in state.in_edges(u))
                for v in neighbors:
                    if v in partition_nodes or v in visited or v not in scope_nodes:
                        continue
                    visited.add(v)
                    queue.append(v)

            components.append(comp)

        return components

    def scope_closed_partition(self, state: SDFGState, region: OrderedSet, boundary: OrderedSet):
        """``region`` grown until every map scope it touches lies wholly inside it, or None.

        A size-1 map around HALF a scope puts a map entry inside the new map and its own exit
        outside it, which is not a scope at all: ``entry_node`` then answers for the wrapping map,
        and validation reports the pair as Map objects that were copied separately. The partition
        arrives that way because the components are cut by dataflow, which a map scope spans.

        None when closing would have to swallow one of the nodes the partitioning deliberately kept
        out -- a GPU map or a device-wide library call. Those are the boundaries the partition
        exists to respect, so the answer there is to leave this group alone rather than to wrap a
        region that reaches across one.
        """
        closed: OrderedSet = OrderedSet(region)
        scope_children = state.scope_children()
        queue = list(region)
        while queue:
            node = queue.pop()
            if isinstance(node, nodes.MapEntry):
                entry = node
            elif isinstance(node, nodes.MapExit):
                entry = state.entry_node(node)
            else:
                continue
            if entry is None:
                continue
            for extra in [entry, state.exit_node(entry), *scope_children[entry]]:
                if extra is None or extra in closed:
                    continue
                if extra in boundary:
                    return None
                closed.add(extra)
                queue.append(extra)
        return closed

    def _remove_all_outer_access_nodes_from_group(self, state: SDFGState, group: set):
        outer_nodes = self._get_entry_nodes(state, group) | self._get_exit_nodes(state, group)
        nodes_to_remove = OrderedSet(node for node in outer_nodes if isinstance(node, nodes.AccessNode))

        while nodes_to_remove:
            group -= nodes_to_remove
            outer_nodes = self._get_entry_nodes(state, group) | self._get_exit_nodes(state, group)
            nodes_to_remove = OrderedSet(node for node in outer_nodes if isinstance(node, nodes.AccessNode))

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
                                           data_names: OrderedSet[str]) -> dict[str, nodes.AccessNode]:
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

    def host_level_scopes(self, state: SDFGState, scope_children: dict) -> list:
        """Every scope of ``state`` that runs as host code: the top level, and each taskloop body.

        A taskloop is host code by construction, so free computation left in its body is host code
        too -- and a device-wide library node beside it does not change that. Descend through
        taskloops only: a scope under a kernel is device code and has no free computation to lift.
        """
        found = [None]
        for entry in found:  # grows as taskloops are met; a taskloop under a kernel is never reached
            for node in scope_children[entry]:
                if isinstance(node, nodes.MapEntry) and node in self.taskloops:
                    found.append(node)
        return found

    def make_size1_map_wrappers(self, sdfg: SDFG, state: SDFGState):
        # Wrapping never adds or removes a taskloop, so the scope list is read once here.
        scopes = self.host_level_scopes(state, state.scope_children())
        new_maps = OrderedSet()
        for scope_entry in scopes:
            new_maps |= self.wrap_free_computation(sdfg, state, scope_entry)
        return new_maps

    def wrap_free_computation(self, sdfg: SDFG, state: SDFGState, scope_entry=None):
        """Lift what is left of ONE host scope into kernels: GPU nodes partition it, the rest is wrapped.

        A library node forces its parent onto the host, so a scope can hold a device-wide call and
        real computation side by side. That computation is not host work -- it is a kernel nobody
        wrapped yet, so a size-1 map around it makes it one.
        """
        scope_children = state.scope_children()
        members = scope_children[scope_entry]
        lib_nodes = OrderedSet(node for node in members
                               if isinstance(node, (nodes.LibraryNode)) and self.has_GPU_schedule(node))
        map_entries = OrderedSet(node for node in members
                                 if isinstance(node, (nodes.MapEntry)) and self.has_GPU_schedule(node))
        map_exits = OrderedSet(state.exit_node(node) for node in map_entries)
        partition_nodes = lib_nodes | map_entries | map_exits
        if scope_entry is not None:
            # The scope's own exit is its boundary, and scope_children lists it beside the body.
            partition_nodes.add(state.exit_node(scope_entry))

        partitions = self._subgraphs_after_removing_partition_nodes(state, partition_nodes, scope_entry, scope_children)
        new_maps = OrderedSet()

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

            # A partition is a dataflow component, and a map scope spans one: covariance hands this
            # a lone MapEntry whose body and exit went to another component.
            partition = self.scope_closed_partition(state, partition, partition_nodes)
            if partition is None:
                continue

            # if anything is left, wrap it
            if partition:
                map_entry, map_exit = self._wrap_region_in_size1_map(state, partition)
                new_maps.add((map_entry, map_exit))

                # Avoid illegal direct map-to-map connections by routing through an access node.
                self._insert_access_between_adjacent_maps(state, map_exit)

                # Ensure all map inputs are also outputs to avoid dace erroneusly labeling them as constants
                self._forward_input_only_map_data(state, map_entry, map_exit)

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
        all_scalars: OrderedSet[str] = OrderedSet(data_name for data_name in sdfg.arrays
                                                  if self._is_scalar(data_name, sdfg))
        all_len1arrays: OrderedSet[str] = OrderedSet(data_name for data_name in sdfg.arrays
                                                     if self._is_length1_array(data_name, sdfg))
        vars: OrderedSet[str] = all_scalars | all_len1arrays

        # 2) with current scheduling heuristic, only toplevel can be GPU
        # 3) tasklets within nested sdfgs are not relevant
        gpu_written: OrderedSet[str] = OrderedSet()
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
