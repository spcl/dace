# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import dace
from dace import dtypes, nodes, registry, Memlet
from dace.sdfg import SDFG, SDFGState, InterstateEdge
from dace.dtypes import StorageType, ScheduleType
from dace.properties import Property, make_properties
from dace.sdfg.utils import find_sink_nodes, concurrent_subgraphs
from dace.sdfg.graph import SubgraphView
from dace.transformation.transformation import SubgraphTransformation


@registry.autoregister
@make_properties
class GPUPersistentKernel(SubgraphTransformation):
    """
    This transformation takes a given subgraph of an SDFG and fuses the 
    given states into a single persistent GPU kernel. Before this transformation can
    be applied the SDFG needs to be transformed to run on the GPU (e.g. with
    the GPUTransformSDFG transformation).
    
    If applicable the transform removes the selected states from the original
    SDFG and places a `launch` state in its place. The removed states will be
    added to a nested SDFG in the launch state. If necessary guard states will
    be added in the nested SDFG, in order to make sure global assignments on
    Interstate edges will be performed in the kernel (this can be disabled with
    the `include_in_assignment` property).
    
    The given subgraph needs to fulfill the following properties to be fused:
    
     - All states in the selected subgraph need to fulfill the following:
        - access only GPU accessible memory
        - all concurrent DFGs inside the state are either sequential or inside
          a GPU_Device map.
     - the selected subgraph has a single point of entry in the form of a 
       single InterstateEdge entering the subgraph (i.e. there is at most one
       state (not part of the subgraph) from which the kernel is entered and
       exactly one state inside the subgraph from which the kernel starts
       execution)
     - the selected subgraph has a single point of exit in the form of a single
       state that is entered after the selected subgraph is left (There can be
       multiple states from which the kernel can be left, but all will leave to
       the same state outside the subgraph)
    """

    validate = Property(
        desc="Validate the sdfg and the nested sdfg",
        dtype=bool,
        default=True,
    )

    include_in_assignment = Property(
        desc="Wether to include global variable assignments of the edge going "
        "into the kernel inside the kernel or have it happen on the "
        "outside. If the assignment is needed in the kernel, it needs to "
        "be included.",
        dtype=bool,
        default=True,
    )

    kernel_prefix = Property(
        desc="Name of the kernel. If no value is given the kerenl will be "
        "refrenced as `kernel`, if a value is given the kernel will be "
        "named `<kernel_prefix>_kernel`. This is useful if multiple "
        "kernels are created.",
        dtype=str,
        default='',
    )

    @staticmethod
    def can_be_applied(sdfg: SDFG, subgraph: SubgraphView):

        if not set(subgraph.nodes()).issubset(set(sdfg.nodes())):
            return False

        # All states need to be GPU states
        for state in subgraph:
            if not GPUPersistentKernel.is_gpu_state(sdfg, state):
                return False

        # for now exactly one inner and one outer entry state
        entry_states_in, entry_states_out = \
            GPUPersistentKernel.get_entry_states(sdfg, subgraph)
        if len(entry_states_in) != 1 or len(entry_states_out) > 1:
            return False

        entry_state_in = entry_states_in.pop()
        if len(entry_states_out) == 1 \
                and len(sdfg.edges_between(entry_states_out.pop(),
                                           entry_state_in)
                        ) > 1:
            return False

        # for now only one outside state allowed, multiple inner exit states
        # allowed
        _, exit_states_out = GPUPersistentKernel.get_exit_states(sdfg, subgraph)
        if len(exit_states_out) > 1:
            return False

        # check reachability
        front = [entry_state_in]
        reachable = {entry_state_in}

        while len(front) > 0:
            current = front.pop(0)
            unseen = [suc for suc in subgraph.successors(current) if suc not in reachable]
            front += unseen
            reachable.update(unseen)

        if reachable != set(subgraph.nodes()):
            return False

        return True

    def apply(self, sdfg: SDFG):
        subgraph = self.subgraph_view(sdfg)

        entry_states_in, entry_states_out = self.get_entry_states(sdfg, subgraph)
        _, exit_states_out = self.get_exit_states(sdfg, subgraph)

        entry_state_in = entry_states_in.pop()
        entry_state_out = entry_states_out.pop() \
            if len(entry_states_out) > 0 else None
        exit_state_out = exit_states_out.pop() \
            if len(exit_states_out) > 0 else None

        launch_state = None
        entry_guard_state = None
        exit_guard_state = None

        # generate entry guard state if needed
        if self.include_in_assignment and entry_state_out is not None:
            entry_edge = sdfg.edges_between(entry_state_out, entry_state_in)[0]
            if len(entry_edge.data.assignments) > 0:
                entry_guard_state = sdfg.add_state(
                    label='{}kernel_entry_guard'.format(self.kernel_prefix + '_' if self.kernel_prefix != '' else ''))
                sdfg.add_edge(entry_state_out, entry_guard_state, InterstateEdge(entry_edge.data.condition))
                sdfg.add_edge(entry_guard_state, entry_state_in, InterstateEdge(None, entry_edge.data.assignments))
                sdfg.remove_edge(entry_edge)

                # Update SubgraphView
                new_node_list = subgraph.nodes()
                new_node_list.append(entry_guard_state)
                subgraph = SubgraphView(sdfg, new_node_list)

                launch_state = sdfg.add_state_before(
                    entry_guard_state,
                    label='{}kernel_launch'.format(self.kernel_prefix + '_' if self.kernel_prefix != '' else ''))

        # generate exit guard state
        if exit_state_out is not None:
            exit_guard_state = sdfg.add_state_before(
                exit_state_out,
                label='{}kernel_exit_guard'.format(self.kernel_prefix + '_' if self.kernel_prefix != '' else ''))

            # Update SubgraphView
            new_node_list = subgraph.nodes()
            new_node_list.append(exit_guard_state)
            subgraph = SubgraphView(sdfg, new_node_list)

            if launch_state is None:
                launch_state = sdfg.add_state_before(
                    exit_state_out,
                    label='{}kernel_launch'.format(self.kernel_prefix + '_' if self.kernel_prefix != '' else ''))

        # If the launch state doesn't exist at this point then there is no other
        # states outside of the kernel, so create a stand alone launch state
        if launch_state is None:
            assert (entry_state_in is None and exit_state_out is None)
            launch_state = sdfg.add_state(label='{}kernel_launch'.format(self.kernel_prefix +
                                                                         '_' if self.kernel_prefix != '' else ''))

        # create sdfg for kernel and fill it with states and edges from
        # ssubgraph dfg will be nested at the end
        kernel_sdfg = SDFG('{}kernel'.format(self.kernel_prefix + '_' if self.kernel_prefix != '' else ''))

        edges = subgraph.edges()
        for edge in edges:
            kernel_sdfg.add_edge(edge.src, edge.dst, edge.data)

        # Setting entry node in nested SDFG if no entry guard was created
        if entry_guard_state is None:
            kernel_sdfg.start_state = kernel_sdfg.node_id(entry_state_in)

        for state in subgraph:
            state.parent = kernel_sdfg

        # remove the now nested nodes from the outer sdfg and make sure the
        # launch state is properly connected to remaining states
        sdfg.remove_nodes_from(subgraph.nodes())

        if entry_state_out is not None \
                and len(sdfg.edges_between(entry_state_out, launch_state)) == 0:
            sdfg.add_edge(entry_state_out, launch_state, InterstateEdge())

        if exit_state_out is not None \
                and len(sdfg.edges_between(launch_state, exit_state_out)) == 0:
            sdfg.add_edge(launch_state, exit_state_out, InterstateEdge())

        # Handle data for kernel
        kernel_data = set(node.data for state in kernel_sdfg for node in state.nodes()
                          if isinstance(node, nodes.AccessNode))

        # move Streams and Register data into the nested SDFG
        # normal data will be added as kernel argument
        kernel_args = []
        for data in kernel_data:
            if (isinstance(sdfg.arrays[data], dace.data.Stream) or
                (isinstance(sdfg.arrays[data], dace.data.Array) and sdfg.arrays[data].storage == StorageType.Register)):
                kernel_sdfg.add_datadesc(data, sdfg.arrays[data])
                del sdfg.arrays[data]
            else:
                copy_desc = copy.deepcopy(sdfg.arrays[data])
                copy_desc.transient = False
                copy_desc.storage = StorageType.Default
                kernel_sdfg.add_datadesc(data, copy_desc)
                kernel_args.append(data)

        # read only data will be passed as input, writeable data will be passed
        # as 'output' otherwise kernel cannot write to data
        kernel_args_read = set()
        kernel_args_write = set()
        for data in kernel_args:
            data_accesses_read_only = [
                node.access == dtypes.AccessType.ReadOnly for state in kernel_sdfg for node in state
                if isinstance(node, nodes.AccessNode) and node.data == data
            ]
            if all(data_accesses_read_only):
                kernel_args_read.add(data)
            else:
                kernel_args_write.add(data)

        # Kernel SDFG is complete at this point
        if self.validate:
            kernel_sdfg.validate()

        # Filling launch state with nested SDFG, map and access nodes
        map_entry, map_exit = launch_state.add_map(
            '{}kernel_launch_map'.format(self.kernel_prefix + '_' if self.kernel_prefix != '' else ''),
            dict(ignore='0'),
            schedule=ScheduleType.GPU_Persistent,
        )

        nested_sdfg = launch_state.add_nested_sdfg(
            kernel_sdfg,
            sdfg,
            kernel_args_read,
            kernel_args_write,
        )

        # Create and connect read only data access nodes
        for arg in kernel_args_read:
            read_node = launch_state.add_read(arg)
            launch_state.add_memlet_path(read_node,
                                         map_entry,
                                         nested_sdfg,
                                         dst_conn=arg,
                                         memlet=Memlet.from_array(arg, sdfg.arrays[arg]))

        # Create and connect writable data access nodes
        for arg in kernel_args_write:
            write_node = launch_state.add_write(arg)
            launch_state.add_memlet_path(nested_sdfg,
                                         map_exit,
                                         write_node,
                                         src_conn=arg,
                                         memlet=Memlet.from_array(arg, sdfg.arrays[arg]))

        # Transformation is done
        if self.validate:
            sdfg.validate()

    @staticmethod
    def is_gpu_state(sdfg: SDFG, state: SDFGState) -> bool:

        # Valid storrage types
        gpu_accessible = [
            StorageType.GPU_Global,
            StorageType.GPU_Shared,
            StorageType.CPU_Pinned,
            StorageType.Register,
        ]

        for node in state.data_nodes():
            if type(node.desc(sdfg)) in [dace.data.Array,
                                        dace.data.Stream] \
                    and node.desc(sdfg).storage not in gpu_accessible:
                return False

        gpu_fused_schedules = [
            ScheduleType.Default,
            ScheduleType.Sequential,
            ScheduleType.GPU_Device,
            ScheduleType.GPU_ThreadBlock,
            ScheduleType.GPU_ThreadBlock_Dynamic,
        ]

        for schedule in [n.map.schedule for n in state.nodes() if isinstance(n, nodes.MapEntry)]:
            if schedule not in gpu_fused_schedules:
                return False

        return True

    @staticmethod
    def get_entry_states(sdfg: SDFG, subgraph):
        entry_states_in = set()
        entry_states_out = set()

        for state in subgraph:
            inner_predecessors = set(subgraph.predecessors(state))
            global_predecessors = set(sdfg.predecessors(state))
            outer_predecessors = global_predecessors - inner_predecessors
            if len(outer_predecessors) > 0:
                entry_states_in.add(state)
                entry_states_out |= outer_predecessors

        return entry_states_in, entry_states_out

    @staticmethod
    def get_exit_states(sdfg: SDFG, subgraph):
        exit_states_in = set()
        exit_states_out = set()

        for state in subgraph:
            inner_successors = set(subgraph.successors(state))
            global_successors = set(sdfg.successors(state))
            outer_successors = global_successors - inner_successors
            if len(outer_successors) > 0:
                exit_states_in.add(state)
                exit_states_out |= outer_successors

        return exit_states_in, exit_states_out
