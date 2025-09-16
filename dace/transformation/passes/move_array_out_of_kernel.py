# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
from typing import Dict, FrozenSet, Set, Tuple, List, Optional
import copy
import functools
from collections import deque

import sympy

import dace
from dace import SDFG, SDFGState, dtypes, data as dt
from dace.sdfg import nodes
from dace.properties import make_properties
from dace.transformation import transformation, helpers
from dace.transformation.pass_pipeline import Pass
from dace.subsets import Range
from dace.sdfg.graph import MultiConnectorEdge
from dace.memlet import Memlet
from dace.symbolic import symbol

import dace.sdfg.utils as sdutil

@make_properties
@transformation.explicit_cf_compatible
class MoveArrayOutOfKernel(Pass):
    """
    This pass supports a legacy use case in the 'ExperimentalCUDACodeGen' backend: the use of
    transient arrays with dtypes.StorageType.GPU_Global inside GPU_Device scheduled maps (kernels).
    Previously, the old 'CUDACodeGen' moved such arrays outside the kernel during codegen, which caused:

    1. Mismatches between the SDFG and the generated code,
    2. Complex, misplaced logic in codegen,
    3. Incorrect semantics — a single shared array was reused instead of per-iteration replication,
       leading to race conditions.

    This pass fixes these issues by explicitly lifting such arrays out of GPU_Device maps
    and creating disjoint arrays per map iteration. Unlike the legacy approach, the transformation
    is now visible and consistent at the SDFG level, avoiding naming collisions and improving clarity.

    NOTE: There is no true "local device (GPU_Device) memory" on GPUs, but DaCe supports this
    pattern for legacy reasons. This pass exists purely for backward compatibility, and its use
    is strongly discouraged.
    """

    def __init__(self):
        """
        Initializes caches for mapping nodes to their states and SDFGs.

        This avoids repeatedly traversing the SDFG structure during the pass.
        The caches are populated in `apply_pass` for convenience.
        """
        self._node_to_state_cache: Dict[nodes.Node, SDFGState] = dict()
        self._node_to_sdfg_cache: Dict[nodes.Node, SDFG] = dict()

    # Entry point
    def apply_pass(self, root_sdfg: SDFG, kernel_entry: nodes.MapEntry, array_name: str) -> None:
        """
        Applies the pass to move a transient GPU_Global array out of a GPU_Device map.

        Args:
            root_sdfg: The top-level SDFG to operate on.
            kernel_entry: The MapEntry node representing the GPU_Device scheduled map (i.e., the kernel)
                        that contains the transient array.
            array_name: The name of the transient array to move. Note that multiple arrays with the
                        same name may exist within the kernel. All will be lifted.
        """
        # Cache every nodes parent state and parent sdfg
        for node, parent in root_sdfg.all_nodes_recursive():
            if isinstance(node, nodes.Node):
                assert isinstance(parent, SDFGState)
                self._node_to_state_cache[node] = parent
                self._node_to_sdfg_cache[node] = parent.sdfg

        # Check if all access nodes to 'array_name' within the kernel are defined in the same SDFG as the map
        kernel_parent_sdfg = self._node_to_sdfg_cache[kernel_entry]
        simple_case = True
        for (_, outermost_sdfg, _, _) in self.collect_array_descriptor_usage(kernel_entry, array_name):
            if outermost_sdfg != kernel_parent_sdfg:
                simple_case = False
                break

        if simple_case:
            # All access nodes are in the same SDFG as the kernel map - easy
            access_nodes = [an for an, _, _ in self.get_access_nodes_within_map(kernel_entry, array_name)]
            self.move_array_out_of_kernel_flat(kernel_entry, array_name, access_nodes)
        else:
            # Access nodes span nested maps or SDFGs —  more involved (more checks, naming conflicts, several seperate
            # array descriptors with the same array_name)
            self.move_array_out_of_kernel_nested(kernel_entry, array_name)

    # Main transformation algorithms and helpers
    def move_array_out_of_kernel_flat(self, kernel_entry: nodes.MapEntry, array_name: str,
                                      access_nodes: List[nodes.AccessNode]) -> None:
        """
        Moves a transient GPU_Global array out of a GPU_Device map (kernel) in the flat case.

        This function handles the simpler case where all access nodes to the array are in the same
        SDFG and state as the kernel map. Therefore, there are no nested SDFGs or naming conflicts
        (since an SDFG cannot define multiple descriptors with the same name).

        The array is reshaped to allocate a disjoint slice per map iteration. For example, given:

            for x, y in dace.map[0:128, 0:32] @ GPU_Device:
                gpu_A = dace.define_local([64], dtype, storage=GPU_Global)

        the array shape will be updated to [128, 32, 64], and memlets will ensure each thread
        accesses [x, y, 0:64].

        Additionally, this method inserts the necessary access nodes and edges to correctly move
        the array out of the map scope and maintain correctness.

        Args:
            kernel_entry: The MapEntry node representing the GPU kernel.
            array_name: Name of the transient array to move.
            access_nodes: List of access nodes referring to the array inside the map.
        """
        # A closest AccessNode of kernel exit is used
        parent_state = self._node_to_state_cache[kernel_entry]
        kernel_exit: nodes.MapExit = parent_state.exit_node(kernel_entry)
        closest_an = self.get_nearest_access_node(access_nodes, kernel_exit)
        array_desc = closest_an.desc(parent_state)

        # Get the chain of MapEntries from the AccessNode up to and including the kernel map entry
        map_entry_chain, _ = self.get_maps_between(kernel_entry, closest_an)

        # Store the original full-range subset of the array.
        # Needed to define correct memlets when moving the array out of the kernel.
        old_subset = [(0, dim - 1, 1) for dim in array_desc.shape]

        # Update the array
        new_shape, new_strides, new_total_size, new_offsets = self.get_new_shape_info(array_desc, map_entry_chain)
        array_desc.set_shape(new_shape=new_shape, strides=new_strides, total_size=new_total_size, offset=new_offsets)

        # Update all memlets
        self.update_memlets(kernel_entry, array_name, closest_an, access_nodes)

        # add new edges to move access Node out of map
        in_connector: str = 'IN_' + array_name
        out_connector: str = 'OUT_' + array_name
        previous_node = closest_an
        previous_out_connector = None
        for next_map_entry in map_entry_chain:

            next_map_exit = parent_state.exit_node(next_map_entry)
            if in_connector not in next_map_exit.in_connectors:
                next_map_state = self._node_to_state_cache[next_map_exit]
                next_map_exit.add_in_connector(in_connector)
                next_map_exit.add_out_connector(out_connector)

                next_entries, _ = self.get_maps_between(kernel_entry, previous_node)

                next_map_state.add_edge(previous_node, previous_out_connector, next_map_exit, in_connector,
                                        Memlet.from_array(array_name, array_desc))

            previous_node = next_map_exit
            previous_out_connector = out_connector

        # New Access Node outside of the target map, connected to the exit
        access_node_outside = parent_state.add_access(array_name)
        parent_state.add_edge(kernel_exit, out_connector, access_node_outside, None,
                              Memlet.from_array(array_name, array_desc))

    def move_array_out_of_kernel_nested(self, kernel_entry: nodes.MapEntry, array_name: str) -> None:
        """
        Moves a transient GPU_Global array out of a GPU_Device map (kernel) in the nested case.

        This function handles the more complex scenario where access nodes to the array may be
        defined inside nested SDFGs within the kernel's parent SDFG. It moves the array out of
        all nested maps and SDFGs, updating shapes and memlets accordingly, and resolves naming
        conflicts that arise from multiple descriptors with the same name in different scopes
        (by renaming).

        The method also ensures that the array is correctly lifted through all nested SDFGs
        between its original definition and the kernel map, updating symbols and connectors
        along the way.

        Args:
            kernel_entry: The MapEntry node representing the GPU kernel.
            array_name: Name of the transient array to move.
        """
        # Collect all information about every distinct data descriptor with the same name "array_name"
        array_descriptor_usage = self.collect_array_descriptor_usage(kernel_entry, array_name)
        original_array_name = array_name
        kernel_parent_sdfg = self._node_to_sdfg_cache[kernel_entry]

        for array_desc, outermost_sdfg, sdfg_defined, access_nodes in array_descriptor_usage:

            if outermost_sdfg == kernel_parent_sdfg:
                # Special case: There are nested accesss nodes, but their descriptor is defined at
                # the same sdfg as the kernel. Thus, we can use the simpler algorithm.
                self.move_array_out_of_kernel_flat(kernel_entry, original_array_name, list(access_nodes))
                continue

            # The outermost node
            nsdfg_node = outermost_sdfg.parent_nsdfg_node
            map_entry_chain, _ = self.get_maps_between(kernel_entry, nsdfg_node)

            # Store the original full-range subset of the array.
            # Needed to define correct memlets when moving the array out of the kernel.
            old_subset = [(0, dim - 1, 1) for dim in array_desc.shape]

            # Update array_descriptor
            new_shape, new_strides, new_total_size, new_offsets = self.get_new_shape_info(array_desc, map_entry_chain)
            array_desc.set_shape(new_shape=new_shape,
                                 strides=new_strides,
                                 total_size=new_total_size,
                                 offset=new_offsets)
            array_desc.transient = False

            # Update memlets data movement
            self.update_memlets(kernel_entry, original_array_name, nsdfg_node, access_nodes)

            # Update name if names conflict
            required, array_name = self.new_name_required(kernel_entry, original_array_name, sdfg_defined)
            if required:
                self.replace_array_name(sdfg_defined, original_array_name, array_name, array_desc)

            # Ensure required symbols are defined
            self.update_symbols(map_entry_chain, kernel_parent_sdfg)

            # Collect all SDFGs from the outermost definition to the target map's parent (inclusive)
            sdfg_hierarchy: List[SDFG] = [outermost_sdfg]
            current_sdfg = outermost_sdfg
            while current_sdfg != kernel_parent_sdfg:
                current_sdfg = current_sdfg.parent_sdfg
                sdfg_hierarchy.append(current_sdfg)

            # Validate collected SDFGs: no None entries
            if any(sdfg is None for sdfg in sdfg_hierarchy):
                raise ValueError("Invalid SDFG hierarchy: contains 'None' entries. This should not happen.")

            # Validate depth: must include at least outer + target SDFG
            if len(sdfg_hierarchy) < 2:
                raise ValueError(f"Invalid SDFG hierarchy: only one SDFG found. "
                                 f"Expected at least two levels, since {outermost_sdfg} is not equal to "
                                 "the kernel map's SDFG and is contained within it — the last entry should "
                                 "be the kernel's parent SDFG.")

            self.lift_array_through_nested_sdfgs(array_name, kernel_entry, sdfg_hierarchy, old_subset)


    def lift_array_through_nested_sdfgs(self, array_name: str, kernel_entry: nodes.MapEntry, sdfg_hierarchy: List[SDFG],
                                        old_subset: List) -> None:
        """
        Lifts a transient array through nested SDFGs.

        For each SDFG in the hierarchy (from inner to outer), this deepcopies the array descriptor
        and adds edges from the NestedSDFG node through any enclosing maps to a new access node.
        This is done until the kernel is exited.
        Memlets are updated using `old_subset` and enclosing map parameters.

        Args:
            array_name: Name of the array to lift.
            kernel_entry: Innermost GPU kernel MapEntry.
            sdfg_hierarchy: Ordered list of nested SDFGs (inner to outer).
            old_subset: Inner array subset used for memlet construction.
        """
        # Move array out ouf the kernel map entry through nested SDFGs
        outer_sdfg = sdfg_hierarchy.pop(0)
        while sdfg_hierarchy:
            inner_sdfg = outer_sdfg
            outer_sdfg = sdfg_hierarchy.pop(0)
            nsdfg_node = inner_sdfg.parent_nsdfg_node
            nsdfg_parent_state = self._node_to_state_cache[nsdfg_node]

            # copy and add the descriptor to the outer sdfg
            old_desc = inner_sdfg.arrays[array_name]
            new_desc = copy.deepcopy(old_desc)
            outer_sdfg.add_datadesc(array_name, new_desc)


            # Get all parent scopes to detect how the data needs to flow.
            # E.g. nsdfg_node -> MapExit  needs to be nsdfg_node -> MapExit -> AccessNode (new)
            parent_scopes: List[nodes.MapEntry] = []
            current_parent_scope = nsdfg_node
            scope_dict = nsdfg_parent_state.scope_dict()
            while scope_dict[current_parent_scope] is not None and current_parent_scope is not kernel_entry:
                parent_scopes.append(scope_dict[current_parent_scope])
                current_parent_scope = scope_dict[current_parent_scope]

            # Get a new AccessNode where the nsdfg node's parent state is.
            # Note: This is in the OUTER sdfg, so this is the first accessNode accessing
            # the current array descriptor
            exit_access_node = nsdfg_parent_state.add_access(array_name)

            # Cache its location
            self._node_to_state_cache[exit_access_node] = nsdfg_parent_state
            self._node_to_sdfg_cache[exit_access_node] = outer_sdfg

            # Create a dataflow path from the NestedSDFG node to the new exit access node,
            # passing through any enclosing map scopes (if the NestedSDFG is nested within maps).
            src = nsdfg_node
            for scope_entry in parent_scopes:
                # next destination is the scope exit
                scope_exit = nsdfg_parent_state.exit_node(scope_entry)
                dst = scope_exit

                # Next, add edge between src and dst in 2 steps:
                # 1.1 Determine source connector name and register it based on src type
                if isinstance(src, nodes.NestedSDFG):
                    src_conn = array_name
                    src.add_out_connector(src_conn)
                elif isinstance(src, nodes.MapExit):
                    src_conn = f"OUT_{array_name}"
                    src.add_out_connector(src_conn)
                else:
                    raise NotImplementedError(
                        f"Unsupported source node type '{type(src).__name__}' — only NestedSDFG or MapExit are expected."
                    )

                # 1.2 Determine destination connector name and register it based on dst type
                if isinstance(dst, nodes.AccessNode):
                    dst_conn = None  # AccessNodes use implicit connectors
                elif isinstance(dst, nodes.MapExit):  # Assuming dst is the entry for parent scope
                    dst_conn = f"IN_{array_name}"
                    dst.add_in_connector(dst_conn)
                else:
                    raise NotImplementedError(
                        f"Unsupported destination node type '{type(dst).__name__}' — expected AccessNode or MapEntry.")

                # 2. Add the edge using the connector names determined in Step 1.
                next_entries, _ = self.get_maps_between(kernel_entry, src)
                memlet_subset = Range(self.get_memlet_subset(next_entries, src) + old_subset)
                nsdfg_parent_state.add_edge(src, src_conn, dst, dst_conn, Memlet.from_array(array_name, new_desc))

                # Continue by setting the dst as source
                src = dst

            # After processing all scopes, the last src (which is either the last MapExit or the intial nsdfg if there are no parent scope)
            # needs to be connected to the exit access node added before
            dst = exit_access_node

            if isinstance(src, nodes.NestedSDFG):
                src_conn = array_name
                src.add_out_connector(src_conn)
            elif isinstance(src, nodes.MapExit):
                src_conn = f"OUT_{array_name}"
                src.add_out_connector(src_conn)
            else:
                raise NotImplementedError(
                    f"Unsupported source node type '{type(src).__name__}' — only NestedSDFG or MapExit are expected.")

            next_entries, _ = self.get_maps_between(kernel_entry, src)
            memlet_subset = Range(self.get_memlet_subset(next_entries, src) + old_subset)
            nsdfg_parent_state.add_edge(src, src_conn, dst, None, Memlet.from_array(array_name, new_desc))

        # At the outermost sdfg we set the array descriptor to be transient again,
        # Since it is not needed beyond it. Furthermore, this ensures that the codegen
        # allocates the array and does not expect it as input to the kernel
        new_desc.transient = True

    # Memlet related helper functions
    def get_memlet_subset(self, map_chain: List[nodes.MapEntry], node: nodes.Node):
        """
        Compute the memlet subset to access an array based on the position of a node within nested GPU maps.

        For each GPU_Device or GPU_ThreadBlock map in the chain:
        - If the node lies inside the map (but is not the map entry or exit itself),
          the subset is the single index corresponding to the map parameter (symbolic).
        - Otherwise, the full range of the map dimension is used.

        This ensures that memlets correctly represent per-thread or per-block slices
        when moving arrays out of kernel scopes.

        Args:
            map_chain: List of MapEntry nodes representing nested maps from outermost to innermost.
            node: The node for which to determine the subset (could be an access node or map entry/exit).

        Returns:
            A list of subsets (start, end, stride) tuples for each map dimension.
        """
        subset = []
        for next_map in map_chain:
            if not next_map.map.schedule in [dtypes.ScheduleType.GPU_Device, dtypes.ScheduleType.GPU_ThreadBlock]:
                continue

            map_parent_state = self._node_to_state_cache[next_map]
            for param, (start, end, stride) in zip(next_map.map.params, next_map.map.range.ndrange()):

                node_is_map = ((isinstance(node, nodes.MapEntry) and node == next_map)
                               or (isinstance(node, nodes.MapExit) and map_parent_state.exit_node(next_map) == node))
                node_state = self._node_to_state_cache[node]
                if helpers.contained_in(node_state, node, next_map) and not node_is_map:
                    index = symbol(param)
                    subset.append((index, index, 1))
                else:
                    subset.append((start, end, stride))

        return subset

    def update_memlets(self, kernel_entry: nodes.MapEntry, array_name: str, outermost_node: nodes.Node,
                       access_nodes: Set[nodes.AccessNode]) -> None:
        """
        Updates all memlets related to a given transient array to reflect correct data
        movement when moving array out of the kernel entry.

        Any map enclosing the `outermost_node` also encloses all access nodes and is
        used to determine which maps are strictly above the access nodes. Based on this,
        we compute the correct memlet subset that includes the additional dimensions
        from the GPU map hierarchy.

        Args:
            kernel_entry: The MapEntry node representing the GPU kernel scope.
            array_name: Name of the transient array being moved out.
            outermost_node: The outermost node.
            access_nodes: Set of AccessNodes inside the kernel that reference the same array.
        """
        map_entry_chain, _ = self.get_maps_between(kernel_entry, outermost_node)
        params_as_ranges = self.get_memlet_subset(map_entry_chain, outermost_node)

        # Update in and out path memlets
        visited: Set[MultiConnectorEdge[Memlet]] = set()
        for access_node in access_nodes:
            # in paths
            for path in self.in_paths(access_node):
                for edge in path:

                    # Guards
                    if edge in visited:
                        continue

                    if edge.data.data == array_name:
                        old_range = edge.data.subset.ndrange()
                        new_range = params_as_ranges + old_range
                        edge.data.subset = Range(new_range)
                        visited.add(edge)

                    elif edge.data.data != array_name and edge.dst is access_node and edge.data.dst_subset is not None:
                        old_range = edge.data.dst_subset.ndrange()
                        new_range = params_as_ranges + old_range
                        edge.data.dst_subset = Range(new_range)
                        visited.add(edge)

                    else:
                        continue

            # out paths
            for path in self.out_paths(access_node):
                for edge in path:
                    if edge in visited:
                        continue

                    if edge.data.data == array_name:
                        old_range = edge.data.subset.ndrange()
                        new_range = params_as_ranges + old_range
                        edge.data.subset = Range(new_range)
                        visited.add(edge)

                    elif (edge.data.data
                          != array_name) and edge.src is access_node and edge.data.src_subset is not None:
                        old_range = edge.data.src_subset.ndrange()
                        new_range = params_as_ranges + old_range
                        edge.data.src_subset = Range(new_range)
                        visited.add(edge)

                    else:
                        continue

    # Array, symbol and renaming related helper functions
    def get_new_shape_info(self, array_desc: dt.Array, map_exit_chain: List[nodes.MapEntry]):
        """
        Calculate the new shape, strides, total size, and offsets for a transient array
        when moving it out of a GPU_Device kernel.

        Each GPU_Device map adds dimensions to allocate disjoint slices per thread.

        For example:

            for x, y in dace.map[0:128, 0:32] @ GPU_Device:
                gpu_A = dace.define_local([64], dtype, storage=GPU_Global)

        gpu_A's shape changes from [64] to [128, 32, 64] to give each thread its own slice
        (i.e. gpu_A[x, y, 64]).

        Args:
            array_desc: Original array descriptor.
            map_exit_chain: List of MapEntry nodes between array and kernel exit.

        Returns:
            Tuple (new_shape, new_strides, new_total_size, new_offsets) for the updated array.
        """
        extended_size = []
        new_strides = list(array_desc.strides)
        new_offsets = list(array_desc.offset)
        for next_map in map_exit_chain:
            if not next_map.map.schedule in [dtypes.ScheduleType.GPU_Device, dtypes.ScheduleType.GPU_ThreadBlock]:
                continue

            map_range: Range = next_map.map.range
            max_elements = map_range.max_element()
            min_elements = map_range.min_element()
            range_size = [max_elem + 1 - min_elem for max_elem, min_elem in zip(max_elements, min_elements)]

            #TODO: check this / clean (maybe support packed C and packed Fortran layouts separately for code readability future)
            old_total_size = array_desc.total_size
            accumulator = old_total_size
            new_strides.insert(0, old_total_size)
            for cur_range_size in range_size[:-1]:
                new_strides.insert(0, accumulator) # insert before (mult with volumes)
                accumulator = accumulator * cur_range_size

            extended_size = range_size + extended_size
            #new_strides = [1 for _ in next_map.map.params] + new_strides  # add 1 per dimension
            new_offsets = [0 for _ in next_map.map.params] + new_offsets  # add 0 per dimension

        new_shape = extended_size + list(array_desc.shape)
        new_total_size = functools.reduce(sympy.Mul, extended_size, 1) * array_desc.total_size

        return new_shape, new_strides, new_total_size, new_offsets

    # TODO: Ask Yakup -> No states test but this should be alright
    def replace_array_name(self, sdfgs: FrozenSet[SDFG], old_name: str, new_name: str, array_desc: dt.Array) -> None:
        """
        Replaces all occurrences of an array name in the given SDFGs, including its data descriptor,
        memlets, connectors and access nodes with a new name.

        Args:
            sdfgs (Set[SDFG]): The SDFGs in which to perform the renaming.
            old_name (str): The original array name to be replaced.
            new_name (str): The new array name.
            new_descriptor (dt.Array): The data descriptor associated with the old and new name.
        """
        for sdfg in sdfgs:

            # Replace by removing the data descriptor and adding it with the new name
            sdfg.remove_data(old_name, False)
            sdfg.add_datadesc(new_name, array_desc)
            sdfg.replace(old_name, new_name)

            # Find all states
            for state in sdfg.states():
                for edge in state.edges():

                    # Update out connectors
                    src = edge.src
                    old_out_conn = f"OUT_{old_name}"
                    new_out_conn = f"OUT_{new_name}"
                    if edge.src_conn == old_out_conn:
                        edge.src_conn = new_out_conn
                        src.remove_out_connector(old_out_conn)
                        src.add_out_connector(new_out_conn)

                    # Update in connectors
                    dst = edge.dst
                    old_in_conn = f"IN_{old_name}"
                    new_in_conn = f"IN_{new_name}"
                    if edge.dst_conn == old_in_conn:
                        edge.dst_conn = new_in_conn
                        dst.remove_in_connector(old_in_conn)
                        dst.add_in_connector(new_in_conn)

    def update_symbols(self, map_entry_chain: List[nodes.MapEntry], top_sdfg: SDFG) -> None:
        """
        Ensures symbols from GPU maps are defined in all nested SDFGs.

        When lifting arrays out of GPU maps, any used symbols (e.g., map indices)
        must be available in nested SDFGs for correct memlet updates.
        This function collects such symbols from the map scopes and adds them to
        the symbol tables and mappings of all nested SDFGs under `top_sdfg`.

        Args:
            map_entry_chain: List of GPU MapEntry nodes whose symbols are relevant.
            top_sdfg: The top-level SDFG under which symbols will be propagated.
        """
        all_symbols = set()
        for next_map in map_entry_chain:
            if not next_map.map.schedule in [
                    dace.dtypes.ScheduleType.GPU_Device, dace.dtypes.ScheduleType.GPU_ThreadBlock
            ]:
                continue
            all_symbols = all_symbols | next_map.used_symbols_within_scope(self._node_to_state_cache[next_map])

        for sdfg in top_sdfg.all_sdfgs_recursive():
            nsdfg_node = sdfg.parent_nsdfg_node
            if nsdfg_node is None:
                continue

            for symbol in all_symbols:
                if str(symbol) not in sdfg.symbols:
                    sdfg.add_symbol(str(symbol), dace.dtypes.int32)
                if str(symbol) not in nsdfg_node.symbol_mapping:
                    nsdfg_node.symbol_mapping[symbol] = dace.symbol(symbol)

    # Array analysis and metadata functions
    def collect_array_descriptor_usage(
            self, map_entry: nodes.MapEntry,
            array_name: str) -> Set[Tuple[dt.Array, SDFG, FrozenSet[SDFG], FrozenSet[nodes.AccessNode]]]:
        """
        Tracks usage of a transient array across nested SDFGs within the scope of a map.

        For each array it collects:
        - the outermost SDFG where it is defined or passed through,
        - all SDFGs in which it is accessed or passed via connectors,
        - all AccessNodes referencing it in those SDFGs.

        Note: By "same array" we mean arrays with the same name and connected via memlets;
        multiple descriptor objects (dt.Array) may exist across SDFGs for the same logical array.

        Args:
            map_entry: The MapEntry node whose scope is used for analysis.
            array_name: The name of the array to analyze.

        Returns:
            A set of tuples, each containing:
                - one of potentially many dt.Array descriptors,
                - the outermost defining or using SDFG,
                - a frozenset of all involved SDFGs,
                - a frozenset of all AccessNodes using this array.
        """
        access_nodes_info: List[Tuple[nodes.AccessNode, SDFGState,
                                      SDFG]] = self.get_access_nodes_within_map(map_entry, array_name)
        
        last_sdfg: SDFG = self._node_to_sdfg_cache[map_entry]

        result: Set[Tuple[dt.Array, SDFG, Set[SDFG], Set[nodes.AccessNode]]] = set()
        visited_sdfgs: Set[SDFG] = set()

        for access_node, state, sdfg in access_nodes_info:

            # Skip visited sdfgs where the array name is defined
            if sdfg in visited_sdfgs:
                continue

            # Get the array_desc (there may be several copies across SDFG, but
            # we are only interested in the information thus this is fine)
            array_desc = access_node.desc(state)

            # Collect all sdfgs and access nodes which refer to the same array
            # (we determine this by inspecting if the array name is passed via connectors)
            sdfg_set: Set[SDFG] = set()
            access_nodes_set: Set[nodes.AccessNode] = set()
            access_nodes_set.add(access_node)

            # Get all parent SDFGs and the outermost sdfg where defined
            current_sdfg = sdfg
            outermost_sdfg = current_sdfg
            while True:
                sdfg_set.add(current_sdfg)

                # We have reached the map's sdfg, so this is the
                # outermost_sdfg we consider
                if current_sdfg == last_sdfg:
                    outermost_sdfg = current_sdfg
                    break

                nsdfg_node = current_sdfg.parent_nsdfg_node
                if array_name in nsdfg_node.in_connectors or array_name in nsdfg_node.out_connectors:
                    current_sdfg = current_sdfg.parent_sdfg
                    outermost_sdfg = current_sdfg
                else:
                    break

            # Get all child SDFGs where the array was also passed to
            queue = [sdfg]
            while queue:
                current_sdfg = queue.pop(0)
                for child_state in current_sdfg.states():
                    for node in child_state.nodes():
                        if not isinstance(node, nodes.NestedSDFG):
                            continue

                        nsdfg_node = node
                        if array_name in nsdfg_node.in_connectors or array_name in nsdfg_node.out_connectors:
                            queue.append(nsdfg_node.sdfg)
                            sdfg_set.add(nsdfg_node.sdfg)

            # Get all access nodes with the array name used in the sdfgs we found
            for current_sdfg in sdfg_set:
                for current_state in current_sdfg.states():
                    for node in current_state.nodes():
                        if isinstance(node, nodes.AccessNode) and node.data == array_name:
                            access_nodes_set.add(node)

            # Update all visited sdfgs
            visited_sdfgs.update(sdfg_set)

            # Finally add information to the result
            result.add((array_desc, outermost_sdfg, frozenset(sdfg_set), frozenset(access_nodes_set)))

        return result

    def new_name_required(self, map_entry: nodes.MapEntry, array_name: str,
                          sdfg_defined: FrozenSet[SDFG]) -> Tuple[bool, str]:
        """
        Returns whether the array_name is also used at an SDFG which is not in the sdfg_defined set.
        This means that the array_name at that SDFG refers to another data descriptor.
        Another new name is suggested if this case occurs.

        Args:
            map_entry: The MapEntry node whose scope is used to determine name usage.
            array_name: The name of the data descriptor of interest
            sdfg_defined: where the data descriptor is defined

        Returns:
            A Tuple where first element is indicatin whether a new name is required, and
            the other is either the same name if no new name is required or otherwise a new name suggestion.
        """
        map_parent_sdfg = self._node_to_sdfg_cache[map_entry]
        taken_names = set()

        for sdfg in map_parent_sdfg.all_sdfgs_recursive():

            # Continue if sdfg is neither the map's parent state
            # or not contained within the map scope
            nsdfg_node = sdfg.parent_nsdfg_node
            state = self._node_to_state_cache[nsdfg_node] if nsdfg_node else None

            if not ((nsdfg_node and state and helpers.contained_in(state, nsdfg_node, map_entry))
                    or sdfg is map_parent_sdfg):
                continue

            # Taken names are all symbol and array identifiers of sdfgs in which
            # the array_name's data descriptor we are interested in IS NOT defined
            if sdfg not in sdfg_defined:
                taken_names.update(sdfg.arrays.keys())
                taken_names.update(sdfg.used_symbols(True))

        if array_name in taken_names:
            counter = 0
            new_name = f"local_{counter}_{array_name}"
            while new_name in taken_names:
                counter += 1
                new_name = f"local_{counter}_{array_name}"

            return True, new_name
        else:
            return False, array_name

    # Utility functions - basic building blocks
    def get_access_nodes_within_map(self, map_entry: nodes.MapEntry,
                                    data_name: str) -> List[Tuple[nodes.AccessNode, SDFGState, SDFG]]:
        """
        Finds all AccessNodes that refer to the given `data_name` and are located inside
        the scope of the specified MapEntry.

        Returns:
            A list of tuples, each consisting of:
                - the matching AccessNode,
                - the SDFGState in which it resides,
                - and the parent SDFG containing the node.
        """
        starting_sdfg = self._node_to_sdfg_cache[map_entry]
        matching_access_nodes = []

        for node, parent_state in starting_sdfg.all_nodes_recursive():

            if (isinstance(node, nodes.AccessNode) and node.data == data_name
                    and helpers.contained_in(parent_state, node, map_entry)):

                parent_sdfg = self._node_to_sdfg_cache[node]
                matching_access_nodes.append((node, parent_state, parent_sdfg))

        return matching_access_nodes

    def get_maps_between(self, stop_map_entry: nodes.MapEntry,
                         node: nodes.Node) -> Tuple[List[nodes.MapEntry], List[nodes.MapExit]]:
        """
        Returns all MapEntry/MapExit pairs between `node` and `stop_map_entry`, inclusive.

        Maps are returned from innermost to outermost, starting at the scope of `node` and
        ending at `stop_map_entry`. Assumes that `node` is (directly or indirectly via a
        nestedSDFG) contained within the `stop_map_entry`'s scope.

        Args:
            stop_map_entry: The outermost MapEntry to stop at (inclusive).
            node: The node from which to begin scope traversal.

        Returns:
            A tuple of two lists:
                - List of MapEntry nodes (from inner to outer scope),
                - List of corresponding MapExit nodes.
        """
        stop_state = self._node_to_state_cache[stop_map_entry]
        stop_exit = stop_state.exit_node(stop_map_entry)

        entries: List[nodes.MapEntry] = []
        exits: List[nodes.MapExit] = []

        current_state = self._node_to_state_cache[node]
        parent_info = helpers.get_parent_map(current_state, node)

        while True:
            if parent_info is None:
                raise ValueError("Expected node to be in scope of stop_map_entry, but no parent map was found.")

            entry, state = parent_info
            exit_node = state.exit_node(entry)

            entries.append(entry)
            exits.append(exit_node)

            if exit_node == stop_exit:
                break

            parent_info = helpers.get_parent_map(state, entry)

        return entries, exits

    def get_nearest_access_node(self, access_nodes: List[nodes.AccessNode], node: nodes.Node) -> nodes.AccessNode:
        """
        Finds the closest access node (by graph distance) to the given node
        within the same state. Direction is ignored.

        Args:
            access_nodes: List of candidate AccessNodes to search from.
            node: The node from which to start the search.

        Returns:
            The closest AccessNode (by number of edges traversed).

        Raises:
            RuntimeError: If no access node is conected in the node's state to the node.
        """
        state = self._node_to_state_cache[node]

        visited = set()
        queue = [node]
        while queue:
            current = queue.pop(0)
            if current in access_nodes:
                return current

            visited.add(current)
            for neighbor in state.neighbors(current):
                if neighbor not in visited:
                    queue.append(neighbor)

        raise RuntimeError(f"No access node found connected to the given node {node}. ")

    def in_paths(self, access_node: nodes.AccessNode) -> List[List[MultiConnectorEdge[Memlet]]]:
        """
        Traces all incoming dataflow paths to the given AccessNode.
        Only searches in the same state where the AccessNode is.

        Returns:
            A list of edge paths (each a list of edges).
        """
        state = self._node_to_state_cache[access_node]

        # Start paths with in-edges to the access node.
        initial_paths = [[edge] for edge in state.in_edges(access_node)]
        queue = deque(initial_paths)
        complete_paths = []

        while queue:
            # Get current path and see whether the starting node has in-edges carrying the access nodes data
            current_path = queue.popleft()
            first_edge = current_path[0]
            current_node = first_edge.src
            incoming_edges = [edge for edge in state.in_edges(current_node)]

            # If no incoming edges found, this path is complete
            if len(incoming_edges) == 0:

                complete_paths.append(current_path)
                continue

            # Otherwise, extend the current path and add it to the queue for further processing
            for edge in incoming_edges:
                if edge in current_path:
                    raise ValueError("Unexpected cycle detected")

                extended_path = [edge] + current_path
                queue.append(extended_path)

        return complete_paths

    def out_paths(self, access_node: nodes.AccessNode) -> List[List[MultiConnectorEdge[Memlet]]]:
        """
        Traces all outgoing dataflow paths to the given AccessNode.
        Only searches in the same state where the AccessNode is.

        Returns:
            A list of edge paths (each a list of edges).
        """
        state: SDFGState = self._node_to_state_cache[access_node]

        initial_paths = [[edge] for edge in state.out_edges(access_node)]
        queue = deque(initial_paths)
        complete_paths = []

        while queue:
            # Get current path and see whether the last node has out-edges carrying the access nodes data
            current_path = queue.popleft()
            last_edge = current_path[-1]
            current_node = last_edge.dst
            outgoing_edges = [edge for edge in state.out_edges(current_node)]

            # If no such edges found, this path is complete
            if len(outgoing_edges) == 0:
                complete_paths.append(current_path)
                continue

            # Otherwise, extend the current path and add it to the queue for further processing
            for edge in outgoing_edges:

                if edge in current_path:
                    raise ValueError("Unexpected cycle detected")

                extended_path = current_path + [edge]
                queue.append(extended_path)

        return complete_paths