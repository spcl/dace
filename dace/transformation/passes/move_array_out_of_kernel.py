# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Pass that hoists kernel-local transients out of GPU kernels into device-global allocations."""
from typing import Dict, FrozenSet, Set, Tuple, List
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


def _tile_extent(max_elem, min_elem):
    """Per-iteration extent of an inner-map range.

    For a tile pattern ``i = start : Min(X, start+Y) + 1`` the extent is the
    static tile width ``Y + 1`` (independent of the outer symbol ``start``).
    Otherwise fall back to the symbolic ``max_elem + 1 - min_elem``; the caller
    must ensure any shape symbols are host-visible at the lift destination.
    """
    if isinstance(max_elem, sympy.Min):
        for arg in max_elem.args:
            diff = sympy.simplify(arg - min_elem)
            if diff.is_Integer and diff >= 0:
                return diff + 1
    return max_elem + 1 - min_elem


@make_properties
@transformation.explicit_cf_compatible
class MoveArrayOutOfKernel(Pass):
    """Lift transient ``GPU_Global`` arrays out of ``GPU_Device`` maps (kernels).

    Each array is replicated per map iteration into a disjoint outer array
    (correct per-iteration semantics instead of a single racing array). GPUs
    have no per-thread ``GPU_Device`` memory, so this is backward-compat only
    and discouraged.
    """

    def __init__(self):
        """Initialize node-to-state and node-to-SDFG caches (populated in :meth:`apply_pass`)."""
        self._node_to_state_cache: Dict[nodes.Node, SDFGState] = dict()
        self._node_to_sdfg_cache: Dict[nodes.Node, SDFG] = dict()

    # Entry point
    def apply_pass(self, root_sdfg: SDFG, kernel_entry: nodes.MapEntry, array_name: str):
        """Move a transient ``GPU_Global`` array out of a ``GPU_Device`` map.

        :param root_sdfg: Top-level SDFG to operate on.
        :param kernel_entry: ``GPU_Device`` kernel MapEntry containing the array.
        :param array_name: Transient array to move; all same-named arrays are lifted.
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
            # Access nodes span nested maps or SDFGs --  more involved (more checks, naming conflicts, several seperate
            # array descriptors with the same array_name)
            self.move_array_out_of_kernel_nested(kernel_entry, array_name)

    # Main transformation algorithms and helpers
    def move_array_out_of_kernel_flat(self, kernel_entry: nodes.MapEntry, array_name: str,
                                      access_nodes: List[nodes.AccessNode]):
        """Move a transient ``GPU_Global`` array out of a kernel (flat case).

        Flat = all access nodes share the kernel map's SDFG/state, so no
        nested SDFGs or naming conflicts. The array is reshaped to a disjoint
        slice per map iteration (e.g. ``[64]`` under a ``[0:128, 0:32]`` kernel
        becomes ``[128, 32, 64]``).

        :param kernel_entry: GPU kernel MapEntry.
        :param array_name: Transient array to move.
        :param access_nodes: Access nodes referring to the array inside the map.
        """
        # Use the AccessNode closest to the kernel exit
        parent_state = self._node_to_state_cache[kernel_entry]
        kernel_exit: nodes.MapExit = parent_state.exit_node(kernel_entry)
        closest_an = self.get_nearest_access_node(access_nodes, kernel_exit)
        array_desc = closest_an.desc(parent_state)

        # MapEntry chain from the AccessNode up to and including the kernel map entry
        map_entry_chain, _ = self.get_maps_between(kernel_entry, closest_an)

        new_shape, new_strides, new_total_size, new_offsets = self.get_new_shape_info(array_desc, map_entry_chain)
        array_desc.set_shape(new_shape=new_shape, strides=new_strides, total_size=new_total_size, offset=new_offsets)

        self.update_memlets(kernel_entry, array_name, closest_an, access_nodes)

        # Add edges to move the AccessNode out of the map
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

                next_map_state.add_edge(previous_node, previous_out_connector, next_map_exit, in_connector,
                                        Memlet.from_array(array_name, array_desc))

            previous_node = next_map_exit
            previous_out_connector = out_connector

        # New AccessNode outside the target map, connected to its exit
        access_node_outside = parent_state.add_access(array_name)
        parent_state.add_edge(kernel_exit, out_connector, access_node_outside, None,
                              Memlet.from_array(array_name, array_desc))

    def move_array_out_of_kernel_nested(self, kernel_entry: nodes.MapEntry, array_name: str):
        """Move a transient ``GPU_Global`` array out of a kernel when its accesses span nested SDFGs.

        Reshapes/rewrites memlets, renames on descriptor-name conflicts, and
        lifts the array through every intermediate nested SDFG.

        :param kernel_entry: MapEntry of the GPU kernel.
        :param array_name: Transient array to move.
        """
        # Info on every distinct descriptor sharing the name ``array_name``
        array_descriptor_usage = self.collect_array_descriptor_usage(kernel_entry, array_name)
        original_array_name = array_name
        kernel_parent_sdfg = self._node_to_sdfg_cache[kernel_entry]

        for array_desc, outermost_sdfg, sdfg_defined, access_nodes in array_descriptor_usage:

            if outermost_sdfg == kernel_parent_sdfg:
                # Nested access nodes, but the descriptor is defined in the kernel's
                # SDFG -- the flat algorithm suffices.
                self.move_array_out_of_kernel_flat(kernel_entry, original_array_name, list(access_nodes))
                continue

            nsdfg_node = outermost_sdfg.parent_nsdfg_node
            map_entry_chain, _ = self.get_maps_between(kernel_entry, nsdfg_node)

            new_shape, new_strides, new_total_size, new_offsets = self.get_new_shape_info(array_desc, map_entry_chain)
            array_desc.set_shape(new_shape=new_shape,
                                 strides=new_strides,
                                 total_size=new_total_size,
                                 offset=new_offsets)
            array_desc.transient = False

            self.update_memlets(kernel_entry, original_array_name, nsdfg_node, access_nodes)

            # Rename on descriptor-name conflict
            required, array_name = self.new_name_required(kernel_entry, original_array_name, sdfg_defined)
            if required:
                self.replace_array_name(sdfg_defined, original_array_name, array_name, array_desc)

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
                                 "the kernel map's SDFG and is contained within it -- the last entry should "
                                 "be the kernel's parent SDFG.")

            self.lift_array_through_nested_sdfgs(array_name, kernel_entry, sdfg_hierarchy)

    def lift_array_through_nested_sdfgs(self, array_name: str, kernel_entry: nodes.MapEntry,
                                        sdfg_hierarchy: List[SDFG]):
        """Lift a transient array out through each nested SDFG up to the kernel boundary.

        :param array_name: Array to lift.
        :param kernel_entry: Innermost GPU kernel MapEntry.
        :param sdfg_hierarchy: Nested SDFGs ordered inner->outer.
        """
        # Lift the array through each nested SDFG up to the kernel boundary
        outer_sdfg = sdfg_hierarchy.pop(0)
        while sdfg_hierarchy:
            inner_sdfg = outer_sdfg
            outer_sdfg = sdfg_hierarchy.pop(0)
            nsdfg_node = inner_sdfg.parent_nsdfg_node
            nsdfg_parent_state = self._node_to_state_cache[nsdfg_node]

            # Copy the descriptor into the outer SDFG
            old_desc = inner_sdfg.arrays[array_name]
            new_desc = copy.deepcopy(old_desc)
            outer_sdfg.add_datadesc(array_name, new_desc)

            # Enclosing map scopes the data must flow back out through
            parent_scopes: List[nodes.MapEntry] = []
            current_parent_scope = nsdfg_node
            scope_dict = nsdfg_parent_state.scope_dict()
            while scope_dict[current_parent_scope] is not None and current_parent_scope is not kernel_entry:
                parent_scopes.append(scope_dict[current_parent_scope])
                current_parent_scope = scope_dict[current_parent_scope]

            # New AccessNode in the OUTER SDFG -- the first node accessing this descriptor
            exit_access_node = nsdfg_parent_state.add_access(array_name)

            self._node_to_state_cache[exit_access_node] = nsdfg_parent_state
            self._node_to_sdfg_cache[exit_access_node] = outer_sdfg

            # Dataflow path from the NestedSDFG node to the new exit access node,
            # through any enclosing map scopes
            src = nsdfg_node
            for scope_entry in parent_scopes:
                scope_exit = nsdfg_parent_state.exit_node(scope_entry)
                dst = scope_exit

                # Source connector, by src node type
                if isinstance(src, nodes.NestedSDFG):
                    src_conn = array_name
                    src.add_out_connector(src_conn)
                elif isinstance(src, nodes.MapExit):
                    src_conn = f"OUT_{array_name}"
                    src.add_out_connector(src_conn)
                else:
                    raise NotImplementedError(
                        f"Unsupported source node type '{type(src).__name__}' -- only NestedSDFG or MapExit are expected."
                    )

                # 1.2 Determine destination connector name and register it based on dst type
                if isinstance(dst, nodes.AccessNode):
                    dst_conn = None  # AccessNodes use implicit connectors
                elif isinstance(dst, nodes.MapExit):  # Assuming dst is the entry for parent scope
                    dst_conn = f"IN_{array_name}"
                    dst.add_in_connector(dst_conn)
                else:
                    raise NotImplementedError(
                        f"Unsupported destination node type '{type(dst).__name__}' -- expected AccessNode or MapEntry.")

                # 2. Add the edge using the connector names determined in Step 1.
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
                    f"Unsupported source node type '{type(src).__name__}' -- only NestedSDFG or MapExit are expected.")

            nsdfg_parent_state.add_edge(src, src_conn, dst, None, Memlet.from_array(array_name, new_desc))

        # At the outermost sdfg we set the array descriptor to be transient again,
        # Since it is not needed beyond it. Furthermore, this ensures that the codegen
        # allocates the array and does not expect it as input to the kernel
        new_desc.transient = True

    # Memlet related helper functions
    def get_memlet_subset(self, map_chain: List[nodes.MapEntry], node: nodes.Node):
        """Memlet subset for accessing an array given a node's position in
        nested GPU maps.

        Per ``GPU_Device``/``GPU_ThreadBlock`` map in the chain: a node
        strictly inside the map yields the single symbolic map-param index;
        otherwise the full map-dimension range. This makes memlets represent
        per-thread/per-block slices when lifting arrays out of kernels.

        :param map_chain: Nested MapEntry nodes, outermost to innermost.
        :param node: Node whose subset is computed (AccessNode or map entry/exit).
        :returns: List of ``(start, end, stride)`` tuples per map dimension.
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
                       access_nodes: Set[nodes.AccessNode]):
        """Rewrite every memlet of a transient array for correct data movement
        after lifting it out of the kernel.

        Maps enclosing ``outermost_node`` also enclose all access nodes; they
        determine which maps sit strictly above and thus the extra GPU-hierarchy
        dimensions to prepend to each subset.

        :param kernel_entry: MapEntry of the GPU kernel scope.
        :param array_name: Transient array being moved out.
        :param outermost_node: The outermost node.
        :param access_nodes: AccessNodes inside the kernel referencing the array.
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
        """New shape, strides, total size and offsets for a transient array
        lifted out of a ``GPU_Device`` kernel.

        Each GPU map prepends dimensions for per-thread disjoint slices, e.g.
        ``gpu_A`` of shape ``[64]`` under ``map[0:128, 0:32]`` becomes
        ``[128, 32, 64]`` (indexed ``gpu_A[x, y, :]``).

        For a tiled ``GPU_ThreadBlock`` map ``i = start : Min(X, start+Y) + 1``
        the per-iteration extent references ``start``, an outer-loop symbol
        invisible at host scope. :func:`_tile_extent` substitutes the tight
        static upper bound ``Y + 1``; non-tiled maps keep ``max - min + 1``.

        :param array_desc: Original array descriptor.
        :param map_exit_chain: MapEntry nodes between array and kernel exit.
        :returns: ``(new_shape, new_strides, new_total_size, new_offsets)``.
        """
        extended_size = []
        new_offsets = list(array_desc.offset)
        for next_map in map_exit_chain:
            if not next_map.map.schedule in [dtypes.ScheduleType.GPU_Device, dtypes.ScheduleType.GPU_ThreadBlock]:
                continue

            map_range: Range = next_map.map.range
            max_elements = map_range.max_element()
            min_elements = map_range.min_element()
            range_size = [_tile_extent(mx, mn) for mx, mn in zip(max_elements, min_elements)]

            extended_size = range_size + extended_size
            new_offsets = [0 for _ in next_map.map.params] + new_offsets  # add 0 per dimension

        new_shape = extended_size + list(array_desc.shape)
        # Packed C-layout strides for the prepended dims: each dimension steps over the full
        # extent of everything nested below it (the more-inner prepended dims plus the original
        # array). Built innermost-first so a dimension's extent multiplies the accumulator only
        # after that dimension's own stride has been recorded. Packed-Fortran support would need
        # a separate stride order here.
        new_strides = list(array_desc.strides)
        accumulator = array_desc.total_size
        for extent in reversed(extended_size):
            new_strides.insert(0, accumulator)
            accumulator = accumulator * extent
        new_total_size = functools.reduce(sympy.Mul, extended_size, 1) * array_desc.total_size

        return new_shape, new_strides, new_total_size, new_offsets

    def replace_array_name(self, sdfgs: FrozenSet[SDFG], old_name: str, new_name: str, array_desc: dt.Array):
        """Rename an array across ``sdfgs`` -- descriptor, memlets, connectors
        and access nodes.

        :param sdfgs: SDFGs in which to rename.
        :param old_name: Original array name.
        :param new_name: New array name.
        :param array_desc: Descriptor to re-register under ``new_name``.
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

    def update_symbols(self, map_entry_chain: List[nodes.MapEntry], top_sdfg: SDFG):
        """Propagate GPU-map symbols (e.g. map indices) into every nested SDFG
        under ``top_sdfg`` so lifted memlets referencing them stay valid.

        :param map_entry_chain: GPU MapEntry nodes whose symbols are relevant.
        :param top_sdfg: Top-level SDFG to propagate symbols under.
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

            for sym in all_symbols:
                name = str(sym)
                if name not in sdfg.symbols:
                    sdfg.add_symbol(name, dace.dtypes.int32)
                if name not in nsdfg_node.symbol_mapping:
                    nsdfg_node.symbol_mapping[name] = dace.symbol(name)

    # Array analysis and metadata functions
    def collect_array_descriptor_usage(
            self, map_entry: nodes.MapEntry,
            array_name: str) -> Set[Tuple[dt.Array, SDFG, FrozenSet[SDFG], FrozenSet[nodes.AccessNode]]]:
        """Track usage of a transient array across nested SDFGs within a map scope.

        "Same array" means same name connected via memlets -- several
        ``dt.Array`` descriptor objects may exist across SDFGs for one
        logical array.

        :param map_entry: MapEntry whose scope is analyzed.
        :param array_name: Array to track.
        :returns: Set of ``(descriptor, outermost SDFG, all involved SDFGs,
            all referencing AccessNodes)`` tuples.
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
        """Detect whether ``array_name`` collides with a different descriptor
        in an SDFG outside ``sdfg_defined``, and suggest a free name if so.

        :param map_entry: MapEntry whose scope bounds the name-usage check.
        :param array_name: Data descriptor name of interest.
        :param sdfg_defined: SDFGs where the descriptor is defined.
        :returns: ``(rename_required, name)`` -- ``name`` is the original when
            no rename is needed, else a fresh suggestion.
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
        """All AccessNodes for ``data_name`` inside ``map_entry``'s scope.

        :returns: ``(AccessNode, SDFGState, parent SDFG)`` tuples.
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
        """All MapEntry/MapExit pairs between ``node`` and ``stop_map_entry``,
        inclusive, innermost to outermost.

        Assumes ``node`` is contained (directly or via a nested SDFG) within
        ``stop_map_entry``'s scope.

        :param stop_map_entry: Outermost MapEntry to stop at (inclusive).
        :param node: Node to begin scope traversal from.
        :returns: ``(MapEntry list, MapExit list)``, inner to outer.
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
        """Closest AccessNode to ``node`` by graph distance within the same
        state (direction-agnostic BFS).

        :param access_nodes: Candidate AccessNodes.
        :param node: Node to start the search from.
        :returns: The closest AccessNode by edges traversed.
        :raises RuntimeError: No candidate is connected to ``node`` in its state.
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
        """All incoming dataflow paths to ``access_node`` within its state.

        :returns: List of edge paths (each a list of edges).
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
        """All outgoing dataflow paths from ``access_node`` within its state.

        :returns: List of edge paths (each a list of edges).
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
