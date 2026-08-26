# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Pass that hoists kernel-local transients out of GPU kernels into device-global allocations."""
from typing import Any, Dict, List, Optional, Tuple
import copy
import warnings

import sympy

from dace import SDFG, SDFGState, data as dt, dtypes, properties, subsets, symbolic, utils
from dace.memlet import Memlet
from dace.sdfg import is_devicelevel_gpu, nodes
from dace.transformation import helpers, pass_pipeline as ppl, transformation
from ordered_set import OrderedSet

# Deliberately NOT ``dtypes.GPU_SCHEDULES``: that also includes dynamic/persistent thread-block
# schedules this pass does not lift.
GPU_HIERARCHY_SCHEDULES = (dtypes.ScheduleType.GPU_Device, dtypes.ScheduleType.GPU_ThreadBlock)


def tile_extent(max_elem, min_elem):
    """Per-iteration extent of an inner-map range.

    For a tile pattern ``i = start : Min(X, start+Y) + 1`` the extent is the static tile width
    ``Y + 1``, independent of the outer symbol ``start``, which is not host-visible at the lift
    destination. Otherwise fall back to the symbolic ``max_elem + 1 - min_elem``.
    """
    if isinstance(max_elem, sympy.Min):
        for arg in max_elem.args:
            diff = symbolic.simplify(arg - min_elem)
            if diff.is_Integer and diff >= 0:
                return diff + 1
    return max_elem + 1 - min_elem


def is_register_demotable(desc: dt.Data, max_elements: int) -> bool:
    """True if ``desc`` is safe and worth demoting to per-thread ``Register``.

    Every shape dimension must be a concrete positive integer -- a symbol would leak into a
    host-side ``cudaMalloc`` and cannot size a per-thread array -- and the element count must
    stay within ``max_elements``. Anything larger is hoisted instead.
    """
    if any(symbolic.issymbolic(dim) for dim in desc.shape):
        return False
    try:
        total = int(utils.prod(desc.shape))
    except (TypeError, ValueError):
        return False  # e.g. sympy.oo: not symbolic, but not a finite integer either
    return 0 < total <= max_elements


def has_wcr_incoming(sdfg: SDFG, data_name: str) -> bool:
    """True if any memlet writes ``data_name`` with a WCR. Demoting such an array to a
    per-thread ``Register`` would silently break the accumulation."""
    return any(e.data.wcr is not None and e.data.data == data_name for nsdfg in sdfg.all_sdfgs_recursive()
               for state in nsdfg.states() for e in state.edges())


@properties.make_properties
@transformation.explicit_cf_compatible
class MoveArrayOutOfKernel(ppl.Pass):
    """Lift transient ``GPU_Global`` arrays out of ``GPU_Device`` maps (kernels).

    Each array is replicated per map iteration into a disjoint outer array
    (correct per-iteration semantics instead of a single racing array). GPUs
    have no per-thread ``GPU_Device`` memory, so this is backward-compat only
    and discouraged.
    """

    register_demotion_max_elements = properties.Property(
        dtype=int,
        default=64,
        desc="Max ``prod(shape)`` for a literal-shape kernel-internal transient to be demoted "
        "from GPU_Global to per-thread Register storage. Larger transients are hoisted.",
    )

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.States | ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    #: SDFG the node-to-state map below was built from; a different root discards it.
    _root: Optional[SDFG] = None
    _node_to_state: Dict[nodes.Node, SDFGState] = {}

    def state_of(self, node: nodes.Node) -> SDFGState:
        """State holding ``node``.

        The map is rebuilt on a miss: the pass adds access nodes as it lifts, and a stale map
        would answer for the graph as it was before the current array was moved.

        :param node: Node to locate.
        :returns: The state ``node`` belongs to.
        """
        if node not in self._node_to_state:
            self._node_to_state = {
                n: parent
                for n, parent in self._root.all_nodes_recursive() if isinstance(parent, SDFGState)
            }
        return self._node_to_state[node]

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Demote or hoist every transient ``GPU_Global`` array defined inside a kernel.

        :returns: Number of arrays handled, or ``None`` if there were none.
        """
        handled = 0
        for data_name, desc, kernel_entry in self.kernel_internal_gpu_global_transients(sdfg):
            if (is_register_demotable(desc, self.register_demotion_max_elements)
                    and not has_wcr_incoming(sdfg, data_name)):
                desc.storage = dtypes.StorageType.Register
            else:
                warnings.warn(f"Transient array '{data_name}' with storage type GPU_Global detected inside kernel "
                              f"{kernel_entry}. GPU_Global memory cannot be allocated within GPU kernels; the array "
                              f"will be lifted outside the kernel as a non-transient GPU_Global array.")
                self.move_array(sdfg, kernel_entry, data_name)
            handled += 1
        self.fail_on_in_kernel_global_global(sdfg)
        return handled or None

    @staticmethod
    def kernel_internal_gpu_global_transients(sdfg: SDFG) -> OrderedSet:
        """Transient ``GPU_Global`` arrays that only ever appear inside a ``GPU_Device`` map.

        A ``(name, desc)`` pair that also appears outside a kernel is left alone: the inner
        access is then a pass-through of an array the host already owns.
        """
        inside, outside = OrderedSet(), OrderedSet()
        for node, parent in sdfg.all_nodes_recursive():
            if not isinstance(node, nodes.AccessNode):
                continue
            desc = node.desc(parent)
            if not (isinstance(desc, dt.Array) and desc.transient and desc.storage is dtypes.StorageType.GPU_Global):
                continue

            kernel_entry = None
            parent_map_info = helpers.get_parent_map(state=parent, node=node)
            while parent_map_info is not None:
                map_entry, map_state = parent_map_info
                if isinstance(map_entry, nodes.MapEntry) and map_entry.map.schedule is dtypes.ScheduleType.GPU_Device:
                    kernel_entry = map_entry
                    break
                parent_map_info = helpers.get_parent_map(map_state, map_entry)

            if kernel_entry is None:
                outside.add((node.data, desc))
            else:
                inside.add((node.data, desc, kernel_entry))

        return OrderedSet([(name, desc, entry) for name, desc, entry in inside if (name, desc) not in outside])

    @staticmethod
    def fail_on_in_kernel_global_global(sdfg: SDFG) -> None:
        """Raise if a transient ``GPU_Global`` copy survives inside a kernel scope: the codegen
        has no host-side allocator there. Non-transient through-flows are connector-bound and fine."""
        offenders: List[str] = []
        for nsdfg in sdfg.all_sdfgs_recursive():
            for state in nsdfg.states():
                for edge in state.edges():
                    if not (isinstance(edge.src, nodes.AccessNode) and isinstance(edge.dst, nodes.AccessNode)):
                        continue
                    if edge.data.is_empty() or edge.data.wcr is not None:
                        continue
                    src_desc, dst_desc = nsdfg.arrays[edge.src.data], nsdfg.arrays[edge.dst.data]
                    if not (src_desc.storage is dtypes.StorageType.GPU_Global
                            and dst_desc.storage is dtypes.StorageType.GPU_Global):
                        continue
                    if not (src_desc.transient or dst_desc.transient):
                        continue
                    if not (is_devicelevel_gpu(nsdfg, state, edge.src) or is_devicelevel_gpu(nsdfg, state, edge.dst)):
                        continue
                    offenders.append(f"  - {edge.src.data} -> {edge.dst.data} in state "
                                     f"'{state.label}' (SDFG '{nsdfg.name}')")
        if offenders:
            raise ValueError("Transient GPU_Global arrays cannot live inside a kernel scope. Offenders:\n" +
                             "\n".join(offenders))

    def move_array_out_of_kernel_flat(self, kernel_entry: nodes.MapEntry, array_name: str,
                                      access_nodes: List[nodes.AccessNode]):
        """Move a transient ``GPU_Global`` array out of a kernel (flat case).

        Flat = all access nodes share the kernel map's SDFG/state, so no
        nested SDFGs or naming conflicts; the array is reshaped to a disjoint
        slice per map iteration (see :meth:`get_new_shape_info`).

        :param access_nodes: Access nodes referring to the array inside the map.
        """
        parent_state = self.state_of(kernel_entry)
        kernel_exit: nodes.MapExit = parent_state.exit_node(kernel_entry)
        closest_an = self.get_nearest_access_node(access_nodes, kernel_exit)
        array_desc = closest_an.desc(parent_state)

        map_entry_chain = self.get_maps_between(kernel_entry, closest_an)

        new_shape, new_strides, new_total_size, new_offsets = self.get_new_shape_info(array_desc, map_entry_chain)
        array_desc.set_shape(new_shape=new_shape, strides=new_strides, total_size=new_total_size, offset=new_offsets)

        self.update_memlets(kernel_entry, array_name, closest_an, access_nodes)

        # ``map_entry_chain`` runs inner to outer, which is the order the path leaves the scopes in.
        parent_state.add_memlet_path(closest_an,
                                     *(parent_state.exit_node(me) for me in map_entry_chain),
                                     parent_state.add_access(array_name),
                                     memlet=Memlet.from_array(array_name, array_desc))

    def move_array(self, root_sdfg: SDFG, kernel_entry: nodes.MapEntry, array_name: str):
        """Move a transient ``GPU_Global`` array out of a ``GPU_Device`` map.

        Each descriptor the array is known by is handled on its own: one living in the kernel's own
        SDFG only needs reshaping, while one behind a nested SDFG is additionally lifted through
        every boundary between it and the kernel, renaming on a descriptor-name conflict.

        :param root_sdfg: Outermost SDFG, which the node-to-state map is keyed on.
        :param kernel_entry: Kernel the array is moved out of.
        :param array_name: Transient array to move; all same-named arrays are lifted.
        """
        if self._root is not root_sdfg:
            self._root, self._node_to_state = root_sdfg, {}
        original_array_name = array_name
        kernel_parent_sdfg = self.state_of(kernel_entry).sdfg

        for array_desc, outermost_sdfg, sdfg_defined, access_nodes in self.collect_array_descriptor_usage(
                kernel_entry, array_name):

            if outermost_sdfg == kernel_parent_sdfg:
                # Descriptor lives in the kernel's own SDFG, so the flat algorithm suffices.
                self.move_array_out_of_kernel_flat(kernel_entry, original_array_name, list(access_nodes))
                continue

            nsdfg_node = outermost_sdfg.parent_nsdfg_node
            map_entry_chain = self.get_maps_between(kernel_entry, nsdfg_node)

            new_shape, new_strides, new_total_size, new_offsets = self.get_new_shape_info(array_desc, map_entry_chain)
            array_desc.set_shape(new_shape=new_shape,
                                 strides=new_strides,
                                 total_size=new_total_size,
                                 offset=new_offsets)
            array_desc.transient = False

            self.update_memlets(kernel_entry, original_array_name, nsdfg_node, access_nodes)

            required, array_name = self.new_name_required(kernel_entry, original_array_name, sdfg_defined)
            if required:
                self.replace_array_name(sdfg_defined, original_array_name, array_name, array_desc)

            self.update_symbols(map_entry_chain, kernel_parent_sdfg)

            sdfg_hierarchy: List[SDFG] = [outermost_sdfg]
            current_sdfg = outermost_sdfg
            while current_sdfg != kernel_parent_sdfg:
                current_sdfg = current_sdfg.parent_sdfg
                sdfg_hierarchy.append(current_sdfg)

            if any(sdfg is None for sdfg in sdfg_hierarchy):
                raise ValueError("Invalid SDFG hierarchy: contains 'None' entries. This should not happen.")

            if len(sdfg_hierarchy) < 2:
                raise ValueError(f"Invalid SDFG hierarchy: only one SDFG found. "
                                 f"Expected at least two levels, since {outermost_sdfg} is not equal to "
                                 "the kernel map's SDFG and is contained within it -- the last entry should "
                                 "be the kernel's parent SDFG.")

            self.lift_array_through_nested_sdfgs(array_name, kernel_entry, sdfg_hierarchy)

    def lift_array_through_nested_sdfgs(self, array_name: str, kernel_entry: nodes.MapEntry,
                                        sdfg_hierarchy: List[SDFG]):
        """Lift a transient array out through each nested SDFG up to the kernel boundary.

        :param sdfg_hierarchy: Nested SDFGs ordered inner->outer.
        """
        outer_sdfg = sdfg_hierarchy.pop(0)
        while sdfg_hierarchy:
            inner_sdfg = outer_sdfg
            outer_sdfg = sdfg_hierarchy.pop(0)
            nsdfg_node = inner_sdfg.parent_nsdfg_node
            nsdfg_parent_state = self.state_of(nsdfg_node)

            old_desc = inner_sdfg.arrays[array_name]
            new_desc = copy.deepcopy(old_desc)
            outer_sdfg.add_datadesc(array_name, new_desc)

            parent_scopes: List[nodes.MapEntry] = []
            current_parent_scope = nsdfg_node
            scope_dict = nsdfg_parent_state.scope_dict()
            while scope_dict[current_parent_scope] is not None and current_parent_scope is not kernel_entry:
                parent_scopes.append(scope_dict[current_parent_scope])
                current_parent_scope = scope_dict[current_parent_scope]

            # ``add_memlet_path`` names and creates the scope connectors along the way. Propagation
            # stays ON: with it off the loop reuses ONE Memlet object for every edge of the path,
            # which validation rejects as a duplicate reference.
            # ``add_memlet_path`` creates the connectors of the scope nodes it passes, but only
            # VERIFIES the two endpoints', so the nested SDFG's own has to exist first.
            nsdfg_node.add_out_connector(array_name)
            nsdfg_parent_state.add_memlet_path(nsdfg_node,
                                               *(nsdfg_parent_state.exit_node(s) for s in parent_scopes),
                                               nsdfg_parent_state.add_access(array_name),
                                               src_conn=array_name,
                                               memlet=Memlet.from_array(array_name, new_desc))

        # Re-mark transient at the outermost SDFG so codegen allocates it instead of expecting a kernel input.
        new_desc.transient = True

    def get_memlet_subset(self, map_chain: List[nodes.MapEntry], node: nodes.Node):
        """Memlet subset for accessing an array given a node's position in
        nested GPU maps.

        Per ``GPU_Device``/``GPU_ThreadBlock`` map in the chain: a node
        strictly inside the map yields the single symbolic map-param index;
        otherwise the full map-dimension range. This makes memlets represent
        per-thread/per-block slices when lifting arrays out of kernels.

        :param map_chain: Nested MapEntry nodes, outermost to innermost.
        :returns: List of ``(start, end, stride)`` tuples per map dimension.
        """
        subset = []
        for next_map in map_chain:
            if next_map.map.schedule not in GPU_HIERARCHY_SCHEDULES:
                continue

            map_parent_state = self.state_of(next_map)
            for param, (start, end, stride) in zip(next_map.map.params, next_map.map.range.ndrange()):

                node_is_map = ((isinstance(node, nodes.MapEntry) and node == next_map)
                               or (isinstance(node, nodes.MapExit) and map_parent_state.exit_node(next_map) == node))
                node_state = self.state_of(node)
                if helpers.contained_in(node_state, node, next_map) and not node_is_map:
                    index = symbolic.symbol(param)
                    subset.append((index, index, 1))
                else:
                    subset.append((start, end, stride))

        return subset

    def update_memlets(self, kernel_entry: nodes.MapEntry, array_name: str, outermost_node: nodes.Node,
                       access_nodes: OrderedSet):
        """Rewrite every memlet of a transient array for correct data movement
        after lifting it out of the kernel.

        Maps enclosing ``outermost_node`` also enclose all access nodes; they
        determine which maps sit strictly above and thus the extra GPU-hierarchy
        dimensions to prepend to each subset.

        :param access_nodes: AccessNodes inside the kernel referencing the array.
        """
        map_entry_chain = self.get_maps_between(kernel_entry, outermost_node)
        params_as_ranges = self.get_memlet_subset(map_entry_chain, outermost_node)

        # edge_bfs visits each edge once, linearly, unlike the old per-path enumeration which
        # was exponential in fan-in/out.
        visited: OrderedSet = OrderedSet()
        for access_node in access_nodes:
            state = self.state_of(access_node)
            incoming = [(edge, True) for edge in state.edge_bfs(access_node, reverse=True)]
            outgoing = [(edge, False) for edge in state.edge_bfs(access_node)]
            for edge, is_incoming in incoming + outgoing:
                if edge in visited:
                    continue
                if edge.data.data == array_name:
                    edge.data.subset = subsets.Range(params_as_ranges + edge.data.subset.ndrange())
                    visited.add(edge)
                elif is_incoming and edge.dst is access_node and edge.data.dst_subset is not None:
                    edge.data.dst_subset = subsets.Range(params_as_ranges + edge.data.dst_subset.ndrange())
                    visited.add(edge)
                elif not is_incoming and edge.src is access_node and edge.data.src_subset is not None:
                    edge.data.src_subset = subsets.Range(params_as_ranges + edge.data.src_subset.ndrange())
                    visited.add(edge)

    def get_new_shape_info(self, array_desc: dt.Array, map_exit_chain: List[nodes.MapEntry]):
        """New shape, strides, total size and offsets for a transient array lifted out of a kernel.

        Each GPU map prepends dimensions for per-thread disjoint slices, e.g. ``gpu_A`` of shape
        ``[64]`` under ``map[0:128, 0:32]`` becomes ``[128, 32, 64]`` (indexed ``gpu_A[x, y, :]``).
        The prepended dimensions are made the slowest-varying ones while the original dimensions
        keep their own layout, so a packed-Fortran array stays packed-Fortran on its own axes.

        :param map_exit_chain: MapEntry nodes between array and kernel exit.
        :returns: ``(new_shape, new_strides, new_total_size, new_offsets)``.
        :raises NotImplementedError: The array is neither packed-C nor packed-Fortran.
        """
        if array_desc.is_packed_c_strides():
            inner_order = list(reversed(range(len(array_desc.shape))))
        elif array_desc.is_packed_fortran_strides():
            inner_order = list(range(len(array_desc.shape)))
        else:
            raise NotImplementedError(f'Cannot lift {array_desc}: only packed C or Fortran strides are supported.')

        extended_size = []
        new_offsets = list(array_desc.offset)
        for next_map in map_exit_chain:
            if next_map.map.schedule not in GPU_HIERARCHY_SCHEDULES:
                continue
            extended_size = [
                tile_extent(mx, mn)
                for mx, mn in zip(next_map.map.range.max_element(), next_map.map.range.min_element())
            ] + extended_size
            new_offsets = [0 for _ in next_map.map.params] + new_offsets

        prepended = len(extended_size)
        # ``strides_from_layout`` takes the dimensions innermost-first: the original axes in their
        # own order, then the prepended ones outermost-last so they end up slowest-varying.
        layout = [d + prepended for d in inner_order] + list(reversed(range(prepended)))

        lifted = array_desc.clone()
        lifted.set_shape(extended_size + list(array_desc.shape))
        new_strides, new_total_size = lifted.strides_from_layout(*layout)
        return list(lifted.shape), list(new_strides), new_total_size, new_offsets

    def replace_array_name(self, sdfgs: OrderedSet, old_name: str, new_name: str, array_desc: dt.Array):
        """Rename an array across ``sdfgs`` -- descriptor, memlets, connectors and access nodes.

        ``SDFG.replace`` reaches the descriptor, the access nodes and the memlets, but not the
        ``IN_``/``OUT_`` scope connectors named after the data, which are renamed here.

        :param sdfgs: SDFGs declaring the descriptor.
        :param old_name: Name to rename from.
        :param new_name: Name to rename to.
        :param array_desc: Descriptor to re-register under ``new_name``.
        """
        renamed = {f"OUT_{old_name}": f"OUT_{new_name}", f"IN_{old_name}": f"IN_{new_name}"}
        for sdfg in sdfgs:
            sdfg.remove_data(old_name, False)
            sdfg.add_datadesc(new_name, array_desc)
            sdfg.replace(old_name, new_name)

            for state in sdfg.states():
                for edge in state.edges():
                    if edge.src_conn in renamed:
                        edge.src.remove_out_connector(edge.src_conn)
                        edge.src_conn = renamed[edge.src_conn]
                        edge.src.add_out_connector(edge.src_conn)
                    if edge.dst_conn in renamed:
                        edge.dst.remove_in_connector(edge.dst_conn)
                        edge.dst_conn = renamed[edge.dst_conn]
                        edge.dst.add_in_connector(edge.dst_conn)

    def update_symbols(self, map_entry_chain: List[nodes.MapEntry], top_sdfg: SDFG):
        """Propagate GPU-map symbols (e.g. map indices) into every nested SDFG
        under ``top_sdfg`` so lifted memlets referencing them stay valid.

        :param map_entry_chain: GPU MapEntry nodes whose symbols are relevant.
        """
        all_symbols = OrderedSet()
        for next_map in map_entry_chain:
            if next_map.map.schedule not in GPU_HIERARCHY_SCHEDULES:
                continue
            all_symbols = all_symbols | next_map.used_symbols_within_scope(self.state_of(next_map))

        for sdfg in top_sdfg.all_sdfgs_recursive():
            nsdfg_node = sdfg.parent_nsdfg_node
            if nsdfg_node is None:
                continue

            for sym in all_symbols:
                name = str(sym)
                if name not in sdfg.symbols:
                    sdfg.add_symbol(name, symbolic.DEFAULT_SYMBOL_TYPE)
                if name not in nsdfg_node.symbol_mapping:
                    # The resolved instance, never a fresh ``symbol(name)``: symbol identity is
                    # name-based, so a re-mint silently aliases another dtype's same-named symbol.
                    nsdfg_node.symbol_mapping[name] = sym

    @staticmethod
    def binds(nsdfg_node: Optional[nodes.NestedSDFG], array_name: str) -> bool:
        """Whether ``nsdfg_node`` carries ``array_name`` across its boundary.

        :param nsdfg_node: Nested SDFG node to test; ``None`` for a top-level SDFG.
        :param array_name: Descriptor name to look for.
        :returns: True if the name is one of the node's connectors.
        """
        return nsdfg_node is not None and (array_name in nsdfg_node.in_connectors
                                           or array_name in nsdfg_node.out_connectors)

    def collect_array_descriptor_usage(self, map_entry: nodes.MapEntry,
                                       array_name: str) -> List[Tuple[dt.Array, SDFG, OrderedSet, OrderedSet]]:
        """Track usage of a transient array across nested SDFGs within a map scope.

        "Same array" means same name connected via memlets -- several
        ``dt.Array`` descriptor objects may exist across SDFGs for one
        logical array.

        :returns: ``(descriptor, outermost SDFG, all involved SDFGs,
            all referencing AccessNodes)`` tuples.
        """
        access_nodes_info: List[Tuple[nodes.AccessNode, SDFGState,
                                      SDFG]] = self.get_access_nodes_within_map(map_entry, array_name)

        last_sdfg: SDFG = self.state_of(map_entry).sdfg

        result: List[Tuple[dt.Array, SDFG, OrderedSet, OrderedSet]] = []
        visited_sdfgs: OrderedSet[SDFG] = OrderedSet()

        for access_node, state, sdfg in access_nodes_info:

            if sdfg in visited_sdfgs:
                continue

            # Any one descriptor copy suffices -- we only read metadata from it.
            array_desc = access_node.desc(state)

            sdfg_set: OrderedSet[SDFG] = OrderedSet()
            access_nodes_set: OrderedSet[nodes.AccessNode] = OrderedSet()
            access_nodes_set.add(access_node)

            # Upwards while the name keeps crossing a connector: that is how far one descriptor reaches.
            outermost_sdfg = sdfg
            while True:
                sdfg_set.add(outermost_sdfg)
                if outermost_sdfg is last_sdfg or not self.binds(outermost_sdfg.parent_nsdfg_node, array_name):
                    break
                outermost_sdfg = outermost_sdfg.parent_sdfg

            queue = [sdfg]
            while queue:
                for nested in (n for st in queue.pop(0).states() for n in st.nodes()
                               if isinstance(n, nodes.NestedSDFG) and self.binds(n, array_name)):
                    queue.append(nested.sdfg)
                    sdfg_set.add(nested.sdfg)

            access_nodes_set.update(node for sub in sdfg_set for sub_state in sub.states()
                                    for node in sub_state.data_nodes() if node.data == array_name)

            visited_sdfgs.update(sdfg_set)

            result.append((array_desc, outermost_sdfg, sdfg_set, access_nodes_set))

        return result

    def new_name_required(self, map_entry: nodes.MapEntry, array_name: str,
                          sdfg_defined: OrderedSet) -> Tuple[bool, str]:
        """Detect whether ``array_name`` collides with a different descriptor
        in an SDFG outside ``sdfg_defined``, and suggest a free name if so.

        :param map_entry: Kernel map bounding the search.
        :param array_name: Name to test.
        :param sdfg_defined: SDFGs where the descriptor is defined.
        :returns: ``(rename_required, name)`` -- ``name`` is the original when
            no rename is needed, else a fresh suggestion.
        """
        map_parent_sdfg = self.state_of(map_entry).sdfg
        taken_names = OrderedSet()

        for sdfg in map_parent_sdfg.all_sdfgs_recursive():

            nsdfg_node = sdfg.parent_nsdfg_node
            state = self.state_of(nsdfg_node) if nsdfg_node else None

            if not ((nsdfg_node and state and helpers.contained_in(state, nsdfg_node, map_entry))
                    or sdfg is map_parent_sdfg):
                continue

            # Taken names exclude SDFGs that already define this descriptor -- renaming only avoids real conflicts.
            if sdfg not in sdfg_defined:
                taken_names.update(sdfg.arrays.keys())
                taken_names.update(sdfg.used_symbols(True))

        new_name = utils.find_new_name(array_name, taken_names)
        return new_name != array_name, new_name

    def get_access_nodes_within_map(self, map_entry: nodes.MapEntry,
                                    data_name: str) -> List[Tuple[nodes.AccessNode, SDFGState, SDFG]]:
        """All AccessNodes for ``data_name`` inside ``map_entry``'s scope.

        :returns: ``(AccessNode, SDFGState, parent SDFG)`` tuples.
        """
        starting_sdfg = self.state_of(map_entry).sdfg
        matching_access_nodes = []

        for node, parent_state in starting_sdfg.all_nodes_recursive():

            if (isinstance(node, nodes.AccessNode) and node.data == data_name
                    and helpers.contained_in(parent_state, node, map_entry)):

                parent_sdfg = self.state_of(node).sdfg
                matching_access_nodes.append((node, parent_state, parent_sdfg))

        return matching_access_nodes

    def get_maps_between(self, stop_map_entry: nodes.MapEntry, node: nodes.Node) -> List[nodes.MapEntry]:
        """Map scopes enclosing ``node`` up to and including ``stop_map_entry``, innermost first.

        Assumes ``node`` is contained (directly or via a nested SDFG) within
        ``stop_map_entry``'s scope.

        :param stop_map_entry: Outermost map to report.
        :param node: Node whose enclosing scopes are walked.
        :returns: The enclosing ``MapEntry`` nodes, inner to outer.
        :raises ValueError: ``node`` is not inside ``stop_map_entry``'s scope.
        """
        stop_state = self.state_of(stop_map_entry)
        stop_exit = stop_state.exit_node(stop_map_entry)

        entries: List[nodes.MapEntry] = []

        current_state = self.state_of(node)
        parent_info = helpers.get_parent_map(current_state, node)

        while True:
            if parent_info is None:
                raise ValueError("Expected node to be in scope of stop_map_entry, but no parent map was found.")

            entry, state = parent_info
            entries.append(entry)
            if state.exit_node(entry) == stop_exit:
                break

            parent_info = helpers.get_parent_map(state, entry)

        return entries

    def get_nearest_access_node(self, access_nodes: List[nodes.AccessNode], node: nodes.Node) -> nodes.AccessNode:
        """Closest AccessNode to ``node`` by graph distance within the same
        state (direction-agnostic BFS).

        :param access_nodes: Candidate AccessNodes.
        :param node: Node to start the search from.
        :returns: The closest AccessNode by edges traversed.
        :raises RuntimeError: No candidate is connected to ``node`` in its state.
        """
        state = self.state_of(node)

        visited = OrderedSet()
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
