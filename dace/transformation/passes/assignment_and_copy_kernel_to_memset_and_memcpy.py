# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import warnings
import dace
import copy
from dace import Tuple, properties
from dace.sdfg.graph import MultiConnectorEdge
from dace.transformation import pass_pipeline as ppl, transformation
from dace.libraries.standard.nodes.copy_node import CopyLibraryNode
from dace.libraries.standard.nodes.memset_node import MemsetLibraryNode
from typing import Dict, List, Set


@properties.make_properties
@transformation.explicit_cf_compatible
class AssignmentAndCopyKernelToMemsetAndMemcpy(ppl.Pass):
    overapproximate_first_dimension = properties.Property(
        dtype=bool,
        default=True,
        desc=
        "If True, the first dimension of the map is overapproximated to be contiguous, even if it is not. This is useful for some cases where the first dimension is always contiguous, but the map range is not.",
    )
    rmid = 0

    def modifies(self) -> ppl.Modifies:
        return ppl.Modeifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def _get_edges_from_path(self, state: dace.SDFGState, node_path: List[dace.nodes.Node]) -> List[MultiConnectorEdge]:
        if len(node_path) == 1:
            return []
        edges = []
        for i in range(len(node_path) - 1):
            src = node_path[i]
            dst = node_path[i + 1]
            oes = {oe for oe in state.out_edges(src) if oe.dst == dst}
            assert len(oes) == 1
            oe = oes.pop()
            edges.append(oe)
        return edges

    def _detect_contiguous_memcpy_paths(self, state: dace.SDFGState, node: dace.nodes.MapEntry):
        paths = list()

        # If map range is not contigous, we can't do contiguous copy detection
        step_equal_one = True
        for (b, e, s) in node.map.range:
            if s != 1:
                step_equal_one = False
                break

        # Non-zero step in map range
        if not step_equal_one:
            return paths

        assert node in state.nodes()
        assert state.exit_node(node) in state.nodes()
        path_candidates = [
            self._get_edges_from_path(state, p)
            for p in state.all_simple_paths(node, state.exit_node(node), as_edges=False)
        ]
        # AN1 -> MapEntry -> Tasklet -> MapExit -> AN2
        # Need to get AN1 and AN2
        for path_candidate in path_candidates:
            if len(path_candidate) != 2:
                continue
            # Gen AN1 by replacing the name of the OUT connector
            if path_candidate[0].dst_conn is None or (not path_candidate[0].src_conn.startswith("OUT_")):
                continue
            ie = next(
                state.in_edges_by_connector(path_candidate[0].src, path_candidate[0].src_conn.replace("OUT_", "IN_")))
            oe = next(
                state.out_edges_by_connector(path_candidate[-1].dst, path_candidate[-1].dst_conn.replace("IN_",
                                                                                                         "OUT_")))
            tasklet = path_candidate[1].src

            # Tasklet in the middle
            if not isinstance(tasklet, dace.nodes.Tasklet):
                continue
            if len(tasklet.in_connectors) != 1 or len(tasklet.out_connectors) != 1:
                continue
            # Output Access Node
            if not isinstance(oe.dst, dace.nodes.AccessNode):
                continue
            # Input Access Node
            if not isinstance(ie.src, dace.nodes.AccessNode):
                continue

            in_conn = next(iter(tasklet.in_connectors))
            out_conn = next(iter(tasklet.out_connectors))
            if tasklet.language == dace.Language.Python:
                tasklet_code_str = tasklet.code.as_string
                if f"{out_conn} = {in_conn}" != tasklet_code_str:
                    continue
            elif tasklet.language == dace.Language.CPP:
                tasklet_code_str = tasklet.code.as_string
                if f"{out_conn} = {in_conn};" != tasklet_code_str:
                    continue
            else:
                continue

            paths.append([ie] + path_candidate + [oe])

        return paths

    def _detect_contiguous_memset_paths(self, state: dace.SDFGState, node: dace.nodes.MapEntry):
        # All tasklets within the map
        paths = list()

        # If map range is not contigous, we can't do contiguous copy detection
        step_equal_one = True
        for (b, e, s) in node.map.range:
            if s != 1:
                step_equal_one = False
                break

        # Non-zero step in map range
        if not step_equal_one:
            return paths

        assert node in state.nodes()
        assert state.exit_node(node) in state.nodes()
        path_candidates = [
            self._get_edges_from_path(state, p)
            for p in state.all_simple_paths(node, state.exit_node(node), as_edges=False)
        ]
        # MapEntry -> Tasklet -> MapExit -> AN2
        # Need to get AN2 only
        for path_candidate in path_candidates:
            if len(path_candidate) != 2:
                continue

            ie = path_candidate[0]
            if ie.src_conn is not None or ie.dst_conn is not None or ie.data.data is not None:
                continue

            oe = next(
                state.out_edges_by_connector(path_candidate[-1].dst, path_candidate[-1].dst_conn.replace("IN_",
                                                                                                         "OUT_")))
            tasklet = path_candidate[1].src

            # Tasklet in the middle
            if not isinstance(tasklet, dace.nodes.Tasklet):
                continue
            if len(tasklet.in_connectors) != 0 or len(tasklet.out_connectors) != 1:
                continue
            # Output Access Node
            if not isinstance(oe.dst, dace.nodes.AccessNode):
                continue

            out_conn = next(iter(tasklet.out_connectors))
            if tasklet.language == dace.Language.Python:
                tasklet_code_str = tasklet.code.as_string
                if f"{out_conn} = 0" != tasklet_code_str:
                    continue
            elif tasklet.language == dace.Language.CPP:
                tasklet_code_str = tasklet.code.as_string
                if f"{out_conn} = 0;" != tasklet_code_str:
                    continue
            else:
                continue

            paths.append(path_candidate + [oe])

        return paths

    def _get_num_tasklets_within_map(self, state: dace.SDFGState, node: dace.nodes.MapEntry):
        assert node in state.nodes(), f"Map entry {node} not in state {state}"
        assert isinstance(node, dace.nodes.MapEntry), f"Node {node} is not a MapEntry"
        assert state.exit_node(node) in state.nodes(), f"Map exit {state.exit_node(node)} not in state {state}"
        n = {n for n in state.all_nodes_between(node, state.exit_node(node)) if isinstance(n, dace.nodes.Tasklet)}
        return len(n)

    # [(b,e,s), ...] is the range
    # Need to find out if it is contiguous, this means:
    # The stride==1 dimension is the first one and strides are ascending where the storage is packed
    # or stride==1 dimension is the last one and strides are descending where the storage is packed
    def _get_packed_fortran_strides(self, array: dace.data.Array) -> List[int]:
        accum = 1
        strides = []
        for shape in array.shape:
            strides.append(accum)
            accum *= shape
        return tuple(strides)

    def _get_packed_c_strides(self, array: dace.data.Array) -> List[int]:
        accum = 1
        strides = []
        # Same as Fortran order if shape is inversed
        for shape in reversed(array.shape):
            strides.append(accum)
            accum *= shape
        return tuple(list(reversed(strides)))

    def _is_packed_fortran_strides(self, array: dace.data.Array, verbose=True) -> bool:
        strides = self._get_packed_fortran_strides(array)
        return tuple(strides) == tuple(array.strides)

    def _is_packed_c_strides(self, array: dace.data.Array, verbose=True) -> bool:
        strides = self._get_packed_c_strides(array)
        return tuple(strides) == tuple(array.strides)

    # let's say arrays strides are [1, N, M*N]
    # then the expression we have needs to cover whole first dimension X-1 if it is not 1 in dimension X
    def _is_contig_subset(self, range_list: List[Tuple], array: dace.data.Array) -> bool:
        if self._is_packed_fortran_strides(array):
            range_list = range_list
            expr_lens = [((e + 1) - b) for (b, e, s) in range_list]
        elif self._is_packed_c_strides(array):
            range_list = list(reversed(range_list))
            expr_lens = [((e + 1) - b) for (b, e, s) in reversed(range_list)]
        else:
            raise ValueError(
                f"Array {array} does not have Fortran or C strides, has strides: {array.strides}, expected Fortran: {self._get_packed_fortran_strides(array)} or C: {self._get_packed_c_strides(array)}"
            )

        expr_lens = [((e + 1) - b) for (b, e, s) in range_list]
        for i, expr_len in enumerate(expr_lens):
            # It can be that this triggers an error because trurth value is rational, then we assume it is less than the shape
            try:
                if expr_len < array.shape[i]:
                    # All next ones must be 1
                    for j in range(i + 1, len(expr_lens)):
                        if expr_lens[j] != 1:
                            return False
                    return True
            except TypeError:
                # All next ones must be 1
                for j in range(i + 1, len(expr_lens)):
                    if expr_lens[j] != 1:
                        return False
                return True

        return True

    def _get_write_begin_and_length(self,
                                    state: dace.SDFGState,
                                    map_entry: dace.nodes.MapEntry,
                                    tasklet: dace.nodes.Tasklet,
                                    verbose=True):
        range_list = {
            dace.symbolic.symbol(p): (b, e, s)
            for (p, (b, e, s)) in zip(map_entry.map.params, map_entry.map.range)
        }

        in_edge = state.in_edges(tasklet)[0]
        out_edge = state.out_edges(tasklet)[0]

        if in_edge.data.data is not None:
            in_data_range = [(b, e, s) for (b, e, s) in in_edge.data.subset]
        out_data_range = [(b, e, s) for (b, e, s) in out_edge.data.subset]

        new_in_data_range = []
        new_out_data_range = []

        if in_edge.data.data is not None:
            for (b, e, s) in in_data_range:
                nb: dace.symbolic.SymExpr = b
                ne: dace.symbolic.SymExpr = e
                ns: dace.symbolic.SymExpr = s
                for (p, (b2, e2, s2)) in range_list.items():
                    nb = nb.subs(p, b2)
                    ne = ne.subs(p, e2)
                    assert ns == 1 and s2 == 1, "Only step of 1 is supported for memcpy detection"
                new_in_data_range.append((nb, ne, ns))

            # If we overapproximate the first dimension, we assume it is contiguous
            if self.overapproximate_first_dimension:
                new_in_data_range[0] = ((0, state.sdfg.arrays[in_edge.data.data].shape[0] - 1, 1))
        for (b, e, s) in out_data_range:
            nb: dace.symbolic.SymExpr = b
            ne: dace.symbolic.SymExpr = e
            ns: dace.symbolic.SymExpr = s
            for (p, (b2, e2, s2)) in range_list.items():
                nb = nb.subs(p, b2)
                ne = ne.subs(p, e2)
                assert ns == 1 and s2 == 1, "Only step of 1 is supported for memcpy/memset detection"
            new_out_data_range.append((nb, ne, ns))
        # If we overapproximate the first dimension, we assume it is contiguous
        if self.overapproximate_first_dimension:
            new_out_data_range[0] = ((0, state.sdfg.arrays[out_edge.data.data].shape[0] - 1, 1))

        new_in_data_subset = dace.subsets.Range(new_in_data_range) if in_edge.data.data is not None else None
        new_out_data_subset = dace.subsets.Range(new_out_data_range) if out_edge.data.data is not None else None

        if in_edge.data.data is not None:
            contig_subset = self._is_contig_subset(new_in_data_range, state.sdfg.arrays[in_edge.data.data])
            if not contig_subset:
                warnings.warn(f"Input array {in_edge.data.data} is not contiguous, cannot remove memcpy/memset.",
                              UserWarning)
                return None, None, None

        if out_edge.data.data is not None:
            contig_subset = self._is_contig_subset(new_out_data_range, state.sdfg.arrays[out_edge.data.data])
            if not contig_subset:
                warnings.warn(f"Output array {out_edge.data.data} is not contiguous, cannot remove memcpy/memset.",
                              UserWarning)
                return None, None, None

        if in_edge.data.data is not None:
            in_begin_exprs = [b for (b, e, s) in new_in_data_range]
            in_length_exprs = [(e + 1) - b for (b, e, s) in new_in_data_range]
        out_begin_exprs = [b for (b, e, s) in new_out_data_range]
        out_length_exprs = [(e + 1) - b for (b, e, s) in new_out_data_range]

        if in_edge.data.data is not None:
            in_begin_collapsed = dace.symbolic.SymExpr(1)
            in_length_collapsed = dace.symbolic.SymExpr(1)
        out_begin_collapsed = dace.symbolic.SymExpr(1)
        out_length_collapsed = dace.symbolic.SymExpr(1)

        # We ensured the subset is contiguous, so we can get the length by multiplying each dimension's length
        if in_edge.data.data is not None:
            for i, b in enumerate(in_begin_exprs):
                in_begin_collapsed *= b

            for i, l in enumerate(in_length_exprs):
                in_length_collapsed *= l

        for i, b in enumerate(out_begin_exprs):
            out_begin_collapsed *= b

        for i, l in enumerate(out_length_exprs):
            out_length_collapsed *= l

        if in_edge.data.data is None:
            in_begin_collapsed = None
            in_length_collapsed = None

        if in_length_collapsed is not None:
            assert in_length_collapsed == out_length_collapsed, f"Input and output lengths must be equal for memcpy detection {in_length_collapsed} != {out_length_collapsed}"

        return new_in_data_range, new_out_data_range, out_length_collapsed

    def remove_memcpy_from_kernel(self, state: dace.SDFGState, node: dace.nodes.MapEntry, verbose=True):
        memcpy_paths = self._detect_contiguous_memcpy_paths(state, node)
        rmed_count = 0
        joined_edges = set()
        for memcpy_path in memcpy_paths:
            for e in memcpy_path:
                joined_edges.add(e)

        for memcpy_path in memcpy_paths:
            src_access_node = memcpy_path[0].src
            map_entry = memcpy_path[0].dst
            tasklet = memcpy_path[1].dst
            map_exit = memcpy_path[2].dst
            dst_access_node = memcpy_path[3].dst
            if src_access_node not in state.nodes() or map_entry not in state.nodes() or tasklet not in state.nodes(
            ) or map_exit not in state.nodes() or dst_access_node not in state.nodes():
                if verbose:
                    warnings.warn(
                        f"Map entry, exit or tasklet not in state: {map_entry} ({map_entry in state.nodes()}), {map_exit} ({map_exit in state.nodes()}), {tasklet} ({tasklet in state.nodes()}). Skipping.",
                        UserWarning)
                assert False, f"Map entry, exit or tasklet not in state: {map_entry} ({map_entry in state.nodes()}), {map_exit} ({map_exit in state.nodes()}), {tasklet} ({tasklet in state.nodes()})."

            # If src and dst types are not the same, we can't do memcpy
            src_desc = state.sdfg.arrays[src_access_node.data]
            dst_desc = state.sdfg.arrays[dst_access_node.data]
            if src_desc.dtype != dst_desc.dtype:
                if verbose:
                    warnings.warn(
                        f"Source and destination types do not match for memcpy removal: {src_desc.dtype} != {dst_desc.dtype}. Skipping.",
                        UserWarning)
                continue
            if src_desc.storage != dst_desc.storage:
                if verbose:
                    warnings.warn(
                        f"Source and destination storage types do not match for memcpy removal: {src_desc.storage} != {dst_desc.storage}. Skipping.",
                        UserWarning)
                continue

            # To calculate the total range,
            # Take input subset of tasklet replace expression with map range
            # For now, we will just use the original range
            # Needs to be before removing the path because it requires edges of the tasklet
            begin_subset, exit_subset, copy_length = self._get_write_begin_and_length(
                state, map_entry, tasklet, verbose)

            # We can now remove the memcpy path
            dyn_inputs = {ie.dst_conn for ie in state.in_edges(map_entry) if not ie.dst_conn.startswith("IN_")}
            in_edges = state.in_edges(map_entry)

            # If src / dst not in the graph anymore, add new ones
            if src_access_node not in state.nodes():
                new_src_access_node = state.add_access(src_access_node.data)
            else:
                new_src_access_node = src_access_node
            if dst_access_node not in state.nodes():
                new_dst_access_node = state.add_access(dst_access_node.data)
            else:
                new_dst_access_node = dst_access_node

            # Add a new memcpy tasklet
            tasklet = CopyLibraryNode(
                name=f"copyLib_{new_src_access_node.data}_{new_dst_access_node.data}_{self.rmid}", )
            state.add_node(tasklet)
            self.rmid += 1
            state.add_edge(new_src_access_node, None, tasklet, "_in",
                           dace.memlet.Memlet(subset=dace.subsets.Range(begin_subset), data=new_src_access_node.data))
            state.add_edge(tasklet, "_out", new_dst_access_node, None,
                           dace.memlet.Memlet(subset=dace.subsets.Range(exit_subset), data=new_dst_access_node.data))
            tasklet.add_in_connector("_in")
            tasklet.add_out_connector("_out")
            for ie in in_edges:
                if not ie.dst_conn.startswith("IN_"):
                    _an = state.add_access(ie.data.data)
                    state.add_edge(_an, None, tasklet, ie.dst_conn, copy.deepcopy(ie.data))
                    tasklet.add_in_connector(ie.dst_conn)

            rmed_count += 1

        for i, e in enumerate(joined_edges):
            assert e in state.edges(), f"{e} not in {state.edges()}"
            state.remove_edge(e)
            if e.src_conn is not None:
                e.src.remove_out_connector(e.src_conn)
            if e.dst_conn is not None:
                e.dst.remove_in_connector(e.dst_conn)

        for n in state.nodes():
            if state.degree(n) == 0:
                state.remove_node(n)

        return rmed_count

    def remove_memset_from_kernel(self, state: dace.SDFGState, node: dace.nodes.MapEntry, verbose=True):
        memset_paths = self._detect_contiguous_memset_paths(state, node)

        joined_edges = set()
        for memcpy_path in memset_paths:
            for e in memcpy_path:
                joined_edges.add(e)

        rmed_count = 0
        for memset_path in memset_paths:
            map_entry = memset_path[0].src
            tasklet = memset_path[0].dst
            map_exit = memset_path[1].dst
            dst_access_node = memset_path[2].dst
            assert isinstance(map_entry, dace.nodes.MapEntry), f"Map entry {map_entry} is not a MapEntry"
            assert isinstance(tasklet, dace.nodes.Tasklet), f"Tasklet {tasklet} is not a Tasklet"
            assert isinstance(map_exit, dace.nodes.MapExit), f"Map exit {map_exit} is not a MapExit"
            assert isinstance(dst_access_node,
                              dace.nodes.AccessNode), f"Destination access node {dst_access_node} is not an AccessNode"

            # To calculate the total range,
            # Take input subset of tasklet replace expression with map range
            # For now, we will just use the original range
            # Needs to be done before removing the memset path
            if map_entry not in state.nodes() or map_exit not in state.nodes() or tasklet not in state.nodes():
                if verbose:
                    warnings.warn(
                        f"Map entry, exit or tasklet not in state: {map_entry} ({map_entry in state.nodes()}), {map_exit} ({map_exit in state.nodes()}), {tasklet} ({tasklet in state.nodes()}). Skipping.",
                        UserWarning)
                continue
            current_tasklets = {
                n
                for n in state.all_nodes_between(map_entry, map_exit) if isinstance(n, dace.nodes.Tasklet)
            }
            if len(memset_paths) != len(current_tasklets):
                if verbose:
                    warnings.warn(
                        f"Number of memset paths {len(memset_paths)} does not match number of tasklets in map {len({n for n in state.all_nodes_between(map_entry, map_exit) if isinstance(n, dace.nodes.Tasklet)})}. Was removed before probably.",
                        UserWarning)
                if tasklet not in current_tasklets:
                    if verbose:
                        warnings.warn(f"Tasklet {tasklet} is not in the current tasklets, skipping.", UserWarning)
                    continue

            begin_subset, exit_subset, copy_length = self._get_write_begin_and_length(state, map_entry, tasklet)

            if begin_subset is None or exit_subset is None or copy_length is None:
                if verbose:
                    warnings.warn(
                        f"Could not determine begin or exit subset or copy length for memset removal (or they are not contiguous) in map {map_entry.map}({map_entry.map.label}). Skipping.",
                        UserWarning)
                continue

            # We can now remove the memset path
            in_edges = state.in_edges(map_entry)

            # Add a new memset tasklet
            tasklet = MemsetLibraryNode(name=f"memsetLib_{dst_access_node.data}_{self.rmid}", )
            tasklet.add_out_connector("_out")
            state.add_node(tasklet)
            self.rmid += 1
            state.add_edge(tasklet, "_out", dst_access_node, None,
                           dace.memlet.Memlet(subset=dace.subsets.Range(exit_subset), data=dst_access_node.data))
            # Redirect all dynamic input connectors
            for ie in in_edges:
                if not ie.dst_conn.startswith("IN_"):
                    _an1 = state.add_access(ie.data.data)
                    state.add_edge(_an1, None, tasklet, ie.dst_conn, copy.deepcopy(ie.data))
                    tasklet.add_in_connector(ie.dst_conn)

            rmed_count += 1

        for i, e in enumerate(joined_edges):
            assert e in state.edges(), f"{e} not in {state.edges()}"
            state.remove_edge(e)
            if e.src_conn is not None:
                e.src.remove_out_connector(e.src_conn)
            if e.dst_conn is not None:
                e.dst.remove_in_connector(e.dst_conn)

        for n in state.nodes():
            if state.degree(n) == 0:
                state.remove_node(n)

        return rmed_count

    def apply_pass(self, sdfg: dace.SDFG, pipeline_res: Dict) -> Dict[int, Dict[dace.SDFGState, Set[dace.SDFGState]]]:
        map_entries = set()

        for n, g in sdfg.all_nodes_recursive():
            if isinstance(n, dace.nodes.MapEntry):
                map_entries.add((n, g))

        rmed_memcpies = dict()
        rmed_memsets = dict()

        for (node, state) in map_entries:
            sdfg.validate()
            assert node in state.nodes(), f"Map entry {node} not in state {state}"
            _e = state.exit_node(node)
            assert state.exit_node(node) in state.nodes(), f"Map exit {state.exit_node(node)} not in state {state}"

            if self._get_num_tasklets_within_map(state, node) == 0:
                continue

            rmed_memcpy = self.remove_memcpy_from_kernel(state, node)
            sdfg.validate()

            # If the map is only used for 1 memcpy, then it might have been already removed
            if node in state.nodes():
                rmed_memset = self.remove_memset_from_kernel(state, node)
            else:
                rmed_memset = 0
            sdfg.validate()

            assert node not in rmed_memsets
            assert node not in rmed_memcpies
            rmed_memcpies[node] = rmed_memcpy
            rmed_memsets[node] = rmed_memset

        num_rmed_memcpies = sum(rmed_memcpies.values())
        num_rmed_memsets = sum(rmed_memsets.values())

        return num_rmed_memcpies + num_rmed_memsets
