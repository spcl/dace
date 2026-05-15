# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lift contiguous zero-assignments and element-wise copies out of maps into Memset / Copy library nodes."""
import warnings
import dace
from dace import properties
from dace.memlet import Memlet
from dace.sdfg.graph import Edge, MultiConnectorEdge
from dace.transformation import helpers, pass_pipeline as ppl, transformation
from dace.libraries.standard.nodes.copy_node import CopyLibraryNode
from dace.libraries.standard.nodes.memset_node import MemsetLibraryNode
from typing import Dict, Iterable, List, Optional, Tuple


@properties.make_properties
@transformation.explicit_cf_compatible
class AssignmentAndCopyKernelToMemsetAndMemcpy(ppl.Pass):
    """Lift contiguous zero-assignments and element-wise copies out of maps.

    Walks every map in the SDFG, identifies data paths that perform a
    constant-zero write or a direct element-wise copy over a contiguous
    region, and replaces them with the corresponding library node. When a
    map mixes compute paths with pure data-movement paths, the map is
    fissioned first so that the data-movement part can be extracted
    independently.
    """

    overapproximate_first_dimension = properties.Property(
        dtype=bool,
        default=False,
        desc="If True, overapproximate the first dimension as contiguous over its stride-one extent, "
        "even if the map range isn't. Useful when the dimension is known to be contiguous in memory.",
    )
    node_label_whitelist = properties.ListProperty(
        element_type=str,
        default=[],
        allow_none=False,
        desc="If non-empty, only map entries whose label appears in this list "
        "are considered for lifting. An empty list means all maps are eligible.",
    )

    rmid = 0

    def __init__(self,
                 overapproximate_first_dimensions: bool = False,
                 node_label_whitelist: Optional[List[str]] = None):
        self.overapproximate_first_dimension = overapproximate_first_dimensions
        self.node_label_whitelist = node_label_whitelist if node_label_whitelist is not None else []

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def _get_edges_from_path(self, state: dace.SDFGState, node_path: List[dace.nodes.Node]) -> List[MultiConnectorEdge]:
        if len(node_path) == 1:
            return []
        edges = []
        for i in range(len(node_path) - 1):
            src = node_path[i]
            dst = node_path[i + 1]
            oes = {oe for oe in state.out_edges(src) if oe.dst == dst}
            if len(oes) != 1:
                # Ambiguous or missing edge between consecutive path nodes.
                return []
            oe = oes.pop()
            edges.append(oe)
        return edges

    @staticmethod
    def _subset_param_order(subset, map_params: List[str]) -> List[str]:
        """Per-dimension list of which map parameter the subset uses.

        Dimensions that don't reference any map param drop out. Used to compare
        in- vs. out-subset access orderings, see :meth:`_in_out_subsets_are_pure_copy`.

        :param subset: Memlet subset to inspect.
        :param map_params: Names of the enclosing map's parameters.
        :returns: One map-parameter name per dimension that references exactly one.
        """
        param_set = set(map_params)
        order = []
        for (b, e, _s) in subset:
            # Treat a [b, e] dim as using a map param iff exactly one map
            # param appears anywhere in ``b`` or ``e``. Per-iteration accesses
            # encode as (p, p, 1); broadcast slices may encode wider but
            # still reference a single param.
            free = set()
            for expr in (b, e):
                free |= {str(s) for s in dace.symbolic.symlist(expr).keys()} & param_set
            if len(free) == 1:
                order.append(next(iter(free)))
        return order

    def _in_out_subsets_are_pure_copy(self, in_subset, out_subset, map_params: List[str]) -> bool:
        """Reject permutations (e.g. transpose) but accept copies and broadcasts.

        ``_out = _in`` is identical for a copy, a broadcast and a transpose;
        only the first two lower safely to ``cudaMemcpyAsync``. A map
        parameter appearing in both in- and out-subsets must keep the same
        relative order -- transpose swaps it, copy/broadcast preserve it.

        :param in_subset: Subset of the tasklet's input memlet.
        :param out_subset: Subset of the tasklet's output memlet.
        :param map_params: Names of the enclosing map's parameters.
        :returns: True iff the in/out ordering is a copy or broadcast, not a permutation.
        """
        in_order = self._subset_param_order(in_subset, map_params)
        out_order = self._subset_param_order(out_subset, map_params)
        shared = set(in_order) & set(out_order)
        if not shared:
            return True
        return [p for p in in_order if p in shared] == [p for p in out_order if p in shared]

    def _detect_contiguous_paths(self, state: dace.SDFGState, node: dace.nodes.MapEntry,
                                 is_memset: bool) -> List[List[MultiConnectorEdge]]:
        """Find ``MapEntry -> tasklet -> MapExit`` data-movement paths under a map.

        Matches a tasklet that is a pure element-wise copy (``is_memset=False``)
        or a constant-zero write (``is_memset=True``).

        :param state: State containing the map.
        :param node: Map entry of the kernel to scan.
        :param is_memset: Match constant-zero writes when True, copies when False.
        :returns: One edge list per matched path; empty if none match.
        """
        if any(s != 1 for (_, _, s) in node.map.range):
            return []

        path_candidates = [
            self._get_edges_from_path(state, p)
            for p in state.all_simple_paths(node, state.exit_node(node), as_edges=False)
        ]

        paths = []
        for path_candidate in path_candidates:
            if len(path_candidate) != 2:
                continue

            tasklet = path_candidate[1].src
            if not isinstance(tasklet, dace.nodes.Tasklet):
                continue

            expected_in_conns = 0 if is_memset else 1
            if len(tasklet.in_connectors) != expected_in_conns or len(tasklet.out_connectors) != 1:
                continue

            oe = next(
                state.out_edges_by_connector(path_candidate[-1].dst, path_candidate[-1].dst_conn.replace("IN_",
                                                                                                         "OUT_")))
            if not isinstance(oe.dst, dace.nodes.AccessNode):
                continue

            out_conn = next(iter(tasklet.out_connectors))
            suffix = ";" if tasklet.language == dace.Language.CPP else ""
            if tasklet.language not in (dace.Language.Python, dace.Language.CPP):
                continue

            if is_memset:
                expected_codes = {f"{out_conn} = 0{suffix}", f"{out_conn} = 0.0{suffix}"}
                if tasklet.code.as_string not in expected_codes:
                    continue
                paths.append(path_candidate + [oe])
            else:
                entry_edge = path_candidate[0]
                if entry_edge.dst_conn is None or not entry_edge.src_conn.startswith("OUT_"):
                    continue
                ie = next(state.in_edges_by_connector(entry_edge.src, entry_edge.src_conn.replace("OUT_", "IN_")))
                if not isinstance(ie.src, dace.nodes.AccessNode):
                    continue
                in_conn = next(iter(tasklet.in_connectors))
                if tasklet.code.as_string != f"{out_conn} = {in_conn}{suffix}":
                    continue
                # Reject permutations (e.g. transpose) -- the tasklet body
                # ``_out = _in`` is identical for copy and transpose, so
                # without this check we'd silently lower a transpose to
                # ``cudaMemcpyAsync``. See ``_in_out_subsets_are_pure_copy``.
                if not self._in_out_subsets_are_pure_copy(path_candidate[0].data.subset, path_candidate[1].data.subset,
                                                          node.map.params):
                    continue
                paths.append([ie] + path_candidate + [oe])

        return paths

    def _detect_contiguous_memcpy_paths(self, state: dace.SDFGState,
                                        node: dace.nodes.MapEntry) -> List[List[MultiConnectorEdge]]:
        return self._detect_contiguous_paths(state, node, is_memset=False)

    def _detect_contiguous_memset_paths(self, state: dace.SDFGState,
                                        node: dace.nodes.MapEntry) -> List[List[MultiConnectorEdge]]:
        return self._detect_contiguous_paths(state, node, is_memset=True)

    def _get_num_tasklets_within_map(self, state: dace.SDFGState, node: dace.nodes.MapEntry) -> int:
        assert node in state.nodes(), f"Map entry {node} not in state {state}"
        assert isinstance(node, dace.nodes.MapEntry), f"Node {node} is not a MapEntry"
        assert state.exit_node(node) in state.nodes(), f"Map exit {state.exit_node(node)} not in state {state}"
        n = {n for n in state.all_nodes_between(node, state.exit_node(node)) if isinstance(n, dace.nodes.Tasklet)}
        return len(n)

    def _get_write_begin_and_length(
            self, state: dace.SDFGState, map_entry: dace.nodes.MapEntry,
            tasklet: dace.nodes.Tasklet) -> Tuple[Optional[List], Optional[List], Optional[dace.symbolic.SymExpr]]:
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
                    assert ns == 1 and s2 == 1, "Only step of 1 is supported for memcpy/memset detection"
                new_in_data_range.append((nb, ne, ns))

            # If we overapproximate the first dimension, we assume it is contiguous
            if self.overapproximate_first_dimension:
                arr = state.sdfg.arrays[in_edge.data.data]
                stride_one_dimension = {(i, d) for i, (d, s) in enumerate(zip(arr.shape, arr.strides)) if s == 1}
                assert len(stride_one_dimension) <= 1  # If a view inside a nested SDFG it can be 0 too
                # If no stride-one-dimension then we can't remove this
                if len(stride_one_dimension) == 0:
                    return None, None, None
                dim_offset, stride_one_dimension = stride_one_dimension.pop()
                new_in_data_range[dim_offset] = ((0, stride_one_dimension - 1, 1))

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
            arr = state.sdfg.arrays[out_edge.data.data]
            stride_one_dimension = {(i, d) for i, (d, s) in enumerate(zip(arr.shape, arr.strides)) if s == 1}
            assert len(stride_one_dimension) <= 1  # If a view inside a nested SDFG it can be 0 too
            # If no stride-one-dimension then we can't remove this
            if len(stride_one_dimension) == 0:
                return None, None, None
            dim_offset, stride_one_dimension = stride_one_dimension.pop()
            new_out_data_range[dim_offset] = ((0, stride_one_dimension - 1, 1))

        new_in_data_subset = dace.subsets.Range(new_in_data_range) if in_edge.data.data is not None else None
        new_out_data_subset = dace.subsets.Range(new_out_data_range) if out_edge.data.data is not None else None

        if in_edge.data.data is not None:
            contig_subset = new_in_data_subset.is_contiguous_subset(state.sdfg.arrays[in_edge.data.data])
            if not contig_subset:
                warnings.warn(f"Input array {in_edge.data.data} is not contiguous, cannot remove memcpy/memset.",
                              UserWarning)
                return None, None, None

        if out_edge.data.data is not None:
            contig_subset = new_out_data_subset.is_contiguous_subset(state.sdfg.arrays[out_edge.data.data])
            if not contig_subset:
                warnings.warn(
                    f"Output array {out_edge.data.data} subset {new_out_data_range} is not contiguous, "
                    "cannot remove memcpy/memset.", UserWarning)
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
            # Inner access is over a non-unit stride dimension
            if in_length_collapsed != out_length_collapsed:
                return None, None, None

        return new_in_data_range, new_out_data_range, out_length_collapsed

    def remove_memcpy_from_kernel(self, state: dace.SDFGState, node: dace.nodes.MapEntry, verbose: bool = True) -> int:
        """Lift every pure element-wise-copy path under map ``node`` to a ``CopyLibraryNode``.

        :param state: State containing the map.
        :param node: Map entry of the kernel to scan.
        :param verbose: Emit warnings for skipped lift opportunities.
        :returns: Number of paths lifted.
        """
        memcpy_paths = self._detect_contiguous_memcpy_paths(state, node)
        rmed_count = 0

        joined_edges = set()

        for memcpy_path in memcpy_paths:
            src_access_node = memcpy_path[0].src
            map_entry = memcpy_path[0].dst
            tasklet = memcpy_path[1].dst
            map_exit = memcpy_path[2].dst
            dst_access_node = memcpy_path[3].dst
            if src_access_node not in state.nodes() or map_entry not in state.nodes() or tasklet not in state.nodes(
            ) or map_exit not in state.nodes() or dst_access_node not in state.nodes():
                warnings.warn(
                    f"Skipping memcpy removal: map {map_entry.map.label} or its tasklet/exit is no longer "
                    "in state.", UserWarning)
                continue

            # If src and dst types are not the same, we can't do memcpy
            src_desc = state.sdfg.arrays[src_access_node.data]
            dst_desc = state.sdfg.arrays[dst_access_node.data]
            if src_desc.dtype != dst_desc.dtype:
                if verbose:
                    warnings.warn(f"Skipping memcpy removal: dtype mismatch ({src_desc.dtype} != {dst_desc.dtype}).",
                                  UserWarning)
                continue
            if src_desc.storage != dst_desc.storage:
                if verbose:
                    warnings.warn(
                        f"Skipping memcpy removal: storage mismatch ({src_desc.storage} != {dst_desc.storage}).",
                        UserWarning)
                continue

            # Must run before the memcpy path is torn down: needs the tasklet's edges.
            begin_subset, exit_subset, copy_length = self._get_write_begin_and_length(state, map_entry, tasklet)

            if begin_subset is None and exit_subset is None and copy_length is None:
                continue

            if self._is_single_element_copy(copy_length):
                continue

            # Skip when either passthrough connector (entry-side IN_X or
            # exit-side IN_X) is shared by other tasklets -- lifting would
            # sever the shared edge that the other tasklets still need.
            # See the matching guard in ``remove_memset_from_kernel``.
            entry_in_conn = memcpy_path[0].dst_conn
            exit_in_conn = memcpy_path[2].dst_conn
            shared_entry = entry_in_conn is not None and len(list(state.in_edges_by_connector(map_entry,
                                                                                              entry_in_conn))) > 1
            shared_exit = exit_in_conn is not None and len(list(state.in_edges_by_connector(map_exit,
                                                                                            exit_in_conn))) > 1
            if shared_entry or shared_exit:
                if verbose:
                    warnings.warn(
                        f"Skipping memcpy lift in map {map_entry.map.label}: "
                        f"passthrough connector(s) shared with other tasklets -- "
                        f"lifting would break their data paths.", UserWarning)
                continue

            # Skip when the parent SDFG has arrays whose names would
            # collide with the new libnode's published connector names
            # (validator rejects connector-vs-array-name collisions).
            # Most common cause: the surrounding SDFG is a CopyLibraryNode
            # expansion wrapper whose parameter arrays are exactly those
            # names -- re-lifting inside it would recreate the clash that
            # motivated the original libnode connector rename.
            clashes = ({CopyLibraryNode.INPUT_CONNECTOR_NAME, CopyLibraryNode.OUTPUT_CONNECTOR_NAME}
                       & set(state.sdfg.arrays))
            if clashes:
                if verbose:
                    warnings.warn(
                        f"Skipping memcpy lift in map {map_entry.map.label}: parent SDFG "
                        f"already has arrays {clashes} which would clash with the new "
                        f"CopyLibraryNode's connectors.", UserWarning)
                continue

            if src_access_node not in state.nodes():
                new_src_access_node = state.add_access(src_access_node.data)
            else:
                new_src_access_node = src_access_node
            if dst_access_node not in state.nodes():
                new_dst_access_node = state.add_access(dst_access_node.data)
            else:
                new_dst_access_node = dst_access_node

            tasklet = CopyLibraryNode(
                name=f"copyLib_{new_src_access_node.data}_{new_dst_access_node.data}_{self.rmid}", )
            state.add_node(tasklet)
            self.rmid += 1
            state.add_edge(new_src_access_node, None, tasklet, CopyLibraryNode.INPUT_CONNECTOR_NAME,
                           dace.memlet.Memlet(subset=dace.subsets.Range(begin_subset), data=new_src_access_node.data))
            state.add_edge(tasklet, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, new_dst_access_node, None,
                           dace.memlet.Memlet(subset=dace.subsets.Range(exit_subset), data=new_dst_access_node.data))
            # Map-entry in-edges are either IN_* data passthroughs (already
            # handled by the libnode's _cpy_in / _cpy_out) or dynamic map-range
            # scalars (any non-IN_*, regardless of naming). The libnode doesn't
            # iterate, so neither belongs on it.

            rmed_count += 1
            joined_edges.update(memcpy_path)

        self.rm_edges(state, joined_edges)

        return rmed_count

    def remove_memset_from_kernel(self, state: dace.SDFGState, node: dace.nodes.MapEntry, verbose: bool = True) -> int:
        """Lift every constant-zero-write path under map ``node`` to a ``MemsetLibraryNode``.

        :param state: State containing the map.
        :param node: Map entry of the kernel to scan.
        :param verbose: Emit warnings for skipped lift opportunities.
        :returns: Number of paths lifted.
        """
        memset_paths = self._detect_contiguous_memset_paths(state, node)

        joined_edges = set()

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

            if map_entry not in state.nodes() or map_exit not in state.nodes() or tasklet not in state.nodes():
                warnings.warn(
                    f"Skipping memset removal: map {map_entry.map.label} or its tasklet/exit is no longer "
                    "in state.", UserWarning)
                continue

            # Must run before the memset path is torn down: needs the tasklet's edges.
            begin_subset, exit_subset, copy_length = self._get_write_begin_and_length(state, map_entry, tasklet)

            if begin_subset is None or exit_subset is None or copy_length is None:
                if verbose:
                    warnings.warn(
                        f"Skipping memset removal in map {map_entry.map.label}: subset or copy length "
                        "could not be determined or is non-contiguous.", UserWarning)
                continue

            if self._is_single_element_copy(copy_length):
                continue

            # Skip when the tasklet->map_exit edge's IN_X passthrough is
            # shared by other tasklets writing the same array (e.g. a
            # boundary memset and a per-thread compute both feeding
            # ``MapExit.IN_y2`` whose ``OUT_y2`` aggregates into a single
            # ``AccessNode(y2)``). Lifting one would sever the shared
            # ``map_exit -> AccessNode`` edge that the other still needs.
            shared_dst_conn = memset_path[1].dst_conn
            if shared_dst_conn is not None and len(list(state.in_edges_by_connector(map_exit, shared_dst_conn))) > 1:
                if verbose:
                    warnings.warn(
                        f"Skipping memset lift in map {map_entry.map.label}: "
                        f"map_exit connector ``{shared_dst_conn}`` is shared with "
                        f"other tasklets -- lifting would break their data paths.", UserWarning)
                continue

            # Same connector-vs-array-name clash guard as the memcpy
            # path above (see comment there).
            if MemsetLibraryNode.OUTPUT_CONNECTOR_NAME in state.sdfg.arrays:
                if verbose:
                    warnings.warn(
                        f"Skipping memset lift in map {map_entry.map.label}: parent SDFG "
                        f"already has an array named "
                        f"``{MemsetLibraryNode.OUTPUT_CONNECTOR_NAME}`` which would clash "
                        f"with the new MemsetLibraryNode's connector.", UserWarning)
                continue

            tasklet = MemsetLibraryNode(name=f"memsetLib_{dst_access_node.data}_{self.rmid}", )
            state.add_node(tasklet)
            self.rmid += 1
            state.add_edge(tasklet, MemsetLibraryNode.OUTPUT_CONNECTOR_NAME, dst_access_node, None,
                           dace.memlet.Memlet(subset=dace.subsets.Range(exit_subset), data=dst_access_node.data))
            # Map-entry in-edges are either IN_* data passthroughs or dynamic
            # map-range scalars (any non-IN_*); neither belongs on the libnode.

            rmed_count += 1
            joined_edges.update(memset_path)

        self.rm_edges(state, joined_edges)

        return rmed_count

    def _has_passthrough_connectors(self, n: dace.nodes.Node) -> bool:
        in_conns = n.in_connectors
        out_conns = n.out_connectors

        has_passtrough = any({c.startswith("IN_") for c in in_conns})
        has_passtrough |= any({c.startswith("OUT_") for c in out_conns})

        return has_passtrough

    def rm_edges(self, state: dace.SDFGState, edges: Iterable[Edge[Memlet]]):
        nodes_to_check = set()
        for i, e in enumerate(edges):
            assert e in state.edges(), f"{e} not in {state.edges()}"
            state.remove_edge(e)
            if e.src_conn is not None:
                e.src.remove_out_connector(e.src_conn)
            if e.dst_conn is not None:
                e.dst.remove_in_connector(e.dst_conn)
            nodes_to_check.add(e.src)
            nodes_to_check.add(e.dst)

        for n in nodes_to_check:
            if isinstance(n, dace.nodes.MapEntry):
                # If it has passthrough connectors then data is left,
                # Otherwise only dynamic connectors and we should remove them
                if (not self._has_passthrough_connectors(n)) and state.out_degree(n) == 0:
                    state.remove_node(n)
            if isinstance(n, dace.nodes.MapExit):
                if not self._has_passthrough_connectors(n) and state.in_degree(n) == 0:
                    state.remove_node(n)

        for n in state.nodes():
            if (state.degree(n) == 0):
                state.remove_node(n)

    @staticmethod
    def _is_single_element_copy(copy_length) -> bool:
        """True iff the lift would write a single element.

        Single-element transfers must not be lifted: the libnode pure expansion
        collapses every singleton dim, yielding an empty map shape that breaks
        memlet propagation. There is also no perf gain over the original tasklet.

        :param copy_length: Collapsed transfer length expression.
        :returns: True iff the length simplifies to the integer 1.
        """
        try:
            return int(dace.symbolic.simplify(copy_length)) == 1
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _is_nested_in_gpu_scope(state: dace.SDFGState, node: dace.nodes.MapEntry) -> bool:
        """True iff ``node`` sits inside any ancestor map with a GPU schedule.

        An in-kernel lift would expand to ``cudaMemcpyAsync`` / ``cudaMemsetAsync``,
        which are host-only and cannot run from device code.

        :param state: State containing the map.
        :param node: Map entry whose ancestor chain is checked.
        :returns: True iff any ancestor map has a GPU schedule.
        """
        parent_tuple = helpers.get_parent_map(state, node)
        while parent_tuple is not None:
            parent_map, parent_state = parent_tuple
            if parent_map.map.schedule in dace.dtypes.GPU_SCHEDULES:
                return True
            parent_tuple = helpers.get_parent_map(parent_state, parent_map)
        return False

    def apply_pass(self, sdfg: dace.SDFG, pipeline_res: Dict) -> int:
        """Walk every map in ``sdfg`` and lift its element-wise-copy / constant-zero paths.

        :param sdfg: SDFG to mutate in place.
        :param pipeline_res: Unused; provided by the pass-pipeline contract.
        :returns: Total number of memcpy + memset paths lifted across the SDFG.
        """
        map_entries = set()

        for n, g in sdfg.all_nodes_recursive():
            if isinstance(n, dace.nodes.MapEntry):
                map_entries.add((n, g))

        rmed_memcpies = dict()
        rmed_memsets = dict()

        for (node, state) in map_entries:
            assert node in state.nodes(), f"Map entry {node} not in state {state}"
            assert state.exit_node(node) in state.nodes(), f"Map exit {state.exit_node(node)} not in state {state}"

            if self.node_label_whitelist != [] and self.node_label_whitelist is not None and node.label not in self.node_label_whitelist:
                continue

            if self._get_num_tasklets_within_map(state, node) == 0:
                continue

            if self._is_nested_in_gpu_scope(state, node):
                continue

            rmed_memcpy = self.remove_memcpy_from_kernel(state, node)

            # If the map is only used for 1 memcpy, then it might have been already removed
            if node in state.nodes():
                rmed_memset = self.remove_memset_from_kernel(state, node)
            else:
                rmed_memset = 0

            assert node not in rmed_memsets
            assert node not in rmed_memcpies
            rmed_memcpies[node] = rmed_memcpy
            rmed_memsets[node] = rmed_memset

        num_rmed_memcpies = sum(rmed_memcpies.values())
        num_rmed_memsets = sum(rmed_memsets.values())

        return num_rmed_memcpies + num_rmed_memsets
