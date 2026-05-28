# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lift contiguous zero-assignments and element-wise copies out of maps into Memset / Copy library nodes."""
import warnings
from typing import Dict, Iterable, List, Optional, Set, Tuple

import dace
from dace import dtypes, properties
from dace.memlet import Memlet
from dace.sdfg import graph, utils as sdutils
from dace.transformation import helpers, pass_pipeline as ppl, transformation
from dace.libraries.standard.helper import CURRENT_STREAM_NAME
from dace.libraries.standard.nodes import copy_node, memset_node


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

    def _get_edges_from_path(self, state: dace.SDFGState,
                             node_path: List[dace.nodes.Node]) -> List[graph.MultiConnectorEdge]:
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
                                 is_memset: bool) -> List[List[graph.MultiConnectorEdge]]:
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
                                        node: dace.nodes.MapEntry) -> List[List[graph.MultiConnectorEdge]]:
        """Element-wise-copy specialization of :meth:`_detect_contiguous_paths`.

        :param state: State containing the map.
        :param node: Map entry of the kernel to scan.
        :returns: One edge list per matched copy path; empty if none match.
        """
        return self._detect_contiguous_paths(state, node, is_memset=False)

    def _detect_contiguous_memset_paths(self, state: dace.SDFGState,
                                        node: dace.nodes.MapEntry) -> List[List[graph.MultiConnectorEdge]]:
        """Constant-zero-write specialization of :meth:`_detect_contiguous_paths`.

        :param state: State containing the map.
        :param node: Map entry of the kernel to scan.
        :returns: One edge list per matched memset path; empty if none match.
        """
        return self._detect_contiguous_paths(state, node, is_memset=True)

    def _get_num_tasklets_within_map(self, state: dace.SDFGState, node: dace.nodes.MapEntry) -> int:
        """Count the tasklets nested inside the scope of map ``node``.

        :param state: State containing the map.
        :param node: Map entry whose body is scanned.
        :returns: Number of distinct tasklets between the map entry and its exit.
        """
        assert node in state.nodes(), f"Map entry {node} not in state {state}"
        assert isinstance(node, dace.nodes.MapEntry), f"Node {node} is not a MapEntry"
        assert state.exit_node(node) in state.nodes(), f"Map exit {state.exit_node(node)} not in state {state}"
        n = {n for n in state.all_nodes_between(node, state.exit_node(node)) if isinstance(n, dace.nodes.Tasklet)}
        return len(n)

    def _subst_and_overapprox(self, data_range: List, range_list: dict, data_name: str,
                              sdfg: dace.SDFG) -> Optional[List]:
        """Substitute map parameters into ``data_range`` and, when
        ``overapproximate_first_dimension`` is set, widen the stride-1 axis
        to the array's full contiguous extent.

        :param data_range: ``(begin, end, step)`` per dimension (map-relative).
        :param range_list: map symbol -> ``(begin, end, step)``.
        :param data_name: array the subset addresses.
        :param sdfg: SDFG owning ``data_name``.
        :returns: the rewritten range, or ``None`` if it cannot be lowered.
        """
        new_range = []
        for (b, e, s) in data_range:
            nb, ne, ns = b, e, s
            for (p, (b2, e2, s2)) in range_list.items():
                nb = nb.subs(p, b2)
                ne = ne.subs(p, e2)
                assert ns == 1 and s2 == 1, "Only step of 1 is supported for memcpy/memset detection"
            new_range.append((nb, ne, ns))

        if self.overapproximate_first_dimension:
            arr = sdfg.arrays[data_name]
            stride_one = {(i, d) for i, (d, s) in enumerate(zip(arr.shape, arr.strides)) if s == 1}
            assert len(stride_one) <= 1  # a view inside a nested SDFG can have 0
            if len(stride_one) == 0:
                return None
            dim_offset, extent = stride_one.pop()
            new_range[dim_offset] = (0, extent - 1, 1)
        return new_range

    @staticmethod
    def _reject_if_not_contiguous(new_range: List, data_name: str, sdfg: dace.SDFG, *, is_input: bool) -> bool:
        """Warn and return ``False`` when ``new_range`` is non-contiguous in its array.

        :param new_range: the rewritten subset range.
        :param data_name: array the range addresses.
        :param sdfg: SDFG owning ``data_name``.
        :param is_input: selects the input vs output warning message.
        :returns: ``True`` iff the subset is contiguous (safe to lower).
        """
        if dace.subsets.Range(new_range).is_contiguous_subset(sdfg.arrays[data_name]):
            return True
        if is_input:
            warnings.warn(f"Input array {data_name} is not contiguous, cannot remove memcpy/memset.", UserWarning)
        else:
            warnings.warn(
                f"Output array {data_name} subset {new_range} is not contiguous, "
                "cannot remove memcpy/memset.", UserWarning)
        return False

    @staticmethod
    def _collapsed_length(new_range: List) -> dace.symbolic.SymExpr:
        """Product of per-dimension lengths of a (contiguous) subset range."""
        total = dace.symbolic.SymExpr(1)
        for (b, e, s) in new_range:
            total *= (e + 1) - b
        return total

    def _get_write_begin_and_length(
            self, state: dace.SDFGState, map_entry: dace.nodes.MapEntry,
            tasklet: dace.nodes.Tasklet) -> Tuple[Optional[List], Optional[List], Optional[dace.symbolic.SymExpr]]:
        range_list = {
            dace.symbolic.symbol(p): (b, e, s)
            for (p, (b, e, s)) in zip(map_entry.map.params, map_entry.map.range)
        }
        in_edge = state.in_edges(tasklet)[0]
        out_edge = state.out_edges(tasklet)[0]
        has_in = in_edge.data.data is not None

        new_out = self._subst_and_overapprox([(b, e, s) for (b, e, s) in out_edge.data.subset], range_list,
                                             out_edge.data.data, state.sdfg)
        if new_out is None:
            return None, None, None
        new_in = []
        if has_in:
            new_in = self._subst_and_overapprox([(b, e, s) for (b, e, s) in in_edge.data.subset], range_list,
                                                in_edge.data.data, state.sdfg)
            if new_in is None:
                return None, None, None

        if has_in and not self._reject_if_not_contiguous(new_in, in_edge.data.data, state.sdfg, is_input=True):
            return None, None, None
        if out_edge.data.data is not None and not self._reject_if_not_contiguous(
                new_out, out_edge.data.data, state.sdfg, is_input=False):
            return None, None, None

        out_length_collapsed = self._collapsed_length(new_out)
        # Reject when the inner access spans a non-unit-stride dimension.
        if has_in and self._collapsed_length(new_in) != out_length_collapsed:
            return None, None, None

        return new_in, new_out, out_length_collapsed

    def _hoist_dynamic_inputs_to_symbols(self, state: dace.SDFGState, map_entry: dace.nodes.MapEntry,
                                         used_symbols: Set[str]) -> bool:
        """Promote dynamic map-input connectors referenced by ``used_symbols`` to in-scope symbols.

        A dynamic map input binds a scalar value to a connector that the map range -- and thus the
        lifted library node's subset -- references as a symbol. Once the map is removed that binding is
        gone, so the scalar is read into the same-named symbol on a state inserted before ``state``; the
        lifted subset already uses the connector name, so no subset rewrite is needed.

        Hoisting is sound only when the source scalar is not written within ``state`` (otherwise the
        hoisted read would observe a stale value). When it is, the caller falls back to nesting the map
        in its own SDFG, where the scalar arrives as a read-only input.

        :param state: The state containing the map.
        :param map_entry: The map entry whose dynamic inputs are promoted.
        :param used_symbols: Symbol names referenced by the lifted subset.
        :returns: True if every referenced dynamic input was promoted; False if any source scalar is
            written in ``state`` (the caller must nest instead).
        """
        dynamic_edges = [e for e in sdutils.dynamic_map_inputs(state, map_entry) if e.dst_conn in used_symbols]
        if not dynamic_edges:
            return True

        written = state.read_and_write_sets()[1]
        if any(not isinstance(e.src, dace.nodes.AccessNode) or e.src.data in written for e in dynamic_edges):
            return False

        sdfg = state.sdfg
        assignments = {}
        for e in dynamic_edges:
            desc = sdfg.arrays[e.src.data]
            # A Scalar is passed by value (referenced bare, like the frontend's own
            # range-bound assignments); an Array is indexed by the edge's subset.
            assignments[e.dst_conn] = e.src.data if isinstance(desc, dace.data.Scalar) else f"{e.src.data}[{e.data.subset}]"
            if e.dst_conn not in sdfg.symbols:
                sdfg.add_symbol(e.dst_conn, desc.dtype)
        state.parent_graph.add_state_before(state, assignments=assignments)
        for e in dynamic_edges:
            state.remove_edge(e)
            if e.dst_conn in map_entry.in_connectors:
                map_entry.remove_in_connector(e.dst_conn)
        return True

    @staticmethod
    def _subset_symbols(*subsets: Optional[List]) -> Set[str]:
        """Collect free-symbol names referenced by one or more ``(begin, end, step)`` range lists."""
        used = set()
        for subset in subsets:
            if subset:
                used |= {str(s) for s in dace.subsets.Range(subset).free_symbols}
        return used

    @staticmethod
    def _needs_nesting_for_dynamic_inputs(state: dace.SDFGState, map_entry: dace.nodes.MapEntry) -> bool:
        """Whether ``map_entry`` has a dynamic-range bound whose source scalar is written in ``state``.

        Such a bound cannot be hoisted to a preceding-state symbol assignment (the read would be
        stale); the map must first be nested in its own SDFG, where the scalar becomes a read-only
        input.

        :param state: The state containing the map.
        :param map_entry: The map entry to inspect.
        :returns: True if a dynamic input's source scalar is written in ``state``.
        """
        dynamic_edges = sdutils.dynamic_map_inputs(state, map_entry)
        if not dynamic_edges:
            return False
        written = state.read_and_write_sets()[1]
        return any(not isinstance(e.src, dace.nodes.AccessNode) or e.src.data in written for e in dynamic_edges)

    def _lift_preconditions_ok(self, state: dace.SDFGState, map_entry: dace.nodes.MapEntry, *, kind: str,
                               passthrough_conns: List, libnode_conn_names: Set[str], begin_subset: Optional[List],
                               exit_subset: List, copy_length: dace.symbolic.SymExpr, verbose: bool) -> bool:
        """Shared skip-checks run before lifting a memcpy / memset path to a library node.

        In order: reject single-element transfers; reject when a passthrough connector is shared with
        other tasklets (lifting would sever their data path); reject when the new library node's
        connector names collide with parent-SDFG array names; finally promote any dynamic-range bound
        to an in-scope symbol (returning False when that requires the nested-SDFG fallback instead).

        :param state: The state containing the map.
        :param map_entry: The map entry being lifted.
        :param kind: ``'memcpy'`` or ``'memset'`` -- used only in warning text.
        :param passthrough_conns: ``(connector, scope_node)`` pairs whose sharing blocks the lift.
        :param libnode_conn_names: connector names the new library node publishes.
        :param begin_subset: source-side range, or ``None`` for memset.
        :param exit_subset: destination-side range.
        :param copy_length: collapsed transfer length.
        :param verbose: emit a warning on each skip.
        :returns: True iff the lift may proceed.
        """
        if self._is_single_element_copy(copy_length):
            return False

        for conn, scope in passthrough_conns:
            if conn is not None and len(list(state.in_edges_by_connector(scope, conn))) > 1:
                if verbose:
                    warnings.warn(
                        f"Skipping {kind} lift in map {map_entry.map.label}: passthrough connector ``{conn}`` "
                        f"is shared with other tasklets -- lifting would break their data paths.", UserWarning)
                return False

        clashes = libnode_conn_names & set(state.sdfg.arrays)
        if clashes:
            if verbose:
                warnings.warn(
                    f"Skipping {kind} lift in map {map_entry.map.label}: parent SDFG already has arrays "
                    f"{clashes} which would clash with the new library node's connectors.", UserWarning)
            return False

        if not self._hoist_dynamic_inputs_to_symbols(state, map_entry,
                                                     self._subset_symbols(begin_subset, exit_subset)):
            if verbose:
                warnings.warn(
                    f"Skipping {kind} lift in map {map_entry.map.label}: a dynamic-range source scalar is "
                    f"written in the same state; nesting fallback required.", UserWarning)
            return False

        return True

    def remove_memcpy_from_kernel(self, state: dace.SDFGState, node: dace.nodes.MapEntry, verbose: bool = True) -> int:
        """Lift every pure element-wise-copy path under map ``node`` to a ``CopyLibraryNode``.

        :param state: State containing the map.
        :param node: Map entry of the kernel to scan.
        :param verbose: Emit warnings for skipped lift opportunities.
        :returns: Number of paths lifted.
        """
        return self._lift_paths(state, node, is_memset=False, verbose=verbose)

    def remove_memset_from_kernel(self, state: dace.SDFGState, node: dace.nodes.MapEntry, verbose: bool = True) -> int:
        """Lift every constant-zero-write path under map ``node`` to a ``MemsetLibraryNode``.

        :param state: State containing the map.
        :param node: Map entry of the kernel to scan.
        :param verbose: Emit warnings for skipped lift opportunities.
        :returns: Number of paths lifted.
        """
        return self._lift_paths(state, node, is_memset=True, verbose=verbose)

    def _lift_paths(self, state: dace.SDFGState, node: dace.nodes.MapEntry, *, is_memset: bool, verbose: bool) -> int:
        """Lift every detected pure-copy / constant-zero path under map ``node`` to a library node.

        Both flavours share one skeleton: detect the contiguous
        ``MapEntry -> tasklet -> MapExit -> AccessNode`` paths, validate each via
        :meth:`_lift_preconditions_ok`, and replace it with a ``CopyLibraryNode``
        (memcpy) or ``MemsetLibraryNode`` (memset). A memcpy additionally carries
        a source AccessNode + input edge and requires matching src/dst dtype and
        storage; a memset writes a constant and has neither.

        :param state: State containing the map.
        :param node: Map entry of the kernel to scan.
        :param is_memset: Lift constant-zero writes when True, element-wise copies when False.
        :param verbose: Emit warnings for skipped lift opportunities.
        :returns: Number of paths lifted.
        """
        if is_memset:
            paths = self._detect_contiguous_memset_paths(state, node)
            libnode_cls, kind = memset_node.MemsetLibraryNode, "memset"
            libnode_conn_names = {libnode_cls.OUTPUT_CONNECTOR_NAME}
        else:
            paths = self._detect_contiguous_memcpy_paths(state, node)
            libnode_cls, kind = copy_node.CopyLibraryNode, "memcpy"
            libnode_conn_names = {libnode_cls.INPUT_CONNECTOR_NAME, libnode_cls.OUTPUT_CONNECTOR_NAME}

        joined_edges = set()
        rmed_count = 0
        for path in paths:
            # Read the common tail from the exit side: ``tasklet -> MapExit -> AccessNode``.
            # A memcpy path additionally prepends ``source AccessNode -> MapEntry`` at ``path[0]``.
            tasklet = path[-2].src
            map_exit = path[-2].dst
            dst_access_node = path[-1].dst
            src_access_node = None if is_memset else path[0].src

            present = [node, tasklet, map_exit, dst_access_node] + ([] if is_memset else [src_access_node])
            if any(n not in state.nodes() for n in present):
                warnings.warn(
                    f"Skipping {kind} removal: map {node.map.label} or its tasklet/exit is no longer "
                    "in state.", UserWarning)
                continue

            # A memcpy lowers to a byte copy, so source and destination must agree on dtype and storage.
            if not is_memset:
                src_desc = state.sdfg.arrays[src_access_node.data]
                dst_desc = state.sdfg.arrays[dst_access_node.data]
                if src_desc.dtype != dst_desc.dtype:
                    if verbose:
                        warnings.warn(
                            f"Skipping memcpy removal: dtype mismatch ({src_desc.dtype} != {dst_desc.dtype}).",
                            UserWarning)
                    continue
                if src_desc.storage != dst_desc.storage:
                    if verbose:
                        warnings.warn(
                            f"Skipping memcpy removal: storage mismatch ({src_desc.storage} != {dst_desc.storage}).",
                            UserWarning)
                    continue

            # Must run before the path is torn down: needs the tasklet's edges. A bail returns all-None.
            begin_subset, exit_subset, copy_length = self._get_write_begin_and_length(state, node, tasklet)
            if copy_length is None:
                if is_memset and verbose:
                    warnings.warn(
                        f"Skipping memset removal in map {node.map.label}: subset or copy length "
                        "could not be determined or is non-contiguous.", UserWarning)
                continue

            # The exit-side IN_X passthrough (destination data) -- and, for memcpy, the entry-side
            # IN_X (source data) -- block the lift if shared with other tasklets.
            passthrough_conns = [(path[-2].dst_conn, map_exit)]
            if not is_memset:
                passthrough_conns.append((path[0].dst_conn, node))
            if not self._lift_preconditions_ok(state, node, kind=kind, passthrough_conns=passthrough_conns,
                                               libnode_conn_names=libnode_conn_names, begin_subset=begin_subset,
                                               exit_subset=exit_subset, copy_length=copy_length, verbose=verbose):
                continue

            if is_memset:
                libnode = libnode_cls(name=f"memsetLib_{dst_access_node.data}_{self.rmid}")
                state.add_node(libnode)
                state.add_edge(libnode, libnode_cls.OUTPUT_CONNECTOR_NAME, dst_access_node, None,
                               dace.memlet.Memlet(subset=dace.subsets.Range(exit_subset), data=dst_access_node.data))
            else:
                libnode = libnode_cls(name=f"copyLib_{src_access_node.data}_{dst_access_node.data}_{self.rmid}")
                state.add_node(libnode)
                state.add_edge(src_access_node, None, libnode, libnode_cls.INPUT_CONNECTOR_NAME,
                               dace.memlet.Memlet(subset=dace.subsets.Range(begin_subset), data=src_access_node.data))
                state.add_edge(libnode, libnode_cls.OUTPUT_CONNECTOR_NAME, dst_access_node, None,
                               dace.memlet.Memlet(subset=dace.subsets.Range(exit_subset), data=dst_access_node.data))
            self._transfer_stream_wiring(state, node, libnode)
            self.rmid += 1
            rmed_count += 1
            joined_edges.update(path)

        self.rm_edges(state, joined_edges)
        return rmed_count

    def _transfer_stream_wiring(self, state: dace.SDFGState, map_entry: dace.nodes.MapEntry,
                                libnode: dace.nodes.LibraryNode):
        """Move the GPU-stream in-wiring from ``map_entry`` onto ``libnode``.

        The pre-lift map carries a ``__dace_current_stream`` in-connector that the
        stream scheduler wired to a ``gpu_streams[i]`` AccessNode. The expanded
        cudaMemcpy*Async tasklet derived from ``libnode`` needs the same stream
        binding, so we re-source the edge onto the libnode. Without this transfer
        the post-expansion scheduler re-entry is gated by ``is_gpu_lowering_applied``
        and the new tasklet never gets a stream.
        """
        if CURRENT_STREAM_NAME not in map_entry.in_connectors:
            return
        stream_in_edges = [e for e in state.in_edges(map_entry) if e.dst_conn == CURRENT_STREAM_NAME]
        if not stream_in_edges:
            return
        libnode.add_in_connector(CURRENT_STREAM_NAME, dtypes.gpuStream_t)
        for e in stream_in_edges:
            state.add_edge(e.src, e.src_conn, libnode, CURRENT_STREAM_NAME, dace.memlet.Memlet.from_memlet(e.data))

    def _has_passthrough_connectors(self, n: dace.nodes.Node) -> bool:
        """Whether ``n`` carries scope-passthrough connectors.

        :param n: Node to inspect (typically a map entry/exit).
        :returns: True if any connector is an ``IN_`` / ``OUT_`` passthrough pair.
        """
        in_conns = n.in_connectors
        out_conns = n.out_connectors

        has_passtrough = any({c.startswith("IN_") for c in in_conns})
        has_passtrough |= any({c.startswith("OUT_") for c in out_conns})

        return has_passtrough

    def rm_edges(self, state: dace.SDFGState, edges: Iterable[graph.Edge[Memlet]]):
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
            # A node may have been nested away by an earlier iteration's fallback.
            if node not in state.nodes():
                continue

            if self.node_label_whitelist != [] and self.node_label_whitelist is not None and node.label not in self.node_label_whitelist:
                continue

            if self._get_num_tasklets_within_map(state, node) == 0:
                continue

            if self._is_nested_in_gpu_scope(state, node):
                continue

            # A dynamic-range bound written in this state cannot be hoisted to a
            # symbol directly; nest the map in its own SDFG (whole arrays passed
            # in, the scalar arriving as a read-only input) and lift inside,
            # where the safe-hoist applies.
            if self._needs_nesting_for_dynamic_inputs(state, node) and (
                    self._detect_contiguous_memcpy_paths(state, node)
                    or self._detect_contiguous_memset_paths(state, node)):
                subgraph = state.scope_subgraph(node, include_entry=True, include_exit=True)
                nsdfg_node = helpers.nest_state_subgraph(state.sdfg, state, subgraph, full_data=True)
                rmed_memcpies[node] = self.apply_pass(nsdfg_node.sdfg, {})
                rmed_memsets[node] = 0
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
