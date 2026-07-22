# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set

from dace import SDFG, SDFGState, data, properties
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.analysis import cfg
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.validation import InvalidSDFGNodeError
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.dataflow import (RedundantArray, RedundantReadSlice, RedundantSecondArray, RedundantWriteSlice,
                                          SqueezeViewRemove, UnsqueezeViewRemove, RemoveSliceView)
from dace.transformation.passes import analysis as ap
from dace.transformation.transformation import SingleStateTransformation
from ordered_set import OrderedSet


def _state_has_read_write_sibling_carrier(state: SDFGState, exclude_data: str) -> bool:
    """Return whether ``state`` contains a container (other than ``exclude_data``)
    with both a read-only source AccessNode (``in_degree == 0`` and
    ``out_degree > 0``) AND a write-only sink AccessNode (``in_degree > 0`` and
    ``out_degree == 0``). Such a container is read AND written in the same state
    -- a write-after-read (WAR) carrier whose read-before-write ordering is
    enforced implicitly by the surrounding dataflow; folding sibling AccessNodes
    of other containers can break that ordering.

    Both TRANSIENT (per-iteration scalar carriers, e.g. s254/s255's ``x``) and
    NON-TRANSIENT (an argument read then updated in place -- e.g. the anti-dep
    snapshot of s212 ``a[i]=a[i]*c[i]; b[i]+=a[i+1]*d[i]``, where ``a`` is a
    read-only source into the snapshot copy + the b-read AND a write-only sink
    of the in-place update) carriers are checked: eliminating ``a``'s snapshot
    because it looks like a redundant copy would redirect the b-read back to the
    updated ``a`` and miscompile.

    :param state: The state to scan.
    :param exclude_data: The data name currently being considered for merge
                         -- skipped because we want to check OTHER containers.
    :returns: ``True`` if such a sibling carrier exists.
    """
    by_name: Dict[str, List[nodes.AccessNode]] = defaultdict(list)
    for n in state.nodes():
        if not isinstance(n, nodes.AccessNode):
            continue
        if n.data == exclude_data:
            continue
        desc = state.sdfg.arrays.get(n.data)
        if desc is None:
            continue
        by_name[n.data].append(n)
    for ans in by_name.values():
        has_read_only = any(state.in_degree(n) == 0 and state.out_degree(n) > 0 for n in ans)
        has_write_only = any(state.in_degree(n) > 0 and state.out_degree(n) == 0 for n in ans)
        if has_read_only and has_write_only:
            return True
    return False


def _is_war_carrier(state: SDFGState, data_name: str) -> bool:
    """True iff ``data_name`` is READ and WRITTEN in ``state`` through distinct AccessNodes:
    a read-only source (``in_degree == 0`` and ``out_degree > 0``) AND a write-only sink
    (``in_degree > 0`` and ``out_degree == 0``). Removing a redundant COPY of such a container
    -- redirecting the copy's readers back to it -- is UNSOUND: it exposes those reads to the
    in-place write, a write-after-read on the copy's source. This is the anti-dependence
    snapshot of s212 (``a[i]=a[i]*c[i]; b[i]+=a[i+1]*d[i]``): ``a`` is a read-only source (into
    the ``a->a_split_snap`` copy and the ``b`` read) AND a write-only sink (the in-place update),
    so folding ``a_split_snap`` back onto ``a`` would make ``b`` read the updated ``a``."""
    anodes = [n for n in state.nodes() if isinstance(n, nodes.AccessNode) and n.data == data_name]
    has_read_only = any(state.in_degree(n) == 0 and state.out_degree(n) > 0 for n in anodes)
    has_write_only = any(state.in_degree(n) > 0 and state.out_degree(n) == 0 for n in anodes)
    return has_read_only and has_write_only


@properties.make_properties
@transformation.explicit_cf_compatible
class ArrayElimination(ppl.Pass):
    """
    Merges and removes arrays and their corresponding accesses. This includes redundant array copies, unnecessary views,
    and duplicate access nodes.
    """

    CATEGORY: str = 'Simplification'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.AccessNodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.AccessNodes

    def depends_on(self):
        return [ap.StateReachability, ap.FindAccessStates]

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Set[str]]:
        """
        Removes redundant arrays and access nodes.

        :param sdfg: The SDFG to modify.
        :param pipeline_results: If in the context of a ``Pipeline``, a dictionary that is populated with prior Pass
                                 results as ``{Pass subclass name: returned object from pass}``. If not run in a
                                 pipeline, an empty dictionary is expected.
        :return: A set of removed data descriptor names, or None if nothing changed.
        """
        result: Set[str] = set()
        reachable: Dict[SDFGState, Set[SDFGState]] = pipeline_results[ap.StateReachability.__name__][sdfg.cfg_id]
        # Get access nodes and modify set as pass continues
        access_sets: Dict[str, Set[SDFGState]] = pipeline_results[ap.FindAccessStates.__name__][sdfg.cfg_id]

        # Traverse SDFG backwards
        try:
            state_order = list(cfg.blockorder_topological_sort(sdfg, recursive=True, ignore_nonstate_blocks=True))
        except KeyError:
            return None
        for state in reversed(state_order):
            # Find all data descriptors that will no longer be used after this state
            removable_data: OrderedSet[str] = OrderedSet(
                s for s in access_sets if state in access_sets[s] and not (access_sets[s] & reachable[state]) - {state})

            # Find duplicate access nodes as an ordered list
            access_nodes: Dict[str, List[nodes.AccessNode]] = defaultdict(list)
            for node in state.nodes():
                if isinstance(node, nodes.AccessNode):
                    access_nodes[node.data].append(node)

            # Merge source and sink access nodes
            removed_nodes = self.merge_access_nodes(state, access_nodes, lambda n: state.in_degree(n) == 0)
            removed_nodes |= self.merge_access_nodes(state, access_nodes, lambda n: state.out_degree(n) == 0)

            # Remove redundant views
            removed_nodes |= self.remove_redundant_views(sdfg, state, access_nodes)

            # Remove redundant copies and views
            removed_nodes |= self.remove_redundant_copies(sdfg, state, removable_data, access_nodes)

            # Update access set if all nodes were removed
            for aname, anodes in access_nodes.items():
                if len(set(anodes) - removed_nodes) == 0:
                    access_sets[aname].remove(state)

            if removed_nodes:
                result.update({n.data for n in removed_nodes})

        # If node is completely removed from graph, erase data descriptor
        for aname, desc in list(sdfg.arrays.items()):
            if isinstance(desc, data.DistributedDescriptor):
                continue
            if not desc.transient or isinstance(desc, data.Scalar):
                continue
            if aname not in access_sets or not access_sets[aname]:
                desc = sdfg.arrays[aname]
                if isinstance(desc, data.Structure) and len(desc.members) > 0:
                    continue
                sdfg.remove_data(aname, validate=False)
                result.add(aname)

        return result or None

    def report(self, pass_retval: Set[str]) -> str:
        return f'Eliminated {len(pass_retval)} arrays: {pass_retval}.'

    def merge_access_nodes(self, state: SDFGState, access_nodes: Dict[str, List[nodes.AccessNode]],
                           condition: Callable[[nodes.AccessNode], bool]):
        """
        Merges access nodes that follow the same conditions together to the first access node.
        """
        removed_nodes: Set[nodes.AccessNode] = set()
        for data_container in access_nodes.keys():
            nodeset = access_nodes[data_container]
            if len(nodeset) > 1:
                # Merge all other access nodes to the first one that fits the condition, if one exists.
                first_node = None
                first_node_idx = 0
                for i, node in enumerate(nodeset[:-1]):
                    if condition(node):
                        first_node = node
                        first_node_idx = i
                        break
                if first_node is None:
                    continue

                # For non-transient data containers, refuse to fold separate
                # source AccessNodes (``in_degree == 0``) when the same
                # container is ALSO written in this state (some AccessNode
                # with ``in_degree > 0``). The topological ordering that
                # keeps each source's downstream reader after the in-state
                # write is only enforced by the presence of distinct
                # predecessor-less nodes; folding the sources frees codegen
                # to reorder a downstream read past the write, and for
                # non-transient externally-observable storage that reorder
                # produces a stale value. Symmetric refusal for sink-merge
                # of a container that is also read in the state. Transients
                # are skipped because the frontend keeps their reads/writes
                # threaded through proper dataflow edges.
                desc = state.sdfg.arrays.get(data_container)
                if desc is not None and not desc.transient:
                    if state.in_degree(first_node) == 0 and any(state.in_degree(n) > 0 for n in nodeset):
                        continue
                    if state.out_degree(first_node) == 0 and any(state.out_degree(n) > 0 for n in nodeset):
                        continue

                # Sibling-carrier guard: source-merging a read-only container
                # ``X`` collapses the topological order that kept ``X``'s
                # distinct source AccessNodes in separate dataflow chains.
                # When a SIBLING transient ``Y`` has both a read-only source
                # (``in_degree == 0``) AND a write-only sink (``out_degree
                # == 0``) in the same state -- i.e. ``Y`` is a per-iteration
                # carrier with a compute chain reading it and a seed-write
                # chain writing the next iteration's value -- the
                # implicit ordering between ``X``'s readers is what kept
                # ``Y``'s read in the compute chain BEFORE the seed-write.
                # Folding ``X``'s sources frees codegen to schedule ``Y``'s
                # seed-write before the compute's read, so the compute reads
                # the NEW value of ``Y`` instead of the carried old value.
                # Pinned by TSVC s254/s255 (``a[i] = (b[i] + x) * 0.5; x =
                # b[i]`` where ``b`` is read-only, ``x`` is the carried
                # scalar). Symmetric guard for sink-merge.
                #
                # The non-transient guard above catches the case where the
                # CARRIER is the same container as the one being merged
                # (``acc[c]`` in s243); this guard catches the orthogonal
                # case where the carrier is a DIFFERENT transient.
                if state.in_degree(first_node) == 0 and len(nodeset) >= 2:
                    if _state_has_read_write_sibling_carrier(state, data_container):
                        continue
                if state.out_degree(first_node) == 0 and len(nodeset) >= 2:
                    if _state_has_read_write_sibling_carrier(state, data_container):
                        continue

                for node in nodeset[first_node_idx + 1:]:
                    if not condition(node):
                        continue

                    # Reconnect edges to first node.
                    # If we are handling views, we do not want to add more than one edge going into a 'views' connector,
                    # so we only merge nodes if the memlets match exactly (which they should). But in that case without
                    # copying the edge.
                    edges: List[MultiConnectorEdge[Memlet]] = state.all_edges(node)
                    other_edges: List[MultiConnectorEdge[Memlet]] = []
                    for edge in edges:
                        if edge.dst is node:
                            if edge.dst_conn == 'views':
                                other_edges = list(state.in_edges_by_connector(first_node, 'views'))
                                if len(other_edges) != 1:
                                    raise InvalidSDFGNodeError('Multiple edges connected to views connector',
                                                               state.sdfg, state.block_id, state.node_id(first_node))
                                other_view_edge = other_edges[0]
                                if other_view_edge.data != edge.data:
                                    # The memlets do not match, skip the node.
                                    continue
                            else:
                                state.add_edge(edge.src, edge.src_conn, first_node, edge.dst_conn, edge.data)
                        else:
                            if edge.src_conn == 'views':
                                other_edges = list(state.out_edges_by_connector(first_node, 'views'))
                                if len(other_edges) != 1:
                                    raise InvalidSDFGNodeError('Multiple edges connected to views connector',
                                                               state.sdfg, state.block_id, state.node_id(first_node))
                                other_view_edge = other_edges[0]
                                if other_view_edge.data != edge.data:
                                    # The memlets do not match, skip the node.
                                    continue
                            else:
                                state.add_edge(first_node, edge.src_conn, edge.dst, edge.dst_conn, edge.data)
                    # Remove merged node and associated edges
                    state.remove_node(node)
                    removed_nodes.add(node)
                access_nodes[data_container] = [n for n in nodeset if n not in removed_nodes]
        return removed_nodes

    def remove_redundant_views(self, sdfg: SDFG, state: SDFGState, access_nodes: Dict[str, List[nodes.AccessNode]]):
        """
        Removes access nodes that contain views, which can be represented normally by memlets. For example, slices.
        """
        removed_nodes: Set[nodes.AccessNode] = set()
        xforms = [RemoveSliceView()]
        state_id = state.block_id

        for nodeset in access_nodes.values():
            for anode in list(nodeset):
                for xform in xforms:
                    # Quick path to setup match
                    candidate = {type(xform).view: anode}
                    xform.setup_match(sdfg, state.parent_graph.cfg_id, state_id, candidate, 0, override=True)

                    # Try to apply
                    if xform.can_be_applied(state, 0, sdfg):
                        xform.apply(state, sdfg)
                        removed_nodes.add(anode)
                        nodeset.remove(anode)
        return removed_nodes

    def remove_redundant_copies(self, sdfg: SDFG, state: SDFGState, removable_data: Set[str],
                                access_nodes: Dict[str, List[nodes.AccessNode]]):
        """
        Removes access nodes that represent redundant copies and/or views.
        """
        removed_nodes: Set[nodes.AccessNode] = set()
        state_id = state.block_id

        # Transformations that remove the first access node
        xforms_first: List[SingleStateTransformation] = [RedundantWriteSlice(), UnsqueezeViewRemove(), RedundantArray()]
        # Transformations that remove the second access node
        xforms_second: List[SingleStateTransformation] = [
            RedundantReadSlice(), SqueezeViewRemove(),
            RedundantSecondArray()
        ]

        # Try the different redundant copy/view transformations on the node
        removed = {1}
        while removed:
            removed = set()
            for aname in removable_data:
                if aname not in access_nodes:  # May be in inter-state edges
                    continue
                for anode in access_nodes[aname]:
                    if anode in removed_nodes:
                        continue
                    if anode not in state.nodes():
                        removed_nodes.add(anode)
                        continue

                    if state.out_degree(anode) == 1:
                        succ = state.successors(anode)[0]
                        if isinstance(succ, nodes.AccessNode):
                            if _is_war_carrier(state, succ.data):
                                continue  # folding anode's read onto a WAR carrier is unsound
                            for xform in xforms_first:
                                # Quick path to setup match
                                candidate = {type(xform).in_array: anode, type(xform).out_array: succ}
                                xform.setup_match(sdfg,
                                                  state.parent_graph.cfg_id,
                                                  state_id,
                                                  candidate,
                                                  0,
                                                  override=True)

                                # Try to apply
                                if xform.can_be_applied(state, 0, sdfg):
                                    ret = xform.apply(state, sdfg)
                                    if ret is not None:  # A view was created
                                        continue
                                    removed_nodes.add(anode)
                                    removed.add(anode)
                                    break

                    if anode in removed_nodes:  # Node was removed, skip second check
                        continue

                    if state.in_degree(anode) == 1:
                        pred = state.predecessors(anode)[0]
                        if isinstance(pred, nodes.AccessNode):
                            if _is_war_carrier(state, pred.data):
                                continue  # folding anode's readers onto a WAR carrier is unsound
                            for xform in xforms_second:
                                # Quick path to setup match
                                candidate = {type(xform).in_array: pred, type(xform).out_array: anode}
                                xform.setup_match(sdfg,
                                                  state.parent_graph.cfg_id,
                                                  state_id,
                                                  candidate,
                                                  0,
                                                  override=True)

                                # Try to apply
                                if xform.can_be_applied(state, 0, sdfg):
                                    ret = xform.apply(state, sdfg)
                                    if ret is not None:  # A view was created
                                        continue
                                    removed_nodes.add(anode)
                                    removed.add(anode)
                                    break

        return removed_nodes
