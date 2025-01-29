# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

from collections import defaultdict
from typing import Dict, Optional, Set, Tuple

from dace import SDFG, SDFGState, symbolic
from dace.sdfg import nodes as nd
from dace.sdfg.replace import replace_datadesc_names
from dace.transformation import pass_pipeline as ppl
from dace import data as dt

from dace.transformation.helpers import state_fission_after
from dace.transformation.passes.analysis.analysis import FindAccessNodes
from dace.transformation.transformation import explicit_cf_compatible


@explicit_cf_compatible
class ExtractNestedStructAccesses(ppl.Pass):
    """
    TODO
    """

    CATEGORY: str = 'Simplification'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.AccessNodes | ppl.Modifies.Tasklets | ppl.Modifies.Memlets | ppl.Modifies.CFG

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.AccessNodes & ppl.Modifies.Tasklets & ppl.Modifies.Memlets | ppl.Modifies.CFG

    def depends_on(self):
        return {}

    def _extract_struct_accesses_nsdfg(self, parent_state: SDFGState, nsdfg_node: nd.NestedSDFG):
        nsdfg = nsdfg_node.sdfg
        removal_candidates = set()
        replacements = dict()
        new_in_conns = dict()
        new_out_conns = dict()
        in_edges = dict()
        out_edges = dict()
        #for in_conn in nsdfg_node.in_connectors.keys():
        #    if isinstance(nsdfg.arrays[in_conn], dt.Structure):
        #        structs[in_conn] = nsdfg.arrays[in_conn]
        #        iedge = parent_state.in_edges_by_connector(nsdfg_node, in_conn)[0]
        #        in_edges[in_conn] = iedge
        #for out_conn in nsdfg_node.out_connectors.keys():
        #    if isinstance(nsdfg.arrays[out_conn], dt.Structure):
        #        structs[out_conn] = nsdfg.arrays[out_conn]
        #        oedges = parent_state.out_edges_by_connector(nsdfg_node, out_conn)
        #        out_edges[out_conn] = oedges
        for block in nsdfg.all_control_flow_blocks():
            if isinstance(block, SDFGState):
                for dnode in block.data_nodes():
                    if '.' in dnode.data:
                        root_data = dnode.root_data
                        if (isinstance(nsdfg.arrays[root_data], dt.Structure) and root_data in nsdfg_node.in_connectors
                                or root_data in nsdfg_node.out_connectors):
                            removal_candidates.add(root_data)
                            is_read = any([oe.data.data is not None for oe in block.out_edges(dnode)])
                            is_write = any([ie.data.data is not None for ie in block.in_edges(dnode)])
                            if dnode.data not in replacements:
                                new_name = dnode.data.replace('.', '_')
                                new_name = nsdfg.add_datadesc(new_name, nsdfg.arrays[dnode.data], find_new_name=True)
                                replacements[dnode.data] = new_name
                            if is_read:
                                new_in_conns[replacements[dnode.data]] = dnode.data
                            if is_write:
                                new_out_conns[replacements[dnode.data]] = dnode.data

        symbolic.safe_replace(replacements,
                              lambda d: replace_datadesc_names(nsdfg, d, skip_repository=True),
                              value_as_string=True)
        ...

    def _apply_to_sdfg(self, sdfg: SDFG):
        for cfg in sdfg.all_control_flow_regions():
            for block in cfg.nodes():
                if isinstance(block, SDFGState):
                    for node in block.nodes():
                        if isinstance(node, nd.NestedSDFG):
                            self._extract_struct_accesses_nsdfg(block, node)

    def apply_pass(self, sdfg: SDFG, _) -> Optional[Dict[str, Set[str]]]:
        """
        TODO
        """
        result = defaultdict(set)

        self._apply_to_sdfg(sdfg)

        return result

    def report(self, pass_retval: Optional[Dict[str, Set[str]]]) -> Optional[str]:
        return 'No modifications performed'


@explicit_cf_compatible
class LowerStructViews(ppl.Pass):
    """
    TODO
    """

    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return (ppl.Modifies.Descriptors | ppl.Modifies.AccessNodes | ppl.Modifies.Tasklets | ppl.Modifies.Memlets |
                ppl.Modifies.CFG)

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.AccessNodes & ppl.Modifies.Tasklets & ppl.Modifies.Memlets & ppl.Modifies.CFG

    def depends_on(self):
        return {FindAccessNodes}

    def _preprocess_state(self, state: SDFGState):
        for dn in state.data_nodes():
            if isinstance(dn.desc(state.sdfg), dt.Structure):
                if state.in_degree(dn) > 0 and state.out_degree(dn) > 0:
                    nstate = state_fission_after(state, dn)
                    self._preprocess_state(state)
                    self._preprocess_state(nstate)
                    return

    def _preprocess_sdfg(self, sdfg: SDFG):
        """
        Preprocess the SDFG by splitting / fissioning states each time we have an access node to a structure that is
        being used for both reads and writes. We do this to move the reads in a subsequent state, guaranteeing the
        lowering does not violate any sequential dependencies.
        """
        for sd in sdfg.all_sdfgs_recursive():
            for cfg in sd.all_control_flow_regions():
                for block in cfg.nodes():
                    if isinstance(block, SDFGState):
                        self._preprocess_state(block)

    def apply_pass(self, sdfg, pipeline_results):
        results = {}

        self._preprocess_sdfg(sdfg)

        for sd in sdfg.all_sdfgs_recursive():
            anodes: Dict[SDFGState, Tuple[Set[nd.AccessNode],
                                          Set[nd.AccessNode]]] = pipeline_results[FindAccessNodes.__name__][sd.cfg_id]
            replacements = {}
            for cfg in sd.all_control_flow_regions():
                for block in cfg.nodes():
                    if isinstance(block, SDFGState):
                        state = block
                        removed_nodes = set()
                        for dn in state.data_nodes():
                            if dn in removed_nodes:
                                continue

                            desc = dn.desc(sd)
                            if isinstance(desc, dt.Structure):
                                read_edges = state.out_edges(dn)
                                write_edges = state.in_edges(dn)

                                if len(read_edges) > 0 and len(write_edges) > 0:
                                    raise RuntimeError('Found a structure attempting to be lowered with both ' +
                                                       'reads and writes')

                                added_writes: Set[nd.AccessNode] = set()
                                added_reads: Set[nd.AccessNode] = set()
                                for write_edge in write_edges:
                                    if isinstance(write_edge.src, nd.AccessNode):
                                        src_desc = write_edge.src.desc(sd)
                                        parts = write_edge.data.data.split('.') if write_edge.data.data else []
                                        if (isinstance(src_desc, dt.View)
                                                and not isinstance(src_desc, dt.StructureView) and len(parts) == 2):
                                            member = parts[1]
                                            full_data = dn.data + '.' + member
                                            for ie in state.in_edges(write_edge.src):
                                                if ie.data.data == write_edge.src.data:
                                                    for e in state.memlet_tree(ie):
                                                        if e.data.data == write_edge.src.data:
                                                            e.data.data = full_data
                                            write_edge.src.data = full_data
                                            write_edge.src.remove_out_connector('views')
                                            added_writes.add(write_edge.src)
                                            state.remove_edge(write_edge)

                                for read_edge in read_edges:
                                    if isinstance(read_edge.dst, nd.AccessNode):
                                        dst_desc = read_edge.dst.desc(sd)
                                        parts = read_edge.data.data.split('.') if read_edge.data.data else []
                                        if (isinstance(dst_desc, dt.View)
                                                and not isinstance(dst_desc, dt.StructureView) and len(parts) == 2):
                                            member = parts[1]
                                            full_data = dn.data + '.' + member
                                            out_edges = state.out_edges(read_edge.dst)
                                            if len(out_edges) == 0:
                                                # This is a view that is being prepared, but is not attached to anything
                                                # reading it. This indicates that it was added for a read on an
                                                # interstate edge or control flow block meta access. If that is the
                                                # case, we may remove it, but only if it is being written to only once
                                                # in the entire SDFG.
                                                found_other_writes = 0
                                                for s in sd.states():
                                                    for anode in anodes[read_edge.dst.data][s][1]:
                                                        if anode is read_edge.dst:
                                                            continue
                                                        found_other_writes += 1
                                                if found_other_writes == 0:
                                                    replacements[read_edge.dst.data] = full_data
                                                    state.remove_edge(read_edge)
                                                    state.remove_node(read_edge.dst)
                                            else:
                                                for oe in out_edges:
                                                    if oe.data.data == read_edge.dst.data:
                                                        for e in state.memlet_tree(oe):
                                                            if e.data.data == read_edge.dst.data:
                                                                e.data.data = full_data
                                                read_edge.dst.data = full_data
                                                read_edge.dst.remove_in_connector('views')
                                                added_reads.add(read_edge.dst)
                                                state.remove_edge(read_edge)

                                if state.out_degree(dn) == 0 and state.in_degree(dn) == 0:
                                    state.remove_node(dn)
                                    removed_nodes.add(dn)

                                #if len(added_reads) > 0 and len(added_writes) > 0:
                                #    for a_read in added_reads:
                                #        for a_write in added_writes:
                                #            if a_write.data == a_read.data:
                                #                for oe in state.out_edges(a_read):
                                #                    state.add_edge(a_write, None, oe.dst, oe.dst_conn, oe.data)
                                #                    state.remove_node(a_read)

            # Perform any replacements of meta accesses and interstate edge reads.
            if replacements:
                sd.replace_dict(replacements, replace_in_graph=True, replace_keys=False)

        return results
