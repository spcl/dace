from copy import deepcopy
from dace.sdfg import nodes, SDFG
from dace.sdfg.state import SDFGState

import dace.transformation.passes.offloading.OffloadToAccelerator_Phases.offloading_helpers as helpers

class SingleElementCopyOptimization():

    # pattern   A -> single access -> Map    becomes    A -> Map -> single access 
    def apply(self, sdfg:SDFG, verbose=False):
        self.verbose = verbose
        self.single_element_copies_into_map(sdfg)


    def single_element_copies_into_map(self, sdfg):
        changes = set()
        for state in sdfg.states():
            for node in state.nodes():
                if not isinstance(node, nodes.MapEntry):
                    continue
                map_entry : nodes.MapEntry = node

                for input in helpers.get_predecessors(state, map_entry):
                    if not isinstance(input, nodes.AccessNode):
                        continue
                    data_name = input.data
                    if not helpers.is_scalar(data_name, sdfg) and not helpers.is_length1_array(data_name, sdfg):
                        continue
                    single_access = input
                    preds = list(helpers.get_predecessors(state, single_access))
                    if len(preds) != 1 or not isinstance(preds[0], nodes.AccessNode):
                        continue

                    changes.add((state, input, map_entry))

        for state, access, map_entry in changes:
            if self.verbose: print(f"Phase 7: ingest single element copy {access} in state {state} into map {map_entry}")
            self.rewire_access_into_map(state, access, map_entry)


    def rewire_access_into_map(self, state:SDFGState, access: nodes.AccessNode, map: nodes.MapEntry) -> None:
        """
        Move an access node from outside a map to inside the map entry boundary.
        Rewires
            B -> access -> map -> C
        into
            B -> map -> access -> C
        for the connector path(s) that currently route through ``access`` into ``map``.
        """
        # 0) get all edges B -> access
        incoming_to_access = list(state.in_edges(access))
        access_to_map_edges = [edge for edge in state.out_edges(access) if edge.dst is map]
        if not access_to_map_edges:
            raise ValueError(f"Access node '{access.label}' does not feed map '{map.label}'.")

        # 1) connect all inputs of access  to map:      B -> map;           access -> map -> C
        # 2) then connect the new out connectors to access:  B -> map -> access; access -> map -> C
        for idx, edge in enumerate(incoming_to_access):
            src, src_conn = edge.src, edge.src_conn
            ext_memlet = deepcopy(edge.data)
            int_memlet = deepcopy(edge.data)
            state.remove_edge(edge)

            conn_idx = 0
            in_conn = f"IN_REWIRE_ACCESS_{idx}_{conn_idx}"
            out_conn = f"OUT_REWIRE_ACCESS_{idx}_{conn_idx}"
            while in_conn in map.in_connectors or out_conn in map.out_connectors:
                conn_idx += 1
                in_conn = f"IN_REWIRE_ACCESS_{idx}_{conn_idx}"
                out_conn = f"OUT_REWIRE_ACCESS_{idx}_{conn_idx}"

            map.add_in_connector(in_conn)
            map.add_out_connector(out_conn)

            state.add_edge(src, src_conn, map, in_conn, ext_memlet) # B -> map
            state.add_edge(map, out_conn, access, None, int_memlet) # B -> map -> access

        # 3) delete the edge and the in_connector of access: B -> map -> access; map -> C
        accesses_to_map = [e for e in state.in_edges(map) if e.src is access]
        assert len(accesses_to_map) == 1, "multiple edges between same two nodes"
        access_to_map = accesses_to_map[0]

        access_out_conns = self.get_corresponding_out_connectors(map, access_to_map.dst_conn)
        map.remove_in_connector(access_to_map.dst_conn)
        state.remove_edge(access_to_map)

        # 4) delete the out_connectors of access and the edges to the output nodes: B -> map -> access; C
        map_out_edges = [e for e in state.out_edges(map) if e.src_conn in access_out_conns]
        for out_conn in access_out_conns:
            map.remove_out_connector(out_conn)

        # 5) connect the output nodes directly to acces: B -> map -> access -> C
        for e in map_out_edges:
            state.add_edge(access, None, e.dst, e.dst_conn, deepcopy(e.data))
            state.remove_edge(e)

    
    def get_corresponding_out_connectors(self, map_entry: nodes.MapEntry, in_connector: str) -> list[str]:
        if not in_connector:
            return []
        suffix = in_connector[3:] # connectors starts with "IN_"
        return [out_conn  for out_conn in map_entry.out_connectors if out_conn.startswith("OUT_") and out_conn[4:] == suffix]
    
        