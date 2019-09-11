""" This module contains classes that implement the map fusion transformation.
"""

from copy import deepcopy as dcpy
from dace import symbolic, types
from dace.graph import nodes, nxutil
from dace.transformation import pattern_matching
from typing import List, Dict, Union
import ast
import networkx as nx


class ASTFindReplace(ast.NodeTransformer):
    def __init__(self, repldict: Dict[str, str]):
        self.repldict = repldict

    def visit_Name(self, node):
        if node.id in self.repldict:
            node.id = self.repldict[node.id]
        return node


def replace(subgraph, name: str, new_name: str):
    """ Finds and replaces all occurrences of a symbol or array in a subgraph.
        @param name: Name to find.
        @param new_name: Name to replace.
    """
    from dace import properties
    import sympy as sp

    symrepl = {
        symbolic.symbol(name):
        symbolic.symbol(new_name) if isinstance(new_name, str) else new_name
    }

    def replsym(symlist):
        if symlist is None:
            return None
        if isinstance(symlist, (symbolic.SymExpr, symbolic.symbol, sp.Basic)):
            return symlist.subs(symrepl)
        for i, dim in enumerate(symlist):
            try:
                symlist[i] = tuple(d.subs(symrepl) for d in dim)
            except TypeError:
                symlist[i] = dim.subs(symrepl)
        return symlist

        # Replace in node properties

    for node in subgraph.nodes():
        for propclass, propval in node.properties():
            pname = propclass.attr_name
            if isinstance(propclass, properties.SymbolicProperty):
                setattr(node, pname, propval.subs({name: new_name}))
            if isinstance(propclass, properties.DataProperty):
                if propval == name:
                    setattr(node, pname, new_name)
            if isinstance(propclass, properties.RangeProperty):
                setattr(node, pname, replsym(propval))
            if isinstance(propclass, properties.CodeProperty):
                for stmt in propval:
                    ASTFindReplace({name: new_name}).visit(stmt)

    # Replace in memlets
    for edge in subgraph.edges():
        if edge.data.data == name:
            edge.data.data = new_name
        edge.data.subset = replsym(edge.data.subset)
        edge.data.other_subset = replsym(edge.data.other_subset)


class MapFusion(pattern_matching.Transformation):
    """ Implements the MapFusion transformation.
        It wil check for all patterns MapExit -> AccessNode -> MapEntry, and
        based on the following rules, fuse them and remove the transient in
        between. There are several possibilities of what it does to this
        transient in between. 

        Essentially, if there is some other place in the
        sdfg where it is required, or if it is not a transient, then it will
        not be removed. In such a case, it will be linked to the MapExit node
        of the new fused map.

        Rules for fusing maps:
          0. The map range of the second map should be a permutation of the
             first map range.
          1. Each of the access nodes that are adjacent to the first map exit
             should have an edge to the second map entry. If it doesn't, then the
             second map entry should not be reachable from this access node.
          2. Any node that has a wcr from the first map exit should not be
             adjacent to the second map entry.
          3. Access pattern for the access nodes in the second map should be
             the same permutation of the map parameters as the map ranges of the
             two maps. Alternatively, this access node should not be adjacent to
             the first map entry.
    """

    _first_map_exit = nodes.ExitNode()
    _some_array = nodes.AccessNode("_")
    _second_map_entry = nodes.EntryNode()

    @staticmethod
    def annotates_memlets():
        return False

    @staticmethod
    def expressions():
        return [
            nxutil.node_path_graph(
                MapFusion._first_map_exit,
                MapFusion._some_array,
                MapFusion._second_map_entry,
            )
        ]

    @staticmethod
    def find_permutation(first_map: nodes.Map,
                         second_map: nodes.Map) -> Union[List[int], None]:
        """ Find permutation between two map ranges.
            @param first_map: First map.
            @param second_map: Second map.
            @return: None if no such permutation exists, otherwise a list of
                     indices L such that L[x]'th parameter of second map has the same range as x'th
                     parameter of the first map.
            """
        result = []

        if len(first_map.range) != len(second_map.range):
            return None

        # Match map ranges with reduce ranges
        for i, tmap_rng in enumerate(first_map.range):
            found = False
            for j, rng in enumerate(second_map.range):
                if tmap_rng == rng and j not in result:
                    result.append(j)
                    found = True
                    break
            if not found:
                break

        # Ensure all map ranges matched
        if len(result) != len(first_map.range):
            return None

        return result

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        first_map_exit = graph.nodes()[candidate[MapFusion._first_map_exit]]
        first_map_entry = graph.entry_node(first_map_exit)
        second_map_entry = graph.nodes()[candidate[
            MapFusion._second_map_entry]]

        for _in_e in graph.in_edges(first_map_exit):
            if _in_e.data.wcr is not None:
                for _out_e in graph.out_edges(second_map_entry):
                    if _out_e.data.data == _in_e.data.data:
                        # wcr is on a node that is used in the second map, quit
                        return False
        # Check whether there is a pattern map -> access -> map.
        intermediate_nodes = set()
        intermediate_data = set()
        for _, _, dst, _, _ in graph.out_edges(first_map_exit):
            if isinstance(dst, nodes.AccessNode):
                intermediate_nodes.add(dst)
                intermediate_data.add(dst.data)
            else:
                return False
        # Check map ranges
        perm = MapFusion.find_permutation(first_map_entry.map,
                                          second_map_entry.map)
        if perm is None:
            return False

        # Create a dict that maps parameters of the first map to those of the
        # second map.
        params_dict = {}
        for _index, _param in enumerate(first_map_entry.map.params):
            params_dict[_param] = second_map_entry.map.params[perm[_index]]

        out_memlets = [e.data for e in graph.in_edges(first_map_exit)]

        # Check that input set of second map is provided by the output set
        # of the first map, or other unrelated maps
        for _, _, _, _, second_memlet in graph.out_edges(second_map_entry):
            # Memlets that do not come from one of the intermediate arrays
            if second_memlet.data not in intermediate_data:
                # however, if intermediate_data eventually leads to
                # second_memlet.data, need to fail.
                for _n in intermediate_nodes:
                    source_node = _n  # graph.find_node(_n.data)
                    destination_node = graph.find_node(second_memlet.data)
                    # NOTE: Assumes graph has networkx version
                    if destination_node in nx.descendants(
                            graph._nx, source_node):
                        return False
                continue

            provided = False
            for first_memlet in out_memlets:
                if first_memlet.data != second_memlet.data:
                    continue
                # If there is an equivalent subset, it is provided
                expected_second_subset = []
                for _tup in first_memlet.subset:
                    new_tuple = []
                    if isinstance(_tup, symbolic.symbol):
                        new_tuple = symbolic.symbol(params_dict[str(_tup)])
                    else:
                        for _sym in _tup:
                            if isinstance(_sym, symbolic.symbol):
                                new_tuple.append(
                                    symbolic.symbol(params_dict[str(_sym)]))
                            else:
                                new_tuple.append(_sym)
                        new_tuple = tuple(new_tuple)
                    expected_second_subset.append(new_tuple)
                if expected_second_subset == list(second_memlet.subset):
                    provided = True
                    break

            # If none of the output memlets of the first map provide the info,
            # fail.
            if provided is False:
                return False

        # Success
        return True

    @staticmethod
    def match_to_str(graph, candidate):
        first_exit = graph.nodes()[candidate[MapFusion._first_map_exit]]
        second_entry = graph.nodes()[candidate[MapFusion._second_map_entry]]

        return " -> ".join(entry.map.label + ": " + str(entry.map.params)
                           for entry in [first_exit, second_entry])

    def apply(self, sdfg):
        """
            This method applies the mapfusion transformation. 
            Other than the removal of the second map entry node (SME), and the first
            map exit (FME) node, it has the following side effects:

            1.  Any transient adjacent to both FME and SME with degree = 2 will be removed. 
                The tasklets that use/produce it shall be connected directly with a 
                scalar/new transient (if the dataflow is more than a single scalar)

            2.  If this transient is adjacent to FME and SME and has other
                uses, it will be adjacent to the new map exit post fusion.
                Tasklet-> Tasklet edges will ALSO be added as mentioned above.

            3.  If an access node is adjacent to FME but not SME, it will be
                adjacent to new map exit post fusion.

            4.  If an access node is adjacent to SME but not FME, it will be
                adjacent to the new map entry node post fusion.

        """
        graph = sdfg.nodes()[self.state_id]
        first_exit = graph.nodes()[self.subgraph[MapFusion._first_map_exit]]
        first_entry = graph.entry_node(first_exit)
        second_entry = graph.nodes()[self.subgraph[
            MapFusion._second_map_entry]]
        second_exit = graph.exit_nodes(second_entry)[0]

        intermediate_nodes = set()
        for _, _, dst, _, _ in graph.out_edges(first_exit):
            intermediate_nodes.add(dst)
            assert isinstance(dst, nodes.AccessNode)

        # Check if an access node refers to non transient memory, or transient
        # is used at another location (cannot erase)
        do_not_erase = set()
        for node in intermediate_nodes:
            if sdfg.arrays[node.data].transient is False:
                do_not_erase.add(node)
            else:
                # If array is used anywhere else in this state.
                num_occurrences = len([
                    n for n in graph.nodes()
                    if isinstance(n, nodes.AccessNode) and n.data == node.data
                ])
                if num_occurrences > 1:
                    return False

                for edge in graph.in_edges(node):
                    if edge.src != first_exit:
                        do_not_erase.add(node)
                        break
                else:
                    for edge in graph.out_edges(node):
                        if edge.dst != second_entry:
                            do_not_erase.add(node)
                            break

        # Find permutation between first and second scopes
        if first_entry.map.params != second_entry.map.params:
            perm = MapFusion.find_permutation(first_entry.map,
                                              second_entry.map)
            params_dict = {}
            for _index, _param in enumerate(first_entry.map.params):
                params_dict[_param] = second_entry.map.params[perm[_index]]

            # Hopefully replaces (in memlets and tasklet) the second scope map
            # indices with the permuted first map indices
            second_scope = graph.scope_subgraph(second_entry)
            for _firstp, _secondp in params_dict.items():
                replace(second_scope, _secondp, _firstp)

        ########Isolate First MapExit node###########
        for _edge in graph.in_edges(first_exit):
            __some_str = _edge.data.data
            _access_node = graph.find_node(__some_str)
            # all outputs of first_exit are in intermediate_nodes set, so all inputs to
            # first_exit should also be!
            if _access_node not in do_not_erase:
                _new_dst = None
                _new_dst_conn = None
                # look at the second map entry out-edges to get the new destination
                for _e in graph.out_edges(second_entry):
                    if _e.data.data == _access_node.data:
                        _new_dst = _e.dst
                        _new_dst_conn = _e.dst_conn
                        break
                if _new_dst is None:
                    # Access node is not even used in the second map
                    graph.remove_node(_access_node)
                    continue
                if _edge.data.data == _access_node.data and isinstance(
                        _edge._src, nodes.AccessNode):
                    _edge.data.data = _edge._src.data
                    _edge.data.subset = "0"
                    graph.add_edge(
                        _edge._src,
                        _edge.src_conn,
                        _new_dst,
                        _new_dst_conn,
                        dcpy(_edge.data),
                    )
                else:
                    if _edge.data.subset.num_elements() == 1:
                        # We will add a scalar
                        local_name = "__s%d_n%d%s_n%d%s" % (
                            self.state_id,
                            graph.node_id(_edge._src),
                            _edge.src_conn,
                            graph.node_id(_edge._dst),
                            _edge.dst_conn,
                        )
                        local_node = sdfg.add_scalar(
                            local_name,
                            dtype=_access_node.desc(graph).dtype,
                            toplevel=False,
                            transient=True,
                            storage=types.StorageType.Register,
                        )
                        _edge.data.data = (
                            local_name)  # graph.add_access(local_name).data
                        _edge.data.subset = "0"
                        graph.add_edge(
                            _edge._src,
                            _edge.src_conn,
                            _new_dst,
                            _new_dst_conn,
                            dcpy(_edge.data),
                        )
                    else:
                        # We will add a transient of size = memlet subset
                        # size
                        local_name = "__s%d_n%d%s_n%d%s" % (
                            self.state_id,
                            graph.node_id(_edge._src),
                            _edge.src_conn,
                            graph.node_id(_edge._dst),
                            _edge.dst_conn,
                        )
                        local_node = graph.add_transient(
                            local_name,
                            _edge.data.subset.size(),
                            dtype=_access_node.desc(graph).dtype,
                            toplevel=False,
                        )
                        _edge.data.data = (
                            local_name)  # graph.add_access(local_name).data
                        _edge.data.subset = ",".join([
                            "0:" + str(_s) for _s in _edge.data.subset.size()
                        ])
                        graph.add_edge(
                            _edge._src,
                            _edge.src_conn,
                            local_node,
                            None,
                            dcpy(_edge.data),
                        )
                        graph.add_edge(local_node, None, _new_dst,
                                       _new_dst_conn, dcpy(_edge.data))
                graph.remove_edge(_edge)
                ####Isolate this node#####
                for _in_e in graph.in_edges(_access_node):
                    graph.remove_edge(_in_e)
                for _out_e in graph.out_edges(_access_node):
                    graph.remove_edge(_out_e)
                graph.remove_node(_access_node)
            else:
                # _access_node will become an output of the second map exit
                for _out_e in graph.out_edges(first_exit):
                    if _out_e.data.data == _access_node.data:
                        graph.add_edge(
                            second_exit,
                            None,
                            _out_e._dst,
                            _out_e.dst_conn,
                            dcpy(_out_e.data),
                        )

                        graph.remove_edge(_out_e)
                        break
                else:
                    raise AssertionError(
                        "No out-edge was found that leads to {}".format(
                            _access_node))
                graph.add_edge(_edge._src, _edge.src_conn, second_exit, None,
                               dcpy(_edge.data))
                ### If the second map needs this node then link the connector
                # that generated this to the place where it is needed, with a
                # temp transient/scalar for memlet to be generated
                for _out_e in graph.out_edges(second_entry):
                    if _out_e.data.data == _access_node.data:
                        if _edge.data.subset.num_elements() == 1:
                            # We will add a scalar
                            local_name = "__s%d_n%d%s_n%d%s" % (
                                self.state_id,
                                graph.node_id(_edge._src),
                                _edge.src_conn,
                                graph.node_id(_edge._dst),
                                _edge.dst_conn,
                            )
                            local_node = sdfg.add_scalar(
                                local_name,
                                dtype=_access_node.desc(graph).dtype,
                                storage=types.StorageType.Register,
                                toplevel=False,
                                transient=True,
                            )
                            _edge.data.data = (
                                local_name
                            )  # graph.add_access(local_name).data
                            _edge.data.subset = "0"
                            graph.add_edge(
                                _edge._src,
                                _edge.src_conn,
                                _out_e._dst,
                                _out_e.dst_conn,
                                dcpy(_edge.data),
                            )
                        else:
                            # We will add a transient of size = memlet subset
                            # size
                            local_name = "__s%d_n%d%s_n%d%s" % (
                                self.state_id,
                                graph.node_id(_edge._src),
                                _edge.src_conn,
                                graph.node_id(_edge._dst),
                                _edge.dst_conn,
                            )
                            local_node = sdfg.add_transient(
                                local_name,
                                _edge.data.subset.size(),
                                dtype=_access_node.desc(graph).dtype,
                                toplevel=False,
                            )
                            _edge.data.data = (
                                local_name
                            )  # graph.add_access(local_name).data
                            _edge.data.subset = ",".join([
                                "0:" + str(_s)
                                for _s in _edge.data.subset.size()
                            ])
                            graph.add_edge(
                                _edge._src,
                                _edge.src_conn,
                                local_node,
                                None,
                                dcpy(_edge.data),
                            )
                            graph.add_edge(
                                local_node,
                                None,
                                _out_e._dst,
                                _out_e.dst_conn,
                                dcpy(_edge.data),
                            )
                        break
                graph.remove_edge(_edge)
        graph.remove_node(first_exit)  # Take a leap of faith

        #############Isolate second_entry node################
        for _edge in graph.in_edges(second_entry):
            _access_node = graph.find_node(_edge.data.data)
            if _access_node in intermediate_nodes:
                # Already handled above, just remove this
                graph.remove_edge(_edge)
                continue
            else:
                # This is an external input to the second map which will now go through the first
                # map.
                graph.add_edge(_edge._src, _edge.src_conn, first_entry, None,
                               dcpy(_edge.data))
                graph.remove_edge(_edge)
                for _out_e in graph.out_edges(second_entry):
                    if _out_e.data.data == _access_node.data:
                        graph.add_edge(
                            first_entry,
                            None,
                            _out_e._dst,
                            _out_e.dst_conn,
                            dcpy(_out_e.data),
                        )
                        graph.remove_edge(_out_e)
                        break
                else:
                    raise AssertionError(
                        "No out-edge was found that leads to {}".format(
                            _access_node))

        graph.remove_node(second_entry)

        # Fix scope exit
        second_exit.map = first_entry.map
        graph.fill_scope_connectors()


pattern_matching.Transformation.register_pattern(MapFusion)
