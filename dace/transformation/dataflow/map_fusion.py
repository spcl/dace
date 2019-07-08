""" This module contains classes that implement the map fusion transformation.
"""

from copy import deepcopy as dcpy
from dace import data, types, subsets, symbolic, graph
from dace.graph import nodes, nxutil
from dace.transformation import pattern_matching
import sympy
from typing import List, Dict
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
        symbolic.symbol(name): symbolic.symbol(new_name)
        if isinstance(new_name, str)
        else new_name
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


def calc_set_union(set_a: subsets.Subset, set_b: subsets.Subset) -> subsets.Range:
    """ Computes the union of two Subset objects. """

    if isinstance(set_a, subsets.Indices) or isinstance(set_b, subsets.Indices):
        raise NotImplementedError("Set union with indices is not implemented.")
    if not (isinstance(set_a, subsets.Range) and isinstance(set_b, subsets.Range)):
        raise TypeError("Can only compute the union of ranges.")
    if len(set_a) != len(set_b):
        raise ValueError("Range dimensions do not match")
    union = []
    for range_a, range_b in zip(set_a, set_b):
        union.append(
            [
                sympy.Min(range_a[0], range_b[0]),
                sympy.Max(range_a[1], range_b[1]),
                sympy.Min(range_a[2], range_b[2]),
            ]
        )
    return subsets.Range(union)


class MapFusion(pattern_matching.Transformation):
    """ Implements the map fusion pattern.

        Map Fusion takes two maps that are connected in series and have the 
        same range, and fuses them to one map. The tasklets in the new map are
        connected in the same manner as they were before the fusion.
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
    def find_permutation(first_map: nodes.Map, second_map: nodes.Map) -> List[int]:
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

    # @staticmethod
    # def _find_adjacent_maps(graph, first_exit, intermediate_nodes):
    #    for node in intermediate_nodes:
    #        for _, _, dst, _, _ in graph.out_edges(node):
    #            if isinstance(dst, nodes.EntryNode):
    #                yield dst

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        first_map_exit = graph.nodes()[candidate[MapFusion._first_map_exit]]

        # first_exit = first_map_entry.map.MapExit
        first_map_entry = graph.entry_node(first_map_exit)
        second_map_entry = graph.nodes()[candidate[MapFusion._second_map_entry]]
        for _in_e in graph.in_edges(first_map_exit):
            if _in_e.data.wcr is not None:
                for _out_e in graph.out_edges(second_map_entry):
                    if _out_e.data.data == _in_e.data.data:
                        #wcr is on a node that is used in the second map, quit
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
        perm = MapFusion.find_permutation(first_map_entry.map, second_map_entry.map)
        if perm is None:
            return False

        # Create a dict that maps parameters of the first map to those of the second map.
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
                    source_node = graph.find_node(_n.data)
                    destination_node = graph.find_node(second_memlet.data)
                    if destination_node in nx.descendants(graph._nx, source_node):
                        return False
                    else:
                        continue
                continue

            provided = False
            for first_memlet in out_memlets:
                if first_memlet.data != second_memlet.data:
                    continue
                # If there is an equivalent subset, it is provided
                first_subset = first_memlet.subset
                expected_second_subset = []
                for _tup in first_memlet.subset:
                    new_tuple = []
                    if isinstance(_tup, symbolic.symbol):
                        new_tuple = symbolic.symbol(params_dict[str(_tup)])
                    else:
                        for _sym in _tup:
                            if isinstance(_sym, symbolic.symbol):
                                new_tuple.append(
                                    symbolic.symbol(params_dict[str(_sym)])
                                )
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
        # candidate[MapFusion._second_map_entry] = graph.nodes().index(second_map_entry)

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        first_exit = graph.nodes()[candidate[MapFusion._first_map_exit]]
        second_entry = graph.nodes()[candidate[MapFusion._second_map_entry]]

        return " -> ".join(
            entry.map.label + ": " + str(entry.map.params)
            for entry in [first_exit, second_entry]
        )

    def apply(self, sdfg):
        graph = sdfg.nodes()[self.state_id]
        first_exit = graph.nodes()[self.subgraph[MapFusion._first_map_exit]]
        first_entry = graph.entry_node(first_exit)
        second_entry = graph.nodes()[self.subgraph[MapFusion._second_map_entry]]
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
                num_occurrences = len(
                    [
                        n
                        for n in graph.nodes()
                        if isinstance(n, nodes.AccessNode) and n.data == node.data
                    ]
                )
                if num_occurrences > 1:
                    return False

                # TODO: NOT SURE
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
            perm = MapFusion.find_permutation(first_entry.map, second_entry.map)
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
                    #_access_node is used somewhere else, but not in the second
                    #map
                    continue
                if _edge.data.data == _access_node.data and isinstance(
                    _edge._src, nodes.AccessNode
                ):
                    _edge.data.data = _edge._src.data
                    _edge.data.subset = "0"
                graph.add_edge(
                    _edge._src,
                    _edge.src_conn,
                    _new_dst,
                    _new_dst_conn,
                    dcpy(_edge.data),
                )
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
                        "No out-edge was found that leads to {}".format(_access_node)
                    )
                graph.add_edge(
                    _edge._src, _edge.src_conn, second_exit, None, dcpy(_edge.data)
                )
                ### If the second map needs this node then link the connector
                # that generated this to the place where it is needed
                for _out_e in graph.out_edges(second_entry):
                    if _out_e.data.data == _access_node.data:
                        graph.add_edge(
                            _edge._src,
                            _edge.src_conn,
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
                graph.add_edge(
                    _edge._src, _edge.src_conn, first_entry, None, dcpy(_edge.data)
                )
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
                        "No out-edge was found that leads to {}".format(_access_node)
                    )

        graph.remove_node(second_entry)

        # Fix scope exit
        second_exit.map = first_entry.map
        graph.fill_scope_connectors()


pattern_matching.Transformation.register_pattern(MapFusion)
