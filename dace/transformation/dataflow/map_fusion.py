""" This module contains classes that implement the map fusion transformation.
"""

from copy import deepcopy as dcpy
from dace import data, types, subsets, symbolic
from dace.graph import nodes, nxutil
from dace.transformation import pattern_matching
import sympy
from typing import List, Dict
import ast


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
        replsym(edge.data.subset)
        replsym(edge.data.other_subset)


def calc_set_union(set_a: subsets.Subset,
                   set_b: subsets.Subset) -> subsets.Range:
    """ Computes the union of two Subset objects. """

    if isinstance(set_a, subsets.Indices) or isinstance(
            set_b, subsets.Indices):
        raise NotImplementedError('Set union with indices is not implemented.')
    if not (isinstance(set_a, subsets.Range)
            and isinstance(set_b, subsets.Range)):
        raise TypeError('Can only compute the union of ranges.')
    if len(set_a) != len(set_b):
        raise ValueError('Range dimensions do not match')
    union = []
    for range_a, range_b in zip(set_a, set_b):
        union.append([
            sympy.Min(range_a[0], range_b[0]),
            sympy.Max(range_a[1], range_b[1]),
            sympy.Min(range_a[2], range_b[2]),
        ])
    return subsets.Range(union)


class MapFusion(pattern_matching.Transformation):
    """ Implements the map fusion pattern.

        Map Fusion takes two maps that are connected in series and have the 
        same range, and fuses them to one map. The tasklets in the new map are
        connected in the same manner as they were before the fusion.
    """

    _first_map_entry = nodes.EntryNode()
    _second_map_entry = nodes.EntryNode()

    @staticmethod
    def annotates_memlets():
        return False

    @staticmethod
    def expressions():
        return [nxutil.node_path_graph(MapFusion._first_map_entry)]

    @staticmethod
    def find_permutation(first_map: nodes.Map,
                         second_map: nodes.Map) -> List[int]:
        """ Find permutation between two map ranges.
            @param first_map: First map.
            @param second_map: Second map.
            @return: None if no such permutation exists, otherwise a list of
                     second map indices (such that perm[i] = j).
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
            if not found: break

        # Ensure all map ranges matched
        if len(result) != len(first_map.range):
            return None

        return result

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        first_map_entry = graph.nodes()[candidate[MapFusion._first_map_entry]]
        first_exit = graph.exit_nodes(first_map_entry)[0]
        if any([e.data.wcr is not None for e in graph.in_edges(first_exit)]):
            return False

        # Check whether there is a pattern map -> access -> map.
        intermediate_nodes = set()
        intermediate_data = set()
        for _, _, dst, _, _ in graph.out_edges(first_exit):
            if isinstance(dst, nodes.AccessNode):
                intermediate_nodes.add(dst)
                intermediate_data.add(dst.data)
            else:
                return False

        second_map_entry = None
        for node in intermediate_nodes:
            for _, _, dst, _, _ in graph.out_edges(node):
                if isinstance(dst, nodes.EntryNode):
                    second_map_entry = dst
                    break
        if second_map_entry is None:  # No second map
            return False

        # Check map ranges
        perm = MapFusion.find_permutation(first_map_entry.map,
                                          second_map_entry.map)
        if perm is None:
            return False

        out_memlets = [e.data for e in graph.in_edges(first_exit)]

        # Check that input set of second map is provided by the output set
        # of the first map, or other unrelated maps
        for _, _, _, _, second_memlet in graph.out_edges(second_map_entry):
            # Memlets that do not come from one of the intermediate arrays
            if second_memlet.data not in intermediate_data:
                continue
            provided = False
            for first_memlet in out_memlets:
                if first_memlet.data != second_memlet.data:
                    continue
                # If there is an equivalent subset, it is provided
                permuted_subset = [first_memlet.subset[i] for i in perm]
                if permuted_subset == second_memlet.subset:
                    provided = True
                    break

            # If none of the output memlets of the first map provide the info,
            # fail.
            if provided is False:
                return False

        # Success
        candidate[MapFusion._second_map_entry] = graph.nodes().index(
            second_map_entry)

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        first_entry = graph.nodes()[candidate[MapFusion._first_map_entry]]
        second_entry = graph.nodes()[candidate[MapFusion._second_map_entry]]

        return ' -> '.join(entry.map.label + ': ' + str(entry.map.params)
                           for entry in [first_entry, second_entry])

    def apply(self, sdfg):
        graph = sdfg.nodes()[self.state_id]
        first_entry = graph.nodes()[self.subgraph[MapFusion._first_map_entry]]
        first_exit = graph.exit_nodes(first_entry)[0]
        second_entry = graph.nodes()[self.subgraph[
            MapFusion._second_map_entry]]
        second_exit = graph.exit_nodes(second_entry)[0]

        intermediate_nodes = set()
        for _, _, dst, _, _ in graph.out_edges(first_exit):
            intermediate_nodes.add(dst)

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
        perm = MapFusion.find_permutation(first_entry.map, second_entry.map)

        # TODO: Replace (in memlets and tasklets of the second scope) map
        #       indices with the permuted first map indices
        second_scope = graph.scope_subgraph(second_entry)
        # replace(second_scope, ...)

        # TODO: For every input edge of the second scope (out edges of
        #       second_entry), reconnect to the corresponding edges in the
        #       outputs of the first scope. If tasklet->tasklet, create a
        #       transient SDFG array in the size of the original memlet and use
        #       that as the new tasklet->tasklet edge

        # TODO: If the array is in do_not_erase, reconnect to array through the
        #       second exit node (second_exit)

        # TODO: Remove first_exit, second entry

        # Fix scope exit
        second_exit.map = first_entry.map


pattern_matching.Transformation.register_pattern(MapFusion)
