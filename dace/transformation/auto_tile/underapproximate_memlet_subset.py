# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" This module contains classes and functions that implement the grid-strided map tiling
    transformation."""


from dace.memlet import Memlet
from dace.sdfg import SDFG, SDFGState
from dace.properties import make_properties
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from dace import subsets
from dace import symbolic
import sympy
from functools import reduce

def filter_symbol(expressions, symbols):
    found_expressions = set()
    for expr in expressions:
        free_symbols = expr.free_symbols
        free_symbol_strings = [str(x) for x in free_symbols]
        for symbol in symbols:
            if expr.has(symbol) or str(symbol) in free_symbol_strings:
                found_expressions.add(expr)
    return found_expressions.pop()

def filter_int(expressions, int_sym):
    found_expressions = set()
    for expr in expressions:
        if (not expr.has(int_sym)) and expr != int_sym:
            found_expressions.add(expr)
    return found_expressions.pop()

def contains_floor_or_min(expr):
    for node in sympy.preorder_traversal(expr):
        if isinstance(node, sympy.Min) or isinstance(node, symbolic.int_floor):
            return True
    return False


@make_properties
class UnderApprorixmateMemletSubsets(transformation.SingleStateTransformation):
    """
    """

    map_entry = transformation.PatternNode(nodes.MapEntry)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        return True

    def apply(self, graph: SDFGState, sdfg: SDFG):
        map_entry = self.map_entry

        map_exit = graph.exit_node(map_entry)

        edges_to_remove = []
        edges_to_add = []
        for edge in graph.out_edges(map_entry) + graph.in_edges(map_exit):
            u, u_conn, v, v_conn, memlet = edge
            memlet_ranges : subsets.Range = memlet.subset
            new_memlet_ranges = []
            if memlet_ranges != None:
                for (beg, end, step) in memlet_ranges:
                    # Consider the expression in the tree form, transform the tree following these 2 rules,
                    # Until there are no Mins or int_floors left
                    # If calling, one side needs to contain the map parameter and the other side needs to be a dimension check
                    # Discard the Min node and

                    if (len(beg.free_symbols) >= 1):
                        expr_to_approximate = sympy.expand(end)
                        new_range_end = None
                        while contains_floor_or_min(expr_to_approximate):
                            filtered_args = []
                            if isinstance(expr_to_approximate, sympy.Min):
                                min_expr_args = expr_to_approximate.args
                                filtered = filter_symbol(min_expr_args, beg.free_symbols)
                                filtered_args.append(filtered)
                            elif isinstance(expr_to_approximate, sympy.Mul) and \
                                isinstance(expr_to_approximate.args[0], sympy.Integer) and \
                                isinstance(expr_to_approximate.args[1], symbolic.int_floor):
                                floor_expr_args = expr_to_approximate.args[1].args
                                filtered = filter_symbol(floor_expr_args, beg.free_symbols)
                                filtered_args.append(filtered)
                                filtered_args.append(-expr_to_approximate.args[0]+1)
                            else:
                                for expr in expr_to_approximate.args:
                                    if isinstance(expr, sympy.Min):
                                        min_expr_args = expr.args
                                        filtered = filter_symbol(min_expr_args, beg.free_symbols)
                                        filtered_args.append(filtered)
                                    elif isinstance(expr, sympy.Mul) and \
                                        isinstance(expr.args[0], sympy.Integer) and \
                                        isinstance(expr.args[1], symbolic.int_floor):
                                        floor_expr_args = expr.args[1].args
                                        filtered = filter_symbol(floor_expr_args, beg.free_symbols)
                                        filtered_args.append(filtered)
                                        filtered_args.append(-expr.args[0]+1)
                                    else:
                                        filtered_args.append(expr)
                            new_range_end = sympy.Add(*filtered_args)
                            expr_to_approximate = new_range_end
                            new_range_end = expr_to_approximate
                        if new_range_end == None:
                            new_range_end = expr_to_approximate
                    else:
                        new_range_end = end
                    assert(step == 1 or step == sympy.Integer(1))
                    new_memlet_ranges.append((beg, new_range_end, step))
            else:
                new_memlet_ranges = None
            if new_memlet_ranges == None:
                new_memlet = Memlet(subset=None, data=memlet.data)
            else:
                new_memlet = Memlet(subset=subsets.Range(new_memlet_ranges), data=memlet.data)
            edges_to_add.append((u, u_conn, v, v_conn, new_memlet))
            edges_to_remove.append(edge)

        for edge in edges_to_remove:
            graph.remove_edge(edge)
        for edge in edges_to_add:
            graph.add_edge(*edge)


    @staticmethod
    def annotates_memlets():
        return True