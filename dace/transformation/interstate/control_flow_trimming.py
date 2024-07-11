# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" State fusion transformation """

from typing import Dict, List, Set

import networkx as nx

from dace import data as dt, dtypes, registry, sdfg as sd, subsets as sbs
from dace.config import Config
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.state import SDFGState
from dace.transformation import transformation
from copy import deepcopy
from dace.transformation.interstate.sdfg_nesting import InlineSDFG
import sympy as sp
import z3
from dace.transformation.dataflow import TrivialMapElimination
import dace

def sympy_to_z3(expr):
    if isinstance(expr, int):
        return z3.IntVal
    elif isinstance(expr, float):
        return z3.RealVal(expr)

    # Define Z3 variables
    symbols = {str(s): z3.Int(str(s)) for s in expr.free_symbols}
    
    def _convert(sympy_expr):
        if isinstance(sympy_expr, sp.Rel):
            lhs = _convert(sympy_expr.lhs)
            rhs = _convert(sympy_expr.rhs)
            if isinstance(sympy_expr, sp.GreaterThan):
                return lhs >= rhs
            elif isinstance(sympy_expr, sp.StrictGreaterThan):
                return lhs > rhs
            elif isinstance(sympy_expr, sp.LessThan):
                return lhs <= rhs
            elif isinstance(sympy_expr, sp.StrictLessThan):
                return lhs < rhs
            elif isinstance(sympy_expr, sp.Equality):
                return lhs == rhs
            elif isinstance(sympy_expr, sp.Unequality):
                return lhs != rhs
            else:
                raise ValueError(f"Unsupported relational operator: {type(sympy_expr)}")
        elif isinstance(sympy_expr, sp.Symbol):
            return symbols[str(sympy_expr)]
        elif isinstance(sympy_expr, sp.Integer):
            return z3.IntVal(sympy_expr)
        elif isinstance(sympy_expr, sp.Add):
            return sum(_convert(arg) for arg in sympy_expr.args)
        elif isinstance(sympy_expr, sp.Mul):
            result = _convert(sympy_expr.args[0])
            for arg in sympy_expr.args[1:]:
                result *= _convert(arg)
            return result
        elif isinstance(sympy_expr, sp.Pow):
            base = _convert(sympy_expr.args[0])
            exponent = _convert(sympy_expr.args[1])
            return base ** exponent
        elif isinstance(sympy_expr, sp.Rational):
            return _convert(sympy_expr.p) / _convert(sympy_expr.q)
        elif isinstance(sympy_expr, sp.Float):
            return z3.RealVal(sympy_expr)
        if isinstance(expr, int):
            return z3.IntVal
        elif isinstance(expr, float):
            return z3.RealVal(expr)
        else:
            raise ValueError(f"Unsupported expression type: {type(sympy_expr)}")
    
    return _convert(expr)

class TrimControlFlow(transformation.SingleStateTransformation):
    """ 
    """

    map_entry = transformation.PatternNode(nodes.MapEntry)
    nested_sdfg = transformation.PatternNode(nodes.NestedSDFG)
    map_exit = transformation.PatternNode(nodes.MapExit)

    @staticmethod
    def annotates_memlets():
        return False

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry, cls.nested_sdfg, cls.map_exit)]


    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        map_entry = self.map_entry
        nested_sdfg = self.nested_sdfg
        map_exit = self.map_exit

        # skip if hard to analyze assignment to the index variable
        for e in nested_sdfg.sdfg.edges():
            if map_entry.map.params[0] in  e.data.assignments.keys():
                return False

        return True
    
    @staticmethod
    def _eliminate_branch(sdfg: sd.SDFG, initial_state: sd.SDFGState):
        state_list = [initial_state]
        while len(state_list) > 0:
            new_state_list = []
            for s in state_list:
                for e in sdfg.out_edges(s):
                    if len(sdfg.in_edges(e.dst)) == 1:
                        new_state_list.append(e.dst)
                sdfg.remove_node(s)
            state_list = new_state_list

    def apply(self, node, sdfg: sd.SDFG):
        map_entry = self.map_entry
        nested_sdfg = self.nested_sdfg

        index = z3.Int(map_entry.map.params[0])
        range_begin = map_entry.map.range.min_element()[0]
        range_end = map_entry.map.range.max_element()[0]
        begin_eq = index >= sympy_to_z3(range_begin)
        end_eq = index <= sympy_to_z3(range_end)
        # if I am inside the loop body then this always holds (for positive stride)
        range_eq = sympy_to_z3(range_begin) < sympy_to_z3(range_end)

        found = True

        while found:
            found = False
            for e in nested_sdfg.sdfg.edges():
                if e.data.is_unconditional():
                    continue
                cond = e.data.condition_sympy()
                if isinstance(cond, sp.Rel):
                    z3_cond = sympy_to_z3(cond)
                    solver = z3.Solver()
                    solver.add(begin_eq)
                    solver.add(end_eq)
                    solver.add(z3_cond)
                    solver.add(range_eq)
                    if solver.check() == z3.unsat:
                        TrimControlFlow._eliminate_branch(nested_sdfg.sdfg, e.dst)
                        if len(nested_sdfg.sdfg.out_edges(e.src)) == 1:
                            nested_sdfg.sdfg.out_edges(e.src)[0].data.condition = dace.properties.CodeBlock("1")
                        found = True
                        break

class MapSplit(transformation.SingleStateTransformation):
    """ 
    """

    map_entry = transformation.PatternNode(nodes.MapEntry)
    nested_sdfg = transformation.PatternNode(nodes.NestedSDFG)
    map_exit = transformation.PatternNode(nodes.MapExit)

    @staticmethod
    def annotates_memlets():
        return False

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry, cls.nested_sdfg, cls.map_exit)]


    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        map_entry = self.map_entry

        range_begin = map_entry.map.range.min_element()[0]
        range_end = map_entry.map.range.max_element()[0]

        # avoid degenerate map
        if range_begin == range_end:
            return False

        return True
    
    @staticmethod
    def _eliminate_branch(sdfg: sd.SDFG, initial_state: sd.SDFGState):
        state_list = [initial_state]
        while len(state_list) > 0:
            new_state_list = []
            for s in state_list:
                for e in sdfg.out_edges(s):
                    if len(sdfg.in_edges(e.dst)) == 1:
                        new_state_list.append(e.dst)
                sdfg.remove_node(s)
            state_list = new_state_list
                        
    def apply(self, state: sd.SDFGState, sdfg: sd.SDFG):
        map_entry = self.map_entry
        nested_sdfg = self.nested_sdfg
        map_exit = self.map_exit

        index = map_entry.map.params[0]
        range_begin = map_entry.map.range.min_element()[0]
        range_end = map_entry.map.range.max_element()[0]

        breaking_point = None
        for n in nested_sdfg.sdfg.nodes():
            for e in nested_sdfg.sdfg.out_edges(n):
                if e.data.is_unconditional():
                    continue
                cond = e.data.condition_sympy()
                if isinstance(cond, sp.Eq):
                    breaking_point = cond.rhs
        
        print(breaking_point)

        if breaking_point is None:
            return

        # breaking_target = deepcopy(nested_sdfg)
        # breaking_target.label = dt.find_new_name(breaking_target.label, sdfg._labels)
        # node.add_node(breaking_target)
        # breaking_point_node = breaking_target.sdfg.add_state('breaking_point')
        # sdfg.add_edge(breaking_point_node, node, sd.InterstateEdge())

        entry_degenerate = deepcopy(map_entry)
        nested_degenerate = deepcopy(nested_sdfg)
        exit_degenerate = deepcopy(map_exit)
        state.add_nodes_from([entry_degenerate, nested_degenerate, exit_degenerate])
        for e in state.in_edges(map_entry):
            state.add_edge(e.src, e.src_conn, entry_degenerate, e.dst_conn, e.data)
        for e in state.out_edges(map_entry):
            state.add_edge(entry_degenerate, e.src_conn, nested_degenerate, e.dst_conn, e.data)
        for e in state.out_edges(nested_sdfg):
            state.add_edge(nested_degenerate, e.src_conn, exit_degenerate, e.dst_conn, e.data)
        for e in state.out_edges(map_exit):
            state.add_edge(exit_degenerate, e.src_conn, e.dst, e.dst_conn, e.data)

        entry_degenerate.map.range = sbs.Range([(breaking_point, breaking_point, 1)])
        exit_degenerate.map.range = sbs.Range([(breaking_point, breaking_point, 1)])

        trim_flow = TrimControlFlow()
        trim_flow.map_entry = entry_degenerate
        trim_flow.nested_sdfg = nested_degenerate
        trim_flow.map_exit = exit_degenerate
        trim_flow.apply(state, sdfg)

        map_elimination = TrivialMapElimination()
        map_elimination.map_entry = entry_degenerate
        map_elimination.apply(state, sdfg)

        if (range_begin == breaking_point) is True:
            map_entry.map.range = sbs.Range([(breaking_point+1, range_end, 1)])
            map_exit.map.range = sbs.Range([(breaking_point+1, range_end, 1)])
        elif (range_end == breaking_point) is True:
            map_entry.map.range = sbs.Range([(range_begin, breaking_point-1, 1)])
            map_exit.map.range = sbs.Range([(range_begin, breaking_point-1, 1)])
        else:
            entry_half = deepcopy(map_entry)
            nested_half = deepcopy(nested_sdfg)
            exit_half = deepcopy(map_exit)
            state.add_nodes_from([entry_half, nested_half, exit_half])
            for e in state.in_edges(map_entry):
                state.add_edge(e.src, e.src_conn, entry_half, e.dst_conn, e.data)
            for e in state.out_edges(map_entry):
                state.add_edge(entry_half, e.src_conn, nested_half, e.dst_conn, e.data)
            for e in state.out_edges(nested_sdfg):
                state.add_edge(nested_half, e.src_conn, exit_half, e.dst_conn, e.data)
            for e in state.out_edges(map_exit):
                state.add_edge(exit_half, e.src_conn, e.dst, e.dst_conn, e.data)
            
            map_entry.map.range = sbs.Range([(range_begin, breaking_point-1, 1)])
            map_exit.map.range = sbs.Range([(range_begin, breaking_point-1, 1)])

            entry_half.map.range = sbs.Range([(breaking_point+1, range_end, 1)])
            exit_half.map.range = sbs.Range([(breaking_point+1, range_end, 1)])

            trim_flow.map_entry = entry_half
            trim_flow.nested_sdfg = nested_half
            trim_flow.map_exit = exit_half
            trim_flow.apply(state, sdfg)
        
        trim_flow.map_entry = map_entry
        trim_flow.nested_sdfg = nested_sdfg
        trim_flow.map_exit = map_exit
        trim_flow.apply(state, sdfg)
