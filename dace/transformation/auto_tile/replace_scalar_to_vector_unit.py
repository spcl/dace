# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import ast
import copy
import dace
from dace.properties import DictProperty, Property, SetProperty, make_properties
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation

import operator


@make_properties
class ReplaceScalarToVectorUnit(transformation.SingleStateTransformation):
    device_map_entry = transformation.PatternNode(nodes.MapEntry)
    supported_operators = DictProperty(
        key_type=type(operator.add),
        value_type=str,
        default={
            operator.add: "add",
        },
    )
    operator_impl = DictProperty(
        key_type=str,
        value_type=str,
        default={"add": "Add()"},
    )
    mmu_size = Property(dtype=int, default=32)

    def __init__(self):
        super().__init__()

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.device_map_entry)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        return True

    def apply(self, state: SDFGState, sdfg: SDFG):
        dev_entry = self.device_map_entry
        dev_exit = state.exit_node(dev_entry)
        nodes_to_check = state.all_nodes_between(dev_entry, dev_exit)

        # If we encounter a tasklet, we need start the check
        found_mult_add_assign_patterns = set()
        for n in nodes_to_check:
            if isinstance(n, dace.nodes.Tasklet):
                assert state.entry_node(n) is not None

                mult_add_assign_pattern, nodetuple = self.is_mult_add_assign_pattern(
                    sdfg, state, n
                )
                if mult_add_assign_pattern:
                    found_mult_add_assign_patterns.add(nodetuple)

                """
                # Assign patterna fter mult_add_assign as assign is part of it
                assign_pattern, nodetuple = self.is_add_assign_pattern(sdfg, state, n)

                if assign_pattern:
                    print("Found assign pattern starting from,", n, nodetuple)

                """

        for nodetuple in found_mult_add_assign_patterns:
            n = nodetuple[0]
            entry = state.entry_node(n)
            outer_entry = state.entry_node(entry)

            if len(entry.map.params) == 1 and len(outer_entry.map.params) == 2:
                # Remove inner map and replace the nodes with a GEMM tasklet
                _exit = state.exit_node(entry)
                gemm_nodes = state.all_nodes_between(entry, _exit)

                for nd in gemm_nodes:
                    if nd == entry or nd == _exit:
                        continue
                    state.remove_node(nd)

                tasklet = dace.nodes.Tasklet(
                    "GEMM",
                    set(
                        [
                            out_conn.replace("OUT_", "_in_")
                            for out_conn in entry.out_connectors
                        ]
                    ),
                    set(
                        [
                            in_conn.replace("IN_", "_out_")
                            for in_conn in _exit.in_connectors
                        ]
                    ),
                    "",
                )
                for ie in state.in_edges(entry):
                    u, uc, v, vc, memlet = ie
                    state.add_edge(
                        v,
                        vc.replace("IN_", "OUT_"),
                        tasklet,
                        vc.replace("IN_", "_in_"),
                        copy.deepcopy(memlet),
                    )
                for oe in state.out_edges(_exit):
                    u, uc, v, vc, memlet = oe
                    state.add_edge(
                        tasklet,
                        uc.replace("OUT_", "_out_"),
                        u,
                        uc.replace("OUT_", "IN_"),
                        copy.deepcopy(memlet),
                    )
                self.increase_step_size(entry, self.mmu_size)
                self.increase_step_size(outer_entry, self.mmu_size)
                params_to_update = set(entry.map.params).union(outer_entry.map.params)
                params_to_update = [dace.symbol(p) for p in params_to_update]
                self.increase_memlet_range(state, entry, params_to_update)

    def increase_memlet_range(self, state: dace.SDFGState, entry, params_to_update):
        edges = state.all_edges(*state.all_nodes_between(entry, state.exit_node(entry)))
        for e in edges:
            _, _, _, _, memlet = e
            if memlet.data is not None:
                new_subsets = []
                for r in memlet.subset:
                    _r = copy.deepcopy(r)
                    beg, end, step = _r
                    for p in params_to_update:
                        beg = beg.subs(p, p * self.mmu_size)
                        end = end.subs(p, p * self.mmu_size)
                        step = step.subs(p, p * self.mmu_size)
                    _r = (beg, end, step)
                    new_subsets.append(_r)
                _subset = dace.subsets.Range(new_subsets)
                memlet.subset = _subset

    def increase_step_size(self, entry, step_size):
        # Increase the step size of the map
        _map = entry.map
        new_ranges = []
        for r in _map.range:
            beg, end, step = r
            step *= step_size
            new_ranges.append((beg, end, step))
        _map.range = dace.subsets.Range(new_ranges)

    def extract_operators_from_code(self, codelist):
        for code in codelist:
            tree = ast.parse(code)
            operators = dict()
            for node in ast.walk(tree):
                if isinstance(node, ast.BinOp):
                    op_type = type(node.op)
                    if op_type in operators:
                        operators[op_type] += 1
                    else:
                        operators[op_type] = 1
            return operators

    def check_is_assignment(self, codelist):
        for code in codelist:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    # Check if the assignment is of the form Name = Name
                    if isinstance(node.targets[0], ast.Name) and isinstance(
                        node.value, ast.Name
                    ):
                        return True
        return False

    def is_add_assign_pattern(self, sdfg, state, n):
        # Pattern is:
        # Tasklet (add) -> AccessNode -> Tasklet (assign)
        # Or tasklet (add) -> Tasklet (assign)
        if not isinstance(n, dace.nodes.Tasklet):
            return False, ()
        operators = self.extract_operators_from_code(n.code.code)
        assert len(operators) <= 1
        # Check if there is an operator
        if len(operators) == 0:
            return False, ()
        if len(operators) == 1:
            # Check if operator is add
            found_operator = list(operators.keys())[0]
            if found_operator == ast.Add:
                # Continue checking next steps
                out_edges = state.out_edges(n)
                # No outgoing edge or too many return false
                if len(out_edges) != 1:
                    return False, ()
                v = out_edges[0].dst
                dst_tasklet = None
                # If access node contiue to next step
                if isinstance(v, dace.nodes.AccessNode):
                    out_edges = state.out_edges(v)
                    if len(out_edges) != 1:
                        return False
                    vv = out_edges[0].dst
                    if isinstance(vv, dace.nodes.Tasklet):
                        dst_tasklet = vv
                    else:
                        return False, ()
                else:
                    if not isinstance(v, dace.nodes.Tasklet):
                        return False, ()
                    else:
                        dst_tasklet = v
                print(self.check_is_assignment(dst_tasklet.code.code))
                if self.check_is_assignment(dst_tasklet.code.code):
                    return True, (n, v, dst_tasklet)
                else:
                    return False, ()
        return False, ()

    def is_mult_add_assign_pattern(self, sdfg, state, n):
        # Pattern is:
        # Tasklet (mult) -> AccessNode -> Add-Assign pattern
        # Or Tasklet (mult) -> Add-Assign pattern
        if not isinstance(n, dace.nodes.Tasklet):
            return False, ()
        operators = self.extract_operators_from_code(n.code.code)
        assert len(operators) <= 1
        # Check if there is an operator
        if len(operators) == 0:
            return False, ()
        if len(operators) == 1:
            # Check if operator is mult
            found_operator = list(operators.keys())[0]
            if found_operator == ast.Mult:
                # Continue checking next steps
                out_edges = state.out_edges(n)
                # No outgoing edge or too many return false
                if len(out_edges) != 1:
                    return False, ()
                v = out_edges[0].dst
                dst_tasklet = None
                # If access node contiue to next step
                if isinstance(v, dace.nodes.AccessNode):
                    out_edges = state.out_edges(v)
                    if len(out_edges) != 1:
                        return False
                    vv = out_edges[0].dst
                    if isinstance(vv, dace.nodes.Tasklet):
                        dst_tasklet = vv
                    else:
                        return False, ()
                else:
                    if not isinstance(v, dace.nodes.Tasklet):
                        return False, ()
                    else:
                        dst_tasklet = v
                add_assign_pattern, node_tuple = self.is_add_assign_pattern(
                    sdfg, state, dst_tasklet
                )
                if add_assign_pattern:
                    return True, (n, *node_tuple)
                else:
                    return False, ()
        return False, ()

    @staticmethod
    def annotates_memlets():
        return True
