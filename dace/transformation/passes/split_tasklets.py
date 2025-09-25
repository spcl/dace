# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import dace
from typing import Dict, Optional, Set

from dace import SDFG
from dace.transformation import pass_pipeline as ppl, transformation

import ast
from dace.sdfg.nodes import CodeBlock

import re


class ASTSplitter:

    def __init__(self):
        self.n = 0
        self.stmts = []

    def temp(self):
        t = f"__t{self.n}"
        self.n += 1
        return t

    def visit(self, node):
        if isinstance(node, ast.BinOp):
            l, r = self.visit(node.left), self.visit(node.right)
            t = self.temp()
            ops = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/', ast.Pow: '**'}
            self.stmts.append(f"{t} = {l} {ops[type(node.op)]} {r}")
            return t
        elif isinstance(node, ast.UnaryOp):
            op = self.visit(node.operand)
            t = self.temp()
            ops = {ast.USub: '-', ast.UAdd: '+'}
            self.stmts.append(f"{t} = {ops[type(node.op)]}{op}")
            return t
        elif isinstance(node, ast.Call):
            func = self.visit(node.func)
            args = [self.visit(arg) for arg in node.args]
            t = self.temp()
            self.stmts.append(f"{t} = {func}({', '.join(args)})")
            return t
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Attribute):
            return f"{self.visit(node.value)}.{node.attr}"
        return ast.unparse(node)


def to_ssa(code):
    tree = ast.parse(code).body[0]
    ssa = ASTSplitter()
    if isinstance(tree, ast.Assign):
        target = tree.targets[0].id if isinstance(tree.targets[0], ast.Name) else ast.unparse(tree.targets[0])
        # Check if RHS is already a simple variable or constant
        if isinstance(tree.value, (ast.Name, ast.Constant)):
            ssa.stmts.append(f"{target} = {ssa.visit(tree.value)}")
        else:
            rhs = ssa.visit(tree.value)
            # Replace the last temp variable with the target
            if ssa.stmts and ssa.stmts[-1].startswith(rhs + ' ='):
                ssa.stmts[-1] = ssa.stmts[-1].replace(rhs, target, 1)
            else:
                ssa.stmts.append(f"{target} = {rhs}")
    else:
        ssa.visit(tree.value if isinstance(tree, ast.Expr) else tree)
    return ssa.stmts


class VarCollector(ast.NodeVisitor):

    def __init__(self):
        self.vars = set()

    def visit_Name(self, node):
        # only collect plain variable names
        self.vars.add(node.id)

    def visit_Attribute(self, node):
        # don't collect attributes like `dace.int64`
        self.generic_visit(node)


def _get_vars(ssa_line: str):
    lhs, rhs = ssa_line.split('=', 1)
    lhs_var = lhs.strip()

    tree = ast.parse(rhs.strip(), mode="eval")
    collector = VarCollector()
    collector.visit(tree)

    return [lhs_var], list(collector.vars)


@transformation.explicit_cf_compatible
class SplitTasklets(ppl.Pass):
    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Tasklets | ppl.Modifies.Descriptors | ppl.Modifies.AccessNodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.Tasklets

    def depends_on(self):
        return {}

    tmp_access_identifier = "_split_"

    def apply_pass(self, sdfg: SDFG, pipeline_results) -> Optional[Dict[str, Set[str]]]:
        split_access_counter = 0

        tasklets_to_split = list()  # tasklet, parent_graph, ssa_statements
        for n, g in sdfg.all_nodes_recursive():
            if isinstance(n, dace.nodes.Tasklet):
                c: CodeBlock = n.code
                # Can't split a tasklet that has >1 outputs
                if len(n.out_connectors) > 1:
                    continue
                input_types = set()
                for ie in g.in_edges(n):
                    if ie.data is None:
                        continue
                    if ie.data.data is None:
                        continue
                    input_types.add(g.sdfg.arrays[ie.data.data].dtype)
                # It is complicated to split a tasklet with mixed precision input
                # Need to bookkeep the mapping of intermediate results to precision
                if len(input_types) > 1 or len(input_types) == 0:
                    continue
                input_type = next(iter(input_types))
                if c.language == dace.dtypes.Language.Python:
                    ssa_statements = to_ssa(c.as_string)
                    if len(ssa_statements) != 1:
                        tasklets_to_split.append((n, g, ssa_statements, input_type))

        # Previous tasklet:
        # i1 -> |         |
        # i2 -> | tasklet | -> o1
        # i3 -> |         |
        # Now they need to be split e.g.:
        # i1 -> |         |
        # i2 -> | tasklet | -> tmp1 -> |         |
        #                        i3 -> | tasklet | -> o1

        # For the case a tasklet goes to a taskelt that needs to be split
        # If we have t1 -> t2 but then split t1 to (t1.1, t1.2) -> t2
        # For each tasklet we split we need to track the new input and output maps
        for tasklet, state, ssa_statements, input_type in tasklets_to_split:
            assert isinstance(state, dace.SDFGState)
            assert isinstance(tasklet, dace.nodes.Tasklet)
            assert tasklet in state.nodes()
            tasklet_input_edges = state.in_edges(tasklet)
            tasklet_output_edges = state.out_edges(tasklet)

            tasklet_in_degree = state.in_degree(tasklet)
            tasklet_in_edges = state.in_edges(tasklet)
            tasklet_out_degree = state.out_degree(tasklet)
            state.remove_node(tasklet)
            added_tasklets = list()
            for i, ssa_statement in enumerate(ssa_statements):  # Since SSA we are going to add in a line
                lhs_vars, rhs_vars = _get_vars(ssa_statement)
                assert len(lhs_vars) == 1
                t = state.add_tasklet(
                    name=f"{tasklet.name}_split_{i}",
                    inputs=set(rhs_vars),
                    outputs=set(lhs_vars),
                    code=ssa_statement,
                )
                for rhs_var in rhs_vars:
                    t.add_in_connector(rhs_var)
                for lhs_var in lhs_vars:
                    t.add_out_connector(lhs_var)
                added_tasklets.append(t)

            added_accesses = dict()
            for i, t in enumerate(added_tasklets):
                assert isinstance(t, dace.nodes.Tasklet)
                # First do inputs
                if i == 0:  # First new tasklet
                    # The inputs should be available in the input data
                    # This might not be matched if a symbol is used in the tasklet
                    # for example X = dace.float64(symbol)
                    matched_in_conns = set()
                    for in_conn in t.in_connectors:
                        matching_in_edges = {ie for ie in tasklet_input_edges if ie.dst_conn == in_conn}
                        assert len(
                            matching_in_edges
                        ) <= 1, f"Required 1 matching in edge always, found: {matching_in_edges}, original tasklet code: {tasklet.code.as_string}, current tasklet code: {t.code.as_string}"

                        if len(matching_in_edges) > 0:
                            matching_in_edge = next(iter(matching_in_edges))

                            state.add_edge(matching_in_edge.src, matching_in_edge.src_conn, t, in_conn,
                                           copy.deepcopy(matching_in_edge.data))
                            matched_in_conns.add(in_conn)

                    for in_conn in list(t.in_connectors.keys()):
                        if in_conn not in matched_in_conns:
                            t.remove_in_connector(in_conn)

                    # dace.float64(symbol) has no sources after split,
                    # but if we for example inside a map we need to add a dependency edge
                    if len(matched_in_conns) == 0 and tasklet_in_degree > 0:
                        for ie in tasklet_in_edges:
                            state.add_edge(ie.src, None, t, None, dace.memlet.Memlet(None))
                else:
                    # Input comes from transient accesses (each unique and needs to be added to the SDFG)
                    # or from the unused in edges
                    for in_conn in t.in_connectors.keys():
                        matching_in_edges = {ie for ie in tasklet_input_edges if ie.dst_conn == in_conn}
                        if len(matching_in_edges) == 0:
                            array_name = f"{in_conn}{self.tmp_access_identifier}{split_access_counter}"
                            if array_name not in state.sdfg.arrays:
                                state.sdfg.add_scalar(
                                    name=array_name,
                                    dtype=input_type,
                                    storage=dace.dtypes.StorageType.Register,
                                    transient=True,
                                )
                                assert array_name not in added_accesses
                                added_accesses[array_name] = state.add_access(array_name)
                            state.add_edge(
                                added_accesses[array_name], None, t, in_conn,
                                dace.memlet.Memlet.from_array(dataname=array_name,
                                                              datadesc=state.sdfg.arrays[array_name]))
                        else:
                            assert len(matching_in_edges) == 1
                            matching_in_edge = next(iter(matching_in_edges))
                            state.add_edge(matching_in_edge.src, matching_in_edge.src_conn, t, in_conn,
                                           copy.deepcopy(matching_in_edge.data))

                # Then do the outputs
                if i == len(added_tasklets) - 1:  # last tasklet
                    # The outputs should be available in the output data
                    for out_conn in t.out_connectors:
                        matching_out_edges = {oe for oe in tasklet_output_edges if oe.src_conn == out_conn}
                        assert len(matching_out_edges) == 1
                        matching_out_edge = next(iter(matching_out_edges))
                        state.add_edge(t, out_conn, matching_out_edge.dst, matching_out_edge.dst_conn,
                                       copy.deepcopy(matching_out_edge.data))
                else:
                    # Output should have been added already
                    assert len(t.out_connectors) == 1
                    out_conn = next(iter(t.out_connectors))
                    array_name = f"{out_conn}{self.tmp_access_identifier}{split_access_counter}"
                    if array_name not in state.sdfg.arrays:
                        state.sdfg.add_scalar(
                            name=array_name,
                            dtype=input_type,
                            storage=dace.dtypes.StorageType.Register,
                            transient=True,
                        )
                        assert array_name not in added_accesses
                        added_accesses[array_name] = state.add_access(array_name)
                    state.sdfg.save("x.sdfgz", compress=True)
                    state.add_edge(
                        t, out_conn, added_accesses[array_name], None,
                        dace.memlet.Memlet.from_array(dataname=array_name, datadesc=state.sdfg.arrays[array_name]))

            split_access_counter += 1

        return None
