# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import re

from sympy import Function
import dace
from typing import Dict, List, Optional, Set, Tuple

from dace import SDFG
from dace import dtypes
from dace.codegen.targets.framecode import infer_expr_type
from dace.transformation import pass_pipeline as ppl, transformation

import ast
from dace.sdfg.nodes import CodeBlock
from dace.codegen.tools.type_inference import infer_types


class ASTSplitter:

    def __init__(self):
        self.n = 0
        self.stmts = []

    def temp(self) -> str:
        t = f"__t{self.n}"
        self.n += 1
        return t

    def visit(self, node):
        if isinstance(node, ast.BinOp):
            l, r = self.visit(node.left), self.visit(node.right)
            t = self.temp()
            ops = {
                ast.Add: '+',
                ast.Sub: '-',
                ast.Mult: '*',
                ast.Div: '/',
                ast.Pow: '**',
                ast.Mod: '%',
                ast.MatMult: '@',
                ast.BitAnd: '&',
                ast.BitOr: '|',
                ast.BitXor: '^',
                ast.LShift: '<<',
                ast.RShift: '>>',
                ast.Or: 'or',
                ast.And: 'and',
                ast.Eq: '==',
            }
            self.stmts.append(f"{t} = {l} {ops[type(node.op)]} {r}")
            return t

        elif isinstance(node, ast.UnaryOp):
            op = self.visit(node.operand)
            t = self.temp()
            ops = {ast.USub: '-', ast.UAdd: '+', ast.Not: 'not ', ast.Invert: '~'}
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

        elif isinstance(node, ast.Compare):
            # Handle comparison operators
            left = self.visit(node.left)
            comparisons = []
            current = left

            for op, comparator in zip(node.ops, node.comparators):
                comp = self.visit(comparator)
                t = self.temp()
                ops = {
                    ast.Eq: '==',
                    ast.NotEq: '!=',
                    ast.Lt: '<',
                    ast.LtE: '<=',
                    ast.Gt: '>',
                    ast.GtE: '>=',
                }
                self.stmts.append(f"{t} = {current} {ops[type(op)]} {comp}")
                comparisons.append(t)
                current = comp

            # If multiple comparisons, combine with 'and'
            if len(comparisons) > 1:
                t = self.temp()
                self.stmts.append(f"{t} = {' and '.join(comparisons)}")
                return t
            return comparisons[0] if comparisons else left

        elif isinstance(node, ast.BoolOp):
            # Handle boolean operators (and, or)
            values = [self.visit(v) for v in node.values]
            t = self.temp()
            op = ' and ' if isinstance(node.op, ast.And) else ' or '
            if op == "or":
                assert isinstance(node.op, ast.Or)
            self.stmts.append(f"{t} = {op.join(values)}")
            return t

        return ast.unparse(node)


def to_ssa(code: str) -> List[str]:
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


def _get_vars(ssa_line: str) -> Tuple[List[str], List[str]]:
    lhs, rhs = ssa_line.split(" = ")
    lhs = lhs.strip()
    rhs = rhs.strip()
    # Also ignore log functions
    function_names = dace.symbolic._builtin_userfunctions.union({
        "log",
        "Log",
        "ln",
        "exp",
        "Exp",
        "or",
        "and",
        "Or",
        "And",
        "OR",
        "AND",
        "math",
        "Math",
        "MATH",
    }).union({"True", "False"})

    return [lhs], list(dace.symbolic.symbols_in_code(rhs, symbols_to_ignore=function_names))


@transformation.explicit_cf_compatible
class SplitTasklets(ppl.Pass):
    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Tasklets | ppl.Modifies.Descriptors | ppl.Modifies.AccessNodes | ppl.Modifies.Edges

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return {}

    tmp_access_identifier = "_split_"

    def token_split_variable_names(self, string_to_check: str) -> Set[str]:
        # Split while keeping delimiters
        tokens = re.split(r'(\s+|[()\[\]])', string_to_check)

        # Replace tokens that exactly match src
        tokens = {token.strip() for token in tokens if token not in ["[", "]", "(", ")"] and token.isidentifier()}

        return tokens

    def _add_missing_symbols(self, sdfg: SDFG):
        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    self._add_missing_symbols(node.sdfg)
        for e in sdfg.all_interstate_edges():
            for k, v in e.data.assignments.items():
                if k not in sdfg.symbols:
                    symexpr = dace.symbolic.SymExpr(v)
                    dtypes = set()
                    funcs = [a.name for a in symexpr.atoms(Function)]
                    for token in funcs:
                        if str(token) in sdfg.arrays:
                            dtypes.add(sdfg.arrays[str(token)].dtype)
                        if str(token) in sdfg.symbols:
                            dtypes.add(sdfg.symbols[str(token)])
                    ktype = None
                    dtype_priority = [
                        dace.float64,
                        dace.float32,
                        dace.int64,
                        dace.uint64,
                        dace.int32,
                        dace.uint32,
                        dace.int16,
                        dace.uint16,
                        dace.int8,
                        dace.uint8,
                        dace.bool_,
                    ]
                    ktype = None
                    if len(dtypes) > 0:
                        for dt in dtype_priority:
                            if dt in dtypes:
                                ktype = dt
                                break
                    if ktype is None:
                        for token in symexpr.free_symbols:
                            if str(token) in sdfg.arrays:
                                dtypes.add(sdfg.arrays[str(token)].dtype)
                            if str(token) in sdfg.symbols:
                                dtypes.add(sdfg.symbols[str(token)])
                    if len(dtypes) > 0:
                        for dt in dtype_priority:
                            if dt in dtypes:
                                ktype = dt
                                break
                    if ktype is None:
                        ktype = dace.float64
                    sdfg.add_symbol(k, ktype)
                    print(f"Adding missing symbol, THIS SHOULD NOT HAPPEN AND IT IS NOT MY FAULT FIX THIS")
                    print(f"Adding missing symbol {k}, deduced type {ktype}")

    def apply_pass(self, sdfg: SDFG, pipeline_results) -> Optional[Dict[str, Set[str]]]:
        self._add_missing_symbols(sdfg)
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

                # Collect symbolic types
                try:
                    tokens = c.as_string.split(" = ")
                    assert len(tokens) == 2
                    lhs = tokens[0].strip()
                    rhs = tokens[1].strip()
                    code_expr_rhs = dace.symbolic.SymExpr(rhs)
                    code_expr_lhs = dace.symbolic.SymExpr(lhs)
                    for free_sym in code_expr_rhs.free_symbols.union(code_expr_lhs.free_symbols):
                        if str(free_sym) in g.sdfg.symbols:
                            input_types.add(g.sdfg.symbols[str(free_sym)])
                except Exception as e:
                    # Nested comparisons might make symexpr / sympify crash
                    code_tokens = self.token_split_variable_names(c.as_string)
                    for free_sym in code_tokens:
                        if str(free_sym) in g.sdfg.symbols:
                            input_types.add(g.sdfg.symbols[str(free_sym)])

                # It is complicated to split a tasklet with mixed precision input
                # Need to bookkeep the mapping of intermediate results to precision
                if len(input_types) > 1:
                    # It might be zero due to symbols
                    has_float_type = any({
                        itype
                        for itype in input_types
                        if itype in {dace.dtypes.float64, dace.dtypes.float32, dace.dtypes.float16}
                    })
                    if has_float_type:
                        input_type = dace.float64
                    else:
                        input_type = dace.int64
                elif len(input_types) == 1:
                    input_type = next(iter(input_types))
                else:
                    # Default to float it consists purely of constants
                    input_type = dace.float64

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
            available_symbols = {str(s)
                                 for s in state.symbols_defined_at(tasklet)
                                 }.union({str(s)
                                          for s in sdfg.symbols.keys()})
            syms_defined_at = {str(s) for s in state.symbols_defined_at(tasklet)}
            state.remove_node(tasklet)
            added_tasklets = list()
            for i, ssa_statement in enumerate(ssa_statements):  # Since SSA we are going to add in a line
                lhs_vars, rhs_vars = _get_vars(ssa_statement)
                assert "True" not in rhs_vars
                assert "False" not in rhs_vars

                #print("Available symbols", available_symbols)
                #print("SDFG syms", sdfg.symbols)
                #print("Symbols defined at", {str(s) for s in syms_defined_at})
                symbol_rhs_vars = {rhs_var for rhs_var in rhs_vars if rhs_var in available_symbols}
                #print("Symbol rhs vars", symbol_rhs_vars)
                rhs_vars = set(rhs_vars) - symbol_rhs_vars
                #print("Rhs vars", rhs_vars)
                assert len(lhs_vars) == 1
                t = state.add_tasklet(
                    name=f"{tasklet.name}_split_{i}",
                    inputs=set(rhs_vars),
                    outputs=set(lhs_vars),
                    code=ssa_statement,
                )
                for rhs_var in rhs_vars:
                    assert rhs_var != "ptare"
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
                            if in_conn == "ptare":
                                raise Exception(t)
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
                    state.add_edge(
                        t, out_conn, added_accesses[array_name], None,
                        dace.memlet.Memlet.from_array(dataname=array_name, datadesc=state.sdfg.arrays[array_name]))

            split_access_counter += 1

        sdfg.validate()
        return None
