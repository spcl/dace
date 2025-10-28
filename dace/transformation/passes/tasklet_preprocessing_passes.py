# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import dace

from typing import Any, Dict, Optional, Set
import ast
import warnings

from dace import SDFG, SDFGState, properties, transformation
from dace.transformation import pass_pipeline as ppl, dataflow as dftrans
from dace.transformation.helpers import CodeBlock
from dace.transformation.passes import analysis as ap, pattern_matching as pmp
from dace.transformation.passes.split_tasklets import SplitTasklets

import ast


class PowerOperatorExpander(ast.NodeTransformer):

    def visit_BinOp(self, node):
        self.generic_visit(node)  # First, rewrite children

        # Match "a ** b"
        if isinstance(node.op, ast.Pow):
            right = node.right
            left = node.left

            # Case 1: integer-like exponent
            if isinstance(right, ast.Constant):
                val = right.value
                if isinstance(val, int):
                    n = val
                elif isinstance(val, float) and val.is_integer():
                    n = int(val)
                else:
                    n = None

                if n is not None:
                    if n > 1:
                        # Expand: x * x * ... * x
                        new_node = ast.copy_location(left, left)
                        for _ in range(n - 1):
                            new_node = ast.BinOp(left=ast.copy_location(new_node, left),
                                                 op=ast.Mult(),
                                                 right=ast.copy_location(ast.fix_missing_locations(left), left))
                        return ast.copy_location(new_node, node)
                    else:
                        # Leave x ** 0 or x ** 1 unchanged
                        return node

            # Case 2: non-integer exponent → use exp(y * log(x))
            log_call = ast.Call(func=ast.Attribute(value=ast.Name(id="math", ctx=ast.Load()),
                                                   attr="log",
                                                   ctx=ast.Load()),
                                args=[ast.copy_location(left, left)],
                                keywords=[])
            mul_expr = ast.BinOp(left=ast.copy_location(right, right), op=ast.Mult(), right=log_call)
            exp_call = ast.Call(func=ast.Attribute(value=ast.Name(id="math", ctx=ast.Load()),
                                                   attr="exp",
                                                   ctx=ast.Load()),
                                args=[mul_expr],
                                keywords=[])
            return ast.copy_location(exp_call, node)

        return node


class DaceCastRemover(ast.NodeTransformer):

    def __init__(self, call_name: str):
        self.call_name = call_name

    def visit_Call(self, node):
        self.generic_visit(node)  # first rewrite children
        # Check if this is a dace.float...() call
        if isinstance(node.func, ast.Attribute):
            # Handle dace.float64(), dace.float32(), etc.
            if (isinstance(node.func.value, ast.Name) and node.func.value.id == 'dace'
                    and node.func.attr.startswith(self.call_name)):
                # Return the first argument (the value being cast)
                if node.args:
                    return node.args[0]
                else:
                    # If no arguments, just remove the call entirely
                    return ast.Constant(value=0.0)

        elif isinstance(node.func, ast.Name):
            # Handle direct calls like float64() if imported
            if node.func.id.startswith(self.call_name) and len(node.func.id) >= len(self.call_name):
                # Return the first argument (the value being cast)
                if node.args:
                    return node.args[0]
                else:
                    # If no arguments, just remove the call entirely
                    return ast.Constant(value=0.0)

        return node


def _expand_pow(src):
    tree = ast.parse(src)
    tree = PowerOperatorExpander().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _remove_dace_float_casts(src):
    tree = ast.parse(src)
    tree = DaceCastRemover(call_name="float").visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _remove_dace_int_casts(src):
    tree = ast.parse(src)
    tree = DaceCastRemover(call_name="int").visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


@properties.make_properties
@transformation.explicit_cf_compatible
class PowerOperatorExpansion(ppl.Pass):
    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Tasklets

    def should_reapply(self, modified: ppl.Modifies):
        return ppl.Modifies.Tasklets

    def depends_on(self):
        return {}

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Set[str]]]:
        for node, _ in sdfg.all_nodes_recursive():
            if isinstance(node, dace.sdfg.nodes.Tasklet):
                if node.code.language == dace.dtypes.Language.Python:
                    ast_str = node.code.as_string
                    new_ast_str = _expand_pow(ast_str)
                    if new_ast_str != ast_str:
                        node.code = CodeBlock(new_ast_str, language=dace.Language.Python)

        return None


@properties.make_properties
@transformation.explicit_cf_compatible
class RemoveFPTypeCasts(ppl.Pass):
    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Tasklets

    def should_reapply(self, modified: ppl.Modifies):
        return ppl.Modifies.Tasklets

    def depends_on(self):
        return {}

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Set[str]]]:
        for node, _ in sdfg.all_nodes_recursive():
            if isinstance(node, dace.sdfg.nodes.Tasklet):
                if node.code.language == dace.dtypes.Language.Python:
                    ast_str = node.code.as_string
                    new_ast_str = _remove_dace_float_casts(ast_str)
                    if new_ast_str != ast_str:
                        node.code = CodeBlock(new_ast_str, language=dace.Language.Python)


@properties.make_properties
@transformation.explicit_cf_compatible
class RemoveIntTypeCasts(ppl.Pass):
    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Tasklets

    def should_reapply(self, modified: ppl.Modifies):
        return ppl.Modifies.Tasklets

    def depends_on(self):
        return {}

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Set[str]]]:
        for node, _ in sdfg.all_nodes_recursive():
            if isinstance(node, dace.sdfg.nodes.Tasklet):
                if node.code.language == dace.dtypes.Language.Python:
                    ast_str = node.code.as_string
                    new_ast_str = _remove_dace_int_casts(ast_str)
                    if new_ast_str != ast_str:
                        node.code = CodeBlock(new_ast_str, language=dace.Language.Python)
