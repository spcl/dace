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


class PowToMulExpander(ast.NodeTransformer):

    def visit_BinOp(self, node):
        self.generic_visit(node)  # first rewrite children

        # Match "var ** integer"
        if isinstance(node.op, ast.Pow) and isinstance(node.right, ast.Constant) and isinstance(node.right.value, int):
            n = node.right.value
            if n > 1:
                # Create x * x * ... * x (n times)
                new_node = node.left
                for _ in range(n - 1):
                    new_node = ast.BinOp(left=ast.copy_location(ast.fix_missing_locations(new_node), node.left),
                                         op=ast.Mult(),
                                         right=ast.copy_location(ast.fix_missing_locations(node.left), node.left))
                return ast.copy_location(new_node, node)

        return node


class DaceFloatRemover(ast.NodeTransformer):

    def visit_Call(self, node):
        self.generic_visit(node)  # first rewrite children

        # Check if this is a dace.float...() call
        if isinstance(node.func, ast.Attribute):
            # Handle dace.float64(), dace.float32(), etc.
            if (isinstance(node.func.value, ast.Name) and node.func.value.id == 'dace'
                    and node.func.attr.startswith('float')):
                # Return the first argument (the value being cast)
                if node.args:
                    return node.args[0]
                else:
                    # If no arguments, just remove the call entirely
                    return ast.Constant(value=0.0)

        elif isinstance(node.func, ast.Name):
            # Handle direct calls like float64() if imported
            if node.func.id.startswith('float') and len(node.func.id) > 5:
                # Return the first argument (the value being cast)
                if node.args:
                    return node.args[0]
                else:
                    # If no arguments, just remove the call entirely
                    return ast.Constant(value=0.0)

        return node


def _expand_pow_to_mul(src):
    tree = ast.parse(src)
    tree = PowToMulExpander().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _remove_dace_float_casts(src):
    tree = ast.parse(src)
    tree = DaceFloatRemover().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


@properties.make_properties
@transformation.explicit_cf_compatible
class IntegerPowerToMult(ppl.Pass):
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
                    new_ast_str = _expand_pow_to_mul(ast_str)
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
