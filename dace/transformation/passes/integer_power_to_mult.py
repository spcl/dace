# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import dace

from typing import Any, Dict, Optional, Set
import ast
import warnings

from dace import SDFG, SDFGState, properties, transformation
from dace.transformation import pass_pipeline as ppl, dataflow as dftrans
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

def expand_pow_to_mul(src):
    tree = ast.parse(src)
    tree = PowToMulExpander().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)

@properties.make_properties
@transformation.explicit_cf_compatible
class IntegerPowerToMult(ppl.Pass):

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies):
        return ppl.Modifies.States | ppl.Modifies.Tasklets | ppl.Modifies.NestedSDFGs | ppl.Modifies.Scopes | ppl.Modifies.Descriptors

    def depends_on(self):
        return {}

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Set[str]]]:
        for node, _ in sdfg.all_nodes_recursive():
            if isinstance(node, dace.sdfg.nodes.Tasklet):
                if isinstance(node.code.language, dace.Language.Python):
                    ast_str = node.code.as_string
                    new_ast_str = expand_pow_to_mul(ast_str)
                    if new_ast_str != ast_str:
                        node.code = dace.codegen.code.CodeBlock(new_ast_str, language=dace.Language.Python)

        return None
