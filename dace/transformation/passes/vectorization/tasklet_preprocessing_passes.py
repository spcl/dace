# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from typing import Any, Dict, Optional, Set
import ast
from dace import SDFG, properties, transformation
from dace.transformation import pass_pipeline as ppl, dataflow as dftrans
from dace.transformation.helpers import CodeBlock
from dace.transformation.passes import analysis as ap, pattern_matching as pmp


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

class FunctionRenamer(ast.NodeTransformer):
    def __init__(self, src_function_name: str, dst_function_name: str):
        self.src_function_name = src_function_name
        self.dst_function_name = dst_function_name

    def visit_Call(self, node):
        # First rewrite children
        self.generic_visit(node)

        # Case 1: Attribute call like math.src_function_name(...)
        if isinstance(node.func, ast.Attribute):
            if (isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'math'
                and node.func.attr == self.src_function_name):
                node.func.attr = self.dst_function_name

        # Case 2: Direct call like src_function_name(...)
        elif isinstance(node.func, ast.Name):
            if node.func.id == self.src_function_name:
                node.func.id = self.dst_function_name

        return node 


class RemoveMathPrefix(ast.NodeTransformer):
    """
    Transform calls of the form math.xxx(...) → xxx(...).
    Only removes the module prefix; does not touch anything else.
    """

    def visit_Call(self, node):
        # Transform children first
        self.generic_visit(node)

        # Check if the function being called is an attribute: A.B
        if isinstance(node.func, ast.Attribute):
            # Check if it is "math.xxx"
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "math":
                # Replace math.xxx(...) → xxx(...)
                node.func = ast.Name(id=node.func.attr, ctx=ast.Load())

        return node

def _expand_pow(src: str):
    tree = ast.parse(src)
    tree = PowerOperatorExpander().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _remove_dace_float_casts(src: str):
    tree = ast.parse(src)
    tree = DaceCastRemover(call_name="float").visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _remove_dace_int_casts(src: str):
    tree = ast.parse(src)
    tree = DaceCastRemover(call_name="int").visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _remove_math_prefix_from_source(source: str) -> str:
    """Returns transformed Python source with math.xxx → xxx."""
    tree = ast.parse(source)
    tree = RemoveMathPrefix().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _replace_function_names(src: str, src_function: str, dst_function: str):
    tree = ast.parse(src)
    tree = FunctionRenamer(src_function_name=src_function, dst_function_name=dst_function).visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)

@properties.make_properties
@transformation.explicit_cf_compatible
class PowerOperatorExpansion(ppl.Pass):
    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Tasklets

    def should_reapply(self, modified: ppl.Modifies):
        return False

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
        sdfg.validate()
        return None


@properties.make_properties
@transformation.explicit_cf_compatible
class RemoveFPTypeCasts(ppl.Pass):
    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Tasklets

    def should_reapply(self, modified: ppl.Modifies):
        return False

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
        sdfg.validate()


@properties.make_properties
@transformation.explicit_cf_compatible
class RemoveIntTypeCasts(ppl.Pass):
    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Tasklets

    def should_reapply(self, modified: ppl.Modifies):
        return False

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
        sdfg.validate()


@properties.make_properties
@transformation.explicit_cf_compatible
class RemoveMathCall(ppl.Pass):
    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Tasklets

    def should_reapply(self, modified: ppl.Modifies):
        return False

    def depends_on(self):
        return {}

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Set[str]]]:
        for node, _ in sdfg.all_nodes_recursive():
            if isinstance(node, dace.sdfg.nodes.Tasklet):
                if node.code.language == dace.dtypes.Language.Python:
                    ast_str = node.code.as_string
                    ast_left, ast_right = ast_str.split(" = ")
                    ast_left = ast_left.strip()
                    ast_right = ast_right.strip()
                    new_ast_right = _remove_math_prefix_from_source(ast_right)
                    if new_ast_right != ast_right:
                        node.code = CodeBlock(ast_left + " = " + new_ast_right, language=dace.Language.Python)
                    assert "math." not in new_ast_right
                    assert "math." not in node.code.as_string
        sdfg.validate()

@properties.make_properties
@transformation.explicit_cf_compatible
class ReplaceSTDLogWithDaCeLog(ppl.Pass):
    CATEGORY: str = 'Optimization Preparation'
    use_safe_implementation = dace.properties.Property(dtype=bool, default=False, allow_none=False)

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Tasklets

    def should_reapply(self, modified: ppl.Modifies):
        return False

    def depends_on(self):
        return {}

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Set[str]]]:
        for node, graph in sdfg.all_nodes_recursive():
            if isinstance(node, dace.sdfg.nodes.Tasklet):
                if node.code.language == dace.dtypes.Language.Python:
                    # We support float->float or double->double
                    ies = graph.in_edges(node)
                    oes = graph.out_edges(node)
                    # Log tasklet should be single input single output
                    if len(ies) == 1 and len(oes) == 1:
                        ie = ies[0]
                        oe = oes[0]
                        ie_data = ie.data.data
                        oe_data = oe.data.data
                        # Check input data exists
                        if ie_data is not None and oe_data is not None:
                            ie_arr = graph.sdfg.arrays[ie_data]
                            oe_arr = graph.sdfg.arrays[oe_data]
                            # Check dtypes
                            if ((ie_arr.dtype == dace.float32 and oe_arr.dtype == dace.float32) or 
                                (ie_arr.dtype == dace.float64 and oe_arr.dtype == dace.float64)):
                                ast_str = node.code.as_string
                                suffix = "f" if (ie_arr.dtype == dace.float32 and oe_arr.dtype == dace.float32) else "d"
                                safe_infix = "" if self.use_safe_implementation is False else "safe_"
                                new_ast_str = _replace_function_names(ast_str, "log", f"dace_log_{safe_infix}{suffix}")
                                if new_ast_str != ast_str:
                                    node.code = CodeBlock(new_ast_str, language=dace.Language.Python)
        
        sdfg.append_global_code('#include "dace/arith.h"')
        sdfg.validate()