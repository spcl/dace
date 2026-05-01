# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from typing import Any, Callable, Dict, Optional, Set, Tuple
import ast
from dace import SDFG, properties, transformation
from dace.transformation import pass_pipeline as ppl
from dace.transformation.helpers import CodeBlock


def _rewrite_python_tasklet_bodies(sdfg: SDFG, rewrite: Callable[[str], str],
                                   filter_node: Optional[Callable[[Any, "dace.SDFGState", "dace.sdfg.nodes.Tasklet"],
                                                                  bool]] = None) -> None:
    """Apply ``rewrite`` to every Python tasklet body in ``sdfg`` (recursively).

    ``rewrite`` is called with the tasklet's source string; if it returns a different
    string the tasklet's code is updated. ``filter_node`` (optional) decides per-tasklet
    whether the rewrite should be attempted at all.
    """
    for node, graph in sdfg.all_nodes_recursive():
        if not isinstance(node, dace.sdfg.nodes.Tasklet):
            continue
        if node.code.language != dace.dtypes.Language.Python:
            continue
        if filter_node is not None and not filter_node(graph, node):
            continue
        ast_str = node.code.as_string
        new_ast_str = rewrite(ast_str)
        if new_ast_str != ast_str:
            node.code = CodeBlock(new_ast_str, language=dace.Language.Python)


class PowerOperatorExpander(ast.NodeTransformer):

    @staticmethod
    def _is_pow_call(call_node: ast.Call) -> bool:
        """True for the two-argument forms ``pow(x, y)`` and ``math.pow(x, y)``."""
        if len(call_node.args) != 2:
            return False
        func = call_node.func
        if isinstance(func, ast.Name) and func.id == "pow":
            return True
        if (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "math"
                and func.attr == "pow"):
            return True
        return False

    def _expand_pow(self, left: ast.AST, right: ast.AST, loc: ast.AST) -> ast.AST:
        """Expand ``left ** right`` according to the two cases handled below.

        Shared between the ``ast.BinOp(op=Pow)`` and ``pow(x, y)`` / ``math.pow(x, y)`` entry
        points so the rewrite logic stays single-sourced.
        """
        # Case 1: integer-like exponent → unrolled multiplication
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
                    new_node = ast.copy_location(left, left)
                    for _ in range(n - 1):
                        new_node = ast.BinOp(left=ast.copy_location(new_node, left),
                                             op=ast.Mult(),
                                             right=ast.copy_location(ast.fix_missing_locations(left), left))
                    return ast.copy_location(new_node, loc)
                # n in {0, 1} → leave the original `left ** right` shape as a BinOp; caller decides
                return ast.copy_location(ast.BinOp(left=left, op=ast.Pow(), right=right), loc)

        # Case 2: non-integer exponent → exp(right * log(left))
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
        return ast.copy_location(exp_call, loc)

    def visit_BinOp(self, node):
        self.generic_visit(node)  # First, rewrite children
        if isinstance(node.op, ast.Pow):
            return self._expand_pow(node.left, node.right, node)
        return node

    def visit_Call(self, node):
        self.generic_visit(node)  # First, rewrite children
        if self._is_pow_call(node):
            return self._expand_pow(node.args[0], node.args[1], node)
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
            if (isinstance(node.func.value, ast.Name) and node.func.value.id == 'math'
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


class _BodyRewritePass(ppl.Pass):
    """Base for vectorization preprocessing passes that rewrite Python tasklet bodies in place.

    Subclasses set ``_rewrite`` (string -> string). The helper walks every tasklet,
    applies the rewrite to its source, and reinstalls the body if changed.
    Validation runs once at the end.
    """
    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Tasklets

    def should_reapply(self, modified: ppl.Modifies):
        return False

    def depends_on(self):
        return {}

    def _rewrite(self, src: str) -> str:
        raise NotImplementedError

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Set[str]]]:
        _rewrite_python_tasklet_bodies(sdfg, self._rewrite)
        sdfg.validate()
        return None


@properties.make_properties
@transformation.explicit_cf_compatible
class PowerOperatorExpansion(_BodyRewritePass):

    def _rewrite(self, src: str) -> str:
        return _expand_pow(src)


@properties.make_properties
@transformation.explicit_cf_compatible
class RemoveFPTypeCasts(_BodyRewritePass):

    def _rewrite(self, src: str) -> str:
        return _remove_dace_float_casts(src)


@properties.make_properties
@transformation.explicit_cf_compatible
class RemoveIntTypeCasts(_BodyRewritePass):

    def _rewrite(self, src: str) -> str:
        return _remove_dace_int_casts(src)


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
        # RemoveMathCall is shaped differently: it splits the tasklet body on " = ", rewrites
        # only the RHS, and asserts the prefix is gone afterwards. The body-rewrite helper does
        # not match this shape, so the loop is open-coded here.
        for node, _ in sdfg.all_nodes_recursive():
            if not isinstance(node, dace.sdfg.nodes.Tasklet):
                continue
            if node.code.language != dace.dtypes.Language.Python:
                continue
            ast_str = node.code.as_string
            if len(ast_str.split(" = ")) != 2:
                continue
            ast_left, ast_right = ast_str.split(" = ")
            ast_left = ast_left.strip()
            ast_right = ast_right.strip()
            new_ast_right = _remove_math_prefix_from_source(ast_right)
            if new_ast_right != ast_right:
                node.code = CodeBlock(ast_left + " = " + new_ast_right, language=dace.Language.Python)
            assert "math." not in new_ast_right
            assert "math." not in node.code.as_string
        sdfg.validate()
        return None


def _matched_fp_dtype_suffix(graph: "dace.SDFGState",
                             node: "dace.sdfg.nodes.Tasklet") -> Optional[str]:
    """Return ``"f"`` / ``"d"`` if the tasklet is a single-input/single-output operation on
    matching float32 / float64 arrays, else None (meaning "skip this tasklet").
    """
    ies = graph.in_edges(node)
    oes = graph.out_edges(node)
    if len(ies) != 1 or len(oes) != 1:
        return None
    ie_data = ies[0].data.data
    oe_data = oes[0].data.data
    if ie_data is None or oe_data is None:
        return None
    ie_arr = graph.sdfg.arrays[ie_data]
    oe_arr = graph.sdfg.arrays[oe_data]
    if ie_arr.dtype == dace.float32 and oe_arr.dtype == dace.float32:
        return "f"
    if ie_arr.dtype == dace.float64 and oe_arr.dtype == dace.float64:
        return "d"
    return None


def _apply_replace_std_function(sdfg: SDFG, *, use_safe_implementation: bool, src_func: str,
                                dst_prefix: str, include_path: str) -> None:
    """Shared body for the STD->dace replacement trio (log/exp/pow).

    Walks all Python tasklets, rewrites ``src_func`` to ``dst_prefix{safe_infix}{f|d}``
    on tasklets whose single input/output arrays are matching float32 or float64,
    then appends ``#include "include_path"`` to the SDFG's global code.
    """
    safe_infix = "" if use_safe_implementation is False else "safe_"
    for node, graph in sdfg.all_nodes_recursive():
        if not isinstance(node, dace.sdfg.nodes.Tasklet):
            continue
        if node.code.language != dace.dtypes.Language.Python:
            continue
        suffix = _matched_fp_dtype_suffix(graph, node)
        if suffix is None:
            continue
        ast_str = node.code.as_string
        new_ast_str = _replace_function_names(ast_str, src_func, f"{dst_prefix}{safe_infix}{suffix}")
        if new_ast_str != ast_str:
            node.code = CodeBlock(new_ast_str, language=dace.Language.Python)

    sdfg.append_global_code(f'#include "{include_path}"\n')
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
        _apply_replace_std_function(sdfg,
                                    use_safe_implementation=self.use_safe_implementation,
                                    src_func="log",
                                    dst_prefix="dace_log_",
                                    include_path="dace/arith/log.h")
        return None


@properties.make_properties
@transformation.explicit_cf_compatible
class ReplaceSTDExpWithDaCeExp(ppl.Pass):
    CATEGORY: str = 'Optimization Preparation'
    use_safe_implementation = dace.properties.Property(dtype=bool, default=False, allow_none=False)

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Tasklets

    def should_reapply(self, modified: ppl.Modifies):
        return False

    def depends_on(self):
        return {}

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Set[str]]]:
        _apply_replace_std_function(sdfg,
                                    use_safe_implementation=self.use_safe_implementation,
                                    src_func="exp",
                                    dst_prefix="dace_exp_",
                                    include_path="dace/arith/exp.h")
        return None


@properties.make_properties
@transformation.explicit_cf_compatible
class ReplaceSTDPowWithDaCePow(ppl.Pass):
    CATEGORY: str = 'Optimization Preparation'
    use_safe_implementation = dace.properties.Property(dtype=bool, default=False, allow_none=False)

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Tasklets

    def should_reapply(self, modified: ppl.Modifies):
        return False

    def depends_on(self):
        return {}

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Set[str]]]:
        _apply_replace_std_function(sdfg,
                                    use_safe_implementation=self.use_safe_implementation,
                                    src_func="pow",
                                    dst_prefix="dace_pow_",
                                    include_path="dace/arith/pow.h")
        return None
