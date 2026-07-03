# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Vectorization-preparation passes that rewrite Python tasklet bodies:
power expansion, type-cast removal, math-prefix stripping, and STD-to-DaCe
math replacement."""
import dace
from typing import Any, Callable, Dict, Optional, Set
import ast
import re
import sympy
from dace import SDFG, properties, transformation
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation.helpers import CodeBlock


def _rewrite_python_tasklet_bodies(
        sdfg: SDFG,
        rewrite: Callable[[str], str],
        filter_node: Optional[Callable[[Any, "dace.SDFGState", "dace.sdfg.nodes.Tasklet"], bool]] = None) -> None:
    """Apply ``rewrite`` to every Python tasklet body in ``sdfg`` recursively.

    :param sdfg: the SDFG whose Python tasklets are rewritten in place.
    :param rewrite: maps a tasklet's source string to its replacement; the
        code is updated only if the result differs.
    :param filter_node: optional per-tasklet predicate; if it returns False
        the rewrite is skipped for that tasklet.
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
    """Expands ``**`` and ``pow``/``math.pow`` calls into multiplications or exp/log."""

    @staticmethod
    def _is_pow_call(call_node: ast.Call) -> bool:
        """Return True for the two-arg forms ``pow(x, y)`` and ``math.pow(x, y)``.

        :param call_node: the call node to test.
        :returns: True if it is a two-argument pow call.
        """
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
        """Expand ``left ** right``: integer exponent to unrolled product, else exp/log.

        :param left: the base expression.
        :param right: the exponent expression.
        :param loc: the original node used for source-location copying.
        :returns: the rewritten expression node.
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

        # Case 2: non-constant / non-integer exponent → leave it as ``left ** right``
        # for the tile binop to lower to ``std::pow``. The former ``exp(right *
        # log(left))`` identity is only valid for a POSITIVE base: ``log(left)`` is NaN
        # for a negative ``left``, so ``sin(x)**2`` -- whose exponent arrives as a
        # connector (numpy's ``power`` ufunc form), NOT a literal, so Case 1 does not
        # fire -- produced NaN on every lane where ``sin(x) < 0`` (npbench arc_distance).
        # ``std::pow`` computes a negative base with an integer exponent correctly and
        # matches numpy's ``**``; ``**`` carries an ISA-less pure lowering
        # (``_PURE_ONLY_MATH_OPS`` / ``_OP_CPP["**"]``) so it vectorizes via libmvec.
        return ast.copy_location(ast.BinOp(left=left, op=ast.Pow(), right=right), loc)

    def visit_BinOp(self, node):
        """Expand a ``**`` binary operation, recursing into children first.

        :param node: the binary-operation node.
        :returns: the expanded node, or the original if not a power op.
        """
        self.generic_visit(node)  # First, rewrite children
        if isinstance(node.op, ast.Pow):
            return self._expand_pow(node.left, node.right, node)
        return node

    def visit_Call(self, node):
        """Expand a ``pow``/``math.pow`` call, recursing into children first.

        :param node: the call node.
        :returns: the expanded node, or the original if not a pow call.
        """
        self.generic_visit(node)  # First, rewrite children
        if self._is_pow_call(node):
            return self._expand_pow(node.args[0], node.args[1], node)
        return node


class DaceCastRemover(ast.NodeTransformer):
    """Strips ``dace.floatNN(...)`` / ``dace.intNN(...)`` casts, keeping the cast value."""

    def __init__(self, call_name: str):
        self.call_name = call_name
        # Match exactly the cast names: ``call_name`` optionally followed by a
        # bit-width (``int``, ``int8``, ``int32``, ``float64``, ...). A bare
        # ``startswith`` would also catch unrelated builtins such as
        # ``int_floor`` / ``int_ceil`` and strip them to their first argument,
        # silently dropping the divisor (TSVC s276: ``int_floor(LEN_1D, 2)``
        # became ``LEN_1D``).
        self._is_cast_name = re.compile(rf"{re.escape(call_name)}\d*$").fullmatch

    def visit_Call(self, node):
        """Drop a matching dace cast call, returning its first argument.

        :param node: the call node.
        :returns: the cast value, ``0.0`` for an empty cast, or the original node.
        """
        self.generic_visit(node)  # first rewrite children
        # Check if this is a dace.float...() call
        if isinstance(node.func, ast.Attribute):
            # Handle dace.float64(), dace.float32(), etc.
            if (isinstance(node.func.value, ast.Name) and node.func.value.id == 'dace'
                    and self._is_cast_name(node.func.attr)):
                # Return the first argument (the value being cast)
                if node.args:
                    return node.args[0]
                else:
                    # If no arguments, just remove the call entirely
                    return ast.Constant(value=0.0)

        elif isinstance(node.func, ast.Name):
            # Handle direct calls like float64() if imported
            if self._is_cast_name(node.func.id):
                # Return the first argument (the value being cast)
                if node.args:
                    return node.args[0]
                else:
                    # If no arguments, just remove the call entirely
                    return ast.Constant(value=0.0)

        return node


class RemoveMathPrefix(ast.NodeTransformer):
    """Rewrites ``math.xxx(...)`` calls to ``xxx(...)``, stripping only the prefix."""

    def visit_Call(self, node):
        """Strip the ``math.`` prefix from a call, recursing into children first.

        :param node: the call node.
        :returns: the (possibly de-prefixed) call node.
        """
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
    """Expand power operators/calls in Python source.

    :param src: Python source string.
    :returns: source with ``**``/``pow`` expanded.
    """
    tree = ast.parse(src)
    tree = PowerOperatorExpander().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _remove_dace_float_casts(src: str):
    """Remove ``dace.floatNN(...)`` casts from Python source.

    :param src: Python source string.
    :returns: source with float casts removed.
    """
    tree = ast.parse(src)
    tree = DaceCastRemover(call_name="float").visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _remove_dace_int_casts(src: str):
    """Remove ``dace.intNN(...)`` casts from Python source.

    :param src: Python source string.
    :returns: source with int casts removed.
    """
    tree = ast.parse(src)
    tree = DaceCastRemover(call_name="int").visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _remove_math_prefix_from_source(source: str) -> str:
    """Rewrite ``math.xxx`` to ``xxx`` in Python source.

    :param source: Python source string.
    :returns: source with the ``math.`` prefix stripped from calls.
    """
    tree = ast.parse(source)
    tree = RemoveMathPrefix().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


class ModuloToPyModExpander(ast.NodeTransformer):
    """Rewrite every ``a % b`` (``ast.Mod`` binop) into a ``py_mod(a, b)`` call.

    Python/NumPy modulo follows the divisor's sign; C's ``%`` follows the
    dividend's. cppunparse lowers a bare ``%`` to C's ``%`` (and is ill-formed
    for floats), so a tasklet ``a % b`` silently miscompiles negative operands.
    ``py_mod`` resolves to ``dace::math::py_mod`` in generated code (the same
    helper the ``np.mod`` ufunc emits), giving Python semantics everywhere. An
    existing ``py_mod(...)`` call is a plain :class:`ast.Call` and is left
    untouched, so the rewrite is idempotent.
    """

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)  # rewrite nested ``%`` first
        if isinstance(node.op, ast.Mod):
            return ast.copy_location(
                ast.Call(func=ast.Name(id="py_mod", ctx=ast.Load()), args=[node.left, node.right], keywords=[]), node)
        return node

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
        self.generic_visit(node)  # rewrite ``%`` in the value expression first
        if isinstance(node.op, ast.Mod):
            # ``t %= b`` -> ``t = py_mod(t, b)`` (the target is also a read here).
            read_target = ast.copy_location(ast.Name(id=node.target.id, ctx=ast.Load()), node.target) \
                if isinstance(node.target, ast.Name) else node.target
            call = ast.Call(func=ast.Name(id="py_mod", ctx=ast.Load()), args=[read_target, node.value], keywords=[])
            return ast.copy_location(ast.Assign(targets=[node.target], value=call), node)
        return node


def _rewrite_modulo(src: str) -> str:
    """Rewrite ``%`` modulo to ``py_mod(...)`` in a Python source string.

    Covers tasklet bodies, control-flow codeblocks (loop bounds, branch
    conditions), and interstate-edge condition / assignment expressions -- any
    place the operator appears as Python ``%``.

    :param src: the Python source.
    :returns: the rewritten source (unchanged when it carries no ``%``).
    """
    if "%" not in src:  # fast path: no modulo to rewrite
        return src
    tree = ast.parse(src)
    tree = ModuloToPyModExpander().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


_PY_MOD = sympy.Function("py_mod")


def _subs_py_mod_symbolic(expr):
    """Rewrite every sympy ``Mod(a, b)`` in ``expr`` to ``py_mod(a, b)``.

    Used for the symbolic sites a tasklet rewrite cannot reach: memlet subsets
    and map ranges (and any other :class:`sympy.Basic`). ``symstr(cpp_mode)``
    lowers a bare ``Mod`` to C's ``%`` just like cppunparse, so an index such as
    ``A[(i - k) % n]`` would miscompile a negative offset; ``py_mod`` renders to
    ``dace::math::py_mod``.

    :param expr: a symbolic expression (or any value; non-symbolic is returned as-is).
    :returns: the rewritten expression, or ``expr`` unchanged when it has no ``Mod``.
    """
    if not isinstance(expr, sympy.Basic) or not expr.has(sympy.Mod):
        return expr
    return expr.replace(sympy.Mod, _PY_MOD)


def _subset_has_mod(subset) -> bool:
    """Whether any component expression of ``subset`` contains a sympy ``Mod``."""
    if isinstance(subset, dace.subsets.Range):
        exprs = [x for rng in subset.ranges for x in rng]
    elif isinstance(subset, dace.subsets.Indices):
        exprs = list(subset.indices)
    else:
        return False
    return any(isinstance(x, sympy.Basic) and x.has(sympy.Mod) for x in exprs)


def _rewrite_subset_modulo(subset):
    """Return a copy of ``subset`` with every ``Mod`` rewritten to ``py_mod``.

    :param subset: a :class:`~dace.subsets.Range` or :class:`~dace.subsets.Indices`.
    :returns: the rewritten subset (a new object), or ``subset`` for other types.
    """
    if isinstance(subset, dace.subsets.Range):
        return dace.subsets.Range([(_subs_py_mod_symbolic(b), _subs_py_mod_symbolic(e), _subs_py_mod_symbolic(s))
                                   for b, e, s in subset.ranges])
    if isinstance(subset, dace.subsets.Indices):
        return dace.subsets.Indices([_subs_py_mod_symbolic(i) for i in subset.indices])
    return subset


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
        """Apply the subclass rewrite to all Python tasklet bodies, then validate.

        :param sdfg: the SDFG rewritten in place.
        :param pipeline_results: unused pipeline results.
        :returns: None.
        """
        _rewrite_python_tasklet_bodies(sdfg, self._rewrite)
        sdfg.validate()
        return None


@properties.make_properties
@transformation.explicit_cf_compatible
class PowerOperatorExpansion(_BodyRewritePass):
    """Pass that expands power operators/calls in every Python tasklet body."""

    def _rewrite(self, src: str) -> str:
        return _expand_pow(src)


@properties.make_properties
@transformation.explicit_cf_compatible
class RewriteModuloToPyMod(_BodyRewritePass):
    """Rewrite every ``%`` modulo to ``py_mod`` everywhere it can appear in an SDFG.

    Run early ("cleaning") in the canonicalize and vectorize pipelines so the
    canonicalized reference, the vectorized body, and the base codegen all agree
    on Python/NumPy modulo semantics (``dace::math::py_mod``) without changing
    core ``cppunparse`` (whose bare ``%`` follows C's dividend-sign rule and
    miscompiles negative operands; ``symstr(cpp_mode)`` lowers a sympy ``Mod`` the
    same way). The operator can appear in five places, all covered here:

    * **tasklet bodies** (Python) -- ``c = a % b``;
    * **loop-range codeblocks** -- a ``LoopRegion`` condition / init / update,
      e.g. ``range(0, x % 7)`` lowered to ``i < x % 7``;
    * **branch conditions** -- a ``ConditionalBlock`` arm, e.g. ``if a % 2 == 0``;
    * **memlet subsets** and **map ranges** -- symbolic, e.g. ``A[(i + k) % n]``;
    * **interstate edges** -- the condition codeblock and every assignment RHS.

    Idempotent: an existing ``py_mod(...)`` call (or a ``Function('py_mod')``) is
    left as-is.
    """

    def modifies(self) -> ppl.Modifies:
        return (ppl.Modifies.Tasklets | ppl.Modifies.Memlets | ppl.Modifies.InterstateEdges | ppl.Modifies.Scopes)

    def _rewrite(self, src: str) -> str:
        # Retained so the pass still composes as a plain body rewrite if reused.
        return _rewrite_modulo(src)

    @staticmethod
    def _rewritten_codeblock(cb):
        """Return a CodeBlock with ``%`` -> ``py_mod``; the original if unchanged.

        :param cb: a :class:`CodeBlock` or ``None``.
        :returns: a new Python CodeBlock when a ``%`` was rewritten, else ``cb``.
        """
        if cb is None or cb.language != dace.dtypes.Language.Python:
            return cb
        src = cb.as_string
        if not src or "%" not in src:
            return cb
        new = _rewrite_modulo(src)
        return CodeBlock(new, language=dace.Language.Python) if new != src else cb

    def _rewrite_control_flow(self, g: SDFG) -> None:
        """Rewrite ``%`` in loop-bound codeblocks and branch conditions of ``g``."""
        for cfg in g.all_control_flow_regions(recursive=True):
            if isinstance(cfg, LoopRegion):
                for attr in ("loop_condition", "init_statement", "update_statement"):
                    cb = getattr(cfg, attr, None)
                    new = self._rewritten_codeblock(cb)
                    if new is not cb:
                        setattr(cfg, attr, new)
            elif isinstance(cfg, ConditionalBlock):
                for branch in cfg.branches:  # each branch is a ``[condition, body]`` pair
                    new = self._rewritten_codeblock(branch[0])
                    if new is not branch[0]:
                        branch[0] = new

    def _rewrite_interstate_edges(self, g: SDFG) -> None:
        """Rewrite ``%`` in interstate-edge conditions and assignment RHS of ``g``."""
        for e in g.all_interstate_edges(recursive=True):
            ise = e.data
            new_cond = self._rewritten_codeblock(ise.condition)
            if new_cond is not ise.condition:
                ise.condition = new_cond
            for var, rhs in list(ise.assignments.items()):
                if "%" in rhs:
                    new_rhs = _rewrite_modulo(rhs)
                    if new_rhs != rhs:
                        ise.assignments[var] = new_rhs

    def _rewrite_memlets_and_ranges(self, g: SDFG) -> None:
        """Rewrite symbolic ``Mod`` in memlet subsets and map ranges of ``g``."""
        for state in g.all_states():
            for e in state.edges():
                m = e.data
                if m is None:
                    continue
                if m.subset is not None and _subset_has_mod(m.subset):
                    m.subset = _rewrite_subset_modulo(m.subset)
                if m.other_subset is not None and _subset_has_mod(m.other_subset):
                    m.other_subset = _rewrite_subset_modulo(m.other_subset)
            for node in state.nodes():
                if isinstance(node, dace.nodes.MapEntry) and _subset_has_mod(node.map.range):
                    node.map.range = _rewrite_subset_modulo(node.map.range)

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Set[str]]]:
        """Rewrite ``%`` -> ``py_mod`` across every location of ``sdfg``, then validate.

        :param sdfg: the SDFG rewritten in place.
        :param pipeline_results: unused pipeline results.
        :returns: None.
        """
        # Tasklet bodies (recurses through nested SDFGs on its own).
        _rewrite_python_tasklet_bodies(sdfg, _rewrite_modulo)
        # The remaining sites are per-SDFG (``all_states`` / ``all_control_flow_regions``
        # / ``all_interstate_edges`` do NOT descend into nested SDFGs), so walk each.
        for g in sdfg.all_sdfgs_recursive():
            self._rewrite_control_flow(g)
            self._rewrite_interstate_edges(g)
            self._rewrite_memlets_and_ranges(g)
        sdfg.validate()
        return None


@properties.make_properties
@transformation.explicit_cf_compatible
class RemoveFPTypeCasts(_BodyRewritePass):
    """Pass that removes ``dace.floatNN(...)`` casts from every Python tasklet body."""

    def _rewrite(self, src: str) -> str:
        return _remove_dace_float_casts(src)


@properties.make_properties
@transformation.explicit_cf_compatible
class RemoveIntTypeCasts(_BodyRewritePass):
    """Pass that removes ``dace.intNN(...)`` casts from every Python tasklet body."""

    def _rewrite(self, src: str) -> str:
        return _remove_dace_int_casts(src)


@properties.make_properties
@transformation.explicit_cf_compatible
class RemoveMathCall(ppl.Pass):
    """Pass that strips the ``math.`` prefix from the RHS of assignment tasklets."""

    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Tasklets

    def should_reapply(self, modified: ppl.Modifies):
        return False

    def depends_on(self):
        return {}

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Set[str]]]:
        """Strip ``math.`` from the RHS of every Python assignment tasklet, then validate.

        :param sdfg: the SDFG rewritten in place.
        :param pipeline_results: unused pipeline results.
        :returns: None.
        """
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
