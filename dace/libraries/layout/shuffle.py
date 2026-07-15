# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The shuffle registry: user-defined value-permutations sigma with C lowerings.

A ``Shuffle(dim, name)`` layout op renumbers the elements along a dimension by a bijection
sigma: the laid-out array holds ``A'[i] = A[sigma(i)]``, so a consumer that logically wants
``A[e]`` reads ``A'[sigma_inverse(e)]``. The bijection is supplied BY THE USER as a pair of
closed-form expressions (forward ``sigma`` and inverse ``sigma^{-1}``) over the reserved index
variable ``i`` plus any SDFG symbols; we do NOT derive the inverse.

Each registered shuffle mints two artifacts from those expressions:

  * a :class:`sympy.Function` subclass per direction (named ``shuffle_<name>`` /
    ``shuffle_inv_<name>``) that can be substituted into a memlet subset -- an unknown sympy
    function prints as a literal C call, so ``A'[sigma_inverse(e)]`` lowers to
    ``A[shuffle_inv_<name>(e, ...)]``; its ``eval`` folds a constant integer index.
  * a C++ definition of each direction (``pyexpr2cpp`` on the expression) injected into an
    SDFG's global code via :func:`emit_shuffle_globals` -- so the emitted call resolves. Any
    SDFG symbols the expression uses become extra function parameters (and extra call args), so
    a symbol-parametrized sigma (e.g. ``(a*i + c) % N``) lowers correctly.

The index variable is ``i``; every other identifier in an expression (that is not a known math
helper) is treated as an SDFG symbol parameter, shared between the forward and inverse.
"""
import ast
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import sympy

import dace
from dace.codegen import cppunparse

#: The reserved index variable in a shuffle expression.
INDEX_VAR = "i"

#: Identifiers that are math helpers / literals, not SDFG symbol parameters.
_KNOWN_NAMES = frozenset({INDEX_VAR, "abs", "min", "max", "int"})

#: A floored modulo (Python ``%`` semantics: result takes the sign of the divisor). C's ``%``
#: truncates toward zero, so ``(i - 3) % 8`` disagrees with Python for a negative dividend; the
#: index-folding and numpy oracle use Python ``%``, so the emitted C must match to stay transparent.
PYMOD_FUNC = "shuffle_pymod"
PYMOD_DEF = (f"\nstatic inline long long {PYMOD_FUNC}(long long a, long long b) "
             f"{{ long long r = a % b; return (r != 0 && ((r < 0) != (b < 0))) ? r + b : r; }}\n")

_REGISTRY: "Dict[str, ShuffleFunction]" = {}


class _FlooredMod(ast.NodeTransformer):
    """Rewrite ``a % b`` to ``shuffle_pymod(a, b)`` so the emitted C floors like Python."""

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Mod):
            return ast.Call(func=ast.Name(id=PYMOD_FUNC, ctx=ast.Load()), args=[node.left, node.right], keywords=[])
        return node


def _expr_to_c(expr: str) -> str:
    """Convert a python index expression to C++, with ``%`` emitted as floored ``shuffle_pymod``.

    ``//`` already lowers to ``dace::math::ifloor`` (floored, matching Python) via ``pyexpr2cpp``.
    """
    tree = _FlooredMod().visit(ast.parse(expr, mode="eval"))
    ast.fix_missing_locations(tree)
    return cppunparse.pyexpr2cpp(ast.unparse(tree))


def _as_sympy(index):
    """Coerce a str / int / sympy value into a sympy expression (for a Function argument)."""
    if isinstance(index, sympy.Basic):
        return index
    return dace.symbolic.pystr_to_symbolic(str(index))


def _symbol_params(*exprs: str) -> Tuple[str, ...]:
    """The sorted SDFG-symbol identifiers used across ``exprs``.

    A free name is a symbol parameter unless it is the reserved index, a known math helper, or a
    function CALLEE (e.g. ``pow`` in ``pow(i, 2)`` -- a called name is a C function, not a symbol).
    """
    names, callees = set(), set()
    for expr in exprs:
        tree = ast.parse(expr, mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                callees.add(node.func.id)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id not in _KNOWN_NAMES:
                names.add(node.id)
    return tuple(sorted(names - callees))


def _make_function_class(cname: str, expr: str, params: Tuple[str, ...]) -> type:
    """Mint a ``sympy.Function`` subclass named ``cname`` that prints as ``cname(i, *params)``.

    The class carries the arity ``1 + len(params)``. Its ``eval`` folds a fully-constant call
    (concrete integer index and no symbol parameters) to the integer result, so constant
    accesses simplify; any symbolic call stays unevaluated and prints as a literal C call.
    """
    fold = None
    if not params:
        # Fold via a restricted eval on integer indices (expression uses only ``i``).
        code = compile(expr, f"<shuffle:{cname}>", "eval")

        def fold(index_value: int) -> int:
            return int(
                eval(code, {"__builtins__": {
                    "abs": abs,
                    "min": min,
                    "max": max,
                    "int": int
                }}, {INDEX_VAR: int(index_value)}))

    def eval_classmethod(cls, *args):
        if fold is not None and len(args) == 1 and args[0].is_Integer:
            return sympy.Integer(fold(int(args[0])))
        return None

    return type(cname, (sympy.Function, ), {
        "nargs": 1 + len(params),
        "eval": classmethod(eval_classmethod),
    })


@dataclass(frozen=True)
class ShuffleFunction:
    """A registered value-permutation: forward/inverse expressions, sympy classes, C defs."""
    name: str
    forward_expr: str
    inverse_expr: str
    params: Tuple[str, ...]
    forward_cls: type
    inverse_cls: type

    def forward_name(self) -> str:
        return f"shuffle_{self.name}"

    def inverse_name(self) -> str:
        return f"shuffle_inv_{self.name}"

    def _param_symbols(self) -> List[sympy.Symbol]:
        return [dace.symbolic.symbol(p) for p in self.params]

    def apply_forward(self, index):
        """The sympy term ``shuffle_<name>(index, *params)`` (folds a constant index)."""
        return self.forward_cls(_as_sympy(index), *self._param_symbols())

    def apply_inverse(self, index):
        """The sympy term ``shuffle_inv_<name>(index, *params)`` (folds a constant index)."""
        return self.inverse_cls(_as_sympy(index), *self._param_symbols())

    def _c_def(self, func_name: str, expr: str) -> str:
        sig_params = "".join(f", long long {p}" for p in self.params)
        return (f"\nstatic inline long long {func_name}(long long {INDEX_VAR}{sig_params}) "
                f"{{ return {_expr_to_c(expr)}; }}\n")

    def forward_c_def(self) -> str:
        """The ``static inline`` C++ definition of the forward map ``shuffle_<name>``."""
        return self._c_def(self.forward_name(), self.forward_expr)

    def inverse_c_def(self) -> str:
        """The ``static inline`` C++ definition of the inverse map ``shuffle_inv_<name>``."""
        return self._c_def(self.inverse_name(), self.inverse_expr)

    def c_definitions(self) -> str:
        """Both ``static inline`` C++ definitions (forward then inverse)."""
        return self.forward_c_def() + self.inverse_c_def()

    def numeric_forward(self) -> Callable:
        """A python callable ``sigma(index, **symbol_values)`` for numpy oracles/tests."""
        return self._numeric(self.forward_expr)

    def numeric_inverse(self) -> Callable:
        return self._numeric(self.inverse_expr)

    def _numeric(self, expr: str) -> Callable:
        code = compile(expr, f"<shuffle-num:{self.name}>", "eval")
        params = self.params

        def fn(index, **symbol_values):
            env = {"__builtins__": {"abs": abs, "min": min, "max": max, "int": int}}
            env[INDEX_VAR] = index
            for p in params:
                env[p] = symbol_values[p]
            return eval(code, env)

        return fn


def register_shuffle(name: str, forward: str, inverse: str) -> ShuffleFunction:
    """Register a value-permutation ``sigma`` under ``name``.

    :param forward:  ``sigma(i)`` as a python expression over ``i`` (+ SDFG symbols), e.g.
                     ``"i ^ 3"`` (XOR swizzle) or ``"(3*i + 1) % N"`` (affine).
    :param inverse:  ``sigma^{-1}(i)`` as a python expression -- the user supplies it; it is the
                     caller's responsibility that ``inverse(forward(i)) == i`` over the domain.
    :returns: the :class:`ShuffleFunction`. Re-registering the same name replaces it.
    """
    forward = str(forward)
    inverse = str(inverse)
    params = _symbol_params(forward, inverse)
    fn = ShuffleFunction(
        name=name,
        forward_expr=forward,
        inverse_expr=inverse,
        params=params,
        forward_cls=_make_function_class(f"shuffle_{name}", forward, params),
        inverse_cls=_make_function_class(f"shuffle_inv_{name}", inverse, params),
    )
    _REGISTRY[name] = fn
    return fn


def get_shuffle(name: str) -> ShuffleFunction:
    """The registered :class:`ShuffleFunction` ``name`` (raises ``KeyError`` if absent)."""
    if name not in _REGISTRY:
        raise KeyError(f"shuffle '{name}' is not registered; call register_shuffle('{name}', forward, inverse) first")
    return _REGISTRY[name]


def is_registered(name: str) -> bool:
    return name in _REGISTRY


def emit_shuffle_globals(sdfg: dace.SDFG, names) -> None:
    """Inject the C++ definitions of the named shuffles into ``sdfg``'s global code.

    The floored-modulo helper and each forward/inverse map are emitted at most once (keyed by the
    exact C function name), so re-emitting a shuffle -- or two shuffles that share a helper -- does
    not duplicate a definition.
    """
    existing = sdfg.global_code.get("frame", None)
    already = existing.code if existing is not None else ""

    def emit_once(func_name: str, definition: str) -> None:
        nonlocal already
        if definition in already:
            return  # exact same definition already emitted -> idempotent
        if f" {func_name}(" in already:
            # Same emitted C name, different body: a genuine collision (e.g. a shuffle named
            # 'inv_x' whose forward name equals shuffle 'x''s inverse). Fail loudly, never bind
            # one shuffle's call to another's body.
            raise ValueError(f"shuffle: C function name collision on '{func_name}' (two shuffles map "
                             f"to the same emitted name; rename one, e.g. avoid naming a shuffle 'inv_<other>').")
        sdfg.append_global_code(definition)
        already += definition

    emit_once(PYMOD_FUNC, PYMOD_DEF)
    for name in dict.fromkeys(names):  # de-dup names, keep order
        fn = get_shuffle(name)
        emit_once(fn.forward_name(), fn.forward_c_def())
        emit_once(fn.inverse_name(), fn.inverse_c_def())
