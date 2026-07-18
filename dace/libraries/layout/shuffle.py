# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The shuffle registry: user-defined value-permutation sigma (forward + inverse expressions), minting sympy Function classes and C++ lowerings for codegen."""
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

#: floored modulo matching Python's ``%`` (C's ``%`` truncates toward zero).
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
    """Convert a Python index expression to C++; ``%`` emitted as floored ``shuffle_pymod``."""
    tree = _FlooredMod().visit(ast.parse(expr, mode="eval"))
    ast.fix_missing_locations(tree)
    return cppunparse.pyexpr2cpp(ast.unparse(tree))


def _as_sympy(index):
    """Coerce a str / int / sympy value into a sympy expression (for a Function argument)."""
    if isinstance(index, sympy.Basic):
        return index
    return dace.symbolic.pystr_to_symbolic(str(index))


def _symbol_params(*exprs: str) -> Tuple[str, ...]:
    """Sorted SDFG-symbol identifiers used across ``exprs`` (excludes index var, math helpers, call callees)."""
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
    """Mint a ``sympy.Function`` subclass ``cname`` printing as ``cname(i, *params)``; folds constant-index calls."""
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
        return self._c_def(self.forward_name(), self.forward_expr)

    def inverse_c_def(self) -> str:
        return self._c_def(self.inverse_name(), self.inverse_expr)

    def c_definitions(self) -> str:
        return self.forward_c_def() + self.inverse_c_def()

    def numeric_forward(self) -> Callable:
        """A Python callable ``sigma(index, **symbol_values)`` for numpy oracles/tests."""
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
    """Register a value-permutation ``sigma`` under ``name`` (inverse is user-supplied); re-registering replaces it."""
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
    """Inject C++ definitions of the named shuffles into ``sdfg``'s global code; each emitted at most once."""
    existing = sdfg.global_code.get("frame", None)
    already = existing.code if existing is not None else ""

    def emit_once(func_name: str, definition: str) -> None:
        nonlocal already
        if definition in already:
            return  # exact same definition already emitted -> idempotent
        if f" {func_name}(" in already:
            # genuine collision (e.g. shuffle 'inv_x' forward name == shuffle 'x' inverse name): fail loudly
            raise ValueError(f"shuffle: C function name collision on '{func_name}' (two shuffles map "
                             f"to the same emitted name; rename one, e.g. avoid naming a shuffle 'inv_<other>').")
        sdfg.append_global_code(definition)
        already += definition

    emit_once(PYMOD_FUNC, PYMOD_DEF)
    for name in dict.fromkeys(names):  # remove duplicate names, preserve order
        fn = get_shuffle(name)
        emit_once(fn.forward_name(), fn.forward_c_def())
        emit_once(fn.inverse_name(), fn.inverse_c_def())
