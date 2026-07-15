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

_REGISTRY: "Dict[str, ShuffleFunction]" = {}


def _as_sympy(index):
    """Coerce a str / int / sympy value into a sympy expression (for a Function argument)."""
    if isinstance(index, sympy.Basic):
        return index
    return dace.symbolic.pystr_to_symbolic(str(index))


def _symbol_params(*exprs: str) -> Tuple[str, ...]:
    """The sorted SDFG-symbol identifiers used across ``exprs`` (every free name but ``i``)."""
    names = set()
    for expr in exprs:
        for node in ast.walk(ast.parse(expr, mode="eval")):
            if isinstance(node, ast.Name) and node.id not in _KNOWN_NAMES:
                names.add(node.id)
    return tuple(sorted(names))


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

    def c_definitions(self) -> str:
        """The two ``static inline`` C++ definitions of the forward and inverse maps."""
        sig_params = "".join(f", long long {p}" for p in self.params)
        fwd_c = cppunparse.pyexpr2cpp(self.forward_expr)
        inv_c = cppunparse.pyexpr2cpp(self.inverse_expr)
        return (f"\nstatic inline long long {self.forward_name()}(long long {INDEX_VAR}{sig_params}) "
                f"{{ return {fwd_c}; }}\n"
                f"static inline long long {self.inverse_name()}(long long {INDEX_VAR}{sig_params}) "
                f"{{ return {inv_c}; }}\n")

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
    """Inject the C++ definitions of the named shuffles into ``sdfg``'s global code (once each)."""
    existing = sdfg.global_code.get("frame", None)
    already = existing.code if existing is not None else ""
    for name in dict.fromkeys(names):  # de-dup, keep order
        fn = get_shuffle(name)
        if f"{fn.forward_name()}(" in already:
            continue
        sdfg.append_global_code(fn.c_definitions())
        already += fn.c_definitions()
