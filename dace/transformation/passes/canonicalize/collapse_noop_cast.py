# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Collapse a no-op cast tasklet ``x = cast(y)`` into a plain assignment ``x = y``.

A Fortran kind coercion / numpy ``astype`` lands as a one-input tasklet whose body
is ``__out = dace.float64(__inp)`` -- the cast spelling ``dace.<typeclass>(...)`` that
``_datatype_converter`` emits (the Fortran frontend and the sympy printer also produce
the bare ``float64(...)`` form). Such a cast is pure noise -- it changes nothing --
exactly when BOTH:

  (a) the argument ``y`` is a data/symbol reference (a ``Name`` or ``Subscript``), NOT a
      numeric constant literal (``float64(2.0)`` is a *typed constant*, left untouched); and
  (b) the destination ``x``'s declared dtype already equals the cast's target dtype --
      the conversion is to the type ``x`` already has.

When both hold the body is rewritten to the plain assignment ``x = y``. That assignment
is then legitimately trivial and may be folded into a direct copy by
``TrivialTaskletElimination``. A genuine cast (differing dtypes) fails (b) and is kept,
so no conversion is ever silently dropped.
"""
import ast
from typing import Any, Dict, Optional, Tuple

from dace import SDFG, dtypes
from dace.frontend.python import astutils
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.transformation import pass_pipeline as ppl, transformation


def string_to_typeclass() -> Dict[str, dtypes.typeclass]:
    """Map every scalar dtype's bare spelling (``float64``, ``int32``, ...) back to its
    typeclass. Built from the registry so it tracks the dtype list -- never hardcoded."""
    return {tc.to_string(): tc for tc in dtypes.TYPECLASS_TO_STRING}


def noop_cast_candidate(code: CodeBlock,
                        casts: Dict[str, dtypes.typeclass]) -> Optional[Tuple[str, ast.AST, dtypes.typeclass]]:
    """If ``code`` is a single ``x = cast(y)`` with ``y`` a data/symbol reference and
    ``cast`` a dtype conversion, return ``(target_conn, arg_ast, cast_dtype)``; else
    ``None``. The dtype-equality checks (a genuine no-op) are applied by the caller."""
    body = code.code
    if len(body) != 1 or not isinstance(body[0], ast.Assign):
        return None
    assign = body[0]
    if len(assign.targets) != 1 or not isinstance(assign.targets[0], ast.Name):
        return None
    call = assign.value
    if not isinstance(call, ast.Call) or len(call.args) != 1 or call.keywords:
        return None
    arg = call.args[0]
    # (a) ``y`` must be a data/symbol reference, not a numeric constant or a compound expression.
    if not isinstance(arg, (ast.Name, ast.Subscript)):
        return None
    cast_name = astutils.rname(call.func).split('.')[-1]
    if cast_name not in casts:
        return None
    return assign.targets[0].id, arg, casts[cast_name]


def base_name(arg: ast.AST) -> str:
    """The referenced data/symbol name of a ``Name`` (``y``) or ``Subscript`` (``y[i]``)."""
    return arg.id if isinstance(arg, ast.Name) else astutils.rname(arg.value)


def destination_dtype(state, tasklet: nodes.Tasklet, out_conn: str, sdfg: SDFG) -> Optional[dtypes.typeclass]:
    """The declared dtype ``x`` is written into: the dtype of the array the output
    connector ``out_conn`` feeds."""
    for edge in state.out_edges(tasklet):
        if edge.src_conn == out_conn and edge.data is not None and edge.data.data is not None:
            return sdfg.arrays[edge.data.data].dtype
    return None


def source_dtype(state, tasklet: nodes.Tasklet, in_name: str, sdfg: SDFG) -> Optional[dtypes.typeclass]:
    """The dtype of the value ``y`` reads: the array behind the matching input connector,
    or -- when ``y`` is a free symbol rather than a connector -- the symbol's dtype."""
    for edge in state.in_edges(tasklet):
        if edge.dst_conn == in_name and edge.data is not None and edge.data.data is not None:
            return sdfg.arrays[edge.data.data].dtype
    return sdfg.symbols.get(in_name)


@transformation.explicit_cf_compatible
class CollapseNoOpCast(ppl.Pass):
    """Rewrite a no-op cast tasklet ``x = cast(y)`` into the plain assignment ``x = y`` --
    only when ``y`` is a non-constant data/symbol reference and source, destination and
    cast-target dtype all coincide (the cast is identity and the store is exact)."""

    CATEGORY: str = 'Canonicalization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Tasklets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[int]:
        casts = string_to_typeclass()
        count = 0
        for nsdfg in sdfg.all_sdfgs_recursive():
            for state in nsdfg.states():
                for tasklet in [n for n in state.nodes() if isinstance(n, nodes.Tasklet)]:
                    candidate = noop_cast_candidate(tasklet.code, casts)
                    if candidate is None:
                        continue
                    target_conn, arg, cast_dtype = candidate
                    dst_dtype = destination_dtype(state, tasklet, target_conn, nsdfg)
                    src_dtype = source_dtype(state, tasklet, base_name(arg), nsdfg)
                    # A genuine no-op: the cast is identity (source already the target dtype) AND
                    # the destination has that same dtype (guard (b)). Then ``cast(y)`` == ``y``
                    # bit-for-bit and the store is exact -- rewriting drops pure noise. Any
                    # mismatch is a real conversion and is left as a cast.
                    if src_dtype != cast_dtype or dst_dtype != cast_dtype:
                        continue
                    tasklet.code = CodeBlock(f'{target_conn} = {astutils.unparse(arg).strip()}')
                    count += 1
        return count or None


__all__ = ['CollapseNoOpCast']
