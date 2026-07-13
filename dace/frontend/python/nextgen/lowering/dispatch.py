# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
The type-directed dispatch seam between syntax rules and lowering mechanisms.

Syntax rules (one per canonical statement form) delegate computations here;
this module inspects *operand types* and routes to the appropriate mechanism:

- any pyobject operand: the computation must run in the interpreter
  (callback mechanism),
- static Python sequences mixed with data: materialize, then elementwise,
- data/symbolic/constant operands: elementwise map or tasklet.

Semantic feature gaps (:class:`UnsupportedFeatureError`) also fall back to the
callback mechanism, preserving totality: no user program fails to lower merely
because a construct has no dedicated mechanism yet. Future registry entries
(NumPy functions, library nodes, user dunders) plug in here rather than as
separate per-library rule sets.
"""
import ast

from dace import dtypes
from dace.frontend.python.nextgen.canonical.cpa import OpaqueStmt, statement_io_sets
from dace.frontend.python.nextgen.common import UnsupportedFeatureError
from dace.frontend.python.nextgen.lowering.access import DataAccess
from dace.frontend.python.nextgen.lowering.registry import LoweringState
from dace.frontend.python.nextgen.lowering.mechanisms import elementwise, static_values


def lower_computation(target: DataAccess, value: ast.expr, statement: ast.stmt, state: LoweringState) -> None:
    """
    Lower a canonical flat expression into a target access, dispatching on
    operand types.
    """
    try:
        value = static_values.fold_static_subscripts(value, state)
        if _consumes_pyobject(value, state):
            fallback_to_callback(statement, state, 'operates on an opaque Python object')
            return
        rewritten = static_values.materialize_operands(value, state)
        elementwise.emit_computation(target, rewritten, statement, state)
    except UnsupportedFeatureError as reason:
        fallback_to_callback(statement, state, str(reason))


def fallback_to_callback(statement: ast.stmt, state: LoweringState, reason: str) -> None:
    """Wrap a statement in a fully specified Python callback."""
    from dace.frontend.python.nextgen.lowering.rules.callbacks import lower_opaque
    reads, writes = statement_io_sets(statement)
    lower_opaque(OpaqueStmt(statement, reason, reads, writes), state)


def _consumes_pyobject(value: ast.expr, state: LoweringState) -> bool:
    """Check whether any operand of a flat expression is an opaque Python object."""
    for node in ast.walk(value):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            binding = state.context.resolve(node.id)
            if binding is not None and binding.kind == 'container':
                descriptor = state.context.containers[binding.container]
                if isinstance(descriptor.dtype, dtypes.pyobject):
                    return True
    return False
