# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Lowering rules for canonical calls in assignment position.

This module is the future integration point for the replacement registry
(NumPy and library functions), nested ``@dace.program`` parsing, and explicit
dataflow constructs. Until those rules land, calls fall back to the callback
path with full I/O specifications — the same totality guarantee as any other
opaque statement.
"""
import ast

from dace.frontend.python.nextgen.lowering.registry import LoweringState


def lower_call_assign(statement: ast.Assign, state: LoweringState) -> None:
    """
    Lower ``target = f(args...)``.

    TODO(phase 3): resolve ``f`` against the replacement registry (NumPy
    functions and methods), nested ``@dace.program`` objects (recursive
    pipeline invocation with a shared repository), and SDFG convertibles.
    Unresolvable callables fall back to callbacks below.
    """
    from dace.frontend.python.nextgen.lowering.dispatch import fallback_to_callback
    fallback_to_callback(statement, state, 'unresolved call')
