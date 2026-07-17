# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Lowering rule for explicit-dataflow tasklets: ``with dace.tasklet:`` blocks,
``@dace.tasklet`` functions, and the tasklet bodies of ``@dace.map`` functions
(recognized during canonicalization as
:class:`~dace.frontend.python.nextgen.canonical.cpa.ExplicitTasklet` markers).

The memlet syntax (``local << A[i]`` for inputs, ``local >> B[i]`` and
``local >> B(1, lambda a, b: a + b)[i]`` for outputs) is parsed with the
shared :func:`~dace.frontend.python.memlet_parser.parse_memlet`, porting the
semantics of the stable frontend's ``TaskletTransformer``. A top-level string
statement provides intrinsic (C++) tasklet code.

Unlike general Python statements, malformed explicit-dataflow constructs raise
frontend errors instead of falling back to callbacks: this is dace-specific
syntax, so a violation is a user error (matching the stable frontend's
``TaskletTransformer`` contract), not an unsupported-language feature.

Global-scope, initialization, and finalization code attach to the tasklet
through ``with dace.tasklet(code_global=..., code_init=..., code_exit=...)``
keyword arguments and land on the emitted :class:`~dace.sdfg.nodes.Tasklet`'s
``code_global``/``code_init``/``code_exit`` properties.
"""
import ast
from typing import Dict, List, Optional

from dace import dtypes
from dace.memlet import Memlet
from dace.sdfg import nodes
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.frontend.python import astutils
from dace.frontend.python.memlet_parser import parse_memlet
from dace.frontend.python.nextgen.canonical.cpa import ExplicitTasklet
from dace.frontend.python.nextgen.common import FrontendError, UnsupportedFeatureError
from dace.frontend.python.nextgen.lowering.registry import LoweringState, rule
from dace.frontend.python.nextgen.semantics.inference import _LocationShim


def _shim(state: LoweringState) -> _LocationShim:
    return _LocationShim(state.context.filename)


@rule(ExplicitTasklet)
def lower_explicit_tasklet(statement: ExplicitTasklet, state: LoweringState) -> None:
    in_memlets: Dict[str, Memlet] = {}
    out_memlets: Dict[str, Memlet] = {}
    code_statements: List[ast.stmt] = []
    intrinsic_code: Optional[str] = None
    defined = state.context.defined_view()

    for body_statement in statement.statements:
        binop = _memlet_binop(body_statement)
        if binop is not None:
            if isinstance(binop.op, ast.LShift):  # local << A[...]
                connector, memlet = parse_memlet(_shim(state), binop.right, binop.left, defined)
                _check_connector(connector, in_memlets, out_memlets, state, body_statement)
                in_memlets[connector] = _to_repository(memlet, state, body_statement)
            else:  # local >> A[...]
                connector, memlet = parse_memlet(_shim(state), binop.left, binop.right, defined)
                _check_connector(connector, in_memlets, out_memlets, state, body_statement)
                out_memlets[connector] = _to_repository(memlet, state, body_statement)
            continue
        if (isinstance(body_statement, ast.Expr) and isinstance(body_statement.value, ast.Constant)
                and isinstance(body_statement.value.value, str)):
            # Intrinsic implementation (defaults to C++, as in the stable frontend)
            if intrinsic_code is not None:
                raise FrontendError('Cannot provide more than one intrinsic implementation for a tasklet',
                                    state.context.filename, body_statement)
            intrinsic_code = body_statement.value.value
            continue
        code_statements.append(body_statement)

    language = _language(statement, intrinsic_code, state)
    if intrinsic_code is not None:
        code = intrinsic_code
    else:
        code = '\n'.join(astutils.unparse(s) for s in code_statements)

    tasklet = nodes.Tasklet(statement.label,
                            set(in_memlets.keys()),
                            set(out_memlets.keys()),
                            code,
                            language=language,
                            code_global=statement.code_global,
                            code_init=statement.code_init,
                            code_exit=statement.code_exit)
    if statement.side_effects is not None:
        tasklet.side_effects = statement.side_effects
    state.emitter.emit(tn.TaskletNode(node=tasklet, in_memlets=in_memlets, out_memlets=out_memlets))


def _memlet_binop(statement: ast.stmt) -> Optional[ast.BinOp]:
    """Return the shift binop of a memlet statement, or None."""
    if (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.BinOp)
            and isinstance(statement.value.op, (ast.LShift, ast.RShift))):
        return statement.value
    return None


def _check_connector(connector: Optional[str], in_memlets: Dict[str, Memlet], out_memlets: Dict[str, Memlet],
                     state: LoweringState, statement: ast.stmt) -> None:
    if connector is None:
        raise UnsupportedFeatureError('Memlet statements require a local connector name', state.context.filename,
                                      statement)
    if connector in in_memlets or connector in out_memlets:
        raise FrontendError(f'Local variable "{connector}" is already a tasklet input or output',
                            state.context.filename, statement)


def _to_repository(memlet: Memlet, state: LoweringState, statement: ast.stmt) -> Memlet:
    """Rewrite a parsed memlet to reference the repository container name."""
    binding = state.context.resolve(memlet.data)
    if binding is None or binding.kind != 'container':
        raise UnsupportedFeatureError(f'Tasklet memlet references unknown container "{memlet.data}"',
                                      state.context.filename, statement)
    memlet.data = binding.container
    return memlet


def _language(statement: ExplicitTasklet, intrinsic_code: Optional[str], state: LoweringState) -> dtypes.Language:
    if statement.language is not None:
        try:
            return dtypes.Language[statement.language]
        except KeyError:
            raise FrontendError(f'Unknown tasklet language "{statement.language}"', state.context.filename, statement)
    return dtypes.Language.CPP if intrinsic_code is not None else dtypes.Language.Python
