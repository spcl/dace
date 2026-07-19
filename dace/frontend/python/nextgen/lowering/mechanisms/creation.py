# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Array-creation mechanism: lowers NumPy creation calls (``zeros``, ``ones``,
``full``, ``empty``, their ``*_like`` variants, ``copy``, and ``arange``) into
frontend-legal schedule tree nodes.

The target container is allocated by the call rule from the registry-inferred
descriptor before this mechanism runs; this module only emits the content
initialization (a fill map, a copy, or nothing for ``empty``).
"""
import ast
from typing import Optional

from dace.memlet import Memlet
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.frontend.python import astutils
from dace.frontend.python.nextgen.common import UnsupportedFeatureError
from dace.frontend.python.nextgen.lowering.access import DataAccess, resolve_access
from dace.frontend.python.nextgen.lowering.mechanisms import elementwise
from dace.frontend.python.nextgen.lowering.registry import LoweringState

#: Creation calls this mechanism lowers, by registry-qualified name.
CREATION_CALLS = frozenset({
    'numpy.zeros', 'numpy.ones', 'numpy.full', 'numpy.empty', 'numpy.zeros_like', 'numpy.ones_like', 'numpy.full_like',
    'numpy.empty_like', 'numpy.copy', 'numpy.arange', 'dace.define_stream', 'dace.define_streamarray'
})


def lower_creation(qualname: str, target: DataAccess, call: ast.Call, statement: ast.stmt,
                   state: LoweringState) -> None:
    """
    Emit the content initialization of an array-creation call into an
    already-allocated target container.

    :raises UnsupportedFeatureError: If the call form is not supported (the
                                     dispatch seam converts this to a callback).
    """
    function = qualname.split('.', 1)[1]
    if function in ('empty', 'empty_like', 'define_stream', 'define_streamarray'):
        return  # Allocation only, contents are undefined (streams start empty)
    if function in ('zeros', 'zeros_like'):
        _emit_fill(target, ast.Constant(value=0), statement, state)
        return
    if function in ('ones', 'ones_like'):
        _emit_fill(target, ast.Constant(value=1), statement, state)
        return
    if function in ('full', 'full_like'):
        if len(call.args) < 2:
            raise UnsupportedFeatureError(f'"{qualname}" requires an explicit fill value',
                                          state.context.filename,
                                          statement,
                                          category='array-creation')
        _emit_fill(target, call.args[1], statement, state)
        return
    if function == 'copy':
        source = resolve_access(call.args[0], state) if call.args else None
        if source is None:
            raise UnsupportedFeatureError(f'"{qualname}" requires a data source',
                                          state.context.filename,
                                          statement,
                                          category='array-creation')
        state.emitter.emit(
            tn.CopyNode(target=target.container,
                        memlet=Memlet(data=source.container, subset=source.subset, other_subset=target.subset)))
        return
    if function == 'arange':
        _emit_arange(target, call, statement, state)
        return
    raise UnsupportedFeatureError(f'Unsupported creation call "{qualname}"',
                                  state.context.filename,
                                  statement,
                                  category='array-creation')


def _emit_fill(target: DataAccess, value: ast.expr, statement: ast.stmt, state: LoweringState) -> None:
    """Fill the target with a (constant, symbolic, or scalar-data) value."""
    ast.fix_missing_locations(ast.copy_location(value, statement))
    elementwise.emit_computation(target, value, statement, state)


def _emit_arange(target: DataAccess, call: ast.Call, statement: ast.stmt, state: LoweringState) -> None:
    """Emit ``__out = start + __i0 * step`` over the 1-D target extent."""
    from dace.frontend.python.nextgen.lowering.access import nondegenerate_shape
    if len(nondegenerate_shape(target.subset)) != 1:
        raise UnsupportedFeatureError('numpy.arange requires a one-dimensional result',
                                      state.context.filename,
                                      statement,
                                      category='array-creation')
    start: Optional[str] = None
    step: Optional[str] = None
    if len(call.args) == 1:
        start, step = '0', '1'
    elif len(call.args) == 2:
        start, step = _scalar_code(call.args[0], state), '1'
    elif len(call.args) >= 3:
        start = _scalar_code(call.args[0], state)
        step = _scalar_code(call.args[2], state)
    if start is None or step is None:
        raise UnsupportedFeatureError('numpy.arange bounds must be constants or symbolic expressions',
                                      state.context.filename,
                                      statement,
                                      category='array-creation')
    elementwise.emit_elementwise(target, f'({start}) + __i0 * ({step})', [], statement, state)


def _scalar_code(node: ast.expr, state: LoweringState) -> Optional[str]:
    """Render a canonical atom as scalar code, if it is compile-time or symbolic."""
    inferred = state.inference.infer(node)
    if inferred.kind in ('constant', 'symbolic'):
        return astutils.unparse(node) if inferred.kind == 'symbolic' else repr(inferred.value)
    return None
