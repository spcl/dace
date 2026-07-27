# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Stream-push mechanism for dataflow scopes.

``stream.push(value)`` normally lowers through the replacement registry
(``Stream.push``, see :mod:`dace.frontend.python.replacements.streams`) as a
deferred :class:`~dace.sdfg.analysis.schedule_tree.treenodes.ReplacementCallNode`.
That expansion adds state machinery and therefore cannot run inside a map or
consume scope, which is exactly where a filtering push belongs::

    for i in dace.map[0:N]:
        if A[i] >= 0.5:
            ostream.push(A[i])

Inside a dataflow scope the push is instead emitted directly as a tasklet with
a dynamic write to the stream -- the same shape the explicit-tasklet rule
produces for ``b >> ostream(-1)``.
"""
import ast

from dace import data, dtypes, nodes, subsets
from dace.memlet import Memlet
from dace.sdfg.analysis.schedule_tree import treenodes as tn

from dace.frontend.python.nextgen.lowering.access import resolve_access
from dace.frontend.python.nextgen.lowering.registry import LoweringState


def lower_stream_push(call: ast.Call, statement: ast.stmt, state: LoweringState) -> bool:
    """
    Emit a dynamic single-element stream push as a tasklet inside the current
    dataflow scope.

    :return: False when the call is not a stream push, when the mechanism is
             not needed (outside a dataflow scope the registry expansion
             produces the classic-identical form, including bulk pushes), or
             when the pushed value is not a single element this tasklet form
             can carry. The syntactic checks come first so that no other call
             form pays for receiver resolution.
    """
    if not state.emitter.in_dataflow_scope:
        return False
    if not isinstance(call.func, ast.Attribute) or call.func.attr != 'push':
        return False
    if not isinstance(call.func.value, (ast.Name, ast.Attribute)):
        return False
    if len(call.args) != 1 or call.keywords:
        # An explicit element count is a static-volume bulk push, which has no
        # tasklet form; fall back rather than silently pushing one element.
        return False

    receiver = resolve_access(call.func.value, state)
    if receiver is None or not isinstance(receiver.descriptor, data.Stream):
        return False
    if isinstance(receiver.descriptor.dtype, dtypes.pyobject):
        return False

    value = resolve_access(call.args[0], state)
    if value is None or value.indirect or value.subset.num_elements() != 1:
        return False

    line = getattr(statement, 'lineno', 0)
    tasklet = nodes.Tasklet(f'push_{line}', {'__inp'}, {'__out'}, '__out = __inp')
    out_memlet = Memlet(data=receiver.container, subset=subsets.Range.from_array(receiver.descriptor), dynamic=True)
    state.emitter.emit(
        tn.TaskletNode(node=tasklet,
                       in_memlets={'__inp': Memlet(data=value.container, subset=value.subset)},
                       out_memlets={'__out': out_memlet}))
    return True
