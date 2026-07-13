# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Lowering rule for :class:`OpaqueStmt` markers: statements that must execute in
the Python interpreter become fully specified :class:`PythonCallbackNode`\\ s.

The callback contract:

- ``input_names``/``output_names`` list the repository containers the callback
  reads and writes, derived from the statement's precomputed I/O sets.
- Written names that have no container yet are registered as ``pyobject``
  scalars so subsequent statements can reference (and pass through) the
  resulting Python objects.
- The node is a side-effect fence: later passes must not reorder memory
  accesses across it. (Enforced by the verifier through the presence of the
  reason and I/O metadata; the tree-to-SDFG lowering maps this onto the
  ``__pystate`` serialization edge of the stable frontend's callback ABI.)
"""
import ast

from dace import data, dtypes
from dace.properties import CodeBlock
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.frontend.python.nextgen.canonical.cpa import OpaqueStmt
from dace.frontend.python.nextgen.lowering.registry import LoweringState, rule


@rule(OpaqueStmt)
def lower_opaque(statement: OpaqueStmt, state: LoweringState) -> None:
    input_names = []
    for name in sorted(statement.inputs):
        binding = state.context.resolve(name)
        if binding is not None and binding.kind == 'container':
            input_names.append(binding.container)

    output_names = []
    for name in sorted(statement.outputs):
        binding = state.context.resolve(name)
        if binding is not None and binding.kind == 'container':
            output_names.append(binding.container)
        else:
            # The callback produces a Python object we cannot type: register an
            # opaque scalar so later statements can bind and pass it through.
            container_name = state.context.add_container(name, data.Scalar(dtypes.pyobject()))
            state.context.bind(name, container_name)
            output_names.append(container_name)

    code = ast.unparse(ast.fix_missing_locations(statement.original))
    state.emitter.emit(
        tn.PythonCallbackNode(code=CodeBlock(code),
                              reason=statement.reason,
                              input_names=input_names,
                              output_names=output_names))
