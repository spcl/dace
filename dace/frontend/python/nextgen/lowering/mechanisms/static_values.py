# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Materialization mechanism for compile-time Python values.

Static sequences (see :mod:`~dace.frontend.python.nextgen.semantics.values`)
stay in the binding layer until an operation forces them into the dataflow
world; this module performs that conversion by registering a constant
container in the repository.
"""
import ast
import copy

import numpy

from dace import data, dtypes, subsets
from dace.frontend.python.nextgen.lowering.access import DataAccess
from dace.frontend.python.nextgen.lowering.registry import LoweringState
from dace.frontend.python.nextgen.semantics import values
from dace.frontend.python.nextgen.semantics.values import StaticSequence


def materialize(sequence: StaticSequence, state: LoweringState, name_hint: str = '__const') -> DataAccess:
    """
    Materialize a static sequence of compile-time constants as a constant
    container in the repository.

    :raises UnsupportedFeatureError: If any element is not a compile-time
        constant (materializing sequences with runtime elements is a planned
        extension that requires per-element initialization).
    :return: A full-range access of the new constant container.
    """
    array = numpy.array(state.inference.sequence_constants(sequence))
    descriptor = data.Array(dtypes.dtype_to_typeclass(array.dtype.type), list(array.shape))
    container_name = state.context.add_constant_container(name_hint, descriptor, array)
    # Bind the repository name to itself so access resolution treats the
    # constant like any other container.
    state.context.bind(container_name, container_name)
    return DataAccess(container_name, subsets.Range.from_array(descriptor), descriptor)


def fold_static_subscripts(value: ast.expr, state: LoweringState) -> ast.expr:
    """
    Replace subscripts of static sequences with their compile-time elements
    inside a canonical flat expression (e.g. ``sizes[1]`` becomes ``20.0``).
    Non-constant indices raise
    :class:`~dace.frontend.python.nextgen.common.UnsupportedFeatureError`
    through the value-domain folding, which the dispatch seam converts to a
    callback.
    """

    class _Folder(ast.NodeTransformer):

        def visit_Subscript(self, subscript_node: ast.Subscript) -> ast.AST:
            if isinstance(subscript_node.ctx, ast.Load) and isinstance(subscript_node.value, ast.Name):
                sequence = state.context.static_value_of(subscript_node.value.id)
                if sequence is not None:
                    folded = values.fold_subscript(sequence, subscript_node, state.inference.constant_int)
                    if isinstance(folded, ast.expr):
                        return ast.copy_location(copy.deepcopy(folded), subscript_node)
                    # Sliced sequences stay static; leave the name reference
                    # for materialization to handle.
                    return subscript_node
            return self.generic_visit(subscript_node)

    return ast.fix_missing_locations(_Folder().visit(copy.deepcopy(value)))


def materialize_operands(value: ast.expr, state: LoweringState) -> ast.expr:
    """
    Replace static-sequence name operands inside a canonical flat expression
    with materialized constant containers, returning the rewritten expression.
    Non-static parts are left untouched.
    """

    class _Materializer(ast.NodeTransformer):

        def visit_Name(self, name_node: ast.Name) -> ast.Name:
            if isinstance(name_node.ctx, ast.Load):
                sequence = state.context.static_value_of(name_node.id)
                if sequence is not None:
                    access = materialize(sequence, state, name_hint=f'__const_{name_node.id}')
                    return ast.copy_location(ast.Name(id=access.container, ctx=ast.Load()), name_node)
            return name_node

    return ast.fix_missing_locations(_Materializer().visit(copy.deepcopy(value)))
