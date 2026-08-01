# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Materialization mechanism for compile-time Python values.

Static sequences (see :mod:`~dace.frontend.python.nextgen.semantics.values`)
stay in the binding layer until an operation forces them into the dataflow
world; this module performs that conversion by registering a constant
container in the repository.
"""
import ast
from numbers import Number
from typing import Optional

import numpy

from dace import data, dtypes, subsets
from dace.frontend.python import astutils
from dace.frontend.python.nextgen.common import UnsupportedFeatureError
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
                        return ast.copy_location(astutils.copy_tree(folded), subscript_node)
                    # Sliced sequences stay static; leave the name reference
                    # for materialization to handle.
                    return subscript_node
            return self.generic_visit(subscript_node)

    return ast.fix_missing_locations(_Folder().visit(astutils.copy_tree(value)))


def fold_descriptor_properties(value: ast.expr, state: LoweringState) -> ast.expr:
    """
    Replace descriptor-property reads inside an expression with the
    compile-time value they denote: ``A.shape[0]`` becomes ``N`` (or ``20``),
    ``A.ndim`` becomes ``2``.

    Inference already resolves these, but every consumer downstream of it
    resolves ATTRIBUTES as data — ``A.shape`` names ``A``, and the operand
    silently becomes the whole array. Folding them here is what lets a shape
    read appear anywhere an ordinary number can (``A[i, j] *= A.shape[0]``, or
    a map bound), not just as an entire right-hand side.

    Registry attributes that need a real data operation (``A.T``, ``.real``,
    ``.flat``) infer as data and are left untouched, as is ``A.dtype``, whose
    value is a typeclass rather than a number.
    """

    def folded(node: ast.expr) -> Optional[ast.expr]:
        try:
            inferred = state.inference.infer(node)
        except UnsupportedFeatureError:
            return None
        if inferred.kind == 'symbolic':
            try:
                replacement = ast.parse(str(inferred.value), mode='eval').body
            except SyntaxError:
                return None
        elif inferred.kind == 'constant' and isinstance(inferred.value, Number):
            replacement = ast.Constant(value=inferred.value)
        else:
            return None
        return ast.fix_missing_locations(ast.copy_location(replacement, node))

    class _Folder(ast.NodeTransformer):

        def visit_Attribute(self, attribute_node: ast.Attribute) -> ast.AST:
            return folded(attribute_node) or self.generic_visit(attribute_node)

        def visit_Subscript(self, subscript_node: ast.Subscript) -> ast.AST:
            if isinstance(subscript_node.value, ast.Attribute):
                replacement = folded(subscript_node)
                if replacement is not None:
                    return replacement
            return self.generic_visit(subscript_node)

    if not any(isinstance(node, ast.Attribute) for node in ast.walk(value)):
        return value
    return _Folder().visit(astutils.copy_tree(value))


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

    return ast.fix_missing_locations(_Materializer().visit(astutils.copy_tree(value)))
