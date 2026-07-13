# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Demand-driven inference for canonical (post-ANF) expressions.

Because lowering only ever sees depth-1 ("flat") expressions, inference here
is intentionally small: it classifies an expression as a container access, a
symbolic expression, or a compile-time constant, and computes the result
descriptor for flat operator expressions. There is no separate whole-program
inference pass — rules ask on demand.

Descriptor inference for library calls (NumPy and friends) is added by the
call-lowering rules through the replacement registry; this module only covers
the operator core.
"""
import ast
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy
from dace import data, dtypes, symbolic
from dace.frontend.python import astutils
from dace.frontend.python.memlet_parser import ParseMemlet, MemletExpr
from dace.frontend.python.nextgen.common import UnsupportedFeatureError
from dace.frontend.python.nextgen.semantics import values
from dace.frontend.python.nextgen.semantics.context import ProgramContext
from dace.frontend.python.nextgen.semantics.values import StaticSequence

#: Comparison and boolean operators always produce booleans.
_BOOLEAN_OPS = (ast.Compare, ast.BoolOp)


def _apply_unary_operator(operator: ast.unaryop, value: Any) -> Any:
    """Apply a unary AST operator to a compile-time constant."""
    if isinstance(operator, ast.USub):
        return -value
    if isinstance(operator, ast.UAdd):
        return +value
    if isinstance(operator, ast.Invert):
        return ~value
    raise TypeError(f'Cannot constant-fold unary operator {type(operator).__name__}')


@dataclass
class Inferred:
    """
    Classification of a canonical expression.

    :param kind: ``'data'`` (container access), ``'symbolic'`` (symbol
                 expression), ``'constant'`` (compile-time value), or
                 ``'static'`` (compile-time Python sequence, see
                 :class:`~dace.frontend.python.nextgen.semantics.values.StaticSequence`).
    :param descriptor: Result data descriptor for ``'data'`` expressions.
    :param value: The symbolic expression, constant, or static sequence otherwise.
    """
    kind: str
    descriptor: Optional[data.Data] = None
    value: Any = None

    @property
    def is_data(self) -> bool:
        return self.kind == 'data'

    @property
    def is_pyobject(self) -> bool:
        return self.kind == 'data' and isinstance(self.descriptor.dtype, dtypes.pyobject)

    @property
    def dtype(self) -> Optional[dtypes.typeclass]:
        if self.descriptor is not None:
            return self.descriptor.dtype
        if self.kind == 'symbolic':
            return symbolic.symtype(self.value)
        if self.kind == 'constant' and isinstance(self.value, tuple(dtypes.dtype_to_typeclass().keys())):
            return dtypes.dtype_to_typeclass(type(self.value))
        return None


class _LocationShim:
    """Minimal visitor stand-in for the shared memlet parser's error reports."""

    def __init__(self, filename: str):
        self.filename = filename


def broadcast_shapes(first: Sequence[Any], second: Sequence[Any]) -> Tuple[Any, ...]:
    """
    NumPy-style shape broadcasting for symbolic shapes.

    :raises UnsupportedFeatureError: If the shapes cannot be broadcast.
    """
    result: List[Any] = []
    for dim_a, dim_b in zip(_padded(first, second), _padded(second, first)):
        if dim_a is None:
            result.append(dim_b)
        elif dim_b is None:
            result.append(dim_a)
        elif dim_a == dim_b or dim_b == 1:
            result.append(dim_a)
        elif dim_a == 1:
            result.append(dim_b)
        else:
            # Symbolically unequal dimensions: assume equality (matches the
            # stable frontend, which defers mismatches to runtime).
            result.append(dim_a)
    return tuple(result)


def _padded(shape: Sequence[Any], other: Sequence[Any]) -> List[Any]:
    pad = max(len(other) - len(shape), 0)
    return [None] * pad + list(shape)


class InferenceService:
    """Classifies canonical expressions against a :class:`ProgramContext`."""

    def __init__(self, context: ProgramContext):
        self.context = context
        self._shim = _LocationShim(context.filename)

    def infer(self, node: ast.expr) -> Inferred:
        """
        Infer the classification and result descriptor of a canonical
        expression.

        :raises UnsupportedFeatureError: If the expression cannot be inferred.
        """
        if isinstance(node, ast.Constant):
            return Inferred(kind='constant', value=node.value)
        if isinstance(node, (ast.List, ast.Tuple)):
            sequence_kind = 'list' if isinstance(node, ast.List) else 'tuple'
            return Inferred(kind='static', value=StaticSequence(elements=list(node.elts), kind=sequence_kind))
        if isinstance(node, ast.Name):
            return self._infer_name(node)
        if isinstance(node, ast.UnaryOp):
            operand = self.infer(node.operand)
            if isinstance(node.op, ast.Not):
                return self._demote_to_bool(operand)
            if operand.kind == 'constant':
                try:
                    return Inferred(kind='constant', value=_apply_unary_operator(node.op, operand.value))
                except TypeError:
                    pass
            if operand.kind == 'symbolic':
                return Inferred(kind='symbolic', value=symbolic.pystr_to_symbolic(astutils.unparse(node)))
            # Data operands: sign/inversion preserves descriptor shape and dtype
            return operand
        if isinstance(node, ast.Subscript):
            return self._infer_subscript(node)
        if isinstance(node, (ast.BinOp, ast.Compare, ast.BoolOp)):
            return self._infer_operator(node)
        raise UnsupportedFeatureError(f'Cannot infer type of expression: {astutils.unparse(node)}',
                                      self.context.filename, node)

    def parse_access(self, node: Union[ast.Name, ast.Subscript]) -> MemletExpr:
        """
        Parse a canonical data access (name or subscript of a name) into a
        memlet expression with an explicit subset, using the shared memlet
        parser.
        """
        return ParseMemlet(self._shim, self.context.defined_view(), node)

    def constant_int(self, node: ast.expr) -> Optional[int]:
        """Resolve a canonical atom to a compile-time integer, or None."""
        value = self.constant_value(node)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        return None

    def constant_value(self, node: ast.expr) -> Any:
        """Resolve a canonical atom to a compile-time value, or None."""
        try:
            inferred = self.infer(node)
        except UnsupportedFeatureError:
            return None
        if inferred.kind == 'constant':
            return inferred.value
        return None

    def sequence_constants(self, sequence: StaticSequence) -> List[Any]:
        """
        Resolve all elements of a static sequence to compile-time values.

        :raises UnsupportedFeatureError: If any element is not a compile-time
            constant (e.g., references runtime data).
        """
        result = []
        for element in sequence.elements:
            value = self.constant_value(element)
            if value is None:
                raise UnsupportedFeatureError(
                    f'Python sequence element "{astutils.unparse(element)}" is not a compile-time constant',
                    self.context.filename, element)
            result.append(value)
        return result

    def sequence_descriptor(self, sequence: StaticSequence) -> data.Array:
        """The constant-array descriptor a static sequence materializes to."""
        array = numpy.array(self.sequence_constants(sequence))
        return data.Array(dtypes.dtype_to_typeclass(array.dtype.type), list(array.shape))

    # ------------------------------------------------------------------ #

    def _infer_name(self, node: ast.Name) -> Inferred:
        binding = self.context.resolve(node.id)
        if binding is not None:
            if binding.kind == 'container':
                return Inferred(kind='data', descriptor=self.context.containers[binding.container])
            if binding.kind == 'symbol':
                return Inferred(kind='symbolic', value=self.context.symbols[node.id])
            if binding.kind == 'static':
                return Inferred(kind='static', value=self.context.static_values[node.id])
        if node.id in self.context.symbols:
            return Inferred(kind='symbolic', value=self.context.symbols[node.id])
        if node.id in self.context.constants:
            return Inferred(kind='constant', value=self.context.constants[node.id][1])
        if node.id in self.context.globals:
            value = self.context.globals[node.id]
            if isinstance(value, symbolic.symbol):
                return Inferred(kind='symbolic', value=value)
            return Inferred(kind='constant', value=value)
        raise UnsupportedFeatureError(f'Use of undefined name "{node.id}"', self.context.filename, node)

    def _infer_subscript(self, node: ast.Subscript) -> Inferred:
        base = self.infer(node.value)
        if base.kind == 'static':
            element = values.fold_subscript(base.value, node, self.constant_int)
            if isinstance(element, ast.expr):
                return self.infer(element)
            return Inferred(kind='static', value=element)
        if not base.is_data:
            raise UnsupportedFeatureError('Subscript of a non-container value', self.context.filename, node)
        expr = self.parse_access(node)
        shape = [s for s in expr.subset.size() if s != 1]
        if not shape:
            return Inferred(kind='data', descriptor=data.Scalar(base.descriptor.dtype))
        return Inferred(kind='data', descriptor=data.Array(base.descriptor.dtype, shape))

    def _infer_operator(self, node: ast.expr) -> Inferred:
        if isinstance(node, ast.BinOp):
            operands = [self.infer(node.left), self.infer(node.right)]
        elif isinstance(node, ast.Compare):
            operands = [self.infer(node.left)] + [self.infer(c) for c in node.comparators]
        else:  # BoolOp
            operands = [self.infer(v) for v in node.values]

        # Operators over Python sequences follow Python semantics at compile
        # time; sequences mixed with data operands materialize as constant
        # arrays and participate in broadcasting instead.
        if isinstance(node, ast.BinOp) and any(op.kind == 'static' for op in operands):
            if not any(op.is_data for op in operands):
                left_sequence = operands[0].value if operands[0].kind == 'static' else None
                right_sequence = operands[1].value if operands[1].kind == 'static' else None
                return Inferred(kind='static',
                                value=values.fold_binop(node, left_sequence, right_sequence, self.constant_int))
            operands = [
                Inferred(kind='data', descriptor=self.sequence_descriptor(op.value)) if op.kind == 'static' else op
                for op in operands
            ]

        # Opaque Python objects poison the expression: consuming them requires
        # the interpreter, which the dispatch seam turns into a callback.
        if any(op.is_pyobject for op in operands):
            return Inferred(kind='data', descriptor=data.Scalar(dtypes.pyobject()))

        boolean_result = isinstance(node, _BOOLEAN_OPS)
        data_operands = [op for op in operands if op.is_data]
        if not data_operands:
            # Purely symbolic/constant expression
            if boolean_result:
                return Inferred(kind='symbolic', value=symbolic.pystr_to_symbolic(astutils.unparse(node)))
            return Inferred(kind='symbolic', value=symbolic.pystr_to_symbolic(astutils.unparse(node)))

        result_dtype = self._result_dtype(operands, boolean_result)
        shape: Tuple[Any, ...] = ()
        for operand in data_operands:
            operand_shape = tuple(operand.descriptor.shape) if isinstance(operand.descriptor, data.Array) else ()
            shape = broadcast_shapes(shape, operand_shape)
        if not shape or all(s == 1 for s in shape):
            return Inferred(kind='data', descriptor=data.Scalar(result_dtype))
        return Inferred(kind='data', descriptor=data.Array(result_dtype, list(shape)))

    def _result_dtype(self, operands: List[Inferred], boolean_result: bool) -> dtypes.typeclass:
        if boolean_result:
            return dtypes.bool_
        known = [op.dtype for op in operands if op.dtype is not None]
        if not known:
            raise UnsupportedFeatureError('Cannot determine operator result type', self.context.filename)
        return dtypes.result_type_of(known[0], *known[1:]) if len(known) > 1 else known[0]

    def _demote_to_bool(self, operand: Inferred) -> Inferred:
        if operand.is_data:
            descriptor = operand.descriptor
            if isinstance(descriptor, data.Array):
                return Inferred(kind='data', descriptor=data.Array(dtypes.bool_, list(descriptor.shape)))
            return Inferred(kind='data', descriptor=data.Scalar(dtypes.bool_))
        return operand
