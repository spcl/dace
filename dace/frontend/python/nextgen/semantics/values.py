# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Compile-time value domain for the next-generation Python frontend.

Native Python containers (lists, tuples) are not data containers: they are
*static values* tracked in the binding layer. Operations on them follow Python
semantics and are evaluated at parse time (concatenation, repetition,
indexing, slicing) without emitting tree nodes. A static value only touches
the dataflow world when it is materialized — e.g., used as an operand of an
array operation or passed to a library call — at which point it becomes a
constant container.

Static sequences store their elements as canonical *atom ASTs*, so elements
may be compile-time constants, symbolic expressions, or references to runtime
data (the ANF pass guarantees element expressions are pre-flattened).
"""
import ast
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union

from dace.frontend.python.nextgen.common import UnsupportedFeatureError


@dataclass
class StaticSequence:
    """
    A compile-time Python sequence (list or tuple) of canonical atom ASTs.
    """
    elements: List[ast.expr] = field(default_factory=list)
    kind: str = 'list'  #: ``'list'`` or ``'tuple'``

    def __len__(self) -> int:
        return len(self.elements)

    def concat(self, other: 'StaticSequence') -> 'StaticSequence':
        """Python ``+`` on sequences."""
        result_kind = self.kind if self.kind == other.kind else 'list'
        return StaticSequence(elements=list(self.elements) + list(other.elements), kind=result_kind)

    def repeat(self, count: int) -> 'StaticSequence':
        """Python ``*`` on sequences."""
        return StaticSequence(elements=list(self.elements) * count, kind=self.kind)

    def getitem(self, index: int) -> ast.expr:
        """Python integer indexing, including negative indices."""
        return self.elements[index]

    def getslice(self, lower, upper, step) -> 'StaticSequence':
        """Python slicing semantics."""
        return StaticSequence(elements=self.elements[slice(lower, upper, step)], kind=self.kind)


#: Resolves a canonical atom AST to a compile-time int, or None.
ConstantResolver = Callable[[ast.expr], Optional[int]]


def fold_subscript(sequence: StaticSequence, node: ast.Subscript,
                   constant_of: ConstantResolver) -> Union[ast.expr, StaticSequence]:
    """
    Apply Python subscript semantics to a static sequence at compile time.

    :param sequence: The static sequence being indexed.
    :param node: The canonical subscript AST.
    :param constant_of: Resolver from atom ASTs to compile-time integers.
    :return: The selected element AST (integer index) or a sliced sequence.
    :raises UnsupportedFeatureError: If the index is not a compile-time value.
    """
    index_node = node.slice
    if isinstance(index_node, ast.Slice):
        parts = []
        for part in (index_node.lower, index_node.upper, index_node.step):
            if part is None:
                parts.append(None)
            else:
                constant = constant_of(part)
                if constant is None:
                    raise UnsupportedFeatureError('Slicing a Python sequence requires compile-time bounds', node=node)
                parts.append(constant)
        return sequence.getslice(*parts)
    constant = constant_of(index_node)
    if constant is None:
        raise UnsupportedFeatureError('Indexing a Python sequence requires a compile-time index', node=node)
    try:
        return sequence.getitem(constant)
    except IndexError:
        raise UnsupportedFeatureError(f'Static sequence index {constant} out of range '
                                      f'(length {len(sequence)})',
                                      node=node)


def fold_binop(node: ast.BinOp, left: Optional[StaticSequence], right: Optional[StaticSequence],
               constant_of: ConstantResolver) -> StaticSequence:
    """
    Apply Python operator semantics to static sequence operands at compile
    time: ``+`` concatenates, ``*`` repeats.

    :raises UnsupportedFeatureError: For any other operator or operand mix.
    """
    if isinstance(node.op, ast.Add) and left is not None and right is not None:
        return left.concat(right)
    if isinstance(node.op, ast.Mult):
        sequence, count_node = (left, node.right) if left is not None else (right, node.left)
        if sequence is not None:
            count = constant_of(count_node)
            if count is not None:
                return sequence.repeat(count)
    raise UnsupportedFeatureError(f'Unsupported operation on a Python sequence: {type(node.op).__name__}', node=node)
