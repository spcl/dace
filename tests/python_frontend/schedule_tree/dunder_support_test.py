# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import ast
import math

import dace
import pytest
from dace.frontend.python import astutils
from dace.frontend.python.common import SDFGConvertible
from dace.frontend.python.schedule_tree import desugar_schedule_tree_expansions
from dace.sdfg.analysis.schedule_tree import treenodes as tn


def _desugar_statements(source: str, *, global_vars=None, known_descriptors=None):
    module = ast.parse(source)
    desugared = desugar_schedule_tree_expansions(module,
                                                 filename='<test>',
                                                 global_vars=dict(global_vars or {}),
                                                 known_descriptors=known_descriptors)
    return [astutils.unparse(statement) for statement in desugared.body]


class DunderHost:

    def __call__(self, value):
        return value

    def __add__(self, value):
        return value

    def __radd__(self, value):
        return value

    def __rmatmul__(self, value):
        return value

    def __iadd__(self, value):
        return value

    def __neg__(self):
        return 1

    def __pos__(self):
        return 1

    def __invert__(self):
        return 1

    def __eq__(self, value):
        return False

    def __contains__(self, value):
        return False

    def __getitem__(self, index):
        return self

    def __setitem__(self, index, value):
        return None

    def __delitem__(self, index):
        return None

    def __hash__(self):
        return 1

    def __repr__(self):
        return 'host'

    def __str__(self):
        return 'host'

    def __bool__(self):
        return True

    def __int__(self):
        return 1

    def __float__(self):
        return 1.0

    def __bytes__(self):
        return b'host'

    def __complex__(self):
        return 1j

    def __format__(self, spec):
        return spec

    def __len__(self):
        return 1

    def __iter__(self):
        return iter(())

    def __reversed__(self):
        return iter(())

    def __next__(self):
        return 1

    def __divmod__(self, value):
        return value

    def __rdivmod__(self, value):
        return value

    def __abs__(self):
        return 1

    def __round__(self, digits=None):
        return 1

    def __trunc__(self):
        return 1

    def __floor__(self):
        return 1

    def __ceil__(self):
        return 1

    def __dir__(self):
        return []


class ClassSubscriptable:

    @classmethod
    def __class_getitem__(cls, value):
        return value


class MetaCheck(type):

    def __instancecheck__(cls, value):
        return True

    def __subclasscheck__(cls, value):
        return True


class Checked(metaclass=MetaCheck):
    pass


@pytest.mark.parametrize(('source', 'expected'), [
    ('return obj(A)', 'return obj.__call__(A)'),
    ('return obj + A', 'return obj.__add__(A)'),
    ('return A + obj', 'return obj.__radd__(A)'),
    ('return A @ obj', 'return obj.__rmatmul__(A)'),
    ('return -obj', 'return obj.__neg__()'),
    ('return +obj', 'return obj.__pos__()'),
    ('return ~obj', 'return obj.__invert__()'),
    ('return obj == A', 'return obj.__eq__(A)'),
    ('return A in obj', 'return obj.__contains__(A)'),
    ('return obj[i]', 'return obj.__getitem__(i)'),
    ('obj[i] = A', 'obj.__setitem__(i, A)'),
    ('del obj[i]', 'obj.__delitem__(i)'),
    ('return hash(obj)', 'return obj.__hash__()'),
    ('return repr(obj)', 'return obj.__repr__()'),
    ('return str(obj)', 'return obj.__str__()'),
    ('return bool(obj)', 'return obj.__bool__()'),
    ('return int(obj)', 'return obj.__int__()'),
    ('return float(obj)', 'return obj.__float__()'),
    ('return bytes(obj)', 'return obj.__bytes__()'),
    ('return complex(obj)', 'return obj.__complex__()'),
    ('return format(obj, spec)', 'return obj.__format__(spec)'),
    ('return len(obj)', 'return obj.__len__()'),
    ('return iter(obj)', 'return obj.__iter__()'),
    ('return reversed(obj)', 'return obj.__reversed__()'),
    ('return next(obj)', 'return obj.__next__()'),
    ('return divmod(obj, A)', 'return obj.__divmod__(A)'),
    ('return divmod(A, obj)', 'return obj.__rdivmod__(A)'),
    ('return abs(obj)', 'return obj.__abs__()'),
    ('return round(obj)', 'return obj.__round__()'),
    ('return round(obj, digits)', 'return obj.__round__(digits)'),
    ('return math.trunc(obj)', 'return obj.__trunc__()'),
    ('return math.floor(obj)', 'return obj.__floor__()'),
    ('return math.ceil(obj)', 'return obj.__ceil__()'),
    ('return dir(obj)', 'return obj.__dir__()'),
    ('return T[A]', 'return T.__class_getitem__(A)'),
    ('return isinstance(x, T)', 'return T.__instancecheck__(x)'),
    ('return issubclass(U, T)', 'return T.__subclasscheck__(U)'),
])
def test_schedule_tree_dunder_desugaring_rewrites_supported_sugar(source, expected):
    statements = _desugar_statements(source,
                                     global_vars={
                                         'obj': DunderHost(),
                                         'math': math,
                                         'T': ClassSubscriptable,
                                         'Checked': Checked,
                                     })

    if 'instancecheck' in expected or 'subclasscheck' in expected:
        statements = _desugar_statements(source, global_vars={'T': Checked})

    assert statements == [expected]


def test_schedule_tree_dunder_desugaring_leaves_parseable_free_function_call_direct():

    def callee(value):
        return value

    statements = _desugar_statements('return callee(A)', global_vars={'callee': callee})

    assert statements == ['return callee(A)']


def test_schedule_tree_dunder_desugaring_prefers_direct_operator_on_distinct_objects():

    class LeftDirect:

        def __matmul__(self, other):
            return other

    class RightReflected:

        def __rmatmul__(self, other):
            return other

    statements = _desugar_statements('return lhs @ rhs', global_vars={'lhs': LeftDirect(), 'rhs': RightReflected()})

    assert statements == ['return lhs.__matmul__(rhs)']


def test_schedule_tree_dunder_desugaring_uses_reflected_operator_when_left_is_missing_direct():

    class LeftPlain:
        pass

    class RightReflected:

        def __rmatmul__(self, other):
            return other

    statements = _desugar_statements('return lhs @ rhs', global_vars={'lhs': LeftPlain(), 'rhs': RightReflected()})

    assert statements == ['return rhs.__rmatmul__(lhs)']


def test_schedule_tree_dunder_desugaring_leaves_class_construction_direct():

    class Builder:

        def __init__(self, value):
            self.value = value

    statements = _desugar_statements('return Builder(A)', global_vars={'Builder': Builder})

    assert statements == ['return Builder(A)']


def test_schedule_tree_dunder_desugaring_rewrites_augassign_before_lowering():
    statements = _desugar_statements('obj += A', global_vars={'obj': DunderHost()})

    assert statements == ['obj = obj.__iadd__(A)']


def test_schedule_tree_dunder_desugaring_rewrites_subscript_augassign_before_lowering():
    statements = _desugar_statements('obj[i] += A', global_vars={'obj': DunderHost(), 'i': 0})

    assert statements == ['obj.__setitem__(i, obj.__getitem__(i).__iadd__(A))']


def test_python_frontend_schedule_tree_callable_object_call_is_inlined():

    class CallableObject:

        @dace.method
        def __call__(self, A: dace.float64[8]):
            return A + 1

    callable_object = CallableObject()

    @dace.program
    def outer(A: dace.float64[8]):
        return callable_object(A)

    stree = outer.to_schedule_tree()

    assert isinstance(stree.children[0], tn.FunctionCallScope)
    assert stree.children[0].call.callee_name == '__call__'
    assert stree.children[0].call.arguments == {'A': 'A'}
    assert isinstance(stree.children[1], tn.ReturnNode)


def test_python_frontend_schedule_tree_parseable_free_function_call_is_inlined():

    def callee(A: dace.float64[8]):
        return A + 1

    @dace.program
    def outer(A: dace.float64[8]):
        return callee(A)

    stree = outer.to_schedule_tree()

    assert isinstance(stree.children[0], tn.FunctionCallScope)
    assert stree.children[0].call.callee_name == 'callee'
    assert stree.children[0].call.arguments == {'A': 'A'}
    assert isinstance(stree.children[1], tn.ReturnNode)


def test_python_frontend_schedule_tree_dunder_add_is_inlined():

    class Adder:

        @dace.method
        def __add__(self, A: dace.float64[8]):
            return A + 1

    adder = Adder()

    @dace.program
    def outer(A: dace.float64[8]):
        return adder + A

    stree = outer.to_schedule_tree()

    assert isinstance(stree.children[0], tn.FunctionCallScope)
    assert stree.children[0].call.callee_name == '__add__'
    assert stree.children[0].call.arguments == {'A': 'A'}
    assert isinstance(stree.children[1], tn.ReturnNode)


def test_python_frontend_schedule_tree_dunder_rmatmul_is_inlined():

    class Reflector:

        @dace.method
        def __rmatmul__(self, A: dace.float64[4, 4]):
            return A + 1

    reflector = Reflector()

    @dace.program
    def outer(A: dace.float64[4, 4]):
        return A @ reflector

    stree = outer.to_schedule_tree()

    assert isinstance(stree.children[0], tn.FunctionCallScope)
    assert stree.children[0].call.callee_name == '__rmatmul__'
    assert stree.children[0].call.arguments == {'A': 'A'}
    assert isinstance(stree.children[1], tn.ReturnNode)


def test_python_frontend_schedule_tree_sdfg_call_stays_opaque():

    @dace.program
    def inner(A: dace.float64[8], B: dace.float64[8]):
        return A + B

    sdfg_obj = inner.to_sdfg()

    @dace.program
    def outer(A: dace.float64[8], B: dace.float64[8]):
        return sdfg_obj(A, B)

    stree = outer.to_schedule_tree()

    assert not any(isinstance(node, tn.FunctionCallScope) for node in stree.preorder_traversal())
    assert not any(isinstance(node, tn.PythonCallbackNode) for node in stree.preorder_traversal())
    assert isinstance(stree.children[0], tn.SDFGCallNode)
    assert isinstance(stree.children[0].sdfg, dace.SDFG)
    assert stree.children[0].sdfg.name == sdfg_obj.name
    assert stree.children[0].call.callee_name.endswith('inner')
    assert stree.children[0].call.arguments == {'A': 'A', 'B': 'B'}
    assert stree.children[0].return_targets == ['__stree_retval']
    assert isinstance(stree.children[1], tn.ReturnNode)
    assert stree.children[1].values[0] == '__stree_retval'


def test_python_frontend_schedule_tree_sdfg_convertible_call_stays_opaque():

    class Convertible(SDFGConvertible):

        def __init__(self):
            self.name = 'convertible'

        def __call__(self, *args, **kwargs):
            raise AssertionError('SDFGConvertible should not execute during schedule-tree generation')

        def __sdfg__(self, A, B):

            @dace.program
            def inner(X: dace.float64[8], Y: dace.float64[8]):
                return X + Y

            return inner.to_sdfg(A, B)

        def __sdfg_signature__(self):
            return ['A', 'B'], []

    convertible = Convertible()

    @dace.program
    def outer(A: dace.float64[8], B: dace.float64[8]):
        return convertible(A, B)

    stree = outer.to_schedule_tree()

    assert not any(isinstance(node, tn.FunctionCallScope) for node in stree.preorder_traversal())
    assert not any(isinstance(node, tn.PythonCallbackNode) for node in stree.preorder_traversal())
    assert isinstance(stree.children[0], tn.SDFGCallNode)
    assert isinstance(stree.children[1], tn.ReturnNode)
