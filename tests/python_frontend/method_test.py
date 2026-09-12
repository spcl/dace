# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests dace.program as class methods """
import pytest
import dace
import numpy as np
import time


class MyTestClass:
    """ Test class with various values, lifetimes, and call types. """
    classvalue = 2

    def __init__(self, n=5) -> None:
        self.n = n

    @dace.method
    def method_jit(self, A):
        return A + self.n

    @dace.method
    def method(self, A: dace.float64[20]):
        return A + self.n

    @dace.method
    def __call__(self, A: dace.float64[20]):
        return A * self.n

    @dace.method
    def other_method_caller(self, A: dace.float64[20]):
        return self.method(A) + 2 + self(A)

    @staticmethod
    @dace.program
    def static(A: dace.float64[20]):
        return A + A

    @staticmethod
    @dace.program
    def static_withclass(A: dace.float64[20]):
        return A + MyTestClass.classvalue

    @classmethod
    @dace.method
    def clsmethod(cls, A):
        return A + cls.classvalue


class MyTestCallAttributesClass:

    class SDFGMethodTestClass:

        def __sdfg__(self, *args, **kwargs):

            @dace.program
            def call(A):
                A[:] = 7.0

            return call.__sdfg__(*args)

        def __sdfg_signature__(self):
            return ['A'], []

    def __init__(self, n=5) -> None:
        self.n = n
        self.call_me = MyTestCallAttributesClass.SDFGMethodTestClass()

    @dace.method
    def method_jit(self, A):
        self.call_me(A)
        return A + self.n

    @dace.method
    def __call__(self, A):
        self.call_me(A)
        return A * self.n

    @dace.method
    def method(self, A: dace.float64[20]):
        self.call_me(A)
        return A + self.n

    @dace.method
    def method_jit_with_scalar_arg(self, A, b):
        self.call_me(A)
        return A + b


def test_method_jit():
    A = np.random.rand(20)
    cls = MyTestClass(10)
    assert np.allclose(cls.method_jit(A), A + 10)


def test_method():
    A = np.random.rand(20)
    cls = MyTestClass(10)
    assert np.allclose(cls.method(A), A + 10)


def test_method_cache():
    A = np.random.rand(20)
    cls1 = MyTestClass(10)
    cls2 = MyTestClass(11)
    assert np.allclose(cls1.method(A), A + 10)
    assert np.allclose(cls1.method(A), A + 10)
    assert np.allclose(cls2.method(A), A + 11)


def test_callable():
    A = np.random.rand(20)
    cls = MyTestClass(12)
    assert np.allclose(cls(A), A * 12)


def test_static():
    A = np.random.rand(20)
    assert np.allclose(MyTestClass.static(A), A + A)


def test_static_withclass():
    A = np.random.rand(20)
    # TODO(later): Make cache strict w.r.t. globals and locals used in program
    # assert np.allclose(MyTestClass.static_withclass(A), A + 2)
    # Modify value
    MyTestClass.classvalue = 3
    assert np.allclose(MyTestClass.static_withclass(A), A + 3)


@pytest.mark.skip(reason="Python 3.13 removed chained @classmethods, making this impossible for now")
def test_classmethod():
    A = np.random.rand(20)
    # Modify value first
    MyTestClass.classvalue = 4
    assert np.allclose(MyTestClass.clsmethod(A), A + 4)


def test_nested_methods():
    A = np.random.rand(20)
    cls = MyTestClass()
    assert np.allclose(cls.other_method_caller(A), (A * 5) + (A + 5) + 2)


def mydec(a):

    def mutator(func):
        dp = dace.program(func)

        @dace.program
        def mmm(A: dace.float64[20]):
            res = dp(A, a)
            return res

        sdfg = mmm.to_sdfg()
        return sdfg

    return mutator


def someprog(A: dace.float64[20], a: dace.float64):
    res = A + a
    return res


def someprog_indirection(a):
    return mydec(a)(someprog)


def test_decorator():

    @dace.program(constant_functions=True)
    def otherprog(A: dace.float64[20]):
        res = np.empty_like(A)
        someprog_indirection(3)(A=A, __return=res)
        return res

    sdfg = otherprog.to_sdfg()
    A = np.random.rand(20)
    assert np.allclose(sdfg(A), A + 3)


def test_sdfgattr_method_jit():
    A = np.random.rand(20)
    cls = MyTestCallAttributesClass(10)
    assert np.allclose(cls.method_jit(A), 17)


def test_sdfgattr_callable_jit():
    A = np.random.rand(20)
    cls = MyTestCallAttributesClass(12)
    assert np.allclose(cls(A), 84)


def test_sdfgattr_method_annotated_jit():
    A = np.random.rand(20)
    cls = MyTestCallAttributesClass(14)
    assert np.allclose(cls.method(A), 21)


def test_sdfgattr_method_jit_with_scalar():
    A = np.random.rand(20)
    cls = MyTestCallAttributesClass(10)
    assert np.allclose(cls.method_jit_with_scalar_arg(A, 2.0), 9.0)


def test_nested_field_in_map():

    class B:

        def __init__(self) -> None:
            self.field = np.random.rand(10, 10)

        @dace.method
        def callee(self):
            return self.field[1, 1]

    class A:

        def __init__(self, nested: B):
            self.nested = nested

        @dace.method
        def tester(self):
            val = np.ndarray([2], np.float64)
            for i in dace.map[0:2]:
                val[i] = self.nested.callee()
            return val

    obj = A(B())
    result = obj.tester()

    assert np.allclose(result, np.array([obj.nested.field[1, 1], obj.nested.field[1, 1]]))


def test_nested_callback_in_map():

    class B:

        def __init__(self) -> None:
            self.field = np.random.rand(10, 10)

        @dace.method
        def callee(self, val, i):
            val[i] = time.time()

    class A:

        def __init__(self, nested: B):
            self.nested = nested

        @dace.method
        def tester(self):
            val = np.ndarray([2], np.float64)
            for i in dace.map[0:2]:
                self.nested.callee(val, i)
            return val

    obj = A(B())
    old_time = time.time()

    with pytest.warns(match="Automatically creating callback"):
        result = obj.tester()

    new_time = time.time()

    assert result[0] >= old_time and result[0] <= new_time


def test_unbounded_method():

    @dace.method
    def tester(a):
        return a + 1

    aa = np.random.rand(20)
    assert np.allclose(tester(aa), aa + 1)


class SharedFieldLeg:
    """One of two objects holding the SAME array, as a pyFV3 stencil pair does."""

    def __init__(self, field: np.ndarray) -> None:
        self.field = field

    @dace.method
    def __call__(self, q: dace.float64[16], out: dace.float64[16]):
        out[:] = q * self.field


class SharedFieldPair:

    def __init__(self, field: np.ndarray) -> None:
        self.first = SharedFieldLeg(field)
        self.second = SharedFieldLeg(field)

    @dace.method
    def __call__(self, q: dace.float64[16], out: dace.float64[16]):
        self.first(q, out)
        self.second(out, out)


def test_shared_closure_field_is_one_argument():
    """The same array reached through two objects must not become two closure arguments."""
    field = np.full(16, 2.0)
    pair = SharedFieldPair(field)
    q = np.arange(16, dtype=np.float64)
    out = np.zeros(16, dtype=np.float64)

    sdfg = pair.__call__.to_sdfg(q, out)
    closure_args = [name for name in sdfg.arglist() if name.startswith('__g_')]
    assert closure_args == ['__g_self_field'], closure_args

    pair(q, out)
    assert np.allclose(out, q * 4.0)


def test_view_closure_field_is_refused():
    """A view reached through the closure is refused, not silently copied behind the kernel's back."""
    if dace.Config.get('compiler', 'allow_view_arguments'):
        pytest.skip('view arguments are allowed in this configuration')

    pair = SharedFieldPair(np.full((4, 16), 2.0)[1])
    q = np.arange(16, dtype=np.float64)
    out = np.zeros(16, dtype=np.float64)

    with pytest.raises(TypeError, match='numpy view'):
        pair(q, out)


if __name__ == '__main__':
    test_method_jit()
    test_method()
    test_method_cache()
    test_callable()
    test_static()
    test_static_withclass()
    #test_classmethod()
    test_nested_methods()
    test_decorator()
    test_sdfgattr_method_jit()
    test_sdfgattr_callable_jit()
    test_sdfgattr_method_annotated_jit()
    test_sdfgattr_method_jit_with_scalar()
    test_nested_field_in_map()
    test_nested_callback_in_map()
    test_unbounded_method()
    test_shared_closure_field_is_one_argument()
    test_view_closure_field_is_refused()
