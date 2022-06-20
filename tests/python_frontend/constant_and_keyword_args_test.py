# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests constants, optional, and keyword arguments. """
from types import SimpleNamespace
import dace
import numpy as np
import pytest

from dace.frontend.python.common import DaceSyntaxError, SDFGConvertible


def test_kwargs():

    @dace.program
    def kwarg(A: dace.float64[20], kw: dace.float64[20]):
        A[:] = kw + 1

    A = np.random.rand(20)
    kw = np.random.rand(20)
    kwarg(A, kw=kw)
    assert np.allclose(A, kw + 1)


def test_kwargs_jit():

    @dace.program
    def kwarg(A, kw):
        A[:] = kw + 1

    A = np.random.rand(20)
    kw = np.random.rand(20)
    kwarg(A, kw=kw)
    assert np.allclose(A, kw + 1)


def test_kwargs_with_default():

    @dace.program
    def kwarg(A: dace.float64[20], kw: dace.float64[20] = np.ones([20])):
        A[:] = kw + 1

    # Call without argument
    A = np.random.rand(20)
    kwarg(A)
    assert np.allclose(A, 2.0)

    # Call with argument
    kw = np.random.rand(20)
    kwarg(A, kw)
    assert np.allclose(A, kw + 1)


def test_var_args_jit():

    @dace.program
    def arg_jit(*args):
        return args[0] + args[1]

    A = np.random.rand(20)
    B = np.random.rand(20)
    C = arg_jit(A, B)
    assert np.allclose(C, A + B)


def test_var_args_aot():
    # This test is supposed to be unsupported
    with pytest.raises(SyntaxError):

        @dace.program
        def arg_aot(*args: dace.float64[20]):
            return args[0] + args[1]

        arg_aot.compile()


def test_var_args_empty():

    @dace.program
    def arg_aot(*args):
        return np.zeros([20])

    arg_aot.compile()


def test_var_kwargs_jit():

    @dace.program
    def kwarg_jit(**kwargs):
        return kwargs['A'] + kwargs['B']

    A = np.random.rand(20)
    B = np.random.rand(20)
    C = kwarg_jit(A=A, B=B)
    assert np.allclose(C, A + B)


def test_var_kwargs_aot():
    # This test is supposed to be unsupported
    with pytest.raises(SyntaxError):

        @dace.program
        def kwarg_aot(**kwargs: dace.float64[20]):
            return kwargs['A'] + kwargs['B']

        kwarg_aot.compile()


def test_none_arrays():

    @dace.program
    def myprog(A: dace.float64[20], B: dace.float64[20]):
        result = np.zeros([20], dtype=dace.float64)
        if B is None:
            if A is not None:
                result[:] = A
            else:
                result[:] = 1
        else:
            result[:] = B
        return result

    # Tests
    A = np.random.rand(20)
    B = np.random.rand(20)
    assert np.allclose(myprog(A, B), B)
    assert np.allclose(myprog(A, None), A)
    assert np.allclose(myprog(None, None), 1)


def test_none_callables():
    myfunc = None

    @dace.program
    def myprog(A: dace.float64[20]):
        if myfunc:
            return myfunc(A)
        return A

    # Tests
    A = np.random.rand(20)
    assert np.allclose(myprog(A), A)

    def modifier(a):
        return a + 1

    myfunc = modifier
    assert np.allclose(myprog(A), A + 1)


def test_none_callables_2():
    myfunc = None

    @dace.program
    def myprog(A: dace.float64[20]):
        if myfunc is not None:
            return myfunc(A)
        return A

    # Tests
    A = np.random.rand(20)
    assert np.allclose(myprog(A), A)

    def modifier(a):
        return a + 1

    myfunc = modifier
    assert np.allclose(myprog(A), A + 1)


def test_none_convertibles():
    myfunc = None

    @dace.program
    def myprog(A: dace.float64[20]):
        if myfunc is not None:
            return myfunc(A)
        return A

    # Tests
    A = np.random.rand(20)
    assert np.allclose(myprog(A), A)

    @dace.program
    def modifier(a):
        return a + 1

    myfunc = modifier
    assert np.allclose(myprog(A), A + 1)


def test_none_convertibles_2():
    myfunc = None

    class AConvertible(SDFGConvertible):

        def __sdfg__(self):

            @dace.program
            def func():
                arr = np.empty([20], np.float64)
                arr[:] = 7.0
                return arr

            return func.to_sdfg()

        def __sdfg_signature__(self):
            return ([], [])

        def __sdfg_closure__(self, reevaluate=None):
            return {}

    @dace.program
    def myprog(A: dace.float64[20]):
        if myfunc is not None:
            return myfunc()
        return A

    # Tests
    A = np.random.rand(20)
    assert np.allclose(myprog(A), A)

    myfunc = AConvertible()
    assert np.allclose(myprog(A), 7)


def test_none_arrays_jit():

    @dace.program
    def myprog_jit(A, B):
        if B is None:
            if A is not None:
                return A
            else:
                return np.ones([20], np.float64)
        else:
            return B

    # Tests
    A = np.random.rand(20)
    B = np.random.rand(20)
    assert np.allclose(myprog_jit(A, B), B)
    assert np.allclose(myprog_jit(A, None), A)
    assert np.allclose(myprog_jit(None, None), 1)


def test_optional_argument_jit():

    @dace.program
    def linear(x, w, bias):
        """ Linear layer with weights w applied to x, and optional bias. """
        if bias is not None:
            return x @ w.T + bias
        else:
            return x @ w.T

    x = np.random.rand(13, 14)
    w = np.random.rand(10, 14)
    b = np.random.rand(10)

    # Try without bias
    assert np.allclose(linear.f(x, w, None), linear(x, w, None))

    # Try with bias
    assert np.allclose(linear.f(x, w, b), linear(x, w, b))


def test_optional_argument_jit_kwarg():

    @dace.program
    def linear(x, w, bias=None):
        """ Linear layer with weights w applied to x, and optional bias. """
        if bias is not None:
            return np.dot(x, w.T) + bias
        else:
            return np.dot(x, w.T)

    x = np.random.rand(13, 14)
    w = np.random.rand(10, 14)
    b = np.random.rand(10)

    # Try without bias
    assert np.allclose(linear.f(x, w), linear(x, w))

    # Try with bias
    assert np.allclose(linear.f(x, w, b), linear(x, w, b))


def test_optional_argument():

    @dace.program
    def linear(x: dace.float64[13, 14], w: dace.float64[10, 14], bias: dace.float64[10] = None):
        """ Linear layer with weights w applied to x, and optional bias. """
        if bias is not None:
            return np.dot(x, w.T) + bias
        else:
            return np.dot(x, w.T)

    x = np.random.rand(13, 14)
    w = np.random.rand(10, 14)
    b = np.random.rand(10)

    # Try without bias
    assert np.allclose(linear.f(x, w), linear(x, w))

    # Try with bias
    assert np.allclose(linear.f(x, w, b), linear(x, w, b))


def test_constant_argument_simple():

    @dace.program
    def const_prog(cst: dace.constant, B: dace.float64[20]):
        B[:] = cst

    # Test program
    A = np.random.rand(20)
    cst = 4.0
    const_prog(cst, A)
    assert np.allclose(A, cst)

    # Test code for folding
    code = const_prog.to_sdfg(cst).generate_code()[0].clean_code
    assert 'cst' not in code


def test_constant_argument_default():

    @dace.program
    def const_prog(B: dace.float64[20], cst: dace.constant = 7):
        B[:] = cst

    # Test program
    A = np.random.rand(20)
    const_prog(A)
    assert np.allclose(A, 7)

    # Test program
    A = np.random.rand(20)
    const_prog(A, cst=4)
    assert np.allclose(A, 4)

    # Test program
    A = np.random.rand(20)
    const_prog(A, cst=5)
    assert np.allclose(A, 5)

    # Test code for folding
    code = const_prog.to_sdfg().generate_code()[0].clean_code
    assert 'cst' not in code


def test_constant_argument_object():
    """
    Tests nested functions with constant parameters passed in as arguments.
    """

    class MyConfiguration:

        def __init__(self, parameter):
            self.p = parameter * 2
            self.q = parameter * 4

        @property
        def get_random_number(self):
            return 4

    @dace.program
    def nested_func(cfg: dace.constant, A: dace.float64[20]):
        return A[cfg.p]

    @dace.program
    def constant_parameter(cfg: dace.constant, cfg2: dace.constant, A: dace.float64[20]):
        A[cfg.q] = nested_func(cfg, A)
        A[cfg.get_random_number] = nested_func(cfg2, A)

    cfg1 = MyConfiguration(3)
    cfg2 = MyConfiguration(4)
    A = np.random.rand(20)
    reg_A = np.copy(A)
    reg_A[12] = reg_A[6]
    reg_A[4] = reg_A[8]

    constant_parameter(cfg1, cfg2, A)
    assert np.allclose(A, reg_A)


def test_none_field():

    class ClassA:

        def __init__(self, field_or_none):
            self.field_or_none = field_or_none

        @dace.method
        def method(self, A):
            if (self.field_or_none is None) and (self.field_or_none is None):
                A[...] = 7.0
            if (self.field_or_none is not None) and (self.field_or_none is not None):
                A[...] += self.field_or_none

    A = np.ones((10, ))
    obja = ClassA(None)
    obja.method(A)
    assert np.allclose(A, 7.0)
    A = np.ones((10, ))
    obja = ClassA(np.ones((10, )))
    obja.method(A)
    assert np.allclose(A, 2.0)


def test_array_by_str_key():

    class AClass:

        def __init__(self):
            self.adict = dict(akey=7.0 * np.ones((10, )))

        @dace.method
        def __call__(self, A):
            A[...] = self.adict['akey']

    aobj = AClass()
    arr = np.empty((10, ))
    aobj(arr)
    assert np.allclose(7.0, arr)


def test_constant_folding():

    @dace.program
    def tofold(A: dace.float64[20], add: dace.constant):
        if add:
            A += 1
        else:
            A -= 1

    A = np.random.rand(20)
    expected = A + 1
    tofold(A, True)

    assert np.allclose(A, expected)


def test_boolglobal():
    some_glob = 124

    @dace.program
    def func(A):
        boolvar = 123 == some_glob
        if boolvar:
            tmp = 0
        else:
            tmp = 1
        A[...] = tmp

    a = np.empty((10, ))
    func(a)
    assert np.allclose(a, 1)


def test_intglobal():
    some_glob = 124

    @dace.program
    def func(A):
        var = some_glob
        tmp = 1
        for it in range(100):
            if 123 == it or (it == var - 1):
                tmp = 0
        A[...] = tmp

    func(np.empty((10, )))


def test_numpynumber_condition():

    @dace.program
    def conditional_val(A: dace.float64[20], val: dace.constant):
        if (val % 4) == 0:
            A[:] = 0
        else:
            A[:] = 1

    # Ensure condition was folded
    sdfg = conditional_val.to_sdfg(val=np.int64(3), simplify=True)
    assert sdfg.number_of_nodes() == 1

    a = np.random.rand(20)
    conditional_val(a, np.int64(3))
    assert np.allclose(a, 1)
    conditional_val(a, np.int64(4))
    assert np.allclose(a, 0)


def test_constant_list_number():
    something = [1, 2, 3]
    n = len(something)

    @dace.program
    def sometest(A):
        for i in dace.unroll(range(n)):
            A += something[i]

    A = np.random.rand(20)
    sometest.to_sdfg(A)


def test_constant_list_function():

    def a(A):
        A += 1

    def b(A):
        A += 2

    def c(A):
        A += 3

    something = [a, b, c]
    n = len(something)

    @dace.program
    def sometest(A):
        for i in dace.unroll(range(n)):
            something[i](A)

    A = np.random.rand(20)
    sometest.to_sdfg(A)


def test_constant_propagation():

    @dace.program
    def conditional_val(A: dace.float64[20], val: dace.constant):
        cval = val % 4
        if cval == 0:
            A[:] = 0
        else:
            A[:] = 1

    # Ensure condition was folded
    sdfg = conditional_val.to_sdfg(val=3, simplify=True)
    from dace.transformation.interstate.state_elimination import DeadStateElimination, ConstantPropagation
    sdfg.apply_transformations_repeated([ConstantPropagation, DeadStateElimination])
    sdfg.simplify()
    assert sdfg.number_of_nodes() == 1

    a = np.random.rand(20)
    conditional_val(a, 3)
    assert np.allclose(a, 1)
    conditional_val(a, 4)
    assert np.allclose(a, 0)


def test_constant_propagation_pass():
    from dace.transformation.passes import constant_propagation as cprop, dead_state_elimination as dse

    @dace.program
    def conditional_val(A: dace.float64[20], val: dace.constant):
        cval = val % 4
        if cval == 0:
            A[:] = 0
        else:
            A[:] = 1

    # Ensure condition was folded
    sdfg_3 = conditional_val.to_sdfg(val=3, simplify=True)
    cprop.ConstantPropagation().apply_pass(sdfg_3, {})
    dse.DeadStateElimination().apply_pass(sdfg_3, {})
    sdfg_3.simplify()
    assert sdfg_3.number_of_nodes() == 1

    # Ensure condition was folded
    sdfg_4 = conditional_val.to_sdfg(val=4, simplify=True)
    cprop.ConstantPropagation().apply_pass(sdfg_4, {})
    dse.DeadStateElimination().apply_pass(sdfg_4, {})
    sdfg_4.simplify()
    assert sdfg_4.number_of_nodes() == 1

    a = np.random.rand(20)
    sdfg_3(a)
    assert np.allclose(a, 1)
    sdfg_4(a)
    assert np.allclose(a, 0)


def test_constant_propagation_2():

    @dace.program
    def conditional_val(A: dace.float64[20], val: dace.int64):
        if val:
            A[:] = 0
        else:
            A[:] = 1

    # Ensure condition was folded
    a = np.random.rand(20)
    conditional_val(a, 1)
    assert np.allclose(a, 0)
    conditional_val(a, 0)
    assert np.allclose(a, 1)


def test_constant_proper_use():

    @dace.program
    def good_function(scal: dace.constant, scal2: dace.constant, arr):
        a_bool = scal == 1
        if a_bool:
            arr[:] = arr[:] + scal2

    @dace.program
    def program(arr, scal: dace.constant):
        arr[:] = arr[:] * scal
        good_function(scal, 3.0, arr)

    arr = np.ones((12), np.float64)
    scal = 2

    program(arr, scal)
    assert np.allclose(arr, 2)


def test_constant_proper_use_2():
    """ Stress test constants with strings. """

    @dace.program
    def good_function(cfg: dace.constant, cfg2: dace.constant, arr):
        print(cfg)
        print(cfg2)

    @dace.program
    def program(arr, cfg: dace.constant):
        arr[:] = arr[:] * scal
        good_function(cfg, 'cfg2', arr)

    arr = np.ones((12), np.float64)
    scal = 2

    program(arr, 'cfg')
    assert np.allclose(arr, 2)


def test_constant_misuse():

    @dace.program
    def bad_function(scal: dace.constant, arr):
        a_bool = scal == 1
        if a_bool:
            arr[:] = arr[:] + 1

    @dace.program
    def program(arr, scal):
        arr[:] = arr[:] * scal
        bad_function(scal, arr)

    arr = np.ones((12), np.float64)
    scal = 2

    with pytest.raises(DaceSyntaxError):
        program(arr, scal)


def test_constant_field():

    def function(ctx: dace.constant, arr, somebool):
        a_bool = ctx.scal == 1
        if a_bool and somebool:
            arr[:] = arr[:] + 1

    @dace.program
    def program(arr, ctx: dace.constant):
        function(ctx, arr, ctx.scal == 1)

    ns = SimpleNamespace(scal=2)
    arr = np.ones((12), np.float64)

    program(arr, ns)


if __name__ == '__main__':
    test_kwargs()
    test_kwargs_jit()
    test_kwargs_with_default()
    test_var_args_jit()
    test_var_args_aot()
    test_var_args_empty()
    test_var_kwargs_jit()
    test_var_kwargs_aot()
    test_none_arrays()
    test_none_arrays_jit()
    test_none_callables()
    test_none_callables_2()
    test_none_convertibles()
    test_none_convertibles_2()
    test_optional_argument_jit()
    test_optional_argument_jit_kwarg()
    test_optional_argument()
    test_constant_argument_simple()
    test_constant_argument_default()
    test_constant_argument_object()
    test_none_field()
    test_array_by_str_key()
    test_constant_folding()
    test_boolglobal()
    test_intglobal()
    test_numpynumber_condition()
    test_constant_list_number()
    test_constant_list_function()
    test_constant_propagation()
    test_constant_propagation_pass()
    test_constant_propagation_2()
    test_constant_proper_use()
    test_constant_proper_use_2()
    test_constant_misuse()
    test_constant_field()
