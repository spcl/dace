# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests automatic detection and baking of callbacks in the Python frontend. """
from typing import Dict, Union
import dace
import numpy as np
import pytest
import time
from dace import config

N = dace.symbol('N')


def dace_inhibitor(f):
    return f


@dace_inhibitor
def almost_gemm(A, alpha, B):
    return alpha * A @ B


@dace_inhibitor
def almost_gemm_2(A, alpha, B):
    return alpha * A @ B, alpha * np.float64(2)


@dace_inhibitor
def scale(C, beta):
    C *= beta


def test_automatic_callback():

    @dace.program
    def autocallback(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N], beta: dace.float64):
        tmp: dace.float64[N, N] = almost_gemm(A, 0.5, B)
        scale(C, beta)
        C += tmp

    A = np.random.rand(24, 24)
    B = np.random.rand(24, 24)
    C = np.random.rand(24, 24)
    beta = np.float64(np.random.rand())
    expected = 0.5 * A @ B + beta * C

    autocallback(A, B, C, beta)

    assert np.allclose(C, expected)


def test_automatic_callback_2():

    @dace.program
    def autocallback(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N], beta: dace.float64):
        tmp: dace.float64[N, N]
        tmp2: dace.float64
        tmp, tmp2 = almost_gemm_2(A, 0.5, B)
        scale(C, beta)
        C += tmp * tmp2

    A = np.random.rand(24, 24)
    B = np.random.rand(24, 24)
    C = np.random.rand(24, 24)
    beta = np.float64(np.random.rand())
    expected = 0.5 * A @ B * 0.5 * 2 + beta * C

    autocallback(A, B, C, beta)

    assert np.allclose(C, expected)


def test_automatic_callback_inference():

    @dace.program
    def autocallback_ret(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N], beta: dace.float64):
        tmp = np.ndarray([N, N], dace.float64)
        tmp[:] = almost_gemm(A, 0.5, B)
        scale(C, beta)
        C += tmp

    A = np.random.rand(24, 24)
    B = np.random.rand(24, 24)
    C = np.random.rand(24, 24)
    beta = np.float64(np.random.rand())
    expected = 0.5 * A @ B + beta * C

    autocallback_ret(A, B, C, beta)

    assert np.allclose(C, expected)


def test_automatic_callback_inference_2():

    @dace.program
    def autocallback_ret(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N], beta: dace.float64):
        tmp = np.ndarray([N, N], dace.float64)
        tmp2 = np.float64(0.0)
        tmp[:], tmp2 = almost_gemm_2(A, 0.5, B)
        scale(C, beta)
        C += tmp * tmp2

    A = np.random.rand(24, 24)
    B = np.random.rand(24, 24)
    C = np.random.rand(24, 24)
    beta = np.float64(np.random.rand())
    expected = 0.5 * A @ B * 0.5 * 2 + beta * C

    autocallback_ret(A, B, C, beta)

    assert np.allclose(C, expected)


def test_automatic_callback_method():

    class NotDace:

        def __init__(self):
            self.q = np.random.rand()

        @dace_inhibitor
        def method(self, a):
            return a * self.q

    nd = NotDace()

    @dace.program
    def autocallback_method(A: dace.float64[N, N]):
        tmp: dace.float64[N, N] = nd.method(A)
        return tmp

    A = np.random.rand(24, 24)

    out = autocallback_method(A)

    assert np.allclose(out, nd.q * A)


@dace.program
def modcallback(A: dace.float64[N, N], B: dace.float64[N]):
    tmp: dace.float64[N] = np.median(A, axis=1)
    B[:] = tmp


def test_callback_from_module():
    N.set(24)
    A = np.random.rand(24, 24)
    B = np.random.rand(24)
    modcallback(A, B)
    diff = np.linalg.norm(B - np.median(A, axis=1))
    print('Difference:', diff)
    assert diff <= 1e-5


def sq(a):
    return a * a


@dace.program
def tasklet_callback(A: dace.float64[N, N], B: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        with dace.tasklet:
            a << A[i, j]
            b >> B[i, j]
            b = sq(a)


@pytest.mark.skip
def test_callback_tasklet():
    A = np.random.rand(24, 24)
    B = np.random.rand(24, 24)
    tasklet_callback(A, B)
    assert np.allclose(A * A, B)


def test_view_callback():

    @dace.program
    def autocallback(A: dace.float64[2 * N, N], B: dace.float64[N, N], C: dace.float64[N, N], beta: dace.float64):
        A[N:, :] = almost_gemm(A[:N, :], 0.5, B)
        scale(C, beta)
        C += A[N:, :]

    A = np.random.rand(48, 24)
    B = np.random.rand(24, 24)
    C = np.random.rand(24, 24)
    beta = np.float64(np.random.rand())
    expected = 0.5 * A[:24] @ B + beta * C

    autocallback(A, B, C, beta)

    assert np.allclose(C, expected)


def test_print():

    @dace.program
    def printprog(a: dace.float64[2, 2]):
        print(a, 'hello')

    a = np.random.rand(2, 2)
    printprog(a)


def test_reorder():
    counter = 0
    should_be_one, should_be_two = 0, 0

    @dace_inhibitor
    def a():
        nonlocal counter
        nonlocal should_be_two
        counter += 1
        should_be_two = counter

    @dace_inhibitor
    def b():
        nonlocal counter
        nonlocal should_be_one
        counter += 1
        should_be_one = counter

    @dace.program
    def do_not_reorder():
        b()
        a()

    sdfg = do_not_reorder.to_sdfg()
    assert list(sdfg.arrays.keys()) == ['__pystate']

    do_not_reorder()
    assert should_be_one == 1
    assert should_be_two == 2


def test_reorder_nested():
    counter = 0
    should_be_one, should_be_two = 0, 0

    @dace_inhibitor
    def a():
        nonlocal counter
        nonlocal should_be_two
        counter += 1
        should_be_two = counter

    def call_a():
        a()

    @dace_inhibitor
    def b():
        nonlocal counter
        nonlocal should_be_one
        counter += 1
        should_be_one = counter

    @dace.program
    def call_b():
        b()

    @dace.program
    def do_not_reorder_nested():
        call_b()
        call_a()

    sdfg = do_not_reorder_nested.to_sdfg()
    assert list(sdfg.arrays.keys()) == ['__pystate']

    do_not_reorder_nested()
    assert should_be_one == 1
    assert should_be_two == 2


def test_callback_samename():
    counter = 0
    should_be_one, should_be_two = 0, 0

    def get_func_a():

        @dace_inhibitor
        def b():
            nonlocal counter
            nonlocal should_be_two
            counter += 1
            should_be_two = counter

        def call_a():
            b()

        return call_a

    def get_func_b():

        @dace_inhibitor
        def b():
            nonlocal counter
            nonlocal should_be_one
            counter += 1
            should_be_one = counter

        @dace.program
        def call_b():
            b()

        return call_b

    call_a = get_func_a()
    call_b = get_func_b()

    @dace.program
    def same_name_nested():
        call_b()
        call_a()

    sdfg = same_name_nested.to_sdfg(simplify=False)
    assert list(sdfg.arrays.keys()) == ['__pystate']

    same_name_nested()
    assert should_be_one == 1
    assert should_be_two == 2


# Cannot run test without cupy
@pytest.mark.skip
def test_gpu_callback():
    import cupy as cp

    @dace_inhibitor
    def cb_with_gpu(arr):
        assert isinstance(arr, cp.ndarray)
        arr *= 2

    @dace.program
    def gpucallback(A):
        tmp = dace.ndarray([20], dace.float64, storage=dace.StorageType.GPU_Global)
        tmp[:] = A
        cb_with_gpu(tmp)
        A[:] = tmp

    a = cp.random.rand(20)
    expected = a * 2
    gpucallback(a)

    assert cp.allclose(a, expected)


def test_bad_closure():
    """ 
    Testing functions that should not be in the closure (must be implemented as
    callbacks).
    """

    @dace.program
    def timeprog(A: dace.float64[20]):
        # Library function that does not return the same value every time
        A[:] = time.time()

    A = np.random.rand(20)
    B = np.random.rand(20)
    now = time.time()
    timeprog(A)
    timeprog(B)

    assert np.all(B > A) and np.all(A > now)


def test_object_with_nested_callback():
    c = np.random.rand(20)

    @dace_inhibitor
    def call_another_function(a, b):
        nonlocal c
        c[:] = a + b

    class MyObject:

        def __call__(self, a, b):
            c = dict(a=a, b=b)
            call_another_function(**c)

    obj = MyObject()

    @dace.program
    def callobj(a, b):
        obj(a, b)

    a = np.random.rand(20)
    b = np.random.rand(20)
    callobj(a, b)
    assert np.allclose(c, a + b)


def test_two_parameters_same_name():

    @dace_inhibitor
    def add(a, b):
        return a + b

    @dace.program
    def calladd(A: dace.float64[20], B: dace.float64[20]):
        B[:] = add(A, A)

    a = np.random.rand(20)
    b = np.random.rand(20)
    calladd(a, b)
    assert np.allclose(b, a + a)


def test_inout_same_name():

    @dace_inhibitor
    def add(a, b):
        return a + b

    @dace.program
    def calladd(A: dace.float64[20]):
        A[:] = add(A, A)

    a = np.random.rand(20)
    expected = a + a
    calladd(a)
    assert np.allclose(expected, a)


def test_inhibit_state_fusion():
    """ Tests that state fusion is inhibited around callbacks if configured as such. """

    @dace_inhibitor
    def add(a, b):
        return a + b

    @dace.program
    def calladd(A: dace.float64[20], B: dace.float64[20], C: dace.float64[20], D: dace.float64[20]):
        A[:] = add(B, C)
        D[:] = add(A, C)

    with config.set_temporary('frontend', 'dont_fuse_callbacks', value=True):
        sdfg = calladd.to_sdfg(simplify=True)
        assert sdfg.number_of_nodes() == 5

    with config.set_temporary('frontend', 'dont_fuse_callbacks', value=False):
        sdfg = calladd.to_sdfg(simplify=True)
        assert sdfg.number_of_nodes() == 1


def test_two_callbacks():
    called_cnt = 0

    @dace_inhibitor
    def call(arr):
        nonlocal called_cnt
        called_cnt += 1

    @dace.program
    def call_twice(arr, scal):
        call(arr)
        arr[:] = arr[:] * scal
        call(arr)

    arr = np.ones((12, ), np.float64)
    scal = 2

    call_twice(arr, scal)
    assert called_cnt == 2


def test_two_callbacks_different_sig():
    called_cnt = 0

    @dace_inhibitor
    def call(*args):
        nonlocal called_cnt
        called_cnt += 1

    @dace.program
    def call_twice(arr, scal):
        call()
        arr[:] = arr[:] * scal
        call(arr)

    @dace.program
    def call_twice_2(arr, scal):
        call_twice(arr, scal)

    arr = np.ones((12, ), np.float64)
    scal = 2

    call_twice_2(arr, scal)
    assert called_cnt == 2


def test_two_callbacks_different_type():
    called_cnt = 0

    @dace_inhibitor
    def call(array):
        nonlocal called_cnt
        called_cnt += 1

    @dace.program
    def call_twice(arr: dace.float64[20], arr2: dace.int32[20, 20]):
        call(arr)
        arr *= arr2[1]
        call(arr2)

    @dace.program
    def call_twice_3(arr: dace.float64[20], arr2: dace.int32[20, 20]):
        call_twice(arr, arr2)

    arr = np.ones((20, ), np.float64)
    arr2 = np.full((20, 20), 2, np.int32)

    call_twice_3(arr, arr2)
    assert called_cnt == 2


def test_disallowed_keyword():

    class Obj:

        def hello(a):
            try:
                return a + 1
            except:
                return a + 2

    @dace
    def prog(a: dace.float64[10]):
        b: dace.float64[10] = Obj.hello(a)
        return b

    a = np.random.rand(10)
    assert np.allclose(prog(a), a + 1)


def test_nested_duplicate_callbacks():
    called = 0

    @dace_inhibitor
    def callback(*args):
        nonlocal called
        called += len(args)

    @dace.program
    def myprogram_1(a):
        callback(a[0])
        for i in range(a.shape[0]):
            a[i] += i
        callback(a[0], a[1])
        return np.sum(a)

    @dace.program
    def myprogram(a):
        myprogram_1(a)

    a = np.random.rand(20, 1)
    sdfg = myprogram.to_sdfg(a)
    build_folder = sdfg.build_folder

    myprogram(a)
    # Ensure the cache is clear
    myprogram._cache.clear()

    myprogram.load_precompiled_sdfg(build_folder, a)
    assert len(myprogram._cache.cache) == 1

    myprogram(a)
    assert called == 6


def test_scalar_retval():

    @dace.program
    def myprogram(a):
        res: float = time.time()
        a[0] = res

    old_time = time.time()
    result = np.random.rand(20)
    myprogram(result)
    new_time = time.time()
    assert result[0] >= old_time and result[0] <= new_time


def test_callback_kwargs():
    called_with = (None, None, None)
    called_2_with = (None, None, None)
    called_3_with = None

    @dace_inhibitor
    def mycb(a, b=1, **kwargs):
        nonlocal called_with
        called_with = (a, b, kwargs['c'])

    @dace_inhibitor
    def mycb2(d, **kwargs):
        nonlocal called_2_with
        called_2_with = (d, kwargs['e'], kwargs['f'])

    @dace_inhibitor
    def mycb3(**kwargs):
        nonlocal called_3_with
        called_3_with = kwargs['ghi']

    # Call three callbacks with similar types to ensure trampolines are unique
    @dace
    def myprogram():
        mycb(a=1, b=2, c=3)
        mycb2(4, f=5, e=6)
        mycb3(ghi=7)

    myprogram()

    assert called_with == (1, 2, 3)
    assert called_2_with == (4, 6, 5)
    assert called_3_with == 7


def test_same_callback_kwargs():
    """ Calls the same callback twice, with different kwargs each time. """
    called_with = (None, None, None)
    called_2_with = (None, None, None)

    @dace_inhibitor
    def mycb(**kwargs):
        nonlocal called_with
        nonlocal called_2_with
        if 'a' in kwargs:
            called_with = (kwargs['a'], kwargs['b'], kwargs['c'])
        else:
            called_2_with = (kwargs['d'], kwargs['e'], kwargs['f'])

    @dace
    def myprogram():
        mycb(a=1, b=2, c=3)
        mycb(d=4, f=5, e=6)

    myprogram()

    assert called_with == (1, 2, 3)
    assert called_2_with == (4, 6, 5)


def test_builtin_callback_kwargs():

    @dace
    def callprint():
        print('hi', end=',\n')

    callprint()


@pytest.mark.parametrize('as_kwarg', (False, True))
def test_callback_literal_list(as_kwarg):
    success = False

    @dace_inhibitor
    def callback(array, arrlist):
        nonlocal success
        if len(arrlist) == 2 and array[0, 0, 0] == arrlist[0][0, 0, 0]:
            success = True

    if as_kwarg:

        @dace
        def caller(a, b):
            callback(arrlist=[a, b], array=a)
    else:

        @dace
        def caller(a, b):
            callback(a, [a, b])

    a = np.zeros((2, 2, 2))
    b = np.ones((2, 2, 2))
    caller(a, b)
    assert success is True


@pytest.mark.parametrize('as_kwarg', (False, True))
def test_callback_literal_dict(as_kwarg):
    success = False

    @dace_inhibitor
    def callback(adict1, adict2):
        nonlocal success
        if len(adict1) == 3 and len(adict2) == 3:
            if adict1['b'][0, 0, 0] == 0.0 and adict1['a'][0, 0, 0] == 1.0 and adict1[1][0, 0, 0] == 0.0:
                if adict2['b'][0, 0, 0] == 1.0 and adict2['a'][0, 0, 0] == 1.0 and adict2[1][0, 0, 0] == 1.0:
                    success = True

    if as_kwarg:

        @dace
        def caller(a, b):
            callback({'b': a, 'a': b, 1: a}, adict2={1: b, 'a': b, 'b': b})

    else:

        @dace
        def caller(a, b):
            callback({'b': a, 'a': b, 1: a}, {1: b, 'a': b, 'b': b})

    a = np.zeros((2, 2, 2))
    b = np.ones((2, 2, 2))
    caller(a, b)
    assert success is True


def test_unused_callback():

    def deg_to_rad(a):
        res = np.zeros((2, ))

        res[0] = a[0] * np.pi / 180.0
        res[1] = a[1] * np.pi / 180.0
        return res

    @dace.program
    def mid_rad(a: dace.float64[2], b: dace.float64[2]) -> dace.float64[2]:
        mid_deg = (a + b) / 2.0
        mid_rad = deg_to_rad(mid_deg)
        return mid_rad

    @dace.program
    def test(point1: dace.float64[2], point2: dace.float64[2]):
        return mid_rad(point1, point2)

    a = np.array([30.0, 60.0])
    b = np.array([40.0, 50.0])
    expected = np.deg2rad((a + b) / 2.0)
    result = test(a, b)
    assert np.allclose(result, expected)


def test_callback_with_nested_calls():
    success = False

    @dace_inhibitor
    def callback(array):
        nonlocal success
        if array == 20.0:
            success = True

    @dace
    def tester(A: dace.float64[20]):
        callback(np.sum(A))

    tester(np.ones([20]))

    assert success is True


def test_string_callback():
    result = (None, None)

    @dace_inhibitor
    def cb(aa, bb):
        nonlocal result
        result = aa, bb

    @dace.program
    def printmystring(a: str):
        cb('hello', a)

    printmystring('world')
    assert result == ('hello', 'world')


def test_unknown_pyobject():
    counter = 1334
    last_seen = counter
    success_counter = 0

    class MyCustomObject:

        def __init__(self) -> None:
            nonlocal counter
            self.q = counter
            counter += 1

        def __str__(self):
            return f'MyCustomObject(q={self.q})'

    @dace_inhibitor
    def checkit(obj: Union[MyCustomObject, Dict[str, Union[int, str]]]):
        nonlocal last_seen
        nonlocal success_counter
        if obj == {'a': 1, 'b': '2'}:
            success_counter += 1
        elif isinstance(obj, MyCustomObject) and obj.q == last_seen:
            success_counter += 1
            last_seen += 1

    @dace
    def tester(unused):
        for _ in dace.unroll(range(10)):
            a = dict(a=1, b='2')
            b = MyCustomObject()
            checkit(a)
            checkit(b)

    tester(np.random.rand(20))
    assert success_counter == 20


def test_pyobject_return():
    counter = 1

    class MyCustomObject:

        @dace_inhibitor
        def __init__(self) -> None:
            nonlocal counter
            self.q = counter
            counter += 1

        def __str__(self):
            return f'MyCustomObject(q={self.q})'

    @dace
    def tester():
        MyCustomObject()
        return MyCustomObject()

    obj = tester()
    assert isinstance(obj, MyCustomObject)
    assert obj.q == 2


def test_pyobject_return_tuple():
    counter = 1

    class MyCustomObject:

        @dace_inhibitor
        def __init__(self) -> None:
            nonlocal counter
            self.q = counter
            counter += 1

        def __str__(self):
            return f'MyCustomObject(q={self.q})'

    @dace
    def tester():
        MyCustomObject()
        return MyCustomObject(), MyCustomObject()

    obj, obj2 = tester()
    assert isinstance(obj, MyCustomObject)
    assert obj.q == 2
    assert isinstance(obj2, MyCustomObject)
    assert obj2.q == 3


def test_custom_generator():

    def reverse_range(sz):
        cur = sz
        for _ in range(sz):
            yield cur
            cur -= 1

    @dace
    def tester(a: dace.float64[20]):
        gen = reverse_range(20)
        for i in range(20):
            val: int = next(gen)
            a[i] = val

    aa = np.ones((20, ), np.float64)
    tester(aa)
    assert np.allclose(aa, np.arange(20, 0, -1))


def test_custom_generator_with_break():

    def reverse_range(sz):
        cur = sz
        for _ in range(sz):
            yield cur
            cur -= 1

    @dace_inhibitor
    def my_next(generator):
        try:
            return next(generator), False
        except StopIteration:
            return None, True

    @dace
    def tester(a: dace.float64[20]):
        gen = reverse_range(20)
        for i in range(21):
            val: int = 0
            stop: bool = True
            val, stop = my_next(gen)
            if stop:
                break
            a[i] = val

    aa = np.ones((21, ), np.float64)
    expected = np.copy(aa)
    expected[:20] = np.arange(20, 0, -1)

    tester(aa)
    assert np.allclose(aa, expected)


@pytest.mark.skip
def test_matplotlib_with_compute():
    """
    Stacked bar plot example from Matplotlib using callbacks and pyobjects.
    https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/bar_stacked.html#sphx-glr-gallery-lines-bars-and-markers-bar-stacked-py
    """
    import matplotlib.pyplot as plt

    menMeans = (20, 35, 30, 35, 27)
    womenMeans = (25, 32, 34, 20, 25)
    menStd = (2, 3, 4, 1, 2)
    womenStd = (3, 5, 2, 3, 3)

    @dace
    def tester():

        ind = np.arange(5)  # the x locations for the groups
        width = 0.35  # the width of the bars: can also be len(x) sequence

        p1 = plt.bar(ind, menMeans, width, yerr=menStd)
        p2 = plt.bar(ind, womenMeans, width, bottom=menMeans, yerr=womenStd)

        plt.ylabel('Scores')
        plt.title('Scores by group and gender')
        plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
        plt.yticks(np.arange(0, 81, 10))
        plt.legend((p1[0], p2[0]), ('Men', 'Women'))
        plt.show()

    tester()


if __name__ == '__main__':
    test_automatic_callback()
    test_automatic_callback_2()
    test_automatic_callback_inference()
    test_automatic_callback_inference_2()
    test_automatic_callback_method()
    test_callback_from_module()
    test_view_callback()
    # test_callback_tasklet()
    test_print()
    test_reorder()
    test_reorder_nested()
    test_callback_samename()
    # test_gpu_callback()
    test_bad_closure()
    test_object_with_nested_callback()
    test_two_parameters_same_name()
    test_inout_same_name()
    test_inhibit_state_fusion()
    test_two_callbacks()
    test_two_callbacks_different_sig()
    test_two_callbacks_different_type()
    test_disallowed_keyword()
    test_nested_duplicate_callbacks()
    test_scalar_retval()
    test_callback_kwargs()
    test_same_callback_kwargs()
    test_builtin_callback_kwargs()
    test_callback_literal_list(False)
    test_callback_literal_list(True)
    test_callback_literal_dict(False)
    test_callback_literal_dict(True)
    test_unused_callback()
    test_callback_with_nested_calls()
    test_string_callback()
    test_unknown_pyobject()
    test_pyobject_return()
    test_pyobject_return_tuple()
    test_custom_generator()
    test_custom_generator_with_break()
    # test_matplotlib_with_compute()
