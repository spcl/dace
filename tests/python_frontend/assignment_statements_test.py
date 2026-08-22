# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest

from dace.frontend.python.common import DaceSyntaxError


@dace.program
def single_target(a: dace.float32[1]):
    b = dace.ndarray((1, ), dtype=dace.float32)
    if (a[0] < 0):
        b = 0
    elif (a[0] < 1):
        b = 1
    else:
        b = a
    return b


def test_single_target():
    a = np.zeros((1, ), dtype=np.float32)
    a[0] = np.pi
    b = single_target(a=a)
    assert (b[0] == np.float32(np.pi))


@dace.program
def single_target_parentheses(a: dace.float32[1]):
    (b) = a
    return b


def test_single_target_parentheses():
    a = np.zeros((1, ), dtype=np.float32)
    a[0] = np.pi
    b = single_target_parentheses(a=a)
    assert (b[0] == np.float32(np.pi))


@dace.program
def multiple_targets(a: dace.float32[1]):
    b, c = a, 2 * a
    return b, c


def test_multiple_targets():
    a = np.zeros((1, ), dtype=np.float32)
    a[0] = np.pi
    b, c = multiple_targets(a=a)
    assert (b[0] == np.float32(np.pi))
    assert (c[0] == np.float32(2) * np.float32(np.pi))


@dace.program
def multiple_targets_parentheses(a: dace.float32[1]):
    (b, c) = (a, 2 * a)
    return b, c


def test_multiple_targets_parentheses():
    a = np.zeros((1, ), dtype=np.float32)
    a[0] = np.pi
    b, c = multiple_targets_parentheses(a=a)
    assert (b[0] == np.float32(np.pi))
    assert (c[0] == np.float32(2) * np.float32(np.pi))


@dace.program
def multiple_targets_unpacking(a: dace.float32[2]):
    b, c = a
    return b, c


def test_multiple_targets_unpacking():
    a = np.zeros((2, ), dtype=np.float32)
    a[0] = np.pi
    a[1] = 2 * np.pi
    b, c = multiple_targets_unpacking(a=a)
    assert (b[0] == a[0])
    assert (c[0] == a[1])


@dace.program
def multiple_targets_unpacking_multidim(a: dace.float64[2, 3, 4]):
    b, c = a
    return b, c


def test_multiple_targets_unpacking_multidim():
    a = np.random.rand(2, 3, 4)
    b, c = multiple_targets_unpacking_multidim(a)
    bref, cref = a
    assert np.allclose(b, bref)
    assert np.allclose(c, cref)


@dace.program
def multiple_targets_unpacking_func(a: dace.float64[2, 3, 4]):
    b, c = np.square(a)
    return b, c


def test_multiple_targets_unpacking_func():
    a = np.random.rand(2, 3, 4)
    b, c = multiple_targets_unpacking_func(a)
    bref, cref = np.square(a)
    assert np.allclose(b, bref)
    assert np.allclose(c, cref)


def test_multiple_targets_unpacking_invalid():

    @dace.program
    def tester(a: dace.float64[2, 3, 4]):
        b, c, d = np.square(a)
        return b, c, d

    with pytest.raises(DaceSyntaxError):
        tester.to_sdfg()


@dace.program
def nested_multiple_targets(a: dace.float32[1]):
    b, (c, d), e = a, (2 * a, 3 * a), 4 * a
    return b, c, d, e


def test_nested_multiple_targets():
    a = np.zeros((1, ), dtype=np.float32)
    a[0] = np.pi
    b, c, d, e = nested_multiple_targets(a=a)
    assert (b[0] == np.float32(np.pi))
    assert (c[0] == np.float32(2) * np.float32(np.pi))
    assert (d[0] == np.float32(3) * np.float32(np.pi))
    assert (e[0] == np.float32(4) * np.float32(np.pi))


@dace.program
def starred_target(a: dace.float32[1]):
    b, *c, d, e = a, 2 * a, 3 * a, 4 * a, 5 * a, 6 * a
    return b, c, d, e


@pytest.mark.skip('Syntax is not yet supported')
def test_starred_target():
    a = np.zeros((1, ), dtype=np.float32)
    a[0] = np.pi
    b, c, d, e = starred_target(a=a)
    assert (b[0] == np.float32(np.pi))
    assert (c[0] == np.float32(2) * np.float32(np.pi))
    assert (c[1] == np.float32(3) * np.float32(np.pi))
    assert (c[2] == np.float32(4) * np.float32(np.pi))
    assert (d[0] == np.float32(5) * np.float32(np.pi))
    assert (e[0] == np.float32(6) * np.float32(np.pi))


mystruct = dace.struct('mystruct', a=dace.int32, b=dace.float32)


@dace.program
def attribute_reference(a: mystruct[1]):
    a.a[0] = 5
    a.b[0] = 6


@pytest.mark.skip('Syntax is not yet supported')
def test_attribute_reference():
    a = np.ndarray((1, ), dtype=np.dtype(mystruct.as_ctypes()))
    attribute_reference(a=a)
    assert (a[0]['a'] == np.int32(5))
    assert (a[0]['b'] == np.float32(6))


@dace.program
def ann_assign_supported_type():
    a: dace.uint16 = 5
    return a


def test_ann_assign_supported_type():
    a = ann_assign_supported_type()
    assert (a.dtype == np.uint16)


def test_assignment_to_nonexistent_variable():

    @dace.program
    def badprog(B: dace.float64):
        A[...] = B

    with pytest.raises(DaceSyntaxError):
        badprog.to_sdfg()


def test_assign_return_symbols():

    @dace.program
    def assign_symbols():
        a = 6
        for i in range(10):
            a = 5
        a -= 1
        return i, a

    result = assign_symbols()
    a = result[1][0]
    assert a == 4


def test_assign_to_compiletime_scalar():

    class MyScalar:

        def __init__(self) -> None:
            self.scalar = 5

        @dace.method
        def method(self):
            self.scalar = 1

    obj = MyScalar()
    with pytest.raises(DaceSyntaxError):
        obj.method()


def test_augassign_to_compiletime_scalar():

    class MyScalar:

        def __init__(self) -> None:
            self.scalar = 5

        @dace.method
        def method(self):
            self.scalar += 1

    obj = MyScalar()
    with pytest.raises(DaceSyntaxError):
        obj.method()


def test_scalar_assignment_copies():

    @dace.program
    def plain(out: dace.float64[2]):
        a = 1.0
        b = a
        b += 1.0
        out[0] = a
        out[1] = b

    sdfg = plain.to_sdfg(simplify=False)
    assert 'a' in sdfg.arrays and 'b' in sdfg.arrays
    assert sdfg.arrays['a'] is not sdfg.arrays['b']

    out = np.zeros((2, ), dtype=np.float64)
    plain(out=out)
    assert np.allclose(out, [1.0, 2.0])


def test_scalar_chained_assignment_copies():

    @dace.program
    def chained(a: dace.float64[8], out: dace.float64[1]):
        tmp = 0.0
        s0 = tmp
        s1 = tmp
        for i in range(4):
            s0 += a[2 * i]
            s1 += a[2 * i + 1]
        out[0] = s0 + s1

    sdfg = chained.to_sdfg(simplify=False)
    assert {'tmp', 's0', 's1'}.issubset(sdfg.arrays.keys())

    a = np.arange(1.0, 9.0)
    out = np.zeros((1, ), dtype=np.float64)
    chained(a=a, out=out)
    assert np.allclose(out[0], a.sum())


def test_scalar_assignment_branch_dispatch():

    @dace.program
    def branches(out: dace.float64[6]):
        rp0 = 10.0
        rp1 = 20.0
        rp2 = 30.0
        for idir in range(3):
            if idir == 0:
                pc = rp0
            elif idir == 1:
                pc = rp1
            else:
                pc = rp2
            out[idir] = pc
        out[3] = rp0
        out[4] = rp1
        out[5] = rp2

    sdfg = branches.to_sdfg(simplify=False)
    assert {'rp0', 'rp1', 'rp2', 'pc'}.issubset(sdfg.arrays.keys())

    out = np.zeros((6, ), dtype=np.float64)
    branches(out=out)
    assert np.allclose(out, [10.0, 20.0, 30.0, 10.0, 20.0, 30.0])


def test_array_assignment_aliases():

    @dace.program
    def aliasing(out: dace.float64[4]):
        a = np.zeros((4, ), dtype=np.float64)
        b = a
        b[0] = 1.0
        out[:] = a

    sdfg = aliasing.to_sdfg(simplify=False)
    # Both names share a single container, as they do in NumPy
    assert len([n for n, desc in sdfg.arrays.items() if desc.transient]) == 1

    out = np.ones((4, ), dtype=np.float64)
    aliasing(out=out)
    assert np.allclose(out, [1.0, 0.0, 0.0, 0.0])


if __name__ == "__main__":
    test_single_target()
    test_single_target_parentheses()
    test_multiple_targets()
    test_multiple_targets_parentheses()
    test_multiple_targets_unpacking()
    test_multiple_targets_unpacking_multidim()
    test_multiple_targets_unpacking_func()
    test_multiple_targets_unpacking_invalid()
    test_nested_multiple_targets()

    # test_starred_target()
    # test_attribute_reference()

    test_ann_assign_supported_type()
    test_assignment_to_nonexistent_variable()

    test_assign_return_symbols()

    test_assign_to_compiletime_scalar()
    test_augassign_to_compiletime_scalar()

    test_scalar_assignment_copies()
    test_scalar_chained_assignment_copies()
    test_scalar_assignment_branch_dispatch()
    test_array_assignment_aliases()
