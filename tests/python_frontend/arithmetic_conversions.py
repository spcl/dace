import dace
import numpy as np


@dace.program
def complex_conversion(a: dace.complex128[1], b: int):
    return a[0] + b


def test_complex_conversion():
    # a = 5 + 6j
    a = np.zeros((1,), dtype=np.complex128)
    a[0] = 5 + 6j
    b = 7
    c = complex_conversion(a=a, b=b)
    assert(c[0] == 12 + 6j)


@dace.program
def float_conversion(a: float, b: int):
    return a + b


def test_float_conversion():
    a = 5.2
    b = 7
    c = float_conversion(a=a, b=b)
    assert(c[0] == 12.2)


if __name__ == "__main__":
    test_complex_conversion()
    test_float_conversion()
