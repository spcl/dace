# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

N = 12


@dace.program
def cr_complex(input, output):

    @dace.map(_[0:N])
    def tasklet(i):
        a << input[i]
        b >> output(1, lambda a, b: a + b)
        b = a


def test_cr_complex():
    print('CR non-atomic (complex value) test')

    A = np.random.rand(N).astype(np.complex128)
    A += np.random.rand(N).astype(np.complex128) * 1j
    B = np.ndarray([1], dtype=A.dtype)
    B[0] = 0

    cr_complex(A, B)

    diff = abs(np.sum(A) - B[0])
    print("Difference:", diff)
    assert diff <= 1e-5


if __name__ == '__main__':
    test_cr_complex()


def test_abs_of_a_complex_value_is_real():
    """``Abs`` is generated for ``abs`` in a tasklet. Its result must be real: pinning the return
    type to the argument turns it back into a complex, and then comparing it has no candidate --
    the generated kernel does not compile. Magnitudes here are exact in binary floating point."""

    @dace.program
    def magnitude(inp: dace.complex128[4], out: dace.float64[4]):
        for i in dace.map[0:4]:
            with dace.tasklet:
                a << inp[i]
                m >> out[i]
                m = abs(a)

    A = np.array([3 + 4j, 5 + 12j, 8 + 6j, 0 + 1j], dtype=np.complex128)
    B = np.zeros(4, dtype=np.float64)
    magnitude(A, B)
    assert np.allclose(B, [5.0, 13.0, 10.0, 1.0]), B


def test_a_complex_magnitude_is_comparable():
    """The failure mode itself: ``abs(z) < t`` on a complex ``z``. With a complex-typed ``Abs`` the
    comparison finds no ``operator<`` and code generation fails at the C++ compile."""

    @dace.program
    def inside(inp: dace.complex128[4], out: dace.int32[4]):
        for i in dace.map[0:4]:
            with dace.tasklet:
                a << inp[i]
                r >> out[i]
                r = 1 if abs(a) < 10.0 else 0

    A = np.array([3 + 4j, 5 + 12j, 8 + 6j, 0 + 1j], dtype=np.complex128)
    B = np.zeros(4, dtype=np.int32)
    inside(A, B)
    assert list(B) == [1, 0, 0, 1], B
