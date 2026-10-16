# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import pytest

import dace


def test_integer_power():

    @dace.program
    def powint(A: dace.float64[20], B: dace.float64[20]):
        for i in dace.map[0:20]:
            with dace.tasklet:
                a << A[i]
                b >> B[i]
                c >> A[i]
                b = a**3
                c = a**3.0

    sdfg = powint.to_sdfg()

    assert ":pow(" not in sdfg.generate_code()[0].clean_code


def test_integer_power_constant():

    @dace.program
    def powint(A: dace.float64[20]):
        for i in dace.map[0:20]:
            with dace.tasklet:
                a << A[i]
                b >> A[i]
                b = a**myconst

    sdfg = powint.to_sdfg()
    sdfg.add_constant("myconst", dace.float32(2.0))

    assert ":pow(" not in sdfg.generate_code()[0].clean_code


def test_equality():

    @dace.program
    def nested(a, b, c):
        pass

    @dace.program
    def program(a: dace.float64[10], b: dace.float64[10]):
        for c in range(2):
            nested(a, b, (c == 1))

    program.to_sdfg(simplify=False).compile()


def test_pow_with_implicit_casting():

    @dace.program
    def f32_pow_failure(array):
        return array**3.3

    rng = np.random.default_rng(42)
    arr = rng.random((10,), dtype=np.float32)
    ref = f32_pow_failure.f(arr)
    val = f32_pow_failure(arr)
    assert np.allclose(ref, val)
    assert ref.dtype == val.dtype


@pytest.mark.gpu
def test_tasklets_with_same_local_name():
    sdfg = dace.SDFG("tester")
    sdfg.add_array("A", [4], dace.float32, dace.StorageType.GPU_Global)
    state = sdfg.add_state()
    me, mx = state.add_map("kernel", dict(i="0:1"), schedule=dace.ScheduleType.GPU_Device)
    t1 = state.add_tasklet(
        "sgn",
        {"a"},
        {"b"},
        """
mylocal: dace.float32
if a > 0:
    mylocal = 1
else:
    mylocal = -1
b = mylocal
    """,
    )
    t2 = state.add_tasklet(
        "sgn",
        {"a"},
        {"b"},
        """
mylocal: dace.float32
if a > 0:
    mylocal = 1
else:
    mylocal = -1
b = mylocal
    """,
    )

    a = state.add_read("A")
    b = state.add_write("A")
    state.add_memlet_path(a, me, t1, dst_conn="a", memlet=dace.Memlet("A[0]"))
    state.add_memlet_path(a, me, t2, dst_conn="a", memlet=dace.Memlet("A[1]"))
    state.add_memlet_path(t1, mx, b, src_conn="b", memlet=dace.Memlet("A[2]"))
    state.add_memlet_path(t2, mx, b, src_conn="b", memlet=dace.Memlet("A[3]"))

    sdfg.compile()


def test_complex_parts():
    """
    The real and imaginary parts of complex numbers (connectors, locals and elements of array connectors) are member
    functions in C++.
    """
    sdfg = dace.SDFG("complex_parts")
    sdfg.add_array("A", [20], dace.complex128)
    sdfg.add_array("B", [20], dace.complex128)
    sdfg.add_array("C", [20], dace.complex64)
    state = sdfg.add_state()
    state.add_mapped_tasklet(
        "parts",
        {"i": "0:20"},
        {"a": dace.Memlet("A[i]"), "b": dace.Memlet("B[i]"), "row": dace.Memlet("B[0:20]")},
        """
re = a.real * b.real - a.imag * b.imag
im = a.real * b.imag + a.imag * b.real
product = dace.complex128(re, im)
c = dace.complex64(product.imag + row[0].real, product.real)
""",
        {"c": dace.Memlet("C[i]")},
        external_edges=True,
    )
    assert "a.real()" in sdfg.generate_code()[0].clean_code

    rng = np.random.default_rng(42)
    a = rng.random(20) + 1j * rng.random(20)
    b = rng.random(20) + 1j * rng.random(20)
    c = np.zeros(20, np.complex64)
    sdfg(A=a, B=b, C=c)
    product = a * b
    assert np.allclose(c, product.imag + b[0].real + 1j * product.real)


if __name__ == "__main__":
    test_integer_power()
    test_integer_power_constant()
    test_equality()
    test_pow_with_implicit_casting()
    test_tasklets_with_same_local_name()
    test_complex_parts()
