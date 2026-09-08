# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest


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

    assert ':pow(' not in sdfg.generate_code()[0].clean_code


def test_integer_power_constant():

    @dace.program
    def powint(A: dace.float64[20]):
        for i in dace.map[0:20]:
            with dace.tasklet:
                a << A[i]
                b >> A[i]
                b = a**myconst

    sdfg = powint.to_sdfg()
    sdfg.add_constant('myconst', dace.float32(2.0))

    assert ':pow(' not in sdfg.generate_code()[0].clean_code


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
    arr = rng.random((10, ), dtype=np.float32)
    ref = f32_pow_failure.f(arr)
    val = f32_pow_failure(arr)
    assert np.allclose(ref, val)
    assert ref.dtype == val.dtype


@pytest.mark.gpu
def test_bitwise_call_spelling_lowers_to_operators():
    """A bitwise operator written as a CALL must reach C++ as the operator.

    ``dace.symbolic`` models every Python bitwise operator as a function -- ``a & b`` parses to
    ``__bitwise_and`` -- so a producer that renders a tasklet body with ``str()`` instead of
    ``symstr`` keeps the call spelling. C++ has no such function, and the body reached the
    compiler as a call to an undeclared name::

        error: '__bitwise_and' was not declared in this scope

    Runs the kernel as well as reading the emitted text: an operator that lowered to the WRONG
    operator would satisfy the text assertions below and still compute the wrong answer.
    """
    sdfg = dace.SDFG("bitwise_call_spelling")
    sdfg.add_array("A", (8, ), dace.int64)
    sdfg.add_array("B", (8, ), dace.int64)
    state = sdfg.add_state()

    read = state.add_read("A")
    write = state.add_write("B")
    tasklet = state.add_tasklet(
        name="crc_step",
        inputs={"_in"},
        outputs={"_out"},
        # The three shapes the crc16 corpus kernel produces: a shift, a xor and a mask.
        code="_out = __bitwise_or(__bitwise_xor(__right_shift(_in, 1), 33800), __bitwise_and(_in, 1))",
    )
    entry, exit_ = state.add_map("m", {"i": "0:8"})
    state.add_memlet_path(read, entry, tasklet, dst_conn="_in", memlet=dace.Memlet("A[i]"))
    state.add_memlet_path(tasklet, exit_, write, src_conn="_out", memlet=dace.Memlet("B[i]"))
    sdfg.validate()

    code = sdfg.generate_code()[0].clean_code
    assert "__bitwise" not in code, f"a bitwise call reached C++ unlowered:\n{code}"
    assert "__right_shift" not in code, f"a shift call reached C++ unlowered:\n{code}"

    a = np.arange(1, 9, dtype=np.int64)
    b = np.zeros(8, dtype=np.int64)
    sdfg(A=a, B=b)
    assert np.array_equal(b, ((a >> 1) ^ 33800) | (a & 1))


def test_tasklets_with_same_local_name():
    sdfg = dace.SDFG('tester')
    sdfg.add_array('A', [4], dace.float32, dace.StorageType.GPU_Global)
    state = sdfg.add_state()
    me, mx = state.add_map('kernel', dict(i='0:1'), schedule=dace.ScheduleType.GPU_Device)
    t1 = state.add_tasklet(
        'sgn', {'a'}, {'b'}, '''
mylocal: dace.float32
if a > 0:
    mylocal = 1
else:
    mylocal = -1
b = mylocal
    ''')
    t2 = state.add_tasklet(
        'sgn', {'a'}, {'b'}, '''
mylocal: dace.float32
if a > 0:
    mylocal = 1
else:
    mylocal = -1
b = mylocal
    ''')

    a = state.add_read('A')
    b = state.add_write('A')
    state.add_memlet_path(a, me, t1, dst_conn='a', memlet=dace.Memlet('A[0]'))
    state.add_memlet_path(a, me, t2, dst_conn='a', memlet=dace.Memlet('A[1]'))
    state.add_memlet_path(t1, mx, b, src_conn='b', memlet=dace.Memlet('A[2]'))
    state.add_memlet_path(t2, mx, b, src_conn='b', memlet=dace.Memlet('A[3]'))

    sdfg.compile()


if __name__ == '__main__':
    test_integer_power()
    test_integer_power_constant()
    test_equality()
    test_pow_with_implicit_casting()
    test_tasklets_with_same_local_name()
