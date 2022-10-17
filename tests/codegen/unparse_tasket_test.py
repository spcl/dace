# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
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


if __name__ == '__main__':
    test_integer_power()
    test_integer_power_constant()
    test_equality()
