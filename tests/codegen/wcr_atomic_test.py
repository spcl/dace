# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests atomic WCR detection in code generation. """
import dace
import numpy as np

N = dace.symbol('N')


def test_wcr_overlapping_atomic():

    @dace.program
    def tester(A: dace.float32[2 * N + 3]):
        for i in dace.map[0:N]:
            A[2 * i:2 * i + 3] += 1

    sdfg = tester.to_sdfg()
    code: str = sdfg.generate_code()[0].code
    assert code.count('atomic') == 1


def test_wcr_strided_atomic():

    @dace.program
    def tester(A: dace.float32[2 * N]):
        for i in dace.map[1:N - 1]:
            A[2 * i - 1:2 * i + 2] += 1

    sdfg = tester.to_sdfg()
    code: str = sdfg.generate_code()[0].code
    assert code.count('atomic') == 1


def test_wcr_strided_nonatomic():

    @dace.program
    def tester(A: dace.float32[2 * N + 3]):
        for i in dace.map[0:N]:
            A[2 * i:2 * i + 2] += 1

    sdfg = tester.to_sdfg()
    code: str = sdfg.generate_code()[0].code
    assert code.count('atomic') == 0


def test_wcr_strided_nonatomic_offset():

    @dace.program
    def tester(A: dace.float32[2 * N]):
        for i in dace.map[1:N - 1]:
            A[2 * i - 1:2 * i + 1] += 1

    sdfg = tester.to_sdfg()
    code: str = sdfg.generate_code()[0].code
    assert code.count('atomic') == 0


if __name__ == '__main__':
    test_wcr_overlapping_atomic()
    test_wcr_strided_atomic()
    test_wcr_strided_nonatomic()
    test_wcr_strided_nonatomic_offset()
