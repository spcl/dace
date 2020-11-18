# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


@dace.program
def uaddtest(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    B[:] = +A


@dace.program
def usubtest(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    B[:] = -A


@dace.program
def nottest(A: dace.bool_[5, 5], B: dace.bool_[5, 5]):
    B[:] = not A


@dace.program
def inverttest(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    B[:] = ~A


if __name__ == '__main__':
    A = np.random.randint(1, 10, size=(5, 5), dtype=np.int64)
    Ab = np.random.randint(0, 2, size=(5, 5)).astype(np.bool_)
    B = np.zeros((5, 5), dtype=np.int64)
    Bb = np.zeros((5, 5), dtype=np.int64).astype(np.bool_)

    failed_tests = set()

    for opname, op in {
            'uadd': '+',
            'usub': '-',
            'not': 'not',
            'invert': '~'
    }.items():

        def test(A, B, np_exec: str = None):
            daceB = B.copy()
            exec('{opn}test(A, daceB)'.format(opn=opname))
            numpyB = B.copy()
            if not np_exec:
                exec('numpyB[:] = {op}A'.format(op=op))
                norm_diff = np.linalg.norm(numpyB - daceB)
            else:
                exec(np_exec.format(op=op))
                norm_diff = 1.0
                if np.array_equal(numpyB, daceB):
                    norm_diff = 0.0
            if norm_diff == 0.0:
                print('Unary operator {opn}: OK'.format(opn=opname))
            else:
                failed_tests.add(opname)
                print('Unary operator {opn}: FAIL ({diff})'.format(
                    opn=opname, diff=norm_diff))

        if opname == 'not':
            test(Ab, Bb, np_exec='numpyB[:] = np.logical_{op}(A, B)')
        else:
            test(A, B)

    if failed_tests:
        print('FAILED TESTS:')
        for t in failed_tests:
            print(t)
        exit(-1)
    exit(0)
