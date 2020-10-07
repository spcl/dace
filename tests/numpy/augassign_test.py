# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


@dace.program
def augaddtest(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    B += A


@dace.program
def augsubtest(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    B -= A


@dace.program
def augmulttest(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    B *= A


@dace.program
def augdivtest(A: dace.float64[5, 5], B: dace.float64[5, 5]):
    B /= A


@dace.program
def augfloordivtest(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    B //= A


@dace.program
def augmodtest(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    B %= A


@dace.program
def augpowtest(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    B **= A


@dace.program
def auglshifttest(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    B <<= A


@dace.program
def augrshifttest(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    B >>= A


@dace.program
def augbitortest(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    B |= A


@dace.program
def augbitxortest(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    B ^= A


@dace.program
def augbitandtest(A: dace.int64[5, 5], B: dace.int64[5, 5]):
    B &= A


if __name__ == '__main__':
    A = np.random.randint(1, 10, size=(5, 5))
    Af = np.random.rand(5, 5)
    B = np.random.randint(1, 10, size=(5, 5))
    Bf = np.random.rand(5, 5)

    failed_tests = set()

    for opname, op in {
            'add': '+',
            'sub': '-',
            'mult': '*',
            'div': '/',
            'floordiv': '//',
            'mod': '%',
            'pow': '**',
            'lshift': '<<',
            'rshift': '>>',
            'bitor': '|',
            'bitxor': '^',
            'bitand': '&'
    }.items():

        def test(A, B):
            daceB = B.copy()
            exec('aug{opn}test(A, daceB)'.format(opn=opname))
            numpyB = B.copy()
            exec('numpyB {op}= A'.format(op=op))
            norm_diff = np.linalg.norm(numpyB - daceB)
            if norm_diff == 0.0:
                print('Augmented {opn}: OK'.format(opn=opname))
            else:
                failed_tests.add(opname)
                print('Augmented {opn}: FAIL ({diff})'.format(opn=opname,
                                                              diff=norm_diff))

        if opname == 'div':
            test(Af, Bf)
        else:
            test(A, B)

    if failed_tests:
        print('FAILED TESTS:')
        for t in failed_tests:
            print(t)
        exit(-1)
    exit(0)
