import dace
import numpy as np


@dace.program
def addtest(A: dace.int64[5, 5], B: dace.int64[5, 5], C: dace.int64[5, 5]):
    C[:] = A + B

@dace.program
def subtest(A: dace.int64[5, 5], B: dace.int64[5, 5], C: dace.int64[5, 5]):
    C[:] = A - B

@dace.program
def multtest(A: dace.int64[5, 5], B: dace.int64[5, 5], C: dace.int64[5, 5]):
    C[:] = A * B

@dace.program
def divtest(A: dace.float64[5, 5], B: dace.float64[5, 5], C: dace.float64[5, 5]):
   C[:] = A / B

@dace.program
def floordivtest(A: dace.int64[5, 5], B: dace.int64[5, 5], C: dace.int64[5, 5]):
    C[:] = A // B

@dace.program
def modtest(A: dace.int64[5, 5], B: dace.int64[5, 5], C: dace.int64[5, 5]):
    C[:] = A % B

@dace.program
def powtest(A: dace.int64[5, 5], B: dace.int64[5, 5], C: dace.int64[5, 5]):
    C[:] = A ** B

@dace.program
def matmulttest(A: dace.int64[5, 5], B: dace.int64[5, 5], C: dace.int64[5, 5]):
    C[:] = A @ B

@dace.program
def lshifttest(A: dace.int64[5, 5], B: dace.int64[5, 5], C: dace.int64[5, 5]):
    C[:] = A << B

@dace.program
def rshifttest(A: dace.int64[5, 5], B: dace.int64[5, 5], C: dace.int64[5, 5]):
    C[:] = A >> B

@dace.program
def bitortest(A: dace.int64[5, 5], B: dace.int64[5, 5], C: dace.int64[5, 5]):
    C[:] = A | B

@dace.program
def bitxortest(A: dace.int64[5, 5], B: dace.int64[5, 5], C: dace.int64[5, 5]):
    C[:] = A ^ B

@dace.program
def bitandtest(A: dace.int64[5, 5], B: dace.int64[5, 5], C: dace.int64[5, 5]):
    C[:] = A & B

@dace.program
def andtest(A: dace.bool[5, 5], B: dace.bool[5, 5], C: dace.bool[5, 5]):
    C[:] = A and B

@dace.program
def ortest(A: dace.bool[5, 5], B: dace.bool[5, 5], C: dace.bool[5, 5]):
    C[:] = A or B

@dace.program
def eqtest(A: dace.int64[5, 5], B: dace.int64[5, 5], C: dace.int64[5, 5]):
    C[:] = A == B

@dace.program
def noteqtest(A: dace.int64[5, 5], B: dace.int64[5, 5], C: dace.int64[5, 5]):
    C[:] = A != B

@dace.program
def lttest(A: dace.int64[5, 5], B: dace.int64[5, 5], C: dace.int64[5, 5]):
    C[:] = A < B

@dace.program
def ltetest(A: dace.int64[5, 5], B: dace.int64[5, 5], C: dace.int64[5, 5]):
    C[:] = A <= B

@dace.program
def gttest(A: dace.int64[5, 5], B: dace.int64[5, 5], C: dace.int64[5, 5]):
    C[:] = A > B

@dace.program
def gtetest(A: dace.int64[5, 5], B: dace.int64[5, 5], C: dace.int64[5, 5]):
    C[:] = A >= B


if __name__ == '__main__':
    A = np.random.randint(1, 10, size=(5, 5))
    Af = np.random.rand(5, 5)
    Apb = np.random.randint(0, 2, size=(5, 5))
    Ab = np.random.randint(0, 2, size=(5, 5)).astype(bool)
    B = np.random.randint(1, 10, size=(5, 5))
    Bf = np.random.rand(5, 5)
    Bpb = np.random.randint(0, 2, size=(5, 5))
    Bb = np.random.randint(0, 2, size=(5, 5)).astype(bool)
    C = np.random.randint(1, 10, size=(5, 5))
    Cf = np.random.rand(5, 5)
    Cpb = np.random.randint(0, 2, size=(5, 5))
    Cb = np.random.randint(0, 2, size=(5, 5)).astype(bool)

    failed_tests = set()

    for opname, op in {'add': '+',
                       'sub': '-',
                       'mult': '*',
                       'div': '/',
                       'floordiv': '//',
                       'mod': '%',
                       'pow': '**',
                       'matmult': '@',
                       'lshift': '<<',
                       'rshift': '>>',
                       'bitor': '|',
                       'bitxor': '^',
                       'bitand': '&',
                       'and': 'and',
                       'or': 'or',
                       'eq': '==',
                       'noteq': '!=',
                       'lt': '<',
                       'lte': '<=',
                       'gt': '>',
                       'gte': '>='}.items():
        
        def test(A, B, C, np_exec: str = None):
            daceC = C.copy()
            exec('{opn}test(A, B, daceC)'.format(opn=opname))
            numpyC = C.copy()
            if not np_exec:
                exec('numpyC[:] = A {op} B'.format(op=op))
                norm_diff = np.linalg.norm(numpyC - daceC)
            else:
                exec(np_exec.format(op=op))
                norm_diff = 1.0
                if np.array_equal(numpyC, daceC):
                    norm_diff = 0.0
            if norm_diff == 0.0:
                print('Augmented {opn}: OK'.format(opn=opname))
            else:
                failed_tests.add(opname)
                print('Augmented {opn}: FAIL ({diff})'.format(opn=opname,
                                                            diff=norm_diff))
        
        if opname == 'div':
            test(Af, Bf, Cf)
        elif opname in {'and', 'or'}:
            test(Ab, Bb, Cb, np_exec='numpyC[:] = np.logical_{op}(A, B)')
        elif opname in {'eq', 'noteq', 'lt', 'lte', 'gt', 'gte'}:
            test(Apb, Bpb, Cpb)
        else:
            test(A, B, C)
        
    if failed_tests:
        print('FAILED TESTS:')
        for t in failed_tests:
            print(t)
        exit(-1)
    exit(0)
