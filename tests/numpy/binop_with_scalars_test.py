import dace
import numpy as np

### Left #####################################################################


@dace.program
def addltest(A: dace.int64[5, 5], B: dace.int64, C: dace.int64[5, 5]):
    C[:] = A + B


@dace.program
def subltest(A: dace.int64[5, 5], B: dace.int64, C: dace.int64[5, 5]):
    C[:] = A - B


@dace.program
def multltest(A: dace.int64[5, 5], B: dace.int64, C: dace.int64[5, 5]):
    C[:] = A * B


@dace.program
def divltest(A: dace.float64[5, 5], B: dace.float64, C: dace.float64[5, 5]):
    C[:] = A / B


@dace.program
def floordivltest(A: dace.int64[5, 5], B: dace.int64, C: dace.int64[5, 5]):
    C[:] = A // B


@dace.program
def modltest(A: dace.int64[5, 5], B: dace.int64, C: dace.int64[5, 5]):
    C[:] = A % B


@dace.program
def powltest(A: dace.int64[5, 5], B: dace.int64, C: dace.int64[5, 5]):
    C[:] = A**B


@dace.program
def matmultltest(A: dace.int64[5, 5], B: dace.int64, C: dace.int64[5, 5]):
    C[:] = A @ B


@dace.program
def lshiftltest(A: dace.int64[5, 5], B: dace.int64, C: dace.int64[5, 5]):
    C[:] = A << B


@dace.program
def rshiftltest(A: dace.int64[5, 5], B: dace.int64, C: dace.int64[5, 5]):
    C[:] = A >> B


@dace.program
def bitorltest(A: dace.int64[5, 5], B: dace.int64, C: dace.int64[5, 5]):
    C[:] = A | B


@dace.program
def bitxorltest(A: dace.int64[5, 5], B: dace.int64, C: dace.int64[5, 5]):
    C[:] = A ^ B


@dace.program
def bitandltest(A: dace.int64[5, 5], B: dace.int64, C: dace.int64[5, 5]):
    C[:] = A & B


@dace.program
def andltest(A: dace.bool[5, 5], B: dace.bool, C: dace.bool[5, 5]):
    C[:] = A and B


@dace.program
def orltest(A: dace.bool[5, 5], B: dace.bool, C: dace.bool[5, 5]):
    C[:] = A or B


@dace.program
def eqltest(A: dace.int64[5, 5], B: dace.int64, C: dace.bool[5, 5]):
    C[:] = A == B


@dace.program
def noteqltest(A: dace.int64[5, 5], B: dace.int64, C: dace.bool[5, 5]):
    C[:] = A != B


@dace.program
def ltltest(A: dace.int64[5, 5], B: dace.int64, C: dace.bool[5, 5]):
    C[:] = A < B


@dace.program
def lteltest(A: dace.int64[5, 5], B: dace.int64, C: dace.bool[5, 5]):
    C[:] = A <= B


@dace.program
def gtltest(A: dace.int64[5, 5], B: dace.int64, C: dace.bool[5, 5]):
    C[:] = A > B


@dace.program
def gteltest(A: dace.int64[5, 5], B: dace.int64, C: dace.bool[5, 5]):
    C[:] = A >= B


### Right #####################################################################


@dace.program
def addrtest(A: dace.int64, B: dace.int64[5, 5], C: dace.int64[5, 5]):
    C[:] = A + B


@dace.program
def subrtest(A: dace.int64, B: dace.int64[5, 5], C: dace.int64[5, 5]):
    C[:] = A - B


@dace.program
def multrtest(A: dace.int64, B: dace.int64[5, 5], C: dace.int64[5, 5]):
    C[:] = A * B


@dace.program
def divrtest(A: dace.float64, B: dace.float64[5, 5], C: dace.float64[5, 5]):
    C[:] = A / B


@dace.program
def floordivrtest(A: dace.int64, B: dace.int64[5, 5], C: dace.int64[5, 5]):
    C[:] = A // B


@dace.program
def modrtest(A: dace.int64, B: dace.int64[5, 5], C: dace.int64[5, 5]):
    C[:] = A % B


@dace.program
def powrtest(A: dace.int64, B: dace.int64[5, 5], C: dace.int64[5, 5]):
    C[:] = A**B


@dace.program
def matmultrtest(A: dace.int64, B: dace.int64[5, 5], C: dace.int64[5, 5]):
    C[:] = A @ B


@dace.program
def lshiftrtest(A: dace.int64, B: dace.int64[5, 5], C: dace.int64[5, 5]):
    C[:] = A << B


@dace.program
def rshiftrtest(A: dace.int64, B: dace.int64[5, 5], C: dace.int64[5, 5]):
    C[:] = A >> B


@dace.program
def bitorrtest(A: dace.int64, B: dace.int64[5, 5], C: dace.int64[5, 5]):
    C[:] = A | B


@dace.program
def bitxorrtest(A: dace.int64, B: dace.int64[5, 5], C: dace.int64[5, 5]):
    C[:] = A ^ B


@dace.program
def bitandrtest(A: dace.int64, B: dace.int64[5, 5], C: dace.int64[5, 5]):
    C[:] = A & B


@dace.program
def andrtest(A: dace.bool, B: dace.bool[5, 5], C: dace.bool[5, 5]):
    C[:] = A and B


@dace.program
def orrtest(A: dace.bool, B: dace.bool[5, 5], C: dace.bool[5, 5]):
    C[:] = A or B


@dace.program
def eqrtest(A: dace.int64, B: dace.int64[5, 5], C: dace.bool[5, 5]):
    C[:] = A == B


@dace.program
def noteqrtest(A: dace.int64, B: dace.int64[5, 5], C: dace.bool[5, 5]):
    C[:] = A != B


@dace.program
def ltrtest(A: dace.int64, B: dace.int64[5, 5], C: dace.bool[5, 5]):
    C[:] = A < B


@dace.program
def ltertest(A: dace.int64, B: dace.int64[5, 5], C: dace.bool[5, 5]):
    C[:] = A <= B


@dace.program
def gtrtest(A: dace.int64, B: dace.int64[5, 5], C: dace.bool[5, 5]):
    C[:] = A > B


@dace.program
def gtertest(A: dace.int64, B: dace.int64[5, 5], C: dace.bool[5, 5]):
    C[:] = A >= B


if __name__ == '__main__':
    A = np.random.randint(1, 10, size=(5, 5))
    Af = np.random.rand(5, 5)
    Apb = np.random.randint(0, 2, size=(5, 5))
    Ab = np.random.randint(0, 2, size=(5, 5)).astype(bool)
    B = np.random.randint(1, 10)
    Bf = np.random.rand()
    Bpb = np.random.randint(0, 2)
    Bb = bool(np.random.randint(0, 2))
    C = np.random.randint(1, 10, size=(5, 5))
    Cf = np.random.rand(5, 5)
    Cpb = np.random.randint(0, 2, size=(5, 5))
    Cb = np.random.randint(0, 2, size=(5, 5)).astype(bool)

    failed_tests = set()

    for opname, op in {
            'add': '+',
            'sub': '-',
            'mult': '*',
            'div': '/',
            'floordiv': '//',
            'mod': '%',
            'pow': '**',
            #    'matmult': '@',
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
            'gte': '>='
    }.items():

        def test(A, B, C, side: str = 'l', np_exec: str = None):
            daceC = C.copy()
            exec('{opn}{s}test(A, B, daceC)'.format(opn=opname, s=side))
            numpyC = C.copy()
            if not np_exec:
                exec('numpyC[:] = A {op} B'.format(op=op))
            else:
                exec(np_exec.format(op=op))
            if C.dtype == 'bool':
                norm_diff = 1.0
                if np.array_equal(numpyC, daceC):
                    norm_diff = 0.0
            else:
                norm_diff = np.linalg.norm(numpyC - daceC)
            if norm_diff == 0.0:
                print('Binary operator {opn}_{s}: OK'.format(opn=opname,
                                                             s=side))
            else:
                failed_tests.add(opname + '_' + side)
                print('Binary operator {opn}_{s}: FAIL ({diff})'.format(
                    opn=opname, s=side, diff=norm_diff))

        if opname == 'div':
            test(Af, Bf, Cf)
            test(Bf, Af, Cf, side='r')
        elif opname in {'and', 'or'}:
            test(Ab, Bb, Cb, np_exec='numpyC[:] = np.logical_{op}(A, B)')
            test(Bb,
                 Ab,
                 Cb,
                 side='r',
                 np_exec='numpyC[:] = np.logical_{op}(A, B)')
        elif opname in {'eq', 'noteq', 'lt', 'lte', 'gt', 'gte'}:
            test(Apb, Bpb, Cb)
            test(Bpb, Apb, Cb, side='r')
        else:
            test(A, B, C)
            test(B, A, C, side='r')

    if failed_tests:
        print('FAILED TESTS:')
        for t in failed_tests:
            print(t)
        exit(-1)
    exit(0)
