import dace
import numpy

N = dace.symbol('N')


@dace.program
def Test(A:dace.float32[N], B:dace.float32[N]):
    return A+B


sdfg = Test.to_sdfg()
csdfg = sdfg.compile()
A = numpy.ndarray(shape = [30])
B = numpy.ndarray(shape = [30])
csdfg(A=A, B=B, N=N)
print(sdfg)
print("Free symbols")
print(sdfg.free_symbols)

print("arglist")
print(sdfg.arglist())

print("signature_arglist")
print(sdfg.signature_arglist())


alist = sdfg.arglist()
print(alist['A'].shape[0])
print(type(alist['A'].shape[0]))

print(alist['A'].dtype.type)
print(type(alist['A'].dtype.type))
