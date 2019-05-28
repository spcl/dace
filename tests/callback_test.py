import dace

# Performs square
def mypythonfunction(inp):
    import numpy
    return numpy.square(inp)

@dace.program(dace.uint32[2], dace.uint32[2], dace.callback(dace.uint32, dace.uint32))
def external_tasklet(A, B, callback_function):
    @dace.map(_[0:2])
    def index(i):
        a << A[i]
        b >> B[i]
        b = callback_function(a)
        #b = a+1.0


if __name__ == '__main__':

    A = dace.ndarray((2, ), dtype=dace.int32)
    B = dace.ndarray((2, ), dtype=dace.int32)

    A[:] = 5
    B[:] = 0

    print(A)
    external_tasklet(A, B, mypythonfunction)
    print(B)
