import dace

N = dace.symbol("N")

@dace.program
def madd(A: dace.float64[N, N],
         B: dace.float64[N, N],
         C: dace.float64[N, N]) -> dace.float64[N, N]:
    for i,  j in dace.map[0:N, 0:N]:
        C[i, j] = A[i, j] + B[i, j]


sdfg1 = madd.to_sdfg()
sdfg1.save("madd1.sdfg")