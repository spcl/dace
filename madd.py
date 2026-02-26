import dace

N = dace.symbol("N")

@dace.program
def madd(A: dace.float64[N, N],
         B: dace.float64[N, N],
         C: dace.float64[N, N]) -> dace.float64[N, N]:
    for i,  j in dace.map[0:N, 0:N]:
        C[i, j] = A[i, j] + B[i, j]


sdfg1 = madd.to_sdfg()
sdfg1.save("madd.sdfgz", compress=True)


@dace.program
def madd_tiled(A: dace.float64[N, N],
         B: dace.float64[N, N],
         C: dace.float64[N, N]) -> dace.float64[N, N]:
    for i,  j in dace.map[0:N:16, 0:N:16]:
        for ti, tj in dace.map[0:16, 0:16]:
            C[i + ti, j + tj] = A[i + ti, j + tj] + B[i + ti, j + tj]

sdfg2 = madd_tiled.to_sdfg()
sdfg2.save("madd_tiled.sdfgz", compress=True)


# externalize map reads