import dace
import numpy as np

M = dace.symbol('M')
K = dace.symbol('K')
N = dace.symbol('N')


@dace.program
def matmul(A: dace.float32[M, K], B: dace.float32[K, N], C: dace.float32[M, N]):
    tmp = np.ndarray([M, N, K], dtype=A.dtype)

    # Multiply every pair of values to a large 3D temporary array
    for i, j, k in dace.map[0:M, 0:N, 0:K]:
        with dace.tasklet:
            in_A << A[i, k]
            in_B << B[k, j]
            out >> tmp[i, j, k]

            out = in_A * in_B

    # Sum last dimension of temporary array to obtain resulting matrix
    dace.reduce(lambda a, b: a + b, tmp, C, axis=2, identity=0)


m = 512
k = 512
n = 512

A = np.random.rand(m, k).astype(np.float32)
B = np.random.rand(k, n).astype(np.float32)
C = np.zeros((m, n), dtype=np.float32)

sdfg = matmul.to_sdfg()
sdfg.simplify()
sdfg.expand_library_nodes()
sdfg.specialize({M: m, N: n, K: k})

for nsdfg in sdfg.all_sdfgs_recursive():
    for state in nsdfg.nodes():
        state.instrument = dace.InstrumentationType.LIKWID_Counters
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                node.instrument = dace.InstrumentationType.LIKWID_Counters

csdfg = sdfg.compile()
for _ in range(1):
    csdfg(A=A, B=B, C=C)

report = sdfg.get_latest_report()
print(report)

measured_flops = sum(report.durations[(0,0,-1)]["RETIRED_SSE_AVX_FLOPS_SINGLE_ALL"])
flops = m * k * (n * 2)

print(f"Expected {flops} FLOPS, measured {measured_flops} FLOPS, diff: {measured_flops - flops}")

