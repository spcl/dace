import dace

TS = dace.symbol("TS")
@dace.program
def example(A: dace.float64[100, 100], B: dace.float64[100, 100], C: dace.float64[100, 100], D: dace.float64[100, 100], E: dace.float64[100]) -> dace.float64[100, 100]:
    for t1 in range(TS):
        for i, j in dace.map[0:100, 0:100]:
            C[i, j] = A[i, j] + B[i, j]
    for t2 in range(2):
        for j in range(100):
            for i in dace.map[0:100]:
                E[i] = E[i] + C[i, j]
        for i in range(1, 100):
            E[i] = (E[i-1] + E[i]) / 100.0
    for t3 in range(2):
        for i, j in dace.map[0:100, 0:100]:
            D[i, j] = E[i] * 2.0 + C[i, j]

sdfg = example.to_sdfg()
sdfg.save("example.sdfg")
sdfg.view()

"""
thesis
introduction: rephrase thesis globals
list contributions
related work: how do others lower things to GPU
implementation: document code and ask claude to summarize
evaluation: test against nbench, test sdfgs which failed previously
"""