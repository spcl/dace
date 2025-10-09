import dace

@dace.program
def double_nest_thread_specialization_2(A: dace.float64[128, 64, 32], c: dace.float64[2]):
    for i in dace.map[0:128]:
        for j in dace.map[0:32]:
            with dace.tasklet:
                out >> A[i, j, 0]
                out = 5.0
            if c[0] > 2.0:
                for k in dace.map[0:16]:
                    with dace.tasklet:
                        out2 >> A[i, j, k]
                        out2 = 5.0

        for j in dace.map[33:60]:
            with dace.tasklet:
                inp << A[i, j, 0]
                out >> A[i, j, 0]
                out = 6.0 + inp
            if c[0] > 2.0:
                for k in dace.map[16:32]:
                    with dace.tasklet:
                        inp2 << A[i, j, k]
                        out2 >> A[i, j, k]
                        out2 = 6.0 + inp2

@dace.program
def double_nest_thread_specialization_1(A: dace.float64[128, 64, 32], c: dace.float64):
    for i in dace.map[0:128]:
        for j in dace.map[0:32]:
            with dace.tasklet:
                out >> A[i, j, 0]
                out = 5.0
            if c > 2.0:
                for k in dace.map[0:16]:
                    with dace.tasklet:
                        out2 >> A[i, j, k]
                        out2 = 5.0

        for j in dace.map[33:60]:
            with dace.tasklet:
                inp << A[i, j, 0]
                out >> A[i, j, 0]
                out = 6.0 + inp
            if c > 2.0:
                for k in dace.map[16:32]:
                    with dace.tasklet:
                        inp2 << A[i, j, k]
                        out2 >> A[i, j, k]
                        out2 = 6.0 + inp2


sdfg = double_nest_thread_specialization_2.to_sdfg()
sdfg.validate()
sdfg.save("s2.sdfg")

sdfg = double_nest_thread_specialization_1.to_sdfg()
sdfg.validate()
sdfg.save("s1.sdfg")