import dace
from dace.transformation.heterogeneous import MultiExpansion
from dace.transformation.heterogeneous import SubgraphFusion
from dace.transformation.heterogeneous.helpers import *
import dace.sdfg.nodes as nodes
import numpy as np



N, M, O, P, Q, R = [dace.symbol(s) for s in ['N', 'M', 'O', 'P', 'Q', 'R']]
N.set(50)
M.set(60)
O.set(70)
P.set(80)
Q.set(90)
R.set(100)


@dace.program
def TEST(A: dace.float64[N], B: dace.float64[M], C: dace.float64[O],
         D: dace.float64[M], E: dace.float64[N], F: dace.float64[P]):

    tmp1 = np.ndarray([N,M,O], dtype = dace.float64)
    for i, j, k in dace.map[0:N, 0:M, 0:O]:
        with dace.tasklet:
            in1 << A[i]
            in2 << B[j]
            in3 << C[k]
            out >> tmp1[i,j,k]

            out = in1 + in2 + in3

    tmp2 = np.ndarray([M,N,P], dtype = dace.float64)
    for j, l, k in dace.map[0:M, 0:P, 0:N]:
        with dace.tasklet:
            in1 << A[k]
            in2 << E[j]
            in3 << F[l]
            out >> tmp2[j,k,l]

            out = in1 + in2 + in3

@dace.program
def TEST2(A: dace.float64[N], B:dace.float64[M],
          C: dace.float64[N], D:dace.float64[M]):

    tmp1 = np.ndarray([N,M,N], dtype = dace.float64)
    for i,j,k in dace.map[0:N, 0:M, 0:N]:
        with dace.tasklet:
            in1 << A[i]
            in2 << B[j]
            in3 << C[k]

            out >> tmp1[i,j,k]

            out = in1+in2+in3

    tmp2 = np.ndarray([M,N], dtype = dace.float64)
    for n,m in dace.map[0:N, 0:M]:
        with dace.tasklet:
            in1 << tmp1[n,m,:]
            in2 << B[m]
            in3 << D[m]

            out >> tmp2[m,n]

            out = in1[0]*in2*in3

@dace.program
def TEST3(A: dace.float64[N], B:dace.float64[M],
          C: dace.float64[N], D:dace.float64[M]):

    tmp = np.ndarray([1], dtype = dace.float64)
    tmp1 = np.ndarray([N,M,N], dtype = dace.float64)
    tmp2 = np.ndarray([N,M,N], dtype = dace.float64)
    tmp3 = np.ndarray([N,M,N], dtype = dace.float64)
    tmp4 = np.ndarray([N,M,N,M], dtype = dace.float64)
    tmp5 = np.ndarray([N,M,N,M], dtype = dace.float64)


    for i,j,k in dace.map[0:N, 0:M, 0:N]:
        with dace.tasklet:
            in1 << A[i]
            in2 << B[j]
            in3 << C[k]

            out1 >> tmp1[i,j,k]
            out2 >> tmp2[i,j,k]
            out3 >> tmp3[i,j,k]

            out1 = in1+in2+in3
            out2 = in1*in2+in3
            out3 = in1*in2*in3

    #for i,j,k,l in dace.map[0:N, 0:M, 0:N, 0:M]:
    for l,k,j,i in dace.map[0:M, 0:N, 0:M, 0:N]:
        with dace.tasklet:
            in1 << tmp1[i,j,k]
            in2 << D[l]
            out >> tmp4[i,j,k,l]

            out = in1*in2

    for i,j,k,l in dace.map[0:N, 0:M, 0:N, 0:M]:
        with dace.tasklet:
            in1 << B[l]
            in2 << tmp2[i,j,k]

            out >> tmp5[i,j,k,l]

            out = in1 + in2 - 42


    dace.reduce('lambda a,b: a+2*b', tmp1, tmp)


@dace.program
def TEST4(A: dace.float64[N], B:dace.float64[N],
          C: dace.float64[N]):

    for i in dace.map[0:N]:
        with dace.tasklet:
            in1 << A[i]
            out1 >> B[i]
            out1 = in1 + 1

    for i in dace.map[0:N]:
        with dace.tasklet:
            in1 << B[i]
            out1 >> C[i]
            out1 = in1 + 1

if __name__ == "__main__":

    symbols = {str(var):var.get() for var in [N,M,O,P,Q,R]}

    sdfg1 = TEST.to_sdfg()
    sdfg2 = TEST2.to_sdfg()
    sdfg3 = TEST3.to_sdfg()
    sdfg4 = TEST4.to_sdfg()
    sdfg1.apply_gpu_transformations()
    sdfg2.apply_gpu_transformations()
    sdfg3.apply_gpu_transformations()
    sdfg4.apply_gpu_transformations()
    
    map_base_variables = [None, None, ['i','j','k'], None]

    # first, let us test the helper functions

    #roof = dace.perf.roofline.Roofline(dace.perf.specs.PERF_CPU_DAVINCI, symbols, debug = True)
    #optimizer = dace.perf.optimizer.SDFGRooflineOptimizer(sdfg1, roof, inplace = False)
    #optimizer.optimize()

    map_base_vars_iter = iter(map_base_variables)
    for test_number in [2,3,4]:
        sdfg = locals()[f'sdfg{test_number}']
        print(f"############ TEST {test_number}")
        state = sdfg.nodes()[0]
        map_entries = [node for node in state.nodes() if isinstance(node, nodes.MapEntry)]
        maps = [node.map for node in state.nodes() if isinstance(node, nodes.MapEntry)]


        common_base_ranges = common_map_base_ranges(maps)
        print("COMMON BASE RANGES")
        print(common_base_ranges)


        reassignment_dict = find_reassignment(maps, common_base_ranges)
        print(reassignment_dict)
        sdfg.view()


        # test transformation
        print("**** MultiExpansion Test")
        transformation = MultiExpansion()
        #transformation.expand(sdfg, state, map_entries)
        transformation.expand(sdfg, state, map_entries, map_base_variables[test_number-1])


        print("**** SubgraphFusion Test")
        transformation = SubgraphFusion()
        #exit()
        transformation.fuse(sdfg, state, map_entries)
        print("VALDIATION:")
        sdfg.validate()
        print("PASS")
        sdfg.view()
