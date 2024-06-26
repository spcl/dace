from dace.transformation.estimator.soap.io_analysis import perform_soap_analysis_from_ir


ir = \
"""
In:
      A
      B
      C
      K
      M
      N
      alpha
      beta
Out:
      D
Program:
M N K : acc[{2},{1}] += A[{2},{0}] * B[{0},{1}]
M N   : alpha_acc[{1},{0}] = alpha * acc[{1},{0}]
M N   : D[{1},{0}] = + beta * C[{1},{0}] + alpha_acc[{1},{0}]"""

ir = \
"""
In:
      K
      Q
      V
      W
Out:
      O2
Program:
mpi_comm_world D L L num_blocks_per_threadblock : A1[{0},{2},{1}] += K[{2},{0}] * Q[{0},{1}]
L L   : A[{1},{0}] = softmax(A1[{1},{0}]) ???
L D L : O1[{2},{1}] += A[{2},{0}] * V[{0},{1}]
D D L : O2[{2},{1}] += O1[{2},{0}] * W[{0},{1}]
"""



if __name__ == '__main__':
    decomp_params = [("P", 4), ("Ss", 12), ("S_shared", 12)]
    for i in range(10):
        decomp_params.append((f"S{i}", 100))
    decomp_params.append(('M', 1000))
    decomp_params.append(('n', 960))
    decomp_params.append(('K', 1000))
    decomp_params.append(('D', 4))
    decomp_params.append(('L', 16))
    decomp_params.append(('alignment', 1))
    soap_result = perform_soap_analysis_from_ir(ir, decomp_params,
                    generate_schedule = True)