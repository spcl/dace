import dace
import numpy as np

@dace.program
def cpu_getstarted(A, B, C):
    C = A + B
    return C

if __name__ == "__main__":
    #a = np.random.rand(2,3)
    # a = 10
    # b = 20
    # call with values
    A = np.ones((20), dtype=np.int32)   # 1,1,1,1,...
    B = np.ones((20), dtype=np.int32)   # 1,1,1,1,...
    C = np.zeros((20), dtype=np.int32)  # 0,0,0,0,...
    print ("before dace(CPU) (a,b)", A, B, C)
    print("after dace(CPU)", cpu_getstarted(A, B, C))
    sdfg = cpu_getstarted.to_sdfg(A, B, C)
    # sdfg.apply_gpu_transformations()