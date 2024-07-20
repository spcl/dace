import dace
import numpy as np

@dace.program
def cpu_getstarted(A, B):
    return A + B

if __name__ == "__main__":
    #a = np.random.rand(2,3)
    a = 10
    b = 20
    print ("before dace(CPU) (a,b)", a, b)
    print("after dace(CPU)", cpu_getstarted(a, b))
    sdfg = cpu_getstarted.to_sdfg(a, b)

    sdfg.save('save_cpu_sdfg.py', use_pickle=True)
    # sdfg.apply_gpu_transformations()