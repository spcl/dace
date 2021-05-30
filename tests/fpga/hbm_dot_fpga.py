from dace import subsets
import dace
import numpy as np

def create_dot_sdfg():
    N = dace.symbol("N")
    M = dace.symbol("M")

    BURST = dace.symbol("BURST")
    BANK = dace.symbol("BANK")
    @dace.program
    def matvec(A : dace.float32[M, N], x : dace.float32[N],y : dace.float32[M]):
        for xpos in dace.map[0:(N // BURST)]:
            tmpX = dace.define_local([BURST], dace.float32)
            for i in dace.map[0:BURST]:
                tmpX[i] = x[xpos * BURST + i]
            for ypos in dace.map[0:(M)]:
                #tmpY = dace.define_local(1, dace.float32)
                #tmpY = y[ypos]
                for i in dace.map[0:BURST]:
                    y = dace.reduce(lambda a, b: a + b, A[ypos, xpos]  * tmpX[i])
    sdfg = matvec.to_sdfg()
    sdfg.validate()
    sdfg.apply_fpga_transformations()
    return sdfg

if __name__ == '__main__':
    sdfg = create_dot_sdfg()
    sdfg.view()