import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass

NP = 32
NQ = 32
NR = 32

@dc.program
def doitgen(A: dc.float64[NR, NQ, NP], C4: dc.float64[NP, NP], S:dc.float64[1]):

    for r in range(NR):
        A[r, :, :] = np.reshape(np.reshape(A[r], (NQ, NP)) @ C4, (NQ, NP))

    S[0] = np.sum(A)


sdfg = doitgen.to_sdfg()

sdfg.save("log_sdfgs/doitgen_forward.sdfg")

add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["S"])

sdfg.save("log_sdfgs/doitgen_backward.sdfg")

