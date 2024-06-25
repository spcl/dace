import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass
NA, NB, Nkz, NE, Nqz, Nw, Norb, N3D = 32,32,32,32,32,32,32,32

@dc.program
def scattering_self_energies(neigh_idx: dc.int32[NA, NB],
                             dH: dc.complex128[NA, NB, N3D, Norb, Norb],
                             G: dc.complex128[Nkz, NE, NA, Norb, Norb],
                             D: dc.complex128[Nqz, Nw, NA, NB, N3D, N3D],
                             Sigma: dc.complex128[Nkz, NE, NA, Norb, Norb], S: dc.float64[1]):

    for k in range(Nkz):
        for E in range(NE):
            for q in range(Nqz):
                for w in range(Nw):
                    for i in range(N3D):
                        for j in range(N3D):
                            for a in range(NA):
                                for b in range(NB):
                                    if E - w >= 0:
                                        dHG = G[k, E - w,
                                                neigh_idx[a, b]] @ dH[a, b, i]
                                        dHD = dH[a, b, j] * D[q, w, a, b, i, j]
                                        Sigma[k, E, a] += dHG @ dHD

    @dc.map(_[0:Nkz, 0:NE, 0:NA, 0:Norb, 0:Norb])
    def summap(i, j, k, l, m):
        s >> S(1, lambda x, y: x + y)[0]
        z << Sigma[i, j, k, l, m]
        s = z



sdfg = scattering_self_energies.to_sdfg()

sdfg.save("log_sdfgs/scattering_self_energies_forward.sdfg")


add_backward_pass(sdfg=sdfg, inputs=["Sigma"], outputs=["S"])

sdfg.save("log_sdfgs/scattering_self_energies_backward.sdfg")

