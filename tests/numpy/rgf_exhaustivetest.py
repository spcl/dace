# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

# Declaration of symbolic variables
N, BS = (dace.symbol(name) for name in ['N', 'BS'])


def Her(A):
    return np.transpose(np.conjugate(A))


def trace(A):
    return np.trace(A)


@dace.program
def hermitian_transpose(A: dace.complex128[BS, BS], B: dace.complex128[BS, BS]):
    @dace.map(_[0:BS, 0:BS])
    def her_trans(i, j):
        inp << A[i, j]
        out >> B[j, i]
        out = dace_conj(inp)


@dace.program
def rgf_dense(HD: dace.complex128[N, BS, BS], HE: dace.complex128[N, BS, BS],
              HF: dace.complex128[N, BS, BS], sigmaRSD: dace.complex128[N, BS,
                                                                        BS],
              sigmaRSE: dace.complex128[N, BS,
                                        BS], sigmaRSF: dace.complex128[N, BS,
                                                                       BS],
              sigmaLSD: dace.complex128[N, BS,
                                        BS], sigmaLSE: dace.complex128[N, BS,
                                                                       BS],
              sigmaLSF: dace.complex128[N, BS,
                                        BS], sigmaGSD: dace.complex128[N, BS,
                                                                       BS],
              sigmaGSE: dace.complex128[N, BS,
                                        BS], sigmaGSF: dace.complex128[N, BS,
                                                                       BS],
              sigRl: dace.complex128[BS, BS], sigRr: dace.complex128[BS, BS],
              gammaleft: dace.complex128[BS,
                                         BS], gammaright: dace.complex128[BS,
                                                                          BS],
              fl: dace.float64[1], fr: dace.float64[1], GL: dace.complex128[N,
                                                                            BS,
                                                                            BS],
              GG: dace.complex128[N, BS, BS], dTGL: dace.complex128[N]):

    gR = np.ndarray((N, BS, BS), np.complex128)
    gL = np.ndarray((N, BS, BS), np.complex128)
    gG = np.ndarray((N, BS, BS), np.complex128)
    GR = np.ndarray((N, BS, BS), np.complex128)
    GLnd = np.ndarray((N, BS, BS), np.complex128)

    her_gR = np.ndarray((BS, BS), np.complex128)
    her_GR = np.ndarray((BS, BS), np.complex128)
    her_M1 = np.ndarray((BS, BS), np.complex128)
    her_M2 = np.ndarray((BS, BS), np.complex128)

    sigLl = fl * 1j * gammaleft
    sigLr = fr * 1j * gammaright
    # sigLl = fl * gammaleft
    # sigLr = fr * gammaright

    sigGr = 1j * (fr - 1) * gammaright
    sigGl = 1j * (fl - 1) * gammaleft
    # sigGr = fr * gammaright
    # sigGl = fl * gammaleft

    for n in range(N):
        # if HE[n] is not None:
        if n < N - 1:
            HE[n] = HE[n] - sigmaRSE[n]
        else:
            HE[n] = -sigmaRSE[n]
        # if HF[n] is not None:
        if n > 0:
            HF[n] = HF[n] - sigmaRSF[n]
        else:
            HF[n] = -sigmaRSF[n]
        HD[n] = HD[n] - sigmaRSD[n]

    # Solve first the left-connected "small g" retarded green function
    # gR[-1] = np.linalg.inv(HD[-1] - sigRr)
    # gL[-1] = gR[-1] @ (sigmaLSD[-1] + sigLr) @ Her(gR[-1])
    # gG[-1] = gR[-1] @ (sigmaGSD[-1] + sigGr) @ Her(gR[-1])
    gR[N - 1] = HD[N - 1] - sigRr
    # Her(gR[-1], her_gR)
    hermitian_transpose(gR[N - 1], her_gR)
    gL[N - 1] = gR[N - 1] @ (sigmaLSD[N - 1] + sigLr) @ her_gR
    gG[N - 1] = gR[N - 1] @ (sigmaGSD[N - 1] + sigGr) @ her_gR

    for n in range(N - 2, -1, -1):
        sig = HF[n] @ gR[n + 1] @ HE[n + 1]
        M = HF[n] @ gR[n + 1] @ sigmaLSE[n + 1]
        hermitian_transpose(M, her_M1)
        # gR[n] = np.linalg.inv(HD[n] - sig)
        # gL[n] = gR[n] @ (HF[n] @ gL[n + 1] @ HE[n + 1] + sigmaLSD[n] - (M - Her(M))) @ Her(gR[n])
        # M = HF[n] @ gR[n + 1] @ sigmaGSE[n + 1]
        # gG[n] = gR[n] @ (HF[n] @ gG[n + 1] @ HE[n + 1] + sigmaGSD[n] - (M - Her(M))) @ Her(gR[n])
        gR[n] = HD[n] - sig
        hermitian_transpose(gR[n], her_gR)
        gL[n] = gR[n] @ (HF[n] @ gL[n + 1] @ HE[n + 1] + sigmaLSD[n] -
                         (M - her_M1)) @ her_gR
        M[:] = HF[n] @ gR[n + 1] @ sigmaGSE[n + 1]
        gG[n] = gR[n] @ (HF[n] @ gG[n + 1] @ HE[n + 1] + sigmaGSD[n] -
                         (M - her_M1)) @ her_gR

    # Solve now the full retarded green function
    M[:] = HF[0] @ gR[1] @ sigmaLSE[1]
    hermitian_transpose(M, her_M1)
    # GR[0] = np.linalg.inv(HD[0] - HF[0] @ gR[1] @ HE[1] - sigRl)
    # GL[0] = GR[0] @ (sigLl + HF[0] @ gL[1] @ HE[1] + sigmaLSD[0] - (M - Her(M))) @ Her(GR[0])
    GR[0] = HD[0] - HF[0] @ gR[1] @ HE[1] - sigRl
    hermitian_transpose(GR[0], her_GR)
    GL[0] = GR[0] @ (sigLl + HF[0] @ gL[1] @ HE[1] + sigmaLSD[0] -
                     (M - her_M1)) @ her_GR

    M[:] = HF[0] @ gR[1] @ sigmaGSE[1]
    hermitian_transpose(M, her_M1)
    # GG[0] = GR[0] @ (sigGl + HF[0] @ gG[1] @ HE[1] + sigmaGSD[0] - (M - Her(M))) @ Her(GR[0])
    # GLnd[0] = -GL[0] @ HF[0] @ Her(gR[1]) - GR[0] @ HF[0] @ gL[1] + GR[0] @ sigmaLSF[0] @ Her(gR[1])
    hermitian_transpose(gR[1], her_gR)
    GG[0] = GR[0] @ (sigGl + HF[0] @ gG[1] @ HE[1] + sigmaGSD[0] -
                     (M - her_M1)) @ her_GR
    GLnd[0] = (-GL[0] @ HF[0] @ her_gR - GR[0] @ HF[0] @ gL[1] +
               GR[0] @ sigmaLSF[0] @ her_gR)

    for n in range(1, N, 1):
        GR[n] = gR[n] + gR[n] @ HE[n] @ GR[n - 1] @ HF[n - 1] @ gR[n]

        hermitian_transpose(gR[n], her_gR)

        # M1 = gR[n] @ HE[n] @ GR[n - 1] @ HF[n - 1] @ gL[n]
        # M2 = gR[n] @ HE[n] @ GR[n - 1] @ sigmaLSF[n - 1] @ Her(gR[n])
        # GL[n] = gL[n] + gR[n] @ HE[n] @ GL[n - 1] @ HF[n - 1] @ Her(gR[n]) + (M1 - Her(M1)) - (M2 - Her(M2))
        M1 = gR[n] @ HE[n] @ GR[n - 1] @ HF[n - 1] @ gL[n]
        M2 = gR[n] @ HE[n] @ GR[n - 1] @ sigmaLSF[n - 1] @ her_gR
        hermitian_transpose(M1, her_M1)
        hermitian_transpose(M2, her_M2)
        GL[n] = gL[n] + gR[n] @ HE[n] @ GL[n - 1] @ HF[n - 1] @ her_gR + (
            M1 - her_M1) - (M2 - her_M2)

        # M1 = gR[n] @ HE[n] @ GR[n - 1] @ HF[n - 1] @ gG[n]
        # M2 = gR[n] @ HE[n] @ GR[n - 1] @ sigmaGSF[n - 1] @ Her(gR[n])
        # GG[n] = gG[n] + gR[n] @ HE[n] @ GG[n - 1] @ HF[n - 1] @ Her(gR[n]) + (M1 - Her(M1)) - (M2 - Her(M2))
        M1[:] = gR[n] @ HE[n] @ GR[n - 1] @ HF[n - 1] @ gG[n]
        M2[:] = gR[n] @ HE[n] @ GR[n - 1] @ sigmaGSF[n - 1] @ her_gR
        hermitian_transpose(M1, her_M1)
        hermitian_transpose(M2, her_M2)
        GG[n] = gG[n] + gR[n] @ HE[n] @ GG[n - 1] @ HF[n - 1] @ her_gR + (
            M1 - her_M1) - (M2 - her_M2)

        if n != N - 1:
            # GLnd[n] = -GL[n] @ HF[n] @ Her(gR[n + 1]) - GR[n] @ HF[n] @ gL[n + 1] + GR[n] @ sigmaLSF[n] @ Her(gR[n + 1])
            hermitian_transpose(gR[n + 1], her_gR)
            GLnd[n] = (-GL[n] @ HF[n] @ her_gR - GR[n] @ HF[n] @ gL[n + 1] +
                       GR[n] @ sigmaLSF[n] @ her_gR)

    for n in range(1, N - 1):
        # dTGL[n] = np.trace(GLnd[n - 1] @ HE[n])
        trace_tmp = GLnd[n - 1] @ HE[n]
        for i in dace.map[0:BS]:
            dTGL[n] += trace_tmp[i, i]

    # dTGL[0] = -np.trace(sigGl @ GL[0] - GG[0] @ sigLl)
    # dTGL[-1] = np.trace(sigGr @ GL[-1] - GG[-1] @ sigLr)
    trace_tmp0 = -sigGl @ GL[0] + GG[0] @ sigLl
    trace_tmp1 = sigGr @ GL[N - 1] + GG[N - 1] @ sigLr
    for i in dace.map[0:BS]:
        dTGL[0] += trace_tmp0[i, i]
        dTGL[N - 1] += trace_tmp1[i, i]


if __name__ == '__main__':

    print("=== Generating SDFG ===")
    sdfg = rgf_dense.to_sdfg(strict=False)
    print("=== Applying dataflow coarsening ===")
    sdfg.coarsen_dataflow()
    print("=== Compiling ===")
    sdfg.compile()
