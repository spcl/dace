import numpy as np

N = 5
K = 4
D = 3
FIN = 2
FOUT = 6
P_init = 2


def monet(H, M, sigma, Wa, Wb, b, pseudo):
    P = np.einsum("ijk,kl->ijl", pseudo, Wb) + b  # b implicit broadcast to NxN

    C = P.reshape(N, N, 1, D) - M.reshape(1, 1, K, D)
    S = sigma.reshape(1, 1, K, D)

    G = np.exp(np.sum(C * S, axis=-1, keepdims=True))

    H_prim = np.sum(
        np.max(np.einsum("ij,jkl->ikl", H, Wa).reshape(1, N, K, FOUT) * G, axis=0),
        axis=1,
    )

    return H_prim



def ggcn(H, A, Wa, Wb, Wc, Wd):
    T = A.reshape(N, N, 1) * np.einsum("ij,jk->ik", H, Wd).reshape(1, N, FOUT)

    S = np.einsum("ij,jk->ik", H, Wc).reshape(N, 1, FOUT) + T

    C = np.einsum("ij,jk->ik", H, Wb).reshape(1, N, FOUT)

    H_prim = np.einsum("ij,jk->ik", H, Wa) + np.sum(S * C, axis=1)

    return H_prim

if __name__ == "__main__":
    H = np.arange(FIN * N).reshape(N, FIN)

    Wa = np.arange(FIN * K * FOUT).reshape(FIN, K, FOUT)
    Wb = np.arange(P_init * D).reshape(P_init, D)
    b = np.arange(D)
    M = np.ones((K, D))
    sigma = np.empty((K, D))
    pseudo = np.ones((N, N, P_init))

    H_prim = monet(H, M, sigma, Wa, Wb, b, pseudo)
    print("MoNet shape:", H_prim.shape)

    Wa = np.arange(FIN * FOUT).reshape(FIN, FOUT)
    Wb = np.arange(FIN * FOUT).reshape(FIN, FOUT)
    Wc = np.arange(FIN * FOUT).reshape(FIN, FOUT)
    Wd = np.arange(FIN * FOUT).reshape(FIN, FOUT)
    A = np.random.randint(2, size=(N, N))

    H_prim = ggcn(H, A, Wa, Wb, Wc, Wd)
    print("G-GCN shape:", H_prim.shape)
