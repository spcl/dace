import dace
import cupy as cp
import numpy as np
# Transformations
from dace.transformation.interstate import GPUTransformSDFG


M = dace.symbol("M")
N = dace.symbol("N")
K = dace.symbol("K")

_M = 8192//2
_N = _M
_K = _M

A = np.random.rand(_M, _K).astype(np.float32)
B = np.random.rand(_K, _N).astype(np.float32)
A2 = cp.asarray(A, cp.float32)
B2 = cp.asarray(B, cp.float32)
C2 = cp.zeros((_M, _N), cp.float32)


nx, ny, nit = (dace.symbol(s, dace.int32) for s in ('nx', 'ny', 'nit'))


@dace.program
def build_up_b(b: dace.float32[ny, nx], rho: dace.float32, dt: dace.float32,
               u: dace.float32[ny, nx], v: dace.float32[ny, nx], dx: dace.float32,
               dy: dace.float32):

    b[1:-1,
      1:-1] = (rho * (1 / dt * ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) +
                                (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                      ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 - 2 *
                      ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                       (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)) -
                      ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))


@dace.program
def pressure_poisson(p: dace.float32[ny, nx], dx: dace.float32, dy: dace.float32,
                     b: dace.float32[ny, nx]):
    pn = np.empty_like(p)
    pn[:] = p.copy()

    for q in range(nit):
        pn[:] = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 +
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                         (2 * (dx**2 + dy**2)) - dx**2 * dy**2 /
                         (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1])

        p[:, -1] = p[:, -2]  # dp/dx = 0 at x = 2
        p[0, :] = p[1, :]  # dp/dy = 0 at y = 0
        p[:, 0] = p[:, 1]  # dp/dx = 0 at x = 0
        p[-1, :] = 0  # p = 0 at y = 2


@dace.program
def cavity_flow(nt: dace.int32, nit: dace.int32, u: dace.float32[ny, nx],
                v: dace.float32[ny, nx], dt: dace.float32, dx: dace.float32,
                dy: dace.float32, p: dace.float32[ny, nx], rho: dace.float32,
                nu: dace.float32):
    un = np.empty_like(u)
    vn = np.empty_like(v)
    b = np.zeros((ny, nx))

    for n in range(nt):
        un[:] = u.copy()
        vn[:] = v.copy()

        build_up_b(b, rho, dt, u, v, dx, dy)
        pressure_poisson(p, dx, dy, b, nit=nit)

        u[1:-1,
          1:-1] = (
            un[1:-1, 1:-1]
            - un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, 0:-2])
            - vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[0:-2, 1:-1])
            - dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) 
            + nu
            * (
              dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2])
              + dt / dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

        v[1:-1,
          1:-1] = (vn[1:-1, 1:-1] - un[1:-1, 1:-1] * dt / dx *
                   (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                   vn[1:-1, 1:-1] * dt / dy *
                   (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) - dt / (2 * rho * dy) *
                   (p[2:, 1:-1] - p[0:-2, 1:-1]) + nu *
                   (dt / dx**2 *
                    (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                    dt / dy**2 *
                    (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

        u[0, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0
        u[-1, :] = 1  # set velocity on cavity lid equal to 1
        v[0, :] = 0
        v[-1, :] = 0
        v[:, 0] = 0
        v[:, -1] = 0
        
        

if __name__ == '__main__':

    # Prerequisite for sample: CUDA compute capability >= 70
    dace.Config.set('compiler', 'cuda', 'cuda_arch', value='70')

    A = np.random.rand(1024, 1024).astype(np.float16)
    B = np.random.rand(1024, 1024).astype(np.float16)
    C = np.random.rand(1024, 1024).astype(np.float32)

    dace_matmul: dace.SDFG = cavity_flow.to_sdfg()

    # Transform the code to run on the GPU, while ensuring that the warp map
    # in the example runs within a single thread-block.
    dace_matmul.apply_transformations(GPUTransformSDFG, options=dict(sequential_innermaps=False))
    # print(f"\n\ntype of dace: {type(dace_matmul)}\n\n")
    
    # A = torch.tensor(A, device='cuda', dtype=torch.float16)
    # B = torch.tensor(B, device='cuda', dtype=torch.float16)
    # C = torch.tensor(C, device='cuda', dtype=torch.float32)
    
    # torch.cuda.synchronize()

    # benchmark_matmul( dace_matmul, 2048, 2048, 2048, num_iterations=10)
    
    # benchmark_matmul( torch.matmul, 2048, 2048, 2048, num_iterations=10)

    dace_matmul(A=A, B=B, C=C, N=1024)

    diff = np.linalg.norm(A @ B - C) / (1024 * 1024)
    print('Difference:', diff)
    exit(1 if diff > 1e-3 else 0)
