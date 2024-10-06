import dace
import cupy as cp
import numpy as np

# Transformations
from dace.transformation.interstate import GPUTransformSDFG
import torch
import triton
import triton.language as tl

M = dace.symbol("M")
N = dace.symbol("N")
K = dace.symbol("K")

_M = 8192 // 2
_N = _M
_K = _M

# A = torch.random.rand(_M, _K).astype(torch.float32)
# B = torch.random.rand(_K, _N).astype(torch.float32)
# A2 = cp.asarray(A, cp.float32)
# B2 = cp.asarray(B, cp.float32)
# C2 = cp.zeros((_M, _N), cp.float32)


nx, ny, nit = (dace.symbol(s, dace.int32) for s in ("nx", "ny", "nit"))


def build_up_b(
    b: dace.float32[ny, nx],
    rho: dace.float32,
    dt: dace.float32,
    u: dace.float32[ny, nx],
    v: dace.float32[ny, nx],
    dx: dace.float32,
    dy: dace.float32,
):

    b[1:-1, 1:-1] = rho * (
        1
        / dt
        * (
            (u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx)
            + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)
        )
        - ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx)) ** 2
        - 2
        * (
            (u[2:, 1:-1] - u[0:-2, 1:-1])
            / (2 * dy)
            * (v[1:-1, 2:] - v[1:-1, 0:-2])
            / (2 * dx)
        )
        - ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) ** 2
    )


def pressure_poisson(
    p: dace.float32[ny, nx],
    dx: dace.float32,
    dy: dace.float32,
    b: dace.float32[ny, nx],
    nit: dace.int32,
):
    # pn = torch.empty_like(p)
    # pn[:] = p.copy()
    # pn = p.clone()

    for q in range(nit):
        pn = p.clone()
        p[1:-1, 1:-1] = (
            (pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2
            + (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2
        ) / (2 * (dx**2 + dy**2)) - dx**2 * dy**2 / (
            2 * (dx**2 + dy**2)
        ) * b[
            1:-1, 1:-1
        ]

        p[:, -1] = p[:, -2]  # dp/dx = 0 at x = 2
        p[0, :] = p[1, :]  # dp/dy = 0 at y = 0
        p[:, 0] = p[:, 1]  # dp/dx = 0 at x = 0
        p[-1, :] = 0  # p = 0 at y = 2


def cavity_flow(
    nt: dace.int32,
    nit: dace.int32,
    u: dace.float32[ny, nx],
    v: dace.float32[ny, nx],
    dt: dace.float32,
    dx: dace.float32,
    dy: dace.float32,
    p: dace.float32[ny, nx],
    rho: dace.float32,
    nu: dace.float32,
):
    # un = torch.empty_like(u)
    # vn = torch.empty_like(v)
    b = torch.zeros((ny, nx))

    for n in range(nt):
        # u is a torch.tensor.
        # copy values of u to un
        un = u.clone()
        vn = v.clone()

        # un[:] = u.copy()
        # vn[:] = v.pcopy()

        build_up_b(b, rho, dt, u, v, dx, dy)
        pressure_poisson(p, dx, dy, b, nit=nit)

        u[1:-1, 1:-1] = (
            un[1:-1, 1:-1]
            - un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, 0:-2])
            - vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[0:-2, 1:-1])
            - dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2])
            + nu
            * (
                dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2])
                + dt / dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])
            )
        )

        v[1:-1, 1:-1] = (
            vn[1:-1, 1:-1]
            - un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2])
            - vn[1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1])
            - dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1])
            + nu
            * (
                dt / dx**2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2])
                + dt / dy**2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])
            )
        )

        u[0, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0
        u[-1, :] = 1  # set velocity on cavity lid equal to 1
        v[0, :] = 0
        v[-1, :] = 0
        v[:, 0] = 0
        v[:, -1] = 0

        print(
            f"un:\n{un.numpy()}\nvn:\n{vn.numpy()}\n u:\n {u.numpy()}\nv:\n {v.numpy()}\n"
        )


def cavity_flow_explicit(
    nt: dace.int32,
    nit: dace.int32,
    u: dace.float32[ny, nx],
    v: dace.float32[ny, nx],
    dt: dace.float32,
    dx: dace.float32,
    dy: dace.float32,
    p: dace.float32[ny, nx],
    rho: dace.float32,
    nu: dace.float32,
):
    # un = torch.empty_like(u)
    # vn = torch.empty_like(v)
    b = torch.zeros((ny, nx))

    for n in range(nt):
        # u is a torch.tensor.
        # copy values of u to un
        un = u.clone()
        vn = v.clone()

        # un[:] = u.copy()
        # vn[:] = v.pcopy()

        # build_up_b(b, rho, dt, u, v, dx, dy)
        # pressure_poisson(p, dx, dy, b, nit=nit)

        # tmp1 = torch.zeros_like(u)
        # tmp2 = torch.zeros_like(v)

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                b[i, j] = rho * (
                    1
                    / dt
                    * (
                        (u[i, j + 1] - u[i, j - 1]) / (2 * dx)
                        + (v[i + 1, j] - v[i - 1, j]) / (2 * dy)
                    )
                    - ((u[i, j + 1] - u[i, j - 1]) / (2 * dx)) ** 2
                    - 2
                    * (
                        (u[i + 1, j] - u[i - 1, j])
                        / (2 * dy)
                        * (v[i, j + 1] - v[i, j - 1])
                        / (2 * dx)
                    )
                    - ((v[i + 1, j] - v[i - 1, j]) / (2 * dy)) ** 2
                )

        for q in range(nit):
            pn = p.clone()
            for i in range(1, nx - 1):
                for j in range(1, ny - 1):
                    p[i, j] = (
                        (pn[i, j + 1] + pn[i, j - 1]) * dy**2
                        + (pn[i + 1, j] + pn[i - 1, j]) * dx**2
                    ) / (2 * (dx**2 + dy**2)) - dx**2 * dy**2 / (
                        2 * (dx**2 + dy**2)
                    ) * b[
                        i, j
                    ]

                    p[:, -1] = p[:, -2]  # dp/dx = 0 at x = 2
                    p[0, :] = p[1, :]  # dp/dy = 0 at y = 0
                    p[:, 0] = p[:, 1]  # dp/dx = 0 at x = 0
                    p[-1, :] = 0  # p = 0 at y = 2

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                u[i][j] = (
                    un[i][j]
                    - un[i][j] * dt / dx * (un[i][j] - un[i][j - 1])
                    - vn[i][j] * dt / dy * (un[i][j] - un[i - 1][j])
                    - dt / (2 * rho * dx) * (p[i][j + 1] - p[i][j - 1])
                    + nu
                    * (
                        dt / dx**2 * (un[i][j + 1] - 2 * un[i][j] + un[i][j - 1])
                        + dt / dy**2 * (un[i + 1][j] - 2 * un[i][j] + un[i - 1][j])
                    )
                )

                v[i][j] = (
                    vn[i][j]
                    - un[i][j] * dt / dx * (vn[i][j] - vn[i][j - 1])
                    - vn[i][j] * dt / dy * (vn[i][j] - vn[i - 1][j])
                    - dt / (2 * rho * dy) * (p[i + 1][j] - p[i - 1][j])
                    + nu
                    * (
                        dt / dx**2 * (vn[i][j + 1] - 2 * vn[i][j] + vn[i][j - 1])
                        + dt / dy**2 * (vn[i + 1][j] - 2 * vn[i][j] + vn[i - 1][j])
                    )
                )

        u[0, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0
        u[-1, :] = 1  # set velocity on cavity lid equal to 1
        v[0, :] = 0
        v[-1, :] = 0
        v[:, 0] = 0
        v[:, -1] = 0
        print(
            f"un:\n{un.numpy()}\nvn:\n{vn.numpy()}\n u:\n {u.numpy()}\nv:\n {v.numpy()}\n"
        )
        
        

def cavity_flow_explicit_tiled_fused(
    nt: dace.int32,
    nit: dace.int32,
    u: dace.float32[ny, nx],
    v: dace.float32[ny, nx],
    dt: dace.float32,
    dx: dace.float32,
    dy: dace.float32,
    p: dace.float32[ny, nx],
    rho: dace.float32,
    nu: dace.float32,
    tile_size_x: dace.int32,
    tile_size_y: dace.int32,
    tile_size_t: dace.int32,
):
    # un = torch.empty_like(u)
    # vn = torch.empty_like(v)
    b = torch.zeros((ny, nx))

    for n in range(nt):
        # u is a torch.tensor.
        # copy values of u to un
        un = u.clone()
        vn = v.clone()

        # un[:] = u.copy()
        # vn[:] = v.pcopy()

        # build_up_b(b, rho, dt, u, v, dx, dy)
        # pressure_poisson(p, dx, dy, b, nit=nit)

        # tmp1 = torch.zeros_like(u)
        # tmp2 = torch.zeros_like(v)

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                b[i, j] = rho * (
                    1
                    / dt
                    * (
                        (u[i, j + 1] - u[i, j - 1]) / (2 * dx)
                        + (v[i + 1, j] - v[i - 1, j]) / (2 * dy)
                    )
                    - ((u[i, j + 1] - u[i, j - 1]) / (2 * dx)) ** 2
                    - 2
                    * (
                        (u[i + 1, j] - u[i - 1, j])
                        / (2 * dy)
                        * (v[i, j + 1] - v[i, j - 1])
                        / (2 * dx)
                    )
                    - ((v[i + 1, j] - v[i - 1, j]) / (2 * dy)) ** 2
                )

        for q in range(nit):
            pn = p.clone()
            for i in range(1, nx - 1):
                for j in range(1, ny - 1):
                    p[i, j] = (
                        (pn[i, j + 1] + pn[i, j - 1]) * dy**2
                        + (pn[i + 1, j] + pn[i - 1, j]) * dx**2
                    ) / (2 * (dx**2 + dy**2)) - dx**2 * dy**2 / (
                        2 * (dx**2 + dy**2)
                    ) * b[
                        i, j
                    ]

                    p[:, -1] = p[:, -2]  # dp/dx = 0 at x = 2
                    p[0, :] = p[1, :]  # dp/dy = 0 at y = 0
                    p[:, 0] = p[:, 1]  # dp/dx = 0 at x = 0
                    p[-1, :] = 0  # p = 0 at y = 2

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                u[i][j] = (
                    un[i][j]
                    - un[i][j] * dt / dx * (un[i][j] - un[i][j - 1])
                    - vn[i][j] * dt / dy * (un[i][j] - un[i - 1][j])
                    - dt / (2 * rho * dx) * (p[i][j + 1] - p[i][j - 1])
                    + nu
                    * (
                        dt / dx**2 * (un[i][j + 1] - 2 * un[i][j] + un[i][j - 1])
                        + dt / dy**2 * (un[i + 1][j] - 2 * un[i][j] + un[i - 1][j])
                    )
                )

                v[i][j] = (
                    vn[i][j]
                    - un[i][j] * dt / dx * (vn[i][j] - vn[i][j - 1])
                    - vn[i][j] * dt / dy * (vn[i][j] - vn[i - 1][j])
                    - dt / (2 * rho * dy) * (p[i + 1][j] - p[i - 1][j])
                    + nu
                    * (
                        dt / dx**2 * (vn[i][j + 1] - 2 * vn[i][j] + vn[i][j - 1])
                        + dt / dy**2 * (vn[i + 1][j] - 2 * vn[i][j] + vn[i - 1][j])
                    )
                )

        u[0, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0
        u[-1, :] = 1  # set velocity on cavity lid equal to 1
        v[0, :] = 0
        v[-1, :] = 0
        v[:, 0] = 0
        v[:, -1] = 0
        print(
            f"un:\n{un.numpy()}\nvn:\n{vn.numpy()}\n u:\n {u.numpy()}\nv:\n {v.numpy()}\n"
        )


import numpy as np

np.set_printoptions(precision=2, suppress=True, linewidth=400)

if __name__ == "__main__":

    # setup parameters
    nt = 2
    nit = 2
    nx = 6
    ny = 6
    # u_def = torch.randn(ny, nx)
    # v_def = torch.randn(ny, nx)
    u_def = torch.ones(ny, nx)
    v_def = torch.ones(ny, nx)
    p_def = torch.zeros(ny, nx)
    u_expl = u_def.clone()
    v_expl = v_def.clone()
    p_expl = p_def.clone()

    u_start = u_def.clone()

    dt = 0.01
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)

    rho = 1
    nu = 0.1
    cavity_flow(
        nt=nt, nit=nit, u=u_def, v=v_def, dt=dt, dx=dx, dy=dy, p=p_def, rho=rho, nu=nu
    )

    cavity_flow_explicit(
        nt=nt,
        nit=nit,
        u=u_expl,
        v=v_expl,
        dt=dt,
        dx=dx,
        dy=dy,
        p=p_expl,
        rho=rho,
        nu=nu,
    )

    if not torch.allclose(u_def, u_expl):
        print("u not equal")
    else:
        print("u equal")
    if not torch.allclose(v_def, v_expl):
        print("v not equal")
    else:
        print("v equal")
    if not torch.allclose(u_start, u_def):
        print("CORRECT: u got updated at all")
    else:
        print("u not updated at all")
    a = 1
