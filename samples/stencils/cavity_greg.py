import copy
import numpy as np
import dace as dc
import sys
import dace.transformation
import torch
import numpy as np
import sympy as sp
import dace

# import cupy as cp
import numpy as np

# Transformations
from dace.transformation.interstate import GPUTransformSDFG


# check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# set precision
dace_dtype = dace.float32
torch_dtype = torch.float32

nx, ny, nit = (dace.symbol(s, dace.int32) for s in ("nx", "ny", "nit"))


# @dace.program
def build_up_b(
    b: dace_dtype[ny, nx],
    rho: dace_dtype,
    dt: dace_dtype,
    u: dace_dtype[ny, nx],
    v: dace_dtype[ny, nx],
    dx: dace_dtype,
    dy: dace_dtype,
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


# @dace.program
def pressure_poisson(
    p: dace_dtype[ny, nx],
    dx: dace_dtype,
    dy: dace_dtype,
    b: dace_dtype[ny, nx],
    nit: dace.int32,
):
    pn = torch.empty_like(p)
    pn[:] = p.clone()

    for q in range(nit):
        pn[:] = p.clone()
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


# @dace.program
def cavity_flow(
    nt: dace.int32,
    nit: dace.int32,
    u: dace_dtype[ny, nx],
    v: dace_dtype[ny, nx],
    dt: dace_dtype,
    dx: dace_dtype,
    dy: dace_dtype,
    p: dace_dtype[ny, nx],
    rho: dace_dtype,
    nu: dace_dtype,
):
    nx, ny = u.shape
    un = torch.empty_like(u)
    vn = torch.empty_like(v)
    b = torch.zeros((nx, ny), dtype=torch_dtype, device=u.device)

    for n in range(nt):
        un[:] = u.clone()
        vn[:] = v.clone()

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


# @dace.program
def cavity_flow_explicit(
    nt: dace.int32,
    nit: dace.int32,
    u: dace_dtype[ny, nx],
    v: dace_dtype[ny, nx],
    dt: dace_dtype,
    dx: dace_dtype,
    dy: dace_dtype,
    p: dace_dtype[ny, nx],
    rho: dace_dtype,
    nu: dace_dtype,
):
    # un = torch.empty_like(u)
    # vn = torch.empty_like(v)
    un = torch.empty_like(u)
    vn = torch.empty_like(v)
    pn = torch.empty_like(p)
    nx, ny = u.shape
    b = torch.zeros((nx, ny))
    # b = torch.zeros_like(u)

    # tile 1
    for n in range(nt):
        # u is a torch.tensor.
        # copy values of u to un
        un[:] = u.clone()
        vn[:] = v.clone()
        # un = u
        # vn = v

        # tile (9*S + 64)^(1/2)/9 - 8/9
        for i in range(1, nx - 1):
            # tile (9*S + 64)^(1/2)/9 - 8/9
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

        # tile (9*S + 64)^(1/2)/6 + 2/3
        for q in range(nit):
            pn[:] = p.clone()
            # pn = p
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

        # tile (9*S + 64)^(1/2)/9 - 8/9
        for i in range(1, nx - 1):
            # tile (9*S + 64)^(1/2)/9 - 8/9
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


# TC_SIZE_M = 16
# TC_SIZE_N = 16

# threadblock tiling
TILE_SHMEM_M = 32
TILE_SHMEM_N = 32
# TILE_SHMEM_K = 32
TILE_SHMEM_T1 = 1
TILE_SHMEM_T2 = 1
HALO_SIZE = TILE_SHMEM_T1 + TILE_SHMEM_T2 - 1

# register tiling
TILE_REG_M = 4
TILE_REG_N = 4

WARPSIZE = 32
# NUM_WARPS = TILE_SHMEM_M // TILE_REG_M
NUM_ACCUMULATORS = TILE_SHMEM_N // TILE_REG_N


@dace.program
def cavity_flow_explicit_gpu(
    nt: dace.int32,
    nit: dace.int32,
    u: dace_dtype[nx, ny] @ dace.StorageType.GPU_Global,
    v: dace_dtype[nx, ny] @ dace.StorageType.GPU_Global,
    dt: dace_dtype,
    dx: dace_dtype,
    dy: dace_dtype,
    p: dace_dtype[nx, ny] @ dace.StorageType.GPU_Global,
    rho: dace_dtype,
    nu: dace_dtype,
):
    # un = torch.empty_like(u)
    # vn = torch.empty_like(v)
    # un = torch.empty_like(u)
    # vn = torch.empty_like(v)
    # pn = torch.empty_like(p)
    # nx, ny = u.shape
    # b = torch.zeros((nx, ny))
    # b = torch.zeros_like(u)

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # --------- THREADBLOCK SCHEDULE --------------
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    for t_shmem_m, t_shmem_n in (
        dace.map[0:nx:TILE_SHMEM_M, 0:ny:TILE_SHMEM_N] @ dace.ScheduleType.GPU_Device
    ):
        # ------------------------------------
        # ---- SHARED MEMORY ALLOCATION ------
        # ------------------------------------

        u_shmem = dace.ndarray(
            [TILE_SHMEM_M + 2 * TILE_SHMEM_T1, TILE_SHMEM_N + 2 * TILE_SHMEM_T1],
            dtype=dace_dtype,
            storage=dace.StorageType.GPU_Shared,
        )
        v_shmem = dace.ndarray(
            [TILE_SHMEM_M + 2 * TILE_SHMEM_T1, TILE_SHMEM_N + 2 * TILE_SHMEM_T1],
            dtype=dace_dtype,
            storage=dace.StorageType.GPU_Shared,
        )
        p_shmem = dace.ndarray(
            [TILE_SHMEM_M + 2 * HALO_SIZE, TILE_SHMEM_N + 2 * HALO_SIZE],
            dtype=dace_dtype,
            storage=dace.StorageType.GPU_Shared,
        )
        b_shmem = dace.ndarray(
            [TILE_SHMEM_M + 2 * HALO_SIZE, TILE_SHMEM_N + 2 * HALO_SIZE],
            dtype=dace_dtype,
            storage=dace.StorageType.GPU_Shared,
        )

        # double buffering
        un_shmem = dace.ndarray(
            [TILE_SHMEM_M + 2 * TILE_SHMEM_T1, TILE_SHMEM_N + 2 * TILE_SHMEM_T1],
            dtype=dace_dtype,
            storage=dace.StorageType.GPU_Shared,
        )
        vn_shmem = dace.ndarray(
            [TILE_SHMEM_M + 2 * TILE_SHMEM_T1, TILE_SHMEM_N + 2 * TILE_SHMEM_T1],
            dtype=dace_dtype,
            storage=dace.StorageType.GPU_Shared,
        )
        pn_shmem = dace.ndarray(
            [TILE_SHMEM_M + 2 * HALO_SIZE, TILE_SHMEM_N + 2 * HALO_SIZE],
            dtype=dace_dtype,
            storage=dace.StorageType.GPU_Shared,
        )
        bn_shmem = dace.ndarray(
            [TILE_SHMEM_M + 2 * HALO_SIZE, TILE_SHMEM_N + 2 * HALO_SIZE],
            dtype=dace_dtype,
            storage=dace.StorageType.GPU_Shared,
        )

        # ------------------------------------
        # ---- REGISTER ALLOCATION ------
        # ------------------------------------

        u_reg = dace.ndarray(
            [TILE_REG_M + 2, TILE_REG_N + 2],
            dtype=dace_dtype,
            storage=dace.StorageType.Register,
        )
        v_reg = dace.ndarray(
            [TILE_REG_M + 2, TILE_REG_N + 2],
            dtype=dace_dtype,
            storage=dace.StorageType.Register,
        )
        p_reg = dace.ndarray(
            [TILE_REG_M + 2, TILE_REG_N + 2],
            dtype=dace_dtype,
            storage=dace.StorageType.Register,
        )
        # p_reg = dace.ndarray(
        #     [TILE_REG_M + HALO_SIZE + 1, TILE_REG_N + HALO_SIZE + 1],
        #     dtype=dace_dtype,
        #     storage=dace.StorageType.Register,
        # )
        b_reg = dace.ndarray(
            [TILE_REG_M, TILE_REG_N],
            dtype=dace_dtype,
            storage=dace.StorageType.Register,
        )

        # double buffering
        un_reg = dace.ndarray(
            [TILE_REG_M + 2, TILE_REG_N + 2],
            dtype=dace_dtype,
            storage=dace.StorageType.Register,
        )
        vn_reg = dace.ndarray(
            [TILE_REG_M + 2, TILE_REG_N + 2],
            dtype=dace_dtype,
            storage=dace.StorageType.Register,
        )
        pn_reg = dace.ndarray(
            [TILE_REG_M + 2, TILE_REG_N + 2],
            dtype=dace_dtype,
            storage=dace.StorageType.Register,
        )
        # pn_reg = dace.ndarray(
        #     [TILE_REG_M + HALO_SIZE + 1, TILE_REG_N + HALO_SIZE + 1],
        #     dtype=dace_dtype,
        #     storage=dace.StorageType.Register,
        # )

        # ------------------------------------
        # load the tiles from global to shared
        # ------------------------------------
        un_shmem[1:-1, 1:-1] = u[
            t_shmem_m : t_shmem_m + TILE_SHMEM_M, t_shmem_n : t_shmem_n + TILE_SHMEM_N
        ]
        vn_shmem[1:-1, 1:-1] = v[
            t_shmem_m : t_shmem_m + TILE_SHMEM_M, t_shmem_n : t_shmem_n + TILE_SHMEM_N
        ]
        pn_shmem[1:-1, 1:-1] = p[
            t_shmem_m : t_shmem_m + TILE_SHMEM_M, t_shmem_n : t_shmem_n + TILE_SHMEM_N
        ]
        # un_shmem[:] = u[
        #     max(t_shmem_m - TILE_SHMEM_T1, 0) : min(
        #         t_shmem_m + TILE_SHMEM_M + TILE_SHMEM_T1, nx
        #     ),
        #     max(t_shmem_n - TILE_SHMEM_T1, 0) : min(
        #         t_shmem_n + TILE_SHMEM_K + TILE_SHMEM_T2, ny
        #     ),
        # ]
        # vn_shmem[:] = v[
        #     max(t_shmem_m - TILE_SHMEM_T1, 0) : min(
        #         t_shmem_m + TILE_SHMEM_M + TILE_SHMEM_T1, nx
        #     ),
        #     max(t_shmem_n - TILE_SHMEM_T1, 0) : min(
        #         t_shmem_n + TILE_SHMEM_K + TILE_SHMEM_T2, ny
        #     ),
        # ]
        # pn_shmem[:] = p[
        #     max(t_shmem_m - TILE_SHMEM_T1, 0) : min(
        #         t_shmem_m + TILE_SHMEM_M + TILE_SHMEM_T1, nx
        #     ),
        #     max(t_shmem_n - TILE_SHMEM_T1, 0) : min(
        #         t_shmem_n + TILE_SHMEM_K + TILE_SHMEM_T2, ny
        #     ),
        # ]

        # __syncthreads();

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # --------- PER-THREAD SCHEDULE --------------
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        for t_tile_m, t_tile_n in (
            dace.map[0:TILE_SHMEM_M:TILE_REG_M, 0:TILE_SHMEM_N:TILE_REG_N]
            @ dace.ScheduleType.GPU_ThreadBlock
        ):

            # ------------------------------------
            # load from shared memory to registers
            # ------------------------------------
            un_reg[:] = un_shmem[
                t_tile_m : t_tile_m + TILE_REG_M + 2,
                t_tile_n : t_tile_n + TILE_REG_N + 2,
            ]
            vn_reg[:] = vn_shmem[
                t_tile_m : t_tile_m + TILE_REG_M + 2,
                t_tile_n : t_tile_n + TILE_REG_N + 2,
            ]
            pn_reg[:] = pn_shmem[
                t_tile_m : t_tile_m + TILE_REG_M + 2,
                t_tile_n : t_tile_n + TILE_REG_N + 2,
            ]
            # vn_reg[i,j] = vn_shmem[t_tile_m + i,t_tile_n + j]
            # pn_reg[i,j] = pn_shmem[t_tile_m + i,t_tile_n + j]

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # MAIN COMPUTATION LOOP
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # ------------------------------------
            # TIME TILING
            # ------------------------------------
            for t_global in dace.map[0:nt:TILE_SHMEM_T1] @ dace.ScheduleType.Sequential:
                # ------------------------------------
                # halo exchange
                # ------------------------------------

                # ------------------------------------
                # store local halo region from registers.
                # If boundary thread, store to global memory.
                # if interior thread, store to shared memory.
                # ------------------------------------

                # north
                if t_tile_m == 0:
                    if t_shmem_m > 0:
                        u[
                            t_shmem_m,
                            t_shmem_n + t_tile_m : t_shmem_n + t_tile_m + TILE_REG_N,
                        ] = u_reg[1, 1 : TILE_REG_N + 1]
                        v[
                            t_shmem_m,
                            t_shmem_n + t_tile_m : t_shmem_n + t_tile_m + TILE_REG_N,
                        ] = v_reg[1, 1 : TILE_REG_N + 1]
                        p[
                            t_shmem_m,
                            t_shmem_n + t_tile_m : t_shmem_n + t_tile_m + TILE_REG_N,
                        ] = p_reg[1, 1 : TILE_REG_N + 1]
                else:
                    un_shmem[t_tile_m + 1, t_tile_n + 1 : t_tile_n + TILE_REG_N + 1] = (
                        u_reg[1, 1 : TILE_REG_N + 1]
                    )
                    vn_shmem[t_tile_m + 1, t_tile_n + 1 : t_tile_n + TILE_REG_N + 1] = (
                        v_reg[1, 1 : TILE_REG_N + 1]
                    )
                    pn_shmem[t_tile_m + 1, t_tile_n + 1 : t_tile_n + TILE_REG_N + 1] = (
                        p_reg[1, 1 : TILE_REG_N + 1]
                    )
                # south
                if t_tile_m == TILE_SHMEM_M - TILE_REG_M:
                    if t_shmem_m < nx - TILE_SHMEM_M:
                        u[
                            t_shmem_m + TILE_SHMEM_M,
                            t_shmem_n + t_tile_m : t_shmem_n + t_tile_m + TILE_REG_N,
                        ] = u_reg[TILE_REG_M + 1, 1 : TILE_REG_N + 1]
                        v[
                            t_shmem_m + TILE_SHMEM_M,
                            t_shmem_n + t_tile_m : t_shmem_n + t_tile_m + TILE_REG_N,
                        ] = v_reg[TILE_REG_M + 1, 1 : TILE_REG_N + 1]
                        p[
                            t_shmem_m + TILE_SHMEM_M,
                            t_shmem_n + t_tile_m : t_shmem_n + t_tile_m + TILE_REG_N,
                        ] = p_reg[TILE_REG_M + 1, 1 : TILE_REG_N + 1]
                else:
                    un_shmem[
                        t_tile_m + TILE_REG_M + 1,
                        t_tile_n + 1 : t_tile_n + TILE_REG_N + 1,
                    ] = u_reg[TILE_REG_M + 1, 1 : TILE_REG_N + 1]
                    vn_shmem[
                        t_tile_m + TILE_REG_M + 1,
                        t_tile_n + 1 : t_tile_n + TILE_REG_N + 1,
                    ] = v_reg[TILE_REG_M + 1, 1 : TILE_REG_N + 1]
                    pn_shmem[
                        t_tile_m + TILE_REG_M + 1,
                        t_tile_n + 1 : t_tile_n + TILE_REG_N + 1,
                    ] = p_reg[TILE_REG_M + 1, 1 : TILE_REG_N + 1]

                # east
                if t_tile_n == 0:
                    if t_shmem_n > 0:
                        u[
                            t_shmem_m + t_tile_m : t_shmem_m + t_tile_m + TILE_REG_M,
                            t_shmem_n,
                        ] = u_reg[1 : TILE_REG_M + 1, 1]
                        v[
                            t_shmem_m + t_tile_m : t_shmem_m + t_tile_m + TILE_REG_M,
                            t_shmem_n,
                        ] = v_reg[1 : TILE_REG_M + 1, 1]
                        p[
                            t_shmem_m + t_tile_m : t_shmem_m + t_tile_m + TILE_REG_M,
                            t_shmem_n,
                        ] = p_reg[1 : TILE_REG_M + 1, 1]
                else:
                    un_shmem[t_tile_m + 1 : t_tile_m + TILE_REG_M + 1, t_tile_n + 1] = (
                        u_reg[1 : TILE_REG_M + 1, 1]
                    )
                    vn_shmem[t_tile_m + 1 : t_tile_m + TILE_REG_M + 1, t_tile_n + 1] = (
                        v_reg[1 : TILE_REG_M + 1, 1]
                    )
                    pn_shmem[t_tile_m + 1 : t_tile_m + TILE_REG_M + 1, t_tile_n + 1] = (
                        p_reg[1 : TILE_REG_M + 1, 1]
                    )

                # west
                if t_tile_n == TILE_SHMEM_N - TILE_REG_N:
                    if t_shmem_n < ny - TILE_SHMEM_N:
                        u[
                            t_shmem_m + t_tile_m : t_shmem_m + t_tile_m + TILE_REG_M,
                            t_shmem_n + TILE_SHMEM_N,
                        ] = u_reg[1 : TILE_REG_M + 1, TILE_REG_N + 1]
                        v[
                            t_shmem_m + t_tile_m : t_shmem_m + t_tile_m + TILE_REG_M,
                            t_shmem_n + TILE_SHMEM_N,
                        ] = v_reg[1 : TILE_REG_M + 1, TILE_REG_N + 1]
                        p[
                            t_shmem_m + t_tile_m : t_shmem_m + t_tile_m + TILE_REG_M,
                            t_shmem_n + TILE_SHMEM_N,
                        ] = p_reg[1 : TILE_REG_M + 1, TILE_REG_N + 1]
                else:
                    un_shmem[
                        t_tile_m + 1 : t_tile_m + TILE_REG_M + 1,
                        t_tile_n + TILE_REG_N + 1,
                    ] = u_reg[1 : TILE_REG_M + 1, TILE_REG_N + 1]
                    vn_shmem[
                        t_tile_m + 1 : t_tile_m + TILE_REG_M + 1,
                        t_tile_n + TILE_REG_N + 1,
                    ] = v_reg[1 : TILE_REG_M + 1, TILE_REG_N + 1]
                    pn_shmem[
                        t_tile_m + 1 : t_tile_m + TILE_REG_M + 1,
                        t_tile_n + TILE_REG_N + 1,
                    ] = p_reg[1 : TILE_REG_M + 1, TILE_REG_N + 1]

                # __syncthreads();

                # ------------------------------------
                # load remote halo region to registers.
                # If boundary thread, load from global memory.
                # if interior thread, load from shared memory.
                # ------------------------------------
                # north
                if t_tile_m == 0:
                    if t_shmem_m > 0:
                        un_reg[0, 1 : TILE_REG_N + 1] = u[
                            t_shmem_m - 1,
                            t_shmem_n + t_tile_m : t_shmem_n + t_tile_m + TILE_REG_N,
                        ]
                        vn_reg[0, 1 : TILE_REG_N + 1] = v[
                            t_shmem_m - 1,
                            t_shmem_n + t_tile_m : t_shmem_n + t_tile_m + TILE_REG_N,
                        ]
                        pn_reg[0, 1 : TILE_REG_N + 1] = p[
                            t_shmem_m - 1,
                            t_shmem_n + t_tile_m : t_shmem_n + t_tile_m + TILE_REG_N,
                        ]
                else:
                    un_reg[0, 1 : TILE_REG_N + 1] = un_shmem[
                        t_tile_m, 1 : TILE_REG_N + 1
                    ]
                    vn_reg[0, 1 : TILE_REG_N + 1] = vn_shmem[
                        t_tile_m, 1 : TILE_REG_N + 1
                    ]
                    pn_reg[0, 1 : TILE_REG_N + 1] = pn_shmem[
                        t_tile_m, 1 : TILE_REG_N + 1
                    ]

                # south
                if t_tile_m == TILE_SHMEM_M - TILE_REG_M:
                    if t_shmem_m < nx - TILE_SHMEM_M:
                        un_reg[TILE_REG_M + 1, 1 : TILE_REG_N + 1] = u[
                            t_shmem_m + TILE_SHMEM_M,
                            t_shmem_n + t_tile_m : t_shmem_n + t_tile_m + TILE_REG_N,
                        ]
                        vn_reg[TILE_REG_M + 1, 1 : TILE_REG_N + 1] = v[
                            t_shmem_m + TILE_SHMEM_M,
                            t_shmem_n + t_tile_m : t_shmem_n + t_tile_m + TILE_REG_N,
                        ]
                        pn_reg[TILE_REG_M + 1, 1 : TILE_REG_N + 1] = p[
                            t_shmem_m + TILE_SHMEM_M,
                            t_shmem_n + t_tile_m : t_shmem_n + t_tile_m + TILE_REG_N,
                        ]
                else:
                    un_reg[TILE_REG_M + 1, 1 : TILE_REG_N + 1] = un_shmem[
                        t_tile_m + TILE_REG_M, 1 : TILE_REG_N + 1
                    ]
                    vn_reg[TILE_REG_M + 1, 1 : TILE_REG_N + 1] = vn_shmem[
                        t_tile_m + TILE_REG_M, 1 : TILE_REG_N + 1
                    ]
                    pn_reg[TILE_REG_M + 1, 1 : TILE_REG_N + 1] = pn_shmem[
                        t_tile_m + TILE_REG_M, 1 : TILE_REG_N + 1
                    ]

                # east
                if t_tile_n == 0:
                    if t_shmem_n > 0:
                        un_reg[1 : TILE_REG_M + 1, 0] = u[
                            t_shmem_m + t_tile_m : t_shmem_m + t_tile_m + TILE_REG_M,
                            t_shmem_n - 1,
                        ]
                        vn_reg[1 : TILE_REG_M + 1, 0] = v[
                            t_shmem_m + t_tile_m : t_shmem_m + t_tile_m + TILE_REG_M,
                            t_shmem_n - 1,
                        ]
                        pn_reg[1 : TILE_REG_M + 1, 0] = p[
                            t_shmem_m + t_tile_m : t_shmem_m + t_tile_m + TILE_REG_M,
                            t_shmem_n - 1,
                        ]
                else:
                    un_reg[1 : TILE_REG_M + 1, 0] = un_shmem[
                        1 : TILE_REG_M + 1, t_tile_n
                    ]
                    vn_reg[1 : TILE_REG_M + 1, 0] = vn_shmem[
                        1 : TILE_REG_M + 1, t_tile_n
                    ]
                    pn_reg[1 : TILE_REG_M + 1, 0] = pn_shmem[
                        1 : TILE_REG_M + 1, t_tile_n
                    ]

                # west
                if t_tile_n == TILE_SHMEM_N - TILE_REG_N:
                    if t_shmem_n < ny - TILE_SHMEM_N:
                        un_reg[1 : TILE_REG_M + 1, TILE_REG_N + 1] = u[
                            t_shmem_m + t_tile_m : t_shmem_m + t_tile_m + TILE_REG_M,
                            t_shmem_n + TILE_SHMEM_N,
                        ]
                        vn_reg[1 : TILE_REG_M + 1, TILE_REG_N + 1] = v[
                            t_shmem_m + t_tile_m : t_shmem_m + t_tile_m + TILE_REG_M,
                            t_shmem_n + TILE_SHMEM_N,
                        ]
                        pn_reg[1 : TILE_REG_M + 1, TILE_REG_N + 1] = p[
                            t_shmem_m + t_tile_m : t_shmem_m + t_tile_m + TILE_REG_M,
                            t_shmem_n + TILE_SHMEM_N,
                        ]
                else:
                    un_reg[1 : TILE_REG_M + 1, TILE_REG_N + 1] = un_shmem[
                        1 : TILE_REG_M + 1, t_tile_n + TILE_REG_N
                    ]
                    vn_reg[1 : TILE_REG_M + 1, TILE_REG_N + 1] = vn_shmem[
                        1 : TILE_REG_M + 1, t_tile_n + TILE_REG_N
                    ]
                    pn_reg[1 : TILE_REG_M + 1, TILE_REG_N + 1] = pn_shmem[
                        1 : TILE_REG_M + 1, t_tile_n + TILE_REG_N
                    ]

                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # FINALLY! DOING SOME COMPUTATION !
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                # ------------------------------------
                # build up b
                # ------------------------------------
                # b_reg[:, :] = un_reg[1:-1, 1:-1]
                b_reg[:, :] = rho * (
                    1
                    / dt
                    * (
                        (un_reg[1:-1, 2:] - un_reg[1:-1, 0:-2]) / (2 * dx)
                        + (vn_reg[2:, 1:-1] - vn_reg[0:-2, 1:-1]) / (2 * dy)
                    )
                    - ((un_reg[1:-1, 2:] - un_reg[1:-1, 0:-2]) / (2 * dx)) ** 2
                    - 2
                    * (
                        (un_reg[2:, 1:-1] - un_reg[0:-2, 1:-1])
                        / (2 * dy)
                        * (vn_reg[1:-1, 2:] - vn_reg[1:-1, 0:-2])
                        / (2 * dx)
                    )
                    - ((vn_reg[2:, 1:-1] - vn_reg[0:-2, 1:-1]) / (2 * dy)) ** 2
                )

                # ------------------------------------
                # pressure poisson equation
                # ------------------------------------
                # TODO: Needs to be fixed! for now it's wrong, no halo exchange
                for _ in range(nit):
                    # p_reg[1:-1, 1:-1] = b_reg[:, :]  # un_reg[1:-1, 1:-1]
                    p_reg[1:-1, 1:-1] = (
                        (pn_reg[1:-1, 2:] + pn_reg[1:-1, 0:-2]) * dy**2
                        + (pn_reg[2:, 1:-1] + pn_reg[0:-2, 1:-1]) * dx**2
                    ) / (2 * (dx**2 + dy**2)) - dx**2 * dy**2 / (
                        2 * (dx**2 + dy**2)
                    ) * b_reg[
                        :, :
                    ]

                    # ------------------------------------
                    # boundary conditions
                    # ------------------------------------

                    # north
                    if t_tile_m == 0 and t_shmem_m == 0:
                        p_reg[0, :] = p_reg[1, :]  # dp/dy = 0 at y = 0

                    # # south
                    # if (
                    #     t_tile_m == TILE_SHMEM_M - TILE_REG_M
                    #     and t_shmem_m == nx - TILE_SHMEM_M
                    # ):
                    #     p_reg[-1, :] = 0  # p_reg = 0 at y = 2

                    # # east
                    # if t_tile_n == 0 and t_shmem_n == 0:
                    #     p_reg[:, 0] = p_reg[:, 1]  # dp/dx = 0 at x = 0

                    # # west
                    # if (
                    #     t_tile_n == TILE_SHMEM_N - TILE_REG_N
                    #     and t_shmem_n == ny - TILE_SHMEM_N
                    # ):
                    #     p_reg[:, -1] = p_reg[:, -2]  # dp/dx = 0 at x = 2

                    pn_reg[:] = p_reg[:]

                # ------------------------------------
                # u and v update
                # ------------------------------------
                u_reg[1:-1, 1:-1] = un_reg[1:-1, 1:-1]
                #     - un_reg[1:-1, 1:-1]
                #     * dt
                #     / dx
                #     * (un_reg[1:-1, 1:-1] - un_reg[1:-1, 0:-2])
                #     - vn_reg[1:-1, 1:-1]
                #     * dt
                #     / dy
                #     * (un_reg[1:-1, 1:-1] - un_reg[0:-2, 1:-1])
                #     - dt / (2 * rho * dx) * (p_reg[1:-1, 2:] - p_reg[1:-1, 0:-2])
                #     + nu
                #     * (
                #         dt
                #         / dx**2
                #         * (
                #             un_reg[1:-1, 2:]
                #             - 2 * un_reg[1:-1, 1:-1]
                #             + un_reg[1:-1, 0:-2]
                #         )
                #         + dt
                #         / dy**2
                #         * (
                #             un_reg[2:, 1:-1]
                #             - 2 * un_reg[1:-1, 1:-1]
                #             + un_reg[0:-2, 1:-1]
                #         )
                #     )
                # )

                v_reg[1:-1, 1:-1] = (
                    vn_reg[1:-1, 1:-1]
                    - un_reg[1:-1, 1:-1]
                    * dt
                    / dx
                    * (vn_reg[1:-1, 1:-1] - vn_reg[1:-1, 0:-2])
                    - vn_reg[1:-1, 1:-1]
                    * dt
                    / dy
                    * (vn_reg[1:-1, 1:-1] - vn_reg[0:-2, 1:-1])
                    - dt / (2 * rho * dy) * (p_reg[2:, 1:-1] - p_reg[0:-2, 1:-1])
                    + nu
                    * (
                        dt
                        / dx**2
                        * (
                            vn_reg[1:-1, 2:]
                            - 2 * vn_reg[1:-1, 1:-1]
                            + vn_reg[1:-1, 0:-2]
                        )
                        + dt
                        / dy**2
                        * (
                            vn_reg[2:, 1:-1]
                            - 2 * vn_reg[1:-1, 1:-1]
                            + vn_reg[0:-2, 1:-1]
                        )
                    )
                )

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # THE END. STORE THE RESULTS FROM REGISTERS BACK TO GLOBAL MEMORY
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            u[
                t_shmem_m + t_tile_m : t_shmem_m + t_tile_m + TILE_REG_M,
                t_shmem_n + t_tile_n : t_shmem_n + t_tile_n + TILE_REG_N,
            ] = u_reg[1:-1, 1:-1]
            v[
                t_shmem_m + t_tile_m : t_shmem_m + t_tile_m + TILE_REG_M,
                t_shmem_n + t_tile_n : t_shmem_n + t_tile_n + TILE_REG_N,
            ] = v_reg[1:-1, 1:-1]
            p[
                t_shmem_m + t_tile_m : t_shmem_m + t_tile_m + TILE_REG_M,
                t_shmem_n + t_tile_n : t_shmem_n + t_tile_n + TILE_REG_N,
            ] = p_reg[1:-1, 1:-1]


def validate_cavity_flow_impls(cavity_flow_dace: dace.SDFG = None):
    # set print options
    np.set_printoptions(
        precision=2, suppress=True, linewidth=3000, threshold=sys.maxsize
    )
    nt = 1
    nit = 1
    nx = 32
    ny = 32

    # u = np.random.rand(nx, ny).astype(np.float32)
    # v = np.random.rand(nx, ny).astype(np.float32)

    u = torch.zeros(nx, ny, dtype=torch_dtype, device=device)
    v = torch.zeros(nx, ny, dtype=torch_dtype, device=device)
    # u[nx // 2, ny // 2] = 1
    # v[nx // 2, ny // 2] = 1
    u[2, 2] = 1
    v[2, 2] = 1
    dt = 1
    dx = 1
    dy = 1
    p = torch.zeros(nx, ny, dtype=torch_dtype, device=device)
    rho = 1
    nu = 1

    u_expl = copy.deepcopy(u)
    v_expl = copy.deepcopy(v)
    p_expl = copy.deepcopy(p)
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

    u_ref = copy.deepcopy(u)
    v_ref = copy.deepcopy(v)
    p_ref = copy.deepcopy(p)
    cavity_flow(
        nt=nt, nit=nit, u=u_ref, v=v_ref, dt=dt, dx=dx, dy=dy, p=p_ref, rho=rho, nu=nu
    )

    if (
        torch.allclose(u_ref, u_expl)
        and torch.allclose(v_ref, v_expl)
        and torch.allclose(p_ref, p_expl)
    ):
        print(f"\nverification successful\n")
    else:
        print(f"\nverification failed\n")
        print(f"u_ref:\n{u_ref}\nu_expl:\n{u_expl}\n")

    if cavity_flow_dace:
        u_dace = copy.deepcopy(u)
        v_dace = copy.deepcopy(v)
        p_dace = copy.deepcopy(p)
        cavity_flow_dace(
            nt=nt,
            nit=nit,
            u=u_dace,
            v=v_dace,
            dt=dt,
            dx=dx,
            dy=dy,
            p=p_dace,
            rho=rho,
            nu=nu,
            nx=nx,
            ny=ny,
        )
        if (
            torch.allclose(u_ref, u_dace)
            and torch.allclose(v_ref, v_dace)
            and torch.allclose(p_ref, p_dace)
        ):
            print(f"\nDace verification successful\n")
        else:
            print(f"\nDace verification failed\n")
            print(
                f"u_ref:\n{p_ref.detach().cpu().numpy()}\nu_dace:\n{p_dace.detach().cpu().numpy()}\n"
            )


if __name__ == "__main__":
    # validate
    # exit()

    sdfg = cavity_flow_explicit_gpu.to_sdfg()
    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated([dace.transformation.dataflow.TaskletFusion])
    # sdfg.apply_transformations_repeated([LoopLifting])
    sdfg.save("tmp.sdfg", hash=False)
    compiled = sdfg.compile()
    validate_cavity_flow_impls(compiled)

    sdfg = dace.SDFG.from_file("tmp.sdfg")
    decomp_params = [("P", 255), ("Ss", 102400)]
    for i in range(10):
        decomp_params.append((f"S{i}", 100))
    decomp_params.append(("TSTEPS", 20))
    decomp_params.append(("dim_m", 20000))
    decomp_params.append(("dim_n", 1000))
    soap_result = perform_soap_analysis_from_sdfg(
        sdfg, decomp_params, generate_schedule=False
    )
    # soap_result.subgraphs[0].get_data_decomposition(0)
    print(soap_result.subgraphs[0].p_grid)
    a = 1
