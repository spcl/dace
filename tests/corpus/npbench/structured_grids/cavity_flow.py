# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""npbench corpus benchmark: ``cavity_flow`` (structured_grids) -- auto-ported from the npbench repo."""
import numpy as np
import dace as dc

dc_float = dc.float64
dc_complex_float = dc.complex128

SIZES = {'ny': 61, 'nx': 61, 'nt': 25, 'nit': 5, 'rho': 1.0, 'nu': 0.1}
INPUT_ARGS = ('ny', 'nx')
ARRAY_ARGS = ('u', 'v', 'p', 'dx', 'dy', 'dt')
SCALARS = {}
OUTPUT_ARGS = ('u', 'v', 'p')

nx, ny, nit = (dc.symbol(s, dc.int64) for s in ('nx', 'ny', 'nit'))


def initialize(ny, nx, datatype=np.float64):
    u = np.zeros((ny, nx), dtype=datatype)
    v = np.zeros((ny, nx), dtype=datatype)
    p = np.zeros((ny, nx), dtype=datatype)
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    dt = 0.1 / ((nx - 1) * (ny - 1))
    return (u, v, p, dx, dy, dt)


# Numpy reference helpers (distinct names so they don't collide with the dace
# ``@dc.program`` build_up_b/pressure_poisson used by the kernel below -- note the
# dace ``pressure_poisson`` takes ``nit`` from a module symbol, not as an argument).
def build_up_b_np(b, rho, dt, u, v, dx, dy):
    b[1:-1, 1:-1] = (rho * (1 / dt * ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) /
                                      (2 * dy)) - ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 - 2 *
                            ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) * (v[1:-1, 2:] - v[1:-1, 0:-2]) /
                             (2 * dx)) - ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))


def pressure_poisson_np(nit, p, dx, dy, b):
    pn = np.empty_like(p)
    pn = p.copy()
    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 + (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                         (2 * (dx**2 + dy**2)) - dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1])
        p[:, -1] = p[:, -2]
        p[0, :] = p[1, :]
        p[:, 0] = p[:, 1]
        p[-1, :] = 0


def reference(nx, ny, nt, nit, u, v, dt, dx, dy, p, rho, nu):
    un = np.empty_like(u)
    vn = np.empty_like(v)
    b = np.zeros((ny, nx), dtype=np.float64)
    for n in range(nt):
        un = u.copy()
        vn = v.copy()
        build_up_b_np(b, rho, dt, u, v, dx, dy)
        pressure_poisson_np(nit, p, dx, dy, b)
        u[1:-1, 1:-1] = un[1:-1, 1:-1] - un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, 0:-2]) - vn[
            1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[0:-2, 1:-1]) - dt / (2 * rho * dx) * (
                p[1:-1, 2:] - p[1:-1, 0:-2]) + nu * (dt / dx**2 *
                                                     (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) + dt / dy**2 *
                                                     (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]))
        v[1:-1, 1:-1] = vn[1:-1, 1:-1] - un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) - vn[
            1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) - dt / (2 * rho * dy) * (
                p[2:, 1:-1] - p[0:-2, 1:-1]) + nu * (dt / dx**2 *
                                                     (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) + dt / dy**2 *
                                                     (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1]))
        u[0, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0
        u[-1, :] = 1
        v[0, :] = 0
        v[-1, :] = 0
        v[:, 0] = 0
        v[:, -1] = 0


@dc.program
def build_up_b(b: dc_float[ny, nx], rho: dc_float, dt: dc_float, u: dc_float[ny, nx], v: dc_float[ny, nx], dx: dc_float,
               dy: dc_float):
    b[1:-1, 1:-1] = rho * (1 / dt * ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) /
                                     (2 * dy)) - ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 - 2 *
                           ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) * (v[1:-1, 2:] - v[1:-1, 0:-2]) /
                            (2 * dx)) - ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2)


@dc.program
def pressure_poisson(p: dc_float[ny, nx], dx: dc_float, dy: dc_float, b: dc_float[ny, nx]):
    pn = np.empty_like(p)
    pn[:] = p.copy()
    for q in range(nit):
        pn[:] = p.copy()
        p[1:-1, 1:-1] = ((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 + (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) / (
            2 * (dx**2 + dy**2)) - dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1]
        p[:, -1] = p[:, -2]
        p[0, :] = p[1, :]
        p[:, 0] = p[:, 1]
        p[-1, :] = 0


@dc.program
def kernel(nt: dc.int64, u: dc_float[ny, nx], v: dc_float[ny, nx], dt: dc_float, dx: dc_float, dy: dc_float,
           p: dc_float[ny, nx], rho: dc_float, nu: dc_float):
    un = np.empty_like(u)
    vn = np.empty_like(v)
    b = np.zeros((ny, nx), dtype=np.float64)
    for n in range(nt):
        un[:] = u.copy()
        vn[:] = v.copy()
        build_up_b(b, rho, dt, u, v, dx, dy)
        pressure_poisson(p, dx, dy, b, nit=nit)
        u[1:-1, 1:-1] = un[1:-1, 1:-1] - un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, 0:-2]) - vn[
            1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[0:-2, 1:-1]) - dt / (2 * rho * dx) * (
                p[1:-1, 2:] - p[1:-1, 0:-2]) + nu * (dt / dx**2 *
                                                     (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) + dt / dy**2 *
                                                     (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]))
        v[1:-1, 1:-1] = vn[1:-1, 1:-1] - un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) - vn[
            1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) - dt / (2 * rho * dy) * (
                p[2:, 1:-1] - p[0:-2, 1:-1]) + nu * (dt / dx**2 *
                                                     (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) + dt / dy**2 *
                                                     (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1]))
        u[0, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0
        u[-1, :] = 1
        v[0, :] = 0
        v[-1, :] = 0
        v[:, 0] = 0
        v[:, -1] = 0


CORPUS = dict(name='cavity_flow',
              dwarf='structured_grids',
              sizes=SIZES,
              input_args=INPUT_ARGS,
              array_args=ARRAY_ARGS,
              scalars=SCALARS,
              output_args=OUTPUT_ARGS,
              initialize=initialize,
              reference=reference,
              program=kernel)
