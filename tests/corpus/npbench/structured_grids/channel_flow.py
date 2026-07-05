# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""npbench corpus benchmark: ``channel_flow`` (structured_grids) -- auto-ported from the npbench repo."""
import numpy as np
import dace as dc

dc_float = dc.float64
dc_complex_float = dc.complex128

SIZES = {'ny': 61, 'nx': 61, 'nit': 5, 'rho': 1.0, 'nu': 0.1, 'F': 1.0}
INPUT_ARGS = ('ny', 'nx')
ARRAY_ARGS = ('u', 'v', 'p', 'dx', 'dy', 'dt')
SCALARS = {}
OUTPUT_ARGS = ('u', 'v', 'p')

nx, ny, nit = (dc.symbol(s, dc.int64) for s in ('nx', 'ny', 'nit'))


def initialize(ny, nx, datatype=np.float64):
    u = np.zeros((ny, nx), dtype=datatype)
    v = np.zeros((ny, nx), dtype=datatype)
    p = np.ones((ny, nx), dtype=datatype)
    dx = datatype(2 / (nx - 1))
    dy = datatype(2 / (ny - 1))
    dt = datatype(0.1 / ((nx - 1) * (ny - 1)))
    return (u, v, p, dx, dy, dt)


# Numpy reference helpers (distinct names so they don't collide with the dace
# ``@dc.program`` build_up_b/pressure_poisson_periodic used by the kernel below;
# the dace ``pressure_poisson_periodic`` takes ``nit`` from a module symbol).
def build_up_b_np(rho, dt, dx, dy, u, v):
    b = np.zeros_like(u)
    b[1:-1, 1:-1] = rho * (1 / dt * ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) /
                                     (2 * dy)) - ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 - 2 *
                           ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) * (v[1:-1, 2:] - v[1:-1, 0:-2]) /
                            (2 * dx)) - ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2)
    b[1:-1, -1] = rho * (1 / dt * ((u[1:-1, 0] - u[1:-1, -2]) / (2 * dx) + (v[2:, -1] - v[0:-2, -1]) /
                                   (2 * dy)) - ((u[1:-1, 0] - u[1:-1, -2]) / (2 * dx))**2 - 2 *
                         ((u[2:, -1] - u[0:-2, -1]) / (2 * dy) * (v[1:-1, 0] - v[1:-1, -2]) /
                          (2 * dx)) - ((v[2:, -1] - v[0:-2, -1]) / (2 * dy))**2)
    b[1:-1, 0] = rho * (1 / dt * ((u[1:-1, 1] - u[1:-1, -1]) / (2 * dx) + (v[2:, 0] - v[0:-2, 0]) /
                                  (2 * dy)) - ((u[1:-1, 1] - u[1:-1, -1]) / (2 * dx))**2 - 2 *
                        ((u[2:, 0] - u[0:-2, 0]) / (2 * dy) * (v[1:-1, 1] - v[1:-1, -1]) /
                         (2 * dx)) - ((v[2:, 0] - v[0:-2, 0]) / (2 * dy))**2)
    return b


def pressure_poisson_periodic_np(nit, p, dx, dy, b):
    pn = np.empty_like(p)
    for q in range(nit):
        pn[:] = p.copy()
        p[1:-1, 1:-1] = ((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 + (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) / (
            2 * (dx**2 + dy**2)) - dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1]
        p[1:-1, -1] = ((pn[1:-1, 0] + pn[1:-1, -2]) * dy**2 + (pn[2:, -1] + pn[0:-2, -1]) * dx**2) / (
            2 * (dx**2 + dy**2)) - dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, -1]
        p[1:-1, 0] = ((pn[1:-1, 1] + pn[1:-1, -1]) * dy**2 + (pn[2:, 0] + pn[0:-2, 0]) * dx**2) / (
            2 * (dx**2 + dy**2)) - dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 0]
        p[-1, :] = p[-2, :]
        p[0, :] = p[1, :]


def reference(nit, u, v, dt, dx, dy, p, rho, nu, F):
    udiff = 1
    stepcount = 0
    while udiff > 0.001:
        un = u.copy()
        vn = v.copy()
        b = build_up_b_np(rho, dt, dx, dy, u, v)
        pressure_poisson_periodic_np(nit, p, dx, dy, b)
        u[1:-1, 1:-1] = un[1:-1, 1:-1] - un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, 0:-2]) - vn[
            1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[0:-2, 1:-1]) - dt / (2 * rho * dx) * (
                p[1:-1, 2:] - p[1:-1, 0:-2]) + nu * (dt / dx**2 *
                                                     (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) + dt / dy**2 *
                                                     (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])) + F * dt
        v[1:-1, 1:-1] = vn[1:-1, 1:-1] - un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) - vn[
            1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) - dt / (2 * rho * dy) * (
                p[2:, 1:-1] - p[0:-2, 1:-1]) + nu * (dt / dx**2 *
                                                     (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) + dt / dy**2 *
                                                     (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1]))
        u[1:-1, -1] = un[1:-1, -1] - un[1:-1, -1] * dt / dx * (un[1:-1, -1] - un[1:-1, -2]) - vn[1:-1, -1] * dt / dy * (
            un[1:-1, -1] - un[0:-2, -1]) - dt / (2 * rho * dx) * (p[1:-1, 0] - p[1:-1, -2]) + nu * (
                dt / dx**2 * (un[1:-1, 0] - 2 * un[1:-1, -1] + un[1:-1, -2]) + dt / dy**2 *
                (un[2:, -1] - 2 * un[1:-1, -1] + un[0:-2, -1])) + F * dt
        u[1:-1, 0] = un[1:-1, 0] - un[1:-1, 0] * dt / dx * (un[1:-1, 0] - un[1:-1, -1]) - vn[1:-1, 0] * dt / dy * (
            un[1:-1, 0] - un[0:-2, 0]) - dt / (2 * rho * dx) * (p[1:-1, 1] - p[1:-1, -1]) + nu * (
                dt / dx**2 * (un[1:-1, 1] - 2 * un[1:-1, 0] + un[1:-1, -1]) + dt / dy**2 *
                (un[2:, 0] - 2 * un[1:-1, 0] + un[0:-2, 0])) + F * dt
        v[1:-1, -1] = vn[1:-1, -1] - un[1:-1, -1] * dt / dx * (vn[1:-1, -1] - vn[1:-1, -2]) - vn[1:-1, -1] * dt / dy * (
            vn[1:-1, -1] - vn[0:-2, -1]) - dt / (2 * rho * dy) * (p[2:, -1] - p[0:-2, -1]) + nu * (
                dt / dx**2 * (vn[1:-1, 0] - 2 * vn[1:-1, -1] + vn[1:-1, -2]) + dt / dy**2 *
                (vn[2:, -1] - 2 * vn[1:-1, -1] + vn[0:-2, -1]))
        v[1:-1, 0] = vn[1:-1, 0] - un[1:-1, 0] * dt / dx * (vn[1:-1, 0] - vn[1:-1, -1]) - vn[1:-1, 0] * dt / dy * (
            vn[1:-1, 0] - vn[0:-2, 0]) - dt / (2 * rho * dy) * (p[2:, 0] - p[0:-2, 0]) + nu * (
                dt / dx**2 * (vn[1:-1, 1] - 2 * vn[1:-1, 0] + vn[1:-1, -1]) + dt / dy**2 *
                (vn[2:, 0] - 2 * vn[1:-1, 0] + vn[0:-2, 0]))
        u[0, :] = 0
        u[-1, :] = 0
        v[0, :] = 0
        v[-1, :] = 0
        udiff = (np.sum(u) - np.sum(un)) / np.sum(u)
        stepcount += 1
    # The validated outputs are the in-place-mutated u/v/p (per ``output_args``);
    # stepcount is just a convergence diagnostic, so don't return it (otherwise the
    # output mapping would bind it to ``u``).


@dc.program
def build_up_b(rho: dc_float, dt: dc_float, dx: dc_float, dy: dc_float, u: dc_float[ny, nx], v: dc_float[ny, nx]):
    b = np.zeros_like(u)
    b[1:-1, 1:-1] = rho * (1 / dt * ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) /
                                     (2 * dy)) - ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 - 2 *
                           ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) * (v[1:-1, 2:] - v[1:-1, 0:-2]) /
                            (2 * dx)) - ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2)
    b[1:-1, -1] = rho * (1 / dt * ((u[1:-1, 0] - u[1:-1, -2]) / (2 * dx) + (v[2:, -1] - v[0:-2, -1]) /
                                   (2 * dy)) - ((u[1:-1, 0] - u[1:-1, -2]) / (2 * dx))**2 - 2 *
                         ((u[2:, -1] - u[0:-2, -1]) / (2 * dy) * (v[1:-1, 0] - v[1:-1, -2]) /
                          (2 * dx)) - ((v[2:, -1] - v[0:-2, -1]) / (2 * dy))**2)
    b[1:-1, 0] = rho * (1 / dt * ((u[1:-1, 1] - u[1:-1, -1]) / (2 * dx) + (v[2:, 0] - v[0:-2, 0]) /
                                  (2 * dy)) - ((u[1:-1, 1] - u[1:-1, -1]) / (2 * dx))**2 - 2 *
                        ((u[2:, 0] - u[0:-2, 0]) / (2 * dy) * (v[1:-1, 1] - v[1:-1, -1]) /
                         (2 * dx)) - ((v[2:, 0] - v[0:-2, 0]) / (2 * dy))**2)
    return b


@dc.program
def pressure_poisson_periodic(p: dc_float[ny, nx], dx: dc_float, dy: dc_float, b: dc_float[ny, nx]):
    pn = np.empty_like(p)
    for q in range(nit):
        pn[:] = p.copy()
        p[1:-1, 1:-1] = ((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 + (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) / (
            2 * (dx**2 + dy**2)) - dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1]
        p[1:-1, -1] = (
            (pn[1:-1, 0] + pn[1:-1, -2]) * dy**2 +
            (pn[2:, -1] + pn[0:-2, -1]) * dx**2) / (2 *
                                                    (dx**2 + dy**2)) - dx**2 * dy**2 / (2 *
                                                                                        (dx**2 + dy**2)) * b[1:-1, -1]
        p[1:-1,
          0] = ((pn[1:-1, 1] + pn[1:-1, -1]) * dy**2 +
                (pn[2:, 0] + pn[0:-2, 0]) * dx**2) / (2 *
                                                      (dx**2 + dy**2)) - dx**2 * dy**2 / (2 *
                                                                                          (dx**2 + dy**2)) * b[1:-1, 0]
        p[-1, :] = p[-2, :]
        p[0, :] = p[1, :]


@dc.program
def kernel(u: dc_float[ny, nx], v: dc_float[ny, nx], dt: dc_float, dx: dc_float, dy: dc_float, p: dc_float[ny, nx],
           rho: dc_float, nu: dc_float, F: dc_float):
    udiff = 1.0
    stepcount = 0
    while udiff > 0.001:
        un = u.copy()
        vn = v.copy()
        b = build_up_b(rho, dt, dx, dy, u, v)
        pressure_poisson_periodic(p, dx, dy, b, nit=nit)
        u[1:-1, 1:-1] = un[1:-1, 1:-1] - un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, 0:-2]) - vn[
            1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[0:-2, 1:-1]) - dt / (2 * rho * dx) * (
                p[1:-1, 2:] - p[1:-1, 0:-2]) + nu * (dt / dx**2 *
                                                     (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) + dt / dy**2 *
                                                     (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])) + F * dt
        v[1:-1, 1:-1] = vn[1:-1, 1:-1] - un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) - vn[
            1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) - dt / (2 * rho * dy) * (
                p[2:, 1:-1] - p[0:-2, 1:-1]) + nu * (dt / dx**2 *
                                                     (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) + dt / dy**2 *
                                                     (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1]))
        u[1:-1, -1] = un[1:-1, -1] - un[1:-1, -1] * dt / dx * (un[1:-1, -1] - un[1:-1, -2]) - vn[1:-1, -1] * dt / dy * (
            un[1:-1, -1] - un[0:-2, -1]) - dt / (2 * rho * dx) * (p[1:-1, 0] - p[1:-1, -2]) + nu * (
                dt / dx**2 * (un[1:-1, 0] - 2 * un[1:-1, -1] + un[1:-1, -2]) + dt / dy**2 *
                (un[2:, -1] - 2 * un[1:-1, -1] + un[0:-2, -1])) + F * dt
        u[1:-1, 0] = un[1:-1, 0] - un[1:-1, 0] * dt / dx * (un[1:-1, 0] - un[1:-1, -1]) - vn[1:-1, 0] * dt / dy * (
            un[1:-1, 0] - un[0:-2, 0]) - dt / (2 * rho * dx) * (p[1:-1, 1] - p[1:-1, -1]) + nu * (
                dt / dx**2 * (un[1:-1, 1] - 2 * un[1:-1, 0] + un[1:-1, -1]) + dt / dy**2 *
                (un[2:, 0] - 2 * un[1:-1, 0] + un[0:-2, 0])) + F * dt
        v[1:-1, -1] = vn[1:-1, -1] - un[1:-1, -1] * dt / dx * (vn[1:-1, -1] - vn[1:-1, -2]) - vn[1:-1, -1] * dt / dy * (
            vn[1:-1, -1] - vn[0:-2, -1]) - dt / (2 * rho * dy) * (p[2:, -1] - p[0:-2, -1]) + nu * (
                dt / dx**2 * (vn[1:-1, 0] - 2 * vn[1:-1, -1] + vn[1:-1, -2]) + dt / dy**2 *
                (vn[2:, -1] - 2 * vn[1:-1, -1] + vn[0:-2, -1]))
        v[1:-1, 0] = vn[1:-1, 0] - un[1:-1, 0] * dt / dx * (vn[1:-1, 0] - vn[1:-1, -1]) - vn[1:-1, 0] * dt / dy * (
            vn[1:-1, 0] - vn[0:-2, 0]) - dt / (2 * rho * dy) * (p[2:, 0] - p[0:-2, 0]) + nu * (
                dt / dx**2 * (vn[1:-1, 1] - 2 * vn[1:-1, 0] + vn[1:-1, -1]) + dt / dy**2 *
                (vn[2:, 0] - 2 * vn[1:-1, 0] + vn[0:-2, 0]))
        u[0, :] = 0
        u[-1, :] = 0
        v[0, :] = 0
        v[-1, :] = 0
        udiff = (np.sum(u) - np.sum(un)) / np.sum(u)
        stepcount += 1
    # Outputs are the in-place-mutated u/v/p; do not return stepcount.


CORPUS = dict(name='channel_flow',
              dwarf='structured_grids',
              sizes=SIZES,
              input_args=INPUT_ARGS,
              array_args=ARRAY_ARGS,
              scalars=SCALARS,
              output_args=OUTPUT_ARGS,
              initialize=initialize,
              reference=reference,
              program=kernel)
