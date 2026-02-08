# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
# Original application code: NPBench - https://github.com/spcl/npbench
import dace.dtypes
import numpy as np
import dace as dc
import pytest
import argparse
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.autodiff import add_backward_pass

# Sample constants
BET_M = 0.5
BET_P = 0.5

I, J, K = (dc.symbol(s, dtype=dc.int64) for s in ('I', 'J', 'K'))


# Adapted from https://github.com/GridTools/gt4py/blob/1caca893034a18d5df1522ed251486659f846589/tests/test_integration/stencil_definitions.py#L111
@dc.program
def vadv_kernel(utens_stage: dc.float64[I, J, K], u_stage: dc.float64[I, J, K], wcon: dc.float64[I + 1, J, K],
                u_pos: dc.float64[I, J, K], utens: dc.float64[I, J, K], dtr_stage: dc.float64):
    ccol = np.ndarray((I, J, K), dtype=utens_stage.dtype)
    dcol = np.ndarray((I, J, K), dtype=utens_stage.dtype)
    data_col = np.ndarray((I, J), dtype=utens_stage.dtype)

    for k in range(1):
        gcv = 0.25 * (wcon[1:, :, k + 1] + wcon[:-1, :, k + 1])
        cs = gcv * BET_M

        ccol[:, :, k] = gcv * BET_P
        bcol = dtr_stage - ccol[:, :, k]

        # update the d column
        correction_term = -cs * (u_stage[:, :, k + 1] - u_stage[:, :, k])
        dcol[:, :, k] = (dtr_stage * u_pos[:, :, k] + utens[:, :, k] + utens_stage[:, :, k] + correction_term)

        # Thomas forward
        divided = 1.0 / bcol
        ccol[:, :, k] = ccol[:, :, k] * divided
        dcol[:, :, k] = dcol[:, :, k] * divided

    for k in range(1, K - 1):
        gav = -0.25 * (wcon[1:, :, k] + wcon[:-1, :, k])
        gcv[:] = 0.25 * (wcon[1:, :, k + 1] + wcon[:-1, :, k + 1])

        as_ = gav * BET_M
        cs[:] = gcv * BET_M

        acol = gav * BET_P
        ccol[:, :, k] = gcv * BET_P
        bcol[:] = dtr_stage - acol - ccol[:, :, k]

        # update the d column
        # correction_term = -as_ * (u_stage[:, :, k - 1] -
        correction_term[:] = -as_ * (u_stage[:, :, k - 1] - u_stage[:, :, k]) - cs * (u_stage[:, :, k + 1] -
                                                                                      u_stage[:, :, k])
        dcol[:, :, k] = (dtr_stage * u_pos[:, :, k] + utens[:, :, k] + utens_stage[:, :, k] + correction_term)

        # Thomas forward
        # divided = 1.0 / (bcol - ccol[:, :, k - 1] * acol)
        divided[:] = 1.0 / (bcol - ccol[:, :, k - 1] * acol)
        ccol[:, :, k] = ccol[:, :, k] * divided
        dcol[:, :, k] = (dcol[:, :, k] - (dcol[:, :, k - 1]) * acol) * divided

    for k in range(K - 1, K):
        gav[:] = -0.25 * (wcon[1:, :, k] + wcon[:-1, :, k])
        # as_ = gav * BET_M
        as_[:] = gav * BET_M
        # acol = gav * BET_P
        acol[:] = gav * BET_P
        # bcol = dtr_stage - acol
        bcol[:] = dtr_stage - acol

        # update the d column
        # correction_term = -as_ * (u_stage[:, :, k - 1] - u_stage[:, :, k])
        correction_term[:] = -as_ * (u_stage[:, :, k - 1] - u_stage[:, :, k])
        dcol[:, :, k] = (dtr_stage * u_pos[:, :, k] + utens[:, :, k] + utens_stage[:, :, k] + correction_term)

        # Thomas forward
        # divided = 1.0 / (bcol - ccol[:, :, k - 1] * acol)
        divided[:] = 1.0 / (bcol - ccol[:, :, k - 1] * acol)
        dcol[:, :, k] = (dcol[:, :, k] - (dcol[:, :, k - 1]) * acol) * divided

    for k in range(K - 1, K - 2, -1):
        datacol = dcol[:, :, k]
        data_col[:] = datacol
        utens_stage[:, :, k] = dtr_stage * (datacol - u_pos[:, :, k])

    for k in range(K - 2, -1, -1):
        # datacol = dcol[:, :, k] - ccol[:, :, k] * data_col[:, :]
        data_col[:] = dcol[:, :, k] - ccol[:, :, k] * data_col[:, :]
        utens_stage[:, :, k] = dtr_stage * (data_col - u_pos[:, :, k])


def vadv_jax_kernel(jnp, lax, utens_stage, u_stage, wcon, u_pos, utens, dtr_stage):
    I, J, K = utens_stage.shape[0], utens_stage.shape[1], utens_stage.shape[2]
    # Allocate working arrays.
    ccol = jnp.empty((I, J, K), dtype=utens_stage.dtype)
    dcol = jnp.empty((I, J, K), dtype=utens_stage.dtype)
    data_col = jnp.empty((I, J), dtype=utens_stage.dtype)

    # --- Loop 1: for k in range(0, 1) ---
    def loop1_body(carry, k):
        ccol, dcol = carry
        # Note: 0+1 is just 1.
        gcv = 0.25 * (wcon[1:, :, 1] + wcon[:-1, :, 1])
        cs = gcv * BET_M
        bs = gcv * BET_P
        bcol = dtr_stage - bs
        # update the d column correction term.
        correction_term = -cs * (u_stage[:, :, k + 1] - u_stage[:, :, k])
        divided = 1.0 / bcol
        ccol = ccol.at[:, :, k].set(bs * divided)
        dcol = dcol.at[:, :, k].set(
            (dtr_stage * u_pos[:, :, k] + utens[:, :, k] + utens_stage[:, :, k] + correction_term) * divided)
        return (ccol, dcol), None

    (ccol, dcol), _ = lax.scan(loop1_body, (ccol, dcol), jnp.arange(0, 1))

    # --- Loop 2: for k in range(1, K-1) ---
    def loop2_body(carry, k):
        ccol, dcol = carry
        gav = -0.25 * (wcon[1:, :, k] + wcon[:-1, :, k])
        gcv = 0.25 * (wcon[1:, :, k + 1] + wcon[:-1, :, k + 1])
        as_ = gav * BET_M
        cs = gcv * BET_M
        bs = gcv * BET_P
        acol = gav * BET_P
        bcol = dtr_stage - acol - bs
        correction_term = (-as_ * (u_stage[:, :, k - 1] - u_stage[:, :, k]) - cs *
                           (u_stage[:, :, k + 1] - u_stage[:, :, k]))
        divided = 1.0 / (bcol - ccol[:, :, k - 1] * acol)
        ccol = ccol.at[:, :, k].set(bs * divided)
        dcol = dcol.at[:, :, k].set(
            ((dtr_stage * u_pos[:, :, k] + utens[:, :, k] + utens_stage[:, :, k] + correction_term) -
             dcol[:, :, k - 1] * acol) * divided)
        return (ccol, dcol), None

    (ccol, dcol), _ = lax.scan(loop2_body, (ccol, dcol), jnp.arange(1, K - 1))

    # --- Loop 3: for k in range(K-1, K) ---
    def loop3_body(dcol, k):
        gav = -0.25 * (wcon[1:, :, k] + wcon[:-1, :, k])
        as_ = gav * BET_M
        acol = gav * BET_P
        bcol = dtr_stage - acol
        correction_term = -as_ * (u_stage[:, :, k - 1] - u_stage[:, :, k])
        divided = 1.0 / (bcol - ccol[:, :, k - 1] * acol)
        dcol = dcol.at[:, :, k].set(
            ((dtr_stage * u_pos[:, :, k] + utens[:, :, k] + utens_stage[:, :, k] + correction_term) -
             dcol[:, :, k - 1] * acol) * divided)
        return dcol, None

    dcol, _ = lax.scan(loop3_body, dcol, jnp.arange(K - 1, K))

    # --- Loop 4: for k in range(K-1, K) ---
    def loop4_body(carry, k):
        data_col, utens_stage = carry
        datacol = dcol[:, :, k]
        data_col = data_col.at[:].set(datacol)
        utens_stage = utens_stage.at[:, :, k].set(dtr_stage * (datacol - u_pos[:, :, k]))
        return (data_col, utens_stage), None

    (data_col, utens_stage), _ = lax.scan(loop4_body, (data_col, utens_stage), jnp.arange(K - 1, K))

    # --- Loop 5: for k in range(0, K-1) with reverse order ---
    def loop5_body(carry, k):
        data_col, utens_stage = carry
        idx = (K - 2) - k  # Reverse order: when k=0, idx=K-2; when k=K-2, idx=0.
        datacol = dcol[:, :, idx] - ccol[:, :, idx] * data_col[:, :]
        data_col = data_col.at[:].set(datacol)
        utens_stage = utens_stage.at[:, :, idx].set(dtr_stage * (datacol - u_pos[:, :, idx]))
        return (data_col, utens_stage), None

    (data_col, utens_stage), _ = lax.scan(loop5_body, (data_col, utens_stage), jnp.arange(0, K - 1))
    return jnp.sum(utens_stage)


def initialize(I, J, K):
    from numpy.random import default_rng
    rng = default_rng(42)

    dtr_stage = 3. / 20.

    # Define arrays
    utens_stage = rng.random((I, J, K))
    u_stage = rng.random((I, J, K))
    wcon = rng.random((I + 1, J, K))
    u_pos = rng.random((I, J, K))
    utens = rng.random((I, J, K))

    return dtr_stage, utens_stage, u_stage, wcon, u_pos, utens


# Adapted from https://github.com/GridTools/gt4py/blob/1caca893034a18d5df1522ed251486659f846589/tests/test_integration/stencil_definitions.py#L111
def ground_truth(utens_stage, u_stage, wcon, u_pos, utens, dtr_stage):
    I, J, K = utens_stage.shape[0], utens_stage.shape[1], utens_stage.shape[2]
    ccol = np.ndarray((I, J, K), dtype=utens_stage.dtype)
    dcol = np.ndarray((I, J, K), dtype=utens_stage.dtype)
    data_col = np.ndarray((I, J), dtype=utens_stage.dtype)

    for k in range(1):
        gcv = 0.25 * (wcon[1:, :, k + 1] + wcon[:-1, :, k + 1])
        cs = gcv * BET_M

        ccol[:, :, k] = gcv * BET_P
        bcol = dtr_stage - ccol[:, :, k]

        # update the d column
        correction_term = -cs * (u_stage[:, :, k + 1] - u_stage[:, :, k])
        dcol[:, :, k] = (dtr_stage * u_pos[:, :, k] + utens[:, :, k] + utens_stage[:, :, k] + correction_term)

        # Thomas forward
        divided = 1.0 / bcol
        ccol[:, :, k] = ccol[:, :, k] * divided
        dcol[:, :, k] = dcol[:, :, k] * divided

    for k in range(1, K - 1):
        gav = -0.25 * (wcon[1:, :, k] + wcon[:-1, :, k])
        gcv = 0.25 * (wcon[1:, :, k + 1] + wcon[:-1, :, k + 1])

        as_ = gav * BET_M
        cs = gcv * BET_M

        acol = gav * BET_P
        ccol[:, :, k] = gcv * BET_P
        bcol = dtr_stage - acol - ccol[:, :, k]

        # update the d column
        correction_term = -as_ * (u_stage[:, :, k - 1] - u_stage[:, :, k]) - cs * (u_stage[:, :, k + 1] -
                                                                                   u_stage[:, :, k])
        dcol[:, :, k] = (dtr_stage * u_pos[:, :, k] + utens[:, :, k] + utens_stage[:, :, k] + correction_term)

        # Thomas forward
        divided = 1.0 / (bcol - ccol[:, :, k - 1] * acol)
        ccol[:, :, k] = ccol[:, :, k] * divided
        dcol[:, :, k] = (dcol[:, :, k] - (dcol[:, :, k - 1]) * acol) * divided

    for k in range(K - 1, K):
        gav = -0.25 * (wcon[1:, :, k] + wcon[:-1, :, k])
        as_ = gav * BET_M
        acol = gav * BET_P
        bcol = dtr_stage - acol

        # update the d column
        correction_term = -as_ * (u_stage[:, :, k - 1] - u_stage[:, :, k])
        dcol[:, :, k] = (dtr_stage * u_pos[:, :, k] + utens[:, :, k] + utens_stage[:, :, k] + correction_term)

        # Thomas forward
        divided = 1.0 / (bcol - ccol[:, :, k - 1] * acol)
        dcol[:, :, k] = (dcol[:, :, k] - (dcol[:, :, k - 1]) * acol) * divided

    for k in range(K - 1, K - 2, -1):
        datacol = dcol[:, :, k]
        data_col[:] = datacol
        utens_stage[:, :, k] = dtr_stage * (datacol - u_pos[:, :, k])

    for k in range(K - 2, -1, -1):
        datacol = dcol[:, :, k] - ccol[:, :, k] * data_col[:, :]
        data_col[:] = datacol
        utens_stage[:, :, k] = dtr_stage * (datacol - u_pos[:, :, k])


def run_vadv(device_type: dace.dtypes.DeviceType):
    '''
    Runs VAdv for the given device
    :return: the SDFG
    '''

    # Initialize data (npbench small size)
    I, J, K = 60, 60, 40
    dtr_stage, utens_stage, u_stage, wcon, u_pos, utens = initialize(I, J, K)
    utens_stage_ref = np.copy(utens_stage)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = vadv_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        sdfg(utens_stage, u_stage, wcon, u_pos, utens, dtr_stage, I=I, J=J, K=K)

    # Compute ground truth and validate
    ground_truth(utens_stage_ref, u_stage, wcon, u_pos, utens, dtr_stage)
    assert np.allclose(utens_stage, utens_stage_ref)
    return sdfg


def run_vadv_autodiff():
    import jax
    import jax.numpy as jnp
    import jax.lax as lax

    # Initialize data (npbench small size)
    I, J, K = 4, 4, 3
    dtr_stage, utens_stage, u_stage, wcon, u_pos, utens = initialize(I, J, K)
    dtr_stage_jax, utens_stage_jax, u_stage_jax, wcon_jax, u_pos_jax, utens_jax = [
        np.copy(arr) for arr in (dtr_stage, utens_stage, u_stage, wcon, u_pos, utens)
    ]

    # Initialize gradient computation data
    gradient_utens = np.zeros_like(utens)
    gradient___return = np.ones((1, ), dtype=np.float64)

    # Define sum reduction for the output
    @dc.program
    def autodiff_kernel(utens_stage: dc.float64[I, J, K], u_stage: dc.float64[I, J, K], wcon: dc.float64[I + 1, J, K],
                        u_pos: dc.float64[I, J, K], utens: dc.float64[I, J, K], dtr_stage: dc.float64):
        vadv_kernel(utens_stage, u_stage, wcon, u_pos, utens, dtr_stage)
        return np.sum(utens_stage)

    # Add the backward pass to the SDFG
    sdfg = autodiff_kernel.to_sdfg()
    add_backward_pass(sdfg=sdfg, inputs=["utens"], outputs=["__return"])
    sdfg(utens_stage,
         u_stage,
         wcon,
         u_pos,
         utens,
         dtr_stage,
         I=I,
         J=J,
         K=K,
         gradient_utens=gradient_utens,
         gradient___return=gradient___return)

    # Enable float64 support
    jax.config.update("jax_enable_x64", True)

    # Numerically validate vs JAX
    jax_kernel = lambda utens_stage, u_stage, wcon, u_pos, utens, dtr_stage: vadv_jax_kernel(
        jnp, lax, utens_stage, u_stage, wcon, u_pos, utens, dtr_stage)
    jax_grad = jax.jit(jax.grad(jax_kernel, argnums=4))
    jax_grad_utens = jax_grad(utens_stage_jax, u_stage_jax, wcon_jax, u_pos_jax, utens_jax, dtr_stage_jax)
    np.testing.assert_allclose(gradient_utens, jax_grad_utens)


def test_cpu(monkeypatch):
    # NOTE: Serialization fails because of "k - k" expression simplified to "0"
    monkeypatch.setenv("DACE_testing_serialization", "0")
    run_vadv(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_vadv(dace.dtypes.DeviceType.GPU)


@pytest.mark.autodiff
def test_autodiff():
    pytest.importorskip("jax", reason="jax not installed. Please install with: pip install dace[ml-testing]")
    run_vadv_autodiff()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_vadv(dace.dtypes.DeviceType.CPU)
        run_vadv_autodiff()
    elif target == "gpu":
        run_vadv(dace.dtypes.DeviceType.GPU)
