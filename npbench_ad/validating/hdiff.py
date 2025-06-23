import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
I, J, K = 32, 32, 32


# Adapted from https://github.com/GridTools/gt4py/blob/1caca893034a18d5df1522ed251486659f846589/tests/test_integration/stencil_definitions.py#L194
@dc.program
def hdiff(in_field: dc.float64[I + 4, J + 4, K], out_field: dc.float64[I, J, K], coeff: dc.float64[I, J, K],
          S: dc.float64[1]):

    lap_field = 4.0 * in_field[1:I + 3, 1:J + 3, :] - (in_field[2:I + 4, 1:J + 3, :] + in_field[0:I + 2, 1:J + 3, :] +
                                                       in_field[1:I + 3, 2:J + 4, :] + in_field[1:I + 3, 0:J + 2, :])

    res1 = lap_field[1:, 1:J + 1, :] - lap_field[:I + 1, 1:J + 1, :]
    flx_field = np.where(
        (res1 * (in_field[2:I + 3, 2:J + 2, :] - in_field[1:I + 2, 2:J + 2, :])) > 0,
        0,
        res1,
    )

    res2 = lap_field[1:I + 1, 1:, :] - lap_field[1:I + 1, :J + 1, :]
    fly_field = np.where(
        (res2 * (in_field[2:I + 2, 2:J + 3, :] - in_field[2:I + 2, 1:J + 2, :])) > 0,
        0,
        res2,
    )

    out_field[:, :, :] = in_field[2:I + 2, 2:J + 2, :] - coeff[:, :, :] * (flx_field[1:, :, :] - flx_field[:-1, :, :] +
                                                                           fly_field[:, 1:, :] - fly_field[:, :-1, :])

    @dc.map(_[0:I, 0:J, 0:K])
    def summap(i, j, k):
        s >> S(1, lambda x, y: x + y)[0]
        z << out_field[i, j, k]
        s = z


sdfg = hdiff.to_sdfg()

sdfg.save("log_sdfgs/hdiff_forward.sdfg")

add_backward_pass(sdfg=sdfg, inputs=["in_field"], outputs=["S"], autooptimize=True
                  )

sdfg.save("log_sdfgs/hdiff_backward.sdfg")
sdfg.compile()


def fwd_initialize(size, datatype=np.float64):
    I, J, K = size
    from numpy.random import default_rng
    rng = default_rng(42)

    # Define arrays
    in_field = rng.random((I + 4, J + 4, K), dtype=datatype)
    out_field = rng.random((I, J, K), dtype=datatype)
    coeff = rng.random((I, J, K), dtype=datatype)
    S = np.zeros(shape=[1], dtype=datatype)
    # Prepare the sizes args dict
    keyword_args = {"I": I, "K": K, "J": J}

    return (in_field, out_field, coeff, S), keyword_args


def bwd_initialize(size, datatype=np.float64):
    I, J, K = size
    fwd_args, fwd_keyword_args = fwd_initialize(size)
    gradient_in_field = np.zeros(shape=(I + 4, J + 4, K), dtype=datatype)
    gradient_S = np.ones(shape=[1], dtype=datatype)

    # Prepare the sizes args dict
    fwd_keyword_args.update({f"gradient_in_field": gradient_in_field, "gradient_S": gradient_S})

    return fwd_args, fwd_keyword_args


fwd_args, fwd_keyword_args = bwd_initialize((I, J, K))
jax_fwd_args = [jnp.array(arg) for arg in fwd_args]
sdfg(*fwd_args, **fwd_keyword_args)
gradient_in_field = fwd_keyword_args["gradient_in_field"]


# JAX Program
def jax_kernel(in_field, out_field, coeff, S):
    I, J, K = out_field.shape[0], out_field.shape[1], out_field.shape[2]
    lap_field = 4.0 * in_field[1:I + 3, 1:J + 3, :] - (in_field[2:I + 4, 1:J + 3, :] + in_field[0:I + 2, 1:J + 3, :] +
                                                       in_field[1:I + 3, 2:J + 4, :] + in_field[1:I + 3, 0:J + 2, :])

    res = lap_field[1:, 1:J + 1, :] - lap_field[:-1, 1:J + 1, :]
    flx_field = jnp.where(
        (res * (in_field[2:I + 3, 2:J + 2, :] - in_field[1:I + 2, 2:J + 2, :])) > 0,
        0,
        res,
    )

    res = lap_field[1:I + 1, 1:, :] - lap_field[1:I + 1, :-1, :]
    fly_field = jnp.where(
        (res * (in_field[2:I + 2, 2:J + 3, :] - in_field[2:I + 2, 1:J + 2, :])) > 0,
        0,
        res,
    )

    out_field = out_field.at[:, :, :].set(
        in_field[2:I + 2, 2:J + 2, :] - coeff[:, :, :] *
        (flx_field[1:, :, :] - flx_field[:-1, :, :] + fly_field[:, 1:, :] - fly_field[:, :-1, :]))

    return jax.block_until_ready(jnp.sum(out_field))


jax_grad = jax.grad(jax_kernel, argnums=(0))

gradient_in_field_jax = jax_grad(*jax_fwd_args)

print(np.max(gradient_in_field - gradient_in_field_jax))
assert np.allclose(gradient_in_field, gradient_in_field_jax)
