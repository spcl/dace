import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.dtypes import DeviceType
N = 32
M = 32


@dc.program
def atax(A: dc.float64[M, N], x: dc.float64[N], B: dc.float64[N], S: dc.float64[1]):

    B[:] = (A @ x) @ A

    S[0] = np.sum(B)


sdfg = atax.to_sdfg()

sdfg.save("log_sdfgs/atax_forward.sdfg")

add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["S"])
sdfg = auto_optimize(sdfg, DeviceType.CPU)
sdfg.save("log_sdfgs/atax_backward_ao.sdfg")
A = np.ones(shape=[M, N])
B = np.ones(shape=[N])
x = np.ones(shape=[N])
gradient_A = np.zeros(shape=[M, N])
gradient_S = np.ones(shape=[1])
S = np.zeros(shape=[1])

sdfg(A, x, B, S, gradient_A=gradient_A, gradient_S=gradient_S)

# JAX
import jax
import jax.numpy as jnp


def k2mm_jax(A, x, B):
    B = (A @ x) @ A
    return jnp.sum(B)


jax_grad = jax.grad(k2mm_jax, argnums=[0])

A = jnp.ones(shape=[M, N])
B = jnp.ones(shape=[N])
x = jnp.ones(shape=[N])

gradient_A_jax = jax_grad(A, x, B)
print(gradient_A)
assert np.allclose(gradient_A_jax, gradient_A)
