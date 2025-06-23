import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass

N = 40
M = 50


@dc.program
def bicg(A: dc.float64[N, M], B: dc.float64[M], D: dc.float64[N], p: dc.float64[M], r: dc.float64[N], S: dc.float64[1]):

    B[:], D[:] = r @ A, A @ p

    S[0] = np.sum(D)


sdfg = bicg.to_sdfg()

sdfg.save("log_sdfgs/bicg_forward.sdfg")

add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["S"], autooptimize=True)

sdfg.save("log_sdfgs/bicg_backward.sdfg")

A = np.ones(shape=[N, M])
B = np.ones(shape=[M])
D = np.ones(shape=[N])
p = np.random.rand(M)
r = np.random.rand(N)
gradient_A = np.zeros(shape=[N, M])
gradient_S = np.ones(shape=[1])
S = np.zeros(shape=[1])

sdfg(A, B, D, p, r, S, gradient_A=gradient_A, gradient_S=gradient_S)

# JAX
import jax
import jax.numpy as jnp


def k2mm_jax(A, B, D, p, r):
    B, D = r @ A, A @ p
    return jnp.sum(D)


jax_grad = jax.grad(k2mm_jax, argnums=[0])

A = jnp.ones(shape=[N, M])
B = jnp.ones(shape=[M])
D = jnp.ones(shape=[N])
p = jnp.copy(p)
r = jnp.ones(shape=[N])

gradient_A_jax = jax_grad(A, B, D, p, r)
print(np.max(gradient_A - gradient_A_jax))
assert np.allclose(gradient_A_jax, gradient_A)
