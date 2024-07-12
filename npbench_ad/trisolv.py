import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass

M = 32
N = 32


@dc.program
def trisolv(L: dc.float64[N, N], x: dc.float64[N], b: dc.float64[N], S: dc.float64[1]):

    for i in range(N):
        x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]

    @dc.map(_[0:N])
    def summap(i):
        s >> S(1, lambda x, y: x + y)[0]
        z << x[i]
        s = z


alpha = 0.2
beta = 1.2

L = np.ones(shape=[N, N])
x = np.ones(shape=[N])
b = np.ones(shape=[N])
S = np.zeros(shape=[1])
gradient_S = np.ones(shape=[1])
gradient_L = np.zeros(shape=[M, N])
sdfg = trisolv.to_sdfg()

sdfg.save("log_sdfgs/trisolv_forward.sdfg")
sdfg(L, x, b, S)

add_backward_pass(sdfg=sdfg, inputs=["L"], outputs=["S"])

sdfg.save("log_sdfgs/trisolv_backward.sdfg")

sdfg(L, x, b, S, gradient_L=gradient_L, gradient_S=gradient_S)

# JAX
import jax
import jax.numpy as jnp


def k2mm_jax(L, x, b):
    for i in range(N):
        x = x.at[i].set((b[i] - L[i, :i] @ x[:i]) / L[i, i])
    return jnp.sum(x)


jax_grad = jax.grad(k2mm_jax, argnums=[0])

L = jnp.ones(shape=[M, M])
x = jnp.ones(shape=[N])
b = jnp.ones(shape=[N])

gradient_A_jax = jax_grad(L, x, b)
print(gradient_A_jax[0])
print(gradient_L)
assert np.allclose(gradient_A_jax[0], gradient_L)
