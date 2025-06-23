import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass

N = 32
M = 32


@dc.program
def go_fast(a: dc.float64[N, N], S: dc.float64[1]):

    trace = 0.0
    for i in range(N):
        trace = trace + np.tanh(a[i, i])

    D = a + trace

    @dc.map(_[0:N, 0:N])
    def summap(i, j):
        s >> S(1, lambda x, y: x + y)[0]
        z << D[i, j]
        s = z

    return a + trace


sdfg = go_fast.to_sdfg()

sdfg.save("log_sdfgs/go_fast_forward.sdfg")

add_backward_pass(sdfg=sdfg, inputs=["a"], outputs=["S"], autooptimize=True)
# sdfg.save("log_sdfgs/go_fast_backward_bs.sdfg")
# sdfg.simplify()
sdfg.save("log_sdfgs/go_fast_backward.sdfg")


a = np.ones(shape=[N, N])
S = np.zeros(shape=[1])
gradient_S = np.ones(shape=[1])
gradient_a = np.zeros(shape=[N, N])
sdfg(a, S, gradient_S=gradient_S, gradient_a=gradient_a)

# JAX
import jax
import jax.numpy as jnp


def k2mm_jax(a):
    trace = 0.0
    for i in range(N):
        trace += jnp.tanh(a[i, i])
    return jnp.sum(a + trace)


jax_grad = jax.grad(k2mm_jax, argnums=[0])

a = jnp.copy(a)

gradient_A_jax = jax_grad(a)
print(np.max(np.abs(gradient_A_jax - gradient_a)))
assert np.allclose(gradient_A_jax, gradient_a)
