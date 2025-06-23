import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass

N = 32
M = 32


@dc.program
def compute(array_1: dc.float64[M, N], array_2: dc.float64[M, N], a: dc.float64, b: dc.float64, B: dc.float64[M, N],
            c: dc.float64, S: dc.float64[1]):

    B[:] = np.minimum(np.maximum(array_1, 2), 10) * a + array_2 * b + c

    @dc.map(_[0:N, 0:M])
    def summap(i, j):
        s >> S(1, lambda x, y: x + y)[0]
        z << B[i, j]
        s = z


sdfg = compute.to_sdfg()

sdfg.save("log_sdfgs/compute_forward.sdfg")

add_backward_pass(sdfg=sdfg, inputs=["array_2"], outputs=["S"])

sdfg.save("log_sdfgs/compute_backward.sdfg")

array_1 = np.ones(shape=[M, N], dtype=np.float64)
array_2 = np.ones(shape=[M, N], dtype=np.float64)
B = np.ones(shape=[M, N], dtype=np.float64)
a = float(10)
b = float(15)
c = float(3)
S = np.zeros(shape=[1], dtype=np.float64)
gradient_S = np.ones(shape=[1], dtype=np.float64)
gradient_A = np.zeros(shape=[N, N], dtype=np.float64)
sdfg(array_1, array_2, a, b, B, c, S, gradient_S=gradient_S, gradient_array_2=gradient_A)

# JAX
import jax
import jax.numpy as jnp


def k2mm_jax(array_1, array_2, a, b, B, c):
    B = np.minimum(np.maximum(array_1, 2), 10) * a + array_2 * b + c
    return jnp.sum(B)


jax_grad = jax.grad(k2mm_jax, argnums=[1])

B = jnp.ones(shape=[M, N])

gradient_A_jax = jax_grad(array_1, array_2, a, b, B, c)
print(gradient_A_jax)
assert np.allclose(gradient_A_jax, gradient_A)
