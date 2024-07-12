import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass

N, H, SM = 16, 16, 128


# Numerically-stable version of softmax
@dc.program
def softmax_gpu(x: dc.float64[N, H, SM, SM], out: dc.float64[N, H, SM, SM], S: dc.float64[1]):
    # tmp_max = np.max(x, axis=-1, keepdims=True)
    tmp_max = np.maximum.reduce(x, axis=-1, keepdims=True, initial=-9999)
    tmp_out = np.exp(x - tmp_max)
    # tmp_sum = np.sum(tmp_out, axis=-1, keepdims=True)
    tmp_sum = np.add.reduce(tmp_out, axis=-1, keepdims=True)
    out[:] = tmp_out / tmp_sum

    S[0] = np.sum(out)


sdfg = softmax_gpu.to_sdfg()

sdfg.save("log_sdfgs/softmax_forward.sdfg")

add_backward_pass(sdfg=sdfg, inputs=["x"], outputs=["S"])

sdfg.save("log_sdfgs/softmax_backward.sdfg")
from numpy.random import default_rng

rng = default_rng(42)
x = rng.random((N, H, SM, SM), dtype=np.float64)
out = np.zeros(shape=[N, H, SM, SM], dtype=np.float64)
S = np.zeros(shape=[1])
gradient_S = np.ones(shape=[1])
gradient_x = np.zeros(shape=[N, H, SM, SM])

sdfg(x, out, S, gradient_S=gradient_S, gradient_x=gradient_x)

# print(gradient_x)
# JAX
import jax
import jax.numpy as jnp


def k2mm_jax(x, out):
    tmp_max = jnp.max(x, axis=-1, keepdims=True, initial=-9999)
    tmp_out = jnp.exp(x - tmp_max)
    tmp_sum = jnp.sum(tmp_out, axis=-1, keepdims=True)
    out = tmp_out / tmp_sum
    return jnp.sum(out)


jax_grad = jax.grad(k2mm_jax, argnums=[0])

x = jnp.copy(x)
out = jnp.zeros(shape=[N, H, SM, SM])

gradient_A_jax = jax_grad(x, out)
print(gradient_A_jax - gradient_x)
assert np.allclose(gradient_A_jax, gradient_x)
