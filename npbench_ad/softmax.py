import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass
from dace.autodiff.optimize_backward_pass_generator import preprocess_fwd_sdfg

# Set NumPy print options
np.set_printoptions(linewidth=100)

N, H, SM = 2, 2, 4


# Numerically-stable version of softmax
@dc.program
def softmax_gpu(x: dc.float32[N, H, SM, SM], out: dc.float32[N, H, SM, SM], S: dc.float32[1]):
    tmp_max = np.maximum.reduce(x, axis=-1, keepdims=True, initial=-9999)
    tmp_out = np.exp(x - tmp_max)
    # tmp_sum = np.sum(tmp_out, axis=-1, keepdims=True)
    tmp_sum = np.add.reduce(tmp_out, axis=-1, keepdims=True)
    out[:] = tmp_out / tmp_sum

    S[0] = np.sum(out)

# sdfg_fwd = softmax_gpu.to_sdfg()
sdfg = softmax_gpu.to_sdfg()

# sdfg.save("log_sdfgs/softmax_forward.sdfg")
# sdfg = dc.SDFG.from_file("log_sdfgs/program.sdfg")
# sdfg.simplify()
# sdfg.simplify()
# preprocess_fwd_sdfg(sdfg)
add_backward_pass(sdfg=sdfg, inputs=["x"], outputs=["S"])

sdfg.save("log_sdfgs/softmax_backward.sdfg")
from numpy.random import default_rng

rng = default_rng(42)
x = rng.random((N, H, SM, SM), dtype=np.float32)
out_d = np.zeros(shape=[N, H, SM, SM], dtype=np.float32)
S = np.zeros(shape=[1], dtype=np.float32)
gradient_S = np.ones(shape=[1], dtype=np.float32)
gradient_x = np.zeros(shape=[N, H, SM, SM], dtype=np.float32)

sdfg(x, out_d, S, gradient_S=gradient_S, gradient_x=gradient_x)

# print(gradient_x)
# JAX
import jax
import jax.numpy as jnp
# jax.config.update("jax_enable_x64", True)

def k2mm_jax(x, out):
    tmp_max = jnp.max(x, axis=-1, keepdims=True, initial=-9999)
    tmp_out = jnp.exp(x - tmp_max)
    tmp_sum = jnp.sum(tmp_out, axis=-1, keepdims=True)
    out = tmp_out / tmp_sum
    return jnp.sum(out)

def k2mm_jax(x, out):
    # tmp_max = jnp.max(x, axis=-1, keepdims=True, initial=-9999)
    # tmp_out = jnp.exp(x - tmp_max)
    # tmp_sum = jnp.sum(tmp_out, axis=-1, keepdims=True)
    # out = tmp_out / tmp_sum
    # return jax.nn.softmax(x, axis=-1)
    return jnp.sum(jax.nn.softmax(x, axis=-1))



jax_grad = jax.grad(k2mm_jax, argnums=[0])
rng = default_rng(42)
x_j = jnp.array(rng.random((N, H, SM, SM), dtype=jnp.float32))
assert np.allclose(x_j, x)

# load sdfg
# sdfg_fwd = dc.SDFG.from_file("log_sdfgs/forward_after_soft.sdfg")

# sdfg_fwd(x, out, S)
# print(out)
# print(np.max(out-jax.nn.softmax(x_j, axis=-1)))
# assert np.allclose(out, k2mm_jax(x, None))

out = jnp.zeros(shape=[N, H, SM, SM])
gradient_A_jax = jax_grad(x, out)
# print(np.max(gradient_A_jax - gradient_x))
# print(gradient_x)
# print(gradient_A_jax)
# assert np.allclose(gradient_A_jax, gradient_x)


def softmax_backward(d_out: np.ndarray, out: np.ndarray):
    """
    Compute the gradient of the softmax function.

    Parameters:
    d_out (np.ndarray): Upstream gradient, same shape as out (N, H, SM, SM).
    out (np.ndarray): Forward softmax output, same shape as d_out.

    Returns:
    d_x (np.ndarray): Gradient of input x, same shape as out.
    """
    d_x = out * (d_out - np.sum(out * d_out, axis=-1, keepdims=True))
    return d_x

print(gradient_A_jax)
dx = softmax_backward(gradient_S, out_d)
print(dx)
assert np.allclose(dx, gradient_A_jax)