import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass

N = 20


@dc.program
def gemver(alpha: dc.float64, beta: dc.float64, A: dc.float64[N, N], u1: dc.float64[N], v1: dc.float64[N],
           u2: dc.float64[N], v2: dc.float64[N], w: dc.float64[N], x: dc.float64[N], y: dc.float64[N], z: dc.float64[N],
           S: dc.float64[1]):

    
    A += np.multiply.outer(u1, v1) + np.multiply.outer(u2, v2)
    x += beta * y @ A + z
    w += alpha * A @ x

    S[0] = np.sum(w)


sdfg = gemver.to_sdfg()

sdfg.save("log_sdfgs/gemver_forward.sdfg")

add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["S"], autooptimize=True)
sdfg.save("log_sdfgs/gemver_backward.sdfg")
# sdfg.simplify()
alpha = 1.5
beta = 1.2
fn = np.float64(N)
# A = np.ones(shape=[N, N], dtype=np.float64)
# u1 = np.ones(shape=[N], dtype=np.float64)
# u2 = np.ones(shape=[N], dtype=np.float64)
# v1 = np.ones(shape=[N], dtype=np.float64)
# v2 = np.ones(shape=[N], dtype=np.float64)
# x = np.ones(shape=[N], dtype=np.float64)
# y = np.ones(shape=[N], dtype=np.float64)
# w = np.ones(shape=[N], dtype=np.float64)
# z = np.ones(shape=[N], dtype=np.float64)
A = np.fromfunction(lambda i, j: (i * j % N) / N, (N, N), dtype=np.float64)
u1 = np.fromfunction(lambda i: i + 1, (N, ), dtype=np.float64)
u2 = np.fromfunction(lambda i: ((i + 1) / fn) / 2.0, (N, ), dtype=np.float64)
v1 = np.fromfunction(lambda i: ((i + 1) / fn) / 4.0, (N, ), dtype=np.float64)
v2 = np.fromfunction(lambda i: ((i + 1) / fn) / 6.0, (N, ), dtype=np.float64)
w = np.zeros((N, ), dtype=np.float64)
x = np.zeros((N, ), dtype=np.float64)
y = np.fromfunction(lambda i: ((i + 1) / fn) / 8.0, (N, ), dtype=np.float64)
z = np.fromfunction(lambda i: ((i + 1) / fn) / 9.0, (N, ), dtype=np.float64)
gradient_A = np.zeros(shape=[N, N], dtype=np.float64)
gradient_S = np.ones(shape=[1], dtype=np.float64)
S = np.zeros(shape=[1], dtype=np.float64)
# sdfg.simplify()
sdfg.save("log_sdfgs/gemver_backward_S.sdfg")
sdfg(alpha, beta, A, u1, v1, u2, v2, w, x, y, z, S, gradient_A=gradient_A, gradient_S=gradient_S)

# JAX
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

def k2mm_jax(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):
    A += np.multiply.outer(u1, v1) + np.multiply.outer(u2, v2)
    x += beta * y @ A + z 
    w += alpha * A @ x
    return jnp.sum(w)


jax_grad = jax.grad(k2mm_jax, argnums=[2])
# A = jnp.ones(shape=[N, N], dtype=jnp.float64)
# u1 = jnp.ones(shape=[N], dtype=jnp.float64)
# u2 = jnp.ones(shape=[N], dtype=jnp.float64)
# v1 = jnp.ones(shape=[N], dtype=jnp.float64)
# v2 = jnp.ones(shape=[N], dtype=jnp.float64)
# x = jnp.ones(shape=[N], dtype=jnp.float64)
# y = jnp.ones(shape=[N], dtype=jnp.float64)
# w = jnp.ones(shape=[N], dtype=jnp.float64)
# z = jnp.ones(shape=[N], dtype=jnp.float64)
A = jnp.fromfunction(lambda i, j: (i * j % N) / N, (N, N), dtype=jnp.float64)
u1 = jnp.fromfunction(lambda i: i + 1, (N, ), dtype=jnp.float64)
u2 = jnp.fromfunction(lambda i: ((i + 1) / fn) / 2.0, (N, ), dtype=jnp.float64)
v1 = jnp.fromfunction(lambda i: ((i + 1) / fn) / 4.0, (N, ), dtype=jnp.float64)
v2 = jnp.fromfunction(lambda i: ((i + 1) / fn) / 6.0, (N, ), dtype=jnp.float64)
w = jnp.zeros((N, ), dtype=jnp.float64)
x = jnp.zeros((N, ), dtype=jnp.float64)
y = jnp.fromfunction(lambda i: ((i + 1) / fn) / 8.0, (N, ), dtype=jnp.float64)
z = jnp.fromfunction(lambda i: ((i + 1) / fn) / 9.0, (N, ), dtype=jnp.float64)
gradient_A_jax = jax_grad(alpha, beta, A, u1, v1, u2, v2, w, x, y, z)
print(np.max(gradient_A_jax- gradient_A))
assert np.allclose(gradient_A_jax, gradient_A)
