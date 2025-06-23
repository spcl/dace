import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass
from jax import numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)
N = 6
M = 7


@dc.program
def gramschmidt(A: dc.float64[M, N], S: dc.float64[1]):

    Q = np.zeros_like(A)
    R = np.zeros((N, N), dtype=A.dtype)

    for k in range(N):
        nrm = np.dot(A[:, k], A[:, k])
        R[k, k] = np.sqrt(nrm)
        Q[:, k] = A[:, k] / R[k, k]
        for j in range(k + 1, N):
            R[k, j] = np.dot(Q[:, k], A[:, j])
            A[:, j] -= Q[:, k] * R[k, j]

    @dc.map(_[0:M, 0:N])
    def summap(i, j):
        s >> S(1, lambda x, y: x + y)[0]
        z << A[i, j]
        s = z


sdfg = gramschmidt.to_sdfg()

sdfg.save("log_sdfgs/gramschmidt_forward.sdfg")

add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["S"])

sdfg.save("log_sdfgs/gramschmidt_backward.sdfg")

sdfg.compile()


def fwd_initialize(size, datatype=np.float64):
    M, N = size
    from numpy.random import default_rng
    rng = default_rng(42)

    A = rng.random((M, N), dtype=datatype)
    while np.linalg.matrix_rank(A) < N:
        A = rng.random((M, N), dtype=datatype)
    S_ = np.zeros(shape=[1], dtype=datatype)

    # Prepare the sizes args dict
    keyword_args = {"N": N, "M": M}
    return (A, S_), keyword_args


(A, S_), keyword_args = fwd_initialize((M, N))
S = np.zeros(shape=[1], dtype=np.float64)
gradient_A = np.zeros(shape=(M, N), dtype=np.float64)
gradient_S = np.ones(shape=[1], dtype=np.float64)

sdfg(A=A, S=S, gradient_A=gradient_A, gradient_S=gradient_S)


# JAX Program
def jax_kernel(A, S):

    Q = jnp.zeros_like(A)
    R = jnp.zeros((A.shape[1], A.shape[1]), dtype=A.dtype)

    for k in range(A.shape[1]):
        nrm = jnp.dot(A[:, k], A[:, k])
        R = R.at[k, k].set(jnp.sqrt(nrm))
        Q = Q.at[:, k].set(A[:, k] / R[k, k])
        for j in range(k + 1, A.shape[1]):
            R = R.at[k, j].set(jnp.dot(Q[:, k], A[:, j]))
            A = A.at[:, j].set(A[:, j] - Q[:, k] * R[k, j])

    return jax.block_until_ready(jnp.sum(A))


def numpy_array_to_jnp(dace_inputs):
    """
    Function to transform a set of numpy arrays to jax arrays
    """

    def convert_element(element):
        if isinstance(element, np.ndarray):
            return jnp.array(element)
        elif isinstance(element, (int, float, complex, np.float32, np.float64)):
            return element
        else:
            raise TypeError(f"Unsupported type {type(element)}")

    return tuple(convert_element(element) for element in dace_inputs)


def jax_initialize(size, datatype=np.float64):
    forward_inputs, _ = fwd_initialize(size)
    return numpy_array_to_jnp(forward_inputs)


jax_grad = jax.grad(jax_kernel, argnums=0)
A_j, S_j = jax_initialize((M, N))
gradient_A_j = jax_grad(A_j, S_j)
print(gradient_A_j)
print(gradient_A)
print(np.max(np.abs(gradient_A - gradient_A_j)))
assert np.allclose(gradient_A, gradient_A_j)
