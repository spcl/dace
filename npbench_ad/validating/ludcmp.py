import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

N = 2000


@dc.program
def ludcmp(A: dc.float64[N, N], b: dc.float64[N], S: dc.float64[1]):

    x = np.zeros_like(b, dtype=np.float64)
    y = np.zeros_like(b, dtype=np.float64)

    for i in range(N):
        for j3 in range(i):
            A[i, j3] -= A[i, :j3] @ A[:j3, j3]
            A[i, j3] /= A[j3, j3]
        for j2 in range(i, N):
            A[i, j2] -= A[i, :i] @ A[:i, j2]
    for i1 in range(N):
        y[i1] = b[i1] - A[i1, :i1] @ y[:i1]
    for i2 in range(N - 1, -1, -1):
        x[i2] = (y[i2] - A[i2, i2 + 1:] @ x[i2 + 1:]) / A[i2, i2]

    S[0] = np.sum(x)


sdfg = ludcmp.to_sdfg()

sdfg.save("log_sdfgs/ludcmp_forward.sdfg")

add_backward_pass(sdfg=sdfg, inputs=["A"], outputs=["S"])

sdfg.save("log_sdfgs/ludcmp_backward.sdfg")

sdfg.compile()


def fwd_initialize(size, datatype=np.float64):
    N, = size
    A = np.empty((N, N), dtype=datatype)
    for i in range(N):
        A[i, :i + 1] = np.fromfunction(lambda j: (-j % N) / N + 1, (i + 1, ), dtype=datatype)
        A[i, i + 1:] = 0.0
        A[i, i] = 1.0
    A[:] = A @ np.transpose(A)
    fn = datatype(N)
    b = np.fromfunction(lambda i: (i + 1) / fn / 2.0 + 4.0, (N, ), dtype=datatype)
    S = np.zeros(shape=[1], dtype=datatype)

    # Prepare the sizes args dict
    keyword_args = {"N": N}

    return (A, b, S), keyword_args


(A_dace, b, S), keyword_args = fwd_initialize((N, ))
gradient_A = np.zeros(shape=(N, N), dtype=np.float64)
gradient_S = np.ones(shape=[1], dtype=np.float64)

sdfg(A=A_dace, b=b, S=S, gradient_A=gradient_A, gradient_S=gradient_S)


def jax_kernel(A, b, S):

    x = jnp.zeros_like(b, dtype=jnp.float64)
    y = jnp.zeros_like(b, dtype=jnp.float64)

    for i in range(A.shape[0]):
        for j in range(i):
            A = A.at[i, j].set(A[i, j] - A[i, :j] @ A[:j, j])
            A = A.at[i, j].set(A[i, j] / A[j, j])
        for j in range(i, A.shape[0]):
            A = A.at[i, j].set(A[i, j] - A[i, :i] @ A[:i, j])
    for i in range(A.shape[0]):
        y = y.at[i].set(b[i] - A[i, :i] @ y[:i])
    for i in range(A.shape[0] - 1, -1, -1):
        x = x.at[i].set((y[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i])

    S = S.at[0].set(jnp.sum(x))
    return S[0]


def jax_initialize(size, datatype=np.float32):
    forward_inputs, _ = fwd_initialize(size)
    return numpy_array_to_jnp(forward_inputs)


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


A_j, b_j, S_j = jax_initialize((N, ))
jax_grad = jax.grad(jax_kernel, argnums=0)
grad_A = jax_grad(A_j, b_j, S_j)
# print(gradient_A)
# print(grad_A)
assert np.allclose(grad_A, gradient_A)
