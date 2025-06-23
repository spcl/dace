import numpy as np
import dace as dc
from dace.autodiff import add_backward_pass
import jax
import jax.numpy as jnp

NX = 4
NY = 4
TMAX = 1


@dc.program
def fdtd_2d(ex: dc.float32[NX, NY], ey: dc.float32[NX, NY], hz: dc.float32[NX, NY], _fict_: dc.float32[TMAX],
            S: dc.float32[1]):

    for t in range(TMAX):
        ey[0, :] = _fict_[t]
        ey[1:, :] -= 0.5 * (hz[1:, :] - hz[:-1, :])
        ex[:, 1:] -= 0.5 * (hz[:, 1:] - hz[:, :-1])
        hz[:-1, :-1] -= 0.7 * (ex[:-1, 1:] - ex[:-1, :-1] + ey[1:, :-1] - ey[:-1, :-1])

    @dc.map(_[0:NX, 0:NY])
    def summap(i, j):
        s >> S(1, lambda x, y: x + y)[0]
        z << hz[i, j]
        s = z


sdfg = fdtd_2d.to_sdfg()

sdfg.save("log_sdfgs/fdtd_2d_forward.sdfg")

add_backward_pass(sdfg=sdfg, inputs=["hz"], outputs=["S"], autooptimize=True)

sdfg.save("log_sdfgs/fdtd_2d_backward.sdfg")

sdfg.compile()


def fwd_initialize(size, datatype=np.float32):
    TMAX, NX, NY = size
    ex = np.empty((NX, NY), dtype=np.float32)
    ey = np.empty((NX, NY), dtype=np.float32)
    hz = np.empty((NX, NY), dtype=np.float32)
    _fict_ = np.empty((TMAX, ), dtype=np.float32)
    for i in range(TMAX):
        _fict_[i] = i
    for i in range(NX):
        for j in range(NY):
            ex[i, j] = (i * (j + 1)) / NX
            ey[i, j] = (i * (j + 2)) / NY
            hz[i, j] = (i * (j + 3)) / NX
    S = np.zeros(shape=[1], dtype=datatype)
    # Prepare the sizes args dict
    keyword_args = {"TMAX": TMAX, "NX": NX, "NY": NY}
    return (ex, ey, hz, _fict_, S), keyword_args


(ex, ey, hz, _fict_, S), keyword_args = fwd_initialize((TMAX, NX, NY))
ex_j, ey_j, hz_j, _fict_j, S_j = jnp.array(ex), jnp.array(ey), jnp.array(hz), jnp.array(_fict_), jnp.array(S)

gradient_hz = np.zeros(shape=(NX, NY), dtype=np.float32)
gradient_S = np.ones(shape=[1], dtype=np.float32)

sdfg(ex=ex, ey=ey, hz=hz, _fict_=_fict_, S=S, gradient_hz=gradient_hz, gradient_S=gradient_S)


def jax_kernel(ex, ey, hz, _fict_, S):
    TMAX = _fict_.shape[0]
    for t in range(TMAX):
        ey = ey.at[0, :].set(_fict_[0])
        ey = ey.at[1:, :].set(ey[1:, :] - 0.5 * (hz[1:, :] - hz[:-1, :]))
        ex = ex.at[:, 1:].set(ex[:, 1:] - 0.5 * (hz[:, 1:] - hz[:, :-1]))
        hz = hz.at[:-1, :-1].set(hz[:-1, :-1] - 0.7 * (ex[:-1, 1:] - ex[:-1, :-1] + ey[1:, :-1] - ey[:-1, :-1]))

    return jax.block_until_ready(jnp.sum(hz))


jax_grad = jax.grad(jax_kernel, argnums=(2))(ex_j, ey_j, hz_j, _fict_j, S_j)
assert np.allclose(gradient_hz, jax_grad)
