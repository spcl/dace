# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Live tests of AI-generated code that calls into an external library.

This is the case where the model has to request a DaCe *environment* -- CMake glue plus headers --
and use the tasklet's initialization and finalization code to manage a resource whose lifetime
spans calls. Needs an API key and a local FFTW installation, so it is marked ``ai`` and excluded
from CI.
"""

import ctypes.util
import os

import numpy as np
import pytest

import dace
from dace import nodes
from dace.libraries.ai import environments as ai_environments
from dace.libraries.ai.nodes import AINode

N = 64

FFT_DESCRIPTION = f"""
Compute the forward discrete Fourier transform of a {N}-element complex double vector, writing the
result to the output. Use FFTW (fftw3.h, the fftw3 library): create the plan once with
fftw_plan_dft_1d in the initialization code, keep it in a state field, execute it here with
fftw_execute_dft, and destroy it with fftw_destroy_plan in the finalization code. The plan must
not be created or destroyed on every call. Use FFTW_ESTIMATE so that planning does not overwrite
the input buffers. Match numpy.fft.fft, which uses the exp(-2*pi*i*k*n/N) sign convention.
""".strip()


@pytest.mark.ai
@pytest.mark.skipif(ctypes.util.find_library('fftw3') is None, reason='needs FFTW installed')
def test_generated_environment_calls_fftw(tmp_path):
    sdfg = dace.SDFG('ai_fftw')
    sdfg.add_array('inp', [N], dace.complex128)
    sdfg.add_array('out', [N], dace.complex128)

    state = sdfg.add_state()
    node = AINode('fft', FFT_DESCRIPTION, inputs={'_inp'}, outputs={'_out'})
    state.add_node(node)
    state.add_edge(state.add_read('inp'), None, node, '_inp', dace.Memlet(f'inp[0:{N}]'))
    state.add_edge(node, '_out', state.add_write('out'), None, dace.Memlet(f'out[0:{N}]'))

    with dace.config.set_temporary('ai', 'environment_dir', value=str(tmp_path)):
        node.expand(state, 'ai')

        # The model had to request an environment, and it was written where it can be read and
        # re-registered in a later session
        written = [f for f in os.listdir(tmp_path) if f.startswith(ai_environments.MODULE_PREFIX)]
        assert written, 'no environment module was generated for the FFTW dependency'

        tasklet = next(n for n in state.nodes() if isinstance(n, nodes.Tasklet))
        assert tasklet.environments, 'the generated tasklet is not linked against any environment'
        # The plan outlives a single call
        assert tasklet.state_fields, 'the FFTW plan was not stored in a state field'
        assert tasklet.code_init.as_string.strip(), 'the plan is not created in the initialization code'
        assert tasklet.code_exit.as_string.strip(), 'the plan is not destroyed in the finalization code'

        rng = np.random.default_rng(0)
        inp = (rng.random(N) + 1j * rng.random(N)).astype(np.complex128)
        out = np.zeros(N, dtype=np.complex128)
        sdfg(inp=inp, out=out)

    assert np.allclose(out, np.fft.fft(inp), rtol=1e-8, atol=1e-8)


if __name__ == '__main__':
    pytest.main([__file__, '-m', 'ai'])
