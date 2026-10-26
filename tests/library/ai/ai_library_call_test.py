# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Live test of generated code that requests a DaCe environment for FFTW. Needs an API key and FFTW, and is
marked ``ai``.
"""

import os
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

import dace
from dace import nodes
from dace.codegen import compiler_family
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


def fftw_usable() -> bool:
    """
    :return: True if a program using FFTW compiles and links, i.e. the header and the development
             library are installed.
    """
    compiler = compiler_family.host_compiler()
    if not compiler or shutil.which(compiler) is None:
        return False
    with tempfile.TemporaryDirectory() as directory:
        result = subprocess.run(
            [compiler, "-x", "c++", "-", "-lfftw3", "-o", os.path.join(directory, "fftw_check")],
            input="#include <fftw3.h>\nint main() { fftw_cleanup(); }\n",
            capture_output=True,
            text=True,
        )
    return result.returncode == 0


@pytest.mark.ai
@pytest.mark.skipif(not fftw_usable(), reason="needs the FFTW header and development library")
def test_generated_environment_calls_fftw():
    sdfg = dace.SDFG("ai_fftw")
    sdfg.add_array("inp", [N], dace.complex128)
    sdfg.add_array("out", [N], dace.complex128)

    state = sdfg.add_state()
    node = AINode("fft", FFT_DESCRIPTION, inputs={"_inp"}, outputs={"_out"})
    state.add_node(node)
    state.add_edge(state.add_read("inp"), None, node, "_inp", dace.Memlet(f"inp[0:{N}]"))
    state.add_edge(node, "_out", state.add_write("out"), None, dace.Memlet(f"out[0:{N}]"))

    node.expand(state, "ai")

    tasklet = next(n for n in state.nodes() if isinstance(n, nodes.Tasklet))
    generated = [path for path in tasklet.environments if path.startswith(ai_environments.MODULE_PREFIX)]
    assert generated, "no environment was generated for the FFTW dependency"
    assert tasklet.state_fields, "the FFTW plan was not stored in a state field"
    assert tasklet.code_init.as_string.strip(), "the plan is not created in the initialization code"
    assert tasklet.code_exit.as_string.strip(), "the plan is not destroyed in the finalization code"

    rng = np.random.default_rng(0)
    inp = (rng.random(N) + 1j * rng.random(N)).astype(np.complex128)
    out = np.zeros(N, dtype=np.complex128)
    sdfg(inp=inp, out=out)

    assert np.allclose(out, np.fft.fft(inp), rtol=1e-8, atol=1e-8)


if __name__ == "__main__":
    test_generated_environment_calls_fftw()
