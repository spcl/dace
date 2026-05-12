# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Minimal pre-GPU SDFG that segfaults on the experimental CUDA codegen.

``CPU scalar -> GPU scalar transient -> CPU array`` (3 nodes, 2 edges, one
state, no tasklet, no NSDFG, no Map). Legacy codegen handles it; experimental
codegen produces a binary that SIGSEGVs inside ``compiled_sdfg.fast_call``.

Original report: ``tests/multistream_copy_cudatest.py::test_copy_sync`` —
this test is its minimal distillation. The SDFG is built and run inside a
subprocess so a crash surfaces as a non-zero exit code rather than killing
the pytest runner.
"""
import os
import subprocess
import sys
import textwrap

import pytest

_REPRO_SCRIPT = textwrap.dedent("""
    import numpy as np
    import dace
    from dace import StorageType

    sdfg = dace.SDFG('h2d_d2h_scalar')
    sdfg.add_scalar('scal_in', dace.float32)
    sdfg.add_scalar('gpu_scal', dace.float32, StorageType.GPU_Global, transient=True)
    sdfg.add_array('output', [1], dace.float32)

    state = sdfg.add_state()
    state.add_nedge(state.add_read('scal_in'), state.add_access('gpu_scal'),
                    dace.Memlet('scal_in'))
    state.add_nedge(state.add_access('gpu_scal'), state.add_write('output'),
                    dace.Memlet('gpu_scal'))

    out = np.zeros(1, dtype=np.float32)
    sdfg(scal_in=np.float32(2), output=out)
    assert out[0] == 2.0, f'expected 2.0, got {out[0]}'
""")


_IS_EXPERIMENTAL = os.environ.get('DACE_compiler_cuda_implementation') == 'experimental'


@pytest.mark.gpu
@pytest.mark.xfail(
    _IS_EXPERIMENTAL,
    reason="Experimental codegen segfaults on CPU scalar -> GPU scalar "
    "transient -> CPU array (issue I1 in the new-gpu-codegen-dev plan). "
    "Legacy codegen passes.",
    strict=True,
)
def test_cpu_gpu_cpu_scalar_roundtrip():
    proc = subprocess.run([sys.executable, '-c', _REPRO_SCRIPT],
                          capture_output=True,
                          text=True,
                          timeout=180)
    assert proc.returncode == 0, (f"exit={proc.returncode}\n"
                                  f"stdout={proc.stdout}\nstderr={proc.stderr}")
