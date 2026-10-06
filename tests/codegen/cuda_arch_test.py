# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The CUDA architecture a CMake build targets when no GPU is visible to ``-arch=native``."""
import os
import subprocess

import dace
from dace.config import set_temporary
from dace.codegen import cuda_arch


def fake_nvcc_run(stderr: str):
    """A ``subprocess.run`` whose probe exits 0 and says ``stderr``."""

    def run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, '', stderr)

    return run


def test_fallback_arch_prefers_the_configured_default():
    """Without a GPU to detect, the build targets sm_80 -- new enough for everything the runtime
    emits, old enough for the cubin to still load on the hardware DaCe is run on."""
    assert cuda_arch.fallback_arch({75, 80, 90}) == 80
    assert cuda_arch.fallback_arch(None) == 80
    assert cuda_arch.FALLBACK_CUDA_ARCH == 80


def test_fallback_arch_drops_to_what_the_toolkit_can_build():
    """A toolkit too old for sm_80 gets the oldest architecture it has that can build fp16 at all.
    Below sm_53 <cuda_fp16.h> declares no ``__half`` operators and no ``half2`` intrinsics."""
    assert cuda_arch.fallback_arch({50, 52, 53, 60}) == 53
    assert cuda_arch.fallback_arch({35, 50}) == cuda_arch.MINIMUM_CUDA_ARCH
    assert cuda_arch.MINIMUM_CUDA_ARCH == 53


def test_native_probe_reads_the_warning_not_the_exit_code(monkeypatch):
    """nvcc does not fail when it finds no GPU for ``-arch=native``: it warns, substitutes a default
    architecture of its own and exits 0, which cannot build a half kernel at all."""
    monkeypatch.setattr(cuda_arch.subprocess, 'run',
                        fake_nvcc_run("nvcc warning : Cannot find valid GPU for '-arch=native', default arch used\n"))
    assert not cuda_arch.can_use_arch_native('/fake/nvcc-without-a-gpu')

    monkeypatch.setattr(cuda_arch.subprocess, 'run', fake_nvcc_run(''))
    assert cuda_arch.can_use_arch_native('/fake/nvcc-with-a-gpu')


def test_cuda_architectures_keeps_native_when_a_gpu_is_there(monkeypatch):
    """Empty is what CMake reads as ``native``, and native is the right answer on a real GPU host."""
    monkeypatch.setattr(cuda_arch, 'find_nvcc', lambda root, path_env: '/fake/nvcc')
    monkeypatch.setattr(cuda_arch, 'can_use_arch_native', lambda nvcc: True)
    with set_temporary('compiler', 'cuda', 'cuda_arch', value=''):
        assert cuda_arch.cuda_architectures() == ''


def test_cuda_architectures_names_an_arch_without_a_gpu(monkeypatch):
    """The GPU-less case: DaCe resolves ``native`` here rather than letting nvcc substitute a default
    that cannot compile the generated fp16."""
    monkeypatch.setattr(cuda_arch, 'find_nvcc', lambda root, path_env: '/fake/nvcc')
    monkeypatch.setattr(cuda_arch, 'can_use_arch_native', lambda nvcc: False)
    monkeypatch.setattr(cuda_arch, 'nvcc_supported_arches', lambda nvcc: frozenset({50, 52, 53, 80}))
    with set_temporary('compiler', 'cuda', 'cuda_arch', value=''):
        assert cuda_arch.cuda_architectures() == '80'


def test_cuda_architectures_honors_the_configured_arch(monkeypatch):
    """An explicit compiler.cuda.cuda_arch wins over detection, and reaches CMake as a ``;`` list."""
    monkeypatch.setattr(cuda_arch, 'find_nvcc', lambda root, path_env: '/fake/nvcc')
    monkeypatch.setattr(cuda_arch, 'can_use_arch_native', lambda nvcc: True)
    with set_temporary('compiler', 'cuda', 'cuda_arch', value='80, 90'):
        assert cuda_arch.cuda_architectures() == '80;90'


def test_cuda_architectures_without_nvcc_keeps_native(monkeypatch):
    """No nvcc to probe: CMake keeps ``native`` and its own toolkit search reports what is missing."""
    monkeypatch.setattr(cuda_arch, 'find_nvcc', lambda root, path_env: None)
    with set_temporary('compiler', 'cuda', 'cuda_arch', value=''):
        assert cuda_arch.cuda_architectures() == ''


def test_cmake_recomputes_the_cuda_architecture_every_configure():
    """The architecture must not be a CACHE entry: cached, it pinned the FIRST configure's choice and
    a later compiler.cuda.cuda_arch on an existing build folder was ignored."""
    cmakelists = os.path.join(os.path.dirname(dace.__file__), 'codegen', 'CMakeLists.txt')
    with open(cmakelists) as f:
        text = f.read()
    block = text[text.index('if (DACE_CUDA_ARCHITECTURES_DEFAULT)'):text.index('set(CMAKE_CUDA_ARCHITECTURES')]

    assert 'LOCAL_CUDA_ARCHITECTURES' in block
    assert 'CACHE' not in block, 'a cached architecture ignores a later compiler.cuda.cuda_arch'
