# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared build caches (recorded commands, CMake configure, precompiled header). All advisory, so
each test asserts the cache ENGAGES -- a declined header or unreplayed recording looks like a
correct build save for wall-clock time.
"""
import contextlib
import json
import os
import subprocess

import numpy as np
import pytest

import dace
from dace.codegen import command_db, compiler

N = dace.symbol('N')


@dace.program
def scaled_add(x: dace.float64[N], y: dace.float64[N]):
    y[:] = x * 3.0 + y


@pytest.fixture
def private_cache(tmp_path, monkeypatch):
    """Point the caches at this test. PCH off: it is ~125 MB, tmp is often a RAM disk, and a failed
    per-test build would change the recording key and defeat the replay under test. Covered below."""
    monkeypatch.setattr(compiler, 'build_cache_root', lambda: str(tmp_path / 'cache'))
    with dace.config.set_temporary('compiler', 'precompiled_header', value=False):
        yield


def check(csdfg):
    """Run a compiled program and assert it computes the right thing."""
    x, y = np.random.rand(64), np.zeros(64)
    csdfg(x=x, y=y, N=64)
    assert np.allclose(y, x * 3.0)
    return csdfg


def make(name, gpu=False):
    sdfg = scaled_add.to_sdfg(simplify=True)
    sdfg.name = name
    if gpu:
        sdfg.apply_gpu_transformations()
    return sdfg


@contextlib.contextmanager
def own_build_folder(tmp_path, name):
    """One fresh build folder per program. Pins ``cache=name`` too, since CI's ``DACE_cache=single``
    shares one directory across SDFGs and these tests need a fresh folder."""
    with dace.config.set_temporary('default_build_folder', value=str(tmp_path / name)):
        with dace.config.set_temporary('cache', value='name'):
            yield


def build_and_check(tmp_path, name, gpu=False):
    """Compile into a private build folder and check the result computes the right thing."""
    with own_build_folder(tmp_path, name):
        sdfg = make(name, gpu)
        csdfg = sdfg.compile()
        build_folder = sdfg.build_folder  # resolved against the config, so read it inside the scope
    check(csdfg)
    return build_folder


def ran_cmake(build_folder):
    return os.path.exists(os.path.join(build_folder, 'build', 'CMakeCache.txt'))


def test_configure_cache_seeds_a_working_build(tmp_path, private_cache):
    """A second program reuses the first's configure and still builds. Command cache off, or it would
    replay and never reach the configure. (CMake aborts on a cache not retargeted to its folder.)"""
    with dace.config.set_temporary('compiler', 'command_cache', value=False):
        build_and_check(tmp_path, 'seedfirst')
        build_and_check(tmp_path, 'seedsecond')


@pytest.mark.skipif(os.name != 'posix', reason='recorded builds need the Ninja generator')
def test_recorded_build_is_replayed(tmp_path, private_cache):
    """The second program of a shape must reuse the first one's commands instead of running CMake."""
    assert ran_cmake(build_and_check(tmp_path, 'recordfirst'))
    assert not ran_cmake(build_and_check(tmp_path, 'recordsecond'))


@pytest.mark.skipif(os.name != 'posix', reason='recorded builds need the Ninja generator')
def test_unusable_recording_falls_back_to_cmake(tmp_path, private_cache):
    """A recording that misdescribes the program costs speed, not correctness. Here it names a TU the
    program lacks -- staleness where every path still substitutes to something plausible."""
    build_and_check(tmp_path, 'staleprime')
    root = compiler.build_cache_root()
    key = os.path.splitext(os.listdir(os.path.join(root, 'commands'))[0])[0]
    poisoned = command_db.load(root, key)
    poisoned.append(dict(poisoned[0], file=poisoned[0]['file'].replace('$NAME', '$NAME_extra')))
    command_db.drop(root, key)
    command_db.publish(root, key, poisoned)

    assert ran_cmake(build_and_check(tmp_path, 'stalevictim'))
    assert not ran_cmake(build_and_check(tmp_path, 'stalerecovered')), 'the bad recording was not replaced'


@pytest.mark.skipif(os.name != 'posix', reason='precompiled headers are only wired up for GCC/Clang')
def test_precompiled_header_is_actually_used(tmp_path):
    """The generated TU must really consume the cached header. A PCH is honored only when its flags
    match the TU's; on drift the compiler ignores it silently. ``-Werror=invalid-pch`` makes that a
    failure. Command cache off so CMake runs and exports the compile line to inspect."""
    with dace.config.set_temporary('compiler', 'command_cache', value=False):
        build_folder = build_and_check(tmp_path, 'pchused')
    database = os.path.join(build_folder, 'build', 'compile_commands.json')
    assert os.path.isfile(database), 'no compilation database was exported'
    with open(database) as fp:
        generated = [e for e in json.load(fp) if 'pchused' in e['file']]
    assert generated, 'no compile command recorded for the generated source'
    command = generated[0]['command']
    assert 'dace_prewarm.h' in command, 'the precompiled header never reached the compile line'
    checked = command.replace(' -c ', ' -Winvalid-pch -Werror=invalid-pch -c ')
    result = subprocess.run(checked, shell=True, cwd=generated[0]['directory'], capture_output=True, text=True)
    assert result.returncode == 0, f'the compiler refused the precompiled header:\n{result.stderr}'


def test_caches_disabled_still_builds(tmp_path):
    """With every cache off the build must still work -- they are optimizations, not requirements."""
    with dace.config.set_temporary('compiler', 'precompiled_header', value=False):
        with dace.config.set_temporary('compiler', 'configure_cache', value=False):
            with dace.config.set_temporary('compiler', 'command_cache', value=False):
                build_and_check(tmp_path, 'nocaches')


@pytest.mark.gpu
def test_many_sdfgs_in_one_process(tmp_path, private_cache):
    """Five CPU then five CPU+GPU programs, back to back. The GPU half is a different shape (adds a
    ``.cu``), so it records separately; within each half only the first runs CMake, all ten stay
    correct."""
    for device in ('cpu', 'gpu'):
        folders = [build_and_check(tmp_path, f'seq{device}{i}', gpu=device == 'gpu') for i in range(5)]
        assert ran_cmake(folders[0])
        assert not any(ran_cmake(f) for f in folders[1:]), f'{device} builds did not replay the recording'


@pytest.mark.mpi
@pytest.mark.gpu
def test_distributed_and_local_builds_interleave(tmp_path, private_cache):
    """Distributed and local builds share one recording. ``distributed_compile`` builds on rank 0 and
    other ranks load from its folder -- the one case a folder is read by processes that did not write
    it. The interleaved local build covers the reverse: it replays rank 0's recipe."""
    from mpi4py import MPI
    from dace.sdfg import utils

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if comm.Get_size() < 2:
        raise ValueError('run this test with at least two processes')

    def distributed(name, gpu):
        """Build on rank 0, load on the rest, and have every rank run the result."""
        if rank != 0:
            check(utils.distributed_compile(None, comm))
            return None
        with own_build_folder(tmp_path, name):
            sdfg = make(name, gpu)
            check(utils.distributed_compile(sdfg, comm))
            return sdfg.build_folder

    cpu_folder = distributed('mpicpu', gpu=False)
    comm.Barrier()

    # A local build of the same shape, between the two distributed ones.
    if rank == 0:
        assert not ran_cmake(build_and_check(tmp_path, 'mpilocal')), 'the local build ignored rank 0 recording'
    comm.Barrier()

    gpu_folder = distributed('mpigpu', gpu=True)
    comm.Barrier()

    if rank == 0:
        assert ran_cmake(cpu_folder), 'the first distributed build should have configured'
        assert ran_cmake(gpu_folder), 'the CPU+GPU shape records separately from the CPU one'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
