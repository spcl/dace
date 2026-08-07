# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared build caches (recorded commands, CMake configure, precompiled header). All advisory, so
each test asserts the cache ENGAGES -- a declined header or unreplayed recording looks like a
correct build save for wall-clock time.
"""
import contextlib
import glob
import json
import os
import shutil
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
    per-test build would change the recording key and defeat the replay under test. Covered below.

    Under the nanobind interface the fixture also warms the helper-archive cache: the first build
    into a cold cache publishes nanobind's helper archive, which ADDS a flag to every later build's
    cmake command -- and with it a second recording key, so the cache only converges on the second
    build of a shape. Warming keeps the keys stable, preserving the "second build replays" semantics
    these tests assert. The warmup's own recordings are dropped so the first measured build still
    runs CMake."""
    monkeypatch.setattr(compiler, 'build_cache_root', lambda: str(tmp_path / 'cache'))
    with dace.config.set_temporary('compiler', 'precompiled_header', value=False):
        if dace.Config.get('compiler', 'interface') == 'nanobind':
            build_and_check(tmp_path, 'archwarmup')
            shutil.rmtree(os.path.join(str(tmp_path / 'cache'), 'commands'), ignore_errors=True)
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
    shares one directory across SDFGs and these tests need a fresh folder. Pins the development
    folder mode too: these tests inspect ``build/`` after compiling, and the production mode CI runs
    under deletes it (the env var must go first, as it beats ``set_temporary``)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv('DACE_compiler_build_folder_mode', raising=False)
        with dace.config.set_temporary('compiler', 'build_folder_mode', value='development'):
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


@pytest.mark.skipif(os.name != 'posix', reason='precompiled headers are only wired up for GCC/Clang')
def test_precompiled_header_separates_source_trees(tmp_path, monkeypatch):
    """Two checkouts sharing a compiler must not share one .gch. The mtime guard cannot catch it: it
    walks THIS tree's runtime and compares against a header built from the other's, so a stale header
    passes while the TU compiles against foreign declarations."""
    monkeypatch.setattr(compiler, 'build_cache_root', lambda: str(tmp_path / 'cache'))
    mine = compiler.prepare_precompiled_header({'cpu'})
    assert mine, 'no precompiled header was produced'

    clone = tmp_path / 'clone' / 'dace'
    real = os.path.dirname(os.path.dirname(os.path.abspath(compiler.__file__)))
    shutil.copytree(os.path.join(real, 'runtime', 'include'), clone / 'runtime' / 'include')
    shutil.copytree(os.path.join(real, 'external'), clone / 'external')  # stream.h reaches into it
    # The runtime path is derived from this module's location, so relocating it is what a second
    # checkout looks like.
    monkeypatch.setattr(compiler, '__file__', str(clone / 'codegen' / 'compiler.py'))
    theirs = compiler.prepare_precompiled_header({'cpu'})

    assert theirs, 'no precompiled header was produced for the second tree'
    assert mine != theirs, 'both trees were handed the same precompiled header'


def test_caches_disabled_still_builds(tmp_path):
    """With every cache off the build must still work -- they are optimizations, not requirements."""
    with dace.config.set_temporary('compiler', 'precompiled_header', value=False):
        with dace.config.set_temporary('compiler', 'configure_cache', value=False):
            with dace.config.set_temporary('compiler', 'command_cache', value=False):
                build_and_check(tmp_path, 'nocaches')


def test_wrongly_named_nanobind_archive_is_ignored(tmp_path, monkeypatch):
    """A cached helper archive whose name does not match the helper target the module really links
    must be ignored, never linked: nanobind names each variant after HOW it was compiled, so a
    leftover of another variant (older DaCe options, changed nanobind naming) is not
    interchangeable. The planted candidate is garbage on purpose -- linking it would fail the
    build, so a passing build proves it was ignored -- and the build must then heal the cache by
    publishing the real archive under its own name."""
    monkeypatch.setattr(compiler, 'build_cache_root', lambda: str(tmp_path / 'cache'))
    cache_dir = None
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv('DACE_compiler_interface', raising=False)
        with dace.config.set_temporary('compiler', 'interface', value='nanobind'):
            with dace.config.set_temporary('compiler', 'precompiled_header', value=False):
                cache_dir = compiler.nanobind_static_cache_dir()
                os.makedirs(cache_dir)
                with open(os.path.join(cache_dir, 'libnanobind-static-stale.a'), 'wb') as fh:
                    fh.write(b'not an archive')
                build_and_check(tmp_path, 'staletolerant')
    published = sorted(os.path.basename(p) for p in glob.glob(os.path.join(cache_dir, 'libnanobind*.a')))
    assert len(published) == 2 and 'libnanobind-static-stale.a' in published, \
        f'the build did not publish the real archive next to the stale one: {published}'


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


def test_cache_key_separates_hosts(monkeypatch):
    """The caches above are reachable from more than one machine -- ``DACE_BUILD_CACHE_DIR`` on shared
    scratch, or the ``default_build_folder`` fallback on a cluster file system. The default cpu args
    carry ``-march=native``, which the key can only see as a literal string, so identical inputs on
    two different CPUs would otherwise collide and hand one host artifacts built for the other's
    instruction set. The host identity in the key is what turns that into a miss."""
    monkeypatch.setattr(compiler, 'host_isa_id', lambda: 'cpu-a')
    on_a = compiler.cache_key('same', 'parts')
    monkeypatch.setattr(compiler, 'host_isa_id', lambda: 'cpu-b')
    assert compiler.cache_key('same', 'parts') != on_a


def test_host_isa_id_is_stable_and_nonempty():
    """A blank or drifting identity silently restores the collision above -- on every host at once,
    since they would then all agree."""
    first = compiler.host_isa_id()
    assert first, 'no host identity derived; every CPU would share one cache key'
    assert compiler.host_isa_id() == first, 'host identity is not stable within a process'
