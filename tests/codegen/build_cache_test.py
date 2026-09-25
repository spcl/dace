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
from dace.codegen import build_cache, command_db, compiler

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


#: Templated name a recording gives the program's shared library (see ``command_db.template``).
PROGRAM_LIBRARY = 'lib$NAME.so'


@pytest.mark.skipif(os.name != 'posix', reason='recorded builds need the Ninja generator')
def test_recording_that_links_a_vanished_library_rebuilds_and_replaces_itself(tmp_path, private_cache):
    """A recording bakes in the absolute library paths CMake's ``find_package`` probed -- libgomp on a
    GNU toolchain among them -- and nothing revalidates them before a replay. A toolchain that moves
    one out from under the recording must therefore cost a rebuild, never correctness, and must not
    keep costing one: the linker rejects the operand it cannot find, the replay reports failure, and
    the caller reconfigures and records the shape again. Distinct from the stale recording above,
    which is refused before a single command runs -- this recipe runs, fails partway, and leaves a
    build folder that has to be cleared before CMake can configure over it.
    """
    build_and_check(tmp_path, 'vanishedprime')
    root = compiler.build_cache_root()
    key = os.path.splitext(os.listdir(os.path.join(root, 'commands'))[0])[0]
    recorded = command_db.load(root, key)
    program_links = [e for e in recorded if e['output'] == PROGRAM_LIBRARY]
    assert program_links, f'no {PROGRAM_LIBRARY} entry: the recording never links the program'
    # Right after the output name, where the driver reads it as one more input file to resolve.
    marker = f'-o {PROGRAM_LIBRARY}'
    assert marker in program_links[0]['command'], 'the recorded link line no longer names its output'
    absent = tmp_path / 'uninstalled-toolchain' / 'libgomp.so'
    relinked = {
        e['output']: dict(e, command=e['command'].replace(marker, f'{marker} {absent}', 1))
        for e in program_links
    }
    # Written straight over the entry, so recovery below is the caller's alone to demonstrate.
    with open(command_db.entry_path(root, key), 'w') as fp:
        json.dump([relinked.get(e['output'], e) for e in recorded], fp)

    assert ran_cmake(build_and_check(tmp_path, 'vanishedvictim')), 'the failed replay never reached CMake'
    assert not ran_cmake(build_and_check(tmp_path, 'vanishedrecovered')), 'the broken recording was not replaced'


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


def test_runtime_digest_sees_an_edit_that_keeps_the_mtime(tmp_path):
    """A whole-second filesystem leaves the mtime of an edit made in the same second unchanged, so
    only the content tells the two headers apart."""
    header = tmp_path / 'include' / 'dace' / 'math.h'
    header.parent.mkdir(parents=True)
    header.write_text('int a;\n')
    stat = header.stat()
    before = build_cache.runtime_digest(str(tmp_path / 'include'))

    header.write_text('int b;\n')
    os.utime(header, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    assert build_cache.runtime_digest(str(tmp_path / 'include')) != before


@pytest.mark.skipif(os.name != 'posix', reason='precompiled headers are only wired up for GCC/Clang')
def test_a_runtime_edit_that_keeps_the_mtime_rebuilds_the_precompiled_header(tmp_path, monkeypatch):
    """The cache sits in /dev/shm with nanosecond mtimes and the runtime on capstor with whole-second
    ones, so a header edited in the second its .gch was built compared as older than the .gch."""
    monkeypatch.setattr(compiler, 'build_cache_root', lambda: str(tmp_path / 'cache'))
    clone = tmp_path / 'clone' / 'dace'
    real = os.path.dirname(os.path.dirname(os.path.abspath(compiler.__file__)))
    shutil.copytree(os.path.join(real, 'runtime', 'include'), clone / 'runtime' / 'include')
    shutil.copytree(os.path.join(real, 'external'), clone / 'external')  # stream.h reaches into it
    monkeypatch.setattr(compiler, '__file__', str(clone / 'codegen' / 'compiler.py'))
    before = compiler.prepare_precompiled_header({'cpu'})
    assert before, 'no precompiled header was produced'

    header = clone / 'runtime' / 'include' / 'dace' / 'math.h'
    stat = header.stat()
    header.write_text(header.read_text() + '\n// edited in the second the header was precompiled\n')
    os.utime(header, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    after = compiler.prepare_precompiled_header({'cpu'})

    assert after and after != before, 'the edited runtime was handed the pre-edit precompiled header'


@pytest.mark.skipif(os.name != 'posix', reason='precompiled headers are only wired up for GCC/Clang')
def test_vanished_precompiled_header_is_not_included(tmp_path, monkeypatch):
    """A build configured with a precompiled header must still build once that header is gone (e.g., it
    lived in /dev/shm before a reboot) and no new one is made: the configured path must not linger."""
    monkeypatch.setattr(compiler, 'build_cache_root', lambda: str(tmp_path / 'cache'))
    with dace.config.set_temporary('compiler', 'command_cache', value=False):
        build_folder = build_and_check(tmp_path, 'pchvanished')
        shutil.rmtree(tmp_path / 'cache' / 'pch')
        # Keep only the configured cache, so the program is compiled again against it
        cmake_folder = os.path.join(build_folder, 'build')
        for entry in os.listdir(cmake_folder):
            if entry != 'CMakeCache.txt':
                path = os.path.join(cmake_folder, entry)
                shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
        with dace.config.set_temporary('compiler', 'precompiled_header', value=False):
            build_and_check(tmp_path, 'pchvanished')


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


@pytest.mark.skipif(os.name != 'posix', reason='recorded builds need the Ninja generator')
def test_a_folder_reconfigured_under_new_flags_publishes_nothing(tmp_path, private_cache):
    """A reconfigure keeps what CMake detected under the old flags, so it must not be filed under the new flags' key."""
    build_and_check(tmp_path, 'reconfigured')
    root = compiler.build_cache_root()
    published = {cache: sorted(os.listdir(os.path.join(root, cache))) for cache in ('configure', 'commands')}
    assert all(published.values()), f'the first build published nothing, so nothing is tested: {published}'
    other_flags = dace.Config.get('compiler', 'cpu', 'args') + ' -DDACE_RECONFIGURED_UNDER_NEW_FLAGS'

    with dace.config.set_temporary('compiler', 'cpu', 'args', value=other_flags):
        build_and_check(tmp_path, 'reconfigured')

    assert {cache: sorted(os.listdir(os.path.join(root, cache))) for cache in published} == published


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


#: One variable per reader that changes what a configure finds or its compiler searches: CMake itself, its
#: modules, the compiler driver, ``find_package``'s ``<PackageName>_DIR``/``_ROOT``, and ROCm's HIP config.
CONFIGURE_INPUTS = ('CMAKE_PREFIX_PATH', 'CC', 'CXX', 'PKG_CONFIG_PATH', 'HIP_PATH', 'CUDA_PATH', 'LD_LIBRARY_PATH',
                    'CPATH', 'C_INCLUDE_PATH', 'CPLUS_INCLUDE_PATH', 'LIBRARY_PATH', 'OPENBLAS_DIR', 'FFTW_ROOT',
                    'TBLIS_ROOT', 'ROCM_PATH')


@pytest.mark.parametrize('name', CONFIGURE_INPUTS)
def test_cache_key_separates_environments_a_configure_reads(name, tmp_path, monkeypatch):
    """Every process of one user shares these caches, whatever environment it runs in. A configure that found a
    package through one process's ``CMAKE_PREFIX_PATH`` must not be handed to a process whose environment lacks it."""
    monkeypatch.setattr(compiler, 'build_cache_root', lambda: str(tmp_path / 'cache'))
    monkeypatch.setenv(name, '/configure/input/a')
    under_a = compiler.cache_key('same', 'parts')
    monkeypatch.setenv(name, '/configure/input/b')
    assert compiler.cache_key('same', 'parts') != under_a, f'{name} does not reach the cache key'


@pytest.mark.parametrize('name', ('PWD', 'OLDPWD', 'SLURM_JOB_ID', 'PYTEST_CURRENT_TEST', 'PYTEST_XDIST_WORKER'))
def test_cache_key_ignores_variables_no_configure_reads(name, tmp_path, monkeypatch):
    """A keyed variable that changes per directory, job or test turns every build into a miss."""
    monkeypatch.setattr(compiler, 'build_cache_root', lambda: str(tmp_path / 'cache'))
    monkeypatch.setenv(name, 'a')
    under_a = compiler.cache_key('same', 'parts')
    monkeypatch.setenv(name, 'b')
    assert compiler.cache_key('same', 'parts') == under_a, f'{name} reaches the cache key'


def test_an_unreadable_cmake_keys_the_whole_environment():
    """Without a CMake installation to derive the inputs from, only keying everything is safe."""
    assert compiler.environment_pattern(None).fullmatch('PYTEST_CURRENT_TEST')


def detected_compiler(build_folder):
    """The compiler detection a build folder's configure used."""
    found = glob.glob(os.path.join(build_folder, 'build', 'CMakeFiles', '[0-9]*', 'CMakeCXXCompiler.cmake'))
    assert found, f'no compiler detection under {build_folder}'
    with open(found[0]) as fp:
        return fp.read()


def test_a_configure_detected_under_another_cpath_is_not_reused(tmp_path, private_cache, monkeypatch):
    """Compiler detection records ``CPATH`` as an implicit include directory, and the configure cache transplants
    that detection into later build folders: seeded across a changed ``CPATH``, a build keeps a search directory its
    own environment never named. Command cache off, or the second build would replay and never configure."""
    marker = tmp_path / 'cpath-marker'
    marker.mkdir()
    with dace.config.set_temporary('compiler', 'command_cache', value=False):
        with monkeypatch.context() as marked:
            marked.setenv('CPATH', str(marker), prepend=os.pathsep)
            marked_folder = build_and_check(tmp_path, 'cpathmarked')
        plain_folder = build_and_check(tmp_path, 'cpathplain')
    assert str(marker) in detected_compiler(marked_folder), 'CPATH never reached the detection, so nothing is tested'
    assert str(marker) not in detected_compiler(plain_folder), 'the configure was seeded from another CPATH'


@pytest.mark.skipif(os.name != 'posix', reason='recorded builds need the Ninja generator')
def test_a_recording_made_under_another_cpath_is_not_replayed(tmp_path, private_cache, monkeypatch):
    """A replay runs the lines CMake authored under the recording's environment, so it must not stand in for a
    build whose ``CPATH`` differs."""
    marker = tmp_path / 'cpath-marker'
    marker.mkdir()
    with monkeypatch.context() as marked:
        marked.setenv('CPATH', str(marker), prepend=os.pathsep)
        assert ran_cmake(build_and_check(tmp_path, 'recordmarked'))
    assert ran_cmake(build_and_check(tmp_path, 'recordplain')), 'a recording made under another CPATH was replayed'
