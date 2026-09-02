# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Handles compilation of code objects. Creates the proper folder structure,
    compiles each target separately, links all targets to one binary, and
    returns the corresponding CompiledSDFG object. """

import atexit
import collections
import contextlib
import getpass
import glob
import hashlib
import io
import os
import pathlib
import platform
import re
import shutil
import shlex
import signal
import subprocess
import tempfile
from typing import Callable, Dict, Iterator, List, Literal, Set, Tuple, TypeVar, Union, Optional, overload
import warnings
from functools import lru_cache

import dace
from dace.config import Config
from dace.codegen import build_cache
from dace.codegen import command_db
from dace.codegen import common
from dace.codegen import compiler_family
from dace.codegen import exceptions as cgx
from dace.codegen.target import TargetCodeGenerator
from dace.codegen.codeobject import CodeObject
from dace.codegen import compiled_sdfg as csd
from dace.codegen.target import make_absolute

T = TypeVar('T')

# Only readability-* fixes safe on include-stripped code (the experimental readable generator's
# clang-tidy pass strips the header block, so a fix depending on types or a variable's full
# use-set rewrites on a half-parse). Excluded: identifier naming/length, magic-numbers,
# cognitive-complexity, uppercase-suffix (noise); non-const-parameter (const-qualifies a pointer
# only forwarded to a nested-SDFG writer -> const vs non-const clash nvcc rejects); and
# modernize-* (type-dependent -> miscompiled the CUDA block-reduction: an empty ``using`` alias
# and a reduction index turned into a range-for value).
# NOTE: the clang-tidy invocation itself (``apply_clang_tidy``) was dropped from
# ``generate_program_folder`` by the build-caching refactor; nothing in ``dace/`` reads this
# constant anymore. Kept only so ``tests/codegen/readable/test_clang_tidy.py`` still collects.
CLANG_TIDY_CHECKS = ('readability-*,'
                     '-readability-identifier-naming,-readability-identifier-length,-readability-magic-numbers,'
                     '-readability-function-cognitive-complexity,-readability-uppercase-literal-suffix,'
                     '-readability-avoid-const-params-in-decls,-readability-non-const-parameter')

#: Compiled artifacts, as opposed to the sources and build-system files beside them in the folder.
BUILD_ARTIFACT_EXTENSIONS = ('.o', '.obj', '.so', '.dylib', '.dll', '.a', '.lib')


def discard_stale_build(out_path: str, sdfg_name: str) -> None:
    """Drop the compiled artifacts of whichever program held THIS program's slot before.

    A build folder is named after the program by default, so two ``@dace.program`` functions that
    share a name -- in one module, or in two modules of the same name -- share a folder, and only
    the build system's timestamp comparison stands between them. That comparison is not sound here:
    a filesystem stamping mtimes at one-second granularity (Lustre and NFS do) leaves a program
    regenerated within the same second as the previous link reading as up to date, so the build is
    skipped and the PREVIOUS program's library is what gets loaded. The symptom is a wrong answer,
    not a build error, and it comes and goes with how fast the two compiles happen.

    Only the artifacts carrying ``sdfg_name`` go: every one is named after the program it belongs
    to -- ``lib<name>`` and ``CMakeFiles/<name>.dir/`` -- and a folder legitimately holds several
    programs at once, whose libraries a caller may still be holding open. Dropping those too would
    make one program's rebuild unload another's, which is what the collision cannot do.

    Called only when the folder's recorded hash says the program changed, so an unchanged rerun
    still reuses everything.
    """
    build_path = os.path.join(out_path, 'build')
    for root, _, files in os.walk(build_path):
        # The stub is the same code for every program; rebuilding it would cost a compile a run.
        owned_directory = f'{sdfg_name}.dir' in os.path.relpath(root, build_path).split(os.sep)
        for filename in files:
            if 'dacestub' in filename:
                continue
            if not owned_directory and sdfg_name not in filename:
                continue
            if os.path.splitext(filename)[1] in BUILD_ARTIFACT_EXTENSIONS:
                try:
                    os.remove(os.path.join(root, filename))
                except OSError:
                    continue  # already gone, or not ours to remove; the build will overwrite it


def generate_program_folder(
    sdfg,
    code_objects: List[CodeObject],
    out_path: str,
    config=None,
    folder_mode: Optional[str] = None,
) -> str:
    """Writes all files required to configure and compile the DaCe program into the specified folder.

    This function respects the ``compiler.build_folder_mode`` configuration variable,
    thus depending on its value the content might be different. However, in any case
    the source files are always generated.

    :param sdfg: The SDFG to generate the program folder for.
    :param code_objects: List of generated code objects.
    :param out_path: The folder in which the build files should be written.
    :param folder_mode: Select which files should be saved in the program build folder;
                        if not given, ``compiler.build_folder_mode`` is used.
    :return: Path to the program folder.

    :note: The ``config`` argument is retained for compatibility and should not be used.
    """

    # NOTE: In older version the argument `config` could be a used to pass a custom
    #   "configuration" (probably a `dict`) object, that would then be written to
    #   `dace.conf` inside the folder. If nothing was provided the content of the
    #   global `dace.Config` would be used. However, since _everything_ is consulting
    #   `dace.Config` for advice, an external configuration, i.e. settings different
    #   from `dace.Config` can not take effect and storing it is wrong. Thus this
    #   feature was dropped.
    if config is not None:
        warnings.warn(
            'Passed a not `None` `config` argument to `generate_program_folder()`.'
            ' This has no effect and will be ignored. Instead `dace.Config` will'
            ' be used.',
            category=UserWarning,
            stacklevel=2,
        )

    if folder_mode is None:
        folder_mode = Config.get('compiler', 'build_folder_mode')

    src_path = os.path.join(out_path, "src")
    filelist = list()

    # Write each code object to a file
    for code_object in code_objects:

        name = code_object.name
        extension = code_object.language
        target_name = code_object.target.target_name
        target_type = code_object.target_type

        # Create target folder
        target_folder = os.path.join(src_path, target_name)
        if target_type:
            target_folder = os.path.join(target_folder, target_type)
        os.makedirs(target_folder, exist_ok=True)

        # Write code to file
        basename = "{}.{}".format(name, extension)
        code_path = os.path.join(target_folder, basename)
        clean_code = code_object.clean_code

        if Config.get_bool('compiler', 'format_code'):
            config_file = Config.get('compiler', 'format_config_file')
            if config_file is not None and config_file != "":
                run_arg_list = ['clang-format', f"-style=file:{config_file}"]
            else:
                run_arg_list = ['clang-format']
            result = subprocess.run(run_arg_list, input=clean_code, text=True, capture_output=True)
            if result.returncode or result.stderr:
                warnings.warn(f'clang-format failed to run: {result.stderr}')
            else:
                clean_code = result.stdout

        # Save the file only if it changed (keeps old timestamps and saves
        # build time)
        if not identical_file_exists(code_path, clean_code):
            with open(code_path, "w") as code_file:
                code_file.write(clean_code)

        if code_object.linkable == True:
            filelist.append("{},{},{}".format(target_name, target_type, basename))

        # Generate the source map.
        if sdfg and (folder_mode in ["development"]):
            if code_object.language == 'cpp' and code_object.title == 'Frame':
                code_object.create_source_map(sdfg)

    # Write list of files
    #  Needed to communicate with `configure_and_compile()`, deleted in production mode.
    with open(os.path.join(out_path, "dace_files.csv"), "w") as filelist_file:
        filelist_file.write("\n".join(filelist))

    # Build a list of environments used
    environments = set()
    for obj in code_objects:
        environments |= obj.environments

    # Write list of environments
    #  Needed to communicate with `configure_and_compile()`, deleted in production mode.
    with open(os.path.join(out_path, "dace_environments.csv"), "w") as env_file:
        env_file.write("\n".join(environments))

    # Save the SDFG itself and its hash
    if sdfg is not None:
        hash = sdfg.save(os.path.join(out_path, "program.sdfgz"), hash=True, compress=True)
        filepath = os.path.join(out_path, 'include', 'hash.h')
        contents = f'#define __HASH_{sdfg.name} "{hash}"\n'
        if not identical_file_exists(filepath, contents):
            # The folder already held a DIFFERENT program, so its artifacts cannot be reused and
            # the build system may not notice on its own.
            if os.path.isfile(filepath):
                discard_stale_build(out_path, sdfg.name)
            with open(filepath, 'w') as hfile:
                hfile.write(contents)

    # Write cachedir tag
    cachedir_tag = os.path.join(out_path, "CACHEDIR.TAG")
    if not os.path.exists(cachedir_tag):
        with open(cachedir_tag, "w") as f:
            f.write("\n".join([
                "Signature: 8a477f597d28d172789f06886806bc55",
                "# This file is a cache directory tag created by DaCe.",
                "# For information about cache directory tags, see:",
                "#	http://www.brynosaurus.com/cachedir/",
            ]))

    # Generate the parts of the folder that are exclusive to the development folder mode.
    if folder_mode in ["development"]:
        # NOTE: There is a bug here, as this only saves they keys inside the configuration
        #   `dict`. It ignores the configuration values set through environment variables.
        #   instead it will store the ones in the `dict`.
        Config.save(os.path.join(out_path, "dace.conf"), all=True)

    # The runtime's `report.save()` uses `std::ofstream` to open `<folder>/perf/report-*.json`.
    #  If `perf/` does not exist it will fail, thus we have to create it if it is needed.
    #  Technically we only need to create it if the SDFG is instrumented. But we will also
    #  create it in development mode. Furthermore, if there is no SDFG given, we also create
    #  it to be on the safe side.
    if (folder_mode in ["development"]) or (sdfg is None) or sdfg.is_instrumented():
        os.makedirs(os.path.join(out_path, 'perf'), exist_ok=True)

    # The folder mode file is always generated. In case it is missing we assume the old version.
    with open(os.path.join(out_path, "FOLDER_MODE"), "w") as version_file:
        version_file.write(folder_mode)

    return out_path


#: Untested on Windows.
CACHES_SUPPORTED = os.name != 'nt'

#: Last-resort cache location inside the build folder. Prefixed so it cannot collide with the
#: per-SDFG folders next to it, which are named after the SDFG.
BUILD_CACHE_FOLDER = '__dace_build_cache'


def build_cache_root() -> str:
    """Directory holding the machine-global build caches, shared by every SDFG.

    All advisory: a miss costs speed, never correctness. RAM-backed when possible, since on HPC
    nodes the temp directory is often a shared file system where re-reading a large precompiled
    header costs more than it saves.
    """
    root = os.environ.get('DACE_BUILD_CACHE_DIR')
    if not root:
        usable = (c for c in ('/dev/shm', tempfile.gettempdir()) if os.path.isdir(c) and os.access(c, os.W_OK))
        root = next(usable, None)
        if root is None:
            return os.path.join(Config.get('default_build_folder'), BUILD_CACHE_FOLDER)
    return os.path.join(root, f'dace_build_cache_{getpass.getuser()}')


#: A PREDICTION of CMake's ``CMAKE_CXX_FLAGS_<CONFIG>`` defaults, only ever used to build the
#: precompiled header with the same flags the translation unit will get; nothing here reaches the
#: real build. A wrong entry costs the PCH speedup, never correctness. NVHPC needs its own row --
#: it differs from GNU in every config. Verified against CMake, not documentation.
BUILD_TYPE_FLAGS_BY_FAMILY = {
    'gnu': {
        'Debug': ['-g'],
        'Release': ['-O3', '-DNDEBUG'],
        'RelWithDebInfo': ['-O2', '-g', '-DNDEBUG'],
        'MinSizeRel': ['-Os', '-DNDEBUG'],
    },
    'nvhpc': {
        'Debug': ['-g', '-O0'],
        'Release': ['-fast', '-O3', '-DNDEBUG'],
        'RelWithDebInfo': ['-O2', '-gopt'],
        'MinSizeRel': ['-O2', '-s', '-DNDEBUG'],
    },
}

#: Clang, IntelLLVM and anything unrecognized use CMake's GNU-like defaults.
CMAKE_BUILD_TYPE_FLAGS = BUILD_TYPE_FLAGS_BY_FAMILY['gnu']


def build_type_flags() -> list:
    """The flags CMake will append for the configured build type and host compiler."""
    family = compiler_family.detect(compiler_family.host_compiler())
    table = BUILD_TYPE_FLAGS_BY_FAMILY.get(family, CMAKE_BUILD_TYPE_FLAGS)
    return list(table.get(Config.get('compiler', 'build_type'), []))


@lru_cache(maxsize=1, typed=True)
def host_isa_id() -> str:
    """Identity of the instruction set ``-march=native`` resolves to on this host.

    The CPU flag list is what the compiler's ``native`` detection reads, so hashing it separates
    hosts that would be given different instructions. ``model name`` comes along for the tuning
    half. Falls back to the platform triple where there is no ``/proc`` (macOS, Windows), which is
    coarser but never wrong in the unsafe direction -- a coarse key over-separates, it cannot merge
    two hosts that differ.
    """
    try:
        with open('/proc/cpuinfo') as fp:  # 'Features' is the aarch64 spelling of 'flags'
            fields = [ln for ln in fp if ln.startswith(('model name', 'flags', 'Features'))][:2]
    except OSError:
        fields = []
    identity = ''.join(fields) if fields else f'{platform.machine()}|{platform.processor()}'
    return hashlib.sha256(identity.encode()).hexdigest()[:12]


def cache_key(*parts: object) -> str:
    # Everything keyed here was produced FOR this host: the default cpu args carry -march=native,
    # and a cache root on shared storage (DACE_BUILD_CACHE_DIR, or the default_build_folder
    # fallback) is reachable from nodes whose CPUs differ. Those must miss, not reuse.
    return hashlib.sha256('\0'.join(str(p) for p in (*parts, host_isa_id())).encode()).hexdigest()[:16]


def newest_mtime(path: str) -> float:
    """Modification time of ``path``, or ``0.0`` if it is not there to stat."""
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def seed_cmake_configure(build_folder: str, key: str) -> bool:
    """Seed a fresh build folder with an earlier configure, returning whether it was seeded.

    ``CMakeCache.txt`` holds the ``find_package`` results, ``CMakeFiles/<version>/`` the compiler
    and ABI detection. Neither differs between two programs configured the same way; both are copied
    together, since seeding one still forces the other half of the work.
    """
    entry = os.path.join(build_cache_root(), 'configure', key)
    if os.path.exists(os.path.join(build_folder, 'CMakeCache.txt')) or not os.path.isdir(entry):
        return False
    try:
        with open(os.path.join(entry, 'CMakeCache.txt')) as fp:
            # CMake refuses a cache it finds anywhere other than where it was created, aborting the
            # configure outright, so retarget that one entry at this build folder.
            cache = re.sub(r'(?m)^CMAKE_CACHEFILE_DIR:INTERNAL=.*$',
                           'CMAKE_CACHEFILE_DIR:INTERNAL=' + build_folder.replace('\\', '/'),
                           fp.read(),
                           count=1)
        with open(os.path.join(build_folder, 'CMakeCache.txt'), 'w') as fp:
            fp.write(cache)
        shutil.copytree(os.path.join(entry, 'CMakeFiles'), os.path.join(build_folder, 'CMakeFiles'), dirs_exist_ok=True)
        build_cache.touch(entry)
        return True
    except OSError:
        shutil.rmtree(os.path.join(build_folder, 'CMakeFiles'), ignore_errors=True)
        return False


def publish_cmake_configure(build_folder: str, key: str) -> None:
    """Publish a fresh configure so the next SDFG can reuse it.

    Only the compiler-detection subdirectory is kept; the rest of ``CMakeFiles/`` holds this
    program's objects, which must never move to another build.
    """
    entry = os.path.join(build_cache_root(), 'configure', key)
    versions = glob.glob(os.path.join(build_folder, 'CMakeFiles', '[0-9]*'))
    if os.path.isdir(entry) or not versions:
        return
    staging = f'{entry}.{os.getpid()}'
    try:
        os.makedirs(os.path.join(staging, 'CMakeFiles'), exist_ok=True)
        shutil.copy2(os.path.join(build_folder, 'CMakeCache.txt'), staging)
        shutil.copytree(versions[0], os.path.join(staging, 'CMakeFiles', os.path.basename(versions[0])))
        os.rename(staging, entry)  # atomic, and loses harmlessly to a concurrent publisher
        build_cache.prune(os.path.join(build_cache_root(), 'configure'))
    except OSError:
        shutil.rmtree(staging, ignore_errors=True)


def prepare_precompiled_header(targets) -> Optional[str]:
    """Precompile ``<dace/dace.h>`` once per (runtime, compiler, flags), returning its dir or ``None``.

    The runtime umbrella header is most of the compile time of a small kernel; caching it across
    SDFGs is what makes precompiling pay. Should the flags drift from CMake's line, the compiler
    silently declines the header and produces the same object.

    The include path is part of the key, not just the flags: the cache is machine-global, so two
    DaCe checkouts sharing a compiler would otherwise share one ``.gch`` and the second would compile
    against the first one's headers. The mtime guard below cannot catch that -- it walks THIS tree's
    runtime and compares against a header built from another's, so it passes while being wrong.
    """
    if not (CACHES_SUPPORTED and Config.get_bool('compiler', 'precompiled_header')):
        return None
    runtime = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'runtime', 'include')
    cxx = make_absolute(compiler_family.host_compiler())
    flags = ([f'-std=c++{common.cpp_standard()}', '-fPIC', '-fopenmp'] + shlex.split(compiler_family.cpu_args() or '') +
             build_type_flags())
    if any(t in ('cuda', 'experimental_cuda') for t in targets):
        flags.append('-DWITH_CUDA')
    pch = os.path.join(build_cache_root(), 'pch', cache_key(runtime, cxx, *flags))
    header = os.path.join(pch, 'dace_prewarm.h')
    newest = max((os.path.getmtime(os.path.join(r, f)) for r, _, fs in os.walk(runtime) for f in fs), default=0.0)
    try:
        # Strictly newer, so a header edit in the same second still invalidates the cached result.
        if not (os.path.exists(header + '.gch') and os.path.getmtime(header + '.gch') > newest):
            os.makedirs(pch, exist_ok=True)
            with open(header, 'w') as fp:
                fp.write('#include <dace/dace.h>\n')
            # Build to a private name and rename, so a concurrent build never sees a partial header.
            staging = f'{header}.gch.{os.getpid()}'
            subprocess.run([cxx] + flags + ['-I', runtime, '-x', 'c++-header', header, '-o', staging],
                           check=True,
                           capture_output=True)
            os.replace(staging, header + '.gch')
            # ~110 MB just landed in the cache, whose default root is RAM. Bound it here, at the
            # only point where it grows.
            build_cache.prune(os.path.join(build_cache_root(), 'pch'))
        build_cache.touch(pch)
        return pch
    except (OSError, subprocess.SubprocessError):
        return None


def run_cmake(cmake_command: str, build_folder: str, configure_key: str, jobs: int, output_stream) -> None:
    """Configure and build ``build_folder``, seeding and publishing the configure cache around it."""
    if Config.get('debugprint') == 'verbose':
        print(f'Running CMake: {cmake_command}')

    cmake_filename = os.path.join(build_folder, 'cmake_configure.sh')
    reuse_configure = CACHES_SUPPORTED and Config.get_bool('compiler', 'configure_cache')
    seeded = reuse_configure and seed_cmake_configure(build_folder, configure_key)
    try:
        if not identical_file_exists(cmake_filename, cmake_command):
            _run_liveoutput(cmake_command, shell=True, cwd=build_folder, output_stream=output_stream)
            if reuse_configure and not seeded:
                publish_cmake_configure(build_folder, configure_key)
    except subprocess.CalledProcessError as ex:
        # Clean CMake directory and try once more
        if Config.get_bool('debugprint'):
            print('Cleaning CMake build folder and retrying...')
        # Drop the seed: a bad one would poison every later build of this shape.
        if seeded:
            shutil.rmtree(os.path.join(build_cache_root(), 'configure', configure_key), ignore_errors=True)
        shutil.rmtree(build_folder, ignore_errors=True)
        os.makedirs(build_folder)
        try:
            _run_liveoutput(cmake_command, shell=True, cwd=build_folder, output_stream=output_stream)
        except subprocess.CalledProcessError as ex:
            # If still unsuccessful, print results
            if Config.get_bool('debugprint'):
                raise cgx.CompilerConfigurationError('Configuration failure')
            else:
                raise cgx.CompilerConfigurationError('Configuration failure:\n' + ex.output)

    with open(cmake_filename, "w") as fp:
        fp.write(cmake_command)

    # ``--parallel`` bounds the build; Ninja would otherwise use every core.
    try:
        _run_liveoutput(f"cmake --build . --config {Config.get('compiler', 'build_type')} --parallel {jobs}",
                        shell=True,
                        cwd=build_folder,
                        output_stream=output_stream)
    except subprocess.CalledProcessError as ex:
        # If unsuccessful, print results
        if Config.get_bool('debugprint'):
            raise cgx.CompilationError('Compiler failure')
        else:
            raise cgx.CompilationError('Compiler failure:\n' + ex.output)


def configure_and_compile(
    program_folder,
    program_name=None,
    output_stream=None,
    folder_mode: Optional[str] = None,
) -> pathlib.Path:
    """
    Configures and compiles a DaCe program in the specified folder into a shared library file.

    This function respects the ``compiler.build_folder_mode`` configuration variable,
    thus depending on its value the content might be different.

    :param program_folder: Folder containing all files necessary to build, equivalent to
                           what was passed to `generate_program_folder`.
    :param output_stream: Additional output stream to write to (used for other clients
                          such as the vscode extension).
    :return: Path to the compiled shared library file.
    """

    if folder_mode is None:
        folder_mode = Config.get('compiler.build_folder_mode')
    assert folder_mode in ["development", "production"]

    # Rejected before any folder is made or any code compiled, so a typo'd mode cannot silently
    # fall back to the other backend after doing half the work.
    build_mode = Config.get('compiler', 'build_mode').strip().lower()
    if build_mode not in ('cmake', 'native'):
        raise cgx.CompilerConfigurationError(
            f"Unknown compiler.build_mode {Config.get('compiler', 'build_mode')!r}; expected 'cmake' or 'native'.")

    if program_name is None:
        program_name = os.path.basename(program_folder)
    program_folder = os.path.abspath(program_folder)
    src_folder = os.path.join(program_folder, "src")

    # Prepare build folder
    build_folder = os.path.join(program_folder, "build")
    os.makedirs(build_folder, exist_ok=True)

    # NOTE: We do not create the instrumentation-report folder (`perf/`) here.
    #   The reason is that this folder is only needed when the SDFG is instrumented,
    #   and to determine this we need the SDFG and we do not have that. Thus the
    #   folder is generated (if needed) by `generate_program_folder()`.

    # Read list of DaCe files to compile.
    # We do this instead of iterating over source files in the directory to
    # avoid globbing files from previous compilations, such that we don't need
    # to wipe the directory for every compilation.
    with open(os.path.join(program_folder, "dace_files.csv"), "r") as f:
        file_list = [line.strip().split(",") for line in f]

    # Get absolute paths and targets for all source files
    files = []
    targets = {}  # {target name: target class}
    for target_name, target_type, file_name in file_list:
        if target_type:
            path = os.path.join(target_name, target_type, file_name)
        else:
            path = os.path.join(target_name, file_name)
        files.append(path)
        targets[target_name] = next(k for k, v in TargetCodeGenerator.extensions().items() if v['name'] == target_name)

    # Windows-only workaround: Override Visual C++'s linker to use
    # Multi-Threaded (MT) mode. This fixes linkage in CUDA applications where
    # CMake fails to do so.
    if os.name == 'nt':
        if '_CL_' not in os.environ:
            os.environ['_CL_'] = '/MT'
        elif '/MT' not in os.environ['_CL_']:
            os.environ['_CL_'] = os.environ['_CL_'] + ' /MT'

    # Resolve the environments the SDFG uses; both build backends take their flags from these.
    with open(os.path.join(program_folder, "dace_environments.csv"), "r") as f:
        environments = set(l.strip() for l in f)
    environments = dace.library.get_environments_and_dependencies(environments)

    # Build the shared library either directly (native) or through CMake. Both write it to the same
    # development-mode location that the shared tail below expects.
    if build_mode == 'native':
        # Lazy: native_compiler imports from this module, so a top-level import would be a cycle.
        from dace.codegen import native_compiler
        native_compiler.build_native(program_folder=program_folder,
                                     program_name=program_name,
                                     files=files,
                                     targets=targets,
                                     environments=environments,
                                     build_folder=build_folder,
                                     output_stream=output_stream)
    else:
        cmake_configure_and_build(program_folder=program_folder,
                                  program_name=program_name,
                                  src_folder=src_folder,
                                  build_folder=build_folder,
                                  files=files,
                                  targets=targets,
                                  environments=environments,
                                  output_stream=output_stream)

    # Get the names of the library files that were generated.
    #  Currently we are still in the `development` folder mode.
    lib_path = get_binary_name(object_folder=program_folder, sdfg_name=program_name, folder_mode="development")
    libstub_path = _get_stub_library_path(lib_path)

    # In production mode, we are now deleting what we need and relocating it.
    if folder_mode == "production":
        lib_path = pathlib.Path(shutil.move(src=lib_path, dst=program_folder))
        libstub_path = pathlib.Path(shutil.move(src=libstub_path, dst=program_folder))
        program_folder = pathlib.Path(program_folder)
        # TODO: Find out where `sample/` are generated and suppress their generation.
        for to_delete in ["include", "src", "build", "sample", "dace_environments.csv", "dace_files.csv"]:
            if (program_folder / to_delete).is_dir():
                shutil.rmtree(os.path.join(program_folder, to_delete))
            else:
                (program_folder / to_delete).unlink()

    return lib_path


def cmake_configure_and_build(
    program_folder: str,
    program_name: str,
    src_folder: str,
    build_folder: str,
    files: List[str],
    targets: Dict,
    environments,
    output_stream=None,
) -> None:
    """Configure and build a prepared program folder with CMake (the default ``build_mode``).

    :param files: source paths relative to ``<program_folder>/src`` (from ``dace_files.csv``).
    :param targets: ``{target name: TargetCodeGenerator}`` for the linkable sources.
    :param environments: resolved environment classes the SDFG uses.
    """
    # Start forming CMake command
    dace_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Ninja's global dependency graph parallelizes a multi-source build better than Make's
    # per-directory one, and it is what can report the commands it ran (see ``command_db``). Not on
    # Windows, where ``-A x64`` is a Visual Studio generator option that Ninja rejects.
    use_ninja = os.name != 'nt' and shutil.which('ninja') is not None
    cmake_command = [
        "cmake",
        "-A x64" if os.name == 'nt' else "",  # Windows-specific flag
        "-G Ninja" if use_ninja else "",
        '"' + os.path.join(dace_path, "codegen") + '"',
        "-DDACE_SRC_DIR=\"{}\"".format(src_folder),
        "-DDACE_FILES=\"{}\"".format(";".join(files)),
        "-DDACE_PROGRAM_NAME={}".format(program_name),
        "-DDACE_CPP_STANDARD={}".format(common.cpp_standard()),
    ]

    environment_flags, cmake_link_flags = get_environment_flags(environments)
    cmake_command += sorted(environment_flags)

    cmake_command += shlex.split(Config.get('compiler', 'extra_cmake_args'))

    # Replace backslashes with forward slashes
    cmake_command = [cmd.replace('\\', '/') for cmd in cmake_command]

    # Generate CMake options for each compiler
    libraries = set()
    cmake_files = []
    for target_name, target in sorted(targets.items()):
        try:
            cmake_command += target.cmake_options()
            cmake_files += target.cmake_files()
            libraries |= unique_flags(Config.get("compiler", target_name, "libs"))
        except KeyError:
            pass
        except ValueError as ex:  # Cannot find compiler executable
            raise cgx.CompilerConfigurationError(str(ex))

    cmake_command.append("-DDACE_LIBS=\"{}\"".format(" ".join(sorted(libraries))))
    cmake_command.append(f"-DDACE_CMAKE_FILES=\"{';'.join(cmake_files)}\"")
    cmake_command.append(f"-DCMAKE_BUILD_TYPE={Config.get('compiler', 'build_type')}")
    # Additive static archive next to the .so (opt-in; matches native build mode). Part of the
    # command ``shape`` below, so a static-archive build never reuses a non-archive recording.
    cmake_command.append(
        "-DDACE_STATIC_ARCHIVE={}".format("ON" if Config.get_bool('compiler', 'static_archive') else "OFF"))
    # Free -- the generator already knows the commands -- and lets tooling see a generated source's
    # exact compile flags.
    cmake_command.append("-DCMAKE_EXPORT_COMPILE_COMMANDS=ON")

    # Set linker and linker arguments, iff they have been specified
    cmake_linker = Config.get('compiler', 'linker', 'executable') or ''
    cmake_linker = cmake_linker.strip()
    if cmake_linker:
        cmake_linker = make_absolute(cmake_linker)
        cmake_command.append(f'-DCMAKE_LINKER="{cmake_linker}"')
    cmake_link_flags = (' '.join(sorted(cmake_link_flags)) + ' ' +
                        (Config.get('compiler', 'linker', 'args') or '')).strip()
    if cmake_link_flags:
        cmake_command.append(f'-DCMAKE_SHARED_LINKER_FLAGS="{cmake_link_flags}"')

    pch_dir = prepare_precompiled_header(targets)
    if pch_dir:
        cmake_command.append(f'-DDACE_PCH_DIR="{pch_dir}"')
    # What the configure DISCOVERS: the command minus the flags naming this program. ``DACE_FILES``
    # reduces to its target subdirectories, which select the languages and packages CMake enables.
    shape = [c for c in cmake_command if not c.startswith(('-DDACE_SRC_DIR=', '-DDACE_FILES=', '-DDACE_PROGRAM_NAME='))]
    configure_key = cache_key(*shape, *sorted({os.path.dirname(f) for f in files}))
    # A replay also pins the exact translation units and the CMake sources behind their flags:
    # editing the CMakeLists or an environment's .cmake changes the compile line but nothing above.
    cmake_sources = sorted({p for c in shape for p in re.findall(r'[^"=;\s]+\.cmake', c)})
    cmake_sources.append(os.path.join(dace_path, 'codegen', 'CMakeLists.txt'))
    command_key = cache_key(*shape, *[f.replace(program_name, '$NAME') for f in files],
                            *(newest_mtime(p) for p in cmake_sources))
    cmake_command = ' '.join(cmake_command)

    ##############################################
    # Build. A recorded build for this exact shape replays directly; anything else -- no recording,
    # or one that turns out not to describe this program -- goes through CMake, which then records.
    reuse_commands = CACHES_SUPPORTED and Config.get_bool('compiler', 'command_cache')
    recorded = command_db.load(build_cache_root(), command_key) if reuse_commands else None
    recipe = command_db.accepts(recorded, build_folder, program_folder, program_name, files) if recorded else None
    jobs = max(1, int(Config.get('compiler', 'build_jobs')))
    replayed = recipe is not None and command_db.replay(recipe, build_folder, jobs)
    if not replayed:
        if recorded:
            command_db.drop(build_cache_root(), command_key)
        if recipe is not None:
            command_db.clear(build_folder)  # it ran and failed partway, so nothing here is trustworthy
        run_cmake(cmake_command, build_folder, configure_key, jobs, output_stream)
        if reuse_commands and use_ninja:
            command_db.publish(
                build_cache_root(), command_key,
                command_db.template(command_db.capture(build_folder), build_folder, program_folder, program_name))


def build_folder_is_disposable(sdfg: 'dace.SDFG') -> bool:
    """Whether ``sdfg``'s build folder is garbage the moment this process is done with it.

    Only under ``cache: unique``, which names the folder after the PID: nothing outside this
    process can ever address it, so the folder is pure garbage once the run ends -- yet it
    survives, and a test sweep leaves one behind per (SDFG name, worker); this repo's tree had 540
    of them at 238 MB. Every other policy names the folder so that a later run finds it again,
    which makes the folder the build cache itself; deleting it would turn every rerun into a full
    recompile.

    An explicitly assigned build folder is the caller's, never ours to remove.
    """
    return sdfg.build_folder_is_default and Config.get('cache') == 'unique'


#: Program folders this process built that no later run could ever address, dropped on the way out.
#: Not tied to ``CompiledSDFG`` lifetime: CPython frees that object through the garbage collector,
#: so "the folder goes when the handle goes" is not a moment anything can observe -- a reference
#: cycle anywhere in the call graph defers it arbitrarily.
_disposable_folders: Set[str] = set()


def register_disposable_folder(sdfg: 'dace.SDFG') -> None:
    """Take ``sdfg``'s build folder into the set dropped when this process ends.

    Called from ``SDFG.compile()`` rather than from the handle that ends up owning the folder,
    because compilation hands the folder to a deepcopy of the SDFG by assigning it -- at which
    point the copy looks like a caller-assigned folder and the policy that produced it is no
    longer visible.
    """
    if build_folder_is_disposable(sdfg):
        _disposable_folders.add(os.path.abspath(sdfg.build_folder))


@atexit.register
def drop_disposable_folders() -> None:
    """Drop every disposable folder on the way out -- past here nothing can name one again.

    Deleting a folder whose library is still mapped is safe on POSIX (the inode outlives the
    unlink); on Windows the delete simply fails and the folder is left behind as before.
    """
    for folder in _disposable_folders:
        shutil.rmtree(folder, ignore_errors=True)


def get_program_handle(
    library_path: Union[pathlib.Path, str],
    sdfg: 'dace.SDFG',
    stub_library_path: Union[pathlib.Path, str, None] = None,
) -> csd.CompiledSDFG:
    """Construct a  ``CompiledSDFG`` form a precompiled library directly.

    This function is similar to the (preferred) ``load_precompiled_sdfg()``. However,
    instead of passing the build folder of the SDFG to the function, the path to the
    compiled library is passed directly.

    :param library_path: Path to the compiled library representing ``sdfg``.
    :param sdfg: The SDFG, will be referenced by the returned ``CompiledSDFG``.
    :param stub_library_path: The path to the stub library.
    """
    library_path = pathlib.Path(library_path)
    if not library_path.is_file():
        raise FileNotFoundError(f'Compiled SDFG library not found: {library_path}')
    libstub_path = _get_stub_library_path(library_path) if stub_library_path is None else pathlib.Path(
        stub_library_path).resolve()
    assert libstub_path.is_file()

    lib = csd.ReloadableDLL(library_filename=library_path, libstub_path=libstub_path)
    return csd.CompiledSDFG(sdfg, lib, sdfg.arg_names)


def load_from_file(sdfg, binary_filename):
    warnings.warn(
        'Used deprecated ``load_from_file()`` function, use ``get_program_handle()`` instead.',
        category=DeprecationWarning,
        stacklevel=2,
    )
    return get_program_handle(library_path=binary_filename, sdfg=sdfg)


@overload
def get_folder_mode(object_folder: Union[pathlib.Path, str], probe: Literal[False] = False) -> str:
    ...


@overload
def get_folder_mode(object_folder: Union[pathlib.Path, str], probe: Literal[True]) -> Optional[str]:
    ...


@overload
def get_folder_mode(object_folder: Union[pathlib.Path, str], probe: bool) -> Optional[str]:
    ...


def get_folder_mode(object_folder: Union[pathlib.Path, str], probe: bool = False) -> Optional[str]:
    """Inspect `object_folder` and determine which save mode the folder has.

    If the function finds the ``FOLDER_MODE`` file it will examine it to get the save mode.
    If the folder mode file is absent the function assumes that it is the ``development``
    format, however, some sanity checks are performed.

    The function also has the optional argument ``probe`` if given and the folder
    save mode could not be inferred the function will return ``None`` instead of
    generating an error.
    """
    object_folder = pathlib.Path(object_folder)

    if not object_folder.is_dir():
        if probe:
            return None
        raise NotADirectoryError("The build folder does not exists.")

    if (object_folder / 'FOLDER_MODE').exists():
        with open(object_folder / 'FOLDER_MODE', 'rt') as F:
            folder_mode = F.readline().strip()
        return folder_mode
    else:
        # This is to check an old style folder, i.e. a cache folder that was generated
        #  before the `FOLDER_MODE` file was introduced. We do some small sanity checks.
        # TODO: Investigate if we should check for `program.sdfgz` and if it is not
        #   pressent assume that we just have some random folder.
        # TODO: Phase out this feature, after there are no old style caches.
        maybe_an_old_style_folder = (object_folder / "build").is_dir()
        for sub_folder in ["map", "src", "include", "sample"]:
            if (object_folder / sub_folder).is_dir() != maybe_an_old_style_folder:
                if probe:
                    # TODO: This is an inconsistent folder, currently it is not an error
                    #   but should it be one?
                    return None
                raise NotADirectoryError(f'The old-style folder ``{object_folder}`` is inconsistent.')

        if maybe_an_old_style_folder:
            # All expected folders where found, so expect that this is a 'development' format folder.
            return "development"
        elif probe:
            # None of the files where found and this is probe, so it is probably just an "empty" folder.
            return None
        else:
            # Up for discussion what to do here.
            raise NotADirectoryError(f'``{object_folder}`` does not appear to be a valid old-style build folder.')


def get_binary_name(
    object_folder: Union[pathlib.Path, str],
    sdfg_name: str,
    lib_extension: Optional[str] = None,
    folder_mode: Optional[str] = None,
) -> pathlib.Path:
    """Returns the supposed location of the compiled library given the boundary conditions.

    If folder mode is not explicitly given, then the function will use `get_folder_mode()`,
    if this fails, then the `compiler.build_folder_mode` key is consulted.

    :param object_folder: The build folder of the SDFG, i.e. `sdfg.build_folder`.
    :param sdfg_name: The name of the SDFG, i.e. `sdfg.name`.
    :param lib_extension: The extension of the library, i.e. file extension.
                          If not given the config option `compiler.library_extension` is used.
    :param folder_mode: The mode of the build folder.
    """
    if lib_extension is None:
        lib_extension = Config.get('compiler', 'library_extension')

    # First try `get_folder_mode()` if that failed, consult the configuration.
    if folder_mode is None:
        folder_mode = get_folder_mode(object_folder, probe=True)
    if folder_mode is None:
        folder_mode = Config.get('compiler', 'build_folder_mode')

    folder_hirarchy = [object_folder]
    if folder_mode == 'development':
        folder_hirarchy.append('build')
    elif folder_mode == 'production':
        # Nothing to add, they are on the top.
        pass
    else:
        raise ValueError(f"Unknown folder mode '{folder_mode}' found.")

    return pathlib.Path(os.path.join(*folder_hirarchy, f'lib{sdfg_name}.{lib_extension}'))


def _get_stub_library_path(sdfg_lib_path: Union[pathlib.Path, str]) -> pathlib.Path:
    """Returns the supposed location of the compiled stub library given the path of the compiled library.
    """
    sdfg_lib_path = pathlib.Path(sdfg_lib_path)
    parent = sdfg_lib_path.parent
    lib_name = sdfg_lib_path.name
    assert lib_name.startswith('lib') and len(lib_name) > 3

    return sdfg_lib_path.parent / ('libdacestub_' + lib_name[3:])


def load_precompiled_sdfg(
    folder: Union[pathlib.Path, str],
    sdfg: Optional['dace.SDFG'] = None,
) -> csd.CompiledSDFG:
    """Loads a precompiled SDFG from ``folder``.

    If ``sdfg`` is not given then the function expects to find the ``program.sdfg(z)``
    dump file inside ``folder``. If the folder does not contain a ``FOLDER_MODE`` file
    it assumes that it is an old style ``development`` folder otherwise, the information
    from ``FOLDER_MODE`` is consulted.

    :param folder: Path to SDFG output folder, i.e. its build folder.
    :param sdfg: If given then ``program.sdfg(z)`` does not need to be present.
    :return: A callable CompiledSDFG object.

    :note: If ``sdfg`` is given then it is referenced by the returned ``CompiledSDFG``.
    """
    folder = pathlib.Path(folder)

    if not folder.is_dir():
        raise NotADirectoryError(f'Can not load the SDFG from folder ``{folder}``.')

    folder_mode = get_folder_mode(folder)

    # Try to find the sdfg from disc, if not given.
    if sdfg is not None:
        assert isinstance(sdfg, dace.SDFG)
    else:
        for name in ['program.sdfgz', 'program.sdfg']:
            if (folder / name).exists():
                sdfg = dace.SDFG.from_file(folder / name)
                break
        else:
            raise ValueError(f"Could not locate the SDFG for `{folder}`.")

    return get_program_handle(library_path=get_binary_name(folder, sdfg_name=sdfg.name, folder_mode=folder_mode),
                              sdfg=sdfg)


def _get_or_eval(value_or_function: Union[T, Callable[[], T]]) -> T:
    """
    Returns a stored value or lazily evaluates it. Used in environments
    for allowing potential runtime (rather than import-time) checks.
    """
    if callable(value_or_function):
        return value_or_function()
    return value_or_function


def get_environment_flags(environments) -> Tuple[List[str], Set[str]]:
    """
    Returns the CMake environment and linkage flags associated with the
    given input environments/libraries.

    :param environments: A list of ``@dace.library.environment``-decorated
                         classes.
    :return: A 2-tuple of (environment CMake flags, linkage CMake flags)
    """
    cmake_minimum_version = [0]
    cmake_variables = collections.OrderedDict()
    cmake_packages = set()
    cmake_includes = set()
    cmake_libraries = set()
    cmake_compile_flags = set()
    cmake_link_flags = set()
    cmake_files = set()
    cmake_module_paths = set()
    for env in environments:
        if (env.cmake_minimum_version is not None and len(env.cmake_minimum_version) > 0):
            version_list = list(map(int, env.cmake_minimum_version.split(".")))
            for i in range(max(len(version_list), len(cmake_minimum_version))):
                if i >= len(version_list):
                    break
                if i >= len(cmake_minimum_version):
                    cmake_minimum_version = version_list
                    break
                if version_list[i] > cmake_minimum_version[i]:
                    cmake_minimum_version = version_list
                    break
                # Otherwise keep iterating
        env_variables = _get_or_eval(env.cmake_variables)
        for var in env_variables:
            if (var in cmake_variables and cmake_variables[var] != env_variables[var]):
                raise KeyError("CMake variable {} was redefined from {} to {}.".format(
                    var, cmake_variables[var], env_variables[var]))
            cmake_variables[var] = env_variables[var]
        cmake_packages |= set(_get_or_eval(env.cmake_packages))
        cmake_includes |= set(_get_or_eval(env.cmake_includes))
        cmake_libraries |= set(_get_or_eval(env.cmake_libraries))
        cmake_compile_flags |= set(_get_or_eval(env.cmake_compile_flags))
        cmake_link_flags |= set(_get_or_eval(env.cmake_link_flags))
        # Make path absolute
        env_dir = os.path.dirname(env._dace_file_path)
        cmake_files |= set(
            (f if os.path.isabs(f) else os.path.join(env_dir, f)) + (".cmake" if not f.endswith(".cmake") else "")
            for f in _get_or_eval(env.cmake_files))
        headers = _get_or_eval(env.headers)
        if not isinstance(headers, dict):
            headers = {'frame': headers}
        for header_group in headers.values():
            for header in header_group:
                if os.path.isabs(header):
                    # Giving an absolute path is not good practice, but allow it
                    # for emergency overriding
                    cmake_includes.add(os.path.dirname(header))
                abs_path = os.path.join(env_dir, header)
                if os.path.isfile(abs_path):
                    # Allow includes stored with the library, specified with a
                    # relative path
                    cmake_includes.add(env_dir)
                    break

    environment_flags = [
        "-DDACE_ENV_MINIMUM_VERSION={}".format(".".join(map(str, cmake_minimum_version))),
        # Make CMake list of key-value pairs
        "-DDACE_ENV_VAR_KEYS=\"{}\"".format(";".join(cmake_variables.keys())),
        "-DDACE_ENV_VAR_VALUES=\"{}\"".format(";".join(cmake_variables.values())),
        "-DDACE_ENV_PACKAGES=\"{}\"".format(" ".join(sorted(cmake_packages))),
        "-DDACE_ENV_INCLUDES=\"{}\"".format(" ".join(sorted(cmake_includes))),
        "-DDACE_ENV_LIBRARIES=\"{}\"".format(" ".join(sorted(cmake_libraries))),
        "-DDACE_ENV_COMPILE_FLAGS=\"{}\"".format(" ".join(cmake_compile_flags)),
        # "-DDACE_ENV_LINK_FLAGS=\"{}\"".format(" ".join(cmake_link_flags)),
        "-DDACE_ENV_CMAKE_FILES=\"{}\"".format(";".join(sorted(cmake_files))),
    ]
    # Escape variable expansions to defer their evaluation
    environment_flags = [cmd.replace("$", "_DACE_CMAKE_EXPAND") for cmd in sorted(environment_flags)]

    return environment_flags, cmake_link_flags


def unique_flags(flags):
    pattern = '[^ ]+[`\'"][^"\'`]+["\'`]|[^ ]+'
    if not isinstance(flags, str):
        flags = " ".join(flags)
    return set(re.findall(pattern, flags))


def identical_file_exists(filename: str, file_contents: str):
    # If file did not exist before, return False
    if not os.path.isfile(filename):
        return False

    # Read file in blocks and compare strings
    block_size = 65536
    with open(filename, 'r') as fp:
        file_buffer = fp.read(block_size)
        while len(file_buffer) > 0:
            block = file_contents[:block_size]
            if file_buffer != block:
                return False
            file_contents = file_contents[block_size:]
            file_buffer = fp.read(block_size)

    # More contents appended to the new file
    if len(file_contents) > 0:
        return False

    return True


#: Environment-variable prefixes an MPI/PMI launcher (srun, mpirun) exports to mark a process as a
#: rank of its job. A child that inherits these and links a PMI/PMIx client -- directly, or
#: transitively through an MPI-wrapper compiler -- treats itself as that rank and blocks in
#: MPI_Init/PMIx_Init awaiting a rendezvous that never comes.
MPI_RANK_ENV_PREFIXES = (
    'PMI_',  # MPICH / Cray / Slurm PMI: PMI_RANK, PMI_SIZE, PMI_FD, PMI_JOBID, ...
    'PMIX_',  # PMIx (OpenMPI 4+): PMIX_RANK, PMIX_NAMESPACE, PMIX_SERVER_URI*, ...
    'OMPI_COMM_WORLD_',  # OpenMPI: OMPI_COMM_WORLD_RANK/SIZE/LOCAL_RANK, ...
    'OMPI_UNIVERSE_',
    'MV2_COMM_WORLD_',  # MVAPICH2
    'MPI_LOCALRANKID',
    'MPI_LOCALNRANKS',
    'SLURM_PROCID',  # Slurm's PMI plugins derive rank from these
    'SLURM_LOCALID',
)


def build_subprocess_env(base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """``base`` (default ``os.environ``) with this process's MPI-rank identity stripped.

    CMake -- and the try_compile test binaries, make/ninja and the compiler driver it spawns --
    otherwise inherit the launcher's rank-identity variables and hang forever in a PMI/PMIx init
    call, which surfaces as a stuck ``cmake`` with defunct children. Compilation never needs an MPI
    identity; everything else (PATH, compiler flags, MCA tuning, ...) is preserved."""
    env = os.environ if base is None else base
    return {k: v for k, v in env.items() if not k.startswith(MPI_RANK_ENV_PREFIXES)}


@contextlib.contextmanager
def build_subprocess_sigmask() -> Iterator[None]:
    """Temporarily unblock ``SIGCHLD`` on the calling thread, so a subprocess forked inside this
    context inherits an unblocked ``SIGCHLD``.

    MPI/Slurm launchers (``srun``, ``mpirun``) start their tasks with ``SIGCHLD`` *blocked*, and
    every child inherits that mask. CMake (KWSys) learns that the helpers it spawns during
    *configure* -- ``uname``, the compiler-id / ABI test binaries, ``make``/``ninja`` -- have
    finished by receiving ``SIGCHLD``; blocked, it is never woken to reap them and spins forever in
    ``select()``. That is the daint compile hang: it looks like a stuck ``cmake`` even though
    nothing is compiling. (Confirmed under srun: every task's ``/proc/self/status`` shows ``SigBlk``
    with the ``SIGCHLD`` bit set, and a trivial ``project()`` configure hangs until the child mask
    is cleared.)

    A child inherits the *forking thread's* mask and ``Popen`` does not reset it, so unblocking
    immediately around the fork is enough. ``pthread_sigmask`` is per-thread, so this never disturbs
    another thread or the process's steady-state mask. No-op where ``pthread_sigmask``/``SIGCHLD``
    are unavailable (Windows)."""
    if os.name != 'posix' or signal.SIGCHLD not in signal.pthread_sigmask(signal.SIG_BLOCK, []):
        yield  # Windows has neither call, or SIGCHLD is already deliverable (the common case).
        return
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGCHLD})
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGCHLD})


def _run_liveoutput(command, output_stream=None, **kwargs):
    # Every build subprocess is forked here -- CMake configure/build and the native backend's
    # compile/link lines alike -- so both launcher safeguards belong at this one point rather than
    # at each call site, where a new caller silently reintroduces the hang. Only the fork itself has
    # to happen inside the sigmask context.
    kwargs['env'] = build_subprocess_env(kwargs.get('env'))
    with build_subprocess_sigmask():
        process = subprocess.Popen(command, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, **kwargs)
    output = io.StringIO()
    while True:
        line = process.stdout.readline().rstrip()
        if not line:
            break
        output.write(line.decode('utf-8') + '\n')
        if Config.get_bool('debugprint'):
            print(line.decode('utf-8'), flush=True)
    stdout, stderr = process.communicate()
    if Config.get_bool('debugprint'):
        print(stdout.decode('utf-8'), flush=True)
        if stderr is not None:
            print(stderr.decode('utf-8'), flush=True)
    if output_stream is not None:
        output_stream.write(stdout.decode('utf-8'), flush=True)
    output.write(stdout.decode('utf-8'))
    if stderr is not None:
        output.write(stderr.decode('utf-8'))

    # An error occurred, raise exception
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command, output.getvalue())


# Allow configuring and compiling a prepared build folder from the commandline.
# This is useful for remote execution.
if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("path", type=str)
    argparser.add_argument("outname", type=str)
    args = vars(argparser.parse_args())

    Config.load(os.path.join(args["path"], "dace.conf"))

    configure_and_compile(args["path"], args["outname"])
