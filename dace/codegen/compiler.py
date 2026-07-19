# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Handles compilation of code objects. Creates the proper folder structure,
    compiles each target separately, links all targets to one binary, and
    returns the corresponding CompiledSDFG object. """

import collections
import contextlib
import io
import os
import pathlib
import re
import shutil
import shlex
import signal
import subprocess
from typing import Callable, List, Literal, Set, Tuple, TypeVar, Union, Optional, overload
import warnings

import dace
from dace.config import Config
from dace.codegen import exceptions as cgx
from dace.codegen.target import TargetCodeGenerator
from dace.codegen.codeobject import CodeObject
from dace.codegen import compiled_sdfg as csd
from dace.codegen.target import make_absolute

T = TypeVar('T')


def deduplicate_lines(code: str, is_candidate: Callable[[str], bool]) -> str:
    """
    Order-preserving de-duplication: drops a line only when ``is_candidate(line.strip())``
    is true AND that stripped line was already kept. Every other line passes through
    untouched (raw, keepends). Used by the experimental readable code generator.
    """
    seen: Set[str] = set()
    out: List[str] = []
    for line in code.splitlines(keepends=True):
        stripped = line.strip()
        if is_candidate(stripped):
            if stripped in seen:
                continue
            seen.add(stripped)
        out.append(line)
    return ''.join(out)


def deduplicate_includes(code: str) -> str:
    """Removes repeated ``#include`` directives (keeping the first occurrence of each)."""
    return deduplicate_lines(code, lambda s: s.startswith('#include'))


def deduplicate_functions(code: str) -> str:
    """
    Removes repeated single-line index / size helper definitions (keeping the first).
    Each helper is a file-scope ``static`` free function on one line, e.g.
    ``static DACE_HDFI constexpr long long A_idx(...) { return ...; }``. A non-inline
    nested-SDFG function has its own function stream that shares the output file with the
    outer stream, so the identical definition can appear more than once; C++ forbids the
    redefinition. Matching is conservative -- only single-line ``static`` definitions naming
    an ``*_idx`` / ``*_size`` helper are considered, and a line is dropped only when
    byte-identical to one already kept, so call sites and other code are never touched.
    """
    return deduplicate_lines(
        code,
        lambda s: s.startswith('static ') and ('_idx(' in s or '_size(' in s) and 'return' in s and s.endswith('}'))


def split_leading_includes(lines: List[str]) -> Tuple[List[str], List[str]]:
    """
    Splits a generated source into (leading header block, body). The header block
    is the contiguous run of comments / blank lines / preprocessor directives
    (``#include``, ``#pragma``, ``#define``, ...) at the top of the file, before
    the first line of real code.
    """
    split = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == '' or stripped.startswith(('//', '/*', '*', '#')):
            split = i + 1
        else:
            break
    return lines[:split], lines[split:]


# Only readability-* fixes safe on include-stripped code (apply_clang_tidy strips the header block,
# so a fix depending on types or a variable's full use-set rewrites on a half-parse). Excluded:
# identifier naming/length, magic-numbers, cognitive-complexity, uppercase-suffix (noise);
# non-const-parameter (const-qualifies a pointer only forwarded to a nested-SDFG writer -> const vs
# non-const clash nvcc rejects); and modernize-* (type-dependent -> miscompiled the CUDA
# block-reduction: an empty ``using`` alias and a reduction index turned into a range-for value).
CLANG_TIDY_CHECKS = ('readability-*,'
                     '-readability-identifier-naming,-readability-identifier-length,-readability-magic-numbers,'
                     '-readability-function-cognitive-complexity,-readability-uppercase-literal-suffix,'
                     '-readability-avoid-const-params-in-decls,-readability-non-const-parameter')


def apply_clang_tidy(code_path: str) -> None:
    """
    Best-effort standalone ``clang-tidy -fix-errors`` on a generated ``.cpp`` /
    ``.cu`` file to improve readability, applied in place -- no CMake / compilation
    database. Only the vetted ``CLANG_TIDY_CHECKS`` run: a fix that needs types or a
    variable's full use-set cannot be trusted here (see that constant).

    The leading ``#include`` block is stripped before tidying and restored after,
    so clang-tidy never parses any header at all: no DaCe runtime, CUDA, cuBLAS,
    or OpenBLAS include path is needed (GPU/vendor libraries do not all live under
    the CUDA prefix, so relying on include discovery would be fragile). External
    functions and types then appear undeclared; ``-fix-errors`` tolerates those as
    black boxes -- their call sites are left intact while the surrounding readable
    code (loops, index functions, tasklets) is tidied. This is also faster, since
    the large runtime headers are not re-parsed.

    Never fails the build: a missing binary or a tidy error only emits a warning.
    """
    tidy = shutil.which('clang-tidy')
    if tidy is None:
        warnings.warn('clang-tidy not found; skipping tidy pass')
        return
    try:
        with open(code_path) as fh:
            lines = fh.readlines()
    except OSError as ex:
        warnings.warn(f'clang-tidy: could not read {code_path}: {ex}')
        return

    header, body = split_leading_includes(lines)
    tmp_path = code_path + '.tidytmp'
    checks = CLANG_TIDY_CHECKS
    # Tidy at the configured C++ standard (the same value CMake compiles with, see
    # DACE_CPP_STANDARD), so a fix is never applied under a different standard than the code
    # is built with.
    std_arg = '-std=c++%s' % str(Config.get('compiler', 'cpp_standard')).strip()
    lang_args = [std_arg]
    if code_path.endswith('.cu'):
        lang_args = ['-x', 'cuda', '--cuda-host-only', '--no-cuda-version-check', std_arg]
    try:
        with open(tmp_path, 'w') as fh:
            fh.writelines(body)
        subprocess.run([
            tidy, '-quiet', '-fix-errors', f'--header-filter={re.escape(os.path.basename(tmp_path))}',
            '-system-headers=0', f'-checks=-*,{checks}', tmp_path, '--'
        ] + lang_args,
                       capture_output=True,
                       text=True,
                       timeout=180)
        with open(tmp_path) as fh:
            tidied_body = fh.readlines()
        with open(code_path, 'w') as fh:
            fh.writelines(header + tidied_body)
    except (subprocess.SubprocessError, OSError) as ex:
        warnings.warn(f'clang-tidy failed to run: {ex}')
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


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

        # The experimental (readable) code generator produces human-oriented code;
        # collapse duplicate headers and format by default for readability.
        readable = Config.get('compiler', 'cpu', 'implementation') == 'experimental_readable'

        if readable:
            clean_code = deduplicate_includes(clean_code)
            clean_code = deduplicate_functions(clean_code)

        if Config.get_bool('compiler', 'format_code') or readable:
            config_file = Config.get('compiler', 'format_config_file')
            if config_file is not None and config_file != "":
                run_arg_list = ['clang-format', f"-style=file:{config_file}"]
            else:
                run_arg_list = ['clang-format']
            try:
                result = subprocess.run(run_arg_list, input=clean_code, text=True, capture_output=True)
                if result.returncode or result.stderr:
                    warnings.warn(f'clang-format failed to run: {result.stderr}')
                else:
                    clean_code = result.stdout
            except FileNotFoundError:
                warnings.warn('clang-format not found; skipping code formatting')

        # Save the file only if it changed (keeps old timestamps and saves
        # build time)
        if not identical_file_exists(code_path, clean_code):
            with open(code_path, "w") as code_file:
                code_file.write(clean_code)

        # Readability tidy-up of the generated CPU (.cpp) and GPU (.cu) files,
        # standalone (no CMake). Run automatically by the experimental readable
        # generator. Best-effort: never fails the build; a missing clang-tidy is
        # a no-op.
        if readable and extension in ('cpp', 'cu'):
            apply_clang_tidy(code_path)

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
        # Copy a full snapshot of configuration script
        Config.save(os.path.join(out_path, "dace.conf"), all=True)

    # The folder mode file is always generated. In case it is missing we assume the old version.
    with open(os.path.join(out_path, "FOLDER_MODE"), "w") as version_file:
        version_file.write(folder_mode)

    return out_path


#: Environment-variable prefixes an MPI/PMI launcher (srun, mpirun) exports to mark a
#: process as a rank of its job. A child that inherits these and links a PMI/PMIx client
#: (directly, or transitively through an MPI-wrapper compiler) treats itself as that rank
#: and blocks in MPI_Init/PMIx_Init awaiting a rendezvous that never comes.
_MPI_RANK_ENV_PREFIXES = (
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


def _build_subprocess_env():
    """`os.environ` with this process's MPI-rank identity stripped, for the CMake
    configure/build subprocesses.

    When DaCe compiles from inside a process launched by an MPI/PMI launcher, CMake --
    and the try_compile test binaries, make/ninja, and the compiler driver it spawns --
    inherit the launcher's rank-identity variables (``_MPI_RANK_ENV_PREFIXES``). Any of
    those children that touches a PMI/PMIx client library then hangs in its init call
    forever (leaving defunct/zombie children), which manifests as a stuck ``cmake``.
    Compilation never needs an MPI identity, so drop those variables from the build
    environment; everything else (PATH, compiler flags, MCA tuning, ...) is preserved."""
    return {k: v for k, v in os.environ.items() if not k.startswith(_MPI_RANK_ENV_PREFIXES)}


@contextlib.contextmanager
def _build_subprocess_sigmask():
    """Temporarily unblock ``SIGCHLD`` on the calling thread so a subprocess forked
    inside this context inherits an unblocked ``SIGCHLD``.

    MPI/Slurm launchers (``srun``, ``mpirun``) start their tasks with ``SIGCHLD`` *blocked*
    in the signal mask, and every child inherits that mask. CMake (KWSys) learns that the
    helper processes it spawns during *configure* -- ``uname`` for system introspection,
    the compiler-id / ABI test binaries, ``make``/``ninja`` -- have finished by receiving
    ``SIGCHLD``; with ``SIGCHLD`` blocked it is never woken to reap them, so ``cmake`` spins
    forever in ``select()`` leaving ``<defunct>`` children. That is the daint compile hang:
    it looks like a stuck ``cmake`` even though nothing is compiling. (Confirmed under srun:
    every task's ``/proc/self/status`` shows ``SigBlk`` with the ``SIGCHLD`` bit set, and a
    trivial ``project()`` configure hangs until the child mask is cleared.)

    A child inherits the *forking thread's* mask, and ``subprocess.Popen`` forks from the
    calling thread without resetting it, so unblocking here -- immediately around the
    ``Popen`` -- is enough. ``pthread_sigmask`` is per-thread, so this never disturbs other
    threads or the process's steady-state mask, and it is restored right after the fork.
    No-op where ``pthread_sigmask``/``SIGCHLD`` are unavailable (e.g. Windows)."""
    if not hasattr(signal, 'pthread_sigmask') or not hasattr(signal, 'SIGCHLD'):
        yield
        return
    if signal.SIGCHLD not in signal.pthread_sigmask(signal.SIG_BLOCK, []):
        yield  # SIGCHLD already deliverable -- nothing to do (the common, non-launcher case)
        return
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGCHLD})
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGCHLD})


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

    if program_name is None:
        program_name = os.path.basename(program_folder)
    program_folder = os.path.abspath(program_folder)
    src_folder = os.path.join(program_folder, "src")

    # Prepare build folder
    build_folder = os.path.join(program_folder, "build")
    os.makedirs(build_folder, exist_ok=True)

    # Prepare performance report folder if requested.
    if folder_mode == "development":
        os.makedirs(os.path.join(program_folder, "perf"), exist_ok=True)

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

    # Resolve the environments the SDFG uses (shared by both build backends).
    with open(os.path.join(program_folder, "dace_environments.csv"), "r") as f:
        environments = set(l.strip() for l in f)
    environments = dace.library.get_environments_and_dependencies(environments)

    # Strip this process's MPI-rank identity from the build subprocesses so the compiler and the
    # children it spawns never hang joining the outer MPI/PMI job (see _build_subprocess_env).
    build_env = _build_subprocess_env()

    # Build the shared library either directly (native) or through CMake (default). Both write it to
    # the same development-mode location that the shared tail below expects.
    build_mode = Config.get('compiler', 'build_mode').strip().lower()
    if build_mode not in ('cmake', 'native'):
        raise cgx.CompilerConfigurationError(
            f"Unknown compiler.build_mode {Config.get('compiler', 'build_mode')!r}; expected 'cmake' or 'native'.")
    if build_mode == 'native':
        from dace.codegen import native_compiler
        native_compiler.build_native(program_folder=program_folder,
                                     program_name=program_name,
                                     files=files,
                                     targets=targets,
                                     environments=environments,
                                     build_folder=build_folder,
                                     build_env=build_env,
                                     output_stream=output_stream)
    else:
        _cmake_configure_and_build(program_folder=program_folder,
                                   program_name=program_name,
                                   src_folder=src_folder,
                                   build_folder=build_folder,
                                   files=files,
                                   targets=targets,
                                   environments=environments,
                                   build_env=build_env,
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


def _cmake_configure_and_build(program_folder,
                               program_name,
                               src_folder,
                               build_folder,
                               files,
                               targets,
                               environments,
                               build_env,
                               output_stream=None) -> None:
    """Configure and build a prepared program folder with CMake (the default ``build_mode``).

    Writes the shared library + loader stub into ``<build_folder>``; the caller
    (:func:`configure_and_compile`) locates them afterwards.
    """
    # Start forming CMake command
    dace_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Prefer the Ninja generator when it is available: the default generator on Linux is Make, whose
    # per-directory dependency graph serializes more of the build than Ninja's global one. This matters
    # most together with ``split_nsdfg_translation_units``, which exists to turn one big translation
    # unit into several that can then compile concurrently. Absence of ninja is NOT an error -- fall
    # back to the default generator, which builds the same sources with the same flags.
    # Not on Windows: the ``-A x64`` platform flag above is a Visual Studio generator option and CMake
    # rejects it for Ninja, so Windows keeps its existing generator untouched.
    use_ninja = os.name != 'nt' and shutil.which('ninja') is not None
    if not use_ninja and Config.get_bool('debugprint'):
        print('ninja not found on PATH; using the default CMake generator')

    cmake_command = [
        "cmake",
        "-A x64" if os.name == 'nt' else "",  # Windows-specific flag
        '-G Ninja' if use_ninja else "",
        '"' + os.path.join(dace_path, "codegen") + '"',
        "-DDACE_SRC_DIR=\"{}\"".format(src_folder),
        "-DDACE_FILES=\"{}\"".format(";".join(files)),
        "-DDACE_PROGRAM_NAME={}".format(program_name),
        "-DDACE_CPP_STANDARD={}".format(Config.get('compiler', 'cpp_standard')),
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
    # Additive static archive next to the .so (opt-in; matches native build mode).
    cmake_command.append(
        "-DDACE_STATIC_ARCHIVE={}".format("ON" if Config.get_bool('compiler', 'static_archive') else "OFF"))

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
    cmake_command = ' '.join(cmake_command)

    if Config.get('debugprint') == 'verbose':
        print(f'Running CMake: {cmake_command}')

    cmake_filename = os.path.join(build_folder, 'cmake_configure.sh')

    ##############################################
    # Configure
    try:
        if not identical_file_exists(cmake_filename, cmake_command):
            _run_liveoutput(cmake_command, shell=True, cwd=build_folder, output_stream=output_stream, env=build_env)
    except subprocess.CalledProcessError as ex:
        # Clean CMake directory and try once more
        if Config.get_bool('debugprint'):
            print('Cleaning CMake build folder and retrying...')
        shutil.rmtree(build_folder, ignore_errors=True)
        os.makedirs(build_folder)
        try:
            _run_liveoutput(cmake_command, shell=True, cwd=build_folder, output_stream=output_stream, env=build_env)
        except subprocess.CalledProcessError as ex:
            # If still unsuccessful, print results
            if Config.get_bool('debugprint'):
                raise cgx.CompilerConfigurationError('Configuration failure')
            else:
                raise cgx.CompilerConfigurationError('Configuration failure:\n' + ex.output)

    with open(cmake_filename, "w") as fp:
        fp.write(cmake_command)

    # Compile and link. ``cmake --build .`` drives whichever generator was configured (Ninja included),
    # so the invocation does not branch on the generator. ``--parallel`` is what actually bounds the
    # build: Make would otherwise be serial, and Ninja would otherwise use every core -- neither is what
    # we want on a shared machine. See ``compiler.build_jobs``.
    build_jobs = max(1, int(Config.get('compiler', 'build_jobs')))
    try:
        _run_liveoutput("cmake --build . --config %s --parallel %d" %
                        (Config.get('compiler', 'build_type'), build_jobs),
                        shell=True,
                        cwd=build_folder,
                        output_stream=output_stream,
                        env=build_env)
    except subprocess.CalledProcessError as ex:
        # If unsuccessful, print results
        if Config.get_bool('debugprint'):
            raise cgx.CompilationError('Compiler failure')
        else:
            raise cgx.CompilationError('Compiler failure:\n' + ex.output)


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
        raise FileNotFoundError('Compiled SDFG library not found: ' + library_path)
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
        # This is to check an old style folder, i.e. a cache folder that was generated before
        #  the `FOLDER_MODE` file was introduced. We do some small sanity checks.
        # TODO: Phase out this feature, after there are no old style caches.
        found_sub_folder = False
        for sub_folder in ["build", "map", "src", "include", "sample"]:
            if (object_folder / sub_folder).is_dir():
                found_sub_folder = True
            elif found_sub_folder:
                # A partial / corrupted cache (some sibling dirs missing). Under ``probe`` the
                # caller wants to know whether this is a usable cache; report "not usable" so
                # the next step regenerates from scratch instead of crashing the build.
                if probe:
                    return None
                raise NotADirectoryError(f'Expected that folder ``{object_folder}`` contains ``{sub_folder}``')

        if found_sub_folder:
            # All expected folders where found, so expect that this is a 'development' format folder.
            return "development"
        elif probe:
            # None of the files where found. Thus this is probably an empty folder that just exist.
            return None
        else:
            # Up for discussion what to do here.
            raise NotADirectoryError(f'``{object_folder}`` does not appear to be a valid build folder.')


def get_binary_name(
    object_folder: Union[pathlib.Path, str],
    sdfg_name: str,
    lib_extension: Optional[str] = None,
    folder_mode: Optional[str] = None,
) -> pathlib.Path:
    """Returns the supposed location of the compiled library given the boundary conditions.

    :param object_folder: The build folder of the SDFG, i.e. `sdfg.build_folder`.
    :param sdfg_name: The name of the SDFG, i.e. `sdfg.name`.
    :param lib_extension: The extension of the library, i.e. file extension.
                          If not given the config option `compiler.library_extension` is used.
    :param folder_mode: The save mode for the build folder. If not given the config
                        option `compiler.build_folder_mode` is used.
    """
    if lib_extension is None:
        lib_extension = Config.get('compiler', 'library_extension')
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
    auxiliary_sources = set()
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
        # Optional ``auxiliary_sources`` field (introduced for library nodes that
        # need a nvcc-compiled ``.cu`` translation unit alongside the SDFG's
        # ``.cpp`` host file -- e.g. the strided ``Scan`` libnode's GPU
        # wrappers). Existing environments without the field continue to work
        # via ``getattr``'s default. Paths are made absolute relative to the
        # environment's file location, matching the ``cmake_files`` convention.
        env_aux = _get_or_eval(getattr(env, 'auxiliary_sources', []))
        auxiliary_sources |= set(s if os.path.isabs(s) else os.path.join(env_dir, s) for s in env_aux)
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
        # Auxiliary source files (``.cu`` translation units, in current usage)
        # to add to the SDFG library target so they are compiled with the right
        # language (e.g. nvcc for ``.cu``) alongside the SDFG's own sources.
        "-DDACE_ENV_AUXILIARY_SOURCES=\"{}\"".format(";".join(sorted(auxiliary_sources))),
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


def _run_liveoutput(command, output_stream=None, **kwargs):
    # Fork the build subprocess (CMake) with SIGCHLD unblocked so it can reap the helper
    # processes it spawns during configure; an MPI/Slurm launcher blocks SIGCHLD in the
    # inherited mask, which otherwise deadlocks cmake in select() (see
    # _build_subprocess_sigmask). Only the fork needs to happen inside the context.
    with _build_subprocess_sigmask():
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
