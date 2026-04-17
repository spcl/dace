# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Handles compilation of code objects. Creates the proper folder structure,
    compiles each target separately, links all targets to one binary, and
    returns the corresponding CompiledSDFG object. """

import collections
import os
import io
import shutil
import shlex
import subprocess
import re
import pathlib
from typing import Any, Callable, Dict, List, Literal, Set, Tuple, TypeVar, Union, Optional, overload
import warnings

import dace
from dace.config import Config
from dace.codegen import exceptions as cgx
from dace.codegen.target import TargetCodeGenerator
from dace.codegen.codeobject import CodeObject
from dace.codegen import compiled_sdfg as csd
from dace.codegen.target import make_absolute

T = TypeVar('T')


def generate_program_folder(
    sdfg,
    code_objects: List[CodeObject],
    out_path: str,
    config=None,
    folder_version: Optional[str] = None,
) -> str:
    """Writes all files required to configure and compile the DaCe program into the specified folder.

    This function respects the ``compiler.build_folder_version`` configuration variable,
    thus depending on its value the content might be different. However, in any case
    the source files are always generated.

    :param sdfg: The SDFG to generate the program folder for.
    :param code_objects: List of generated code objects.
    :param out_path: The folder in which the build files should be written.
    :param folder_version: Version of the program folder that should be generated,
                           if not given ``compiler.build_folder_version`` is used.
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

    if folder_version is None:
        folder_version = Config.get('compiler', 'build_folder_version')

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
        if sdfg and (folder_version in ["development"]):
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
        if folder_version in ["development"]:
            hash = sdfg.save(os.path.join(out_path, "program.sdfgz"), hash=True, compress=True)
        else:
            hash = sdfg.hash_sdfg()
        filepath = os.path.join(out_path, 'include', 'hash.h')
        contents = f'#define __HASH_{sdfg.name} "{hash}"\n'
        if not identical_file_exists(filepath, contents):
            with open(filepath, 'w') as hfile:
                hfile.write(contents)

    # Generate the parts of the folder that are exclusive to the development folder mode.
    if folder_version in ["development"]:

        # Copy a full snapshot of configuration script
        Config.save(os.path.join(out_path, "dace.conf"), all=True)

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

    # The version file is always generated. In case it is missing we assume the old version.
    with open(os.path.join(out_path, "VERSION"), "w") as version_file:
        version_file.write(folder_version)

    return out_path


def configure_and_compile(
    program_folder,
    program_name=None,
    output_stream=None,
    folder_version: Optional[str] = None,
) -> pathlib.Path:
    """
    Configures and compiles a DaCe program in the specified folder into a shared library file.

    This function respects the ``compiler.build_folder_version`` configuration variable,
    thus depending on its value the content might be different.

    :param program_folder: Folder containing all files necessary to build, equivalent to
                           what was passed to `generate_program_folder`.
    :param output_stream: Additional output stream to write to (used for other clients
                          such as the vscode extension).
    :return: Path to the compiled shared library file.
    """

    if folder_version is None:
        folder_version = Config.get('compiler.build_folder_version')
    assert folder_version in ["development", "production"]

    if program_name is None:
        program_name = os.path.basename(program_folder)
    program_folder = os.path.abspath(program_folder)
    src_folder = os.path.join(program_folder, "src")

    # Prepare build folder
    build_folder = os.path.join(program_folder, "build")
    os.makedirs(build_folder, exist_ok=True)

    # Prepare performance report folder if requested.
    if folder_version == "development":
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

    # Start forming CMake command
    dace_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmake_command = [
        "cmake",
        "-A x64" if os.name == 'nt' else "",  # Windows-specific flag
        '"' + os.path.join(dace_path, "codegen") + '"',
        "-DDACE_SRC_DIR=\"{}\"".format(src_folder),
        "-DDACE_FILES=\"{}\"".format(";".join(files)),
        "-DDACE_PROGRAM_NAME={}".format(program_name),
        "-DDACE_CPP_STANDARD={}".format(Config.get('compiler', 'cpp_standard')),
    ]

    # Get required environments are retrieve the CMake information
    with open(os.path.join(program_folder, "dace_environments.csv"), "r") as f:
        environments = set(l.strip() for l in f)

    environments = dace.library.get_environments_and_dependencies(environments)

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
            _run_liveoutput(cmake_command, shell=True, cwd=build_folder, output_stream=output_stream)
    except subprocess.CalledProcessError as ex:
        # Clean CMake directory and try once more
        if Config.get_bool('debugprint'):
            print('Cleaning CMake build folder and retrying...')
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

    # Compile and link
    try:
        _run_liveoutput("cmake --build . --config %s" % (Config.get('compiler', 'build_type')),
                        shell=True,
                        cwd=build_folder,
                        output_stream=output_stream)
    except subprocess.CalledProcessError as ex:
        # If unsuccessful, print results
        if Config.get_bool('debugprint'):
            raise cgx.CompilationError('Compiler failure')
        else:
            raise cgx.CompilationError('Compiler failure:\n' + ex.output)

    # Get the names of the library files that were generated.
    #  Currently we are still in the `development` folder mode.
    lib_path = get_binary_name(object_folder=program_folder, sdfg_name=program_name, folder_version="development")
    libstub_path = _get_stub_library_path(lib_path)

    # In production mode, we are now deleting what we need and relocating it.
    if folder_version == "production":
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
def get_folder_version(object_folder: Union[pathlib.Path, str], probe: Literal[False] = False) -> str:
    ...


@overload
def get_folder_version(object_folder: Union[pathlib.Path, str], probe: Literal[True]) -> Optional[str]:
    ...


@overload
def get_folder_version(object_folder: Union[pathlib.Path, str], probe: bool) -> Optional[str]:
    ...


def get_folder_version(object_folder: Union[pathlib.Path, str], probe: bool = False) -> Optional[str]:
    """Inspect `object_folder` and determine which version the folder has.

    If the function find the ``VERSION`` file it will examine it to get the version.
    If the version file is absent the function assumes that it is the ``development``
    format, however, some sanity checks are performed.

    The function also has the optional argument ``probe`` if given and the folder
    version could not be inferred the function will return ``None`` instead of
    generating an error.
    """
    object_folder = pathlib.Path(object_folder)

    if not object_folder.is_dir():
        if probe:
            return None
        raise NotADirectoryError("The build folder does not exists.")

    if (object_folder / 'VERSION').exists():
        with open(object_folder / 'VERSION', 'rt') as F:
            folder_version = F.readline().strip()
        return folder_version
    else:
        # This is to check an old style folder, i.e. a cache folder that was generated before
        #  the `VERSION` file was introduced. We do some small sanity checks.
        # TODO: Phase out this feature, after there are no old style caches.
        found_sub_folder = False
        for sub_folder in ["build", "map", "src", "include", "sample"]:
            if (object_folder / sub_folder).is_dir():
                found_sub_folder = True
            elif found_sub_folder:
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
    folder_version: Optional[str] = None,
) -> pathlib.Path:
    """Returns the supposed location of the compiled library given the boundary conditions.

    :param object_folder: The build folder of the SDFG, i.e. `sdfg.build_folder`.
    :param sdfg_name: The name of the SDFG, i.e. `sdfg.name`.
    :param lib_extension: The extension of the library, i.e. file extension.
                          If not given the config option `compiler.library_extension` is used.
    :param folder_version: The version of the build folder. If not given the config option
                           `compiler.build_folder_version` is used.
    """
    if lib_extension is None:
        lib_extension = Config.get('compiler', 'library_extension')
    if folder_version is None:
        folder_version = Config.get('compiler', 'build_folder_version')

    folder_hirarchy = [object_folder]
    if folder_version == 'development':
        folder_hirarchy.append('build')
    elif folder_version == 'production':
        # Nothing to add, they are on the top.
        pass
    else:
        raise ValueError(f"Unknown folder version '{folder_version}' found.")

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
    dump file inside ``folder``. If the folder does not contain a ``VERSION`` file
    it assumes that it is an old style ``development`` folder otherwise, the information
    from ``VERSION`` is consulted.

    :param folder: Path to SDFG output folder, i.e. its build folder.
    :param sdfg: If given then ``program.sdfg(z)`` does not need to be present.
    :return: A callable CompiledSDFG object.

    :note: If ``sdfg`` is given then it is referenced by the returned ``CompiledSDFG``.
    """
    folder = pathlib.Path(folder)

    if not folder.is_dir():
        raise NotADirectoryError(f'Can not load the SDFG from folder ``{folder}``.')

    folder_version = get_folder_version(folder)

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

    return get_program_handle(library_path=get_binary_name(folder, sdfg_name=sdfg.name, folder_version=folder_version),
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


def _run_liveoutput(command, output_stream=None, **kwargs):
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
