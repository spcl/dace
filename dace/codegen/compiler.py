# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Handles compilation of code objects. Creates the proper folder structure,
    compiles each target separately, links all targets to one binary, and
    returns the corresponding CompiledSDFG object. """

from __future__ import print_function

import collections
import os
import six
import shutil
import subprocess
import re
from typing import Any, Callable, Dict, List, Set, Tuple, TypeVar, Union

import dace
from dace.config import Config
from dace.codegen import exceptions as cgx
from dace.codegen.targets.target import TargetCodeGenerator
from dace.codegen.codeobject import CodeObject
from dace.codegen import compiled_sdfg as csd
from dace.codegen.targets.target import make_absolute

T = TypeVar('T')


def generate_program_folder(sdfg, code_objects: List[CodeObject], out_path: str, config=None):
    """ Writes all files required to configure and compile the DaCe program
        into the specified folder.

        :param sdfg: The SDFG to generate the program folder for.
        :param code_objects: List of generated code objects.
        :param out_path: The folder in which the build files should be written.
        :return: Path to the program folder.
    """

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

        # Save the file only if it changed (keeps old timestamps and saves
        # build time)
        if not identical_file_exists(code_path, clean_code):
            with open(code_path, "w") as code_file:
                code_file.write(clean_code)

        if code_object.linkable == True:
            filelist.append("{},{},{}".format(target_name, target_type, basename))

    # Write list of files
    with open(os.path.join(out_path, "dace_files.csv"), "w") as filelist_file:
        filelist_file.write("\n".join(filelist))

    # Build a list of environments used
    environments = set()
    for obj in code_objects:
        environments |= obj.environments

    # Write list of environments
    with open(os.path.join(out_path, "dace_environments.csv"), "w") as env_file:
        env_file.write("\n".join(environments))

    # Copy snapshot of configuration script
    if config is not None:
        config.save(os.path.join(out_path, "dace.conf"))
    else:
        Config.save(os.path.join(out_path, "dace.conf"))

    if sdfg is not None:
        # Save the SDFG itself and its hash
        hash = sdfg.save(os.path.join(out_path, "program.sdfg"), hash=True)
        filepath = os.path.join(out_path, 'include', 'hash.h')
        contents = f'#define __HASH_{sdfg.name} "{hash}"\n'
        if not identical_file_exists(filepath, contents):
            with open(filepath, 'w') as hfile:
                hfile.write(contents)

    return out_path


def configure_and_compile(program_folder, program_name=None, output_stream=None):
    """ Configures and compiles a DaCe program in the specified folder into a
        shared library file.

        :param program_folder: Folder containing all files necessary to build,
                               equivalent to what was passed to
                               `generate_program_folder`.
        :param output_stream: Additional output stream to write to (used for
                              other clients such as the vscode extension).
        :return: Path to the compiled shared library file.
    """

    if program_name is None:
        program_name = os.path.basename(program_folder)
    program_folder = os.path.abspath(program_folder)
    src_folder = os.path.join(program_folder, "src")

    # Prepare build folder
    build_folder = os.path.join(program_folder, "build")
    os.makedirs(build_folder, exist_ok=True)

    # Prepare performance report folder
    os.makedirs(os.path.join(program_folder, "perf"), exist_ok=True)

    # Read list of DaCe files to compile.
    # We do this instead of iterating over source files in the directory to
    # avoid globbing files from previous compilations, such that we don't need
    # to wipe the directory for every compilation.
    file_list = [line.strip().split(",") for line in open(os.path.join(program_folder, "dace_files.csv"), "r")]

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
    ]

    # Get required environments are retrieve the CMake information
    environments = set(l.strip() for l in open(os.path.join(program_folder, "dace_environments.csv"), "r"))

    environments = dace.library.get_environments_and_dependencies(environments)

    environment_flags, cmake_link_flags = get_environment_flags(environments)
    cmake_command += sorted(environment_flags)

    # Replace backslashes with forward slashes
    cmake_command = [cmd.replace('\\', '/') for cmd in cmake_command]

    # Generate CMake options for each compiler
    libraries = set()
    for target_name, target in sorted(targets.items()):
        try:
            cmake_command += target.cmake_options()
            libraries |= unique_flags(Config.get("compiler", target_name, "libs"))
        except KeyError:
            pass
        except ValueError as ex:  # Cannot find compiler executable
            raise cgx.CompilerConfigurationError(str(ex))

    cmake_command.append("-DDACE_LIBS=\"{}\"".format(" ".join(sorted(libraries))))

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
        shutil.rmtree(build_folder)
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

    shared_library_path = os.path.join(build_folder, "lib{}.{}".format(program_name,
                                                                       Config.get('compiler', 'library_extension')))

    return shared_library_path


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


def get_program_handle(library_path, sdfg):
    lib = csd.ReloadableDLL(library_path, sdfg.name)
    # Load and return the compiled function
    return csd.CompiledSDFG(sdfg, lib, sdfg.arg_names)


def load_from_file(sdfg, binary_filename):
    if not os.path.isfile(binary_filename):
        raise FileNotFoundError('File not found: ' + binary_filename)

    # Load the generated library
    lib = csd.ReloadableDLL(binary_filename, sdfg.name)

    # Load and return the compiled function
    return csd.CompiledSDFG(sdfg, lib)


def get_binary_name(object_folder, object_name, lib_extension=Config.get('compiler', 'library_extension')):
    name = None
    name = os.path.join(object_folder, "build", 'lib%s.%s' % (object_name, lib_extension))
    return name


def _run_liveoutput(command, output_stream=None, **kwargs):
    process = subprocess.Popen(command, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, **kwargs)
    output = six.StringIO()
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
