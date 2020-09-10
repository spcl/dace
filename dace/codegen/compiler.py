# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
""" Handles compilation of code objects. Creates the proper folder structure,
    compiles each target separately, links all targets to one binary, and
    returns the corresponding CompiledSDFG object. """

from __future__ import print_function

import ctypes
import os
import six
import shutil
import subprocess
import re
from typing import Any, Dict, List
import numpy as np
import sympy as sp
import warnings

import dace
from dace.frontend import operations
from dace import symbolic, data as dt, dtypes
from dace.config import Config
from dace.codegen.targets.target import TargetCodeGenerator
from dace.codegen.codeobject import CodeObject
from dace.codegen.targets.target import make_absolute


# Specialized exception classes
class DuplicateDLLError(Exception):
    """ An exception that is raised whenever a library is loaded twice. """
    pass


class CompilerConfigurationError(Exception):
    """ An exception that is raised whenever CMake encounters a configuration
        error. """
    pass


class CompilationError(Exception):
    """ An exception that is raised whenever a compilation error occurs. """
    pass


class ReloadableDLL(object):
    """ A reloadable shared object (or dynamically linked library), which
        bypasses Python's dynamic library reloading issues. """
    def __init__(self, library_filename, program_name):
        """ Creates a new reloadable shared object.
            :param library_filename: Path to library file.
            :param program_name: Name of the DaCe program (for use in finding
                                 the stub library loader).
        """
        self._stub_filename = os.path.join(
            os.path.dirname(os.path.realpath(library_filename)),
            'libdacestub_%s.%s' %
            (program_name, Config.get('compiler', 'library_extension')))
        self._library_filename = os.path.realpath(library_filename)
        self._stub = None
        self._lib = None

    def get_symbol(self, name, restype=ctypes.c_int):
        """ Returns a symbol (e.g., function name) in the loaded library. """

        if self._lib is None or self._lib.value is None:
            raise ReferenceError('ReloadableDLL can only be used with a ' +
                                 '"with" statement or with load() and unload()')

        func = self._stub.get_symbol(self._lib, ctypes.c_char_p(name.encode()))
        if func is None:
            raise KeyError('Function %s not found in library %s' %
                           (name, os.path.basename(self._library_filename)))

        return ctypes.CFUNCTYPE(restype)(func)

    def load(self):
        """ Loads the internal library using the stub. """

        # If internal library is already loaded, skip
        if self._lib is not None and self._lib.value is not None:
            return
        self._stub = ctypes.CDLL(self._stub_filename)

        # Set return types of stub functions
        self._stub.load_library.restype = ctypes.c_void_p
        self._stub.get_symbol.restype = ctypes.c_void_p

        # Check if library is already loaded
        is_loaded = True
        lib_cfilename = None
        while is_loaded:
            # Convert library filename to string according to OS
            if os.name == 'nt':
                # As UTF-16
                lib_cfilename = ctypes.c_wchar_p(self._library_filename)
            else:
                # As UTF-8
                lib_cfilename = ctypes.c_char_p(
                    self._library_filename.encode('utf-8'))

            is_loaded = self._stub.is_library_loaded(lib_cfilename)
            if is_loaded == 1:
                warnings.warn('Library %s already loaded, renaming file' %
                              self._library_filename)
                try:
                    shutil.copyfile(self._library_filename,
                                    self._library_filename + '_')
                    self._library_filename += '_'
                except shutil.Error:
                    raise DuplicateDLLError(
                        'Library %s is already loaded somewhere else ' %
                        os.path.basename(self._library_filename) +
                        'and cannot be unloaded. Please use a different name ' +
                        'for the SDFG/program.')

        # Actually load the library
        self._lib = ctypes.c_void_p(self._stub.load_library(lib_cfilename))

        if self._lib.value is None:
            # Try to understand why the library is not loading, if dynamic
            # linker is used
            reason = ''
            if os.name == 'posix':
                result = subprocess.run(['ld', self._library_filename],
                                        capture_output=True)
                stderr = result.stderr.decode('utf-8')
                reason = 'Reason:\n' + '\n'.join(
                    [l for l in stderr.split('\n') if '_start' not in l])
            raise RuntimeError(
                'Could not load library %s. %s' %
                (os.path.basename(self._library_filename), reason))

    def unload(self):
        """ Unloads the internal library using the stub. """

        if self._stub is None:
            return

        self._stub.unload_library(self._lib)
        self._lib = None
        del self._stub
        self._stub = None

    def __enter__(self, *args, **kwargs):
        self.load()
        return self

    def __exit__(self, *args, **kwargs):
        self.unload()


class CompiledSDFG(object):
    """ A compiled SDFG object that can be called through Python. """
    def __init__(self, sdfg, lib: ReloadableDLL):
        self._sdfg = sdfg
        self._lib = lib
        self._initialized = False
        self._lastargs = ()
        self._return_arrays: List[np.ndarray] = []
        self._return_kwarrays: Dict[str, np.ndarray] = {}
        self._return_syms: Dict[str, Any] = {}
        lib.load()  # Explicitly load the library
        self._init = lib.get_symbol('__dace_init_{}'.format(sdfg.name))
        self._exit = lib.get_symbol('__dace_exit_{}'.format(sdfg.name))
        self._cfunc = lib.get_symbol('__program_{}'.format(sdfg.name))

    @property
    def filename(self):
        return self._lib._library_filename

    @property
    def sdfg(self):
        return self._sdfg

    def __del__(self):
        if self._initialized is True:
            self.finalize(*self._lastargs)
            self._initialized = False
        self._lib.unload()

    def _construct_args(self, **kwargs):
        """ Main function that controls argument construction for calling
            the C prototype of the SDFG.

            Organizes arguments first by `sdfg.arglist`, then data descriptors
            by alphabetical order, then symbols by alphabetical order.
        """
        # Return value initialization (for values that have not been given)
        kwargs.update({
            k: v
            for k, v in self._initialize_return_values(kwargs).items()
            if k not in kwargs
        })

        # Argument construction
        sig = self._sdfg.signature_arglist(with_types=False)
        typedict = self._sdfg.arglist()
        if len(kwargs) > 0:
            # Construct mapping from arguments to signature
            arglist = []
            argtypes = []
            argnames = []
            for a in sig:
                try:
                    arglist.append(kwargs[a])
                    argtypes.append(typedict[a])
                    argnames.append(a)
                except KeyError:
                    raise KeyError("Missing program argument \"{}\"".format(a))
        else:
            arglist = []
            argtypes = []
            argnames = []
            sig = []

        # Type checking
        for a, arg, atype in zip(argnames, arglist, argtypes):
            if not dtypes.is_array(arg) and isinstance(atype, dt.Array):
                raise TypeError(
                    'Passing an object (type %s) to an array in argument "%s"' %
                    (type(arg).__name__, a))
            elif dtypes.is_array(arg) and not isinstance(atype, dt.Array):
                raise TypeError(
                    'Passing an array to a scalar (type %s) in argument "%s"' %
                    (atype.dtype.ctype, a))
            elif not isinstance(atype, dt.Array) and not isinstance(
                    atype.dtype, dace.callback) and not isinstance(
                        arg, (atype.dtype.type, sp.Basic)) and not (isinstance(
                            arg, symbolic.symbol) and arg.dtype == atype.dtype):
                if isinstance(arg, int) and atype.dtype.type == np.int64:
                    pass
                elif isinstance(arg, float) and atype.dtype.type == np.float64:
                    pass
                else:
                    print(
                        'WARNING: Casting scalar argument "%s" from %s to %s' %
                        (a, type(arg).__name__, atype.dtype.type))
            elif (isinstance(atype, dt.Array) and isinstance(arg, np.ndarray)
                  and atype.dtype.as_numpy_dtype() != arg.dtype):
                # Make exception for vector types
                if (isinstance(atype.dtype, dtypes.vector)
                        and atype.dtype.vtype.as_numpy_dtype() != arg.dtype):
                    print(
                        'WARNING: Passing %s array argument "%s" to a %s array'
                        % (arg.dtype, a, atype.dtype.type.__name__))

        # Call a wrapper function to make NumPy arrays from pointers.
        for index, (arg, argtype) in enumerate(zip(arglist, argtypes)):
            if isinstance(argtype.dtype, dace.callback):
                arglist[index] = argtype.dtype.get_trampoline(arg, kwargs)

        # Retain only the element datatype for upcoming checks and casts
        arg_ctypes = [t.dtype.as_ctypes() for t in argtypes]

        sdfg = self._sdfg

        # Obtain SDFG constants
        constants = sdfg.constants

        # Remove symbolic constants from arguments
        callparams = tuple(
            (arg, actype, atype)
            for arg, actype, atype in zip(arglist, arg_ctypes, argtypes)
            if not symbolic.issymbolic(arg) or (
                hasattr(arg, 'name') and arg.name not in constants))

        # Replace symbols with their values
        callparams = tuple(
            (actype(arg.get()), actype,
             atype) if isinstance(arg, symbolic.symbol) else (arg, actype,
                                                              atype)
            for arg, actype, atype in callparams)

        # Replace arrays with their base host/device pointers
        newargs = tuple(
            (ctypes.c_void_p(_array_interface_ptr(arg, atype)), actype,
             atype) if dtypes.is_array(arg) else (arg, actype, atype)
            for arg, actype, atype in callparams)

        newargs = tuple(
            actype(arg) if (not isinstance(arg, ctypes._SimpleCData)) else arg
            for arg, actype, atype in newargs)

        self._lastargs = newargs
        return self._lastargs

    def _initialize_return_values(self, kwargs):
        # Obtain symbol values from arguments and constants
        syms = dict()
        syms.update(
            {k: v
             for k, v in kwargs.items() if k not in self.sdfg.arrays})
        syms.update(self.sdfg.constants)

        if self._initialized:
            if self._return_syms == syms:
                return self._return_kwarrays

        self._return_syms = syms

        # Initialize return values with numpy arrays
        self._return_arrays = []
        self._return_kwarrays = {}
        for arrname, arr in sorted(self.sdfg.arrays.items()):
            if arrname.startswith('__return'):
                if isinstance(arr, dt.Stream):
                    raise NotImplementedError('Return streams are unsupported')
                if arr.storage in [
                        dace.dtypes.StorageType.GPU_Global,
                        dace.dtypes.StorageType.FPGA_Global
                ]:
                    raise NotImplementedError('Non-host return values are '
                                              'unsupported')

                # Create an array with the properties of the SDFG array
                self._return_arrays.append(
                    np.ndarray([symbolic.evaluate(s, syms) for s in arr.shape],
                               arr.dtype.as_numpy_dtype(),
                               buffer=np.zeros(
                                   [symbolic.evaluate(arr.total_size, syms)],
                                   arr.dtype.as_numpy_dtype()),
                               strides=[
                                   symbolic.evaluate(s, syms) * arr.dtype.bytes
                                   for s in arr.strides
                               ]))
                self._return_kwarrays[arrname] = self._return_arrays[-1]

        # Set up return_arrays field
        if len(self._return_arrays) == 0:
            self._return_arrays = None
        elif len(self._return_arrays) == 1:
            self._return_arrays = self._return_arrays[0]
        else:
            self._return_arrays = tuple(self._return_arrays)

        return self._return_kwarrays

    def initialize(self, *argtuple):
        if self._init is not None:
            res = self._init(*argtuple)
            if res != 0:
                raise RuntimeError('DaCe application failed to initialize')

        self._initialized = True

    def finalize(self, *argtuple):
        if self._exit is not None:
            self._exit(*argtuple)

    def __call__(self, **kwargs):
        try:
            argtuple = self._construct_args(**kwargs)

            # Call initializer function if necessary, then SDFG
            if self._initialized is False:
                self._lib.load()
                self.initialize(*argtuple)

            # PROFILING
            if Config.get_bool('profiling'):
                operations.timethis(self._sdfg.name, 'DaCe', 0, self._cfunc,
                                    *argtuple)
            else:
                self._cfunc(*argtuple)

            return self._return_arrays
        except (RuntimeError, TypeError, UnboundLocalError, KeyError,
                DuplicateDLLError, ReferenceError):
            self._lib.unload()
            raise


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


def generate_program_folder(sdfg,
                            code_objects: List[CodeObject],
                            out_path: str,
                            config=None):
    """ Writes all files required to configure and compile the DaCe program
        into the specified folder.

        :param sdfg: The SDFG to generate the program folder for.
        :param code_objects: List of generated code objects.
        :param out_path: The folder in which the build files should be written.
        :return: Path to the program folder.
    """

    src_path = os.path.join(out_path, "src")

    os.makedirs(src_path, exist_ok=True)

    filelist = []
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
        clean_code = re.sub(r'[ \t]*////__DACE:[^\n]*', '', code_object.code)

        # Save the file only if it changed (keeps old timestamps and saves
        # build time)
        if not identical_file_exists(code_path, clean_code):
            with open(code_path, "w") as code_file:
                code_file.write(clean_code)

        if code_object.linkable == True:
            filelist.append("{},{},{}".format(target_name, target_type,
                                              basename))

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
        # Save the SDFG itself
        sdfg.save(os.path.join(out_path, "program.sdfg"))

    return out_path


def configure_and_compile(program_folder,
                          program_name=None,
                          output_stream=None):
    """ Configures and compiles a DaCe program in the specified folder into a
        shared library file.

        :param program_folder: Folder containing all files necessary to build,
                               equivalent to what was passed to
                               `generate_program_folder`.
        :param output_stream: Additional output stream to write to (used for
                              DIODE client).
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
    file_list = [
        line.strip().split(",")
        for line in open(os.path.join(program_folder, "dace_files.csv"), "r")
    ]

    # Get absolute paths and targets for all source files
    files = []
    targets = {}  # {target name: target class}
    for target_name, target_type, file_name in file_list:
        if target_type:
            path = os.path.join(target_name, target_type, file_name)
        else:
            path = os.path.join(target_name, file_name)
        files.append(path)
        targets[target_name] = next(
            k for k, v in TargetCodeGenerator.extensions().items()
            if v['name'] == target_name)

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
    environments = set(l.strip() for l in open(
        os.path.join(program_folder, "dace_environments.csv"), "r"))
    cmake_minimum_version = [0]
    cmake_variables = dict()
    cmake_packages = set()
    cmake_includes = set()
    cmake_libraries = set()
    cmake_compile_flags = set()
    cmake_link_flags = set()
    cmake_files = set()
    cmake_module_paths = set()
    for env_name in environments:
        env = dace.library.get_environment(env_name)
        if (env.cmake_minimum_version is not None
                and len(env.cmake_minimum_version) > 0):
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
        for var in env.cmake_variables:
            if (var in cmake_variables
                    and cmake_variables[var] != env.cmake_variables[var]):
                raise KeyError(
                    "CMake variable {} was redefined from {} to {}.".format(
                        var, cmake_variables[var], env.cmake_variables[var]))
            cmake_variables[var] = env.cmake_variables[var]
        cmake_packages |= set(env.cmake_packages)
        cmake_includes |= set(env.cmake_includes)
        cmake_libraries |= set(env.cmake_libraries)
        cmake_compile_flags |= set(env.cmake_compile_flags)
        cmake_link_flags |= set(env.cmake_link_flags)
        # Make path absolute
        env_dir = os.path.dirname(env._dace_file_path)
        cmake_files |= set(
            (f if os.path.isabs(f) else os.path.join(env_dir, f)) +
            (".cmake" if not f.endswith(".cmake") else "")
            for f in env.cmake_files)
        for header in env.headers:
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
        "-DDACE_ENV_MINIMUM_VERSION={}".format(".".join(
            map(str, cmake_minimum_version))),
        # Make CMake list of key-value pairs
        "-DDACE_ENV_VAR_KEYS=\"{}\"".format(";".join(cmake_variables.keys())),
        "-DDACE_ENV_VAR_VALUES=\"{}\"".format(";".join(
            cmake_variables.values())),
        "-DDACE_ENV_PACKAGES=\"{}\"".format(" ".join(cmake_packages)),
        "-DDACE_ENV_INCLUDES=\"{}\"".format(" ".join(cmake_includes)),
        "-DDACE_ENV_LIBRARIES=\"{}\"".format(" ".join(cmake_libraries)),
        "-DDACE_ENV_COMPILE_FLAGS=\"{}\"".format(" ".join(cmake_compile_flags)),
        # "-DDACE_ENV_LINK_FLAGS=\"{}\"".format(" ".join(cmake_link_flags)),
        "-DDACE_ENV_CMAKE_FILES=\"{}\"".format(";".join(cmake_files)),
    ]
    # Escape variable expansions to defer their evaluation
    environment_flags = [
        cmd.replace("$", "_DACE_CMAKE_EXPAND") for cmd in environment_flags
    ]
    cmake_command += environment_flags

    # Replace backslashes with forward slashes
    cmake_command = [cmd.replace('\\', '/') for cmd in cmake_command]

    # Generate CMake options for each compiler
    libraries = set()
    for target_name, target in targets.items():
        try:
            cmake_command += target.cmake_options()
            libraries |= unique_flags(
                Config.get("compiler", target_name, "libs"))
        except KeyError:
            pass
        except ValueError as ex:  # Cannot find compiler executable
            raise CompilerConfigurationError(str(ex))

    cmake_command.append("-DDACE_LIBS=\"{}\"".format(" ".join(libraries)))

    # Override linker and linker arguments
    if Config.get('compiler', 'linker', 'executable'):
        cmake_command.append("-DCMAKE_LINKER=\"{}\"".format(
            make_absolute(Config.get('compiler', 'linker', 'executable'))))
    if Config.get('compiler', 'linker', 'args'):
        cmake_command.append(
            "-DCMAKE_SHARED_LINKER_FLAGS=\"{}\"".format(
                Config.get('compiler', 'linker', 'args') + " " +
                " ".join(cmake_link_flags)), )
    cmake_command = ' '.join(cmake_command)

    cmake_filename = os.path.join(build_folder, 'cmake_configure.sh')
    ##############################################
    # Configure
    try:
        _run_liveoutput(cmake_command,
                        shell=True,
                        cwd=build_folder,
                        output_stream=output_stream)
    except subprocess.CalledProcessError as ex:
        # Clean CMake directory and try once more
        if Config.get_bool('debugprint'):
            print('Cleaning CMake build folder and retrying...')
        shutil.rmtree(build_folder)
        os.makedirs(build_folder)
        try:
            _run_liveoutput(cmake_command,
                            shell=True,
                            cwd=build_folder,
                            output_stream=output_stream)
        except subprocess.CalledProcessError as ex:
            # If still unsuccessful, print results
            if Config.get_bool('debugprint'):
                raise CompilerConfigurationError('Configuration failure')
            else:
                raise CompilerConfigurationError('Configuration failure:\n' +
                                                 ex.output)

        with open(cmake_filename, "w") as fp:
            fp.write(cmake_command)

    # Compile and link
    try:
        _run_liveoutput("cmake --build . --config %s" %
                        (Config.get('compiler', 'build_type')),
                        shell=True,
                        cwd=build_folder,
                        output_stream=output_stream)
    except subprocess.CalledProcessError as ex:
        # If unsuccessful, print results
        if Config.get_bool('debugprint'):
            raise CompilationError('Compiler failure')
        else:
            raise CompilationError('Compiler failure:\n' + ex.output)

    shared_library_path = os.path.join(
        build_folder,
        "lib{}.{}".format(program_name,
                          Config.get('compiler', 'library_extension')))

    return shared_library_path


def get_program_handle(library_path, sdfg):
    lib = ReloadableDLL(library_path, sdfg.name)
    # Load and return the compiled function
    return CompiledSDFG(sdfg, lib)


def load_from_file(sdfg, binary_filename):
    if not os.path.isfile(binary_filename):
        raise FileNotFoundError('File not found: ' + binary_filename)

    # Load the generated library
    lib = ReloadableDLL(binary_filename, sdfg.name)

    # Load and return the compiled function
    return CompiledSDFG(sdfg, lib)


def get_binary_name(object_folder,
                    object_name,
                    lib_extension=Config.get('compiler', 'library_extension')):
    name = None
    name = os.path.join(object_folder, "build",
                        'lib%s.%s' % (object_name, lib_extension))
    return name


def _run_liveoutput(command, output_stream=None, **kwargs):
    process = subprocess.Popen(command,
                               stderr=subprocess.STDOUT,
                               stdout=subprocess.PIPE,
                               **kwargs)
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
        raise subprocess.CalledProcessError(process.returncode, command,
                                            output.getvalue())


def _array_interface_ptr(array: Any, array_type: dt.Array) -> int:
    """
    If the given array implements ``__array_interface__`` (see
    ``dtypes.is_array``), returns the base host or device pointer to the
    array's allocated memory.
    :param array: Array object that implements NumPy's array interface.
    :param array_type: Data descriptor of the array (used to get storage
                       location to determine whether it's a host or GPU device
                       pointer).
    :return: A pointer to the base location of the allocated buffer.
    """
    if hasattr(array, 'data_ptr'):
        return array.data_ptr()
    if array_type.storage == dace.StorageType.GPU_Global:
        return array.__cuda_array_interface__['data'][0]
    return array.__array_interface__['data'][0]


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
