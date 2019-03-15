#!/usr/bin/python3
""" Handles compilation of code objects. Creates the proper folder structure,
    compiles each target separately, links all targets to one binary, and
    returns the corresponding CompiledSDFG object. """

from __future__ import print_function

import ctypes
import os
import re
import six
import shutil
import subprocess
import string
import subprocess as sp
import re
from typing import List
import numpy as np

import dace
from dace.frontend import operations
from dace.frontend.python import ndarray
from dace import symbolic, types
from dace.config import Config
from dace.codegen import codegen
from dace.codegen.codeobject import CodeObject
from dace.codegen.targets.cpu import CPUCodeGen
from dace.codegen.targets.target import make_absolute

from dace.codegen.instrumentation.perfsettings import PerfSettings, PerfMetaInfoStatic


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
            @param library_filename: Path to library file.
            @param program_name: Name of the DaCe program (for use in finding
                                 the stub library loader).
        """
        self._stub_filename = os.path.join(
            os.path.dirname(os.path.realpath(library_filename)),
            'libdacestub_%s.%s' %
            (program_name, Config.get('compiler', 'library_extension')))
        self._library_filename = library_filename
        self._stub = None
        self._lib = None

    def get_symbol(self, name, restype=ctypes.c_int):
        """ Returns a symbol (e.g., function name) in the loaded library. """

        if self._lib is None or self._lib.value is None:
            raise ReferenceError(
                'ReloadableDLL can only be used with a ' +
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

        # Convert library filename to string according to OS
        if os.name == 'nt':
            # As UTF-16
            lib_cfilename = ctypes.c_wchar_p(self._library_filename)
        else:
            # As UTF-8
            lib_cfilename = ctypes.c_char_p(
                self._library_filename.encode('utf-8'))

        # Check if library is already loaded
        is_loaded = self._stub.is_library_loaded(lib_cfilename)
        if is_loaded == 1:
            raise DuplicateDLLError(
                'Library %s is already loaded somewhere else, ' %
                os.path.basename(self._library_filename) +
                'either unload it or use a different name ' +
                'for the SDFG/program.')

        # Actually load the library
        self._lib = ctypes.c_void_p(self._stub.load_library(lib_cfilename))

        if self._lib.value is None:
            raise RuntimeError('Could not load library %s' % os.path.basename(
                self._library_filename))

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
        lib.load()  # Explicitly load the library
        self._init = lib.get_symbol('__dace_init')
        self._exit = lib.get_symbol('__dace_exit')
        self._cfunc = lib.get_symbol('__program_{}'.format(sdfg.name))

    @property
    def sdfg(self):
        return self._sdfg

    def __del__(self):
        if self._initialized == True:
            self.finalize(*self._lastargs)
            self._initialized = False
        self._lib.unload()

    def _construct_args(self, *args, **kwargs):
        """ Main function that controls argument construction for calling
            the C prototype of the SDFG. 
            
            Organizes arguments first by `sdfg.arglist`, then data descriptors
            by alphabetical order, then symbols by alphabetical order.
        """

        if len(kwargs) > 0 and len(args) > 0:
            raise AttributeError(
                'Compiled SDFGs can only be called with either arguments ' +
                '(e.g. "program(a,b,c)") or keyword arguments ' +
                '("program(A=a,B=b)"), but not both')

        # Argument construction
        sig = []
        if len(kwargs) > 0:
            # Construct mapping from arguments to signature
            sig = self._sdfg.signature_arglist(with_types=False)
            arglist = []
            for a in sig:
                try:
                    arglist.append(kwargs[a])
                except KeyError:
                    raise KeyError("Missing kernel argument \"{}\"".format(a))
        elif len(args) > 0:
            arglist = list(args)
        else:
            arglist = []

        sdfg = self._sdfg

        # As in compilation, add symbols used in array sizes to parameters
        symparams = {}
        for symname in sdfg.undefined_symbols(False):
            # Ignore arguments (as they may not be symbols but constants,
            # see below)
            if symname in sdfg.arg_types: continue
            try:
                symval = symbolic.symbol(symname)
                symparams[symname] = symval.get()
            except UnboundLocalError:
                try:
                    symparams[symname] = kwargs[symname]
                except KeyError:
                    raise UnboundLocalError('Unassigned symbol %s' % symname)

        arglist.extend(
            [symparams[k] for k in sorted(symparams.keys()) if k not in sig])

        # Obtain SDFG constants
        constants = sdfg.constants

        # Remove symbolic constants from arguments
        callparams = tuple(
            arg for arg in arglist if not symbolic.issymbolic(arg) or (
                hasattr(arg, 'name') and arg.name not in constants))

        # Replace symbols with their values
        callparams = tuple(
            symbolic.eval(arg) if symbolic.issymbolic(arg, constants) else arg
            for arg in callparams)

        # Replace arrays with their pointers
        newargs = tuple(
            ctypes.c_void_p(arg.__array_interface__['data'][0]) if (isinstance(
                arg, ndarray.ndarray) or isinstance(arg, np.ndarray)) else arg
            for arg in callparams)

        newargs = tuple(types._FFI_CTYPES[type(arg)](arg)
                        if type(arg) in types._FFI_CTYPES else arg
                        for arg in newargs)

        self._lastargs = newargs
        return self._lastargs

    def initialize(self, *argtuple):
        if self._init is not None:
            res = self._init(*argtuple)
            if res != 0:
                raise RuntimeError('DaCe application failed to initialize')

        self._initialized = True

    def finalize(self, *argtuple):
        if self._exit is not None:
            self._exit(*argtuple)

    def __call__(self, *args, **kwargs):
        argtuple = self._construct_args(*args, **kwargs)

        # Call initializer function if necessary, then SDFG
        if self._initialized == False:
            self.initialize(*argtuple)

        # PROFILING
        if Config.get_bool('profiling'):
            operations.timethis(self._sdfg.name, 'DaCe', 0, self._cfunc,
                                *argtuple)
        else:
            return self._cfunc(*argtuple)


def unique_flags(flags):
    pattern = '[^ ]+[`\'"][^"\'`]+["\'`]|[^ ]+'
    if not isinstance(flags, str):
        flags = " ".join(flags)
    return set(re.findall(pattern, flags))


def generate_program_folder(code_objects: List[CodeObject], out_path):
    """ Writes all files required to configure and compile the DaCe program
        into the specified folder.

        @param code_objects: List of generated code objects.
        @param out_path: The folder in which the build files should be written.
        @return: Path to the program folder.
    """

    src_path = os.path.join(out_path, "src")

    try:
        os.makedirs(src_path)
    except FileExistsError:
        pass

    filelist = []
    # Write each code object to a file
    for code_object in code_objects:

        name = code_object.name
        extension = code_object.language
        target_name = code_object.target.target_name

        # Create target folder
        target_folder = os.path.join(src_path, target_name)
        try:
            os.makedirs(target_folder)
        except FileExistsError:
            pass

        # Write code to file
        basename = "{}.{}".format(name, extension)
        code_path = os.path.join(target_folder, basename)
        with open(code_path, "w") as code_file:
            clean_code = re.sub(r'[ \t]*////__DACE:[^\n]*', '',
                                code_object.code)

            if PerfSettings.perf_enable_vectorization_analysis():
                # Generate line number information from the code
                # TODO: Make per code stream
                code_object.perf_meta_info.resolve(clean_code)
            code_file.write(clean_code)

        filelist.append("{},{}".format(target_name, basename))

    # Write list of files
    with open(os.path.join(out_path, "dace_files.csv"), "w") as filelist_file:
        filelist_file.write("\n".join(filelist))

    # Copy snapshot of configuration script
    Config.save(os.path.join(out_path, "dace.conf"))

    return out_path


def configure_and_compile(program_folder, program_name=None):
    """ Configures and compiles a DaCe program in the specified folder into a
        shared library file.

        @param program_folder: Folder containing all files necessary to build,
                               equivalent to what was passed to
                               `generate_program_folder`.
        @return: Path to the compiled shared library file.
    """

    if program_name is None:
        program_name = os.path.basename(program_folder)
    program_folder = os.path.abspath(program_folder)
    src_folder = os.path.join(program_folder, "src")

    # Prepare build folder
    build_folder = os.path.join(program_folder, "build")
    try:
        os.makedirs(build_folder)
    except FileExistsError:
        pass

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
    for target_name, file_name in file_list:
        path = os.path.join(src_folder, target_name, file_name)
        files.append(path)
        targets[target_name] = codegen.STRING_TO_TARGET[target_name]

    # Start forming CMake command
    dace_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmake_command = [
        "cmake",
        "-A x64" if os.name == 'nt' else "",  # Windows-specific flag
        '"' + os.path.join(dace_path, "codegen") + '"',
        "-DDACE_FILES=\"{}\"".format(";".join(files)),
        "-DDACE_PROGRAM_NAME={}".format(program_name),
    ]

    # Replace backslashes with forward slashes
    cmake_command = [cmd.replace('\\', '/') for cmd in cmake_command]

    # Generate CMake options for each compiler
    libraries = set()
    for target_name, target in targets.items():
        cmake_command += target.cmake_options()
        try:
            libraries |= unique_flags(
                Config.get("compiler", target_name, "libs"))
        except KeyError:
            pass

    # TODO: it should be possible to use the default arguments/compilers
    #       found by CMake
    cmake_command += [
        "-DDACE_LIBS=\"{}\"".format(" ".join(libraries)),
        "-DCMAKE_LINKER=\"{}\"".format(
            make_absolute(Config.get('compiler', 'linker', 'executable'))),
        "-DCMAKE_SHARED_LINKER_FLAGS=\"{}\"".format(
            Config.get('compiler', 'linker', 'args') +
            Config.get('compiler', 'linker', 'additional_args')),
    ]

    ##############################################
    # Configure
    try:
        _run_liveoutput(" ".join(cmake_command), shell=True, cwd=build_folder)
    except subprocess.CalledProcessError as ex:
        # Clean CMake directory and try once more
        if Config.get_bool('debugprint'):
            print('Cleaning CMake build folder and retrying...')
        shutil.rmtree(build_folder)
        os.makedirs(build_folder)
        try:
            _run_liveoutput(
                " ".join(cmake_command), shell=True, cwd=build_folder)
        except subprocess.CalledProcessError as ex:
            # If still unsuccessful, print results
            if Config.get_bool('debugprint'):
                raise CompilerConfigurationError('Configuration failure')
            else:
                raise CompilerConfigurationError('Configuration failure:\n' +
                                                 ex.output)

    # Compile and link
    try:
        _run_liveoutput(
            "cmake --build . --config %s" % (Config.get(
                'compiler', 'build_type')),
            shell=True,
            cwd=build_folder)
    except subprocess.CalledProcessError as ex:
        # If unsuccessful, print results
        if Config.get_bool('debugprint'):
            raise CompilationError('Compiler failure')
        else:
            raise CompilationError('Compiler failure:\n' + ex.output)

    shared_library_path = os.path.join(
        build_folder, "lib{}.{}".format(
            program_name, Config.get('compiler', 'library_extension')))

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


def get_binary_name(object_name,
                    object_hash=None,
                    lib_extension=Config.get('compiler', 'library_extension')):
    name = None
    if object_hash is None:
        name = os.path.join('.dacecache', object_name, "build",
                            'lib%s.%s' % (object_name, lib_extension))
    else:
        name = os.path.join(
            '.dacecache', object_name, "build",
            'lib%s_%s.%s' % (object_name, object_hash, lib_extension))
    return name


def _run_liveoutput(command, **kwargs):
    process = subprocess.Popen(
        command, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, **kwargs)
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
    output.write(stdout.decode('utf-8'))
    if stderr is not None:
        output.write(stderr.decode('utf-8'))

    # An error occurred, raise exception
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command,
                                            output.getvalue())


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
