# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains functionality to load, use, and invoke compiled SDFG libraries. """
import ctypes
import os
import re
import shutil
import subprocess
from typing import Any, Dict, List, Tuple
import warnings

import numpy as np
import sympy as sp

from dace import data as dt, dtypes, symbolic
from dace.codegen import exceptions as cgx
from dace.config import Config
from dace.frontend import operations


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
                    raise cgx.DuplicateDLLError(
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
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
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
    if array_type.storage == dtypes.StorageType.GPU_Global:
        return array.__cuda_array_interface__['data'][0]
    return array.__array_interface__['data'][0]


class CompiledSDFG(object):
    """ A compiled SDFG object that can be called through Python. """
    def __init__(self, sdfg, lib: ReloadableDLL):
        self._sdfg = sdfg
        self._lib = lib
        self._initialized = False
        self._libhandle = ctypes.c_void_p(0)
        self._lastargs = ()
        self._return_arrays: List[np.ndarray] = []
        self._return_kwarrays: Dict[str, np.ndarray] = {}
        self._return_syms: Dict[str, Any] = {}
        lib.load()  # Explicitly load the library
        self._init = lib.get_symbol('__dace_init_{}'.format(sdfg.name))
        self._init.restype = ctypes.c_void_p
        self._exit = lib.get_symbol('__dace_exit_{}'.format(sdfg.name))
        self._cfunc = lib.get_symbol('__program_{}'.format(sdfg.name))

    @property
    def filename(self):
        return self._lib._library_filename

    @property
    def sdfg(self):
        return self._sdfg

    def initialize(self, *argtuple):
        if self._init is not None:
            res = ctypes.c_void_p(self._init(*argtuple))
            if res == ctypes.c_void_p(0):
                raise RuntimeError('DaCe application failed to initialize')

            self._libhandle = res
            self._initialized = True

    def finalize(self):
        if self._exit is not None:
            self._exit(self._libhandle)

    def __call__(self, **kwargs):
        try:
            argtuple, initargtuple = self._construct_args(**kwargs)

            # Call initializer function if necessary, then SDFG
            if self._initialized is False:
                self._lib.load()
                self.initialize(*initargtuple)

            # PROFILING
            if Config.get_bool('profiling'):
                operations.timethis(self._sdfg, 'DaCe', 0, self._cfunc,
                                    self._libhandle, *argtuple)
            else:
                self._cfunc(self._libhandle, *argtuple)

            return self._return_arrays
        except (RuntimeError, TypeError, UnboundLocalError, KeyError,
                cgx.DuplicateDLLError, ReferenceError):
            self._lib.unload()
            raise

    def __del__(self):
        if self._initialized is True:
            self.finalize()
            self._initialized = False
            self._libhandle = ctypes.c_void_p(0)
        self._lib.unload()

    def _construct_args(self, **kwargs) -> Tuple[Tuple[Any], Tuple[Any]]:
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
                    atype.dtype, dtypes.callback) and not isinstance(
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
            if isinstance(argtype.dtype, dtypes.callback):
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

        initargs = tuple(atup for atup in callparams
                         if not dtypes.is_array(atup[0]))

        newargs = tuple(
            actype(arg) if (not isinstance(arg, ctypes._SimpleCData)) else arg
            for arg, actype, atype in newargs)

        initargs = tuple(
            actype(arg) if (not isinstance(arg, ctypes._SimpleCData)) else arg
            for arg, actype, atype in initargs)

        self._lastargs = newargs, initargs
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
                        dtypes.StorageType.GPU_Global,
                        dtypes.StorageType.FPGA_Global
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
