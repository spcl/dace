# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains functionality to load, use, and invoke compiled SDFG libraries. """
import ctypes
import os
import re
import shutil
import subprocess
from typing import Any, Callable, Dict, List, Tuple, Optional, Type
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
            'libdacestub_%s.%s' % (program_name, Config.get('compiler', 'library_extension')))
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
            raise KeyError('Function %s not found in library %s' % (name, os.path.basename(self._library_filename)))

        return ctypes.CFUNCTYPE(restype)(func)

    def is_loaded(self) -> bool:
        """ Checks if the library is already loaded. """

        # If internal library is already loaded, skip
        if self._lib is not None and self._lib.value is not None:
            return True
        if not os.path.isfile(self._stub_filename):
            return False
        self._stub = ctypes.CDLL(self._stub_filename)

        # Set return types of stub functions
        self._stub.load_library.restype = ctypes.c_void_p
        self._stub.get_symbol.restype = ctypes.c_void_p

        lib_cfilename = None
        # Convert library filename to string according to OS
        if os.name == 'nt':
            # As UTF-16
            lib_cfilename = ctypes.c_wchar_p(self._library_filename)
        else:
            # As UTF-8
            lib_cfilename = ctypes.c_char_p(self._library_filename.encode('utf-8'))

        return self._stub.is_library_loaded(lib_cfilename) == 1

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
                lib_cfilename = ctypes.c_char_p(self._library_filename.encode('utf-8'))

            is_loaded = self._stub.is_library_loaded(lib_cfilename)
            if is_loaded == 1:
                warnings.warn('Library %s already loaded, renaming file' % self._library_filename)
                try:
                    shutil.copyfile(self._library_filename, self._library_filename + '_')
                    self._library_filename += '_'
                except shutil.Error:
                    raise cgx.DuplicateDLLError('Library %s is already loaded somewhere else ' %
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
                result = subprocess.run(['ld', self._library_filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stderr = result.stderr.decode('utf-8')
                reason = 'Reason:\n' + '\n'.join([l for l in stderr.split('\n') if '_start' not in l])
            raise RuntimeError('Could not load library %s. %s' % (os.path.basename(self._library_filename), reason))

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
    def __init__(self, sdfg, lib: ReloadableDLL, argnames: List[str] = None):
        self._sdfg = sdfg
        self._lib = lib
        self._initialized = False
        self._libhandle = ctypes.c_void_p(0)
        self._lastargs = ()

        lib.load()  # Explicitly load the library
        self._init = lib.get_symbol('__dace_init_{}'.format(sdfg.name))
        self._init.restype = ctypes.c_void_p
        self._exit = lib.get_symbol('__dace_exit_{}'.format(sdfg.name))
        self._cfunc = lib.get_symbol('__program_{}'.format(sdfg.name))

        # Cache SDFG return values
        self._create_new_arrays: bool = True
        self._return_syms: Dict[str, Any] = None
        self._retarray_shapes: List[Tuple[str, np.dtype, dtypes.StorageType, Tuple[int], Tuple[int], int]] = []
        self._return_arrays: List[np.ndarray] = []
        self._callback_retval_references: List[Any] = []  # Avoids garbage-collecting callback return values

        # Cache SDFG argument properties
        self._typedict = self._sdfg.arglist()
        self._sig = self._sdfg.signature_arglist(with_types=False, arglist=self._typedict)
        self._free_symbols = self._sdfg.free_symbols
        self.argnames = argnames

    def get_exported_function(self, name: str, restype=None) -> Optional[Callable[..., Any]]:
        """
        Tries to find a symbol by name in the compiled SDFG, and convert it to a callable function
        with the (optionally) given return type (void by default). If no such function exists, returns None.
        :param name: Name of the function to query.
        :return: Callable to the function, or None if doesn't exist.
        """
        try:
            return self._lib.get_symbol(name, restype=restype)
        except KeyError:  # Function not found
            return None

    def get_state_struct(self) -> ctypes.Structure:
        """ Attempt to parse the SDFG source code and extract the state struct. This method will parse the first
            consecutive entries in the struct that are pointers. As soon as a non-pointer or other unparseable field is
            encountered, the method exits early. All fields defined until then will nevertheless be available in the
            structure.
            :returns: the ctypes.Structure representation of the state struct.
        """

        return ctypes.cast(self._libhandle, ctypes.POINTER(self._try_parse_state_struct())).contents

    def _try_parse_state_struct(self) -> Optional[Type[ctypes.Structure]]:
        # the path of the main sdfg file containing the state struct
        main_src_path = os.path.join(os.path.dirname(os.path.dirname(self._lib._library_filename)), "src", "cpu",
                                     self._sdfg.name + ".cpp")
        code = open(main_src_path, 'r').read()

        code_flat = code.replace("\n", " ")

        # try to find the first struct definition that matches the name we are looking for in the sdfg file
        match = re.search(f"struct {self._sdfg.name}_t {{(.*?)}};", code_flat)
        if match is None or len(match.groups()) != 1:
            return None

        # get the definitions from the struct
        struct_defn = match[1]

        fields = []
        for field_str in struct_defn.split(";"):
            field_str = field_str.strip()

            match_name = re.match(r'(?:const)?\s*(.*)(?:\s+\*\s*|\s*\*\s+\_\_restrict\_\_\s+)([a-zA-Z_][a-zA-Z_0-9]*)$',
                                  field_str)
            if match_name is None:
                # reached a non-ptr field or something unparsable, we have to abort here
                break

            # we have a ptr field
            name = match_name[2]
            fields.append((name, ctypes.c_void_p))

        class State(ctypes.Structure):
            _fields_ = fields

        return State

    @property
    def filename(self):
        return self._lib._library_filename

    @property
    def sdfg(self):
        return self._sdfg

    def _initialize(self, argtuple):
        if self._init is not None:
            res = ctypes.c_void_p(self._init(*argtuple))
            if res == ctypes.c_void_p(0):
                raise RuntimeError('DaCe application failed to initialize')

            self._libhandle = res
            self._initialized = True

    def initialize(self, *args, **kwargs):
        """
        Initializes the compiled SDFG without invoking it. 
        :param args: Arguments to call SDFG with.
        :param kwargs: Keyword arguments to call SDFG with.
        :return: If successful, returns the library handle (as a ctypes pointer).
        :note: This call requires the same arguments as it would when normally calling the program.
        """
        if self._initialized:
            return

        if len(args) > 0 and self.argnames is not None:
            kwargs.update({aname: arg for aname, arg in zip(self.argnames, args)})

        # Construct arguments in the exported C function order
        _, initargtuple = self._construct_args(kwargs)
        self._initialize(initargtuple)
        return self._libhandle

    def finalize(self):
        if self._exit is not None:
            self._exit(self._libhandle)
            self._initialized = False

    def __call__(self, *args, **kwargs):
        # Update arguments from ordered list
        if len(args) > 0 and self.argnames is not None:
            kwargs.update({aname: arg for aname, arg in zip(self.argnames, args)})

        try:
            argtuple, initargtuple = self._construct_args(kwargs)

            # Call initializer function if necessary, then SDFG
            if self._initialized is False:
                self._lib.load()
                self._initialize(initargtuple)
            # PROFILING
            if Config.get_bool('profiling'):
                operations.timethis(self._sdfg, 'DaCe', 0, self._cfunc, self._libhandle, *argtuple)
            else:
                self._cfunc(self._libhandle, *argtuple)

            return self._return_arrays
        except (RuntimeError, TypeError, UnboundLocalError, KeyError, cgx.DuplicateDLLError, ReferenceError):
            self._lib.unload()
            raise

    def __del__(self):
        if self._initialized is True:
            self.finalize()
            self._initialized = False
            self._libhandle = ctypes.c_void_p(0)
        self._lib.unload()

    def _construct_args(self, kwargs) -> Tuple[Tuple[Any], Tuple[Any]]:
        """ Main function that controls argument construction for calling
            the C prototype of the SDFG.

            Organizes arguments first by `sdfg.arglist`, then data descriptors
            by alphabetical order, then symbols by alphabetical order.
        """
        # Return value initialization (for values that have not been given)
        self._initialize_return_values(kwargs)
        if self._return_arrays is not None:
            if len(self._retarray_shapes) == 1:
                kwargs[self._retarray_shapes[0][0]] = self._return_arrays
            else:
                for desc, arr in zip(self._retarray_shapes, self._return_arrays):
                    kwargs[desc[0]] = arr

        # Argument construction
        sig = self._sig
        typedict = self._typedict
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
        for i, (a, arg, atype) in enumerate(zip(argnames, arglist, argtypes)):
            if not dtypes.is_array(arg) and isinstance(atype, dt.Array):
                if isinstance(arg, list):
                    print('WARNING: Casting list argument "%s" to ndarray' % a)
                elif arg is None:
                    if atype.optional is False:  # If array cannot be None
                        raise TypeError(f'Passing a None value to a non-optional array in argument "{a}"')
                    # Otherwise, None values are passed as null pointers below
                else:
                    raise TypeError('Passing an object (type %s) to an array in argument "%s"' %
                                    (type(arg).__name__, a))
            elif dtypes.is_array(arg) and not isinstance(atype, dt.Array):
                # GPU scalars are pointers, so this is fine
                if atype.storage != dtypes.StorageType.GPU_Global:
                    raise TypeError('Passing an array to a scalar (type %s) in argument "%s"' % (atype.dtype.ctype, a))
            elif not isinstance(atype, dt.Array) and not isinstance(atype.dtype, dtypes.callback) and not isinstance(
                    arg,
                (atype.dtype.type, sp.Basic)) and not (isinstance(arg, symbolic.symbol) and arg.dtype == atype.dtype):
                if isinstance(arg, int) and atype.dtype.type == np.int64:
                    pass
                elif isinstance(arg, float) and atype.dtype.type == np.float64:
                    pass
                elif (isinstance(arg, int) and atype.dtype.type == np.int32 and abs(arg) <= (1 << 31) - 1):
                    pass
                elif (isinstance(arg, int) and atype.dtype.type == np.uint32 and arg >= 0 and arg <= (1 << 32) - 1):
                    pass
                elif (isinstance(arg, str) or arg is None) and atype.dtype == dtypes.string:
                    if arg is None:
                        arglist[i] = ctypes.c_char_p(None)
                    else:
                        # Cast to bytes
                        arglist[i] = ctypes.c_char_p(arg.encode('utf-8'))
                else:
                    warnings.warn(f'Casting scalar argument "{a}" from {type(arg).__name__} to {atype.dtype.type}')
                    arglist[i] = atype.dtype.type(arg)
            elif (isinstance(atype, dt.Array) and isinstance(arg, np.ndarray)
                  and atype.dtype.as_numpy_dtype() != arg.dtype):
                # Make exception for vector types
                if (isinstance(atype.dtype, dtypes.vector) and atype.dtype.vtype.as_numpy_dtype() == arg.dtype):
                    pass
                else:
                    print('WARNING: Passing %s array argument "%s" to a %s array' %
                          (arg.dtype, a, atype.dtype.type.__name__))
            elif (isinstance(atype, dt.Array) and isinstance(arg, np.ndarray) and arg.base is not None
                  and not '__return' in a and not Config.get_bool('compiler', 'allow_view_arguments')):
                raise TypeError(f'Passing a numpy view (e.g., sub-array or "A.T") "{a}" to DaCe '
                                'programs is not allowed in order to retain analyzability. '
                                'Please make a copy with "numpy.copy(...)". If you know what '
                                'you are doing, you can override this error in the '
                                'configuration by setting compiler.allow_view_arguments '
                                'to True.')

        # Explicit casting
        for index, (arg, argtype) in enumerate(zip(arglist, argtypes)):
            # Call a wrapper function to make NumPy arrays from pointers.
            if isinstance(argtype.dtype, dtypes.callback):
                arglist[index] = argtype.dtype.get_trampoline(arg, kwargs, self._callback_retval_references)
            # List to array
            elif isinstance(arg, list) and isinstance(argtype, dt.Array):
                arglist[index] = np.array(arg, dtype=argtype.dtype.type)
            # Null pointer
            elif arg is None and isinstance(argtype, dt.Array):
                arglist[index] = ctypes.c_void_p(0)

        # Retain only the element datatype for upcoming checks and casts
        arg_ctypes = [t.dtype.as_ctypes() for t in argtypes]

        sdfg = self._sdfg

        # Obtain SDFG constants
        constants = sdfg.constants

        # Remove symbolic constants from arguments
        callparams = tuple((arg, actype, atype, aname)
                           for arg, actype, atype, aname in zip(arglist, arg_ctypes, argtypes, argnames)
                           if not symbolic.issymbolic(arg) or (hasattr(arg, 'name') and arg.name not in constants))

        # Replace symbols with their values
        callparams = tuple((actype(arg.get()) if isinstance(arg, symbolic.symbol) else arg, actype, atype, aname)
                           for arg, actype, atype, aname in callparams)

        # Construct init args, which only consist of the symbols
        symbols = self._free_symbols
        initargs = tuple(
            actype(arg) if (not isinstance(arg, ctypes._SimpleCData)) else arg
            for arg, actype, atype, aname in callparams if aname in symbols)

        # Replace arrays with their base host/device pointers
        newargs = tuple((ctypes.c_void_p(_array_interface_ptr(arg, atype)), actype,
                         atype) if dtypes.is_array(arg) else (arg, actype, atype)
                        for arg, actype, atype, _ in callparams)

        try:
            newargs = tuple(
                actype(arg) if (not isinstance(arg, ctypes._SimpleCData)) else arg for arg, actype, atype in newargs)
        except TypeError:
            # Pinpoint bad argument
            for i, (arg, actype, _) in enumerate(newargs):
                try:
                    if not isinstance(arg, ctypes._SimpleCData):
                        actype(arg)
                except TypeError as ex:
                    raise TypeError(f'Invalid type for scalar argument "{callparams[i][3]}": {ex}')

        self._lastargs = newargs, initargs
        return self._lastargs

    def clear_return_values(self):
        self._create_new_arrays = True

    def _create_array(self, _: str, dtype: np.dtype, storage: dtypes.StorageType, shape: Tuple[int],
                      strides: Tuple[int], total_size: int):
        ndarray = np.ndarray
        zeros = np.empty

        if storage is dtypes.StorageType.GPU_Global:
            try:
                import cupy

                # Set allocator to GPU
                def ndarray(*args, buffer=None, **kwargs):
                    if buffer is not None:
                        buffer = buffer.data
                    return cupy.ndarray(*args, memptr=buffer, **kwargs)

                zeros = cupy.empty
            except (ImportError, ModuleNotFoundError):
                raise NotImplementedError('GPU return values are unsupported if cupy is not installed')
        if storage is dtypes.StorageType.FPGA_Global:
            raise NotImplementedError('FPGA return values are unsupported')

        # Create an array with the properties of the SDFG array
        return ndarray(shape, dtype, buffer=zeros(total_size, dtype), strides=strides)

    def _initialize_return_values(self, kwargs):
        # Obtain symbol values from arguments and constants
        syms = dict()
        syms.update({k: v for k, v in kwargs.items() if k not in self.sdfg.arrays})
        syms.update(self.sdfg.constants)

        # Clear references from last call (allow garbage collection)
        self._callback_retval_references.clear()

        if self._initialized:
            if self._return_syms == syms:
                if not self._create_new_arrays:
                    return
                else:
                    self._create_new_arrays = False
                    # Use stored sizes to recreate arrays (fast path)
                    if self._return_arrays is None:
                        return
                    elif isinstance(self._return_arrays, tuple):
                        self._return_arrays = tuple(kwargs[desc[0]] if desc[0] in kwargs else self._create_array(*desc)
                                                    for desc in self._retarray_shapes)
                        return
                    else:  # Single array return value
                        desc = self._retarray_shapes[0]
                        arr = (kwargs[desc[0]] if desc[0] in kwargs else self._create_array(*desc))
                        self._return_arrays = arr
                        return

        self._return_syms = syms
        self._create_new_arrays = False

        # Initialize return values with numpy arrays
        self._retarray_shapes = []
        self._return_arrays = []
        for arrname, arr in sorted(self.sdfg.arrays.items()):
            if arrname.startswith('__return') and not arr.transient:
                if arrname in kwargs:
                    self._return_arrays.append(kwargs[arrname])
                    self._retarray_shapes.append((arrname, ))
                    continue

                if isinstance(arr, dt.Stream):
                    raise NotImplementedError('Return streams are unsupported')

                shape = tuple(symbolic.evaluate(s, syms) for s in arr.shape)
                dtype = arr.dtype.as_numpy_dtype()
                total_size = int(symbolic.evaluate(arr.total_size, syms))
                strides = tuple(symbolic.evaluate(s, syms) * arr.dtype.bytes for s in arr.strides)
                shape_desc = (arrname, dtype, arr.storage, shape, strides, total_size)
                self._retarray_shapes.append(shape_desc)

                # Create an array with the properties of the SDFG array
                arr = self._create_array(*shape_desc)
                self._return_arrays.append(arr)

        # Set up return_arrays field
        if len(self._return_arrays) == 0:
            self._return_arrays = None
        elif len(self._return_arrays) == 1:
            self._return_arrays = self._return_arrays[0]
        else:
            self._return_arrays = tuple(self._return_arrays)
