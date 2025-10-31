# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains functionality to load, use, and invoke compiled SDFG libraries. """
import ctypes
import os
import re
import shutil
import subprocess
from typing import Any, Callable, Dict, List, Tuple, Optional, Type, Union, Sequence
import warnings
import tempfile
import pickle
import sys

import numpy as np
import sympy as sp

from dace import data as dt, dtypes, hooks, symbolic
from dace.codegen import exceptions as cgx
from dace.config import Config
from dace.frontend import operations


class ReloadableDLL(object):
    """
    A reloadable shared object (or dynamically linked library), which
    bypasses Python's dynamic library reloading issues.
    """

    def __init__(self, library_filename, program_name):
        """
        Creates a new reloadable shared object.

        :param library_filename: Path to library file.
        :param program_name: Name of the DaCe program (for use in finding
                             the stub library loader).
        """
        self._stub_filename = os.path.join(os.path.dirname(os.path.realpath(library_filename)),
                                           f'libdacestub_{program_name}.{Config.get("compiler", "library_extension")}')
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
            raise KeyError(f'Function {name} not found in library {os.path.basename(self._library_filename)}')

        return ctypes.CFUNCTYPE(restype)(func)

    def is_loaded(self) -> bool:
        """ Checks if the library is already loaded. """

        # If internal library is already loaded, skip
        if self._lib is not None and self._lib.value is not None:
            return True
        if not os.path.isfile(self._stub_filename):
            return False
        try:
            self._stub = ctypes.CDLL(self._stub_filename)
        except OSError:
            return False

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
                warnings.warn(f'Library {self._library_filename} already loaded, renaming file')
                try:
                    shutil.copyfile(self._library_filename, self._library_filename + '_')
                    self._library_filename += '_'
                except shutil.Error:
                    raise cgx.DuplicateDLLError(f'Library {os.path.basename(self._library_filename)}'
                                                'is already loaded somewhere else and cannot be unloaded. '
                                                'Please use a different name for the SDFG/program.')

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
            raise RuntimeError(f'Could not load library {os.path.basename(self._library_filename)}. {reason}')

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
    """ A compiled SDFG object that can be called through Python.

    Essentially this class makes an SDFG callable. Normally a user will not create it
    directly but instead it is generated by some utilities such as `SDFG.compile()`.

    The class performs the following tasks:
    - It ensures that the SDFG object is properly initialized, either by a direct
        call to `initialize()` or the first time it is called. Furthermore, it will
        also take care of the finalization if it does out of scope.
    - It transforms Python arguments into C arguments.

    Technically there are two ways how the SDFG can be called, the first is using
    `__call__()`, i.e. as a normal function. However, this will always processes
    the arguments and does some error checking and is thus slow. The second way
    is the advanced interface, which allows to decompose the calling into different
    subset. For more information see `construct_arguments()`, `fast_call()` and
    `convert_return_values()`.

    :note: In previous version the arrays used as return values were sometimes reused.
        However, this was changed and every time `construct_arguments()` is called
        new arrays are allocated.
    :note: It is not possible to return scalars. Note that currently using scalars
        as return values is a validation error. The only exception are (probably)
        Python objects.
    """

    def __init__(self, sdfg, lib: ReloadableDLL, argnames: List[str] = None):
        from dace.sdfg import SDFG
        self._sdfg: SDFG = sdfg
        self._lib = lib
        self._initialized = False
        self._libhandle = ctypes.c_void_p(0)
        self.do_not_execute = False

        # Contains the pointer arguments that where used to call the SDFG, `__call__()`
        #  was used. It is also used by `get_workspace_size()`.
        # NOTE: Using its content might be dangerous as only the pointers to arrays are
        #   stored. It is the users responsibility to ensure that they are valid.
        self._lastargs = None

        lib.load()  # Explicitly load the library
        self._init = lib.get_symbol('__dace_init_{}'.format(sdfg.name))
        self._init.restype = ctypes.c_void_p
        self._exit = lib.get_symbol('__dace_exit_{}'.format(sdfg.name))
        self._exit.restype = ctypes.c_int
        self._cfunc = lib.get_symbol('__program_{}'.format(sdfg.name))

        # Cache SDFG return values
        self._return_syms: Dict[str, Any] = None
        # It will contain the shape of the array or the name if the return array is passed as argument.
        self._retarray_shapes: List[Tuple[str, np.dtype, dtypes.StorageType, Tuple[int], Tuple[int], int]] = []
        # Is only `True` if teh return value is a scalar _and_ a `pyobject`.
        self._retarray_is_pyobject: List[bool] = []
        self._return_arrays: List[np.ndarray] = []
        self._callback_retval_references: List[Any] = []  # Avoids garbage-collecting callback return values

        # If there are return values then this is `True` it is is a single value. Note that
        #  `False` either means that a tuple is returned or there are no return values.
        # NOTE: Needed to handle the case of a tuple with one element.
        self._is_single_value_ret: bool = False
        if '__return' in self._sdfg.arrays:
            assert not any(aname.startswith('__return_') for aname in self._sdfg.arrays.keys())
            self._is_single_value_ret = True

        # Cache SDFG argument properties
        self._typedict = self._sdfg.arglist()
        self._sig = self._sdfg.signature_arglist(with_types=False, arglist=self._typedict)
        self._free_symbols = self._sdfg.free_symbols
        self._constants = self._sdfg.constants
        self.argnames = argnames

        if self.argnames is None and len(sdfg.arg_names) != 0:
            warnings.warn('You passed `None` as `argnames` to `CompiledSDFG`, but the SDFG you passed has positional'
                          ' arguments. This is allowed but deprecated.')

        self.has_gpu_code = False
        self.external_memory_types = set()
        for _, _, aval in self._sdfg.arrays_recursive():
            if aval.storage in dtypes.GPU_STORAGES:
                self.has_gpu_code = True
                break
            if aval.lifetime == dtypes.AllocationLifetime.External:
                self.external_memory_types.add(aval.storage)
        if not self.has_gpu_code:
            for node, _ in self._sdfg.all_nodes_recursive():
                if getattr(node, 'schedule', False) in dtypes.GPU_SCHEDULES:
                    self.has_gpu_code = True
                    break

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

            :return: the ctypes.Structure representation of the state struct.
        """
        if not self._libhandle:
            raise ValueError('Library was not initialized')

        return ctypes.cast(self._libhandle, ctypes.POINTER(self._try_parse_state_struct())).contents

    def _try_parse_state_struct(self) -> Optional[Type[ctypes.Structure]]:
        from dace.codegen.targets.cpp import mangle_dace_state_struct_name  # Avoid import cycle
        # the path of the main sdfg file containing the state struct
        main_src_path = os.path.join(os.path.dirname(os.path.dirname(self._lib._library_filename)), "src", "cpu",
                                     self._sdfg.name + ".cpp")
        code = open(main_src_path, 'r').read()

        code_flat = code.replace("\n", " ")

        # try to find the first struct definition that matches the name we are looking for in the sdfg file
        match = re.search(f"struct {mangle_dace_state_struct_name(self._sdfg)} {{(.*?)}};", code_flat)
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

    def get_workspace_sizes(self) -> Dict[dtypes.StorageType, int]:
        """
        Returns the total external memory size to be allocated for this SDFG.

        Note that the function queries the sizes of the last call that was made by
        `__call__()` or `initialize()`. Calls made by `fast_call()` or `safe_call()`
        will not be considered.

        :return: A dictionary mapping storage types to the number of bytes necessary
                 to allocate for the SDFG to work properly.
        :note: It is the users responsibility that all arguments, especially the array
            arguments, remain valid between the call to `__call__()` or `initialize()`
            and the call to this function.
        """
        if not self._initialized:
            raise ValueError('Compiled SDFG is uninitialized, please call ``initialize`` prior to '
                             'querying external memory size.')
        if self._lastargs is None:
            raise ValueError('To use `get_workspace_sizes()` `__call__()` or `initialize()` must be called before.')

        result: Dict[dtypes.StorageType, int] = {}
        for storage in self.external_memory_types:
            func = self._lib.get_symbol(f'__dace_get_external_memory_size_{storage.name}')
            func.restype = ctypes.c_size_t
            result[storage] = func(self._libhandle, *self._lastargs[1])

        return result

    def set_workspace(self, storage: dtypes.StorageType, workspace: Any):
        """
        Sets the workspace for the given storage type to the given buffer.

        Note that the function queries the sizes of the last call that was made by
        `__call__()` or `initialize()`. Calls made by `fast_call()` or `safe_call()`
        will not be considered.

        :param storage: The storage type to fill.
        :param workspace: An array-convertible object (through ``__[cuda_]array_interface__``,
                          see ``array_interface_ptr``) to use for the workspace.
        :note: It is the users responsibility that all arguments, especially the array
            arguments, remain valid between the call to `__call__()` or `initialize()`
            and the call to this function.
        """
        if not self._initialized:
            raise ValueError('Compiled SDFG is uninitialized, please call ``initialize`` prior to '
                             'setting external memory.')
        if storage not in self.external_memory_types:
            raise ValueError(f'Compiled SDFG does not specify external memory of {storage}')
        if self._lastargs is None:
            raise ValueError('To use `get_workspace_sizes()` `__call__()` or `initialize()` must be called before.')

        func = self._lib.get_symbol(f'__dace_set_external_memory_{storage.name}', None)
        ptr = dtypes.array_interface_ptr(workspace, storage)
        func(self._libhandle, ctypes.c_void_p(ptr), *self._lastargs[1])

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

        # Construct arguments in the exported C function order
        callargtuple, initargtuple = self.construct_arguments(*args, **kwargs)
        self._initialize(initargtuple)

        # The main reason for setting `_lastargs` here is, to allow calls to `get_workspace_size()`.
        self._lastargs = (callargtuple, initargtuple)

        return self._libhandle

    def finalize(self):
        if self._exit is not None:
            res: int = self._exit(self._libhandle)
            self._initialized = False
            if res != 0:
                raise RuntimeError(
                    f'An error was detected after running "{self._sdfg.name}": {self._get_error_text(res)}')

    def _get_error_text(self, result: Union[str, int]) -> str:
        from dace.codegen import common  # Circular import
        if self.has_gpu_code:
            if isinstance(result, int):
                result = common.get_gpu_runtime().get_error_string(result)
            return (f'{result}. Consider enabling synchronous debugging mode (environment variable: '
                    'DACE_compiler_cuda_syncdebug=1) to see where the issue originates from.')
        else:
            return result

    def __call__(self, *args, **kwargs):
        """
        Forwards the Python call to the compiled ``SDFG``.

        The order of the positional arguments is expected to be the same as in the
        ``argnames`` member. The function will perform the following tasks:
        - Calling ``construct_arguments()`` and creating the argument vector and
            allocating the memory for the return values.
        - Performing the actual call by means of ``fast_call()``, with enabled error
            checks.
        - Then it will convert the return value into the expected format by means of
            ``convert_return_values()`` and return that value.

        :note: The memory for the return values is only allocated the first
               time this function is called. Thus, this function will always
               return the same objects. To force the allocation of new memory
               you can call ``clear_return_values()`` in advance.
        """
        argtuple, initargtuple = self.construct_arguments(*args, **kwargs)  # Missing arguments will be detected here.
        self._lastargs = (argtuple, initargtuple)
        self.fast_call(argtuple, initargtuple, do_gpu_check=True)
        return self.convert_return_values()

    def safe_call(self, *args, **kwargs):
        """
        Forwards the Python call to the compiled ``SDFG`` in a separate process to avoid crashes in the main process. Raises an exception if the SDFG execution fails.

        Note the current implementation lacks the proper handling of return values.
        Thus output can only be transmitted through inout arguments.
        """
        if any(aname == '__return' or aname.startswith('__return_') for aname in self.sdfg.arrays.keys()):
            raise NotImplementedError('`CompiledSDFG.safe_call()` does not support return values.')

        # Pickle the SDFG and arguments
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            pickle.dump({
                'library_path': self._lib._library_filename,
                "sdfg": self.sdfg,
                'args': args,
                'kwargs': kwargs
            }, f)
            temp_path = f.name

        # Call the SDFG in a separate process
        result = subprocess.run([
            sys.executable, '-c', f'''
import pickle
from dace.codegen import compiled_sdfg as csd

with open(r"{temp_path}", "rb") as f:
    data = pickle.load(f)
library_path = data['library_path']
sdfg = data['sdfg']

lib = csd.ReloadableDLL(library_path, sdfg.name)
obj = csd.CompiledSDFG(sdfg, lib, sdfg.arg_names)
obj(*data['args'], **data['kwargs'])

with open(r"{temp_path}", "wb") as f:
    pickle.dump({{
        'args': data['args'],
        'kwargs': data['kwargs']
    }}, f)
             '''
        ])

        # Receive the result
        with open(temp_path, 'rb') as f:
            data = pickle.load(f)
            for i in range(len(args)):
                if hasattr(args[i], '__setitem__'):
                    args[i].__setitem__(slice(None), data['args'][i])
            for k in kwargs:
                if hasattr(kwargs[k], '__setitem__'):
                    kwargs[k].__setitem__(slice(None), data['kwargs'][k])

        # Clean up
        os.remove(temp_path)
        if result.returncode != 0:
            raise RuntimeError(f'SDFG execution failed with return code {result.returncode}.')

    def fast_call(
        self,
        callargs: Sequence[Any],
        initargs: Sequence[Any],
        do_gpu_check: bool = False,
    ) -> None:
        """
        Calls the underlying binary functions directly and bypassing argument sanitation.

        This is a faster, but less user friendly version of ``__call__()``. While
        ``__call__()`` will transforms its Python arguments such that they can be
        forwarded and allocate memory for the return values, this function assumes
        that this processing was already done by the user.
        To build the argument vectors you should use `self.construct_arguments()`.

        :param callargs:        Arguments passed to the actual computation.
        :param initargs:        Arguments passed to the initialization function.
        :param do_gpu_check:    Check if errors happened on the GPU.

        :note: This is an advanced interface.
        :note: In previous versions this function also called `convert_return_values()`.
        """
        try:
            # Call initializer function if necessary, then SDFG
            if self._initialized is False:
                self._lib.load()
                self._initialize(initargs)

            with hooks.invoke_compiled_sdfg_call_hooks(self, callargs):
                if self.do_not_execute is False:
                    self._cfunc(self._libhandle, *callargs)

            # Optionally get errors from call
            if do_gpu_check and self.has_gpu_code:
                from dace.codegen import common  # Circular import and avoid import in the hot path.
                try:
                    lasterror = common.get_gpu_runtime().get_last_error_string()
                except RuntimeError as ex:
                    warnings.warn(f'Could not get last error from GPU runtime: {ex}')
                    lasterror = None

                if lasterror is not None:
                    raise RuntimeError(
                        f'An error was detected when calling "{self._sdfg.name}": {self._get_error_text(lasterror)}')
            return
        except (RuntimeError, TypeError, UnboundLocalError, KeyError, cgx.DuplicateDLLError, ReferenceError):
            self._lib.unload()
            raise

    def __del__(self):
        if self._initialized is True:
            self.finalize()
            self._initialized = False
            self._libhandle = ctypes.c_void_p(0)
        self._lib.unload()

    def construct_arguments(self, *args: Any, **kwargs: Any) -> Tuple[Tuple[Any], Tuple[Any]]:
        """Construct the argument vectors suitable for from its argument.

        The function returns a pair of tuple, that are suitable for `fast_call()`.
        The first element of is `callargs`, i.e. the full arguments, while the
        second element is `initargs`, which is only used/needed the first time
        an SDFG is called.

        It is important that this function will also allocate new return values.
        The array objects are managed by `self` and remain valid until this
        function is called again. However, they are also returned by `self.__call__()`.

        It is also possible to pass the array, that should be used to return a value,
        directly as argument. In that case the allocation for that return value will
        be skipped.

        :note: In case of arrays, the returned argument vectors only contains the
            pointers to the underlying memory. Thus it is the user's responsibility
            to ensure that the memory remains allocated until the argument vector
            is used.
        :note: This is an advanced interface.
        """
        if self.argnames is None and len(args) != 0:
            raise KeyError(f"Passed positional arguments to an SDFG that does not accept them.")
        elif len(args) > 0 and self.argnames is not None:
            positional_arguments = {aname: avalue for aname, avalue in zip(self.argnames, args)}
            if not positional_arguments.keys().isdisjoint(kwargs.keys()):
                raise ValueError(
                    f'The arguments where passed once as positional and named arguments: {set(positional_arguments.keys()).intersection(kwargs.keys())}'
                )
            kwargs.update(positional_arguments)

        # NOTE: This might invalidate the elements associated to the return values of
        #   all argument vectors that were created before.
        self._initialize_return_values(kwargs)

        # Add the return values to the arguments, since they are part of the C signature.
        for desc, arr in zip(self._retarray_shapes, self._return_arrays):
            kwargs[desc[0]] = arr

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
            if len(sig) > 0:
                raise KeyError(f"Missing program arguments: {', '.join(sig)}")
            arglist = []
            argtypes = []
            argnames = []
            sig = []

        no_view_arguments = not Config.get_bool('compiler', 'allow_view_arguments')
        cargs = tuple(
            dt.make_ctypes_argument(aval,
                                    atype,
                                    aname,
                                    allow_views=not no_view_arguments,
                                    symbols=kwargs,
                                    callback_retval_references=self._callback_retval_references)
            for aval, atype, aname in zip(arglist, argtypes, argnames))

        symbols = self._free_symbols
        callparams = tuple((carg, aname) for arg, carg, aname in zip(arglist, cargs, argnames)
                           if not ((hasattr(arg, 'name') and arg.name in self._constants) and symbolic.issymbolic(arg)))
        newargs = tuple(carg for carg, _aname in callparams)
        initargs = tuple(carg for carg, aname in callparams if aname in symbols)

        return (newargs, initargs)

    def convert_return_values(self) -> Union[Any, Tuple[Any, ...]]:
        """Convert the return arguments.

        Execute the `return` statement and return. This function should only be called
        after `fast_call()` has been run.
        Keep in mid that it is not possible to return scalars (with the exception of
        `pyobject`s), they will be always returned as an array with shape `(1,)`.

        :note: This is an advanced interface.
        :note: After `fast_call()` returns it is only allowed to call this function once.
        """
        # TODO: Make sure that the function is called only once by checking it.
        # NOTE: Currently it is not possible to return a scalar value, see `tests/sdfg/scalar_return.py`
        if not self._return_arrays:
            return None
        elif self._is_single_value_ret:
            assert len(self._return_arrays) == 1
            return self._return_arrays[0].item() if self._retarray_is_pyobject[0] else self._return_arrays[0]
        else:
            return tuple(r.item() if is_pyobj else r
                         for r, is_pyobj in zip(self._return_arrays, self._retarray_is_pyobject))

    def clear_return_values(self):
        warnings.warn(
            'The "CompiledSDFG.clear_return_values" API is deprecated, as this behaviour has'
            ' become the new default, and is a noops.', DeprecationWarning)
        pass

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
        sdfg_arrays = self.sdfg.arrays
        syms.update({k: v for k, v in kwargs.items() if k not in sdfg_arrays})
        syms.update(self.sdfg.constants)

        # Clear references from last call (allow garbage collection)
        self._callback_retval_references.clear()

        if self._initialized and self._return_syms == syms:
            # Use stored sizes to recreate arrays (fast path)
            self._return_arrays = tuple(kwargs[desc[0]] if desc[0] in kwargs else self._create_array(*desc)
                                        for desc in self._retarray_shapes)
            return

        self._return_syms = syms
        self._return_arrays = []
        self._retarray_shapes = []
        for arrname, arr in sorted(self.sdfg.arrays.items()):
            if arrname.startswith('__return'):
                if arr.transient:
                    raise ValueError(f'Used the special array name "{arrname}" as transient.')

                elif arrname in kwargs:
                    # The return value is passed as an argument, in that case store the name in `self._retarray_shapes`.
                    warnings.warn(f'Return value "{arrname}" is passed as a regular argument.', stacklevel=2)
                    self._return_arrays.append(kwargs[arrname])
                    self._retarray_shapes.append((arrname, ))

                elif isinstance(arr, dt.Stream):
                    raise NotImplementedError('Return streams are unsupported')

                else:
                    shape = tuple(symbolic.evaluate(s, syms) for s in arr.shape)
                    dtype = arr.dtype.as_numpy_dtype()
                    total_size = int(symbolic.evaluate(arr.total_size, syms))
                    strides = tuple(symbolic.evaluate(s, syms) * arr.dtype.bytes for s in arr.strides)
                    shape_desc = (arrname, dtype, arr.storage, shape, strides, total_size)
                    self._retarray_shapes.append(shape_desc)

                    # Create an array with the properties of the SDFG array
                    return_array = self._create_array(*shape_desc)
                    self._return_arrays.append(return_array)

                # BUG COMPATIBILITY(PR#2206):
                #   In the original version `_retarray_is_pyobject` was named `_retarray_is_scalar`, however
                #   since scalars could not be returned on an [implementation level](https://github.com/spcl/dace/pull/1609)
                #   it was essentially useless. But was used for `pyobject` in _some_ cases. And indeed,
                #   since `pyobject`s are essentially `void` pointers is was, in principle possible, to return/pass
                #   them as "scalars", read "not inside an array".
                #   However, if the return value was passed as argument, i.e. the first `elif`, then it
                #   was ignored if `arr` was a `pyobject`. Only if the return value was managed by `self`,
                #   i.e. the `else` case, then it was considered, in a way at least. The problem was, that it was
                #   done using the following check:
                #       `isinstance(arr, dt.Scalar) or isinstance(arr.dtype, dtypes.pyobject)`
                #   Because of the `or` that is used, _everything_ whose `dtype` is `pyobject` was classified
                #   as a scalar `pyobject`, i.e. one element, even if it was in fact an array of millions of `pyobject`s.
                #   The correct behaviour would be to change the `or` to an `and` but then several unit
                #   tests (`test_pyobject_return`, `test_pyobject_return_tuple` and `test_nested_autoparse[False]`
                #   in `tests/python_frontend/callee_autodetect_test.py`) will fail.
                #   The following code is bug compatible and also allows to pass a `pyobject` directly, i.e.
                #   through `kwargs`.
                if isinstance(arr.dtype, dtypes.pyobject):
                    if isinstance(arr, dt.Scalar):
                        # Proper scalar.
                        self._retarray_is_pyobject.append(True)
                    elif isinstance(arr, dt.Array):
                        # An array, let's check if it is just a wrapper for a single value.
                        if not (len(arr.shape) == 1 and arr.shape[0] == 1):
                            warnings.warn(f'Decay an array of `pyobject`s with shape {arr.shape} to a single one.',
                                          stacklevel=2)
                        self._retarray_is_pyobject.append(True)
                    else:
                        raise ValueError(
                            f'Does not know how to handle "{arrname}", which is a {type(arr).__name__} of `pyobject`.')
                else:
                    self._retarray_is_pyobject.append(False)

        assert (not self._is_single_value_ret) or (len(self._return_arrays) == 1)
        assert len(self._return_arrays) == len(self._retarray_shapes) == len(self._retarray_is_pyobject)
        self._return_arrays = tuple(self._return_arrays)
