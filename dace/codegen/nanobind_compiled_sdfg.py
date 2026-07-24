# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The user-facing wrapper for compiled SDFGs built as nanobind modules.

Loading (importlib under the ``dace.generated.*`` namespace) lives in
``dace.codegen.compiler``; this module only contains the wrapper class.
"""

from typing import Union, List, Any, Tuple, Dict, Optional, Callable
from types import ModuleType
import pathlib
import warnings

import numpy as np
import ctypes

import dace
from dace import data as dt, dtypes, hooks, symbolic
from dace.codegen import compiler
from dace.config import Config

# Try importing CuPy once (avoid hot path).
try:
    import cupy
except (ImportError, ModuleNotFoundError):
    cupy = None


class NanobindCompiledSDFG:
    """Interface to a compiled SDFG using the ``nanobind`` bindings.

    It allows to call a compiled SDFG binary from Python. Unlike ``CompiledSDFG``
    it does not use ``ctypes`` but ``nanobind``.

    - It ensures that the SDFG object is properly initialized, either by a
        direct call to ``initialize()`` or the first time it is called.
    - Marshalling Python arguments into C arguments, such that it can be called.
        Most of the transformation happens in the bindings and it is thus faster.

    Unlike ``CompiledSDFG`` there is only ``__call__()``, the advanced
    three-step interface (``construct_arguments()`` / ``fast_call()`` /
    ``convert_return_values()``) is not provided. Otherwise it implements the same
    interface as ``CompiledSDFG``, with some deviations listed bellow.

    :param sdfg: The ``SDFG`` this wrapper was compiled from; used to evaluate
                 return-array shapes and exposed via the ``sdfg`` property.
    :param module: The imported nanobind extension module.
    :param arg_names: The user-facing positional argument order, i.e. ``sdfg.arg_names``,
                      used to map positional call arguments to their names.

    :note: The allocation of the return arrays is performed in Python and will slow
           down calling. By setting ``compiler.nanobind_allow_return_override`` it
           is possible to pass them, i.e. the special ``__return*`` arguments,
           explicitly to ``__call__()``.
    :note: Return values are arrays only; unlike the ctypes ``CompiledSDFG`` the
           nanobind interface returns neither Python scalars nor pyobjects.
    :note: Some symbolic arguments are automatically inferred by the bindings, such
           as symbols used as shape arguments. However, it is restricted to arrays
           of fundamental types. And symbols needed for the return values have to be
           provided explicitly.
    :note: Marshalling of Python callbacks is done in Python.
    :note: There is no caching of the "previous call arguments", i.e.
           ``CompiledSDFG._lastargs``. This means that the symbolic sizes must be
           explicitly passed to :meth:`get_workspace_sizes` and :meth:`set_workspace`.
           take the symbol values they depend on as arguments of that call.
    :note: Initialization is not thread safe. Calling the SDFG is thread safe only
           if ``self`` is already initialized and the SDFG does not have persistent
           or external memory. Furthermore, ``finalize()`` and the retrieval of
           GPU erros is not thread safe.
    :note: This class will not unload the module.
    """

    def __init__(self, sdfg: "dace.SDFG", module: ModuleType, arg_names: List[str], gpu_error_check: bool = True):
        self._sdfg: "dace.SDFG" = sdfg
        self._module: ModuleType = module
        self._arg_names: List[str] = list(arg_names or [])
        self._handle: Any = module.make_compiled_sdfg()  # TODO: create a protocol for it.
        #: When True, calls skip the program execution (argument processing and
        #: hooks still run). Toggled by hooks such as ``dace.profile``, which
        #: runs the repetitions itself and then suppresses the hooked call.
        self.do_not_execute: bool = False

        # Codegen-time call metadata comes from the handle: the `__return*`
        # naming convention and the callback detection live in the bindings
        # generator, not here.
        self._return_values: Tuple[str, ...] = tuple(self._handle.return_names)
        self._is_single_value_ret: bool = self._handle.is_single_value_ret

        # Callback arguments; each call wraps the passed callable in a ctypes `CFUNCTYPE` (see _process_callbacks).
        arglist = sdfg.arglist()
        self._callback_args: Dict[str, Any] = {name: arglist[name].dtype for name in self._handle.callback_names}
        self._callback_keepalive: List[Any] = []
        self._callback_refs: List[Any] = []

        # No return arrays to allocate and no callbacks to wrap: unless hooks
        # are registered, a call needs no Python-side processing at all.
        self._simple_call: bool = not (self._return_values or self._callback_args)

        # Static per module; cached for the per-call GPU error check.
        self._has_gpu_code: bool = bool(self._handle.has_gpu_code)
        # See the ``gpu_error_check`` property. Read directly on the (GPU-only) call path.
        self._gpu_error_check: bool = bool(gpu_error_check)

    @property
    def sdfg(self) -> "dace.SDFG":
        return self._sdfg

    @property
    def module(self) -> ModuleType:
        """The extension module used to construct ``self``."""
        return self._module

    @property
    def filename(self) -> str:
        """The resolved absolute path to the loaded extension module (the built .so).
        """
        return str(pathlib.Path(self._module.__file__).resolve())

    @property
    def has_gpu_code(self) -> bool:
        return self._handle.has_gpu_code

    @property
    def gpu_error_check(self) -> bool:
        """Whether each call on a GPU SDFG checks the GPU runtime's last error afterwards.

        Parity with the ctypes ``fast_call`` ``do_gpu_check``. Defaults to the
        constructor argument (``True``); set to ``False`` to skip the check (a
        small speedup). Has no effect on a CPU-only SDFG.
        """
        return self._gpu_error_check

    @gpu_error_check.setter
    def gpu_error_check(self, value: bool) -> None:
        self._gpu_error_check = bool(value)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the compiled SDFG.

        The function will forward the call to compiled executable. The arguments can
        either be passed as positional, if they are listed in ``arg_names`` or as
        keyword arguments. If the SDFG is not initialized it will be initialized first.

        The interface is able to infer some symbolic arguments from other arguments.
        The example is a symbol that is also used in the shape of an array. However,
        this is only possible if it was not listed in ``arg_names`` and as sources
        only arrays of fundamental types are considered, i.e. no arrays of structs
        or ``ContainerArray``s. Furthermore, symbols that are needed to compute the
        size of the return values, must be explicitly passed.

        Passing a ``__return*`` buffer explicitly is refused unless
        ``compiler.nanobind_allow_return_override`` is enabled.
        """
        # Fast path - no return arrays, no callbacks, no hooks: hand the
        # arguments straight to the compiled dispatcher.
        hooks_active = bool(hooks._COMPILED_SDFG_CALL_HOOKS)
        if self._simple_call and not hooks_active:
            if self.do_not_execute is False:
                self._handle(*args, **kwargs)
                if self._has_gpu_code and self._gpu_error_check:
                    self._check_gpu_error()
            return None

        # Handle positional arguments and move them into `kwargs`.
        if args:
            if len(args) > len(self._arg_names):
                raise TypeError(f'Too many positional arguments (got {len(args)}, '
                                f'expected at most {len(self._arg_names)}).')
            for name, value in zip(self._arg_names, args, strict=False):
                if name in kwargs:
                    raise TypeError(f'Argument "{name}" passed both positionally and as a keyword.')
                kwargs[name] = value

        # Process the callbacks.
        if self._callback_args:
            self._process_callbacks(kwargs)

        # No return value given.
        if not self._return_values:
            if hooks_active:
                self._call_handle_with_hooks(kwargs)
            elif self.do_not_execute is False:
                self._handle(**kwargs)
            if self._has_gpu_code and self._gpu_error_check:
                self._check_gpu_error()
            return None

        # NOTE: Return shapes are evaluated here in Python, so symbols they
        # depend on must be passed explicitly - the compiled shape inference
        # only serves the kernel call, and running the (sympy-based) Python
        # inference per call would be slow.

        # Will add the return arrays also to `kwargs`.
        return_arrays = self._allocate_return_arrays(kwargs)

        # Perform the call to the compiled extension.
        if hooks_active:
            self._call_handle_with_hooks(kwargs)
        elif self.do_not_execute is False:
            self._handle(**kwargs)
        if self._has_gpu_code and self._gpu_error_check:
            self._check_gpu_error()

        # Process the return value.
        if self._is_single_value_ret:
            return return_arrays[0]
        return tuple(return_arrays)

    def _check_gpu_error(self) -> None:
        """Raises if the GPU runtime recorded an error during the last call.

        Note failing to obtain the runtime only warns. It is also important that this
        function is not thread safe.
        """
        from dace.codegen import common  # Circular import; the CPU hot path never pays it.
        try:
            lasterror = common.get_gpu_runtime().get_last_error_string()
        except RuntimeError as ex:
            warnings.warn(f'Could not get last error from GPU runtime: {ex}')
            return
        if lasterror is not None:
            raise RuntimeError(f'An error was detected when calling "{self._sdfg.name}": {lasterror}. '
                               'Consider enabling synchronous debugging mode (environment variable: '
                               'DACE_compiler_cuda_syncdebug=1) to see where the issue originates from.')

    def _call_handle_with_hooks(self, kwargs: Dict[str, Any]) -> None:
        """Runs the handle inside the registered compiled-SDFG call hooks.

        On this interface a hook's ``args`` parameter is a 1-tuple holding the
        processed keyword arguments (marshalling happens in compiled code, so
        there is no ctypes-style C-argument tuple to hand out); re-invoking the
        program from a hook is ``compiled_sdfg._handle(**args[0])``.
        """
        with hooks.invoke_compiled_sdfg_call_hooks(self, (kwargs, )):
            # Checked inside the hook context: a hook may toggle the flag
            # (dace.profile does) before the program call would run.
            if self.do_not_execute is False:
                self._handle(**kwargs)

    def _process_callbacks(self, kwargs: Dict[str, Any]) -> None:
        """Replaces callback callables in ``kwargs`` with C function-pointer addresses.

        Each callable is wrapped in the same trampoline + ctypes CFUNCTYPE the
        ctypes interface uses - the libffi thunk re-acquires the GIL on entry,
        so the kernel may invoke it while running GIL-free (also from worker
        threads). Exceptions raised inside a callback follow ctypes semantics:
        they are printed, not propagated to the caller.
        """
        # The CFUNCTYPE objects own the synthesized C entry points. They are
        # retained for the wrapper's lifetime rather than per call: the opaque
        # SDFG state may keep a pointer passed at initialization, so per-call
        # replacement could leave a stored pointer dangling. Return-value
        # references only need to survive until the kernel has consumed them,
        # i.e. the next call.
        self._callback_refs.clear()
        for name, cbtype in self._callback_args.items():
            if name not in kwargs:
                continue
            trampoline = cbtype.get_trampoline(kwargs[name], kwargs, self._callback_refs, None)
            cfunc = cbtype.as_ctypes()(trampoline)
            self._callback_keepalive.append(cfunc)
            kwargs[name] = ctypes.cast(cfunc, ctypes.c_void_p).value

    def _allocate_return_arrays(self, kwargs: Dict[str, Any]) -> List[Any]:
        """Allocates the ``__return*`` arrays.

        The function will return them _and_ add them to ``kwargs``.
        It is important that a caller-provided return buffer is accepted only when
        ``compiler.nanobind_allow_return_override`` is enabled and all layout related
        symbols must be provided.
        """
        arrays = self._sdfg.arrays
        syms = {k: v for k, v in kwargs.items() if k not in arrays}
        syms.update(self._sdfg.constants)

        # Config lookups are not free; resolve lazily, at most once per call.
        nanobind_allow_return_override: Optional[bool] = None

        return_arrays = []
        for name in self._return_values:
            desc = arrays[name]
            assert isinstance(desc, dt.Array)  # Non-array returns are refused by the bindings generator at codegen

            if name in kwargs:
                # Caller-provided output buffer (opt-in): the kernel writes into
                # and returns it. No shape/dtype validation - the caller owns
                # correctness; the binding rejects what it cannot accept.
                if nanobind_allow_return_override is None:
                    nanobind_allow_return_override = Config.get_bool('compiler', 'nanobind_allow_return_override')

                if not nanobind_allow_return_override:
                    raise ValueError(f'The implicit output argument `{name}` can not be passed as an explicit '
                                     f'input argument; set compiler.nanobind_allow_return_override=true to allow '
                                     f'reusing a caller-provided output buffer.')
                arr = kwargs[name]
            else:
                shape = tuple(int(symbolic.evaluate(s, syms)) for s in desc.shape)
                dtype = desc.dtype.as_numpy_dtype()
                total_size = int(symbolic.evaluate(desc.total_size, syms))
                strides = tuple(int(symbolic.evaluate(s, syms)) * desc.dtype.bytes for s in desc.strides)
                if desc.storage is dtypes.StorageType.GPU_Global:
                    if cupy is None:
                        raise NotImplementedError('GPU return values are unsupported if cupy is not installed')
                    arr = cupy.ndarray(shape, dtype, memptr=cupy.zeros(total_size, dtype).data, strides=strides)
                else:
                    arr = np.ndarray(shape, dtype, buffer=np.zeros(total_size, dtype), strides=strides)
            kwargs[name] = arr
            return_arrays.append(arr)

        return return_arrays

    def initialize(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the SDFG state eagerly, without running it.

        Accepts the same arguments as :meth:`__call__` (positional arguments in
        ``arg_names`` order and/or keywords); only the values needed to
        initialize the state (the init symbols) are actually used. Calling
        this is optional - :meth:`__call__` initializes on demand - but it is
        required before querying external-memory workspace sizes. Furthermore, it
        is not thread safe.
        """
        self._handle.initialize(**self._named_call_arguments(args, kwargs))

    def _named_call_arguments(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Add the positional arguments to ``kwargs`` and wraps callback callables."""
        if args:
            assert not (multiple_names :=
                        kwargs.keys() & self._arg_names[:len(args)]), f"Specified '{multiple_names}' multiple times."
            kwargs.update(zip(self._arg_names, args, strict=False))
        if self._callback_args:
            self._process_callbacks(kwargs)
        return kwargs

    def finalize(self) -> None:
        """Finalizes the compiled SDFG explicitly.

        This function will deallocate the internal state and free all persistent memory.
        Note that this is not thread safe and needs synchronization.
        It is possible to reinitialize a previously finalized compiled SDFG.
        """
        rc = self._handle.finalize()
        if rc != 0:
            raise RuntimeError(f'An error was detected after running "{self._sdfg.name}": {self._get_error_text(rc)}')

    def _get_error_text(self, result: Union[int, str]) -> str:
        """Translates a ``__dace_exit`` code into text (ctypes ``_get_error_text`` parity).

        With GPU code the numeric code is a GPU error code and goes through the
        GPU runtime's ``get_error_string``; without, it is reported as-is.
        """
        from dace.codegen import common  # Circular import
        if self._has_gpu_code:
            if isinstance(result, int):
                result = common.get_gpu_runtime().get_error_string(result)
            return (f'{result}. Consider enabling synchronous debugging mode (environment variable: '
                    'DACE_compiler_cuda_syncdebug=1) to see where the issue originates from.')
        else:
            return result

    def safe_call(self, *args: Any, **kwargs: Any) -> Any:
        """Runs the SDFG in a separate process, so a crash raises here instead of killing the caller.

        Output travels through the in/out arguments (return values are not
        supported); delegates to the interface-agnostic ``safe_call_precompiled``.
        """
        return compiler.safe_call_precompiled(self._sdfg, args, kwargs)

    def get_workspace_sizes(self, *args: Any, **kwargs: Any) -> Dict[dtypes.StorageType, int]:
        """Returns the external-memory sizes per storage type.

        Unlike the version provided by ``CompiledSDFG`` the symbolic sizes must be provided.
        Any subset of the :meth:`__call__` arguments is accepted; only the needed values are consumed.
        """
        kwargs = self._named_call_arguments(args, kwargs)
        return {getattr(dtypes.StorageType, k): v for k, v in self._handle.get_workspace_sizes(**kwargs).items()}

    def set_workspace(self, storage: Union[str, dtypes.StorageType], workspace: int, *args: Any, **kwargs: Any):
        """Sets the workspace for the given storage type to the given buffer.

        As with :meth:`get_workspace_sizes`, the symbol values the external
        memory depends on are taken from this call's arguments (any subset of
        the :meth:`__call__` arguments is accepted).
        """
        name = storage.name if isinstance(storage, dtypes.StorageType) else dtypes.StorageType(storage).name
        self._handle.set_workspace(name, workspace, **self._named_call_arguments(args, kwargs))

    def state_fields(self) -> list[str]:
        """Names of the pointer fields in the state struct.
        """
        return list(self._handle.state_fields())

    def get_state_struct(self) -> ctypes.Structure:
        """Returns a live, mutable ``ctypes.Structure`` view of the state struct.

        Parity with ``CompiledSDFG.get_state_struct``: the structure overlays the
        live state memory and exposes the leading pointer fields (as
        ``c_void_p``) by name, so callers can ``getattr``/``setattr`` them. The
        state must be initialized; querying it beforehand raises.

        :note: The structure aliases state memory owned by the handle; it must
               not be used after :meth:`finalize`.
        """
        # ``state_pointer`` raises if the state is uninitialized or finalized.
        ptr = self._handle.state_pointer
        fields = [(name, ctypes.c_void_p) for name in self._handle.state_fields()]
        state_struct_t = type('State', (ctypes.Structure, ), {'_fields_': fields})
        return state_struct_t.from_address(ptr)

    def get_exported_function(self, name: str, restype=None) -> Optional[Callable]:
        """Returns an arbitrary exported symbol as a callable, or None if absent.

        Resolved with ``ctypes.CDLL`` on the already-imported module file
        (which returns the same library handle); the wrapper is attached to
        the returned function as ``__compiled_sdfg__`` to keep the module
        alive.

        :note: Reaching for this function should be considered a bug - it is a
               low-level escape hatch that bypasses the typed interface, and
               anything it is used for is most likely better done another way.
        """
        lib = ctypes.CDLL(self.filename)
        try:
            func = getattr(lib, name)
        except AttributeError:
            return None
        if restype is not None:
            func.restype = restype
        func.__compiled_sdfg__ = self
        return func
