# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The user-facing wrapper for compiled SDFGs built as nanobind modules.

Loading (importlib under the ``dace.generated.*`` namespace) lives in
``dace.codegen.compiler``; this module only contains the wrapper class.
"""

from typing import Union, List, Any, Tuple, Dict, Optional, Callable
from types import ModuleType
import pathlib
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
    """A compiled SDFG object that can be called; the nanobind counterpart of ``CompiledSDFG``.

    Do not construct this class directly; it is created by utilities such as
    ``SDFG.compile()`` when ``compiler.interface`` is set to ``nanobind``.

    The class performs the same tasks as ``CompiledSDFG``, but delegates them
    to the generated nanobind module:

    - It ensures that the SDFG object is properly initialized, either by a
        direct call to ``initialize()`` or the first time it is called.
        Furthermore, it will also take care of the finalization if the handle
        goes out of scope.
    - Marshalling Python arguments into C arguments happens in the generated
        C++ code (nanobind's dispatcher), with the GIL released around the
        C calls. This includes struct arguments, whose raw pointer the
        generated code reads via the Python buffer protocol.

    Unlike ``CompiledSDFG`` there is only ``__call__()``; the advanced
    three-step interface (``construct_arguments()`` / ``fast_call()`` /
    ``convert_return_values()``) is not provided, because the argument
    processing already happens in compiled code — there is one fast path.
    Both classes satisfy (structurally, without inheriting)
    :class:`~dace.codegen.compiled_sdfg.CompiledSDFGProtocol` — the surface
    interface-agnostic code may rely on.

    :param sdfg: The ``SDFG`` this wrapper was compiled from; used to evaluate
                 return-array shapes and exposed via the ``sdfg`` property.
    :param module: The imported nanobind extension module (the generated shared
                   library) providing ``make_compiled_sdfg()``.
    :param arg_names: The user-facing positional argument order
                      (``sdfg.arg_names``), used to map positional call
                      arguments to their names.
    :note: The arrays used as return values are allocated fresh on every call,
           unless ``compiler.nanobind_allow_return_override`` is enabled and the
           caller passes their own ``__return*`` buffer.
    :note: Return values are arrays only; unlike the ctypes ``CompiledSDFG`` the
           nanobind interface returns neither Python scalars nor pyobjects.
           A return array with ``GPU_Global`` storage is allocated with (and
           returned as) a CuPy array; without CuPy installed such a call
           raises ``NotImplementedError``.
    :note: No argument or symbol values are stored between calls; unlike the
           ctypes ``CompiledSDFG`` (which reuses the last call's arguments,
           ``_lastargs``), :meth:`get_workspace_sizes` and :meth:`set_workspace`
           take the symbol values they depend on as arguments of that call.
    :note: Compiled-SDFG call hooks (``dace.hooks``, ``dace.profile``) are
           supported; a hook's ``args`` parameter is a 1-tuple holding the
           processed keyword arguments (there is no ctypes-style C-argument
           tuple - marshalling happens in compiled code).

    :note: **Not thread-safe** (accepted by design: a handle is not meant to
           be shared across threads). Calls carry all their per-call data,
           but the lazy state initialization is an unsynchronized
           check-then-act that runs with the GIL released, concurrent calls
           share the single SDFG state struct (persistent transients,
           workspace), and ``finalize()`` frees that state without
           synchronizing with in-flight calls. Distinct instances are
           independent. The ctypes ``CompiledSDFG`` has the same
           initialization race and is additionally unsafe under mere
           GIL-interleaving (``_lastargs``).
    """

    def __init__(self, sdfg: "dace.SDFG", module: ModuleType, arg_names: List[str]):
        self._sdfg: "dace.SDFG" = sdfg
        self._module: ModuleType = module
        self._arg_names: List[str] = list(arg_names or [])
        self._handle: Any = module.make_compiled_sdfg()  # TODO: create a protocol for it.
        #: When True, calls skip the program execution (argument processing and
        #: hooks still run). Toggled by hooks such as ``dace.profile``, which
        #: runs the repetitions itself and then suppresses the hooked call.
        self.do_not_execute: bool = False

        if '__return' in sdfg.arrays:
            self._return_values: Tuple[str] = ('__return', )
            self._is_single_value_ret = True
        else:
            # Can not use `sorted()` here, because then `__return_11` would be ordered before `__return_2`.
            return_names = {name for name in sdfg.arrays if name.startswith('__return_')}
            self._return_values = tuple(f'__return_{i}' for i in range(len(return_names)))
            self._is_single_value_ret = False
            assert return_names == set(self._return_values)

        # Callback arguments; each call wraps the passed callable in a ctypes
        # CFUNCTYPE (see _process_callbacks).
        self._callback_args: Dict[str, Any] = {
            name: desc.dtype
            for name, desc in sdfg.arglist().items() if isinstance(desc.dtype, dtypes.callback)
        }
        self._callback_keepalive: List[Any] = []
        self._callback_refs: List[Any] = []

        # No return arrays to allocate and no callbacks to wrap: unless hooks
        # are registered, a call needs no Python-side processing at all.
        self._simple_call: bool = not self._return_values and not self._callback_args

    @property
    def sdfg(self) -> "dace.SDFG":
        return self._sdfg

    @property
    def module(self) -> ModuleType:
        return self._module

    @property
    def filename(self) -> str:
        """The resolved absolute path to the loaded extension module (the built .so).

        Parity with ``CompiledSDFG.filename``; some callers rely on the path
        being absolute. Backed by the imported module's ``__file__``.
        """
        return str(pathlib.Path(self._module.__file__).resolve())

    @property
    def has_gpu_code(self) -> bool:
        return self._handle.has_gpu_code

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Runs the SDFG, initializing the state on demand on the first call.

        Arguments are passed positionally (in ``arg_names`` order) and/or by
        keyword. Return-value arrays are freshly allocated and returned - a
        single array, or a tuple when there are several; with no return values
        the result is ``None``. Passing a ``__return*`` buffer explicitly is
        refused unless ``compiler.nanobind_allow_return_override`` is enabled.
        """
        # Fast path - no return arrays, no callbacks, no hooks: hand the
        # arguments straight to the compiled dispatcher.
        hooks_active = bool(hooks._COMPILED_SDFG_CALL_HOOKS)
        if self._simple_call and not hooks_active:
            if self.do_not_execute is False:
                self._handle(*args, **kwargs)
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

        # Process the return value.
        if self._is_single_value_ret:
            return return_arrays[0]
        return tuple(return_arrays)

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
        """Allocates the ``__return*`` arrays (fresh each call) and adds them to ``kwargs``.

        A caller-provided return buffer is accepted only when
        ``compiler.nanobind_allow_return_override`` is enabled; its shape and
        dtype are then the caller's responsibility.
        """
        arrays = self._sdfg.arrays
        syms = {k: v for k, v in kwargs.items() if k not in arrays}
        syms.update(self._sdfg.constants)

        # Config lookups are not free; resolve lazily, at most once per call.
        nanobind_allow_return_override: Union[bool, None] = None

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
        initialize the state (the init symbols) are actually consumed. Calling
        this is optional - :meth:`__call__` initializes on demand - but it is
        required before querying external-memory workspace sizes.
        """
        self._handle.initialize(**self._named_call_arguments(args, kwargs))

    def _named_call_arguments(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Maps positional arguments onto their names and wraps callback callables."""
        if args:
            assert not (multiple_names :=
                        kwargs.keys() & self._arg_names[:len(args)]), f"Specified '{multiple_names}' multiple times."
            kwargs.update(zip(self._arg_names, args, strict=False))
        if self._callback_args:
            self._process_callbacks(kwargs)
        return kwargs

    def finalize(self) -> None:
        self._handle.finalize()

    def safe_call(self, *args: Any, **kwargs: Any) -> Any:
        """Runs the SDFG in a separate process, so a crash raises here instead of killing the caller.

        Output travels through the in/out arguments (return values are not
        supported); delegates to the interface-agnostic ``safe_call_precompiled``.
        """
        return compiler.safe_call_precompiled(self._sdfg, args, kwargs)

    def get_workspace_sizes(self, *args: Any, **kwargs: Any) -> Dict[dtypes.StorageType, int]:
        """Returns the external-memory sizes per storage type.

        Symbol values are never stored on the handle, so the sizes are computed
        from the arguments of *this* call: pass the symbols the sizes depend
        on. Any subset of the :meth:`__call__` arguments is accepted; only the
        needed values are consumed.
        """
        # The handle keys the sizes by storage-type *name* (stable across
        # changes to the enum values); the Python enum is not exposed to C++,
        # so the conversion back happens here.
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

        Baked into the module at code-generation time - no parsing of the
        generated sources.
        """
        return list(self._handle.state_fields())

    def get_state_struct(self) -> int:
        """Returns the *raw pointer* to the state struct as an integer.

        The state must be initialized; querying it beforehand raises. Combine
        it with :meth:`state_fields` if you need to interpret the memory.
        Callers must know exactly what they are doing - and whatever they are
        doing with this pointer is most likely wrong.

        :note: The return value differs from the ctypes
               ``CompiledSDFG.get_state_struct()``, which returns a
               ``ctypes.Structure`` dynamically assembled from the parsed source
               (its ``_try_parse_state_struct()``); here only the integer
               pointer value is returned.
        :todo: Align this function with the `ctypes` version.
        """
        # ``state_pointer`` raises if the state is uninitialized or finalized.
        return self._handle.state_pointer

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
        lib = ctypes.CDLL(self._module.__file__)
        try:
            func = getattr(lib, name)
        except AttributeError:
            return None
        if restype is not None:
            func.restype = restype
        func.__compiled_sdfg__ = self
        return func
