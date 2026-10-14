# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The user-facing wrapper for compiled SDFGs built as nanobind modules.

Loading (importlib under the ``dace.generated.*`` namespace) lives in
``dace.codegen.compiler``; this module only contains the wrapper class.
"""

import ctypes
import dataclasses
import pathlib
import warnings
from collections.abc import Callable
from types import ModuleType
from typing import Any, ClassVar

import numpy as np

import dace
from dace import dtypes, hooks
from dace.codegen import compiler


@dataclasses.dataclass
class NanobindInterface:
    """Descriptor for a decided nanobind compilation (see ``resolve_compiler_interface``).

    Carries the pre-codegen content hash of the source SDFG from ``compile()``
    to the bindings generator, which bakes it into the module for
    loaded-module reuse. ``None`` when reuse is disabled.
    """

    #: The ``compiler.interface`` config value and ``INTERFACE`` marker content.
    name: ClassVar[str] = "nanobind"

    source_hash: str | None = None


def _unwrap_array_likes(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
    """Swaps array-like WRAPPER objects for zero-copy ndarray views, or ``None``.

    The ctypes marshaller accepted any ``dtypes.is_array`` object (NDSL's
    Quantity, GT4Py storages, ...); nanobind's dispatcher ingests numpy/DLPack
    objects only. This repair runs only after a failed dispatch and replaces
    each such wrapper by a zero-copy view of its buffer, preserving
    by-reference semantics. Returns ``None`` when nothing was unwrapped, so
    the caller re-raises the original dispatch error.
    """

    def _view(value):
        if isinstance(value, np.ndarray) or not dtypes.is_array(value):
            return None
        if hasattr(value, "__cuda_array_interface__"):
            import cupy

            return cupy.asarray(value)
        if hasattr(value, "__array_interface__"):
            return np.asarray(value)
        return None

    changed = False
    new_args = []
    for value in args:
        view = _view(value)
        changed |= view is not None
        new_args.append(value if view is None else view)
    new_kwargs = {}
    for name, value in kwargs.items():
        view = _view(value)
        changed |= view is not None
        new_kwargs[name] = value if view is None else view
    return (tuple(new_args), new_kwargs) if changed else None


class NanobindCompiledSDFG:
    """Interface to a compiled SDFG using the ``nanobind`` bindings.

    It allows to call a compiled SDFG binary from Python; unlike
    ``CompiledSDFG`` the marshalling happens in the compiled bindings, not in
    ``ctypes``. It implements the same interface as ``CompiledSDFG``, with
    these deviations:

    - The advanced three-step interface (``construct_arguments()`` /
      ``fast_call()`` / ``convert_return_values()``) is not provided;
      ``__call__()`` forwards straight to the compiled dispatcher.
    - Return values are arrays only (a ``pyobject`` return decays to the
      contained object); Python scalars are never returned. Return arrays are
      allocated inside the binding; see :meth:`__call__` for buffer overrides
      and symbol deduction.
    - There is no caching of the previous call arguments
      (``CompiledSDFG._lastargs``): :meth:`get_workspace_sizes` and
      :meth:`set_workspace` take the symbol values they depend on as
      arguments of that call.
    - Initialization, ``finalize()`` and the retrieval of GPU errors are not
      thread safe; calling is thread safe only once initialized and without
      persistent or external memory.
    - The module is never unloaded.

    :param sdfg: The ``SDFG`` this wrapper was compiled from.
    :param module: The imported nanobind extension module.
    :param arg_names: The user-facing positional argument order, i.e.
                      ``sdfg.arg_names``.
    """

    def __init__(self, sdfg: "dace.SDFG", module: ModuleType, arg_names: list[str], gpu_error_check: bool = True):
        self._sdfg: dace.SDFG = sdfg
        self._module: ModuleType = module
        self._arg_names: list[str] = list(arg_names or [])
        self._handle: Any = module.make_compiled_sdfg()  # TODO: create a protocol for it.
        #: When True, calls skip the program execution (argument processing and
        #: hooks still run). Toggled by hooks such as ``dace.profile``, which
        #: runs the repetitions itself and then suppresses the hooked call.
        self.do_not_execute: bool = False

        # Return handling (naming convention, allocation, tuple-vs-single,
        # buffer validation) lives in the binding; the names stay exposed for
        # introspection.
        self._return_values: tuple[str, ...] = tuple(self._handle.return_names)

        # Static per module; used to translate __dace_exit codes in _get_error_text.
        self._has_gpu_code: bool = bool(self._handle.has_gpu_code)
        self._handle.gpu_error_check = bool(gpu_error_check)

    @property
    def sdfg(self) -> "dace.SDFG":
        return self._sdfg

    @property
    def module(self) -> ModuleType:
        """The extension module used to construct ``self``."""
        return self._module

    @property
    def filename(self) -> str:
        """The resolved absolute path to the loaded extension module (the built .so)."""
        return str(pathlib.Path(self._module.__file__).resolve())

    @property
    def has_gpu_code(self) -> bool:
        return self._handle.has_gpu_code

    @property
    def is_initialized(self) -> bool:
        """Whether the SDFG state is currently initialized.

        True after :meth:`initialize` (or the first call) ran and no
        :meth:`finalize` happened since.
        """
        # state_pointer raises on an uninitialized or finalized state; the
        # handle has no non-throwing probe.
        try:
            return bool(self._handle.state_pointer)
        except RuntimeError:
            return False

    @property
    def gpu_error_check(self) -> bool:
        """Whether each call on a GPU SDFG raises the error its generated code recorded.

        Same mechanism as the ctypes ``fast_call`` ``do_gpu_check``: the
        compiled binding reads the SDFG's OWN error record after each program
        call, never the process-global CUDA last-error slot. No effect on a
        CPU-only SDFG (the check is not even compiled in there).
        """
        return bool(self._handle.gpu_error_check)

    @gpu_error_check.setter
    def gpu_error_check(self, value: bool) -> None:
        self._handle.gpu_error_check = bool(value)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the compiled SDFG.

        Arguments can be passed positionally (if listed in ``arg_names``) or
        as keywords; an uninitialized SDFG is initialized first.

        A symbol not listed in ``arg_names`` may be omitted: its value is
        deduced from the shape or strides of a passed array of fundamental
        type. The deduction expression may reference symbols listed in
        ``arg_names`` (always passed explicitly): with ``A[a + b]`` and ``b``
        in ``arg_names``, an omitted ``a`` is computed as ``A.shape(0) - b``.
        An explicitly passed value takes precedence; a symbol that cannot be
        deduced must be passed.

        Return arrays are allocated and returned by the compiled binding
        itself (after symbol inference, so inferred symbols may size them). A
        ``__return*`` buffer passed explicitly is used as the output buffer
        and returned, validated against the symbol-derived return size (too
        small is rejected). When :attr:`do_not_execute` suppresses the program
        run, ``None`` is returned.
        """
        # This is _named_call_arguments (which the cold entry points call),
        # inlined to keep the hot path free of the extra call - keep in sync.
        if args:
            if multiple_names := kwargs.keys() & set(self._arg_names[: len(args)]):
                raise TypeError(f"Argument(s) {sorted(multiple_names)} passed both positionally and as keywords.")
            if len(args) > len(self._arg_names):
                warnings.warn(
                    f"SDFG '{self._sdfg.name}': {len(args) - len(self._arg_names)} extra positional "
                    f"argument(s) beyond arg_names are ignored (ctypes-interface parity)."
                )
            kwargs.update(zip(self._arg_names, args, strict=False))

        # Fast path - no hooks: hand the arguments straight to the compiled
        # dispatcher. A failed dispatch is repaired once by unwrapping
        # array-like wrappers; native argument types never pay for this.
        if not hooks._COMPILED_SDFG_CALL_HOOKS:
            if self.do_not_execute is False:
                try:
                    return self._handle(**kwargs)
                except TypeError:
                    repaired = _unwrap_array_likes((), kwargs)
                    if repaired is None:
                        raise
                    return self._handle(**repaired[1])
            return None

        return self._call_handle_with_hooks(kwargs)

    def _call_handle_with_hooks(self, kwargs: dict[str, Any]) -> Any:
        """Runs the handle inside the registered compiled-SDFG call hooks.

        On this interface a hook's ``args`` parameter is a 1-tuple holding the
        processed keyword arguments (marshalling happens in compiled code, so
        there is no ctypes-style C-argument tuple to hand out); re-invoking
        the program from a hook is ``compiled_sdfg._handle(**args[0])``.
        """
        # A hook that runs the program itself and then suppresses the hooked
        # call (dace.profile) deposits its last invocation's return value here.
        self._hook_result: Any = None
        result = None
        with hooks.invoke_compiled_sdfg_call_hooks(self, (kwargs,)):
            # Checked inside the hook context: a hook may toggle the flag.
            if self.do_not_execute is False:
                try:
                    result = self._handle(**kwargs)
                except TypeError:
                    repaired = _unwrap_array_likes((), kwargs)
                    if repaired is None:
                        raise
                    result = self._handle(**repaired[1])
        if result is None:
            result = self._hook_result
        return result

    def initialize(self, *args: Any, **kwargs: Any) -> ctypes.c_void_p:
        """Initializes the SDFG state eagerly, without running it.

        Accepts the same arguments as :meth:`__call__`; only the init symbols
        are actually used. Calling this is optional - :meth:`__call__`
        initializes on demand - but required before querying external-memory
        workspace sizes. Not thread safe.

        :return: The state pointer (ctypes-interface parity): functions
                 obtained through :meth:`get_exported_function` take it as
                 their state argument.
        """
        self._handle.initialize(**self._named_call_arguments(args, kwargs))
        return ctypes.c_void_p(self._handle.state_pointer)

    def _named_call_arguments(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
        """Adds the positional arguments to ``kwargs`` (in ``arg_names`` order).

        Positional arguments bind by ``arg_names`` ONLY - never by the
        binding's parameter order, which continues past ``arg_names`` with
        internal ordering. Extras beyond ``arg_names`` are dropped with a
        warning (ctypes zip-truncates them silently, and callers rely on
        that); a name passed both positionally and as a keyword is an error
        (the trailing kwargs absorber would otherwise let the positional win
        silently).
        """
        if args:
            if multiple_names := kwargs.keys() & set(self._arg_names[: len(args)]):
                raise TypeError(f"Argument(s) {sorted(multiple_names)} passed both positionally and as keywords.")
            if len(args) > len(self._arg_names):
                warnings.warn(
                    f"SDFG '{self._sdfg.name}': {len(args) - len(self._arg_names)} extra positional "
                    f"argument(s) beyond arg_names are ignored (ctypes-interface parity)."
                )
            kwargs.update(zip(self._arg_names, args, strict=False))
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

    def _get_error_text(self, result: int | str) -> str:
        """Translates a ``__dace_exit`` code into text (ctypes ``_get_error_text`` parity).

        With GPU code the numeric code is a GPU error code and goes through the
        GPU runtime's ``get_error_string``; without, it is reported as-is.
        """
        from dace.codegen import common  # Circular import

        if self._has_gpu_code:
            if isinstance(result, int):
                result = common.get_gpu_runtime().get_error_string(result)
            return (
                f"{result}. Consider enabling synchronous debugging mode (environment variable: "
                "DACE_compiler_cuda_syncdebug=1) to see where the issue originates from."
            )
        else:
            return result

    def safe_call(self, *args: Any, **kwargs: Any) -> Any:
        """Runs the SDFG in a separate process, so a crash raises here instead of killing the caller.

        Output travels through the in/out arguments (return values are not
        supported); delegates to the interface-agnostic ``safe_call_precompiled``.
        """
        return compiler.safe_call_precompiled(self._sdfg, args, kwargs)

    def get_workspace_sizes(self, *args: Any, **kwargs: Any) -> dict[dtypes.StorageType, int]:
        """Returns the external-memory sizes per storage type.

        Unlike the version provided by ``CompiledSDFG`` the symbolic sizes must be provided.
        Any subset of the :meth:`__call__` arguments is accepted; only the needed values are consumed.
        """
        kwargs = self._named_call_arguments(args, kwargs)
        return {getattr(dtypes.StorageType, k): v for k, v in self._handle.get_workspace_sizes(**kwargs).items()}

    def set_workspace(self, storage: str | dtypes.StorageType, workspace: int, *args: Any, **kwargs: Any):
        """Sets the workspace for the given storage type to the given buffer.

        As with :meth:`get_workspace_sizes`, the symbol values the external
        memory depends on are taken from this call's arguments (any subset of
        the :meth:`__call__` arguments is accepted).
        """
        name = storage.name if isinstance(storage, dtypes.StorageType) else dtypes.StorageType(storage).name
        self._handle.set_workspace(name, workspace, **self._named_call_arguments(args, kwargs))

    def state_fields(self) -> list[str]:
        """Names of the pointer fields in the state struct."""
        return list(self._handle.state_fields())

    def get_state_struct(self) -> ctypes.Structure:
        """Returns a live, mutable ``ctypes.Structure`` view of the state struct.

        The structure overlays the live state memory and exposes the leading
        pointer fields (as ``c_void_p``) by name. The state must be
        initialized, and the view must not be used after :meth:`finalize`.
        """
        ptr = self._handle.state_pointer
        fields = [(name, ctypes.c_void_p) for name in self._handle.state_fields()]
        state_struct_t = type("State", (ctypes.Structure,), {"_fields_": fields})
        return state_struct_t.from_address(ptr)

    def get_exported_function(self, name: str, restype=None) -> Callable | None:
        """Returns an arbitrary exported symbol as a callable, or None if absent.

        Resolved with ``ctypes.CDLL`` on the already-imported module file
        (the same library handle); the wrapper is attached to the returned
        function as ``__compiled_sdfg__`` to keep the module alive. A
        low-level escape hatch that bypasses the typed interface.
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
