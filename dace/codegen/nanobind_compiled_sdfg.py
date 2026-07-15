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
from dace import data as dt, dtypes, symbolic
from dace.codegen import compiler
from dace.config import Config


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
    """

    def __init__(self, sdfg: "dace.SDFG", module: ModuleType, arg_names: List[str]):
        self._sdfg: "dace.SDFG" = sdfg
        self._module: ModuleType = module
        self._arg_names: List[str] = list(arg_names or [])
        self._handle: Any = module.make_compiled_sdfg()  # TODO: create a protocol for it.

        if '__return' in sdfg.arrays:
            self._return_values: Tuple[str] = ('__return', )
            self._is_single_value_ret = True
        else:
            # Can not use `sorted()` here, because then `__return_11` would be ordered before `__return_2`.
            return_names = {name for name in sdfg.arrays if name.startswith('__return_')}
            self._return_values = tuple(f'__return_{i}' for i in range(len(return_names)))
            self._is_single_value_ret = False
            assert return_names == set(self._return_values)

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
        # Early exit: no return values, no need to do more processing.
        if not self._return_values:
            self._handle(*args, **kwargs)
            return None

        # With return values, positional arguments must be mapped to names in
        # Python, since the return-shape evaluation needs the symbol values.
        if args:
            if len(args) > len(self._arg_names):
                raise TypeError(f'Too many positional arguments (got {len(args)}, '
                                f'expected at most {len(self._arg_names)}).')
            for name, value in zip(self._arg_names, args):
                if name in kwargs:
                    raise TypeError(f'Argument "{name}" passed both positionally and as a keyword.')
                kwargs[name] = value

        # Will add the return arrays also to `kwargs`.
        return_arrays = self._allocate_return_arrays(kwargs)

        self._handle(**kwargs)

        # A single value is returned bare; a tuple return (even one element) stays a tuple.
        if self._is_single_value_ret:
            return return_arrays[0]
        return tuple(return_arrays)

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
        if args:
            assert not (multiple_names :=
                        kwargs.keys() & self._arg_names[:len(args)]), f"Specified '{multiple_names}' multiple times."
            kwargs.update(zip(self._arg_names, args, strict=False))
        self._handle.initialize(**kwargs)

    def finalize(self) -> None:
        self._handle.finalize()

    def safe_call(self, *args: Any, **kwargs: Any) -> Any:
        """Runs the SDFG in a separate process, so a crash raises here instead of killing the caller.

        Output travels through the in/out arguments (return values are not
        supported); delegates to the interface-agnostic ``safe_call_precompiled``.
        """
        return compiler.safe_call_precompiled(self._sdfg, args, kwargs)

    def get_workspace_sizes(self) -> Dict[dtypes.StorageType, int]:
        """Returns the external-memory sizes per storage type.

        The handle keys the sizes by storage-type *name* (stable across changes
        to the enum values); the conversion back to the ``StorageType`` enum
        happens here, since the Python enum is not exposed to C++.
        """
        return {getattr(dtypes.StorageType, k): v for k, v in self._handle.get_workspace_sizes().items()}

    def set_workspace(self, storage: Union[str, dtypes.StorageType], workspace: int):
        """Sets the workspace for the given storage type to the given buffer."""
        name = storage.name if isinstance(storage, dtypes.StorageType) else dtypes.StorageType(storage).name
        self._handle.set_workspace(name, workspace)

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
