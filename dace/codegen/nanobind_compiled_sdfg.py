# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The user-facing wrapper for compiled SDFGs built as nanobind modules.

Loading (importlib under the ``dace.generated.*`` namespace) lives in
``dace.codegen.compiler``; this module only contains the wrapper class.
"""

from dace import data as dt, dtypes, symbolic
from dace.codegen import compiler

import pathlib
import numpy as np
import ctypes


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
        C calls.

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
    :note: The arrays used as return values are allocated fresh on every call.
    :note: It is not possible to return Python scalars, exactly as with
           ``CompiledSDFG``.
    """

    def __init__(self, sdfg, module, arg_names):
        self._sdfg = sdfg
        self._module = module
        self._arg_names = list(arg_names or [])
        self._handle = module.make_compiled_sdfg()

        # Name of the return values (special DaCe names).
        self._return_values = tuple(
            sorted(name for name, desc in sdfg.arrays.items() if name.startswith('__return') and (not desc.transient)))

        # Arguments that are structures. Needs to be a `set` for fast lookups.
        self._struct_args = frozenset(name for name, desc in sdfg.arrays.items()
                                      if (not desc.transient) and isinstance(desc, dt.Structure))

    @property
    def sdfg(self):
        return self._sdfg

    @property
    def module(self):
        return self._module

    @property
    def filename(self):
        """The resolved absolute path to the loaded extension module (the built .so).

        Parity with ``CompiledSDFG.filename``; some callers rely on the path
        being absolute. Backed by the imported module's ``__file__``.
        """
        return str(pathlib.Path(self._module.__file__).resolve())

    @property
    def has_gpu_code(self) -> bool:
        return self._handle.has_gpu_code

    def __call__(self, *args, **kwargs):
        """Runs the SDFG, initializing the state on demand on the first call.

        Arguments are passed positionally (in ``arg_names`` order) and/or by
        keyword. Return-value arrays are freshly allocated and returned - a
        single array, or a tuple when there are several; with no return values
        the result is ``None``.

        Structure arguments may be passed as the ``ctypes.Structure`` the ctypes
        path uses; they are forwarded as a pointer to it.
        """
        if self._struct_args:
            # keepalive keeps the ctypes objects alive until the handle returns.
            args, kwargs, keepalive = self._marshal_structures(args, kwargs)

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

        if len(return_arrays) == 1:
            return return_arrays[0]
        return tuple(return_arrays)

    def _marshal_structures(self, args, kwargs):
        """Replaces each ``Structure`` argument value with the address of the ctypes object.

        The function returns the modified `args` and `kwargs` entities and as a third
        value a list of all ``Structure`` that were previously stored in them, but
        where replaced with their address.
        """
        keepalive = []

        def to_address(value):
            keepalive.append(value)
            return ctypes.addressof(value)

        args = tuple(
            to_address(v) if (i < len(self._arg_names) and self._arg_names[i] in self._struct_args) else v
            for i, v in enumerate(args))
        kwargs = {k: (to_address(v) if k in self._struct_args else v) for k, v in kwargs.items()}
        return args, kwargs, keepalive

    def _allocate_return_arrays(self, kwargs):
        """Allocates the ``__return*`` arrays (fresh each call) and adds them to ``kwargs``.

        There is no pyobject handling here (including the PR#2206 bug-compatible
        decay of pyobject arrays to a single value): pyobject is deferred to
        part 2 of the nanobind port, and the bindings generator already rejects
        such SDFGs at codegen time. Whether part 2 replicates the PR#2206
        behavior or fixes it is an open decision recorded there.
        """
        arrays = self._sdfg.arrays
        syms = {k: v for k, v in kwargs.items() if k not in arrays}
        syms.update(self._sdfg.constants)

        return_arrays = []
        for name in self._return_values:
            desc = arrays[name]
            if name in kwargs:
                # This constraint also simplifies the implementation of struct handling.
                raise ValueError(f'The implicit output argument `{name}` can not be passed as explicit input argument.')
            if not isinstance(desc, dt.Array):
                # This is also a limitation of the current DaCe interface, as scalars are always passed as values.
                raise NotImplementedError(f'Nanobind interface: return value "{name}" of type '
                                          f'{type(desc).__name__} is not supported yet.')
            shape = tuple(int(symbolic.evaluate(s, syms)) for s in desc.shape)
            dtype = desc.dtype.as_numpy_dtype()
            total_size = int(symbolic.evaluate(desc.total_size, syms))
            strides = tuple(int(symbolic.evaluate(s, syms)) * desc.dtype.bytes for s in desc.strides)
            arr = np.ndarray(shape, dtype, buffer=np.zeros(total_size, dtype), strides=strides)
            # A dtypes.struct element is bound as nb::ndarray<uint8_t> (nanobind
            # rejects numpy record arrays), so hand the handle a byte view that
            # shares the buffer; the caller still gets the structured array back.
            kwargs[name] = arr.view(np.uint8) if isinstance(desc.dtype, dtypes.struct) else arr
            return_arrays.append(arr)
        return return_arrays

    def initialize(self, *args, **kwargs):
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

    def finalize(self):
        self._handle.finalize()

    def safe_call(self, *args, **kwargs):
        """Runs the SDFG in a separate process, so a crash raises here instead of killing the caller.

        Output travels through the in/out arguments (return values are not
        supported); delegates to the interface-agnostic ``safe_call_precompiled``.
        """
        return compiler.safe_call_precompiled(self._sdfg, args, kwargs)

    def get_workspace_sizes(self):
        """Returns the external-memory sizes per storage type.

        The handle keys the sizes by storage-type *name* (stable across changes
        to the enum values); the conversion back to the ``StorageType`` enum
        happens here, since the Python enum is not exposed to C++.
        """
        return {getattr(dtypes.StorageType, k): v for k, v in self._handle.get_workspace_sizes().items()}

    def set_workspace(self, storage, workspace):
        """Sets the workspace for the given storage type to the given buffer."""
        name = storage.name if isinstance(storage, dtypes.StorageType) else dtypes.StorageType(storage).name
        self._handle.set_workspace(name, workspace)

    def state_fields(self):
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
        """
        # ``state_pointer`` raises if the state is uninitialized or finalized.
        return self._handle.state_pointer

    def get_exported_function(self, name: str, restype=None):
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
