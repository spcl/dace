# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The user-facing wrapper for compiled SDFGs built as nanobind modules.

Loading (importlib under the ``dace.generated.*`` namespace) lives in
``dace.codegen.compiler``; this module only contains the wrapper class.
"""


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
        # Cached so calls without return values skip the shape logic entirely.
        self._has_return_values = any(name.startswith('__return') for name in sdfg.arrays.keys())

    @property
    def sdfg(self):
        return self._sdfg

    @property
    def module(self):
        return self._module

    @property
    def has_gpu_code(self) -> bool:
        return self._handle.has_gpu_code

    def __call__(self, *args, **kwargs):
        """Runs the SDFG, initializing the state on demand on the first call.

        Arguments are passed positionally (in ``arg_names`` order) and/or by
        keyword. Return-value arrays are freshly allocated and returned - a
        single array, or a tuple when there are several; with no return values
        the result is ``None``.
        """
        # Early exit: no return values, no shape logic - nanobind's dispatcher
        # does the positional/keyword matching and casting itself.
        if not self._has_return_values:
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

        return_arrays = self._allocate_return_arrays(kwargs)
        self._handle(**kwargs)
        if len(return_arrays) == 1:
            return return_arrays[0]
        return tuple(return_arrays)

    def _allocate_return_arrays(self, kwargs):
        """Allocates the ``__return*`` arrays (fresh each call) and adds them to ``kwargs``.

        There is no pyobject handling here (including the PR#2206 bug-compatible
        decay of pyobject arrays to a single value): pyobject is deferred to
        part 2 of the nanobind port, and the bindings generator already rejects
        such SDFGs at codegen time. Whether part 2 replicates the PR#2206
        behavior or fixes it is an open decision recorded there.
        """
        import numpy as np
        from dace import data as dt, symbolic

        arrays = self._sdfg.arrays
        syms = {k: v for k, v in kwargs.items() if k not in arrays}
        syms.update(self._sdfg.constants)

        return_arrays = []
        for name in sorted(n for n in arrays.keys() if n.startswith('__return')):
            desc = arrays[name]
            if name in kwargs:
                return_arrays.append(kwargs[name])
                continue
            if not isinstance(desc, dt.Array):
                raise NotImplementedError(f'Nanobind interface: return value "{name}" of type '
                                          f'{type(desc).__name__} is not supported yet.')
            shape = tuple(int(symbolic.evaluate(s, syms)) for s in desc.shape)
            dtype = desc.dtype.as_numpy_dtype()
            total_size = int(symbolic.evaluate(desc.total_size, syms))
            strides = tuple(int(symbolic.evaluate(s, syms)) * desc.dtype.bytes for s in desc.strides)
            arr = np.ndarray(shape, dtype, buffer=np.zeros(total_size, dtype), strides=strides)
            kwargs[name] = arr
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

    def get_workspace_sizes(self):
        """Returns the external-memory sizes per storage type.

        The handle keys the sizes by storage-type *name* (stable across changes
        to the enum values); the conversion back to the ``StorageType`` enum
        happens here, since the Python enum is not exposed to C++.
        """
        from dace import dtypes
        return {getattr(dtypes.StorageType, k): v for k, v in self._handle.get_workspace_sizes().items()}

    def set_workspace(self, storage, workspace):
        """Sets the workspace for the given storage type to the given buffer."""
        from dace import dtypes
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
        import ctypes
        lib = ctypes.CDLL(self._module.__file__)
        try:
            func = getattr(lib, name)
        except AttributeError:
            return None
        if restype is not None:
            func.restype = restype
        func.__compiled_sdfg__ = self
        return func
