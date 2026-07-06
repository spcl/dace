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

        Unlike ``CompiledSDFG._initialize_return_values()`` there is no
        pyobject handling here (including the PR#2206 bug-compatible decay of
        pyobject arrays to a single value): pyobject is deferred to part 2 of
        the nanobind port, and the bindings generator already rejects such
        SDFGs at codegen time. Whether part 2 replicates the PR#2206 behavior
        or fixes it is an open decision recorded there.
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
        # The name mapping is needed here too, but for a different reason
        # than in __call__: the C++ initialize's positional parameters are
        # the *init symbols* only (e.g. ``initialize(int N, **kwargs)``),
        # while callers pass positionals in user-facing arg_names order
        # (e.g. ``initialize(a, N=20)``). A plain pass-through would match
        # ``a`` positionally against ``N``.
        if args:
            for name, value in zip(self._arg_names, args):
                kwargs.setdefault(name, value)
        self._handle.initialize(**kwargs)

    def finalize(self):
        self._handle.finalize()

    def get_workspace_sizes(self):
        """Returns the external-memory sizes per storage type (see ``CompiledSDFG``).

        The handle speaks raw storage-type values; the conversion to the
        ``StorageType`` enum happens here, since the Python enum is not
        exposed to C++.
        """
        from dace import dtypes
        return {dtypes.StorageType(k): v for k, v in self._handle.get_workspace_sizes().items()}

    def set_workspace(self, storage, workspace):
        """Sets the workspace for the given storage type to the given buffer (see ``CompiledSDFG``)."""
        from dace import dtypes
        value = storage.value if isinstance(storage, dtypes.StorageType) else int(storage)
        self._handle.set_workspace(value, workspace)

    def state_fields(self):
        """Names of the pointer fields in the state struct.

        Baked into the module at code-generation time - no parsing of the
        generated sources, unlike the ctypes path.
        """
        return list(self._handle.state_fields())

    def get_state_struct(self) -> int:
        """Returns the *raw pointer* to the state struct as an integer.

        Unlike ``CompiledSDFG.get_state_struct()`` this does not reconstruct a
        ``ctypes.Structure`` view; combine it with ``state_fields()`` if
        needed. Callers must know exactly what they are doing - and whatever
        they are doing with this pointer is most likely wrong.
        """
        return self._handle.state_pointer

    def get_exported_function(self, name: str, restype=None):
        """Returns an arbitrary exported symbol as a callable, or None if absent.

        Resolved with ``ctypes.CDLL`` on the already-imported module file
        (which returns the same library handle); the wrapper is attached to
        the returned function as ``__compiled_sdfg__`` to keep the module
        alive.
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
