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
        """Allocates the ``__return*`` arrays (fresh each call) and adds them to ``kwargs``."""
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

    def initialize(self, **kwargs):
        self._handle.initialize(**kwargs)

    def finalize(self):
        self._handle.finalize()
