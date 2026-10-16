# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Contains functionality to compile and invoke Python-generated SDFG code."""

import builtins
import linecache
import re
import types
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dace.codegen.codeobject import CodeObject

#: A return-value array is named exactly ``__return`` (single value) or
#: ``__return_<int>`` (tuple element).  Transients that merely share the
#: ``__return`` prefix -- e.g. the ``__return_tile`` / ``__return_tile_out``
#: buffers the tile vectorizer creates when the program's return value is
#: written through a cuTile kernel -- are NOT return values and must be
#: excluded from return marshaling.
_RETURN_ARRAY_RE = re.compile(r'^__return(_[0-9]+)?$')


def _is_return_array_name(name: str) -> bool:
    """Whether ``name`` is a genuine SDFG return-value array name.

    :param name: A data-descriptor name from ``sdfg.arrays``.
    :returns: ``True`` for ``__return`` and ``__return_<int>`` only.
    """
    return _RETURN_ARRAY_RE.match(name) is not None


def _build_aux_module(co: 'CodeObject') -> types.ModuleType:
    """Build a Python module from a CodeObject without registering it in sys.modules."""
    pseudo_filename = f'<dace_generated_module_{co.name}_{id(co)}>'
    code_lines = co.code.splitlines(True)
    linecache.cache[pseudo_filename] = (len(co.code), None, code_lines, pseudo_filename)
    mod = types.ModuleType(co.name)
    mod.__file__ = pseudo_filename
    compiled = compile(co.code, pseudo_filename, 'exec')
    exec(compiled, mod.__dict__)
    return mod


def _make_import_hook(aux_modules: Dict[str, types.ModuleType]):
    """Build a replacement for ``__import__`` that resolves ``aux_modules`` first.

    Top-level absolute imports whose name matches an auxiliary module return
    that module directly. All other imports fall through to the real importer,
    so stdlib and third-party packages behave normally and ``sys.modules`` is
    never touched for our generated modules.
    """
    real_import = builtins.__import__

    def __import__(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name in aux_modules:
            return aux_modules[name]
        return real_import(name, globals, locals, fromlist, level)

    return __import__


class PythonCompiledSDFG:
    """
    A compiled SDFG object for the Python backend.

    Unlike ``CompiledSDFG``, which loads a shared library via ctypes,
    this class executes the generated Python source code directly using
    ``exec()`` and extracts the callable function from the resulting
    namespace.
    """

    def __init__(self, sdfg, code: str, *,
                 aux_modules: Optional[Dict[str, types.ModuleType]] = None):
        from dace.sdfg import SDFG
        self._sdfg: SDFG = sdfg
        self._code: str = code
        self._initialized = False
        self._finalized = False
        self._aux_modules: Dict[str, types.ModuleType] = aux_modules or {}

        # Register generated source under a pseudo filename so inspect can
        # retrieve source lines for nested/generated functions.
        # This is necessary for features like CuTile that generate kernels and
        # rely on inspect.getsource() to retrieve their source code for compilation.
        pseudo_filename = (
            f'<dace_generated_python_sdfg_{sdfg.name}_{id(self)}>'
        )
        code_lines = self._code.splitlines(True)
        linecache.cache[pseudo_filename] = (len(self._code), None, code_lines,
                            pseudo_filename)

        # Build a custom builtins so that imports of auxiliary modules resolve
        # locally instead of polluting sys.modules.
        custom_builtins = dict(builtins.__dict__)
        custom_builtins['__import__'] = _make_import_hook(self._aux_modules)

        # Execute the generated code in an isolated namespace
        self._namespace: Dict[str, Any] = {
            '__builtins__': custom_builtins,
            '__file__': pseudo_filename,
        }
        compiled_code = compile(self._code, pseudo_filename, 'exec')
        exec(compiled_code, self._namespace)

        # Extract the generated function (name matches the SDFG name)
        func_name = sdfg.name
        if func_name not in self._namespace:
            raise RuntimeError(
                f"Generated Python code does not define function '{func_name}'"
            )
        self._func = self._namespace[func_name]
        self._check_assumptions = self._namespace.get(f'__dace_check_assumptions_{func_name}')
        self._init = self._namespace.get(f'__dace_init_{func_name}')
        self._exit = self._namespace.get(f'__dace_exit_{func_name}')

        # --- Return-value metadata ---
        # Single return (__return) vs tuple return (__return_0, __return_1, ...)
        self._is_single_value_ret: bool = False
        if '__return' in self._sdfg.arrays:
            assert not any(
                _is_return_array_name(aname) and aname != '__return'
                for aname in self._sdfg.arrays.keys()
            )
            self._is_single_value_ret = True

        # Whether the SDFG has any genuine return arrays (cached for fast
        # __call__); tile transients sharing the __return prefix don't count.
        self._has_returns: bool = any(
            _is_return_array_name(aname) for aname in self._sdfg.arrays
        )

        # Argument name list for positional arg conversion (includes __return*).
        # Computed lazily by _get_argnames() because arglist() can fail on
        # nested SDFGs with undeclared runtime symbols.
        self._argnames: Optional[List[str]] = None

        # Sorted __return* array names (lazy cache).
        self._return_names: Optional[List[str]] = None

        # Create cached cfunc wrapper for profiler compatibility
        func = self._func
        def _cfunc_wrapper(_handle, *args, **kwargs):
            return func(*args, **kwargs)
        self._cfunc_cached = _cfunc_wrapper

        # Profiler compatibility: these attributes mirror the C++ CompiledSDFG
        # interface so that CompiledSDFGProfiler (and similar tools like
        # npbench) can use the same code path for both backends.
        self.do_not_execute: bool = False
        self._libhandle = None

    @property
    def sdfg(self):
        return self._sdfg

    @property
    def code(self) -> str:
        return self._code

    @property
    def _cfunc(self):
        """Return a callable compatible with the C++ CompiledSDFG interface.

        The C++ backend exposes ``_cfunc`` as a ctypes function pointer whose
        first argument is ``_libhandle`` (a ``ctypes.c_void_p``).  Profiling
        tools such as ``CompiledSDFGProfiler`` call
        ``compiled_sdfg._cfunc(compiled_sdfg._libhandle, *args)`` to bypass
        the normal ``__call__`` path.

        This property returns a cached thin wrapper around ``self._func``
        that accepts (and ignores) the leading handle argument so the same
        calling convention works for the Python backend.
        """
        return self._cfunc_cached

    def _get_argnames(self) -> List[str]:
        """Return cached arglist keys, computing them lazily.

        Returns ``sdfg.arglist().keys()`` which is equivalent to
        ``CompiledSDFG.argnames`` for positional-arg mapping.
        Computed lazily because ``arglist()`` can raise on nested SDFGs
        with undeclared runtime symbols during ``__init__``.
        """
        if self._argnames is None:
            self._argnames = list(self._sdfg.arglist().keys())
        return self._argnames

    def initialize(self, *args, **kwargs):
        if self._initialized:
            return
        if self._init is not None:
            self._init(*args, **kwargs)
        self._initialized = True
        self._finalized = False

    def finalize(self):
        if self._finalized:
            return
        if self._exit is not None:
            self._exit()
        # Clear persistent transients so that the next initialize() cycle
        # starts fresh, matching the semantics of a full re-initialization.
        # TODO: This needs to be part of the generated code, not here
        if '__dace_persistent_transients' in self._namespace:
            self._namespace['__dace_persistent_transients'].clear()
        self._initialized = False
        self._finalized = True

    def _bind_positional(self, args: tuple, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Convert positional arguments to keyword arguments.

        :param args: Positional argument values, mapped onto ``arglist()`` order.
        :param kwargs: Keyword arguments (not modified).
        :returns: A new kwargs dict including the positional bindings.
        :raises KeyError: If the SDFG takes no arguments but ``args`` is nonempty.
        :raises TypeError: If more positional arguments are passed than the
            SDFG accepts.
        :raises ValueError: If an argument is passed both ways.
        """
        if not args:
            return dict(kwargs)
        argnames = self._get_argnames()
        if not argnames:
            raise KeyError(
                "Passed positional arguments to an SDFG that does "
                "not accept them.")
        if len(args) > len(argnames):
            raise TypeError(f"Passed {len(args)} positional arguments to an SDFG that "
                            f"accepts at most {len(argnames)} ({argnames}).")
        positional = dict(zip(argnames, args))
        if not positional.keys().isdisjoint(kwargs.keys()):
            raise ValueError(
                "Arguments passed as both positional and keyword: "
                f"{set(positional) & set(kwargs)}")
        merged = dict(kwargs)
        merged.update(positional)
        return merged

    def _marshal_arguments(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Coerce runtime arguments to the Python-backend calling convention.

        Scalar data arguments are bound as 0-d numpy buffers of the declared
        dtype (generated code reads them as ``x[()]`` and writes ``x[...]``),
        and numpy scalars passed for symbols are converted to native Python
        values (kernel launch grids and ``range()`` bounds require them).

        :param kwargs: Keyword arguments as passed by the caller.
        :returns: A new dict with marshalled values.
        :raises TypeError: If a non-scalar array is passed for a Scalar
            argument, if an ndarray argument's dtype does not match the
            declared dtype, or if a scalar value cannot be losslessly
            converted to the declared dtype.
        """
        from dace import data

        marshalled = dict(kwargs)
        for name, value in kwargs.items():
            desc = self._sdfg.arrays.get(name)
            if isinstance(desc, data.Scalar):
                nptype = desc.dtype.as_numpy_dtype()
                if isinstance(value, np.ndarray):
                    if value.size != 1:
                        raise TypeError(f'Argument {name!r}: expected a scalar, '
                                        f'got an array of shape {value.shape}')
                    # A dtype-mismatched buffer would have to be copied
                    # (np.asarray), so in-SDFG writes would land in the copy
                    # and the caller's buffer would keep the stale value.
                    # Mirror the C backend's array behavior and reject it.
                    if value.dtype != nptype:
                        raise TypeError(f'Argument {name!r}: passed an ndarray of dtype '
                                        f'{value.dtype}, expected {np.dtype(nptype)}')
                    # Same-dtype reshape is a view: in-SDFG writes propagate.
                    marshalled[name] = value.reshape(())
                elif isinstance(value, (bool, int, float, complex, np.generic)):
                    coerced = np.asarray(value, dtype=nptype)
                    # Coercing a plain value to an integer/bool dtype must be
                    # exact (e.g. 3.9 -> int32 would silently truncate to 3,
                    # and out-of-range numpy ints silently wrap).
                    if coerced.dtype.kind in 'iub' and coerced[()] != value:
                        raise TypeError(f'Argument {name!r}: value {value!r} cannot be '
                                        f'losslessly converted to declared dtype '
                                        f'{np.dtype(nptype)}')
                    marshalled[name] = coerced
                # Anything else (e.g. a cupy scalar buffer) passes through.
            elif desc is None:
                # Symbol values: native Python scalars only.
                if isinstance(value, np.generic):
                    marshalled[name] = value.item()
                elif isinstance(value, np.ndarray) and value.ndim == 0:
                    marshalled[name] = value.item()
        return marshalled

    def _get_return_names(self) -> List[str]:
        """Sorted names of ``__return*`` arrays in the SDFG."""
        if self._return_names is None:
            self._return_names = sorted(
                n for n in self._sdfg.arrays if _is_return_array_name(n)
            )
        return self._return_names

    def _allocate_return_array(self, name: str, syms: Dict[str, Any]) -> np.ndarray:
        """Allocate a return-value array, evaluating symbolic shapes.

        :param name: The ``__return*`` array name in the SDFG.
        :param syms: Symbol-to-value mapping for shape evaluation.
        :returns: A newly allocated numpy (or cupy) array.
        """
        from dace import dtypes, symbolic

        desc = self._sdfg.arrays[name]
        if desc.transient:
            raise ValueError(
                f'Used the special array name "{name}" as transient.')
        shape = tuple(int(symbolic.evaluate(s, syms)) for s in desc.shape)
        dtype = desc.dtype.as_numpy_dtype()
        if desc.storage is dtypes.StorageType.GPU_Global:
            try:
                import cupy
                return cupy.empty(shape, dtype=dtype)
            except (ImportError, ModuleNotFoundError):
                raise NotImplementedError(
                    'GPU return values require cupy to be installed')
        return np.empty(shape, dtype=dtype)

    def __call__(self, *args, **kwargs):
        # Fast path: no return values
        if not self._has_returns:
            if args:
                try:
                    self._get_argnames()
                except Exception:
                    # arglist() can fail on nested SDFGs with undeclared
                    # runtime symbols; fall back to raw positional args
                    # (keyword arguments are still marshalled below).
                    pass
                else:
                    # Only the argname resolution above is fallible; binding
                    # errors (duplicate/excess arguments) must propagate.
                    kwargs = self._bind_positional(args, kwargs)
                    args = ()
            kwargs = self._marshal_arguments(kwargs)
            self.initialize(*args, **kwargs)
            if self.do_not_execute:
                return None
            return self._func(*args, **kwargs)

        # Convert positional args to keyword args
        kwargs = self._bind_positional(args, kwargs)
        kwargs = self._marshal_arguments(kwargs)

        if self._check_assumptions is not None:
            self._check_assumptions(**kwargs)

        # Resolve symbols for shape evaluation
        syms = {k: v for k, v in kwargs.items()
                if k not in self._sdfg.arrays}
        syms.update(self._sdfg.constants)

        # Allocate (or reuse user-provided) return arrays
        return_arrays = []
        for name in self._get_return_names():
            if name not in kwargs:
                kwargs[name] = self._allocate_return_array(name, syms)
            return_arrays.append(kwargs[name])

        self.initialize(**kwargs)
        if not self.do_not_execute:
            self._func(**kwargs)

        # Marshal return values
        if self._is_single_value_ret:
            return return_arrays[0]
        return tuple(return_arrays)

    def __del__(self):
        try:
            self.finalize()
        except AttributeError: # can happen if __init__ raised an exception
            pass


def compile_python_sdfg(sdfg, code_objects: 'list[CodeObject]') -> PythonCompiledSDFG:
    """
    Compile a Python-backend SDFG from the generated code objects.

    The first code object is the frame (main SDFG function). Subsequent
    linkable code objects are built as in-memory Python modules that are
    resolvable by import statements in the frame code via a private
    ``__import__`` hook. ``sys.modules`` is never modified. Non-linkable
    objects (e.g. SampleMain) are skipped.

    :param sdfg: The SDFG that was compiled.
    :param code_objects: List of CodeObject instances from code generation.
    :return: A callable PythonCompiledSDFG.
    """
    if not code_objects:
        raise RuntimeError("No code objects generated for Python backend")

    frame_co = code_objects[0]

    aux_modules: Dict[str, types.ModuleType] = {}
    for co in code_objects[1:]:
        if not co.linkable:
            continue
        aux_modules[co.name] = _build_aux_module(co)

    return PythonCompiledSDFG(sdfg, frame_co.code, aux_modules=aux_modules)
