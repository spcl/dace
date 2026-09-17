# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Generator for the nanobind bindings of a compiled SDFG.

Emits one C++ source file that is compiled into the program library itself
(the nanobind extension module *is* the shared library). The module exposes a
``CompiledSDFGHandle`` class wrapping one state handle, and a
``make_compiled_sdfg()`` builder; several handles may share one loaded module.

The bound methods use typed positional/keyword parameters, so argument
matching and casting happen in nanobind's dispatcher rather than in
hand-written kwargs lookups. A trailing ``nb::kwargs`` parameter absorbs
extra keyword arguments, which the old ctypes interface allowed.
"""

from typing import Dict, List, Optional, Set, Tuple

import numpy
import sympy

from dace import data as dt, dtypes, symbolic
from dace.codegen.common import sym2cpp
from dace.config import Config

# ml_dtypes-backed low-precision types: numpy cannot export their arrays via
# DLPack or the buffer protocol, so nanobind cannot ingest them; they are
# outside the interface's scope (see argument_unsupported).
_LOWP_TYPES = (dtypes.bfloat16, dtypes.float8_e4m3fn, dtypes.float8_e5m2)

# dtype_traits specialization advertising dace::float16 (= dace::half) as a
# 16-bit DLPack float, so nb::ndarray<dace::float16, ...> accepts a numpy/cupy
# float16 array. nanobind's own detection uses std::is_floating_point, false for
# the half struct. Emitted only when a float16 ndarray argument exists.
_FLOAT16_TRAITS = '''
namespace nanobind { namespace detail {
template <> struct dtype_traits<dace::float16> {
    static constexpr dlpack::dtype value{
        (uint8_t) dlpack::dtype_code::Float, 16, 1
    };
    static constexpr auto name = const_name("float16");
};
}}
'''

# Boolean scalar arguments need their own caster: nanobind's bool caster only
# ever accepts an exact Python bool, and the integer-caster detour this
# interface used to take (binding the parameter as uint8_t) broke when
# nanobind 2.14 narrowed integer conversion to the __index__ protocol just as
# numpy removed __index__ from numpy.bool_ (int() still works, index() does
# not). The caster restores the intended acceptance set - Python bool,
# numpy.bool_, and integer-like values (Python int, numpy integer scalars) -
# and nothing more: floats have no __index__ and keep being rejected. Kept as
# a caster rather than per-argument setup code so it slots uniformly into
# call() and initialize().
_DACE_BOOL_CASTER = '''
// The implicit bool conversion is load-bearing: a bool SYMBOL's binding
// parameter is passed by its raw name into init_impl (and the workspace
// methods), where the extern "C" signature takes a plain bool.
struct dace_bool {
    uint8_t value;
    operator bool() const { return value != 0; }
};
namespace nanobind { namespace detail {
template <> struct type_caster<dace_bool> {
    NB_TYPE_CASTER(dace_bool, const_name("bool"))
    bool from_python(handle src, uint8_t flags, cleanup_list *) noexcept {
        PyObject *o = src.ptr();
        if (o == Py_True) { value.value = 1; return true; }
        if (o == Py_False) { value.value = 0; return true; }
        int64_t i;
        if (load_i64(o, flags, &i)) { value.value = (uint8_t) (i != 0); return true; }
        // numpy.bool_ answers to neither of the above; accept it by type. The
        // type object resolves lazily (numpy may legitimately be absent) and
        // is deliberately leaked - it lives as long as numpy itself.
        static PyObject *np_bool_type = []() -> PyObject * {
            PyObject *np = PyImport_ImportModule("numpy");
            if (!np) { PyErr_Clear(); return nullptr; }
            PyObject *t = PyObject_GetAttrString(np, "bool_");
            Py_DECREF(np);
            if (!t) PyErr_Clear();
            return t;
        }();
        if (np_bool_type && PyObject_TypeCheck(o, (PyTypeObject *) np_bool_type)) {
            int r = PyObject_IsTrue(o);
            if (r < 0) { PyErr_Clear(); return false; }
            value.value = (uint8_t) r;
            return true;
        }
        return false;
    }
    static handle from_cpp(dace_bool src, rv_policy, cleanup_list *) noexcept {
        return handle(src.value ? Py_True : Py_False).inc_ref();
    }
};
}}
'''


def argument_unsupported(name: str, desc: dt.Data) -> Optional[str]:
    """The reason argument ``(name, desc)`` is outside the nanobind interface's scope, or ``None``.

    This predicate is the single source of truth for the interface's data
    scope: :func:`_argument_binding` raises exactly when it returns a reason,
    and :func:`unsupported_reason` (the ``compiler.interface=auto`` detector in
    ``dace.codegen.compiler``) walks the same predicate over the arglist - so
    the codegen refusals and the automatic ctypes fallback cannot diverge.

    In scope: scalars and arrays of primitive data (integers, floats incl.
    float16 arrays, complex, bool, vectors of them), string scalars, nullable
    and GPU arrays, ``pyobject`` scalars and arrays, and array return values.
    Out of scope (kept on the ctypes interface): ``Structure``,
    ``ContainerArray``, record-dtype (``dtypes.struct``) arrays, callbacks,
    ml_dtypes-backed low-precision data, float16 and vector scalars, string
    arrays, non-array returns and returns with a nonzero offset.
    """
    hint = 'use the ctypes interface (compiler.interface=ctypes, or the default "auto").'
    if isinstance(desc, dt.Structure):
        return f'argument "{name}" is a Structure, which is out of scope; {hint}'
    if isinstance(desc, dt.ContainerArray):
        return f'argument "{name}" is a ContainerArray, which is out of scope; {hint}'
    if isinstance(desc.dtype, dtypes.callback):
        return f'callback argument "{name}" is out of scope (needs the ctypes trampoline machinery); {hint}'
    if desc.dtype.base_type in _LOWP_TYPES:
        return (f'argument "{name}" of low-precision type {desc.dtype} is out of scope '
                f'(nanobind cannot ingest ml_dtypes-backed data); {hint}')
    if name.startswith('__return'):
        if not isinstance(desc, dt.Array):
            return (f'return value "{name}" of type {type(desc).__name__} is not supported; '
                    f'returns are arrays only.')
        if isinstance(desc.dtype, dtypes.struct):
            return f'return value "{name}" is a record-dtype (dtypes.struct) array, which is out of scope; {hint}'
        if any(str(o) != '0' for o in desc.offset):
            return (f'return value "{name}" has a nonzero offset; in-binding allocation '
                    f'assumes offset 0; {hint}')
        return None
    if isinstance(desc, dt.Array):
        if isinstance(desc.dtype, dtypes.struct):
            return f'argument "{name}" is a record-dtype (dtypes.struct) array, which is out of scope; {hint}'
        if desc.dtype.base_type == dtypes.string:
            return f'argument "{name}" is a string array (char-buffer form), which is out of scope; {hint}'
        return None
    if isinstance(desc, dt.Scalar):
        if isinstance(desc.dtype, dtypes.pyobject) or desc.dtype == dtypes.string:
            return None
        if desc.dtype.base_type == dtypes.float16:
            return (f'float16 scalar argument "{name}" is out of scope '
                    f'(dace::half would need a value type-caster); {hint}')
        if isinstance(desc.dtype, dtypes.vector):
            return f'vector scalar argument "{name}" is out of scope; {hint}'
        return None
    return f'argument "{name}" of type {type(desc).__name__} is out of scope; {hint}'


def unsupported_reason(sdfg) -> Optional[str]:
    """The reason the nanobind interface cannot compile ``sdfg``, or ``None`` if it can.

    Walks :func:`argument_unsupported` over the full ``sdfg.arglist()`` - the
    exact descriptors the bindings generator binds, return values included.
    ``compiler.resolve_compiler_interface`` consults this for
    ``compiler.interface=auto``; under an explicit ``nanobind`` setting the
    same predicate raises at code generation instead.
    """
    for name, desc in sdfg.arglist().items():
        reason = argument_unsupported(name, desc)
        if reason is not None:
            return reason
    return None


def _symbol_fallbacks(arglist: Dict[str, dt.Data], arg_names: List[str],
                      symbols: Dict[str, dtypes.typeclass]) -> Tuple[Set[str], Dict[str, str]]:
    """Determines the optional "artifact" symbols and their C++ shape-inference fallbacks.

    Numeric **SDFG symbols** (``symbols``) that are not in the
    user-facing ``arg_names`` (array sizes and the like) become optional
    parameters; when omitted, their value is derived from an array argument's
    shape or strides. Data scalars are never omittable. The expression is
    inverted with sympy once, at code-generation time, so the run-time
    fallback is plain arithmetic on ``<array>.shape(<dim>)`` /
    ``<array>.stride(<dim>)``.

    Inference sources are user-facing plain arrays only: nullable
    (``optional=True``), struct, container and vector arrays are excluded
    (their run-time shape is either unavailable or does not equal the
    descriptor shape). Nullability is opt-in, so every source binds as a
    plain ``nb::ndarray`` whose shape is always readable.

    A dim expression may reference further symbols besides the target, as
    long as each of them is itself listed in ``arg_names``: such a symbol is
    explicitly passed (a plain required parameter, in scope at the setup
    statement), so ``a`` in ``A[a + b]`` is inferable as ``A.shape(0) - b``
    once ``b`` is promised in ``arg_names``. Inferred (omittable) symbols are
    never expression terms - no ordering fixed-point is needed.

    :param arglist: The full ``sdfg.arglist()``, in C signature order.
    :param arg_names: The user-facing positional argument names.
    :param symbols: The SDFG's symbol registry (``sdfg.symbols``), mapping
                    each symbol name to its typeclass.
    :return: A pair ``(optional_symbols, fallbacks)``: the names that bind as
             optional parameters, and the fallback C++ expression for those
             that have an inference source. An optional symbol absent from
             ``fallbacks`` has no source - omitting it raises at run time.
    """

    def _plain_numeric_symbol(name) -> bool:
        # An SDFG symbol of a plain integer or floating typeclass. Subclassed
        # typeclasses (strings, callbacks, pyobjects, vectors, ...) fall
        # outside by construction, and so do data scalars (not in the symbol
        # registry). The type comes from the registry - a symbol has no data
        # descriptor of its own; its arglist entry is only a synthesized
        # Scalar wrapper.
        if name not in symbols:
            return False
        dtype = symbols[name]
        return (type(dtype) is dtypes.typeclass
                and (numpy.issubdtype(dtype.type, numpy.integer) or numpy.issubdtype(dtype.type, numpy.floating)))

    arg_names_set = set(arg_names)
    candidates = {name for name in arglist if name not in arg_names_set and _plain_numeric_symbol(name)}
    if not candidates:
        return set(), {}

    # Symbols the caller promised to pass: every symbol listed in arg_names
    # binds as a plain required parameter under its own name, so a dim
    # expression referencing one stays evaluable in the generated fallback
    # (arg_names arrives pre-filtered to arglist members).
    explicit_symbols = {name for name in arg_names_set if _plain_numeric_symbol(name)}

    # The shape and strides of these array arguments can be used to infer
    # symbol values. Struct/container/low-precision arrays cannot occur here
    # (out of scope, refused before this runs); vector arrays are excluded
    # because their run-time shape differs from the descriptor shape.
    sources = [(name, desc) for name, desc in arglist.items()
               if name in arg_names_set and isinstance(desc, dt.Array) and not isinstance(desc.dtype, dtypes.vector)
               and desc.optional is not True and not name.startswith('__return')]

    dace_infer_src = f'__dace_infer_src_{id(arglist)}'
    placeholder = sympy.Symbol(dace_infer_src)

    def _invert(sym_name: str, ctype: str) -> Optional[str]:
        for aname, desc in sources:
            # Strides are sources like shapes: DaCe descriptor strides and DLPack (nb::ndarray::stride)
            # both count elements. Unlike NumPy's Python level `.strides`.
            for accessor, dims in (('shape', desc.shape), ('stride', desc.strides)):
                for i, dim in enumerate(dims):
                    dim_symbols = symbolic.symlist(dim)
                    free = set(dim_symbols.keys())
                    # Invertible when the target occurs in the dim and every
                    # other free symbol is explicitly passed (in arg_names);
                    # those render by name into the fallback expression.
                    if sym_name not in free or (free - {sym_name}) - explicit_symbols:
                        continue
                    try:
                        solutions = sympy.solve(dim - placeholder, dim_symbols[sym_name])
                    except Exception:
                        continue
                    if len(solutions) != 1:
                        continue
                    expr = sym2cpp(solutions[0]).replace(dace_infer_src, f'{aname}.{accessor}({i})')
                    return f'static_cast<{ctype}>({expr})'
        return None

    fallbacks = {}
    for sym_name in candidates:
        expr = _invert(sym_name, arglist[sym_name].dtype.ctype)
        if expr is not None:
            fallbacks[sym_name] = expr
    return candidates, fallbacks


def _ndarray_device(desc) -> str:
    """DLPack device annotation for an array argument's storage.

    Only ``GPU_Global`` is device memory from the caller's perspective;
    everything else (including ``CPU_Pinned``, which numpy reports as a CPU
    DLPack device) binds as host memory. The configured backend is consulted
    at codegen time; only ``'auto'`` falls back to ``get_gpu_backend()``,
    whose hardware probe is lru-cached (and so blind to config changes) -
    and it runs only when a GPU array actually exists.
    """
    if desc.storage == dtypes.StorageType.GPU_Global:
        backend = Config.get('compiler', 'cuda', 'backend')
        if backend in (None, '', 'auto'):
            from dace.codegen import common
            backend = common.get_gpu_backend()
        return 'nb::device::cuda' if backend == 'cuda' else 'nb::device::rocm'
    return 'nb::device::cpu'


def _has_gpu_code(sdfg) -> bool:
    """Same detection as the ctypes ``CompiledSDFG``, evaluated at codegen time."""
    for _, _, desc in sdfg.arrays_recursive():
        if desc.storage in dtypes.GPU_STORAGES:
            return True
    for node, _ in sdfg.all_nodes_recursive():
        if getattr(node, 'schedule', False) in dtypes.GPU_SCHEDULES:
            return True
    return False


def _argument_binding(arglist: Dict[str, dt.Data],
                      binding_order: List[str],
                      optional_symbols: Set[str],
                      symbol_fallbacks: Dict[str, str],
                      sdfg=None) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Generates the per-argument pieces of the bound ``call()``/``initialize()`` methods.

    Arrays are taken without implicit conversion: nanobind would otherwise
    silently pass a converted *copy*, breaking DaCe's by-reference argument
    semantics. Scalars keep nanobind's overflow-checked conversion.

    :param arglist: The arguments to bind (``sdfg.arglist()`` or the init
                    subset), in C signature order.
    :param binding_order: The order of the bound parameters (the user-facing
                          positional order); pass ``list(arglist.keys())`` for
                          the natural order.
    :param optional_symbols: Names of the arguments that may be omitted (see
                             :func:`_symbol_fallbacks`); pass ``set()`` for
                             none.
    :param symbol_fallbacks: The shape-inference fallback expression per
                             optional symbol that has one; omitting a symbol
                             absent here raises at run time. Pass ``{}`` for
                             none.
    :return: A 4-tuple of C++ fragment lists:
             the parameter declarations (in ``binding_order``),
             the program-call argument expressions (in ``arglist`` order, the
             C signature order),
             the ``nb::arg`` annotations (in ``binding_order``), and
             the setup statements that must run under the GIL before the
             kernel call: first the must-pass symbol extractions (omittable-
             bound symbols without an inference source - never legitimately
             None), then the rest (inferable-symbol deductions, struct buffer
             acquisition, callback pointer recovery) in ``arglist`` order.
             Empty without struct/callback/omittable-symbol arguments.
    """
    # params and nb::args are keyed by name so they can be reordered to
    # binding_order at the end; call_args are collected directly in arglist
    # order, which is the C signature order the program call needs.
    params_by_name = {}
    nb_args_by_name = {}
    call_args = []

    # C++ statements that must run under the GIL before the kernel call - used
    # to extract raw pointers from `nb::object` struct arguments via the Python
    # buffer protocol, and to declare the typed locals of the omittable
    # symbols. The must-pass symbols (omittable-bound but with no inference
    # source, so never legitimately None) are collected separately and emitted
    # FIRST: their values are then established at the top of call(), so a
    # deduction statement may reference an explicitly-passed symbol.
    setup_stmts = []
    must_pass_setup = []
    # Return-array setup runs LAST: allocation sizes reference symbols, so the
    # must-pass extractions and inference deductions must already have run.
    return_setup = []

    strict_scalar = Config.get_bool('compiler', 'nanobind_strict_scalar_cast')
    # Whether a caller-provided return buffer is accepted is decided AT CODE
    # GENERATION TIME and baked into the module: the binding never consults the
    # config at run time (changing the option requires a recompile).
    allow_return_override = Config.get_bool('compiler', 'nanobind_allow_return_override')

    for name, desc in arglist.items():
        # The scope predicate is the one refusal site: everything it flags is
        # kept on the ctypes interface (and `compiler.interface=auto` routes
        # such SDFGs there instead of ever reaching this raise).
        reason = argument_unsupported(name, desc)
        if reason is not None:
            raise NotImplementedError(f'Nanobind interface: {reason}')

        # A pyobject is an opaque PyObject* (`typedef void *pyobject` in
        # pyinterop.h). A SCALAR pyobject argument passes through: the
        # nb::object parameter holds a reference for the duration of the call
        # and the raw pointer is forwarded - reading `.ptr()` needs no GIL,
        # and the program must not retain the pointer beyond the call
        # (ctypes-interface parity, which passes a borrowed ctypes.py_object
        # the same way).
        # Returns fall through to the __return branch below, which allocates the object array
        # and decays it to the single contained object on the way out (ctypes parity).
        if isinstance(desc.dtype, dtypes.pyobject) and not name.startswith('__return'):
            if isinstance(desc, dt.Array):
                # An ARRAY of pyobjects, i.e. numpy dtype=object: the buffer is a flat run of
                # PyObject* slots. nb::ndarray cannot ingest it (DLPack refuses object arrays
                # outright), so the pointer comes from __array_interface__ exactly as for the
                # ml_dtypes-backed low-precision arrays below. The itemsize guard is
                # sizeof(void*); the kind letter is 'O'.
                #
                # Lifetime is the caller's array: it owns references to the contained objects
                # and the nb::object parameter keeps it alive across the call, so the slots
                # stay valid while the program dereferences them. This is what the ctypes
                # marshaller does too - it hands out pointers into the same buffer.
                setup_stmts.append(
                    f'nb::object {name}__ai = {name}.attr("__array_interface__");\n'
                    f'        const std::string {name}__ts = nb::cast<std::string>({name}__ai["typestr"]);\n'
                    f'        if ({name}__ts.size() < 2 || {name}__ts[1] != \'O\')\n'
                    f'            throw std::invalid_argument("SDFG argument error: argument \'{name}\': expected '
                    f'an object array (numpy dtype=object), got typestr \'" + {name}__ts + "\'.");\n'
                    f'        const std::uintptr_t {name}__ptr = '
                    f'nb::cast<std::uintptr_t>(nb::tuple({name}__ai["data"])[0]);')
                params_by_name[name] = f'nb::object {name}'
                call_args.append(f'reinterpret_cast<{desc.dtype.ctype} *>({name}__ptr)')
                nb_args_by_name[name] = f'nb::arg("{name}")'
                continue
            params_by_name[name] = f'nb::object {name}'
            call_args.append(f'reinterpret_cast<{desc.dtype.ctype}>({name}.ptr())')
            nb_args_by_name[name] = f'nb::arg("{name}")'
            continue

        if name.startswith('__return'):
            # Return arrays bind as DEFAULTED nb::object parameters and are
            # allocated INSIDE the binding when omitted (None) - after the
            # compiled symbol inference ran, so inferred symbols may size them.
            # The allocation deliberately goes through NumPy/CuPy via the
            # Python API (in setup, GIL held): ownership, alignment and dtype
            # semantics stay exactly those of the former Python-side
            # allocation. The binding returns the array object(s) at the end
            # of call().
            is_vector_ret = isinstance(desc.dtype, dtypes.vector)
            is_pyobj_ret = isinstance(desc.dtype, dtypes.pyobject)
            gpu = desc.storage == dtypes.StorageType.GPU_Global
            if is_vector_ret:
                # A vector array allocates as its base scalar with a trailing
                # veclen dimension - the exact layout numpy produced for the
                # former subarray-dtype allocation (numpy auto-expands
                # subarray dtypes into extra dimensions).
                nb_scalar = _ndarray_scalar_ctype(desc.dtype)
                device = _ndarray_device(desc)
                dtype_expr = f'__mod.attr("dtype")("{desc.dtype.base_type.as_numpy_dtype().name}")'
            elif is_pyobj_ret:
                # numpy dtype "object": a flat run of PyObject* slots, allocated zeroed (i.e.
                # None-filled) exactly as the ctypes allocator does.
                nb_scalar = device = None
                dtype_expr = '__mod.attr("dtype")("object")'
            else:
                nb_scalar = _ndarray_scalar_ctype(desc.dtype)
                device = _ndarray_device(desc)
                dtype_expr = f'__mod.attr("dtype")("{desc.dtype.as_numpy_dtype().name}")'
            # Only scalar constants can appear in shape/stride/size
            # expressions; array-valued compile-time constants must not reach
            # sympy's subs (sympify rejects numpy arrays).
            constants = {
                k: v
                for k, v in (sdfg.constants if sdfg is not None else {}).items()
                if isinstance(v, (int, float, complex, numpy.number, numpy.bool_))
            }

            def _cx(v):
                e = symbolic.pystr_to_symbolic(str(v))
                if constants:
                    e = e.subs(constants)
                return sym2cpp(e)

            # Vector arrays: base-scalar layout with a trailing veclen
            # dimension (contiguous inner); descriptor strides count VECTOR
            # elements, so their byte strides use the full vector width.
            shape_dims = list(desc.shape) + ([desc.dtype.veclen] if is_vector_ret else [])
            dims = ', '.join(_cx(s) for s in shape_dims)
            total = _cx(desc.total_size) + (f' * {desc.dtype.veclen}' if is_vector_ret else '')
            stride_terms = [f'({_cx(s)}) * {desc.dtype.bytes}' for s in desc.strides]
            if is_vector_ret:
                stride_terms.append(str(desc.dtype.base_type.bytes))
            strides_b = ', '.join(stride_terms)
            # Mirrors the former Python allocation: a zeroed flat buffer
            # wrapped with the descriptor's shape and (byte) strides.
            # cupy.ndarray takes memptr/strides positionally after dtype.
            # An object array must be allocated with numpy even for GPU storage - a pyobject
            # return is a host-side Python object, never device memory.
            if is_pyobj_ret:
                gpu = False
            if gpu:
                alloc = (f'[&]() {{ nb::object __mod = nb::module_::import_("cupy");\n'
                         f'            nb::object __dt = {dtype_expr};\n'
                         f'            return __mod.attr("ndarray")(nb::make_tuple({dims}), __dt, '
                         f'__mod.attr("zeros")({total}, __dt).attr("data"), nb::make_tuple({strides_b})); }}()')
            else:
                alloc = (f'[&]() {{ nb::object __mod = nb::module_::import_("numpy");\n'
                         f'            nb::object __dt = {dtype_expr};\n'
                         f'            return __mod.attr("ndarray")(nb::make_tuple({dims}), __dt, '
                         f'__mod.attr("zeros")({total}, __dt), 0, nb::make_tuple({strides_b})); }}()')

            if not allow_return_override:
                obtain = (f'if (!{name}.is_none())\n'
                          f'            throw std::invalid_argument("SDFG argument error: the implicit output '
                          f'\'{name}\' cannot be passed explicitly: this module was compiled with '
                          f'compiler.nanobind_allow_return_override=false; enable the option and recompile.");\n'
                          f'        nb::object {name}__obj = {alloc};')
            else:
                obtain = (f'nb::object {name}__obj = {name};\n'
                          f'        if ({name}__obj.is_none()) {{\n'
                          f'            {name}__obj = {alloc};\n'
                          f'        }}')
            if is_pyobj_ret:
                # DLPack refuses object arrays, so nb::ndarray can never ingest one: the
                # pointer comes from the array-interface dict instead. The return object is
                # already an nb::object here, so unlike the argument path there is no type
                # caster to bypass - only the nb::cast is replaced.
                iface = '__cuda_array_interface__' if gpu else '__array_interface__'
                extract = (f'nb::object {name}__ai = {name}__obj.attr("{iface}");\n'
                           f'        const std::uintptr_t {name}__ptr = '
                           f'nb::cast<std::uintptr_t>(nb::tuple({name}__ai["data"])[0]);')
                if allow_return_override:
                    # ``size`` rather than the nd view (which does not exist here); both NumPy
                    # and CuPy arrays expose it. Same contract as below: too small is refused,
                    # larger is legitimate.
                    extract += (f'\n        if (!{name}.is_none() && '
                                f'nb::cast<size_t>({name}__obj.attr("size")) < '
                                f'static_cast<size_t>({total}))\n'
                                f'            throw std::invalid_argument("SDFG argument error: return buffer '
                                f'\'{name}\' has a wrong shape (smaller than the symbol-derived return size).");')
                return_setup.append(f'{obtain}\n        {extract}')
                call_args.append(f'reinterpret_cast<{desc.dtype.ctype} *>({name}__ptr)')
            else:
                extract = f'auto {name}__nd = nb::cast<nb::ndarray<{nb_scalar}, {device}>>({name}__obj, false);'
                if allow_return_override:
                    # The program writes through the DESCRIPTOR's shape and
                    # strides regardless of the buffer's own: the guard is
                    # against out-of-bounds writes, so a too-SMALL buffer is
                    # rejected while a larger one is a legitimate pattern (a
                    # caller may hand in a longer buffer whose tail must stay
                    # untouched - see local_storage_test's test_uneven).
                    extract += (f'\n        if (!{name}.is_none() && {name}__nd.size() < '
                                f'static_cast<size_t>({total}))\n'
                                f'            throw std::invalid_argument("SDFG argument error: return buffer '
                                f'\'{name}\' has a wrong shape (smaller than the symbol-derived return size).");')
                return_setup.append(f'{obtain}\n        {extract}')
                call_args.append(f'reinterpret_cast<{desc.dtype.ctype} *>({name}__nd.data())')
            params_by_name[name] = f'nb::object {name}'
            nb_args_by_name[name] = f'nb::arg("{name}").none() = nb::none()'
            continue

        ctype = desc.dtype.ctype
        if isinstance(desc, dt.Scalar) and desc.dtype == dtypes.string:
            # A string scalar is a C string (int8_t*). std::optional<std::string>
            # lets None become a null pointer - ctypes-interface parity, which
            # marshals a None string argument to a NULL char* - and the owned
            # copy keeps the buffer valid across the GIL release. .none() is
            # required: nanobind rejects None by default.
            params_by_name[name] = f'std::optional<std::string> {name}'
            call_args.append(
                f'{name}.has_value() ? reinterpret_cast<{ctype}>(const_cast<char *>({name}->c_str())) : nullptr')
            nb_args_by_name[name] = f'nb::arg("{name}").none()'

        elif isinstance(desc, dt.Array):
            # The ndarray scalar type may differ from the cast target: a vector
            # dtype binds as its base scalar, but the kernel pointer stays dace::vec*.
            nb_scalar = _ndarray_scalar_ctype(desc.dtype)
            device = _ndarray_device(desc)
            if desc.optional:
                # Nullable array: None becomes a null pointer. .none() is
                # required - nanobind rejects None by default.
                params_by_name[name] = f'std::optional<nb::ndarray<{nb_scalar}, {device}>> {name}'
                call_args.append(f'{name}.has_value() ? reinterpret_cast<{ctype} *>({name}->data()) : nullptr')
                nb_args_by_name[name] = f'nb::arg("{name}").noconvert().none()'
            else:
                params_by_name[name] = f'nb::ndarray<{nb_scalar}, {device}> {name}'
                call_args.append(f'reinterpret_cast<{ctype} *>({name}.data())')
                nb_args_by_name[name] = f'nb::arg("{name}").noconvert()'

        elif isinstance(desc, dt.Scalar) and name in optional_symbols:
            # An "artifact" argument (a size symbol outside the user-facing signature): omittable.
            # When omitted, the value comes from an array's shape via the codegen-time inverted
            # expression - or, with no source, a clear error instead of nanobind's missing-argument
            # message. The typed local under the real name keeps init_call and the program call
            # unchanged. The strict_scalar option does not apply here (inference is inherently a conversion).
            params_by_name[name] = f'std::optional<{ctype}> {name}__opt'
            if name in symbol_fallbacks:
                setup_stmts.append(
                    f'const {ctype} {name} = {name}__opt.has_value() ? *{name}__opt : {symbol_fallbacks[name]};')
            else:
                # Must-pass: goes into the leading setup group (see above).
                must_pass_setup.append(f'if (!{name}__opt.has_value())\n'
                                       f'            throw std::invalid_argument("SDFG argument error: '
                                       f'missing argument \'{name}\' (not inferable from any array argument).");\n'
                                       f'        const {ctype} {name} = *{name}__opt;')
            call_args.append(name)
            nb_args_by_name[name] = f'nb::arg("{name}") = nb::none()'

        elif isinstance(desc, dt.Scalar) and desc.dtype.base_type == dtypes.bool_:
            # Bound through the emitted dace_bool caster (see _DACE_BOOL_CASTER): Python bool,
            # numpy.bool_ and integer-like values. Deliberately exempt from strict_scalar's
            # .noconvert(), which would re-reject numpy.bool_.
            params_by_name[name] = f'dace_bool {name}'
            call_args.append(f'static_cast<{ctype}>({name}.value)')
            nb_args_by_name[name] = f'nb::arg("{name}")'

        elif isinstance(desc, dt.Scalar):
            # `.noconvert()` (strict option on) makes nanobind reject even a safe
            # widening scalar cast (e.g. int -> double). A lossy cast (float ->
            # int) is rejected by nanobind regardless. `.noconvert()` also
            # disables the `__index__` path, so it rejects every numpy scalar,
            # even at the exact width (numpy.int32 -> int32_t): strict means
            # built-in Python scalar types only.
            params_by_name[name] = f'{ctype} {name}'
            call_args.append(name)
            nb_args_by_name[name] = (f'nb::arg("{name}").noconvert()' if strict_scalar else f'nb::arg("{name}")')

        else:
            raise NotImplementedError(f'Nanobind interface: argument type {type(desc).__name__} '
                                      f'(argument "{name}") is not supported yet.')
    params = [params_by_name[n] for n in binding_order]
    nb_args = [nb_args_by_name[n] for n in binding_order]
    return params, call_args, nb_args, must_pass_setup + setup_stmts + return_setup


def _ndarray_scalar_ctype(dtype):
    """The C++ scalar type for an ``nb::ndarray<...>`` parameter.

    nanobind's ndarray needs a real scalar type, so a vector dtype binds as its
    base scalar (e.g. ``float``); the caller keeps ``dtype.ctype`` (``dace::vec
    <T, N>``) as the ``reinterpret_cast`` target for the kernel pointer. All
    other dtypes bind as their own ctype.
    """
    if isinstance(dtype, dtypes.vector):
        return dtype.vtype.ctype
    return dtype.ctype


def _uses_half_ndarray(arglist) -> bool:
    """True iff some argument binds a ``dace::float16`` ndarray scalar.

    When this holds the generated TU needs the ``dtype_traits<dace::float16>``
    specialization (plain, nullable, GPU and vector-of-half arrays all bind
    that scalar).
    """
    for desc in arglist.values():
        if isinstance(desc, dt.Array) and _ndarray_scalar_ctype(desc.dtype) == dtypes.float16.ctype:
            return True
    return False


def _pointer_field_names(statestruct):
    """Leading consecutive pointer-field names of the state struct (codegen source of truth).

    Mirrors the ctypes ``_try_parse_state_struct``: only the leading run of
    pointer fields is returned, stopping at the first non-pointer or unparseable
    declaration. This lets the names back a consecutively-laid-out
    ``ctypes.Structure`` overlay (get_state_struct) with correct field offsets.
    """
    names = []
    for decl in statestruct or []:
        decl = decl.strip().rstrip(';').strip()
        if '*' not in decl:
            break
        token = decl.split()[-1].lstrip('*')
        if not token.isidentifier():
            break
        names.append(token)
    return names


def _external_memory_storages(sdfg):
    """Storage types with ``AllocationLifetime.External`` arrays (same scan as framecode)."""
    storages = set()
    for _, _, desc in sdfg.arrays_recursive():
        if desc.lifetime == dtypes.AllocationLifetime.External:
            storages.add(desc.storage)
    return sorted(storages, key=lambda s: s.name)


def generate_bindings_code(sdfg, statestruct=None, gpu_backend=None) -> str:
    """Returns the C++ source of the nanobind module for ``sdfg``.

    :param statestruct: The frame generator's state-struct field declarations;
                        used to bake the pointer-field names into the module.
    :param gpu_backend: ``'cuda'``/``'hip'`` when the CUDA target emitted its
                        init/exit pair for this program (which is exactly when
                        ``__dace_gpu_last_error`` exists), ``None`` otherwise.
                        Gates the per-call read of the SDFG's own GPU error
                        record; declaring the accessor without the target
                        would be a link error, so the caller passes the code
                        objects' truth rather than a storage/schedule
                        heuristic.
    """
    from dace.codegen.targets.cpp import mangle_dace_state_struct_name

    name = sdfg.name
    state_t = mangle_dace_state_struct_name(name)
    arglist = sdfg.arglist()
    # The C++ namespace of the generated types carries the SDFG's content hash:
    # nanobind's process-wide type registry keys by type name, so same-named
    # but different SDFGs (loadable side by side under distinct folder-magic
    # module keys) need distinct type identities to not dispatch into each
    # other. Only disambiguation is needed, not stability: hash_sdfg() is not
    # guaranteed stable across DaCe versions, and a regenerated artifact with
    # a fresh namespace merely forgoes sharing a type identity with older
    # loads (identical copies of one .so still share theirs - the namespace
    # is baked into the file).
    type_ns = f'{name}_{sdfg.hash_sdfg()[:12]}'

    # Extern declarations reuse the exact signature strings framecode emits.
    sig_decl = sdfg.signature(with_types=True, arglist=arglist)
    init_decl = sdfg.init_signature()
    init_call = sdfg.init_signature(for_call=True)

    # GPU error record (mirrors the ctypes interface): after every program call
    # read - and thereby clear - the error the generated code recorded for THIS
    # SDFG, via the accessor the CUDA target exports alongside its init/exit
    # pair. The process-global CUDA last-error slot is deliberately never
    # consulted: it is per-host-thread and shared with every other GPU user in
    # the process, so it can carry third-party state.
    if gpu_backend is not None:
        be = gpu_backend
        gpu_runtime_include = ('\n#include <cuda_runtime.h>' if be == 'cuda' else '\n#include <hip/hip_runtime.h>')
        gpu_check_decl = f'\nint __dace_gpu_last_error({state_t} *__state);'
        gpu_check_method = f'''
    // Raise the error the generated code recorded for this SDFG, if any (the
    // read clears the record, so a failure is delivered exactly once).
    void check_gpu_error() {{
        if (!m_gpu_error_check || !m_state) return;
        const int __err = __dace_gpu_last_error(m_state);
        if (__err != 0)
            throw std::runtime_error(
                "An error was detected when calling \\"{name}\\": " +
                std::string({be}GetErrorString(static_cast<{be}Error_t>(__err))) +
                ". Consider enabling synchronous debugging mode (environment "
                "variable: DACE_compiler_cuda_syncdebug=1) to see where the "
                "issue originates from.");
    }}
'''
        gpu_check_call = '        check_gpu_error();\n'
    else:
        gpu_runtime_include = ''
        gpu_check_decl = ''
        gpu_check_method = ''
        gpu_check_call = ''

    # External-memory guard: an SDFG with external (workspace) memory must not
    # run before set_workspace() - the generated code would dereference a null
    # workspace pointer (a segfault; ctypes runs into the same UB silently).
    # One flag per external storage, set by set_workspace and reset with the
    # state (exit_impl): the association dies with the state it was set on.
    guard_storages = _external_memory_storages(sdfg)
    if guard_storages:
        ws_guard_members = '\n    '.join(f'bool m_ws_set_{s.name} = false;' for s in guard_storages)
        ws_guard_checks = '\n        '.join(
            f'if (!m_ws_set_{s.name})\n'
            f'            throw std::runtime_error(\n'
            f'                "SDFG \'{name}\': external memory (storage {s.name}) was not set; "\n'
            f'                "call set_workspace() before calling.");' for s in guard_storages)
        ws_guard_method = (f'\n    // External-memory guard; see set_workspace/exit_impl for the flag lifecycle.\n'
                           f'    void check_workspace_set() const {{\n'
                           f'        {ws_guard_checks}\n'
                           f'    }}\n    {ws_guard_members}\n')
        ws_check_lead = 'check_workspace_set();\n        '
        ws_reset = '\n        ' + '\n        '.join(f'm_ws_set_{s.name} = false;' for s in guard_storages)
    else:
        ws_guard_method = ''
        ws_check_lead = ''
        ws_reset = ''

    # Bound-parameter order = user-facing positional order (arg_names first,
    # as the old interface's positional calls expect), rest in arglist order -
    # with the omittable symbols last, since defaulted parameters must follow
    # the required ones.
    arg_names = [n for n in (sdfg.arg_names or []) if n in arglist]
    if any(n.startswith('__return') for n in arg_names):
        raise ValueError(f"SDFG '{sdfg.name}': return values cannot be listed in arg_names "
                         f"(they are defaulted keyword-only parameters of the binding).")
    optional_symbols, symbol_fallbacks = _symbol_fallbacks(arglist, arg_names, sdfg.symbols)
    rest = [n for n in arglist.keys() if n not in set(arg_names)]
    return_params = [n for n in rest if n.startswith('__return')]
    # Returns go last: they are defaulted (None -> in-binding allocation), and
    # defaulted parameters must follow the required ones.
    binding_order = (arg_names + [n for n in rest if n not in optional_symbols and n not in return_params] +
                     [n for n in rest if n in optional_symbols] + return_params)
    params, call_args, nb_args, setup_stmts = _argument_binding(arglist,
                                                                binding_order,
                                                                optional_symbols,
                                                                symbol_fallbacks,
                                                                sdfg=sdfg)

    free_symbols = sorted(k for k in sdfg.used_symbols(all_symbols=False) if not k.startswith('__dace'))
    init_arglist = {k: v for k, v in arglist.items() if k in free_symbols}
    init_params, _, init_nb_args, init_setup = _argument_binding(init_arglist, list(init_arglist.keys()), set(), {})

    program_params = f'{state_t} *__state' + (f', {sig_decl}' if sig_decl else '')
    program_args = 'm_state' + (f', {", ".join(call_args)}' if call_args else '')

    call_param_list = ', '.join(params + ['nb::kwargs'])
    init_param_list = ', '.join(init_params + ['nb::kwargs'])
    # The trailing nb::kwargs absorber needs an annotation too.
    call_def_args = ''.join(f', {a}' for a in nb_args + ['nb::arg("_extra_kwargs")'])
    init_def_args = ''.join(f', {a}' for a in init_nb_args + ['nb::arg("_extra_kwargs")'])
    has_gpu = 'true' if _has_gpu_code(sdfg) else 'false'

    # Codegen-time call metadata, exposed on the handle so the Python wrapper
    # does not re-derive it (the __return naming convention lives in one place).
    def _ret_obj(n: str) -> str:
        """The expression yielding the Python object handed back for return ``n``.

        A ``pyobject`` return decays to the single contained object, matching the ctypes
        interface, which returns ``self._return_arrays[i].item()`` whenever the return is a
        pyobject (a proper Scalar, or an Array that wraps one). Everything else hands back the
        array itself.
        """
        desc = sdfg.arrays.get(n)
        if desc is not None and isinstance(desc.dtype, dtypes.pyobject):
            return f'{n}__obj.attr("item")()'
        return f'{n}__obj'

    if '__return' in sdfg.arrays:
        return_names = ('__return', )
        # The single-value convention returns the bare array, not a 1-tuple.
        ret_expr = _ret_obj('__return')
    else:
        found_returns = {n for n in sdfg.arrays if n.startswith('__return_')}
        return_names = tuple(f'__return_{i}' for i in range(len(found_returns)))
        if found_returns != set(return_names):
            raise ValueError(f"SDFG '{sdfg.name}': non-contiguous return-array numbering: {sorted(found_returns)}")
        ret_expr = ('nb::make_tuple(' + ', '.join(_ret_obj(n)
                                                  for n in return_names) + ')') if return_names else 'nb::none()'
    return_names_def = ', '.join(f'"{n}"' for n in return_names)

    # Symbol values are never stored on the handle: the external-memory entry
    # points (framecode.py, generate_external_memory_management) take the init
    # symbols as arguments, so the caller passes them per call - the bound
    # methods accept the full __call__-style argument set and the dispatcher
    # picks the ones needed (the trailing nb::kwargs absorbs the rest).
    init_symbol_names = list(init_arglist.keys())
    init_sym_args = ''.join(f', {s}' for s in init_symbol_names)
    init_comma_decl = f', {init_decl}' if init_decl else ''

    ext_storages = _external_memory_storages(sdfg)
    ext_decls = '\n'.join(f'size_t __dace_get_external_memory_size_{s.name}({state_t} *__state{init_comma_decl});\n'
                          f'void __dace_set_external_memory_{s.name}({state_t} *__state, char *ptr{init_comma_decl});'
                          for s in ext_storages)
    # Storage types are keyed by their *name* (e.g. "CPU_Heap"), not their enum
    # value: the value depends on the declaration order of the StorageType enum,
    # so a module compiled against one DaCe version could be misread by another.
    # The Python side restores the enum via getattr(dtypes.StorageType, name).
    ws_size_entries = '\n        '.join(
        f'sizes["{s.name}"] = nb::int_(__dace_get_external_memory_size_{s.name}(m_state{init_sym_args}));'
        for s in ext_storages)
    ws_set_entries = '\n        '.join(
        f'if (storage == "{s.name}") {{\n'
        f'            __dace_set_external_memory_{s.name}(m_state, reinterpret_cast<char *>(buffer.data()){init_sym_args});\n'
        f'            m_ws_set_{s.name} = true;\n'
        f'            return;\n'
        f'        }}' for s in ext_storages)

    # Setup statements for the workspace methods (their init-symbol
    # parameters arrive like initialize()'s).
    ws_init_setup = ''.join(f'{stmt}\n        ' for stmt in init_setup)

    state_field_appends = '\n            '.join(f'fields.append("{f}");' for f in _pointer_field_names(statestruct))

    # The pre-codegen content hash of the SOURCE SDFG, threaded in by
    # compile() (absent in standalone bindings generation): baked as a module
    # attribute so a later compile() of an unchanged SDFG can mint a fresh
    # handle from the loaded module instead of rename-and-recompile (see
    # compiler.nanobind_reuse_loaded). The post-codegen hash would never
    # match - codegen mutates the SDFG - hence the threading.
    source_hash = getattr(sdfg, '_source_sdfg_hash', None)
    source_hash_attr = f'\n    m.attr("source_sdfg_hash") = "{source_hash}";' if source_hash else ''

    # nanobind's ndarray dtype detection uses std::is_floating_point, which is
    # false for dace::half (on the host path a 2-byte struct of raw IEEE-754
    # half bits). Teach it that dace::float16 is a 16-bit DLPack float so a
    # numpy/cupy float16 array binds by reference - its bytes are exactly an
    # array of dace::half, so the reinterpret_cast is a no-op. Emitted only when
    # a float16 ndarray argument exists.
    float16_traits_block = _FLOAT16_TRAITS if _uses_half_ndarray(arglist) else ''
    bool_caster_block = _DACE_BOOL_CASTER if any(
        isinstance(d, dt.Scalar) and d.dtype.base_type == dtypes.bool_ for d in arglist.values()) else ''

    # setup_stmts (pyobject-array pointer extraction, omittable-symbol locals,
    # return allocation) may need the Python API, so with them the GIL is
    # released only around the kernel call. Without them, call() keeps the
    # simpler whole-body release. initialize() gets the same treatment for its
    # own (init-symbol) setup statements.

    if setup_stmts:
        setup_block = '\n        '.join(setup_stmts)
        call_body = (f'{ws_check_lead}{setup_block}\n'
                     f'        {{\n'
                     f'            nb::gil_scoped_release _nogil;\n'
                     f'            init_impl({init_call});\n'
                     f'            __program_{name}({program_args});\n'
                     f'        }}\n'
                     f'{gpu_check_call}'
                     f'        return {ret_expr};')
    else:
        call_body = (f'{ws_check_lead}// Reading ndarray fields (.data()) needs no Python API, so the whole\n'
                     f'        // init + program call runs with the GIL released, as ctypes did.\n'
                     f'        {{\n'
                     f'            nb::gil_scoped_release _nogil;\n'
                     f'            init_impl({init_call});\n'
                     f'            __program_{name}({program_args});\n'
                     f'        }}\n'
                     f'{gpu_check_call}'
                     f'        return nb::none();')

    if init_setup:
        init_setup_block = '\n        '.join(init_setup)
        initialize_body = (f'{init_setup_block}\n'
                           f'        {{\n'
                           f'            nb::gil_scoped_release _nogil;\n'
                           f'            init_impl({init_call});\n'
                           f'        }}')
    else:
        initialize_body = (f'// GIL released around the C call only; parameter handling stays under\n'
                           f'        // the GIL (a call_guard would copy Python objects without it).\n'
                           f'        nb::gil_scoped_release _nogil;\n'
                           f'        init_impl({init_call});')

    return f'''// Auto-generated nanobind bindings for SDFG '{name}'.
// The include set lives in the runtime umbrella header so the binary-header
// machinery can precompile it once for every generated module.
#include <dace/nanobind.h>{gpu_runtime_include}

namespace nb = nanobind;
{float16_traits_block}{bool_caster_block}
// The generated types live in dace::generated::<name>_<content hash>. nanobind
// shares its type registry across all modules in-process and keys it by type
// name, so the namespace must distinguish not only different SDFG names but
// also same-named SDFGs with different content (which may be loaded side by
// side, each under its own dace.generated.<folder magic>.<name> module): without
// the hash, a handle from one program could silently dispatch into the other's
// methods. Identical content in two modules (a copied artifact) shares the
// type identity - harmless, the code is identical.
extern "C" {{
struct {state_t};
{state_t} *__dace_init_{name}({init_decl});
int __dace_exit_{name}({state_t} *__state);
void __program_{name}({program_params});{gpu_check_decl}
{ext_decls}
}}

namespace dace {{ namespace generated {{ namespace {type_ns} {{

// Not thread-safe (accepted by design - a handle is not meant to be shared
// across threads): the lazy init is an unsynchronized check-then-act on
// m_state and runs with the GIL released, concurrent call()s share the one
// SDFG state struct, and finalize() frees it without synchronizing with
// in-flight calls. Per-call data is all locals, so distinct handles are
// independent.
struct DaceHandle_{name} {{
    {state_t} *m_state = nullptr;
    // Honored by the compiled per-call GPU error check; inert when this module
    // has no GPU code (the check method is only emitted with a GPU target).
    bool m_gpu_error_check = true;
{gpu_check_method}{ws_guard_method}
    void require_state() const {{
        if (!m_state)
            throw std::runtime_error(
                "SDFG '{name}': the state is not initialized (or has been finalized).");
    }}

    DaceHandle_{name}() = default;
    DaceHandle_{name}(const DaceHandle_{name} &) = delete;
    DaceHandle_{name} &operator=(const DaceHandle_{name} &) = delete;
    // Never throw from the destructor; the explicit `finalize()` reports errors.
    ~DaceHandle_{name}() {{
        if (m_state) (void)exit_impl();
    }}

    void init_impl({init_decl}) {{
        if (m_state) return;
        m_state = __dace_init_{name}({init_call});
        if (!m_state) throw std::runtime_error("SDFG '{name}': __dace_init failed.");
    }}

    nb::dict get_workspace_sizes({init_param_list}) {{
        require_state();
        {ws_init_setup}nb::dict sizes;
        {ws_size_entries}
        return sizes;
    }}

    void set_workspace(const std::string &storage, nb::ndarray<> buffer, {init_param_list}) {{
        require_state();
        {ws_init_setup}{ws_set_entries}
        throw std::invalid_argument("SDFG '{name}': no external memory of storage type " + storage);
    }}

    // The state counts as deallocated even on failure (old-interface behavior).
    int exit_impl() {{
        int rc = __dace_exit_{name}(m_state);
        m_state = nullptr;{ws_reset}
        return rc;
    }}

    void initialize({init_param_list}) {{
        {initialize_body}
    }}

    // Returns __dace_exit's code (0 on success, or if already finalized);
    // raising is left to the Python wrapper, which can translate GPU error
    // codes through the GPU runtime.
    int finalize() {{
        if (!m_state) return 0;
        return exit_impl();
    }}

    nb::object call({call_param_list}) {{
        {call_body}
    }}
}};

}} }} }} // namespace dace::generated::{type_ns}

NB_MODULE({name}, m) {{
    using namespace dace::generated::{type_ns};
    nb::class_<DaceHandle_{name}>(m, "CompiledSDFGHandle")
        .def("initialize", &DaceHandle_{name}::initialize{init_def_args})
        .def("finalize", &DaceHandle_{name}::finalize)
        .def("__call__", &DaceHandle_{name}::call{call_def_args})
        .def("get_workspace_sizes", &DaceHandle_{name}::get_workspace_sizes{init_def_args})
        .def("set_workspace", &DaceHandle_{name}::set_workspace,
             nb::arg("storage"), nb::arg("buffer"){init_def_args})
        .def("state_fields", [](DaceHandle_{name} &) {{
            // Baked in at code generation time; only pointer fields.
            nb::list fields;
            {state_field_appends}
            return fields;
        }})
        .def_prop_ro("has_gpu_code", [](DaceHandle_{name} &) {{ return {has_gpu}; }})
        .def_prop_rw("gpu_error_check",
                     [](DaceHandle_{name} &h) {{ return h.m_gpu_error_check; }},
                     [](DaceHandle_{name} &h, bool v) {{ h.m_gpu_error_check = v; }})
        .def_prop_ro("return_names", [](DaceHandle_{name} &) {{ return nb::make_tuple({return_names_def}); }})
        .def_prop_ro("state_pointer", [](DaceHandle_{name} &h) {{
            h.require_state();
            return reinterpret_cast<std::uintptr_t>(h.m_state);
        }});
    m.def("make_compiled_sdfg", []() {{ return new DaceHandle_{name}(); }});{source_hash_attr}
}}
'''
