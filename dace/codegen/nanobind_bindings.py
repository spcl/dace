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

from dace import data as dt, dtypes
from dace.config import Config

# RAII wrapper around a Python buffer view; nested inside the generated handle,
# emitted only when the SDFG has struct arguments. PyBUF_SIMPLE yields the raw
# contiguous bytes regardless of the compound dtype that makes nb::ndarray
# reject record arrays; the resulting address is the struct's/array's data
# pointer. The guard outlives call()'s nested GIL release, so PyBuffer_Release
# always runs with the GIL held.
_PYBUFFER_HELPER = '''    struct _DacePyBuffer {
        Py_buffer view{};
        bool ok = false;
        explicit _DacePyBuffer(PyObject *o) {
            ok = (PyObject_GetBuffer(o, &view, PyBUF_SIMPLE) == 0);
            if (!ok) throw nb::python_error();
        }
        _DacePyBuffer(_DacePyBuffer &&other) noexcept : view(other.view), ok(other.ok) { other.ok = false; }
        _DacePyBuffer(const _DacePyBuffer &) = delete;
        _DacePyBuffer &operator=(const _DacePyBuffer &) = delete;
        ~_DacePyBuffer() { if (ok) PyBuffer_Release(&view); }
        void *data() const { return view.buf; }
    };
'''


def _symbol_fallbacks(arglist: Dict[str, dt.Data], arg_names: List[str],
                      symbol_names: Set[str]) -> Tuple[Set[str], Dict[str, str]]:
    """Determines the optional "artifact" symbols and their C++ shape-inference fallbacks.

    Numeric scalar **SDFG symbols** (``symbol_names``) that are not in the
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

    :param arglist: The full ``sdfg.arglist()``, in C signature order.
    :param arg_names: The user-facing positional argument names.
    :param symbol_names: The names of the SDFG's symbols (``sdfg.symbols``).
    :return: A pair ``(optional_symbols, fallbacks)``: the names that bind as
             optional parameters, and the fallback C++ expression for those
             that have an inference source. An optional symbol absent from
             ``fallbacks`` has no source - omitting it raises at run time.
    """
    import numpy
    import sympy
    from dace import symbolic
    from dace.codegen.common import sym2cpp

    def _omittable(name, desc) -> bool:
        # Inclusive support test: an SDFG symbol bound as a scalar of a plain
        # integer or floating typeclass. Data scalars and subclassed
        # typeclasses (strings, callbacks, pyobjects, vectors, ...) fall
        # outside by construction.
        return (name in symbol_names and isinstance(desc, dt.Scalar) and type(desc.dtype) is dtypes.typeclass and
                (numpy.issubdtype(desc.dtype.type, numpy.integer) or numpy.issubdtype(desc.dtype.type, numpy.floating)))

    arg_names_set = set(arg_names)
    candidates = {name for name, desc in arglist.items() if name not in arg_names_set and _omittable(name, desc)}
    if not candidates:
        return set(), {}

    sources = [(name, desc) for name, desc in arglist.items()
               if name in arg_names_set and isinstance(desc, dt.Array) and not isinstance(desc, dt.ContainerArray)
               and not isinstance(desc.dtype, (dtypes.struct, dtypes.vector)) and desc.optional is not True
               and not name.startswith('__return')]

    dace_infer_src = f'__dace_infer_src_{id(arglist)}'
    placeholder = sympy.Symbol(dace_infer_src)

    def _invert(sym_name: str, ctype: str) -> Optional[str]:
        for aname, desc in sources:
            # Strides are sources like shapes: DaCe descriptor strides and DLPack (nb::ndarray::stride)
            # both count elements. Unlike NumPy's Python level `.strides`.
            for accessor, dims in (('shape', desc.shape), ('stride', desc.strides)):
                for i, dim in enumerate(dims):
                    dim_symbols = symbolic.symlist(dim)
                    if set(dim_symbols.keys()) != {sym_name}:
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


def _argument_binding(arglist: Dict[str, dt.Data], binding_order: List[str], optional_symbols: Set[str],
                      symbol_fallbacks: Dict[str, str]) -> Tuple[List[str], List[str], List[str], List[str]]:
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
             kernel call (in ``arglist`` order; empty without struct/callback
             arguments).
    """
    # params and nb::args are keyed by name so they can be reordered to
    # binding_order at the end; call_args are collected directly in arglist
    # order, which is the C signature order the program call needs.
    params_by_name = {}
    nb_args_by_name = {}
    call_args = []

    # C++ statements (in arglist order) that must run under the GIL before the
    # kernel call - used to extract raw pointers from `nb::object` struct
    # arguments via the Python buffer protocol. Empty for SDFGs without structs.
    setup_stmts = []

    strict_scalar = Config.get_bool('compiler', 'nanobind_strict_scalar_cast')

    for name, desc in arglist.items():
        # A pyobject's ctype is not a C++ type, so refuse here instead of
        # emitting code that fails to compile. Returns are permanently out
        # (arrays only); arguments are callbacks, deferred to part 2 of the port.
        if isinstance(desc.dtype, dtypes.pyobject):
            if name.startswith('__return'):
                raise NotImplementedError(f'Nanobind interface: pyobject return value "{name}" is not supported; '
                                          f'the nanobind interface returns arrays only.')
            raise NotImplementedError(f'Nanobind interface: pyobject argument "{name}" is not supported yet '
                                      f'(callbacks are deferred to part 2 of the port); '
                                      f'use the ctypes interface (compiler.interface=ctypes).')

        # A callback's ctype is not a C++ type, so it cannot go through the
        # generic branches (hence the early `continue`). The wrapper passes the
        # address of a ctypes CFUNCTYPE - whose libffi thunk re-acquires the
        # GIL - and the typed pointer is recovered under the real name, so
        # init_call, the program call and sym_stores all see the right type.
        if isinstance(desc.dtype, dtypes.callback):
            params_by_name[name] = f'std::uintptr_t {name}__addr'
            setup_stmts.append(f'{desc.dtype.as_arg(name)} = reinterpret_cast<{desc.dtype.as_arg("")}>({name}__addr);')
            call_args.append(name)
            nb_args_by_name[name] = f'nb::arg("{name}")'
            continue

        # A non-array return (scalar, structure) cannot carry output back.
        if name.startswith('__return') and not isinstance(desc, dt.Array):
            raise NotImplementedError(f'Nanobind interface: return value "{name}" of type '
                                      f'{type(desc).__name__} is not supported; returns are arrays only.')

        # float16 maps to dace::half, which nanobind's ndarray cannot take as a
        # scalar dtype; a proper mapping is a TODO.
        if desc.dtype.base_type == dtypes.float16:
            raise NotImplementedError(f'Nanobind interface: float16 argument/return value "{name}" is not '
                                      f'supported yet (dace::half is not a valid nanobind ndarray dtype); '
                                      f'use the ctypes interface (compiler.interface=ctypes).')

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

        elif isinstance(desc, dt.ContainerArray):
            # Array of structures: the caller passes a numpy array of
            # per-element pointers (e.g. ctypes.addressof), whose data pointer
            # is forwarded cast to the element-pointer type - no device
            # constraint, the pointer table is passed through as-is. Must
            # precede the Array branch - ContainerArray subclasses Array.
            params_by_name[name] = f'nb::ndarray<uint64_t> {name}'
            call_args.append(f'reinterpret_cast<{ctype} *>({name}.data())')
            nb_args_by_name[name] = f'nb::arg("{name}").noconvert()'

        elif isinstance(desc, dt.Structure):
            # Thin pointer passthrough: the caller builds the C struct as a
            # ctypes.Structure, whose address the buffer protocol yields
            # directly. Marshalling a Python object field by field instead
            # would need the full struct definition; deferred.
            params_by_name[name] = f'nb::object {name}'
            setup_stmts.append(f'_DacePyBuffer {name}_buf({name}.ptr());')
            call_args.append(f'reinterpret_cast<{ctype}>({name}_buf.data())')
            nb_args_by_name[name] = f'nb::arg("{name}")'

        elif isinstance(desc, dt.Array) and isinstance(desc.dtype, dtypes.struct):
            # nb::ndarray rejects a numpy record array (DLPack has no compound dtype),
            # so take a generic object and pull the raw bytes via the buffer protocol.
            # A nullable one accepts None -> null pointer.
            params_by_name[name] = f'nb::object {name}'
            if desc.optional:
                setup_stmts.append(
                    f'std::optional<_DacePyBuffer> {name}_buf; if (!{name}.is_none()) {name}_buf.emplace({name}.ptr());'
                )
                call_args.append(f'{name}.is_none() ? nullptr : reinterpret_cast<{ctype} *>({name}_buf->data())')
                nb_args_by_name[name] = f'nb::arg("{name}").none()'
            else:
                setup_stmts.append(f'_DacePyBuffer {name}_buf({name}.ptr());')
                call_args.append(f'reinterpret_cast<{ctype} *>({name}_buf.data())')
                nb_args_by_name[name] = f'nb::arg("{name}")'

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
                setup_stmts.append(f'if (!{name}__opt.has_value())\n'
                                   f'            throw std::invalid_argument("SDFG argument error: '
                                   f'missing argument \'{name}\' (not inferable from any array argument).");\n'
                                   f'        const {ctype} {name} = *{name}__opt;')
            call_args.append(name)
            nb_args_by_name[name] = f'nb::arg("{name}") = nb::none()'

        elif isinstance(desc, dt.Scalar) and desc.dtype.base_type == dtypes.bool_:
            # nanobind's bool caster accepts only a Python bool; uint8_t also takes numpy bools
            # and ints. Deliberately exempt from strict_scalar's .noconvert(), which would re-reject numpy.bool_.
            params_by_name[name] = f'uint8_t {name}'
            call_args.append(f'static_cast<{ctype}>({name})')
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
    return params, call_args, nb_args, setup_stmts


def _referenced_struct_name(desc):
    """The C struct type name a ``Structure`` / ``ContainerArray``-of-Structure references, else ``None``.

    For a ContainerArray the innermost element type decides (nested containers
    of arrays reference builtins like ``double`` and need no declaration).
    """
    if isinstance(desc, dt.Structure):
        return desc.dtype.ctype.rstrip(' *')  # e.g. "CSRMatrix*" -> "CSRMatrix"
    if isinstance(desc, dt.ContainerArray):
        stype = desc.stype
        while isinstance(stype, dt.ContainerArray):
            stype = stype.stype
        if isinstance(stype, dt.Structure):
            return stype.dtype.ctype.rstrip(' *')
    if isinstance(desc, dt.Array) and isinstance(desc.dtype, dtypes.struct):
        return desc.dtype.ctype
    return None


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


def _structure_forward_decls(arglist):
    """Forward declarations for the C structs referenced by Structure / ContainerArray arguments.

    The bindings only pass pointers to these structs (thin passthrough), so a
    forward declaration is enough - the full definition lives in the frame code.
    """
    decls = []
    seen = set()
    for desc in arglist.values():
        struct_name = _referenced_struct_name(desc)
        if struct_name and struct_name not in seen:
            seen.add(struct_name)
            decls.append(f'struct {struct_name};')
    return '\n'.join(decls)


def _pointer_field_names(statestruct):
    """Names of the pointer fields in the state-struct declarations (codegen source of truth)."""
    names = []
    for decl in statestruct or []:
        decl = decl.strip().rstrip(';').strip()
        if '*' not in decl:
            continue
        token = decl.split()[-1].lstrip('*')
        if token.isidentifier():
            names.append(token)
    return names


def _external_memory_storages(sdfg):
    """Storage types with ``AllocationLifetime.External`` arrays (same scan as framecode)."""
    storages = set()
    for _, _, desc in sdfg.arrays_recursive():
        if desc.lifetime == dtypes.AllocationLifetime.External:
            storages.add(desc.storage)
    return sorted(storages, key=lambda s: s.name)


def generate_bindings_code(sdfg, statestruct=None) -> str:
    """Returns the C++ source of the nanobind module for ``sdfg``.

    :param statestruct: The frame generator's state-struct field declarations;
                        used to bake the pointer-field names into the module.
    """
    from dace.codegen.targets.cpp import mangle_dace_state_struct_name

    name = sdfg.name
    state_t = mangle_dace_state_struct_name(name)
    arglist = sdfg.arglist()

    # Extern declarations reuse the exact signature strings framecode emits.
    sig_decl = sdfg.signature(with_types=True, arglist=arglist)
    init_decl = sdfg.init_signature()
    init_call = sdfg.init_signature(for_call=True)

    # Bound-parameter order = user-facing positional order (arg_names first,
    # as the old interface's positional calls expect), rest in arglist order -
    # with the omittable symbols last, since defaulted parameters must follow
    # the required ones.
    arg_names = [n for n in (sdfg.arg_names or []) if n in arglist]
    optional_symbols, symbol_fallbacks = _symbol_fallbacks(arglist, arg_names, set(sdfg.symbols.keys()))
    rest = [n for n in arglist.keys() if n not in set(arg_names)]
    binding_order = (arg_names + [n for n in rest if n not in optional_symbols] +
                     [n for n in rest if n in optional_symbols])
    params, call_args, nb_args, setup_stmts = _argument_binding(arglist, binding_order, optional_symbols,
                                                                symbol_fallbacks)

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
    # does not re-derive it (the __return naming convention and the callback
    # detection live in one place).
    if '__return' in sdfg.arrays:
        return_names = ('__return', )
        is_single_ret = 'true'
    else:
        found_returns = {n for n in sdfg.arrays if n.startswith('__return_')}
        return_names = tuple(f'__return_{i}' for i in range(len(found_returns)))
        if found_returns != set(return_names):
            raise ValueError(f"SDFG '{sdfg.name}': non-contiguous return-array numbering: {sorted(found_returns)}")
        is_single_ret = 'false'
    return_names_def = ', '.join(f'"{n}"' for n in return_names)
    callback_names_def = ', '.join(f'"{n}"' for n in arglist if isinstance(arglist[n].dtype, dtypes.callback))

    # Symbol values are never stored on the handle: the external-memory entry
    # points (framecode.py, generate_external_memory_management) take the init
    # symbols as arguments, so the caller passes them per call - the bound
    # methods accept the full __call__-style argument set and the dispatcher
    # picks the ones needed (the trailing nb::kwargs absorbs the rest).
    #
    # Callbacks are among these init symbols: the Python frontend registers a
    # callback via `sdfg.add_symbol(name, dtypes.callback(...))`, riding the
    # symbol machinery because a callback shares a symbol's defining property -
    # a scalar, runtime-constant value that parameterizes the execution (here a
    # pointer-sized function pointer). That is why they flow through
    # `used_symbols()` into the init signature, and why the init setup
    # statements (the pointer recovery) apply to the workspace methods too.
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
        f'            return;\n'
        f'        }}' for s in ext_storages)

    # Callback-pointer recovery for the workspace methods (their init-symbol
    # parameters arrive like initialize()'s).
    ws_init_setup = ''.join(f'{stmt}\n        ' for stmt in init_setup)

    state_field_appends = '\n            '.join(f'fields.append("{f}");' for f in _pointer_field_names(statestruct))

    # Forward declarations for structs passed by pointer (Structure arguments).
    struct_decls = _structure_forward_decls(arglist)
    struct_fwd_block = f'\n{struct_decls}' if struct_decls else ''

    # setup_stmts (struct pointer extraction, callback pointer recovery) may
    # need the Python API, so with them the GIL is released only around the
    # kernel call - the RAII buffer guards outlive that nested scope and
    # release with the GIL re-acquired. Without them, call() keeps the simpler
    # whole-body release. initialize() gets the same treatment for its own
    # (init-symbol) setup statements - callbacks are init symbols.
    if setup_stmts:
        pybuffer_helper = _PYBUFFER_HELPER
        setup_block = '\n        '.join(setup_stmts)
        call_body = (f'{setup_block}\n'
                     f'        {{\n'
                     f'            nb::gil_scoped_release _nogil;\n'
                     f'            init_impl({init_call});\n'
                     f'            __program_{name}({program_args});\n'
                     f'        }}')
    else:
        pybuffer_helper = ''
        call_body = (f'// Reading ndarray fields (.data()) needs no Python API, so the whole\n'
                     f'        // init + program call runs with the GIL released, as ctypes did.\n'
                     f'        nb::gil_scoped_release _nogil;\n'
                     f'        init_impl({init_call});\n'
                     f'        __program_{name}({program_args});')

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
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

// DaCe runtime types used in the extern "C" program signature, the argument
// casts, and the nb::ndarray scalar types: dace::uint, dace::complex64/128
// (aliases of unsigned int / std::complex<...>), dace::vec<T, N>, and the
// `pyobject` typedef (pyinterop.h) that callback signatures may reference.
#include <dace/types.h>
#include <dace/vector.h>
#include <dace/pyinterop.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

// The generated types live in dace::generated::<name>, mirroring the Python-side
// dace.generated.<name> module. The per-SDFG namespace also keeps the identically
// structured handle types from different modules from colliding in-process.
extern "C" {{
struct {state_t};{struct_fwd_block}
{state_t} *__dace_init_{name}({init_decl});
int __dace_exit_{name}({state_t} *__state);
void __program_{name}({program_params});
{ext_decls}
}}

namespace dace {{ namespace generated {{ namespace {name} {{

// Not thread-safe (accepted by design - a handle is not meant to be shared
// across threads): the lazy init is an unsynchronized check-then-act on
// m_state and runs with the GIL released, concurrent call()s share the one
// SDFG state struct, and finalize() frees it without synchronizing with
// in-flight calls. Per-call data is all locals, so distinct handles are
// independent.
struct DaceHandle_{name} {{
{pybuffer_helper}    {state_t} *m_state = nullptr;

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
        m_state = nullptr;
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

    void call({call_param_list}) {{
        {call_body}
    }}
}};

}} }} }} // namespace dace::generated::{name}

NB_MODULE({name}, m) {{
    using namespace dace::generated::{name};
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
        .def_prop_ro("return_names", [](DaceHandle_{name} &) {{ return nb::make_tuple({return_names_def}); }})
        .def_prop_ro("is_single_value_ret", [](DaceHandle_{name} &) {{ return {is_single_ret}; }})
        .def_prop_ro("callback_names", [](DaceHandle_{name} &) {{ return nb::make_tuple({callback_names_def}); }})
        .def_prop_ro("state_pointer", [](DaceHandle_{name} &h) {{
            h.require_state();
            return reinterpret_cast<std::uintptr_t>(h.m_state);
        }});
    m.def("make_compiled_sdfg", []() {{ return new DaceHandle_{name}(); }});
}}
'''
