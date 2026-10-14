# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Generator for the nanobind bindings of a compiled SDFG.

Emits one C++ source file that is compiled into the program library itself
(the nanobind extension module *is* the shared library). The module exposes a
``CompiledSDFGHandle`` class wrapping one state handle, and a
``make_compiled_sdfg()`` builder; several handles may share one loaded module.
The bound methods use typed parameters, so argument matching and casting
happen in nanobind's dispatcher.
"""

import io
from collections import OrderedDict
from hashlib import md5

import numpy
import sympy

from dace import data as dt
from dace import dtypes, symbolic
from dace.codegen.common import get_gpu_backend, sdfg_has_gpu_code, sym2cpp
from dace.config import Config

# ml_dtypes-backed low-precision types: numpy cannot export their arrays via
# DLPack or the buffer protocol, so nanobind cannot ingest them; they are
# outside the interface's scope (see argument_unsupported).
_LOWP_TYPES = (dtypes.bfloat16, dtypes.float8_e4m3fn, dtypes.float8_e5m2)


def _ndarray_inexpressible(dtype) -> bool:
    """True when ``nb::ndarray`` has no dtype for ``dtype``'s element type.

    Expressible elements are the plain primitive typeclasses (bool and float16
    included, via the helpers header's casters), vectors thereof, and pyobject
    (marshalled as ``PyObject *``, not through ``nb::ndarray``); other
    subclassed typeclasses (opaque, pointer, ...) have no DLPack dtype and
    would render as their raw ctype inside ``nb::ndarray<...>``.
    """
    elem = dtype.vtype if isinstance(dtype, dtypes.vector) else dtype
    if isinstance(elem, dtypes.pyobject):
        return False
    return type(elem) is not dtypes.typeclass


def argument_unsupported(name: str, desc: dt.Data) -> str | None:
    """The reason argument ``(name, desc)`` is outside the nanobind interface's scope, or ``None``.

    Single source of truth for the interface's data scope:
    :func:`_argument_binding` raises exactly when it returns a reason, and
    :func:`unsupported_reason` (behind ``compiler.interface=auto``) walks the
    same predicate, so the codegen refusals and the automatic ctypes fallback
    cannot diverge.
    """
    hint = 'use the ctypes interface (compiler.interface=ctypes, or the default "auto").'
    if isinstance(desc, dt.Structure):
        return f'argument "{name}" is a Structure, which is out of scope; {hint}'
    if isinstance(desc, dt.ContainerArray):
        return f'argument "{name}" is a ContainerArray, which is out of scope; {hint}'
    if isinstance(desc.dtype, dtypes.callback):
        return f'callback argument "{name}" is out of scope (needs the ctypes trampoline machinery); {hint}'
    if desc.dtype.base_type in _LOWP_TYPES:
        return (
            f'argument "{name}" of low-precision type {desc.dtype} is out of scope '
            f"(nanobind cannot ingest ml_dtypes-backed data); {hint}"
        )
    if name.startswith("__return"):
        if not isinstance(desc, dt.Array):
            return f'return value "{name}" of type {type(desc).__name__} is not supported; returns are arrays only.'
        if isinstance(desc.dtype, dtypes.struct):
            return f'return value "{name}" is a record-dtype (dtypes.struct) array, which is out of scope; {hint}'
        if _ndarray_inexpressible(desc.dtype):
            return f'return value "{name}" has non-primitive element type {desc.dtype}, which is out of scope; {hint}'
        if any(str(o) != "0" for o in desc.offset):
            return f'return value "{name}" has a nonzero offset; in-binding allocation assumes offset 0; {hint}'
        return None
    if isinstance(desc, dt.Array):
        if isinstance(desc.dtype, dtypes.struct):
            return f'argument "{name}" is a record-dtype (dtypes.struct) array, which is out of scope; {hint}'
        if desc.dtype.base_type == dtypes.string:
            return f'argument "{name}" is a string array (char-buffer form), which is out of scope; {hint}'
        if _ndarray_inexpressible(desc.dtype):
            return f'argument "{name}" has non-primitive element type {desc.dtype}, which is out of scope; {hint}'
        return None
    if isinstance(desc, dt.Scalar):
        if isinstance(desc.dtype, dtypes.pyobject) or desc.dtype == dtypes.string:
            return None
        if desc.dtype.base_type == dtypes.float16:
            return (
                f'float16 scalar argument "{name}" is out of scope (dace::half would need a value type-caster); {hint}'
            )
        if isinstance(desc.dtype, dtypes.vector):
            return f'vector scalar argument "{name}" is out of scope; {hint}'
        if type(desc.dtype) is not dtypes.typeclass:
            return f'scalar argument "{name}" of non-primitive type {desc.dtype} is out of scope; {hint}'
        return None
    return f'argument "{name}" of type {type(desc).__name__} is out of scope; {hint}'


def unsupported_reason(sdfg) -> str | None:
    """The reason the nanobind interface cannot compile ``sdfg``, or ``None`` if it can.

    Walks :func:`argument_unsupported` over the full ``sdfg.arglist()``,
    return values included.
    """
    for name, desc in sdfg.arglist().items():
        reason = argument_unsupported(name, desc)
        if reason is not None:
            return reason
    return None


def source_content_hash(sdfg) -> str:
    """The pre-codegen content hash keying loaded-module reuse for ``sdfg``.

    ``compile()`` computes it on the source SDFG (codegen mutates the SDFG, so
    a post-codegen hash would never match) and :func:`generate_bindings_code`
    bakes it into the module as ``source_sdfg_hash``. ``hash_sdfg()`` considers
    neither the instrumentation nor the configuration, but both influence the
    generated code, so they join the key. Hashing the WHOLE effective
    configuration (deterministic: sorted YAML keys, env applied at load time)
    errs toward a redundant rebuild, never a wrong reuse.
    """
    instr_sig = ",".join(
        [sdfg.instrument.name]
        + [n.instrument.name for n, _ in sdfg.all_nodes_recursive() if getattr(n, "instrument", None) is not None]
    )
    config_dump = io.StringIO()
    Config.save(file=config_dump, all=True)
    codegen_sig = f"{instr_sig}:{config_dump.getvalue()}"
    return f"{sdfg.hash_sdfg()}:{md5(codegen_sig.encode('utf-8')).hexdigest()}"


def _symbol_fallbacks(
    arglist: dict[str, dt.Data], arg_names: list[str], symbols: dict[str, dtypes.typeclass]
) -> tuple[set[str], dict[str, str]]:
    """Determines the optional "artifact" symbols and their C++ shape-inference fallbacks.

    Numeric SDFG symbols absent from the user-facing ``arg_names`` (array
    sizes and the like) become optional parameters; when omitted, their value
    is derived from an array argument's shape or strides. The expression is
    inverted with sympy once, at code-generation time, so the run-time
    fallback is plain arithmetic on ``<array>.shape(<dim>)`` /
    ``<array>.stride(<dim>)``.

    :param arglist: The full ``sdfg.arglist()``, in C signature order.
    :param arg_names: The user-facing positional argument names.
    :param symbols: The SDFG's symbol registry (``sdfg.symbols``).
    :return: A pair ``(optional_symbols, fallbacks)``: the names that bind as
             optional parameters, and the fallback C++ expression for those
             that have an inference source. An optional symbol absent from
             ``fallbacks`` has no source - omitting it raises at run time.
    """

    def _plain_numeric_symbol(name) -> bool:
        # Plain integer/floating SDFG symbols only: subclassed typeclasses
        # (strings, callbacks, pyobjects, vectors, ...) and data scalars (not
        # in the symbol registry) fall outside by construction.
        if name not in symbols:
            return False
        dtype = symbols[name]
        return type(dtype) is dtypes.typeclass and (
            numpy.issubdtype(dtype.type, numpy.integer) or numpy.issubdtype(dtype.type, numpy.floating)
        )

    arg_names_set = set(arg_names)
    candidates = {name for name in arglist if name not in arg_names_set and _plain_numeric_symbol(name)}
    if not candidates:
        return set(), {}

    # Symbols in arg_names bind as plain required parameters under their own
    # name, so a dim expression referencing one stays evaluable in the
    # generated fallback.
    explicit_symbols = {name for name in arg_names_set if _plain_numeric_symbol(name)}

    # Inference sources: plain user-facing arrays. Nullable arrays are
    # excluded (may arrive as None), vector arrays too (their run-time shape
    # differs from the descriptor shape).
    sources = [
        (name, desc)
        for name, desc in arglist.items()
        if name in arg_names_set
        and isinstance(desc, dt.Array)
        and not isinstance(desc.dtype, dtypes.vector)
        and desc.optional is not True
        and not name.startswith("__return")
    ]

    dace_infer_src = f"__dace_infer_src_{id(arglist)}"
    placeholder = sympy.Symbol(dace_infer_src)

    def _invert(sym_name: str, ctype: str) -> str | None:
        for aname, desc in sources:
            # Strides are sources like shapes: DaCe descriptor strides and DLPack (nb::ndarray::stride)
            # both count elements. Unlike NumPy's Python level `.strides`.
            for accessor, dims in (("shape", desc.shape), ("stride", desc.strides)):
                for i, dim in enumerate(dims):
                    dim_symbols = symbolic.symlist(dim)
                    free = set(dim_symbols.keys())
                    # Invertible when the target occurs in the dim and every
                    # other free symbol is explicitly passed.
                    if sym_name not in free or (free - {sym_name}) - explicit_symbols:
                        continue
                    try:
                        solutions = sympy.solve(dim - placeholder, dim_symbols[sym_name])
                    except Exception:
                        continue
                    if len(solutions) != 1:
                        continue
                    expr = sym2cpp(solutions[0]).replace(dace_infer_src, f"{aname}.{accessor}({i})")
                    return f"static_cast<{ctype}>({expr})"
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
    DLPack device) binds as host memory.
    """
    if desc.storage == dtypes.StorageType.GPU_Global:
        return "nb::device::cuda" if get_gpu_backend() == "cuda" else "nb::device::rocm"
    return "nb::device::cpu"


def _argument_binding(
    arglist: dict[str, dt.Data],
    binding_order: list[str],
    optional_symbols: set[str],
    symbol_fallbacks: dict[str, str],
    sdfg=None,
    binding_only: set[str] = frozenset(),
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Generates the per-argument pieces of the bound ``call()``/``initialize()`` methods.

    Arrays are taken without implicit conversion: nanobind would otherwise
    silently pass a converted *copy*, breaking DaCe's by-reference argument
    semantics. Scalars keep nanobind's overflow-checked conversion.

    :param arglist: The arguments to bind (``sdfg.arglist()`` or the init
                    subset), in C signature order.
    :param binding_order: The order of the bound parameters (the user-facing
                          positional order).
    :param optional_symbols: Names of the arguments that may be omitted (see
                             :func:`_symbol_fallbacks`).
    :param symbol_fallbacks: The shape-inference fallback expression per
                             optional symbol that has one.
    :return: A 4-tuple of C++ fragment lists: the parameter declarations and
             the ``nb::arg`` annotations (both in ``binding_order``), the
             program-call argument expressions (in ``arglist`` order, the C
             signature order), and the setup statements that must run under
             the GIL before the kernel call.
    """
    # params and nb::args are keyed by name so they can be reordered to
    # binding_order at the end; call_args are collected directly in arglist
    # order, which is the C signature order the program call needs.
    params_by_name = {}
    nb_args_by_name = {}
    call_args = []

    # Setup statements run under the GIL before the kernel call, in three
    # groups: must-pass symbol extractions FIRST (so a deduction may reference
    # an explicitly-passed symbol), then the rest, then return-array setup
    # LAST (allocation sizes reference the symbols established before).
    setup_stmts = []
    must_pass_setup = []
    return_setup = []

    for name, desc in arglist.items():
        reason = argument_unsupported(name, desc)
        if reason is not None:
            raise NotImplementedError(f"Nanobind interface: {reason}")

        # A pyobject is an opaque PyObject* (`typedef void *pyobject` in
        # pyinterop.h): the nb::object parameter holds a reference for the
        # duration of the call and the raw pointer is forwarded, as ctypes'
        # borrowed ctypes.py_object does; the program must not retain it.
        # pyobject returns take the __return branch below instead.
        if isinstance(desc.dtype, dtypes.pyobject) and not name.startswith("__return"):
            if isinstance(desc, dt.Array):
                # numpy dtype=object: a flat run of PyObject* slots. DLPack
                # refuses object arrays, so the pointer comes from
                # __array_interface__ ('O' typestr). The caller's array owns
                # the contained references and the nb::object parameter keeps
                # it alive across the call.
                setup_stmts.append(
                    f'nb::object {name}__ai = {name}.attr("__array_interface__");\n'
                    f'        const std::string {name}__ts = nb::cast<std::string>({name}__ai["typestr"]);\n'
                    f"        if ({name}__ts.size() < 2 || {name}__ts[1] != 'O')\n"
                    f"            throw std::invalid_argument(\"SDFG argument error: argument '{name}': expected "
                    f'an object array (numpy dtype=object), got typestr \'" + {name}__ts + "\'.");\n'
                    f"        const std::uintptr_t {name}__ptr = "
                    f'nb::cast<std::uintptr_t>(nb::tuple({name}__ai["data"])[0]);'
                )
                params_by_name[name] = f"nb::object {name}"
                call_args.append(f"reinterpret_cast<{desc.dtype.ctype} *>({name}__ptr)")
                nb_args_by_name[name] = f'nb::arg("{name}")'
                continue
            params_by_name[name] = f"nb::object {name}"
            call_args.append(f"reinterpret_cast<{desc.dtype.ctype}>({name}.ptr())")
            nb_args_by_name[name] = f'nb::arg("{name}")'
            continue

        if name.startswith("__return"):
            # Return arrays bind as DEFAULTED nb::object parameters and are
            # allocated inside the binding when omitted (None), through
            # NumPy/CuPy via the Python API (GIL held) so ownership, alignment
            # and dtype semantics match the ctypes interface's allocation.
            is_vector_ret = isinstance(desc.dtype, dtypes.vector)
            is_pyobj_ret = isinstance(desc.dtype, dtypes.pyobject)
            gpu = desc.storage == dtypes.StorageType.GPU_Global
            if is_vector_ret:
                # Base scalar with a trailing veclen dimension - the layout
                # numpy itself expands a subarray dtype into.
                nb_scalar = _ndarray_scalar_ctype(desc.dtype)
                device = _ndarray_device(desc)
                dtype_expr = f'__mod.attr("dtype")("{desc.dtype.base_type.as_numpy_dtype().name}")'
            elif is_pyobj_ret:
                # numpy dtype=object, allocated zeroed (None-filled) as the
                # ctypes allocator does.
                nb_scalar = device = None
                dtype_expr = '__mod.attr("dtype")("object")'
            else:
                nb_scalar = _ndarray_scalar_ctype(desc.dtype)
                device = _ndarray_device(desc)
                dtype_expr = f'__mod.attr("dtype")("{desc.dtype.as_numpy_dtype().name}")'
            # Only scalar constants: sympy's subs rejects numpy arrays.
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

            # Descriptor strides count VECTOR elements, so a vector return's
            # byte strides use the full vector width, with a contiguous
            # trailing veclen dimension appended.
            shape_dims = list(desc.shape) + ([desc.dtype.veclen] if is_vector_ret else [])
            dims = ", ".join(_cx(s) for s in shape_dims)
            total = _cx(desc.total_size) + (f" * {desc.dtype.veclen}" if is_vector_ret else "")
            stride_terms = [f"({_cx(s)}) * {desc.dtype.bytes}" for s in desc.strides]
            if is_vector_ret:
                stride_terms.append(str(desc.dtype.base_type.bytes))
            strides_b = ", ".join(stride_terms)
            # A pyobject return is a host-side Python object, never device
            # memory, so it allocates with numpy even for GPU storage.
            if is_pyobj_ret:
                gpu = False
            if gpu:
                alloc = (
                    f'[&]() {{ nb::object __mod = nb::module_::import_("cupy");\n'
                    f"            nb::object __dt = {dtype_expr};\n"
                    f'            return __mod.attr("ndarray")(nb::make_tuple({dims}), __dt, '
                    f'__mod.attr("zeros")({total}, __dt).attr("data"), nb::make_tuple({strides_b})); }}()'
                )
            else:
                alloc = (
                    f'[&]() {{ nb::object __mod = nb::module_::import_("numpy");\n'
                    f"            nb::object __dt = {dtype_expr};\n"
                    f'            return __mod.attr("ndarray")(nb::make_tuple({dims}), __dt, '
                    f'__mod.attr("zeros")({total}, __dt), 0, nb::make_tuple({strides_b})); }}()'
                )

            obtain = (
                f"nb::object {name}__obj = {name};\n"
                f"        if ({name}__obj.is_none()) {{\n"
                f"            {name}__obj = {alloc};\n"
                f"        }}"
            )
            if is_pyobj_ret:
                # DLPack refuses object arrays, so the pointer comes from the
                # array-interface dict; the size check uses .size (no nd view
                # exists here), same contract as the non-pyobject branch.
                iface = "__cuda_array_interface__" if gpu else "__array_interface__"
                extract = (
                    f'nb::object {name}__ai = {name}__obj.attr("{iface}");\n'
                    f"        const std::uintptr_t {name}__ptr = "
                    f'nb::cast<std::uintptr_t>(nb::tuple({name}__ai["data"])[0]);'
                )
                extract += (
                    f"\n        if (!{name}.is_none() && "
                    f'nb::cast<size_t>({name}__obj.attr("size")) < '
                    f"static_cast<size_t>({total}))\n"
                    f'            throw std::invalid_argument("SDFG argument error: return buffer '
                    f"'{name}' has a wrong shape (smaller than the symbol-derived return size).\");"
                )
                return_setup.append(f"{obtain}\n        {extract}")
                call_args.append(f"reinterpret_cast<{desc.dtype.ctype} *>({name}__ptr)")
            else:
                extract = f"auto {name}__nd = nb::cast<nb::ndarray<{nb_scalar}, {device}>>({name}__obj, false);"
                # The program writes through the DESCRIPTOR's shape and
                # strides regardless of the buffer's own, so the guard is
                # against out-of-bounds writes: too SMALL is rejected, a
                # larger caller-provided buffer is legitimate (its tail stays
                # untouched).
                extract += (
                    f"\n        if (!{name}.is_none() && {name}__nd.size() < "
                    f"static_cast<size_t>({total}))\n"
                    f'            throw std::invalid_argument("SDFG argument error: return buffer '
                    f"'{name}' has a wrong shape (smaller than the symbol-derived return size).\");"
                )
                return_setup.append(f"{obtain}\n        {extract}")
                call_args.append(f"reinterpret_cast<{desc.dtype.ctype} *>({name}__nd.data())")
            params_by_name[name] = f"nb::object {name}"
            nb_args_by_name[name] = f'nb::arg("{name}").none() = nb::none()'
            continue

        ctype = desc.dtype.ctype
        if isinstance(desc, dt.Scalar) and desc.dtype == dtypes.string:
            # A string scalar is a C string (int8_t*): std::optional lets None
            # become a null pointer (as ctypes marshals it), and the owned
            # std::string copy keeps the buffer valid across the GIL release.
            params_by_name[name] = f"std::optional<std::string> {name}"
            call_args.append(
                f"{name}.has_value() ? reinterpret_cast<{ctype}>(const_cast<char *>({name}->c_str())) : nullptr"
            )
            nb_args_by_name[name] = f'nb::arg("{name}").none()'

        elif isinstance(desc, dt.Array):
            # The ndarray scalar type may differ from the cast target: a vector
            # dtype binds as its base scalar, but the kernel pointer stays dace::vec*.
            nb_scalar = _ndarray_scalar_ctype(desc.dtype)
            device = _ndarray_device(desc)
            if desc.optional:
                # Nullable array: None becomes a null pointer.
                params_by_name[name] = f"std::optional<nb::ndarray<{nb_scalar}, {device}>> {name}"
                call_args.append(f"{name}.has_value() ? reinterpret_cast<{ctype} *>({name}->data()) : nullptr")
                nb_args_by_name[name] = f'nb::arg("{name}").noconvert().none()'
            else:
                params_by_name[name] = f"nb::ndarray<{nb_scalar}, {device}> {name}"
                call_args.append(f"reinterpret_cast<{ctype} *>({name}.data())")
                nb_args_by_name[name] = f'nb::arg("{name}").noconvert()'

        elif isinstance(desc, dt.Scalar) and name in optional_symbols:
            # Omittable size symbol: when omitted, the value comes from an
            # array's shape via the codegen-time inverted expression. The
            # typed local under the real name keeps init_call and the program
            # call unchanged.
            params_by_name[name] = f"std::optional<{ctype}> {name}__opt"
            if name in symbol_fallbacks:
                setup_stmts.append(
                    f"const {ctype} {name} = {name}__opt.has_value() ? *{name}__opt : {symbol_fallbacks[name]};"
                )
            else:
                must_pass_setup.append(
                    f"if (!{name}__opt.has_value())\n"
                    f'            throw std::invalid_argument("SDFG argument error: '
                    f"missing argument '{name}' (not inferable from any array argument).\");\n"
                    f"        const {ctype} {name} = *{name}__opt;"
                )
            if name not in binding_only:
                call_args.append(name)
            nb_args_by_name[name] = f'nb::arg("{name}") = nb::none()'

        elif isinstance(desc, dt.Scalar) and desc.dtype.base_type == dtypes.bool_:
            # Bound through the shared dace_bool caster (<dace/nanobind_helpers.h>): Python
            # bool, numpy.bool_ and integer-like values.
            params_by_name[name] = f"dace::nanobind_detail::dace_bool {name}"
            call_args.append(f"static_cast<{ctype}>({name}.value)")
            nb_args_by_name[name] = f'nb::arg("{name}")'

        elif isinstance(desc, dt.Scalar):
            # nanobind's standard conversions apply: a safe widening cast (e.g.
            # int -> double) is accepted, a lossy cast (float -> int) rejected.
            params_by_name[name] = f"{ctype} {name}"
            if name not in binding_only:
                call_args.append(name)
            nb_args_by_name[name] = f'nb::arg("{name}")'

        else:
            raise NotImplementedError(
                f'Nanobind interface: argument type {type(desc).__name__} (argument "{name}") is not supported yet.'
            )
    params = [params_by_name[n] for n in binding_order]
    nb_args = [nb_args_by_name[n] for n in binding_order]
    return params, call_args, nb_args, must_pass_setup + setup_stmts + return_setup


def _ndarray_scalar_ctype(dtype):
    """The C++ scalar type for an ``nb::ndarray<...>`` parameter.

    nanobind's ndarray needs a real scalar type, so a vector dtype binds as
    its base scalar; the caller keeps ``dtype.ctype`` (``dace::vec<T, N>``) as
    the ``reinterpret_cast`` target for the kernel pointer.
    """
    if isinstance(dtype, dtypes.vector):
        return dtype.vtype.ctype
    return dtype.ctype


def _pointer_field_names(statestruct):
    """Leading consecutive pointer-field names of the state struct (codegen source of truth).

    Mirrors the ctypes ``_try_parse_state_struct``: only the leading run of
    pointer fields is returned, stopping at the first non-pointer or unparseable
    declaration. This lets the names back a consecutively-laid-out
    ``ctypes.Structure`` overlay (get_state_struct) with correct field offsets.
    """
    names = []
    for decl in statestruct or []:
        decl = decl.strip().rstrip(";").strip()
        if "*" not in decl:
            break
        token = decl.split()[-1].lstrip("*")
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


def generate_bindings_code(sdfg, statestruct=None, gpu_backend=None, source_hash=None) -> str:
    """Returns the C++ source of the nanobind module for ``sdfg``.

    :param statestruct: The frame generator's state-struct field declarations;
                        used to bake the pointer-field names into the module.
    :param gpu_backend: ``'cuda'``/``'hip'`` when the CUDA target emitted its
                        code object for this program (which is exactly when
                        ``__dace_gpu_last_error`` exists - declaring the
                        accessor without the target would be a link error),
                        ``None`` otherwise.
    :param source_hash: The :func:`source_content_hash` of the SOURCE SDFG
                        (codegen has mutated ``sdfg`` by now), baked into the
                        module for loaded-module reuse; ``None`` omits it.
    """
    from dace.codegen.targets.cpp import mangle_dace_state_struct_name

    name = sdfg.name
    state_t = mangle_dace_state_struct_name(name)
    arglist = sdfg.arglist()
    # The generated types' C++ namespace carries the SDFG's content hash:
    # nanobind's process-wide type registry keys by type name, so same-named
    # but different SDFGs (loadable side by side under distinct folder-magic
    # module keys) need distinct type identities to not dispatch into each
    # other. Only disambiguation is needed, not hash stability across DaCe
    # versions.
    type_ns = f"{name}_{sdfg.hash_sdfg()[:12]}"

    # Extern declarations reuse the exact signature strings framecode emits.
    sig_decl = sdfg.signature(with_types=True, arglist=arglist)
    init_decl = sdfg.init_signature()
    init_call = sdfg.init_signature(for_call=True)

    # GPU error record (mirrors the ctypes interface): after every program
    # call read - and thereby clear - the error the generated code recorded
    # for THIS SDFG. The process-global CUDA last-error slot is never
    # consulted: it is per-host-thread and shared with every other GPU user in
    # the process. The raising lives in the runtime header
    # (dace::nanobind_detail::check_gpu_error); only the program-specific
    # pieces are baked in.
    if gpu_backend is not None:
        be = gpu_backend
        gpu_runtime_include = "\n#include <cuda_runtime.h>" if be == "cuda" else "\n#include <hip/hip_runtime.h>"
        gpu_check_decl = f"\nint __dace_gpu_last_error({state_t} *__state);"

        gpu_check_method = f"""
    void check_gpu_error() {{
        dace::nanobind_detail::check_gpu_error(
            m_state, m_gpu_error_check, __dace_gpu_last_error,
            [](int e) {{ return {be}GetErrorString(static_cast<{be}Error_t>(e)); }}, "{name}");
    }}
"""
        gpu_check_call = "        check_gpu_error();\n"
    else:
        gpu_runtime_include = ""
        gpu_check_decl = ""
        gpu_check_method = ""
        gpu_check_call = ""

    # External-memory guard: running before set_workspace() would dereference
    # a null workspace pointer. One flag per external storage, set by
    # set_workspace and reset with the state (exit_impl).
    guard_storages = _external_memory_storages(sdfg)
    if guard_storages:
        ws_guard_members = "\n    ".join(f"bool m_ws_set_{s.name} = false;" for s in guard_storages)
        ws_guard_checks = "\n        ".join(
            f"if (!m_ws_set_{s.name})\n"
            f"            throw std::runtime_error(\n"
            f"                \"SDFG '{name}': external memory (storage {s.name}) was not set; \"\n"
            f'                "call set_workspace() before calling.");'
            for s in guard_storages
        )
        ws_guard_method = (
            f"\n    // External-memory guard; see set_workspace/exit_impl for the flag lifecycle.\n"
            f"    void check_workspace_set() const {{\n"
            f"        {ws_guard_checks}\n"
            f"    }}\n    {ws_guard_members}\n"
        )
        ws_check_lead = "check_workspace_set();\n        "
        ws_reset = "\n        " + "\n        ".join(f"m_ws_set_{s.name} = false;" for s in guard_storages)
    else:
        ws_guard_method = ""
        ws_check_lead = ""
        ws_reset = ""

    # Bound-parameter order: arg_names first (the user-facing positional
    # order), rest in arglist order, defaulted parameters (omittable symbols,
    # returns) after the required ones.
    arg_names = [n for n in (sdfg.arg_names or []) if n in arglist]
    if any(n.startswith("__return") for n in arg_names):
        raise ValueError(
            f"SDFG '{sdfg.name}': return values cannot be listed in arg_names "
            f"(they are defaulted keyword-only parameters of the binding)."
        )

    # A symbol appearing ONLY in a return descriptor's shape/strides is absent
    # from arglist(), but the in-binding return allocation renders it into
    # C++, so it must still be a bindable parameter. Such BINDING-ONLY symbols
    # are never forwarded to the program call (the extern "C" signature does
    # not take them).
    binding_only = []
    for ret_name in sorted(n for n in sdfg.arrays if n.startswith("__return")):
        for sym in sorted(map(str, sdfg.arrays[ret_name].free_symbols)):
            if sym in arglist or sym in binding_only or sym.startswith("__dace"):
                continue
            if sym not in sdfg.symbols:
                raise ValueError(
                    f"SDFG '{sdfg.name}': return array '{ret_name}' uses symbol '{sym}', "
                    f"which is not in the SDFG's symbol registry."
                )
            binding_only.append(sym)
    if binding_only:
        arglist = OrderedDict(arglist)
        for sym in binding_only:
            arglist[sym] = dt.Scalar(sdfg.symbols[sym])
    binding_only = frozenset(binding_only)

    optional_symbols, symbol_fallbacks = _symbol_fallbacks(arglist, arg_names, sdfg.symbols)
    rest = [n for n in arglist.keys() if n not in set(arg_names)]
    return_params = [n for n in rest if n.startswith("__return")]
    binding_order = (
        arg_names
        + [n for n in rest if n not in optional_symbols and n not in return_params]
        + [n for n in rest if n in optional_symbols]
        + return_params
    )
    params, call_args, nb_args, setup_stmts = _argument_binding(
        arglist, binding_order, optional_symbols, symbol_fallbacks, sdfg=sdfg, binding_only=binding_only
    )

    free_symbols = sorted(k for k in sdfg.used_symbols(all_symbols=False) if not k.startswith("__dace"))
    init_arglist = {k: v for k, v in arglist.items() if k in free_symbols}
    init_params, _, init_nb_args, init_setup = _argument_binding(init_arglist, list(init_arglist.keys()), set(), {})

    program_params = f"{state_t} *__state" + (f", {sig_decl}" if sig_decl else "")
    program_args = "m_state" + (f", {', '.join(call_args)}" if call_args else "")

    call_param_list = ", ".join(params + ["nb::kwargs"])
    init_param_list = ", ".join(init_params + ["nb::kwargs"])
    # The trailing nb::kwargs absorber needs an annotation too.
    call_def_args = "".join(f", {a}" for a in nb_args + ['nb::arg("_extra_kwargs")'])
    init_def_args = "".join(f", {a}" for a in init_nb_args + ['nb::arg("_extra_kwargs")'])
    has_gpu = "true" if sdfg_has_gpu_code(sdfg) else "false"

    def _ret_obj(n: str) -> str:
        """The expression yielding the Python object handed back for return ``n``.

        A ``pyobject`` return decays to the single contained object (ctypes
        parity); everything else hands back the array itself.
        """
        desc = sdfg.arrays.get(n)
        if desc is not None and isinstance(desc.dtype, dtypes.pyobject):
            return f'{n}__obj.attr("item")()'
        return f"{n}__obj"

    if "__return" in sdfg.arrays:
        return_names = ("__return",)
        # The single-value convention returns the bare array, not a 1-tuple.
        ret_expr = _ret_obj("__return")
    else:
        found_returns = {n for n in sdfg.arrays if n.startswith("__return_")}
        return_names = tuple(f"__return_{i}" for i in range(len(found_returns)))
        if found_returns != set(return_names):
            raise ValueError(f"SDFG '{sdfg.name}': non-contiguous return-array numbering: {sorted(found_returns)}")
        ret_expr = (
            ("nb::make_tuple(" + ", ".join(_ret_obj(n) for n in return_names) + ")") if return_names else "nb::none()"
        )
    return_names_def = ", ".join(f'"{n}"' for n in return_names)

    # Symbol values are never stored on the handle: the external-memory entry
    # points take the init symbols as arguments, so the caller passes them per
    # call (the trailing nb::kwargs absorbs the rest of the argument set).
    init_symbol_names = list(init_arglist.keys())
    init_sym_args = "".join(f", {s}" for s in init_symbol_names)
    init_comma_decl = f", {init_decl}" if init_decl else ""

    ext_storages = _external_memory_storages(sdfg)
    ext_decls = "\n".join(
        f"size_t __dace_get_external_memory_size_{s.name}({state_t} *__state{init_comma_decl});\n"
        f"void __dace_set_external_memory_{s.name}({state_t} *__state, char *ptr{init_comma_decl});"
        for s in ext_storages
    )
    # Storage types are keyed by *name*, not enum value: the value depends on
    # the StorageType declaration order, so a module compiled against one DaCe
    # version could be misread by another.
    ws_size_entries = "\n        ".join(
        f'sizes["{s.name}"] = nb::int_(__dace_get_external_memory_size_{s.name}(m_state{init_sym_args}));'
        for s in ext_storages
    )
    ws_set_entries = "\n        ".join(
        f'if (storage == "{s.name}") {{\n'
        f"            __dace_set_external_memory_{s.name}(m_state, reinterpret_cast<char *>(buffer.data()){init_sym_args});\n"
        f"            m_ws_set_{s.name} = true;\n"
        f"            return;\n"
        f"        }}"
        for s in ext_storages
    )

    # Setup statements for the workspace methods (their init-symbol
    # parameters arrive like initialize()'s).
    ws_init_setup = "".join(f"{stmt}\n        " for stmt in init_setup)

    state_field_appends = "\n            ".join(f'fields.append("{f}");' for f in _pointer_field_names(statestruct))

    source_hash_attr = f'\n    m.attr("source_sdfg_hash") = "{source_hash}";' if source_hash else ""

    # Setup statements may need the Python API, so with them the GIL is
    # released only around the kernel call; without them, call() and
    # initialize() keep the simpler whole-body release.
    if setup_stmts:
        setup_block = "\n        ".join(setup_stmts)
        call_body = (
            f"{ws_check_lead}{setup_block}\n"
            f"        {{\n"
            f"            nb::gil_scoped_release _nogil;\n"
            f"            init_impl({init_call});\n"
            f"            __program_{name}({program_args});\n"
            f"        }}\n"
            f"{gpu_check_call}"
            f"        return {ret_expr};"
        )
    else:
        call_body = (
            f"{ws_check_lead}// Reading ndarray fields (.data()) needs no Python API, so the whole\n"
            f"        // init + program call runs with the GIL released, as ctypes did.\n"
            f"        {{\n"
            f"            nb::gil_scoped_release _nogil;\n"
            f"            init_impl({init_call});\n"
            f"            __program_{name}({program_args});\n"
            f"        }}\n"
            f"{gpu_check_call}"
            f"        return nb::none();"
        )

    if init_setup:
        init_setup_block = "\n        ".join(init_setup)
        initialize_body = (
            f"{init_setup_block}\n"
            f"        {{\n"
            f"            nb::gil_scoped_release _nogil;\n"
            f"            init_impl({init_call});\n"
            f"        }}"
        )
    else:
        initialize_body = (
            f"// GIL released around the C call only; parameter handling stays under\n"
            f"        // the GIL (a call_guard would copy Python objects without it).\n"
            f"        nb::gil_scoped_release _nogil;\n"
            f"        init_impl({init_call});"
        )

    return f"""// Auto-generated nanobind bindings for SDFG '{name}'.
// The include set lives in the runtime umbrella header so the binary-header
// machinery can precompile it once for every generated module.
#include <dace/nanobind.h>{gpu_runtime_include}

namespace nb = nanobind;

// The generated types live in dace::generated::<name>_<content hash>: nanobind
// keys its process-wide type registry by type name, so same-named SDFGs with
// different content (loadable side by side) need distinct type identities -
// without the hash, a handle from one program could silently dispatch into
// the other's methods.
extern "C" {{
struct {state_t};
{state_t} *__dace_init_{name}({init_decl});
int __dace_exit_{name}({state_t} *__state);
void __program_{name}({program_params});{gpu_check_decl}
{ext_decls}
}}

namespace dace {{ namespace generated {{ namespace {type_ns} {{

// Not thread-safe by design: a handle is not meant to be shared across
// threads (lazy init and finalize() do not synchronize with in-flight calls).
// Distinct handles are independent.
struct DaceHandle_{name} {{
    {state_t} *m_state = nullptr;
    // Honored by the per-call GPU error check; inert without GPU code.
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
"""
