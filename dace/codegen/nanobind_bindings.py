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

from dace import data as dt, dtypes
from dace.config import Config


def _has_gpu_code(sdfg) -> bool:
    """Same detection as the ctypes ``CompiledSDFG``, evaluated at codegen time."""
    for _, _, desc in sdfg.arrays_recursive():
        if desc.storage in dtypes.GPU_STORAGES:
            return True
    for node, _ in sdfg.all_nodes_recursive():
        if getattr(node, 'schedule', False) in dtypes.GPU_SCHEDULES:
            return True
    return False


def _argument_binding(arglist, binding_order=None):
    """Per-argument (C++ parameter, program-call expression, nb::arg annotation) lists.

    ``binding_order`` controls the order of the bound parameters (the
    user-facing positional order); the program-call expressions are always
    returned in ``arglist`` order, which is the C signature order.

    Arrays are taken without implicit conversion: nanobind would otherwise
    silently pass a converted *copy*, breaking DaCe's by-reference argument
    semantics. Scalars keep nanobind's overflow-checked conversion.
    """
    # params and nb::args are keyed by name so they can be reordered to
    # binding_order at the end; call_args are collected directly in arglist
    # order, which is the C signature order the program call needs.
    params_by_name = {}
    nb_args_by_name = {}
    call_args = []
    strict_scalar = Config.get_bool('compiler', 'nanobind_strict_scalar_cast')
    for name, desc in arglist.items():
        # pyobject (arguments and returns, incl. the PR#2206 bug-compatible
        # decay of pyobject arrays) is deferred to part 2 of the port; its
        # ctype is not a C++ type, so refuse here instead of emitting code
        # that fails to compile.
        if isinstance(desc.dtype, dtypes.pyobject):
            # pyobject returns are dropped (the nanobind interface returns arrays
            # only); pyobject arguments are callbacks, deferred to part 2.
            if name.startswith('__return'):
                raise NotImplementedError(f'Nanobind interface: pyobject return value "{name}" is not supported; '
                                          f'the nanobind interface returns arrays only.')
            raise NotImplementedError(f'Nanobind interface: pyobject argument "{name}" is not supported yet '
                                      f'(callbacks are deferred to part 2 of the port); '
                                      f'use the ctypes interface (compiler.interface=ctypes).')
        # Callbacks (a dtypes.callback scalar) are not a pyobject subclass, so
        # the check above misses them; their ctype ("dace.callback") is not a
        # C++ type, so refuse here instead of emitting code that fails to
        # compile. Callback support is deferred to part 2 of the port.
        if isinstance(desc.dtype, dtypes.callback):
            raise NotImplementedError(f'Nanobind interface: callback argument "{name}" is not supported yet '
                                      f'(callbacks are deferred to part 2 of the port); '
                                      f'use the ctypes interface (compiler.interface=ctypes).')
        # Return values must be arrays; a non-array return (scalar, structure)
        # cannot carry output back, so refuse it here at codegen time.
        if name.startswith('__return') and not isinstance(desc, dt.Array):
            raise NotImplementedError(f'Nanobind interface: return value "{name}" of type '
                                      f'{type(desc).__name__} is not supported; returns are arrays only.')
        # float16 maps to dace::half, a custom struct nanobind's ndarray cannot
        # accept as a scalar dtype; refuse loudly instead of emitting code that
        # fails to compile. A proper mapping is deferred to a future slice.
        if desc.dtype.base_type == dtypes.float16:
            raise NotImplementedError(f'Nanobind interface: float16 argument/return value "{name}" is not '
                                      f'supported yet (dace::half is not a valid nanobind ndarray dtype); '
                                      f'use the ctypes interface (compiler.interface=ctypes).')
        ctype = desc.dtype.ctype
        if isinstance(desc, dt.Scalar) and desc.dtype == dtypes.string:
            # A string scalar is a C string (int8_t*). Marshal a Python str -
            # or None, which the ctypes path also accepts - through
            # std::optional<std::string>: None becomes a null pointer, and the
            # owned std::string keeps its buffer valid across the GIL release.
            # .none() is required: nanobind rejects None by default, and
            # std::optional only accepts it implicitly on some versions.
            params_by_name[name] = f'std::optional<std::string> {name}'
            call_args.append(
                f'{name}.has_value() ? reinterpret_cast<{ctype}>(const_cast<char *>({name}->c_str())) : nullptr')
            nb_args_by_name[name] = f'nb::arg("{name}").none()'

        elif isinstance(desc, dt.ContainerArray):
            # ContainerArray (an array of structures/containers): the caller
            # passes a numpy array of pointers - one per element, e.g.
            # ``ctypes.addressof`` of a per-element ctypes.Structure - exactly as
            # on the ctypes path. Forward that array's data pointer, cast to the
            # element-pointer type (``ctype`` is the element's pointer, so the
            # buffer is ``ctype *``). Structs referenced this way are
            # forward-declared (see generate_bindings_code). Checked before the
            # Array branch, since ContainerArray subclasses Array.
            params_by_name[name] = f'nb::ndarray<uint64_t, nb::device::cpu> {name}'
            call_args.append(f'reinterpret_cast<{ctype} *>({name}.data())')
            nb_args_by_name[name] = f'nb::arg("{name}").noconvert()'

        elif isinstance(desc, dt.Structure):
            # Thin pointer passthrough, mirroring the ctypes path: the caller
            # builds the C struct (as a ``ctypes.Structure`` filled with the raw
            # pointers of its members) and we forward a pointer to it. The struct
            # type is forward-declared in the module (see generate_bindings_code).
            # A richer alternative would marshal a Python dict/object field by
            # field into the struct inside C++ - but that needs the full struct
            # definition and per-member handling (including array pointers and
            # nested structs), so it is deferred as a future improvement.
            params_by_name[name] = f'std::uintptr_t {name}'
            call_args.append(f'reinterpret_cast<{ctype}>({name})')
            nb_args_by_name[name] = f'nb::arg("{name}")'

        elif isinstance(desc, dt.Array) and isinstance(desc.dtype, dtypes.struct):
            # Array of a C struct declared with `dace.struct(...)`. Because `nanobind`
            # rejects such inputs, we pass it as a byte array (`NanobindCompiledSDFG`
            # byte-views the caller's record array) and cast it back here. A nullable
            # one (optional is not False) accepts None -> null pointer, like plain
            # arrays; .none() is required.
            if desc.optional is False:
                params_by_name[name] = f'nb::ndarray<uint8_t, nb::device::cpu> {name}'
                call_args.append(f'reinterpret_cast<{ctype} *>({name}.data())')
                nb_args_by_name[name] = f'nb::arg("{name}").noconvert()'
            else:
                params_by_name[name] = f'std::optional<nb::ndarray<uint8_t, nb::device::cpu>> {name}'
                call_args.append(f'{name}.has_value() ? reinterpret_cast<{ctype} *>({name}->data()) : nullptr')
                nb_args_by_name[name] = f'nb::arg("{name}").noconvert().none()'

        elif isinstance(desc, dt.Array):
            # The ndarray scalar type may differ from the cast target: a vector
            # dtype binds as its base scalar, but the kernel pointer stays dace::vec*.
            nb_scalar = _ndarray_scalar_ctype(desc.dtype)
            if desc.optional is False:
                params_by_name[name] = f'nb::ndarray<{nb_scalar}, nb::device::cpu> {name}'
                call_args.append(f'reinterpret_cast<{ctype} *>({name}.data())')
                nb_args_by_name[name] = f'nb::arg("{name}").noconvert()'
            else:
                # Nullable array: None (allowed unless optional is explicitly
                # False) becomes a null pointer, matching the ctypes marshaller.
                # .none() is required (see the string case above).
                params_by_name[name] = f'std::optional<nb::ndarray<{nb_scalar}, nb::device::cpu>> {name}'
                call_args.append(f'{name}.has_value() ? reinterpret_cast<{ctype} *>({name}->data()) : nullptr')
                nb_args_by_name[name] = f'nb::arg("{name}").noconvert().none()'

        elif isinstance(desc, dt.Scalar) and desc.dtype.base_type == dtypes.bool_:
            # nanobind's `bool` caster accepts only a Python bool (rejects
            # numpy.bool_); bind as uint8_t (whose caster accepts Python/numpy
            # bools and ints) and cast back to bool for the kernel call. Matches
            # ctypes' permissive coercion. Bool deliberately ignores the
            # strict_scalar `.noconvert()` option, which would re-reject
            # numpy.bool_ and defeat this.
            params_by_name[name] = f'uint8_t {name}'
            call_args.append(f'static_cast<{ctype}>({name})')
            nb_args_by_name[name] = f'nb::arg("{name}")'

        elif isinstance(desc, dt.Scalar):
            # `.noconvert()` (strict option on) makes nanobind reject even a safe
            # widening scalar cast (e.g. int -> double). A lossy cast (float ->
            # int) is rejected by nanobind regardless.
            params_by_name[name] = f'{ctype} {name}'
            call_args.append(name)
            nb_args_by_name[name] = (f'nb::arg("{name}").noconvert()' if strict_scalar else f'nb::arg("{name}")')

        else:
            raise NotImplementedError(f'Nanobind interface: argument type {type(desc).__name__} '
                                      f'(argument "{name}") is not supported yet.')
    binding_order = list(arglist.keys()) if binding_order is None else binding_order
    params = [params_by_name[n] for n in binding_order]
    nb_args = [nb_args_by_name[n] for n in binding_order]
    return params, call_args, nb_args


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
    # as the old interface's positional calls expect), rest in arglist order.
    arg_names = [n for n in (sdfg.arg_names or []) if n in arglist]
    binding_order = arg_names + [n for n in arglist.keys() if n not in set(arg_names)]
    params, call_args, nb_args = _argument_binding(arglist, binding_order)

    free_symbols = sorted(k for k in sdfg.used_symbols(all_symbols=False) if not k.startswith('__dace'))
    init_arglist = {k: v for k, v in arglist.items() if k in free_symbols}
    init_params, _, init_nb_args = _argument_binding(init_arglist)

    program_params = f'{state_t} *__state' + (f', {sig_decl}' if sig_decl else '')
    program_args = 'm_state' + (f', {", ".join(call_args)}' if call_args else '')

    call_param_list = ', '.join(params + ['nb::kwargs'])
    init_param_list = ', '.join(init_params + ['nb::kwargs'])
    # The trailing nb::kwargs absorber needs an annotation too.
    call_def_args = ''.join(f', {a}' for a in nb_args + ['nb::arg("_extra_kwargs")'])
    init_def_args = ''.join(f', {a}' for a in init_nb_args + ['nb::arg("_extra_kwargs")'])
    has_gpu = 'true' if _has_gpu_code(sdfg) else 'false'

    # Init symbols are stored on the handle: the external-memory entry points
    # take them as arguments (framecode.py, generate_external_memory_management),
    # mirroring the ctypes wrapper's use of the last call's arguments.
    init_symbol_names = list(init_arglist.keys())
    sym_members = '\n'.join(f'    {init_arglist[s].dtype.ctype} m_sym_{s} = {{}};' for s in init_symbol_names)
    sym_stores = '\n        '.join(f'm_sym_{s} = {s};' for s in init_symbol_names)
    sym_args = ''.join(f', m_sym_{s}' for s in init_symbol_names)
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
        f'sizes["{s.name}"] = nb::int_(__dace_get_external_memory_size_{s.name}(m_state{sym_args}));'
        for s in ext_storages)
    ws_set_entries = '\n        '.join(
        f'if (storage == "{s.name}") {{\n'
        f'            __dace_set_external_memory_{s.name}(m_state, reinterpret_cast<char *>(buffer.data()){sym_args});\n'
        f'            return;\n'
        f'        }}' for s in ext_storages)

    state_field_appends = '\n            '.join(f'fields.append("{f}");' for f in _pointer_field_names(statestruct))

    # Forward declarations for structs passed by pointer (Structure arguments).
    struct_decls = _structure_forward_decls(arglist)
    struct_fwd_block = f'\n{struct_decls}' if struct_decls else ''

    return f'''// Auto-generated nanobind bindings for SDFG '{name}'.
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

// DaCe runtime types used in the extern "C" program signature, the argument
// casts, and the nb::ndarray scalar types: dace::uint, dace::complex64/128
// (aliases of unsigned int / std::complex<...>), and dace::vec<T, N>.
#include <dace/types.h>
#include <dace/vector.h>

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

struct DaceHandle_{name} {{
    {state_t} *m_state = nullptr;
{sym_members}

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
        {sym_stores}
        m_state = __dace_init_{name}({init_call});
        if (!m_state) throw std::runtime_error("SDFG '{name}': __dace_init failed.");
    }}

    nb::dict get_workspace_sizes() {{
        require_state();
        nb::dict sizes;
        {ws_size_entries}
        return sizes;
    }}

    void set_workspace(const std::string &storage, nb::ndarray<> buffer) {{
        require_state();
        {ws_set_entries}
        throw std::invalid_argument("SDFG '{name}': no external memory of storage type " + storage);
    }}

    // The state counts as deallocated even on failure (old-interface behavior).
    int exit_impl() {{
        int rc = __dace_exit_{name}(m_state);
        m_state = nullptr;
        return rc;
    }}

    void initialize({init_param_list}) {{
        // GIL released around the C call only; parameter handling stays under
        // the GIL (a call_guard would copy Python objects without it).
        nb::gil_scoped_release _nogil;
        init_impl({init_call});
    }}

    void finalize() {{
        if (!m_state) return;
        int rc = exit_impl();
        if (rc != 0)
            throw std::runtime_error("An error was detected after running '{name}': exit code " +
                                     std::to_string(rc));
    }}

    void call({call_param_list}) {{
        // Reading ndarray fields (.data()) needs no Python API, so the whole
        // init + program call runs with the GIL released, as ctypes did.
        nb::gil_scoped_release _nogil;
        init_impl({init_call});
        __program_{name}({program_args});
    }}
}};

}} }} }} // namespace dace::generated::{name}

NB_MODULE({name}, m) {{
    using namespace dace::generated::{name};
    nb::class_<DaceHandle_{name}>(m, "CompiledSDFGHandle")
        .def("initialize", &DaceHandle_{name}::initialize{init_def_args})
        .def("finalize", &DaceHandle_{name}::finalize)
        .def("__call__", &DaceHandle_{name}::call{call_def_args})
        .def("get_workspace_sizes", &DaceHandle_{name}::get_workspace_sizes)
        .def("set_workspace", &DaceHandle_{name}::set_workspace,
             nb::arg("storage"), nb::arg("buffer"))
        .def("state_fields", [](DaceHandle_{name} &) {{
            // Baked in at code generation time; only pointer fields.
            nb::list fields;
            {state_field_appends}
            return fields;
        }})
        .def_prop_ro("has_gpu_code", [](DaceHandle_{name} &) {{ return {has_gpu}; }})
        .def_prop_ro("state_pointer", [](DaceHandle_{name} &h) {{
            h.require_state();
            return reinterpret_cast<std::uintptr_t>(h.m_state);
        }});
    m.def("make_compiled_sdfg", []() {{ return new DaceHandle_{name}(); }});
}}
'''
