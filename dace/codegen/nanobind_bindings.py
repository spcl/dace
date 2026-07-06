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
    per_name = {}
    for name, desc in arglist.items():
        ctype = desc.dtype.ctype
        if isinstance(desc, dt.Array):
            per_name[name] = (f'nb::ndarray<{ctype}, nb::device::cpu> {name}',
                              f'reinterpret_cast<{ctype} *>({name}.data())', f'nb::arg("{name}").noconvert()')
        elif isinstance(desc, dt.Scalar):
            per_name[name] = (f'{ctype} {name}', name, f'nb::arg("{name}")')
        else:
            raise NotImplementedError(f'Nanobind interface: argument type {type(desc).__name__} '
                                      f'(argument "{name}") is not supported yet.')
    binding_order = list(arglist.keys()) if binding_order is None else binding_order
    params = [per_name[n][0] for n in binding_order]
    call_args = [per_name[n][1] for n in arglist.keys()]
    nb_args = [per_name[n][2] for n in binding_order]
    return params, call_args, nb_args


def generate_bindings_code(sdfg) -> str:
    """Returns the C++ source of the nanobind module for ``sdfg``."""
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

    return f'''// Auto-generated nanobind bindings for SDFG '{name}'.
#include <cstdint>
#include <stdexcept>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

extern "C" {{
struct {state_t};
{state_t} *__dace_init_{name}({init_decl});
int __dace_exit_{name}({state_t} *__state);
void __program_{name}({program_params});
}}

namespace {{

struct DaceHandle_{name} {{
    {state_t} *m_state = nullptr;

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

}} // namespace

NB_MODULE({name}, m) {{
    nb::class_<DaceHandle_{name}>(m, "CompiledSDFGHandle")
        .def("initialize", &DaceHandle_{name}::initialize{init_def_args})
        .def("finalize", &DaceHandle_{name}::finalize)
        .def("__call__", &DaceHandle_{name}::call{call_def_args})
        .def_prop_ro("has_gpu_code", [](DaceHandle_{name} &) {{ return {has_gpu}; }})
        .def_prop_ro("state_pointer", [](DaceHandle_{name} &h) {{
            if (!h.m_state)
                throw std::runtime_error(
                    "SDFG '{name}': the state is not initialized (or has been finalized).");
            return reinterpret_cast<std::uintptr_t>(h.m_state);
        }});
    m.def("make_compiled_sdfg", []() {{ return new DaceHandle_{name}(); }});
}}
'''
