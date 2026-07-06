# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Generator for the nanobind bindings of a compiled SDFG.

Emits one C++ source file that is compiled into the program library itself
(the nanobind extension module *is* the shared library). The module exposes a
``CompiledSDFGHandle`` class wrapping one state handle, and a
``make_compiled_sdfg()`` builder; several handles may share one loaded module.
"""

from dace import data as dt


def _cpp_marshalling(arglist) -> str:
    """Generates per-argument extraction of C values from the Python kwargs."""
    lines = []
    for name, desc in arglist.items():
        ctype = desc.dtype.ctype
        if isinstance(desc, dt.Array):
            # No implicit conversion: nanobind would silently pass a converted
            # *copy*, breaking DaCe's by-reference argument semantics.
            lines.append(f'        nb::object _obj_{name} = kwargs["{name}"];')
            lines.append(f'        auto _nd_{name} = nb::cast<nb::ndarray<{ctype}, nb::device::cpu>>'
                         f'(_obj_{name}, false);')
            lines.append(f'        {ctype} *{name} = reinterpret_cast<{ctype} *>(_nd_{name}.data());')
        elif isinstance(desc, dt.Scalar):
            lines.append(f'        {ctype} {name} = nb::cast<{ctype}>(kwargs["{name}"]);')
        else:
            raise NotImplementedError(f'Nanobind interface: argument type {type(desc).__name__} '
                                      f'(argument "{name}") is not supported yet.')
    return '\n'.join(lines)


def generate_bindings_code(sdfg) -> str:
    """Returns the C++ source of the nanobind module for ``sdfg``."""
    from dace.codegen.targets.cpp import mangle_dace_state_struct_name

    name = sdfg.name
    state_t = mangle_dace_state_struct_name(name)
    arglist = sdfg.arglist()

    sig_decl = sdfg.signature(with_types=True, arglist=arglist)
    sig_call = sdfg.signature(with_types=False, for_call=True, arglist=arglist)
    init_decl = sdfg.init_signature()
    init_call = sdfg.init_signature(for_call=True)

    free_symbols = sorted(k for k in sdfg.used_symbols(all_symbols=False) if not k.startswith('__dace'))
    init_arglist = {k: v for k, v in arglist.items() if k in free_symbols}

    call_marshalling = _cpp_marshalling(arglist)
    init_marshalling = _cpp_marshalling(init_arglist)

    program_params = f'{state_t} *__state' + (f', {sig_decl}' if sig_decl else '')
    program_args = 'm_state' + (f', {sig_call}' if sig_call else '')

    return f'''// Auto-generated nanobind bindings for SDFG '{name}'.
#include <cstdint>
#include <stdexcept>

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

struct CompiledSDFGHandle {{
    {state_t} *m_state = nullptr;

    CompiledSDFGHandle() = default;
    CompiledSDFGHandle(const CompiledSDFGHandle &) = delete;
    CompiledSDFGHandle &operator=(const CompiledSDFGHandle &) = delete;
    ~CompiledSDFGHandle() {{ finalize(); }}

    void initialize(nb::kwargs kwargs) {{
        if (m_state) return;
{init_marshalling}
        m_state = __dace_init_{name}({init_call});
        if (!m_state) throw std::runtime_error("SDFG '{name}': __dace_init failed.");
    }}

    void finalize() {{
        if (m_state) {{
            __dace_exit_{name}(m_state);
            m_state = nullptr;
        }}
    }}

    void call(nb::kwargs kwargs) {{
        if (!m_state) initialize(kwargs);
{call_marshalling}
        {{
            nb::gil_scoped_release _nogil;
            __program_{name}({program_args});
        }}
    }}
}};

}} // namespace

NB_MODULE({name}, m) {{
    nb::class_<CompiledSDFGHandle>(m, "CompiledSDFGHandle")
        .def("initialize", &CompiledSDFGHandle::initialize)
        .def("finalize", &CompiledSDFGHandle::finalize)
        .def("__call__", &CompiledSDFGHandle::call)
        .def_prop_ro("state_pointer", [](CompiledSDFGHandle &h) {{
            return reinterpret_cast<std::uintptr_t>(h.m_state);
        }});
    m.def("make_compiled_sdfg", []() {{ return new CompiledSDFGHandle(); }});
}}
'''
