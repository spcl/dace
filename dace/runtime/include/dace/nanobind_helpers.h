// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
// Static nanobind helpers shared by every generated bindings module, so the
// generator emits per-program content only. The definitions here are
// deliberately IDENTICAL for all modules: nanobind's type registry is
// process-wide and keyed by type name, so one shared caster/traits definition
// is exactly what keeps arguments dispatching consistently across modules
// (per-program types, by contrast, live in the generated per-program
// namespace).
#pragma once

#include <Python.h>

#include <cstdint>    // uint8_t
#include <stdexcept>  // std::runtime_error
#include <string>     // std::string

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>  // dtype_traits, dlpack::dtype

#include <dace/types.h>  // dace::float16

namespace dace {
namespace nanobind_detail {

// Raise the error the generated code recorded for this SDFG, if any (the read
// clears the record, so a failure is delivered exactly once). `last_error` is
// the program's own `__dace_gpu_last_error` accessor and `error_string` the
// backend's error formatter, both handed in by the generated module - this
// header stays backend-agnostic and is compiled into CPU-only modules too.
template <typename StateT, typename GetLastError, typename ErrorString>
inline void check_gpu_error(StateT *state, bool enabled, GetLastError &&last_error,
                            ErrorString &&error_string, const char *program_name) {
    if (!enabled || !state) return;
    const int err = last_error(state);
    if (err != 0)
        throw std::runtime_error(
            std::string("An error was detected when calling \"") + program_name + "\": " +
            error_string(err) +
            ". Consider enabling synchronous debugging mode (environment "
            "variable: DACE_compiler_cuda_syncdebug=1) to see where the "
            "issue originates from.");
}

// Boolean scalar arguments need their own caster: nanobind's bool caster only
// ever accepts an exact Python bool, and the integer-caster detour this
// interface used to take (binding the parameter as uint8_t) broke when
// nanobind 2.14 narrowed integer conversion to the __index__ protocol just as
// numpy removed __index__ from numpy.bool_ (int() still works, index() does
// not). The caster restores the intended acceptance set - Python bool,
// numpy.bool_, and integer-like values (Python int, numpy integer scalars) -
// and nothing more: floats have no __index__ and keep being rejected. Kept as
// a caster rather than per-argument setup code so it slots uniformly into
// call() and initialize().
//
// The implicit bool conversion is load-bearing: a bool SYMBOL's binding
// parameter is passed by its raw name into init_impl (and the workspace
// methods), where the extern "C" signature takes a plain bool.
struct dace_bool {
    uint8_t value;
    operator bool() const { return value != 0; }
};

}  // namespace nanobind_detail
}  // namespace dace

namespace nanobind {
namespace detail {

// dtype_traits specialization advertising dace::float16 (= dace::half) as a
// 16-bit DLPack float, so nb::ndarray<dace::float16, ...> accepts a numpy/cupy
// float16 array. nanobind's own detection uses std::is_floating_point, false
// for the half struct.
template <> struct dtype_traits<dace::float16> {
    static constexpr dlpack::dtype value{
        (uint8_t) dlpack::dtype_code::Float, 16, 1
    };
    static constexpr auto name = const_name("float16");
};

template <> struct type_caster<dace::nanobind_detail::dace_bool> {
    NB_TYPE_CASTER(dace::nanobind_detail::dace_bool, const_name("bool"))
    bool from_python(handle src, uint8_t, cleanup_list *) noexcept {
        PyObject *o = src.ptr();
        if (o == Py_True) { value.value = 1; return true; }
        if (o == Py_False) { value.value = 0; return true; }
        // Integer-likes (Python int, numpy integer scalars) enter through the
        // __index__ protocol; floats have no __index__ and stay rejected.
        // Public CPython API only - nanobind's internal load_* helpers change
        // signature across major versions (2.x -> 3.x broke the build).
        if (PyObject *idx = PyNumber_Index(o)) {
            long long i = PyLong_AsLongLong(idx);
            Py_DECREF(idx);
            if (i == -1 && PyErr_Occurred()) { PyErr_Clear(); return false; }
            value.value = (uint8_t) (i != 0);
            return true;
        }
        PyErr_Clear();
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
    static handle from_cpp(dace::nanobind_detail::dace_bool src, rv_policy, cleanup_list *) noexcept {
        return handle(src.value ? Py_True : Py_False).inc_ref();
    }
};

}  // namespace detail
}  // namespace nanobind
