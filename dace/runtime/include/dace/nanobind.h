// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_NANOBIND_H
#define __DACE_NANOBIND_H

// Umbrella header for the generated nanobind bindings: everything a bindings
// translation unit needs beyond its own generated code. When the nanobind
// Python package is available, the binary-header machinery precompiles this
// header alongside <dace/dace.h> (see codegen/compiler.py,
// prepare_precompiled_header), so keep it self-contained and free of
// program-specific content.

// Standard headers the generated code relies on, listed explicitly so the
// bindings never depend on transitive includes (GCC's libstdc++ leaks them,
// LLVM's libc++ does not):
#include <cstdint>    // std::uintptr_t
#include <optional>   // std::optional
#include <stdexcept>  // std::invalid_argument, std::runtime_error
#include <string>     // std::string

// DaCe runtime types used in the extern "C" program signature, the argument
// casts, and the nb::ndarray scalar types: dace::uint, dace::complex64/128
// (aliases of unsigned int / std::complex<...>), dace::vec<T, N>, and the
// `pyobject` typedef (pyinterop.h) that pyobject scalar arguments reference.
#include <dace/types.h>
#include <dace/vector.h>
#include <dace/pyinterop.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
// nanobind's STL type casters are opt-in per header: a std::complex,
// std::optional or std::string parameter binds only when the matching header
// is included -- without it nanobind reports "incompatible function
// arguments" at call time, not a compile error. Including the superset here
// is harmless for bindings that use none of them.
#include <nanobind/stl/complex.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

// Static helpers shared by every generated module (the dace_bool caster and
// the dace::float16 dtype_traits), so the generated TU carries per-program
// content only.
#include <dace/nanobind_helpers.h>

#endif  // __DACE_NANOBIND_H
