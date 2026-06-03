// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_ITE_H
#define __DACE_ITE_H

// ``ITE`` (If-Then-Else): canonical 3-input ternary blend across the
// pipeline. Templated to accept any pair of compatible arms (``int``
// literal vs ``int64_t`` symbol, ``double`` vs ``double``, ...). Top-level
// (un-namespaced) so the codegen can emit ``ITE(c, t, e)`` alongside
// ``left_shift``, ``ROUND``, etc. — see ``math.h`` for the same convention.
template <typename C, typename T, typename E>
static DACE_CONSTEXPR DACE_HDFI auto ITE(C c, T t, E e) -> decltype(c ? t : e) {
  return c ? t : e;
}

// ``merge`` is the legacy spelling of ``ITE``. Retained as a thin
// pointer-typed overload to keep existing C++ emission paths working
// while the pipeline is migrated to the new name.
template <typename T>
static DACE_CONSTEXPR DACE_HDFI T merge(bool c, T a, T b) {
  return c ? a : b;
}

#endif  // __DACE_ITE_H
