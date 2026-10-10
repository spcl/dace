// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_ITE_H
#define __DACE_ITE_H

#include <type_traits>

// ``ITE(c, a, b)``: top-level ternary blend emitted by codegen. The arms may differ in type; the result is
// their common type.
template <typename TA, typename TB>
static DACE_CONSTEXPR DACE_HDFI typename std::common_type<TA, TB>::type ITE(bool c, TA a, TB b) {
  return c ? a : b;
}

#endif  // __DACE_ITE_H
