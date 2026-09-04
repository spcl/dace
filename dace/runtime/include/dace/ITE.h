// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_ITE_H
#define __DACE_ITE_H

#include <type_traits>

// Top-level (un-namespaced) so the codegen can emit ``ITE(c, a, b)``
// alongside ``left_shift``, ``ROUND``, etc. -- see ``math.h`` for the same
// convention. ``ITE(c, a, b)`` is the unified canonical name for the
// ternary blend, emitted by ASTSplitter (Python ternaries) and by the
// branch-normalization passes.
//
// The two arms may differ in type: a blend against a literal (``ITE(c, x, 0)``
// where ``x`` is ``double``) reaches codegen with an ``int`` arm, and a single
// ``T`` would fail template deduction (``double`` vs ``int``). Accept the arms
// independently and return their common type -- the ``c ? a : b`` body already
// applies the usual arithmetic conversions, so this only fixes deduction.
template <typename TA, typename TB>
static DACE_CONSTEXPR DACE_HDFI typename std::common_type<TA, TB>::type ITE(bool c, TA a, TB b) {
  return c ? a : b;
}

#endif  // __DACE_ITE_H
