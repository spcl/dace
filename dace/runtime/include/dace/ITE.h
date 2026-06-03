// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_ITE_H
#define __DACE_ITE_H

// Top-level (un-namespaced) so the codegen can emit ``ITE(c, a, b)``
// alongside ``left_shift``, ``ROUND``, etc. -- see ``math.h`` for the same
// convention. ``ITE(c, a, b)`` is the unified canonical name for the
// ternary blend, emitted by ASTSplitter (Python ternaries) and by the
// branch-normalization passes.
template <typename T>
static DACE_CONSTEXPR DACE_HDFI T ITE(bool c, T a, T b) {
  return c ? a : b;
}

#endif  // __DACE_ITE_H
