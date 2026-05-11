// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_MERGE_H
#define __DACE_MERGE_H

// Top-level (un-namespaced) so the codegen can emit ``merge(c, a, b)``
// alongside ``left_shift``, ``ROUND``, etc. — see ``math.h`` for the same
// convention.
template <typename T>
static DACE_CONSTEXPR DACE_HDFI T merge(bool c, T a, T b) { return c ? a : b; }

#endif  // __DACE_MERGE_H
