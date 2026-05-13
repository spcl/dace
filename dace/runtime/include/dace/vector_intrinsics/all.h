// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
//
// Convenience aggregate that pulls in every per-op vector intrinsic header.
//
// During the H1..H12 migration this header progressively absorbs ops as they
// move from the legacy ``cpu_vectorizable_math_<arch>.h`` monoliths into
// per-op files under ``vector_intrinsics/``. Once H12 lands and the four
// monoliths are deleted, ``cpu_vectorizable_math.h`` reduces to a single
// ``#include "dace/vector_intrinsics/all.h"`` line.

#pragma once

#include "dace/vector_intrinsics/_common.h"

// Pre-existing per-op intrinsics (already in this layout from earlier work).
#include "dace/vector_intrinsics/break_safe_mask.h"
#include "dace/vector_intrinsics/gather.h"
#include "dace/vector_intrinsics/multiplex.h"
#include "dace/vector_intrinsics/scatter.h"
#include "dace/vector_intrinsics/shift_left.h"
#include "dace/vector_intrinsics/strided_load.h"
#include "dace/vector_intrinsics/strided_store.h"
