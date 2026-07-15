// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_RUNTIME_H
#define __DACE_RUNTIME_H

// Necessary headers
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <tuple>

// The order in which these are included matters - sorting them
// alphabetically causes compilation to fail.
#include "alloc.h"
#include "comm.h"
#include "complex.h"
#include "copy.h"
#include "intset.h"
#include "math.h"
#include "os.h"
#include "perf/reporting.h"
#include "pyinterop.h"
#include "reduction.h"
#include "serialization.h"
#include "stream.h"
#include "types.h"
#include "vector.h"

#if defined(__CUDACC__) || defined(__HIPCC__)
#include "cuda/copy.cuh"
#include "cuda/cudacommon.cuh"
#include "cuda/dynmap.cuh"
#else
#include "cudainterop.h"
#endif

#endif  // __DACE_RUNTIME_H
