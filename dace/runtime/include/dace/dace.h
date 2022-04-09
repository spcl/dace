// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_RUNTIME_H
#define __DACE_RUNTIME_H

// Necessary headers
#include <cstdio>
#include <cmath>
#include <numeric>
#include <tuple>
#include <cstring>

// The order in which these are included matters - sorting them
// alphabetically causes compilation to fail.
#include "types.h"
#include "vector.h"
#include "intset.h"
#include "math.h"
#include "complex.h"
#include "pyinterop.h"
#include "reduction.h"
#include "copy.h"
#include "stream.h"
#include "os.h"
#include "perf/reporting.h"
#include "comm.h"
#include "serialization.h"

#if defined(__CUDACC__) || defined(__HIPCC__)
#include "cuda/cudacommon.cuh"
#include "cuda/copy.cuh"
#include "cuda/dynmap.cuh"
#else
#include "cudainterop.h"
#endif

#ifdef DACE_XILINX
#include "xilinx/host.h"
#endif

#ifdef DACE_INTELFPGA
#include "intel_fpga/host.h"
#endif

#include "fpga_common.h"

#endif  // __DACE_RUNTIME_H
