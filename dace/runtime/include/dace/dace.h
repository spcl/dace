// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_RUNTIME_H
#define __DACE_RUNTIME_H

#ifdef DACE_ASCEND
#ifndef __CCE_KT_TEST__
#endif

// Necessary headers
#include <cstdio>
#include <cmath>
#include <numeric>
#include <tuple>
#include <cstring>
#include <cstdlib>

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
#ifndef DACE_ASCEND
#include "stream.h"
#endif
#include "os.h"
#ifndef DACE_ASCEND
#include "perf/reporting.h"
#endif
#include "comm.h"
#ifndef DACE_ASCEND
#include "serialization.h"
#endif

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

#ifdef DACE_ASCEND
#include "ascendc/ascendccommon.h"
#endif

#ifdef DACE_ASCEND
#endif
#endif

#endif  // __DACE_RUNTIME_H

