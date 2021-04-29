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

DACE_HDFI float4 exp(float4 v) {
    float4 result;
    result.x = exp(v.x);
    result.y = exp(v.y);
    result.z = exp(v.z);
    result.w = exp(v.w);
    return result;
}

DACE_HDFI float4 operator+(float f, float4 v) {
    return make_float4(v.x + f, v.y + f, v.z + f, v.w + f);
}

DACE_HDFI float4 operator*(float4 u, float4 v) {
    return make_float4(u.x * v.x, u.y * v.y, u.z * v.z, u.w * v.w);
}

DACE_HDFI float4 log(float4 v) {
    float4 result;
    result.x = log(v.x);
    result.y = log(v.y);
    result.z = log(v.z);
    result.w = log(v.w);
    return result;
}

DACE_HDFI float4 tanh(float4 v) {
    float4 result;
    result.x = tanh(v.x);
    result.y = tanh(v.y);
    result.z = tanh(v.z);
    result.w = tanh(v.w);
    return result;
}

#endif  // __DACE_RUNTIME_H
