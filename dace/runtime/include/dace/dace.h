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


DACE_HDFI dace::vec<float, 4> exp(dace::vec<float, 4> v) {
    dace::vec<float, 4> result;
    result.x = exp(v.x);
    result.y = exp(v.y);
    result.z = exp(v.z);
    result.w = exp(v.w);
    return result;
}

DACE_HDFI dace::vec<float, 4> operator+(float f, dace::vec<float, 4> v) {
    dace::vec<float, 4> result;
    result.x = v.x + f;
    result.y = v.y + f;
    result.z = v.z + f;
    result.w = v.w + f;
    return result;
}

DACE_HDFI dace::vec<float, 4> operator*(dace::vec<float, 4> v, float f) {
    dace::vec<float, 4> result;
    result.x = v.x * f;
    result.y = v.y * f;
    result.z = v.z * f;
    result.w = v.w * f;
    return result;
}


DACE_HDFI dace::vec<float, 4> log(dace::vec<float, 4> v) {
    dace::vec<float, 4> result;
    result.x = log(v.x);
    result.y = log(v.y);
    result.z = log(v.z);
    result.w = log(v.w);
    return result;
}

DACE_HDFI dace::vec<float, 4> tanh(dace::vec<float, 4> v) {
    dace::vec<float, 4> result;
    result.x = tanh(v.x);
    result.y = tanh(v.y);
    result.z = tanh(v.z);
    result.w = tanh(v.w);
    return result;
}

#endif  // __DACE_RUNTIME_H
