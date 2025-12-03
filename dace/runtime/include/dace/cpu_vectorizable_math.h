// Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#if defined(__DACE_USE_INTRINSICS)
    #if defined(__AVX512F__)
        #include "dace/cpu_vectorizable_math_avx512.h"
    #else
        #if defined(__DACE_USE_SVE)
            #if defined(__ARM_FEATURE_SVE2)
                #include "dace/cpu_vectorizable_math_arm_sve.h"
            #else
                #include "dace/cpu_vectorizable_math_scalar.h"
            #endif
        #else
            #if defined(__ARM_NEON)
                #include "dace/cpu_vectorizable_math_arm_neon.h"
            #else
                #include "dace/cpu_vectorizable_math_scalar.h"
            #endif
        #endif
    #endif
#else

#include "dace/cpu_vectorizable_math_scalar.h"

#endif