// Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#define STR(x) #x
#define XSTR(x) STR(x)

#pragma message("__DACE_USE_INTRINSICS = " XSTR(__DACE_USE_INTRINSICS))
#pragma message("__DACE_USE_SVE = " XSTR(__DACE_USE_SVE))
#pragma message("__ARM_FEATURE_SVE = " XSTR(__ARM_FEATURE_SVE))
#pragma message("__ARM_NEON = " XSTR(__ARM_NEON))


#if defined(__DACE_USE_INTRINSICS) && (__DACE_USE_INTRINSICS == 1)
    #if defined(__AVX512F__) && defined(__DACE_USE_AVX512) && (__DACE_USE_AVX512 == 1)
        #pragma message("Including AVX512 Intrinsics")
        #include "dace/cpu_vectorizable_math_avx512.h"
    #else
        #if defined(__DACE_USE_SVE) && (__DACE_USE_SVE == 1)
            #if defined(__ARM_FEATURE_SVE)
                #pragma message("Including ARM SVE")
                #include "dace/cpu_vectorizable_math_arm_sve.h"
            #else
                #pragma message("Including Scalar Fallback Intrinsics (SVE not defined)")
                #include "dace/cpu_vectorizable_math_scalar.h"
            #endif
        #else
            #if defined(__ARM_NEON)
                #pragma message("Including NEON Intrinsics")
                #include "dace/cpu_vectorizable_math_arm_neon.h"
            #else
                #pragma message("Including Scalar Fallback Intrinsics (NEON not defined)")
                #include "dace/cpu_vectorizable_math_scalar.h"
            #endif
        #endif
    #endif
#else

#pragma message("Including Scalar Fallback Intrinsics (DaCe Use Intrinsics not defined)")
#include "dace/cpu_vectorizable_math_scalar.h"

#endif
