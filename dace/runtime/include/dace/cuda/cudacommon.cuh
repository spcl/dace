// Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_CUDACOMMON_CUH
#define __DACE_CUDACOMMON_CUH

#if defined(__HIPCC__) || defined(WITH_HIP)
typedef hipStream_t gpuStream_t;
typedef hipEvent_t gpuEvent_t;

#define DACE_CUDA_CHECK(err) do {                                            \
    hipError_t errr = (err);                                                 \
    if(errr != (hipError_t)0)                                                \
    {                                                                        \
        printf("HIP ERROR at %s:%d, code: %d\n", __FILE__, __LINE__, errr);  \
    }                                                                        \
} while(0)

#else

typedef cudaStream_t gpuStream_t;
typedef cudaEvent_t gpuEvent_t;

#define DACE_CUDA_CHECK(err) do {                                            \
    cudaError_t errr = (err);                                                \
    if(errr != (cudaError_t)0)                                               \
    {                                                                        \
        printf("CUDA ERROR at %s:%d, code: %d\n", __FILE__, __LINE__, errr); \
    }                                                                        \
} while(0)
#endif

namespace dace {
    namespace cuda {
        extern gpuStream_t __streams[];
        extern gpuEvent_t __events[];
        extern int num_streams;
        extern int num_events;
    }  // namespace cuda
}  // namespace dace

#endif  // __DACE_CUDACOMMON_CUH
