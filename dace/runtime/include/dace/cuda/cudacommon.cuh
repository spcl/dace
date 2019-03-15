#ifndef __DACE_CUDACOMMON_CUH
#define __DACE_CUDACOMMON_CUH

#define DACE_CUDA_CHECK(err) do {                                            \
    cudaError_t errr = (err);                                                \
    if(errr != (cudaError_t)0)                                               \
    {                                                                        \
        printf("CUDA ERROR at %s:%d, code: %d\n", __FILE__, __LINE__, errr); \
    }                                                                        \
} while(0)

namespace dace {
    namespace cuda {
        extern cudaStream_t __streams[];
        extern cudaEvent_t __events[];
    }  // namespace cuda
}  // namespace dace

#endif  // __DACE_CUDACOMMON_CUH
