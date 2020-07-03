#ifndef __DACE_CUDAREDUCE_CUH
#define __DACE_CUDAREDUCE_CUH

namespace dace {

template <typename T, typename BinaryOp, int size = sizeof(T)>
struct WarpAllreduce {
    static __forceinline__ __device__ T run(T v) {
        #pragma unroll
        for (int i = 1; i < warpSize; i = i * 2) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
            v = BinaryOp(v, __shfl_xor_sync(0xffffffff, v, i));
#elif defined(__CUDA_ARCH__) || defined(__HIPCC__)
            v = BinaryOp(v, __shfl_xor(v, i));
#endif
        }
        return v;
    }
};

}  // namespace dace

#endif  // __DACE_CUDAREDUCE_CUH
