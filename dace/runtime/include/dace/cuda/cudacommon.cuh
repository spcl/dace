// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_CUDACOMMON_CUH
#define __DACE_CUDACOMMON_CUH

#include <cstddef>  // size_t

// The ONE place the two GPU runtimes are reconciled. A library node emits ``gpuMalloc`` /
// ``gpuMemcpyAsync`` / ``gpuError_t`` and compiles under either backend unchanged, which is what
// keeps an expansion from needing a second, hand-maintained AMD copy of itself.
#if defined(__HIPCC__) || defined(WITH_HIP)
// The precompiled runtime header is force-included ahead of the generated file's own includes.
#include <hip/hip_runtime.h>
typedef hipStream_t gpuStream_t;
typedef hipEvent_t gpuEvent_t;
typedef hipError_t gpuError_t;
#define gpuGetLastError hipGetLastError
#define gpuPeekAtLastError hipPeekAtLastError
#define gpuGetErrorString hipGetErrorString
#define gpuStreamSynchronize hipStreamSynchronize
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuEventSynchronize hipEventSynchronize
#define gpuSuccess hipSuccess
#define gpuErrorMemoryAllocation hipErrorOutOfMemory
#define gpuLaunchKernel hipLaunchKernel
#define gpuMalloc hipMalloc
#define gpuMallocAsync hipMallocAsync
#define gpuFree hipFree
#define gpuFreeAsync hipFreeAsync
#define gpuMemset hipMemset
#define gpuMemsetAsync hipMemsetAsync
#define gpuMemcpy hipMemcpy
#define gpuMemcpyAsync hipMemcpyAsync
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
// hipFreeHost is deprecated in favour of hipHostFree; the pair of gpuMallocHost.
#define gpuFreeHost hipHostFree
#else
// nvcc implies this, a host .cpp translation unit does not -- and the aliases below are consumed
// from both.
#include <cuda_runtime.h>
typedef cudaStream_t gpuStream_t;
typedef cudaEvent_t gpuEvent_t;
typedef cudaError_t gpuError_t;
#define gpuGetLastError cudaGetLastError
#define gpuPeekAtLastError cudaPeekAtLastError
#define gpuGetErrorString cudaGetErrorString
#define gpuStreamSynchronize cudaStreamSynchronize
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuEventSynchronize cudaEventSynchronize
#define gpuSuccess cudaSuccess
#define gpuErrorMemoryAllocation cudaErrorMemoryAllocation
#define gpuLaunchKernel cudaLaunchKernel
#define gpuMalloc cudaMalloc
#define gpuMallocAsync cudaMallocAsync
#define gpuFree cudaFree
#define gpuFreeAsync cudaFreeAsync
#define gpuMemset cudaMemset
#define gpuMemsetAsync cudaMemsetAsync
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyAsync cudaMemcpyAsync
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuFreeHost cudaFreeHost
#endif

// Pinned-host allocation, in the shape the code generator emits. ``cudaMallocHost`` is a TEMPLATE
// accepting any ``T**``; the HIP spelling takes a bare ``void**`` and is deprecated in favour of
// ``hipHostMalloc``, so emitting the vendor name directly fails to compile for any non-void
// pointer (measured on an ``int**`` index array). One wrapper gives both backends the same
// signature, so codegen names one function rather than branching.
template <typename T>
static inline gpuError_t gpuMallocHost(T **ptr, size_t size) {
#if defined(__HIPCC__) || defined(WITH_HIP)
  return hipHostMalloc(reinterpret_cast<void **>(ptr), size);
#else
  return cudaMallocHost(reinterpret_cast<void **>(ptr), size);
#endif
}

// The context guard covers the calls checked during __dace_init_cuda before the context has been
// constructed (the runtime warm-up allocation). The message is printed either way; only the
// recording needs a context to record into.
#define DACE_GPU_CHECK(err)                                               \
  do {                                                                    \
    gpuError_t errr = (err);                                              \
    if (errr != (gpuError_t)0) {                                          \
      printf("GPU runtime error at %s:%d: %s (%d)\n", __FILE__, __LINE__, \
             gpuGetErrorString(errr), errr);                              \
      if (__state->gpu_context) {                                         \
        __state->gpu_context->record_error(errr);                         \
      }                                                                   \
    }                                                                     \
  } while (0)

#define DACE_KERNEL_LAUNCH_CHECK(err, kernel_name, gdimx, gdimy, gdimz, bdimx, \
                                 bdimy, bdimz)                                 \
  do {                                                                         \
    if (err != (gpuError_t)0) {                                                \
      printf(                                                                  \
          "ERROR launching kernel %s: %s (%d). Grid dimensions: "              \
          "(%u, %u, %u); Block dimensions: (%u, %u, %u).\n",                   \
          kernel_name, gpuGetErrorString(err), (int)err,                       \
          (unsigned int)(gdimx), (unsigned int)(gdimy), (unsigned int)(gdimz), \
          (unsigned int)(bdimx), (unsigned int)(bdimy),                        \
          (unsigned int)(bdimz));                                              \
      __state->gpu_context->record_error(err);                                 \
    }                                                                          \
  } while (0)

namespace dace {
namespace cuda {
struct Context {
  int num_streams;
  int num_events;
  gpuStream_t *streams;
  gpuStream_t *internal_streams;
  gpuEvent_t *events;
  gpuError_t lasterror;
  Context(int nstreams, int nevents)
      : num_streams(nstreams), num_events(nevents), lasterror((gpuError_t)0) {
    streams = new gpuStream_t[nstreams];
    internal_streams = new gpuStream_t[nstreams];
    events = new gpuEvent_t[nevents];
  }
  ~Context() {
    delete[] streams;
    delete[] internal_streams;
    delete[] events;
  }
  // Keep the first error. One failure tends to produce more, and only the first names the call that
  // actually broke: a failed CUB size query leaves its workspace unsized, and the reduction that
  // then reads it reports a second, later error that describes a consequence.
  void record_error(gpuError_t err) {
    if (lasterror == (gpuError_t)0) {
      lasterror = err;
    }
  }
};

}  // namespace cuda
}  // namespace dace

#ifdef __CUDACC__
DACE_DFI dace::vec<float, 4> operator+(float f, dace::vec<float, 4> v) {
  dace::vec<float, 4> result;
  result.x = v.x + f;
  result.y = v.y + f;
  result.z = v.z + f;
  result.w = v.w + f;
  return result;
}

DACE_DFI dace::vec<float, 4> operator/(float f, dace::vec<float, 4> v) {
  dace::vec<float, 4> result;
  result.x = f / v.x;
  result.y = f / v.y;
  result.z = f / v.z;
  result.w = f / v.w;
  return result;
}

DACE_DFI dace::vec<float, 4> operator/(dace::vec<float, 4> v, float f) {
  dace::vec<float, 4> result;
  result.x = v.x / f;
  result.y = v.y / f;
  result.z = v.z / f;
  result.w = v.w / f;
  return result;
}

DACE_DFI dace::vec<float, 4> operator-(dace::vec<float, 4> v) {
  dace::vec<float, 4> result;
  result.x = -v.x;
  result.y = -v.y;
  result.z = -v.z;
  result.w = -v.w;
  return result;
}

DACE_DFI dace::vec<float, 4> operator-(float f, dace::vec<float, 4> v) {
  dace::vec<float, 4> result;
  result.x = f - v.x;
  result.y = f - v.y;
  result.z = f - v.z;
  result.w = f - v.w;
  return result;
}

DACE_DFI dace::vec<float, 4> operator-(dace::vec<float, 4> u,
                                       dace::vec<float, 4> v) {
  dace::vec<float, 4> result;
  result.x = u.x - v.x;
  result.y = u.y - v.y;
  result.z = u.z - v.z;
  result.w = u.w - v.w;
  return result;
}

DACE_DFI dace::vec<float, 4> operator*(float f, dace::vec<float, 4> v) {
  dace::vec<float, 4> result;
  result.x = v.x * f;
  result.y = v.y * f;
  result.z = v.z * f;
  result.w = v.w * f;
  return result;
}

DACE_DFI dace::vec<float, 4> operator*(dace::vec<float, 4> v, float f) {
  dace::vec<float, 4> result;
  result.x = v.x * f;
  result.y = v.y * f;
  result.z = v.z * f;
  result.w = v.w * f;
  return result;
}

namespace dace {
namespace math {

DACE_DFI dace::vec<float, 2> exp(dace::vec<float, 2> v) {
  dace::vec<float, 2> result;
  result.x = exp(v.x);
  result.y = exp(v.y);
  return result;
}

DACE_DFI dace::vec<float, 4> exp(dace::vec<float, 4> v) {
  dace::vec<float, 4> result;
  result.x = exp(v.x);
  result.y = exp(v.y);
  result.z = exp(v.z);
  result.w = exp(v.w);
  return result;
}

DACE_DFI dace::vec<float, 4> log(dace::vec<float, 4> v) {
  dace::vec<float, 4> result;
  result.x = log(v.x);
  result.y = log(v.y);
  result.z = log(v.z);
  result.w = log(v.w);
  return result;
}

DACE_DFI dace::vec<float, 4> log10(dace::vec<float, 4> v) {
  dace::vec<float, 4> result;
  result.x = log10(v.x);
  result.y = log10(v.y);
  result.z = log10(v.z);
  result.w = log10(v.w);
  return result;
}

DACE_DFI dace::vec<float, 4> tanh(dace::vec<float, 4> v) {
  dace::vec<float, 4> result;
  result.x = tanh(v.x);
  result.y = tanh(v.y);
  result.z = tanh(v.z);
  result.w = tanh(v.w);
  return result;
}

DACE_DFI dace::vec<float, 4> heaviside(const dace::vec<float, 4> &a) {
  dace::vec<float, 4> result;
  result.x = (a.x > 0) ? 1.0f : 0.0f;
  result.y = (a.y > 0) ? 1.0f : 0.0f;
  result.z = (a.z > 0) ? 1.0f : 0.0f;
  result.w = (a.w > 0) ? 1.0f : 0.0f;
  return result;
}
}  // namespace math
}  // namespace dace
using dace::math::exp;
using dace::math::heaviside;
using dace::math::log;
using dace::math::log10;
using dace::math::tanh;
#endif

#endif  // __DACE_CUDACOMMON_CUH
