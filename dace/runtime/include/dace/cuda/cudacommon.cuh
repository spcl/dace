// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_CUDACOMMON_CUH
#define __DACE_CUDACOMMON_CUH

#if defined(__HIPCC__) || defined(WITH_HIP)
typedef hipStream_t gpuStream_t;
typedef hipEvent_t gpuEvent_t;
typedef hipError_t gpuError_t;
#define gpuGetLastError hipGetLastError
#define gpuGetErrorString hipGetErrorString
#else
typedef cudaStream_t gpuStream_t;
typedef cudaEvent_t gpuEvent_t;
typedef cudaError_t gpuError_t;
#define gpuGetLastError cudaGetLastError
#define gpuGetErrorString cudaGetErrorString
#endif

#define DACE_GPU_CHECK(err)                                               \
  do {                                                                    \
    gpuError_t errr = (err);                                              \
    if (errr != (gpuError_t)0) {                                          \
      printf("GPU runtime error at %s:%d: %s (%d)\n", __FILE__, __LINE__, \
             gpuGetErrorString(err), errr);                               \
      throw;                                                              \
    }                                                                     \
  } while (0)

#define DACE_KERNEL_LAUNCH_CHECK(err, kernel_name, gdimx, gdimy, gdimz, bdimx, \
                                 bdimy, bdimz)                                 \
  do {                                                                         \
    if (err != decltype(err)(0)) {                                             \
      printf(                                                                  \
          "ERROR launching kernel %s: %s (%d). Grid dimensions: "              \
          "(%d, %d, %d); Block dimensions: (%d, %d, %d).\n",                   \
          kernel_name, gpuGetErrorString(err), (int)err, gdimx, gdimy, gdimz,  \
          bdimx, bdimy, bdimz);                                                \
      throw;                                                                   \
    }                                                                          \
  } while (0)

namespace dace {
namespace cuda {
struct Context {
  int num_streams;
  int num_events;
  gpuStream_t *streams;
  gpuEvent_t *events;
  Context(int nstreams, int nevents)
      : num_streams(nstreams), num_events(nevents) {
    streams = new gpuStream_t[nstreams];
    events = new gpuEvent_t[nevents];
  }
  ~Context() {
    delete[] streams;
    delete[] events;
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
