// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_GPUCUB_CUH
#define __DACE_GPUCUB_CUH

// CUB and hipCUB expose the same device-wide primitives under the same signatures
// (DeviceReduce::Sum(void*, size_t&, InIt, OutIt, NumItemsT, stream) and friends), so a
// libnode's emitted code differs only in the namespace it names. Aliasing it here keeps the
// expansions backend-neutral instead of carrying a second copy of every CUB template.
#if defined(__HIPCC__) || defined(WITH_HIP)
#include <hipcub/hipcub.hpp>
namespace gpucub = hipcub;
#else
#include <cub/cub.cuh>
namespace gpucub = cub;
#endif

// CUB's error-check macro is named after its backend, so the one spelling a caller writes has to be
// selected here alongside the namespace.
#if defined(__HIPCC__) || defined(WITH_HIP)
#define DACE_GPUCUB_DEBUG(e) HipcubDebug(e)
#else
#define DACE_GPUCUB_DEBUG(e) CubDebug(e)
#endif

#endif  // __DACE_GPUCUB_CUH
