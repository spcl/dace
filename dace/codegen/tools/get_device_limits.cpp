// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
// Prints the first visible GPU's multiprocessor count, resident threads and resident blocks per multiprocessor.
#ifdef DACE_PROBE_HIP
#include <hip/hip_runtime.h>
#define GPU(name) hip##name
typedef hipDeviceProp_t DeviceProp;
#else
#include <cuda_runtime.h>
#define GPU(name) cuda##name
typedef cudaDeviceProp DeviceProp;
#endif

#include <cstdio>

int main() {
  DeviceProp prop;
  if (GPU(GetDeviceProperties)(&prop, 0) != GPU(Success)) return 1;
  std::printf("%d %d %d\n", prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor,
              prop.maxBlocksPerMultiProcessor);
  return 0;
}
