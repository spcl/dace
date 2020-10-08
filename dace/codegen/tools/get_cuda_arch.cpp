// Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
#include <cuda_runtime.h>

#include <iostream>
#include <set>
#include <sstream>
#include <string>

int main(int argc, char **argv) {
  int count;
  if (cudaGetDeviceCount(&count) != cudaSuccess) return 1;

  std::set<std::string> architectures;
  // Loop over all GPU architectures
  for (int i = 0; i < count; ++i) {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, i) != cudaSuccess ||
        (prop.major == 99 && prop.minor == 99))
      continue;
    std::stringstream ss;
    ss << prop.major << prop.minor;
    architectures.insert(ss.str());
  }

  // Print out architectures
  for (std::set<std::string>::iterator iter = architectures.begin();
       iter != architectures.end(); ++iter)
    std::cout << *iter << " ";

  return 0;
}
