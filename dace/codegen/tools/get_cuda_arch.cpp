// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <set>
#include <sstream>
#include <string>

int main() {
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

  if (architectures.empty()) {
    return 1;
  }

  std::copy(architectures.begin(), std::prev(architectures.end(), 1),
            std::ostream_iterator<std::string>(std::cout, ";"));
  std::cout << *architectures.rbegin();

  return 0;
}
