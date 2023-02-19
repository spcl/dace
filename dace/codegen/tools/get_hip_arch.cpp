// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#include <hip/hip_runtime.h>

#include <iostream>
#include <set>
#include <string>

int main(int argc, char **argv) {
  int count;
  if (hipGetDeviceCount(&count) != hipSuccess) return 1;

  std::set<std::string> architectures;
  // Loop over all GPU architectures
  for (int i = 0; i < count; ++i) {
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, i) != hipSuccess ||
        (prop.major == 99 && prop.minor == 99))
      continue;

    // Find architecture, ignoring anything that succeeds a colon
    void *colon = memchr(prop.gcnArchName, ':', 256);
    if (colon)
      *(char *)colon = '\0';

    std::string str = prop.gcnArchName;
    architectures.insert(str);
  }

  // Print out architectures
  for (std::set<std::string>::iterator iter = architectures.begin();
       iter != architectures.end(); ++iter)
    std::cout << *iter << " ";

  return 0;
}
