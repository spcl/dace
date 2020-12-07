// Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#if !defined(DACE_XILINX) && !defined(DACE_INTELFPGA)
#error "Either DACE_XILINX or DACE_INTELFPGA must be defined."
#endif

#include <iostream>
#include <unordered_map>

namespace dace {

namespace fpga {

class Context {
 public:
  Context() = default;
  ~Context() = default;

  inline hlslib::ocl::Context &Get(int device_id = 0) {
    auto c = contexts_.find(device_id);
    if (c != contexts_.end()) {
      return c->second;
    } else {
      contexts_.emplace(device_id, device_id);
      return contexts_.at(device_id);
    }
  }

 private:
  // Don't allow copying or moving
  Context(Context const &) = delete;
  Context(Context &&) = delete;
  Context &operator=(Context const &) = delete;
  Context &operator=(Context &&) = delete;

  std::unordered_map<int, hlslib::ocl::Context> contexts_;
};

extern Context *_context;

}  // namespace fpga

}  // namespace dace

