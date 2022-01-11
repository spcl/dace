// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#if !defined(DACE_XILINX) && !defined(DACE_INTELFPGA)
#error "Either DACE_XILINX or DACE_INTELFPGA must be defined."
#endif

#include <iostream>
#include <unordered_map>

struct dace_fpga_context {
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
  dace_fpga_context(dace_fpga_context const &) = delete;
  dace_fpga_context(dace_fpga_context &&) = delete;
  dace_fpga_context &operator=(dace_fpga_context const &) = delete;
  dace_fpga_context &operator=(dace_fpga_context &&) = delete;

  std::unordered_map<int, hlslib::ocl::Context> contexts_;
};
