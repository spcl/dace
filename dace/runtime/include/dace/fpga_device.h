// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#ifdef DACE_XILINX
#include "xilinx/device.h"
#endif

#ifdef DACE_INTELFPGA
#include "intel_fpga/device.h"
#endif

namespace dace {  namespace fpga {
  class Context;
}  // namespace fpga
}  // namespace dace
