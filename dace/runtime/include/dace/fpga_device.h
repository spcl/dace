// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#ifdef DACE_XILINX
#include "xilinx/device.h"
#endif

#ifdef DACE_INTELFPGA
#include "intel_fpga/device.h"
#endif

// Defined as a struct rather than a class for C compatibility with OpenCL
// For definition, see fpga_host.h
struct dace_fpga_context;
