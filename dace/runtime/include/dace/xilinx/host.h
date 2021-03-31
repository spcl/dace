// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#include <vector>  // For concurrent kernel launches

#include "hlslib/xilinx/OpenCL.h"

#include <dace/fpga_host.h>  // Must be included after hlslib/xilinx/OpenCL.h
#include <dace/os.h>
#include <dace/types.h>
