// Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#include <vector>  // For concurrent kernel launches

#include "hlslib/xilinx/SDAccel.h"

#include <dace/fpga_host.h>  // Must be included after hlslib/xilinx/SDAccel.h
#include <dace/os.h>
#include <dace/types.h>
