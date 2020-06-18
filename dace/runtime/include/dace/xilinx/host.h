#pragma once

#include <vector>  // For concurrent kernel launches

#include "hlslib/xilinx/SDAccel.h"

#include <dace/fpga_host.h>  // Must be included after hlslib/xilinx/SDAccel.h
#include <dace/os.h>
#include <dace/types.h>
