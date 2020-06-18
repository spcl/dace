#pragma once

#include <vector>  // For concurrent kernel launches

#include "hlslib/intel/OpenCL.h"

#include <dace/fpga_host.h>  // Must be included after hlslib/intel/OpenCL.h
#include <dace/os.h>
#include <dace/types.h>
