// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#include <cuda_runtime.h>
#include "nccl.h"

#include <cstddef>    // size_t
#include <stdexcept>  // std::runtime_error
#include <string>     // std::to_string
#include <unordered_map>


namespace dace {
namespace nccl {

static void CheckNcclError(ncclResult_t const& status) {
    if (status != ncclSuccess) {
        throw std::runtime_error("nccl failed with error code: " + 
                                 std::to_string(status));
    }
}

}  // namespace nccl

}  // namespace dace