// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef  __DACE_NCCL_H
#define  __DACE_NCCL_H

#include <cuda_runtime.h>

#include <cstddef>    // size_t
#include <stdexcept>  // std::runtime_error
#include <string>     // std::to_string
#include <unordered_map>

#include "nccl.h"

#define DACE_NCCL_CHECK(err) do {                                            \
    ncclResult_t errr = (err);                                                \
    if(errr != (ncclResult_t)0)                                               \
    {                                                                        \
        printf("NCCL ERROR at %s:%d, code: %d\n", __FILE__, __LINE__, errr); \
    }                                                                        \
} while(0)

#endif  // __DACE_NCCL_H
