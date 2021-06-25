// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#include <cuda_runtime.h>
#include <nccl.h>

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

class NcclHandle {
 public:
  NcclHandle() = default;
  NcclHandle(const int _nDev): nDev{_nDev}{
    nDevs = new int[nDev];
    for (int i = 0; i < nDev; i++)
        nDevs[i] = i;
    CheckNcclError(ncclGroupStart());
    CheckNcclError(ncclGetUniqueId(&id));
    for (int i = 0; i < nDev; i++){
        cudaSetDevice(nDevs[i]);
        ncclComm_t comm;
        ncclCommInitRank(&comm, nDev, id, nDevs[i]);
        ncclCommunicators.insert(nDevs[i], comm);
    }
    CheckNcclError(ncclGroupEnd());
  };
  NcclHandle(NcclHandle const&) = delete;
  
  ~NcclHandle() {
      delete ncclCommunicators;
  }  
  NcclHandle& operator=(NcclHandle const&) = delete;
  
  ncclUniqueId id;
  const int nDev;
  int *nDevs;
  std::unordered_map<int, ncclComm_t> ncclCommunicators;
}

}  // namespace nccl

}  // namespace dace