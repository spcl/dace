// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#include "acl/acl.h"

#define DACE_ACL_CHECK(x)                                               \
  do {                                                                  \
    aclError __ret = x;                                                 \
    if (__ret != ACL_ERROR_NONE) {                                      \
      std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret \
                << std::endl;                                           \
    }                                                                   \
  } while (0);

namespace dace {
namespace ascendc {
struct Context {
  int num_streams;
  int num_events;
  aclrtStream *streams;

  Context(int nstreams, int nevents)
      : num_streams(nstreams),
        num_events(nevents),
        streams(new aclrtStream[nstreams]) {}

  ~Context() { delete[] streams; }
};

}  // namespace ascendc
}  // namespace dace
