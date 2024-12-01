// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#ifndef __CCE_KT_TEST__
#include "acl/acl.h"

#ifdef __CCE_KT_TEST__
#include <iostream>
#endif

#define GM_HALF __gm__ dace::float16* __restrict__
#define GM_FLOAT __gm__ dace::float32* __restrict__

#ifdef __CCE_KT_TEST__
#define DACE_ACL_CHECK(x)                                               \
  do {                                                                  \
    aclError __ret = x;                                                 \
    if (__ret != ACL_ERROR_NONE) {                                      \
      std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret \
                << std::endl;                                           \
    }                                                                   \
  } while (0);
#else
#define DACE_ACL_CHECK(x)                                               \
  do {                                                                  \
    aclError __ret = x;                                                 \
    if (__ret != ACL_ERROR_NONE) {                                      \
     /* "TODO, find a way to print" */                                  \
    }                                                                   \
  } while (0);
#endif

namespace dace {
namespace ascendc {
struct Context {
  int num_streams;
  int num_events;
  aclrtStream *streams;
  aclrtStream *internal_streams;
  aclrtContext aclrt_context;

  Context(int nstreams, int nevents)
      : num_streams(nstreams),
        num_events(nevents),
        streams(new aclrtStream[nstreams]),
        internal_streams(new aclrtStream[nstreams]),
        aclrt_context() {}

  ~Context() { delete[] streams; }
};

}  // namespace ascendc
}  // namespace dace
#endif