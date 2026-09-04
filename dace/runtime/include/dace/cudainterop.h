// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_CUDAINTEROP_H
#define __DACE_CUDAINTEROP_H

#ifdef WITH_CUDA
// The backend runtime header and every gpu* alias come from cudacommon.cuh, which is the one place
// the two runtimes are reconciled; this file used to carry a second copy of four of them.
#include "cuda/cudacommon.cuh"
#endif  // WITH_CUDA

#endif  // __DACE_CUDAINTEROP_H
