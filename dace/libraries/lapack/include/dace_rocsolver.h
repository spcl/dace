// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

// rocSOLVER is built ON rocBLAS: it takes a rocblas_handle, reports a rocblas_status, and names its
// enums (rocblas_fill, rocblas_operation) from there. So the handle, the error check and the
// constants all come from the rocBLAS header rather than being restated here.
#include "../../blas/include/dace_rocblas.h"

#include <rocsolver/rocsolver.h>

namespace dace {

namespace lapack {

// rocSOLVER returns a rocblas_status, so the rocBLAS check is the right one; this alias exists so a
// libnode expansion can name a lapack-side symbol and not reach across into blas::.
inline void CheckRocsolverError(rocblas_status const& status) {
  dace::blas::CheckRocblasError(status);
}

}  // namespace lapack

}  // namespace dace
