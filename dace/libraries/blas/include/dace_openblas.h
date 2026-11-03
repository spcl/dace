// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#include <omp.h>

#include <algorithm>
#include <vector>

// Weak, so a program linked against a BLAS without these OpenBLAS extensions (netlib through the distro
// alternatives) still links; their addresses are then null.
extern "C" {
int openblas_get_parallel(void);
void openblas_set_num_threads(int num_threads);
}
#pragma weak openblas_get_parallel
#pragma weak openblas_set_num_threads

namespace dace {

namespace blas {

// A pthreads OpenBLAS (openblas_get_parallel() == 1) runs its own thread pool beside libgomp's team, and a
// call between two OpenMP regions competes for the cores with the team's still-spinning workers:
// contour_integral's LU solves ran 12x slower that way than with OpenBLAS kept on one thread. DaCe
// parallelizes through OpenMP, so such a build is kept single-threaded; an OpenMP build shares the team.
inline void DeferOpenBlasThreadsToOpenMP() {
  if (openblas_get_parallel != nullptr && openblas_set_num_threads != nullptr && openblas_get_parallel() == 1) {
    openblas_set_num_threads(1);
  }
}

}  // namespace blas

}  // namespace dace
