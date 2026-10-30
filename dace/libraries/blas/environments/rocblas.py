# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from dace.libraries.blas.environments.hipblas import hipBLAS

#: The former rocBLAS environment; the HIP BLAS expansions call hipBLAS, which runs on rocBLAS.
rocBLAS = hipBLAS
