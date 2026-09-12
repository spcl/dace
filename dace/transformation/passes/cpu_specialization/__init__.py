# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""CPU device-specialization rewrites: forms that bake in CPU scheduling and are
therefore NOT part of the device-neutral canonical output.

:func:`~dace.transformation.passes.cpu_specialization.pipeline.cpu_specialize` runs the whole band
in order; it is a separate stage from canonicalization and runs after it.
"""
from dace.transformation.passes.cpu_specialization.calibrate_thresholds import CalibrateCpuThresholds
from dace.transformation.passes.cpu_specialization.chunk_anti_dependence import ChunkAntiDependence
from dace.transformation.passes.cpu_specialization.hoist_parallel_region import HoistParallelRegion
from dace.transformation.passes.cpu_specialization.recompute_oversized_intermediates import (
    RecomputeOversizedIntermediates)
from dace.transformation.passes.cpu_specialization.sequentialize_unprofitable_parallel_scopes import SequentializeUnprofitableParallelScopes
from dace.transformation.passes.cpu_specialization.specialize_cpu_transfers import SpecializeCpuTransfers
from dace.transformation.passes.cpu_specialization.pipeline import cpu_specialize
