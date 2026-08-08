# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""CPU device-specialization rewrites: forms that bake in CPU scheduling and are
therefore NOT part of the device-neutral canonical output.
"""
from dace.transformation.passes.cpu_specialization.chunk_anti_dependence import ChunkAntiDependence
from dace.transformation.passes.cpu_specialization.sequentialize_parallel_scopes import SequentializeParallelScopes
from dace.transformation.passes.cpu_specialization.specialize_cpu_transfers import SpecializeCpuTransfers
