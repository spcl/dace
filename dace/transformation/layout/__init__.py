# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Data-layout transformations (SC26 layout algebra: Pad, Permute, Block, Shuffle, Zip/Unzip)."""
from dace.transformation.layout.permute_dimensions import PermuteDimensions
from dace.transformation.layout.split_dimensions import SplitDimensions
from dace.transformation.layout.unblock_dimensions import UnblockDimensions
from dace.transformation.layout.pad_dimensions import PadDimensions
from dace.transformation.layout.block_aware_map_tiling import BlockAwareMapTiling
from dace.transformation.layout.normalize_schedule import NormalizeScheduleForLayout, normalize_schedule_for_layout
from dace.transformation.layout.timing import (InsertLayoutTiming, add_fusion_barrier, is_copy_state, time_compute,
                                               compute_region_timer)
from dace.transformation.layout.zip_arrays import ZipArrays
from dace.transformation.layout.unzip_arrays import UnzipArrays
from dace.transformation.layout.shuffle_elements import ShuffleElements
from dace.transformation.layout.rewrite_libnodes import (GemmToTensorDot, RewriteCopyForLayout, transform_einsum,
                                                         remap_contracted_axes, permute_reduce, block_scan_stride,
                                                         copy_permutation_axes)
from dace.transformation.layout.split_array import SplitArray
from dace.transformation.layout.brute_force import (sweep, best, time_cpu, permutation_candidates, block_candidates,
                                                    shuffle_candidates, indirection_candidates, SweepResult)
from dace.transformation.layout.indirect_access import (IndirectAccess, indirect_accesses, index_bindings,
                                                        resolve_index_source)
from dace.transformation.layout.prepare import prepare_for_layout
