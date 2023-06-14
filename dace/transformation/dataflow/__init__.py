# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" This module initializes the dataflow transformations package. """

# Map-related
from .mapreduce import MapReduceFusion, MapWCRFusion
from .map_expansion import MapExpansion
from .map_collapse import MapCollapse
from .map_for_loop import MapToForLoop
from .map_interchange import MapInterchange
from .map_dim_shuffle import MapDimShuffle
from .map_fusion import MapFusion
from .map_fission import MapFission
from .map_unroll import MapUnroll
from .trivial_map_elimination import TrivialMapElimination
from .trivial_map_range_elimination import TrivialMapRangeElimination
from .otf_map_fusion import OTFMapFusion

# Data movement
from .strip_mining import StripMining
from .tiling import MapTiling
from .tiling_with_overlap import MapTilingWithOverlap
from .buffer_tiling import BufferTiling
from .vectorization import Vectorization
from .copy_to_map import CopyToMap
from .gpu_grid_stride_tiling import GPUGridStridedTiling

# Data-related
from .stream_transient import StreamTransient, AccumulateTransient
from .local_storage import InLocalStorage, OutLocalStorage
from .double_buffering import DoubleBuffering
from .streaming_memory import StreamingMemory, StreamingComposition
from .reduce_expansion import ReduceExpansion

# Complexity reduction
from .dedup_access import DeduplicateAccess
from .redundant_array import (RedundantArray, RedundantSecondArray, SqueezeViewRemove, UnsqueezeViewRemove,
                              RedundantReadSlice, RedundantWriteSlice, RemoveSliceView, RemoveIntermediateWrite)
from .redundant_array_copying import (RedundantArrayCopyingIn, RedundantArrayCopying, RedundantArrayCopying2,
                                      RedundantArrayCopying3)
from .merge_arrays import InMergeArrays, OutMergeArrays, MergeSourceSinkArrays
from .prune_connectors import PruneConnectors, PruneSymbols
from .wcr_conversion import AugAssignToWCR
from .tasklet_fusion import TaskletFusion
from .trivial_tasklet_elimination import TrivialTaskletElimination

# Device-related
from .copy_to_device import CopyToDevice
from .gpu_transform import GPUTransformMap
from .gpu_transform_local_storage import GPUTransformLocalStorage
from .mpi import MPITransformMap
from .warp_tiling import WarpTiling
from .bank_split import BankSplit
from .hbm_transform import HbmTransform

# Algorithmic
from .matrix_product_transpose import MatrixProductTranspose
from .lift_einsum import LiftEinsum

# Distributions
from .map_distribution import (ElementWiseArrayOperation, ElementWiseArrayOperation2D, RedundantComm2D)