""" This module initializes the dataflow transformations package. """

# Map-related
from .mapreduce import MapReduceFusion
from .map_expansion import MapExpansion
from .map_collapse import MapCollapse
from .map_for_loop import MapToForLoop
from .map_interchange import MapInterchange
from .map_fusion import MapFusion

# Data movement
from .strip_mining import StripMining
from .tiling import MapTiling
from .vectorization import Vectorization

# Data-related
from .stream_transient import StreamTransient
from .local_storage import InLocalStorage, OutLocalStorage
from .reduce_expansion import ReduceExpansion
from .double_buffering import DoubleBuffering

# Complexity reduction
from .redundant_array import RedundantArray
from .redundant_array_copying import (
    RedundantArrayCopying, RedundantArrayCopying2, RedundantArrayCopying3)
from .merge_arrays import MergeArrays

# TensorFlow-specific transformations
from .tensorflow_redundant_array import TensorflowRedundantArray

# Device-related
from .copy_to_device import CopyToDevice
from .gpu_transform import GPUTransformMap
from .gpu_transform_local_storage import GPUTransformLocalStorage
from .mpi import MPITransformMap
