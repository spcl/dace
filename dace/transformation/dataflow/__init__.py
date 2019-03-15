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
from .tiling import OrthogonalTiling
from .vectorization import Vectorization

# Data-related
from .stream_transient import StreamTransient, InLocalStorage, OutLocalStorage
from .reduce_expansion import ReduceExpansion

# Complexity reduction
from .redundant_array import RedundantArray
from .redundant_array_copying import (
    RedundantArrayCopying, RedundantArrayCopying2, RedundantArrayCopying3)

# Device-related
from .copy_to_device import CopyToDevice
from .gpu_transform import GPUTransformMap
from .gpu_transform_local_storage import GPUTransformLocalStorage
from .fpga_transform import FPGATransformMap
from .mpi import MPITransformMap
