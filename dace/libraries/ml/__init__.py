# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
from dace.library import register_library
from .nodes import *
from .environments import *
from .transformations import *

register_library(__name__, "ml")

default_implementation = "pure"

FuseConvolutionActivation.fusable.add(Convolution)
