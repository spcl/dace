# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from .stencil import *

dace.library.register_library(__name__, "stencil")
