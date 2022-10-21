# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace.library import register_library
from .nodes import *
from .environments import *

register_library(__name__, "ttranspose")