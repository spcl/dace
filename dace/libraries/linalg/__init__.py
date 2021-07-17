# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace.library import register_library
from .nodes import *

register_library(__name__, "linalg")
