from dace.library import register_library
from .nodes import *
from .environments import *

register_library(__name__, "blas")

