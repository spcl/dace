from dace.library import register_library
from .environments import *
from .nodes import *
from .schema import *

register_library(__name__, "onnx")
