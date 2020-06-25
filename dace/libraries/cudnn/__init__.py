from dace.library import register_library
from .conv2d import Conv2D
from .cudnn import cuDNN

register_library(__name__, "cudnn")
