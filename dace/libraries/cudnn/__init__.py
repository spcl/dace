from dace.library import register_library
from .conv2d import Conv2D
from .conv2dbackpropinput import Conv2DBackpropInput
from .conv2dbackpropfilter import Conv2DBackpropFilter
from .cudnn import cuDNN

register_library(__name__, "cudnn")
