import dace.library

from . import torch_integration
from . import library
from . import python_frontend

dace.library.register_library(__name__, "autodiff")
