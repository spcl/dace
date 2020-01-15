import dace.library
from dace.config import Config
from .nodes.dot import Dot

@dace.library.library
class BLAS:

   nodes = [Dot]
   transformations = []
   default_implementation = Config.get("experimental", "blas_implementation")

