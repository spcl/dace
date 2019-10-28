import dace.library
from .nodes.dot import Dot

@dace.library.library
class BLAS:

   nodes = [Dot]
   transformations = []
