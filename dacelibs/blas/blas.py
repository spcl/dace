from dace.library import library
from .nodes.dot import Dot, ExpandDotOpenBLAS

@library
class BLAS:

   nodes = [Dot]
   transformations = [ExpandDotOpenBLAS]
