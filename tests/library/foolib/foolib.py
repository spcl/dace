from dace.graph.nodes import LibraryNode
from dace.transformation.pattern_matching import ExpandTransformation
import dace.library
import barlib

@dace.library.library
class FooLib:
    nodes = []
    transformations = []
    default_implementation = None

def herp():
    print("herp")
