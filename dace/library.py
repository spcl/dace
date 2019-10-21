import dace
from dace.graph.nodes import LibraryNode
from dace.transformation.pattern_matching import ExpandTransformation

# Use to decorate DaCe libraries
def library(lib):
    if not hasattr(lib, "nodes"):
        raise ValueError("DaCe library class must expose the field a "
                         "list of library nodes in the field \"nodes\".")
    if not hasattr(lib, "transformations"):
        raise ValueError(
            "DaCe library class must expose the field a "
            "list of transformations in the field \"transformations\".")
    _DACE_REGISTERED_LIBRARIES.append(lib)
    return lib

# Use to decorate DaCe library nodes
def node(n):
    if not (n, LibraryNode):
        raise TypeError("Library node class \"" + type(n).__name__ +
                        "\" must derive from dace.graph.nodes.LibraryNode")
    if not hasattr(n, "implementations"):
        raise ValueError("Library node class \"" + type(n).__name__ +
                        "\" must define implementations.")
    if not hasattr(n, "default_implementation"):
        raise ValueError("Library node class \"" + type(n).__name__ +
                        "\" must define default_implementation (can be None).")
    return dace.properties.make_properties(n)

# Use to decorate DaCe library transformations
def transformation(t):
    if not issubclass(t, ExpandTransformation):
        raise TypeError("Library node \"" + type(n).__name__ +
                        "\"must derive from dace.graph.nodes.LibraryNode")
    return dace.properties.make_properties(t)

_DACE_REGISTERED_LIBRARIES = []
