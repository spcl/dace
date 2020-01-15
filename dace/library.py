import dace
import inspect
from dace.graph.nodes import LibraryNode
from dace.transformation.pattern_matching import ExpandTransformation


# Use to decorate DaCe libraries
def library(lib):
    lib = dace.properties.make_properties(lib)
    if lib.__name__ in _DACE_REGISTERED_LIBRARIES:
        raise ValueError("Duplicate library found: " + lib.__name__)
    if not hasattr(lib, "nodes"):
        raise ValueError("DaCe library class must implement a "
                         "list of library nodes in the field \"nodes\".")
    if not hasattr(lib, "default_implementation"):
        raise ValueError("DaCe library class must implement the field "
                         "\"default_implementation\" (can be None).")
    if not hasattr(lib, "transformations"):
        raise ValueError(
            "DaCe library class must expose a "
            "list of transformations in the field \"transformations\".")
    # Go into every node and expansion associated with this library, and mark
    # them as belonging to this library, such that we can keep track of when a
    # library becomes a real dependency
    for node in lib.nodes:
        if not hasattr(node, "_dace_library_node"):
            raise ValueError(str(node) + " is not a DaCe library node.")
        node._dace_library_name = lib.__name__
        for trans in node.implementations.values():
            if not hasattr(trans, "_dace_library_expansion"):
                raise ValueError(
                    str(trans) + " is not a DaCe library expansion.")
            trans._dace_library_name = lib.__name__
    lib._dace_library = True
    _DACE_REGISTERED_LIBRARIES[lib.__name__] = lib
    return lib


# Use to decorate DaCe library nodes
def node(n):
    n = dace.properties.make_properties(n)
    if not issubclass(n, LibraryNode):
        raise TypeError("Library node class \"" + n.__name__ +
                        "\" must derive from dace.graph.nodes.LibraryNode")
    if not hasattr(n, "implementations"):
        raise ValueError("Library node class \"" + n.__name__ +
                         "\" must define implementations.")
    if not hasattr(n, "default_implementation"):
        raise ValueError(
            "Library node class \"" + n.__name__ +
            "\" must define default_implementation (can be None).")
    # Add the node type to all implementations for matching
    for Transformation in n.implementations.values():
        Transformation._match_node = n("__" + Transformation.__name__)
    n._dace_library_node = True
    return n


# Use to decorate DaCe library expansions
def expansion(exp):
    exp = dace.properties.make_properties(exp)
    if not issubclass(exp, ExpandTransformation):
        raise TypeError("Library node expansion \"" + type(n).__name__ +
                        "\"must derive from ExpandTransformation")
    if not hasattr(exp, "environments"):
        raise ValueError("Library node expansion must define environments "
                         "(can be an empty list).")
    for dep in exp.environments:
        if not hasattr(dep, "_dace_library_environment"):
            raise ValueError(str(dep) + " is not a DaCe library environment.")
    exp._dace_library_expansion = True
    return exp


# Use to decorate DaCe library environments
def environment(env):
    env = dace.properties.make_properties(env)
    if env.__name__ in _DACE_REGISTERED_ENVIRONMENTS:
        raise ValueError("Duplicate environment specification: " + env.__name__)
    for field in [
            "cmake_minimum_version",
            "cmake_packages",
            "cmake_variables",
            "cmake_includes",
            "cmake_libraries",
            "cmake_compile_flags",
            "cmake_link_flags",
            "cmake_files",
            "headers",
            "init_code",
            "finalize_code",
    ]:
        if not hasattr(env, field):
            raise ValueError(
                "DaCe environment specification must implement the field \"" +
                field + "\".")
    env._dace_library_environment = True
    # Retrieve which file this was called from
    caller_file = inspect.stack()[1].filename
    env._dace_file_path = caller_file
    _DACE_REGISTERED_ENVIRONMENTS[env.__name__] = env
    return env


# Mapping from string to DaCe environment
def get_environment(env_name):
    try:
        env = dace.library._DACE_REGISTERED_ENVIRONMENTS[env_name]
    except KeyError:
        raise KeyError("Undefined DaCe environment {}.".format(env_name))
    return env


# Mapping from string to dacelet
def get_library(lib_name):
    try:
        lib = dace.library._DACE_REGISTERED_LIBRARIES[lib_name]
    except KeyError:
        raise KeyError("Undefined DaCe library {}.".format(lib_name))
    return lib


_DACE_REGISTERED_LIBRARIES = {}
_DACE_REGISTERED_ENVIRONMENTS = {}
