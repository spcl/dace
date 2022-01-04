# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import inspect
import sys
import types
from typing import Set, List
import contextlib
import networkx as nx
import types

import dace.properties
from dace.sdfg.nodes import LibraryNode, full_class_path
from dace.transformation.transformation import (Transformation, ExpandTransformation)


def register_implementation(implementation_name, expansion_cls, node_cls):
    """Associate a given library node expansion class with a library node
       class. This is done automatically for expansions defined in a DaCe
       library module, but this function can be used to add additional
       expansions from an external context."""
    if not issubclass(expansion_cls, ExpandTransformation):
        raise TypeError("Expected ExpandTransformation class, got: {}".format(type(node_cls).__name__))
    if not issubclass(node_cls, LibraryNode):
        raise TypeError("Expected LibraryNode class, got: {}".format(type(node_cls).__name__))
    if (hasattr(expansion_cls, "_dace_library_node") and expansion_cls._dace_library_node != node_cls):
        raise ValueError("Transformation {} is already registered with a "
                         "different library node: {}".format(expansion_cls.__name__, expansion_cls._dace_library_node))
    expansion_cls._dace_library_node = node_cls
    if implementation_name in node_cls.implementations:
        if node_cls.implementations[implementation_name] != expansion_cls:
            raise ValueError("Implementation {} registered with multiple expansions.".format(implementation_name))
    else:
        node_cls.implementations[implementation_name] = expansion_cls

    # Update library as necessary
    if hasattr(node_cls, "_dace_library_name"):
        expansion_cls._dace_library_name = node_cls._dace_library_name
        library = _DACE_REGISTERED_LIBRARIES[node_cls._dace_library_name]
        if expansion_cls not in library._dace_library_expansions:
            library._dace_library_expansions.append(expansion_cls)


def register_node(node_cls, library):
    """Associate a given library node class with a DaCe library. This is done
       automatically for library nodes defined in a DaCe library module,
       but this function can be used to add additional node classes from an
       external context."""
    if not issubclass(node_cls, LibraryNode):
        raise TypeError("Expected LibraryNode class, got: {}".format(type(node_cls).__name__))
    if not isinstance(library, types.ModuleType):
        raise TypeError("Expected Python module, got: {}".format(type(library).__name__))
    if not hasattr(node_cls, "_dace_library_node"):
        raise ValueError("Library node class {} must be decorated "
                         "with @dace.library.node.".format(node_cls.__name__))
    if (hasattr(node_cls, "_dace_library_name") and node_cls._dace_library_name != library.__name__):
        raise ValueError("Node class {} registered with multiple libraries: {} and {}".format(
            node_cls.__name__, node_cls._dace_library_name, library.__name__))
    if node_cls not in library._dace_library_nodes:
        library._dace_library_nodes.append(node_cls)
        node_cls._dace_library_name = library._dace_library_name
    for name, impl in node_cls.implementations.items():
        register_implementation(name, impl, node_cls)


def register_transformation(transformation_cls, library):
    """Associate a given transformation with a DaCe library. This is done
       automatically for transformations defined in a DaCe library module,
       but this function can be used to add additional transformations from an
       external context."""

    if not issubclass(transformation_cls, Transformation):
        raise TypeError("Expected Transformation, got: {}".format(transformation_cls.__name__))
    if not isinstance(library, types.ModuleType):
        raise TypeError("Expected Python module, got: {}".format(type(library).__name__))
    if (hasattr(transformation_cls, "_dace_library_name")
            and transformation_cls._dace_library_name != library.__name__):
        raise ValueError("Transformation class {} registered with multiple "
                         "libraries: {} and {}".format(transformation_cls.__name__,
                                                       transformation_cls._dace_library_name, library.__name__))
    if transformation_cls not in library._dace_library_transformations:
        library._dace_library_transformations.append(transformation_cls)
        transformation_cls._dace_library_name = library.__name__


def register_library(module_name, name):
    """Called from a library's __init__.py to register it with DaCe."""
    module = sys.modules[module_name]
    module._dace_library_name = name
    for attr in ["_dace_library_nodes", "_dace_library_transformations", "_dace_library_expansions"]:
        herp = hasattr(module, attr)
        if not hasattr(module, attr):
            setattr(module, attr, [])
    if not hasattr(module, "default_implementation"):
        module.default_implementation = None
    _DACE_REGISTERED_LIBRARIES[name] = module
    # Register content
    for key, value in module.__dict__.items():
        if isinstance(value, type):
            if issubclass(value, LibraryNode):
                register_node(value, module)
            elif issubclass(value, Transformation) and not issubclass(value, ExpandTransformation):
                register_transformation(value, module)


# Use to decorate DaCe library nodes
def node(n):
    n = dace.properties.make_properties(n)
    if not issubclass(n, LibraryNode):
        raise TypeError("Library node class \"" + n.__name__ + "\" must derive from dace.sdfg.nodes.LibraryNode")
    if not hasattr(n, "implementations"):
        raise ValueError("Library node class \"" + n.__name__ + "\" must define implementations.")
    if not hasattr(n, "default_implementation"):
        raise ValueError("Library node class \"" + n.__name__ + "\" must define default_implementation (can be None).")
    # Remove and re-register each implementation
    implementations = n.implementations
    n.implementations = type(n.implementations)()
    for name, transformation_type in implementations.items():
        n.register_implementation(name, transformation_type)
    n._dace_library_node = True
    return n


# Use to decorate DaCe library expansions
def expansion(exp):
    exp = dace.properties.make_properties(exp)
    if not issubclass(exp, ExpandTransformation):
        raise TypeError("Library node expansion \"" + exp.__name__ + "\" must derive from ExpandTransformation")
    if not hasattr(exp, "environments"):
        raise ValueError("Library node expansion must define environments " "(can be an empty list).")
    for dep in exp.environments:
        if not hasattr(dep, "_dace_library_environment"):
            raise ValueError(str(dep) + " is not a DaCe library environment.")
    exp._dace_library_expansion = True
    return exp


def register_expansion(library_node: LibraryNode, expansion_name: str):
    """ Defines and registers an expansion. """
    def expander(exp: ExpandTransformation):
        result = expansion(exp)
        library_node.register_implementation(expansion_name, exp)
        return result

    return expander


# Use to decorate DaCe library environments
def environment(env):
    env = dace.properties.make_properties(env)
    for field in [
            "cmake_minimum_version", "cmake_packages", "cmake_variables", "cmake_includes", "cmake_libraries",
            "cmake_compile_flags", "cmake_link_flags", "cmake_files", "headers", "state_fields", "init_code",
            "finalize_code", "dependencies"
    ]:
        if not hasattr(env, field):
            raise ValueError("DaCe environment specification must implement the field \"" + field + "\".")
    env._dace_library_environment = True
    # Retrieve which file this was called from
    caller_file = inspect.getmodule(env).__file__
    env._dace_file_path = caller_file
    env.full_class_path = types.MethodType(full_class_path, env)
    _DACE_REGISTERED_ENVIRONMENTS[env.full_class_path()] = env
    return env


def get_environments_and_dependencies(names: Set[str]) -> List:
    """ Get the environment objects from names. Also resolve the dependencies.

        :names: set of environment names.
        :return: a list of environment objects, ordered such that environments with dependencies appear after their
                 dependencies.
    """

    # get all environments: add dependencies until no new dependencies are found
    environments = {get_environment(name) for name in names}
    while True:
        added = {dep for env in environments for dep in env.dependencies if dep not in environments}
        if len(added) == 0:
            break
        environments = environments.union(added)

    # construct dependency graph
    dep_graph = nx.DiGraph()
    dep_graph.add_nodes_from(environments)

    for env in environments:
        for dep in env.dependencies:
            dep_graph.add_edge(dep, env)

    try:
        return list(nx.topological_sort(dep_graph))
    except nx.NetworkXUnfeasible:
        raise ValueError("Detected cycle in dependency graph.")


# Mapping from string to DaCe environment
def get_environment(env_name):
    try:
        env = dace.library._DACE_REGISTERED_ENVIRONMENTS[env_name]
    except KeyError:
        raise KeyError("Undefined DaCe environment {}.".format(env_name))
    return env


# Mapping from string to library
def get_library(lib_name):
    try:
        lib = dace.library._DACE_REGISTERED_LIBRARIES[lib_name]
    except KeyError:
        raise KeyError("Undefined DaCe library {}.".format(lib_name))
    return lib


_DACE_REGISTERED_LIBRARIES = {}
_DACE_REGISTERED_ENVIRONMENTS = {}


@contextlib.contextmanager
def change_default(library, implementation):
    old_default = library.default_implementation
    library.default_implementation = implementation
    yield
    library.default_implementation = old_default
