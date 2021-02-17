# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Specialized exception classes for code generator. """


class DuplicateDLLError(Exception):
    """ An exception that is raised whenever a library is loaded twice. """
    pass


class CompilerConfigurationError(Exception):
    """ An exception that is raised whenever CMake encounters a configuration
        error. """
    pass


class CompilationError(Exception):
    """ An exception that is raised whenever a compilation error occurs. """
    pass


class CodegenError(Exception):
    """ An exception that is raised within SDFG code generation. """
    pass
