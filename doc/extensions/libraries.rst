.. _libraries:

Libraries and Environments
==========================

A *library* in DaCe is a collection of :ref:`library node types <libnodes>`, their
expansions (implementations), library-specific transformations, and *environments*
that describe how to compile and link the resulting code against external
dependencies (e.g., BLAS, MPI, cuBLAS). Libraries are the primary mechanism
through which DaCe integrates high-performance third-party code into generated
SDFGs while keeping the IR portable and target-independent.

Libraries live under :mod:`dace.libraries` and are registered with DaCe at import
time. Each library is a regular Python package whose ``__init__.py`` calls
:func:`dace.library.register_library`. External libraries can be developed in
their own Python packages and registered the same way - they do not need to
live inside the DaCe source tree.

Defining a Library
------------------

The minimal skeleton of a library package looks like this::

    mylib/
      __init__.py
      nodes/
        __init__.py
        my_node.py
      environments/
        __init__.py
        my_env.py

The package's ``__init__.py`` imports the library nodes and environments and
then registers itself with DaCe:

.. code-block:: python

    # mylib/__init__.py
    from dace.library import register_library
    from .nodes import *
    from .environments import *

    register_library(__name__, "mylib")

    # Optional: set a default expansion implementation for all nodes in this
    # library. Users can override this at runtime by assigning to
    # ``mylib.default_implementation``.
    default_implementation = "pure"

Once registered, the library can be discovered via
:func:`dace.library.get_library` and any of its :ref:`library nodes <libnodes>`
can be expanded using one of its registered implementations.

Library Nodes and Expansions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Library nodes are defined by subclassing :class:`~dace.sdfg.nodes.LibraryNode`
and decorating the class with :func:`@dace.library.node <dace.library.node>`.
Each node declares an ``implementations`` dictionary that maps an
implementation name (string) to an
:class:`~dace.transformation.transformation.ExpandTransformation` subclass, and
a ``default_implementation`` attribute. Expansions are decorated with
:func:`@dace.library.expansion <dace.library.expansion>` (or registered with
:func:`@dace.library.register_expansion <dace.library.register_expansion>`),
and must declare an ``environments`` list that names the environments the
expansion's generated code depends on.

See :ref:`libnodes` for a full example (the ``Einsum`` library node) and a
description of how expansion is performed during compilation.

The ``"pure"`` Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By convention, most built-in library nodes provide an expansion named
``"pure"``. A *pure* implementation is a native SDFG expansion that lowers the
library node into ordinary DaCe constructs (maps, tasklets, nested SDFGs) and
declares no external environments. Because it relies only on the DaCe runtime
and the target's standard compiler, the ``"pure"`` implementation is the most
portable - it works on every backend, requires no third-party libraries, and
can be transformed and analyzed like any other SDFG subgraph - but it is
typically slower than vendor-tuned alternatives such as ``"MKL"``,
``"cuBLAS"``, or ``"OpenBLAS"``. When defining a new library node, providing a
``"pure"`` expansion is recommended so that the node always has a working
fallback regardless of the user's environment.

Environments
------------

An *environment* is a declarative description of an external dependency that
generated code may rely on. Environments tell the DaCe code generator and the
CMake build system which packages to find, which headers to include, which
libraries to link against, and which initialization or finalization code to
emit. They isolate platform- and library-specific build logic from the
expansion code, so the same expansion can target multiple backends simply by
changing its ``environments`` list.

An environment is a Python class decorated with
:func:`@dace.library.environment <dace.library.environment>`. The decorator
verifies that the class implements every required field listed below and
registers it in DaCe's global environment registry, where it can be looked up
by fully qualified class path with :func:`dace.library.get_environment`.

Each environment **must** define every one of the following attributes (use an
empty list, empty dict, or empty string when a field does not apply):

* ``cmake_minimum_version`` - minimum required CMake version, or ``None``.
* ``cmake_packages`` - list of CMake packages to ``find_package``
  (e.g., ``["BLAS"]``, ``["MPI"]``).
* ``cmake_variables`` - dict of CMake variables to set before package lookup
  (e.g., ``{"BLA_VENDOR": "Intel10_64lp"}``).
* ``cmake_includes`` - list of additional include directories.
* ``cmake_libraries`` - list of libraries to link against.
* ``cmake_compile_flags`` - list of extra compile flags.
* ``cmake_link_flags`` - list of extra linker flags.
* ``cmake_files`` - list of additional CMake files (without the ``.cmake``
  suffix) to be included by the generated build script. See
  :ref:`custom-cmake` below.
* ``headers`` - list of C/C++ headers that expansions of this environment
  emit ``#include`` directives for. May be a flat list, or a dict keyed by
  code-generator target (e.g., ``{'frame': [...], 'cuda': [...]}``) when
  different headers are needed per backend.
* ``state_fields`` - list of C/C++ field declarations to add to the generated
  SDFG runtime state struct (e.g., ``["cublasHandle_t cublas_handle;"]``).
* ``init_code`` - C/C++ snippet emitted during program initialization
  (typically used to populate ``state_fields``).
* ``finalize_code`` - C/C++ snippet emitted during program teardown.
* ``dependencies`` - list of other environment classes that this environment
  depends on; DaCe topologically sorts these when resolving the full
  dependency graph for a compiled SDFG.

Any of the ``cmake_*`` and ``headers`` attributes can also be defined as
``@staticmethod``\ s instead of plain class attributes. This is useful when
their values must be computed at runtime, for example to locate a vendor SDK
via environment variables. The Intel MKL environment uses this pattern to
locate ``mkl.h`` and ``libmkl_rt`` from ``MKLROOT`` or a Conda prefix:

.. code-block:: python

    import os
    import dace.library

    @dace.library.environment
    class IntelMKL:
        """Intel Math Kernel Library environment."""

        cmake_minimum_version = None
        cmake_packages = []
        cmake_variables = {"BLA_VENDOR": "Intel10_64lp"}
        cmake_compile_flags = []
        cmake_link_flags = []
        cmake_files = []

        headers = ["mkl.h", "../include/dace_blas.h"]
        state_fields = []
        init_code = ""
        finalize_code = ""
        dependencies = []

        @staticmethod
        def cmake_includes():
            if 'MKLROOT' in os.environ:
                return [os.path.join(os.environ['MKLROOT'], 'include')]
            return []

        @staticmethod
        def cmake_libraries():
            ...

An expansion opts into an environment by listing it on its
``environments`` attribute:

.. code-block:: python

    from dace.libraries.blas.environments import IntelMKL

    @dace.library.register_expansion(Gemm, "MKL")
    class ExpandGemmMKL(xf.ExpandTransformation):
        environments = [IntelMKL]

        @staticmethod
        def expansion(node, parent_state, parent_sdfg):
            ...

When an SDFG that uses this expansion is compiled, DaCe collects every
environment referenced by every expanded library node, resolves their
``dependencies`` graph, and merges all CMake fragments into the generated
build script.

.. _custom-cmake:

Custom CMake Files
------------------

For dependencies that cannot be expressed by simply calling ``find_package``
and listing libraries, an environment may ship one or more custom CMake files
and reference them through the ``cmake_files`` attribute. This mechanism is
useful when integrating a vendor SDK that requires non-trivial discovery
logic, custom compile/link rules, or auto-generated source files.

The procedure is:

1. Place one or more ``<name>.cmake`` files next to the environment's Python
   module (or anywhere within the library package).
2. List the files in ``cmake_files``, **without** the ``.cmake`` extension and
   **without** a directory prefix - DaCe locates them relative to the file
   that defined the environment (the path is captured by the
   :func:`@dace.library.environment <dace.library.environment>` decorator
   from ``inspect.getmodule(env).__file__``).
3. Inside the custom CMake file, define any variables, targets, or commands
   required, and use the standard CMake ``include_directories``,
   ``target_link_libraries``, etc., as needed. The generated SDFG build
   script will ``include()`` your file before linking the program.

For example, given the layout::

    mylib/
      environments/
        my_env.py
        FindMyDep.cmake

the environment can be declared as:

.. code-block:: python

    @dace.library.environment
    class MyDep:
        cmake_minimum_version = "3.15"
        cmake_packages = ["MyDep"]
        cmake_variables = {}
        cmake_includes = []
        cmake_libraries = ["MyDep::MyDep"]
        cmake_compile_flags = []
        cmake_link_flags = []
        cmake_files = ["FindMyDep"]   # resolves to FindMyDep.cmake next to my_env.py

        headers = ["mydep.h"]
        state_fields = []
        init_code = ""
        finalize_code = ""
        dependencies = []

If your CMake module exposes a target (e.g., ``MyDep::MyDep``), it is usually
sufficient to list it in ``cmake_libraries`` and let the standard target
resolution machinery propagate include directories and compile flags.

Built-in Libraries
------------------

DaCe ships a number of libraries under :mod:`dace.libraries`. They are loaded
automatically when ``dace`` is imported, and their library nodes are emitted
by the :ref:`Python frontend <python-frontend>` whenever a matching NumPy or
SciPy call is encountered.

* :mod:`dace.libraries.blas` - Basic Linear Algebra Subprograms (GEMM, GEMV,
  AXPY, DOT, GER, ...) with reference, OpenBLAS, Intel MKL, cuBLAS, and
  rocBLAS expansions. See :ref:`blas`.
* :mod:`dace.libraries.lapack` - Dense linear algebra solvers (Cholesky, LU,
  triangular solves) backed by LAPACK and cuSOLVER.
* :mod:`dace.libraries.linalg` - Higher-level linear algebra routines
  (e.g., matrix inverse, tensor contractions) layered on top of BLAS, LAPACK,
  and cuTENSOR.
* :mod:`dace.libraries.fft` - Fast Fourier transforms with pure and cuFFT
  expansions.
* :mod:`dace.libraries.sparse` - Sparse linear algebra primitives (e.g.,
  SpMM, SpMV) with pure, Intel MKL, and cuSPARSE expansions.
* :mod:`dace.libraries.mpi` - Point-to-point and collective MPI operations
  (Send, Recv, Bcast, Reduce, Allreduce, ...) for distributed SDFGs.
* :mod:`dace.libraries.pblas` - Distributed dense linear algebra via
  ScaLAPACK/PBLAS, with Intel MKL and reference variants over OpenMPI/MPICH.
* :mod:`dace.libraries.standard` - Common building blocks (reductions,
  transposes, code-region nodes) that are useful across other libraries and
  user code.
* :mod:`dace.libraries.stencil` - Stencil computation library node with
  CPU and GPU expansions.
* :mod:`dace.libraries.onnx` - Library nodes for ONNX operators, enabling
  import and execution of ONNX models inside SDFGs.
* :mod:`dace.libraries.torch` - Interoperability with PyTorch tensors and
  modules, including environments for libtorch headers and linkage.
