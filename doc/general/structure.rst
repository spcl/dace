Project Structure
=================

DaCe is divided into the following subfolders:

    * Interfaces and entry points:
        * ``frontend``: Different language frontends that can be converted to SDFGs:

          * ``frontend/python``: Python frontend. See :doc:`../frontend/daceprograms` and :ref:`python-frontend`
          * ``frontend/common``: Common utilities for multiple frontends
        * ``cli``: :doc:`../ide/cli` that can view, compile, and profile SDFGs
    * Intermediate representation:
        * ``sdfg``: The :ref:`sdfg-api`
        
          * ``sdfg/graph.py``: General graph API for manipulation and traversal of nodes, edges, and their contents
          * ``sdfg/{nodes, scope, sdfg, state}.py``: Interfaces for specific SDFG components
          * ``sdfg/validation.py``: Functions that validate the correctness of SDFGs. See :ref:`sdfg-validation`
          * ``sdfg/propagation.py``: :ref:`memprop`
        * ``dtypes.py``: Basic enumerations and data types that are used in DaCe
        * ``data.py``: Definitions for usage and creation of :ref:`descriptors`
        * ``memlet.py``: Definition of a :ref:`Memlet <sdfg-memlet>`
        * ``subsets.py``: Subset types used in memlets (e.g., :class:`~dace.subsets.Range`)
        * ``symbolic.py``: Symbolic types, expressions, conversion, and analysis functions. See :ref:`sdfg-symbol` and :ref:`symbolic`
    * Optimization and transformation:
        * ``transformation``: Transformation classes and helpers. See :ref:`transformations`

            * ``transformations/{dataflow, interstate, subgraph}``: Built-in DaCe transformations
        * ``optimization``: Automatic SDFG tuning interfaces
    * Backends:
        * ``codegen``: :ref:`codegen`
        * ``runtime``: Thin runtime that supports DaCe-generated code. See :ref:`runtime`
        * ``sourcemap.py``: Source mapping capabilities that maps frontend code <--> SDFG <--> generated code
    * Library nodes and libraries (See :ref:`libraries`):
        * ``library.py``: Library interface
        * ``libraries``: Built-in libraries
    * Configuration
        * ``config.py``: Configuration-related classes. See :ref:`config` 
        * ``config_schema.yml``: Configuration specification file (schema) with defaults and documentation
    * Miscellaneous
        * ``serialize.py`` and ``registry.py``: Functionality that supports serialization and extensibility. See :ref:`properties`
        * ``external``: Git submodules containing necessary external projects
        * ``viewer``: Contains infrastructure for rendering SDFGs. Used in Jupyter notebooks and in :ref:`sdfv`
