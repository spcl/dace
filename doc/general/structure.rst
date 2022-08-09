Project Structure
=================

DaCe is divided into the following subfolders:

    * Interfaces and entry points:
        * frontend
        * cli
    * Intermediate representation:
        * graph
        * sdfg
        * dtypes.py
        * data.py
        * memlet.py
        * subsets.py
        * symbolic.py
    * Optimization and transformation:
        * transformation
        * optimization
    * Backends:
        * codegen
        * runtime
        * sourcemap.py
    * Library nodes and libraries:
        * library.py
        * libraries - for more information, see
    * Configuration
        * config.py - configuration related
        * config_schema.yml
    * Miscellaneous
        * serialize.py and registry.py --
        * external
        * viewer
