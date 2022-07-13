Command-Line Utilities
======================

DaCe provides a set of command-line utilities to simplify interaction with SDFGs and tools
or workflows within the DaCe framework.

.. _sdfgcc:

:code:`sdfgcc` - SDFG Command-Line Compiler
-------------------------------------------

The SDFG Command-Line Compiler :code:`sdfgcc` enables compilation of SDFG files from the command-line.

| Usage:
| :code:`sdfgcc [-o OUT] [-O] <filepath>`

+-----------------------+--------------+----------------------------------------------------------+
| Argument              | Required     | Description                                              |
+=======================+==============+==========================================================+
| **<filepath>**        | Yes          | Path to the SDFG file to compile.                        |
+-----------------------+--------------+----------------------------------------------------------+
| :code:`-O,--optimize` |              | If set, invokes the command-line optimization interface. |
+-----------------------+--------------+----------------------------------------------------------+
| :code:`-o,--out`      |              | If provided, saves the library as the given file or in   |
|                       |              | the specified path, together with a header file.         |
+-----------------------+--------------+----------------------------------------------------------+

.. _sdfv:

:code:`sdfv` - SDFG Viewer
--------------------------

The SDFG Viewer :code:`sdfv` displays SDFGs in the system's HTML viewer.

| Usage:
| :code:`sdfv <filepath>`

+-----------------------+--------------+----------------------------------------------------------+
| Argument              | Required     | Description                                              |
+=======================+==============+==========================================================+
| **<filepath>**        | Yes          | Path to the SDFG file to show. If a directory is         |
|                       |              | provided, the tool searches for a file called            |
|                       |              | :code:`program.sdfg` in said folder                      |
+-----------------------+--------------+----------------------------------------------------------+

.. _sdprof:

:code:`sdprof` - SDFG Profile Viewer
------------------------------------

The SDFG Profile Viewer :code:`sdprof` shows summarizations of SDFG profiling and instrumentation
reports in the command-line.

| Usage:
| :code:`sdprof [-s CRITERION] [-a] <filepath>`

+-----------------------+--------------+-----------------------------------------------------------+
| Argument              | Required     | Description                                               |
+=======================+==============+===========================================================+
| **<filepath>**        | Yes          | Path to the file containing the report.                   |
+-----------------------+--------------+-----------------------------------------------------------+
| :code:`-s,--sort`     |              | Sort by a specific criterion. Choices are:                |
|                       |              |                                                           |
|                       |              | - :code:`min|max|mean|median`:                            |
|                       |              |   Sort by the minimum/maximum/mean/median observed value. |
|                       |              | - :code:`counter`: Sort by counter name/type.             |
|                       |              | - :code:`value`: Sort by the observed value.              |
+-----------------------+--------------+-----------------------------------------------------------+
| :code:`-a,--ascending`|              | If given, sort in ascending order.                        |
+-----------------------+--------------+-----------------------------------------------------------+

For a more detailed guide on how to profile SDFGs and work with the resulting data, see
`this tutorial <https://nbviewer.org/github/spcl/dace/blob/master/tutorials/benchmarking.ipynb#Benchmarking-and-Instrumentation-API>`_.
