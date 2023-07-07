Command-Line Utilities
======================

DaCe provides a set of command-line utilities to simplify interaction with SDFGs and tools
or workflows within the DaCe framework.

.. _sdfgcc:

:code:`sdfgcc` - SDFG Command-line Compiler
-------------------------------------------

The SDFG Command-line Compiler :code:`sdfgcc` enables compilation of SDFG files from the command-line.

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

The SDFG Viewer :code:`sdfv` displays SDFGs in the system's HTML viewer. It reads an SDFG and creates a temporary
HTML file that contains a standalone viewer, which is then opened.

| Usage:
| :code:`sdfv <filepath>`

+-----------------------+--------------+----------------------------------------------------------+
| Argument              | Required     | Description                                              |
+=======================+==============+==========================================================+
| **<filepath>**        | Yes          | Path to the SDFG file to show. If a directory is         |
|                       |              | provided, the tool searches for a file called            |
|                       |              | :code:`program.sdfg` in said folder                      |
+-----------------------+--------------+----------------------------------------------------------+

.. _daceprof:

:code:`daceprof` - Profiler and Report Viewer
---------------------------------------------

The DaCe profiler is a versatile profiling and analysis tool that can provide performance results
and performance modeling for DaCe programs. Calling a Python script or module with ``daceprof`` instead
of ``python`` will profile/instrument each individual call to a DaCe program and print the latest
report at the end.

The tool can also be used to view a profiling report directly in the console with the ``-i`` flag.

If ``--type`` is given, performs instrumentation of specific elements in the invoked DaCe program. If
nothing is given, the tool will time the entire execution of each program using :func:`~dace.builtin_hooks.profile`.


| Usage:
| :code:`daceprof [ARGUMENTS] myscript.py [SCRIPT ARGUMENTS]`
| :code:`daceprof [ARGUMENTS] -m package.module [MODULE ARGUMENTS]`
| :code:`daceprof [ARGUMENTS] -i profile.json`

+---------------------------+--------------+-----------------------------------------------------------+
| Argument                  | Required     | Description                                               |
+===========================+==============+===========================================================+
| **Execution arguments**   |              |                                                           |
+---------------------------+--------------+-----------------------------------------------------------+
| <scriptpath>        OR    | Yes          | Path to the script file, report, or Python module.        |
| -m <modulepath>     OR    |              |                                                           |
| -i <filepath>             |              |                                                           |
+---------------------------+--------------+-----------------------------------------------------------+
| **Profiling arguments**   |              |                                                           |
+---------------------------+--------------+-----------------------------------------------------------+
| :code:`-r,--repetitions`  |              | Runs each profiled program for the specified number of    |
| ``REPETITIONS``           |              | repetitions (default: 100).                               |
+---------------------------+--------------+-----------------------------------------------------------+
| :code:`-w,--warmup`       |              | Number of additional repetitions to run before measuring  |
| ``WARMUP``                |              | runtime (default: 0).                                     |
+---------------------------+--------------+-----------------------------------------------------------+
| :code:`-t,--type` ``TYPE``|              | Followed by :class:`~dace.dtypes.InstrumentationType`,    |
|                           |              | specified which instrumentation type to use. If not given,|
|                           |              | times the entire SDFG with a wall-clock timer.            |
+---------------------------+--------------+-----------------------------------------------------------+
| :code:`--instrument`      |              | A comma-separated list specifying which SDFG elements to  |
| INSTRUMENT                |              | instrument. Can be a comma-separated list of element types|
|                           |              | from the following: ``map, tasklet, state, sdfg``.        |
+---------------------------+--------------+-----------------------------------------------------------+
| :code:`--sequential`      |              | Disable CPU multi-threading in code generation.           |
+---------------------------+--------------+-----------------------------------------------------------+
| **Data instrumentation**  |              |                                                           |
+---------------------------+--------------+-----------------------------------------------------------+
| :code:`-ds,--save-data`   |              | Enable data instrumentation and store all (or filtered)   |
|                           |              | arrays.                                                   |
+---------------------------+--------------+-----------------------------------------------------------+
| :code:`-dr,--restore-data`|              | Reproducibly run code by restoring all (or filtered)      |
|                           |              | arrays.                                                   |
+---------------------------+--------------+-----------------------------------------------------------+
| **Filtering arguments**   |              |                                                           |
+---------------------------+--------------+-----------------------------------------------------------+
| :code:`-f,--filter`       |              | Specifies a filter for elements to instrument.            |
+---------------------------+--------------+-----------------------------------------------------------+
| :code:`-df,--filter-data` |              | Specifies a filter for data containers to serialize.      |
+---------------------------+--------------+-----------------------------------------------------------+
| **Report arguments**      |              |                                                           |
+---------------------------+--------------+-----------------------------------------------------------+
| :code:`-s,--sort`         |              | Sort by a specific criterion. Choices are:                |
|                           |              |                                                           |
|                           |              | - :code:`min|max|mean|median`:                            |
|                           |              |   Sort by the minimum/maximum/mean/median observed value. |
|                           |              | - :code:`counter`: Sort by counter name/type.             |
|                           |              | - :code:`value`: Sort by the observed value.              |
+---------------------------+--------------+-----------------------------------------------------------+
| :code:`-a,--ascending`    |              | If given, sort in ascending order.                        |
+---------------------------+--------------+-----------------------------------------------------------+
| :code:`-o,--output`       |              | If given, saves report in output path.                    |
+---------------------------+--------------+-----------------------------------------------------------+
| :code:`--csv`             |              | Use Comma-Separated Values (CSV) for reporting.           |
+---------------------------+--------------+-----------------------------------------------------------+

For a more detailed guide on how to profile SDFGs and work with the resulting data, see :ref:`profiling` and
`this tutorial <https://nbviewer.org/github/spcl/dace/blob/master/tutorials/benchmarking.ipynb#Benchmarking-and-Instrumentation-API>`_.
