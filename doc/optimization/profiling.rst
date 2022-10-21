.. _profiling:

Profiling and Instrumentation
=============================

.. note::

  For more information and examples, see the `Benchmarking and Instrumentation <https://nbviewer.jupyter.org/github/spcl/dace/blob/master/tutorials/benchmarking.ipynb>`_ tutorial.

Simple profiling
----------------

Full-program profiling of Python DaCe programs can be performed by using the :envvar:`profiling` configuration entry. 
This simple profiling mode is performed within Python using timers, calling the same program for a configurable number (:envvar:`treps`)
of times and printing the median execution time. It is not as accurate as the other profiling modes, but it is easy to
use and does not require any additional tools.


The profiling configuration entry can be set globally through an environment variable or within code. Every time an SDFG
is invoked, a profiling report CSV file will be generated in the following format: ``.dacecache/<program name>/profiling/results-<timestamp>.csv``.
For example, the following code will print the execution time of the ``my_function`` function after running it 100 times:

.. code-block:: python

  import dace
  import numpy as np

  @dace.program
  def my_function(A: dace.float64[10000]):
      return A + 1

  A = np.random.rand(10000)
  
  with dace.config.set_temporary('profiling', value=True):  # Enable profiling
      with dace.config.set_temporary('treps', value=100):   # Run 100 times
          my_function(A)


.. note::

  This mode executes the same program multiple times. If the output would be affected by this (e.g., if an array is
  incremented), either use :envvar:`treps` = 1 or use the :ref:`instrumentation` mode.

.. _instrumentation:

Instrumentation
---------------

Instrumentation is a more accurate profiling mode that generates specific measurement code on an SDFG or any sub-part
of it (for example, a single Map). When the SDFG is called, the instrumentation API generates a JSON file for each
execution, containing the measured metrics (see file format below) and places it in the ``.dacecache/<program name>/perf``
directory.

The instrumentation API can be used by setting ``element.instrument`` to the desired instrumentation type (see :class:`~dace.dtypes.InstrumentationType`
for a list of the default available types). ``element`` can be almost any SDFG element, from the SDFG itself, through a
state, to a variety of nodes, such as a Map, a Tasklet, or a NestedSDFG. The generated report can then be read programmatically
as a :class:`~dace.codegen.instrumentation.report.InstrumentationReport` object. The SDFG class provides the methods 
:func:`~dace.sdfg.sdfg.SDFG.get_latest_report` and :func:`~dace.sdfg.sdfg.SDFG.get_instrumentation_reports` to read the last or 
all generated reports, respectively. See :class:`~dace.sdfg.sdfg.SDFG` for more methods related to instrumentation reports.

A simple example use of SDFG instrumentation would be to mimic the simple profiling mode from above with a 
:class:`~dace.dtypes.InstrumentationType.Timer` instrumentation applied on the whole SDFG:

.. code-block:: python

  import dace
  import numpy as np

  @dace.program
  def twomaps(A):
      B = np.sin(A)
      return B * 2.0

  a = np.random.rand(1000, 1000)
  sdfg = twomaps.to_sdfg(a)
  sdfg.instrument = dace.InstrumentationType.Timer  # Instrument the whole SDFG

  sdfg(a)

  # Print the execution time in a human-readable tabular format
  report = sdfg.get_latest_report()
  print(report)


More in-depth instrumentation can be performed by applying instrumentation to specific nodes. For example, the following
code will instrument the individual Map scopes in the above application:

.. code-block:: python

  # Instrument the individual Map scopes
  for state in sdfg.nodes():
      for node in state.nodes():
          if isinstance(node, dace.nodes.MapEntry):
              node.instrument = dace.InstrumentationType.Timer

  # The report will now contain information on each individual map. Example printout:
  # Instrumentation report
  # SDFG Hash: 0f02b642249b861dc94b7cbc729190d4b27cab79607b8f28c7de3946e62d5977
  # ---------------------------------------------------------------------------
  # Element                          Runtime (ms)
  #               Min            Mean           Median         Max            
  # ---------------------------------------------------------------------------
  # SDFG (0)                                                                   
  # |-State (0)                                                                
  # | |-Node (0)                                                               
  # | | |Map _numpy_sin__map:                                                  
  # | | |          11.654         11.654         11.654         11.654         
  # | |-Node (5)                                                               
  # | | |Map _Mult__map:                                                       
  # | | |          1.524          1.524          1.524          1.524          
  # ---------------------------------------------------------------------------


There are more instrumentation types available, such as fine-grained GPU kernel timing with :class:`~dace.dtypes.InstrumentationType.GPU_Events`.
Instrumentation can also collect performance counters on CPUs and GPUs using `LIKWID <https://github.com/RRZE-HPC/likwid>`_.
The :class:`~dace.dtypes.InstrumentationType.LIKWID_Counters` instrumentation type can be configured to collect
a wide variety of performance counters on CPUs and GPUs. An example use can be found in the
`LIKWID instrumentation code sample <https://github.com/spcl/dace/blob/master/samples/instrumentation/matmul_likwid.py>`_.


Instrumentation file format
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instrumentation uses a JSON file in the `Chrome Trace Event <https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview>`_ 
format to store the collected metrics. You can view it in several ways:

  * In the Visual Studio Code extension, laid on top of a program (see :ref:`vscode_trace` for an example)
  * Separately, using `Speedscope <https://www.speedscope.app/>`_ (or, if you have Google Chrome, a viewer can also be
    accessed locally at  `<chrome://tracing>`_)
  * Printed out in the console with :ref:`sdprof`



Data Instrumentation
~~~~~~~~~~~~~~~~~~~~

Similarly to timing events, data containers and their contents can be serialized for performance and validation 
reproducibility purposes. This is done by setting the ``instrument`` property of an :class:`~dace.sdfg.nodes.AccessNode`
to a :class:`~dace.dtypes.DataInstrumentationType`, such as :class:`~dace.dtypes.DataInstrumentationType.Save`.
The data will be serialized (keeping each version if the access node is encountered multiple times) in the 
``.dacecache/<program name>/data`` directory. The data can then be reloaded in subsequent executions by setting the
``instrument`` property to :class:`~dace.dtypes.DataInstrumentationType.Restore`.

This feature is crucial for reproducibility and validation purposes, as it allows to run the same program multiple times
with the same input data, and compare the output data to the original output data. Data instrumentation powers cutout-based
auto-tuning (:class:`~dace.optimization.cutout_tuner.CutoutTuner`), which looks at subsets of a program at a time.

The folder structure of a data report is as follows: ``.dacecache/<program name>/data/<array name>/<uuid>_<version>.bin``,
where ``<array name>`` is the data container name in the SDFG, ``<uuid>`` is a unique identifier to the access node from
which this array was saved, and ``<version>`` is a running number for the currently-saved array (e.g., when an access node
is written to multiple times in a loop).

The instrumented data report can be read in the Python API via the :class:`~dace.codegen.instrumentation.data.data_report.InstrumentedDataReport`
class, which can be obtained by calling :func:`~dace.sdfg.sdfg.SDFG.get_instrumented_data` on the SDFG object.
The files themselves are direct binary representations of the whole data (with padding and strides), for complete
reproducibility. When accessed from Python, a numpy wrapper shows the user-accessible view of that array.

Example of creating and reading such a report is as follows:

.. code-block:: python

    @dace.program
    def data_instrumentation(A: dace.float64[1000, 1000]):
        versioned = np.zeros_like(A)
        for i in range(10):
          versioned += A
        return versioned

    sdfg = data_instrumentation.to_sdfg()
    
    # ... Set instrument to Save on the AccessNodes and run the SDFG ...

    dreport = sdfg.get_instrumented_data()  # Returns an InstrumentedDataReport
    print(dreport.keys())                   # Will print "'A', 'versioned'"
    array = dreport['A']  # return value is a single array if there is only one version
    varrays = dreport['versioned']  # otherwise, return value is a sorted list of versions
    
    # after loading, arrays can be used normally with numpy
    assert np.allclose(array, real_A)
    for arr in varrays:
        print(arr[5, :])

