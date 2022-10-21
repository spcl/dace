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


full SDFG timer instrumentation

In-depth instrumentation and types

The file can be found in ...

Instrumentation can also collect performance counters on CPUs and GPUs using `LIKWID <https://github.com/RRZE-HPC/likwid>`_.
Example use can be found in LIKWID sample

Data Instrumentation
~~~~~~~~~~~~~~~~~~~~


Instrumentation file format
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Chrome tracing format. You can view it in several ways:

* In the Visual Studio Code extension, laid on top of a program (see :ref:`vscode_trace` for an example)
* Separately, using `Speedscope <https://www.speedscope.app/>`_ (or, if you have Google Chrome, a viewer can also be
  accessed locally at  `<chrome://tracing>`_)
* Printed out in the console with :ref:`sdprof`


