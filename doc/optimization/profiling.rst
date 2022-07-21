.. _profiling:

Profiling and Instrumentation
=============================

``DACE_profiling``

the files

full SDFG timer instrumentation

In-depth instrumentation and types

The file can be found in ...


Chrome tracing format. You can view it in several ways:

* In the Visual Studio Code extension, laid on top of a program (see :ref:`vscode_trace` for an example)
* Separately, using `Speedscope <https://www.speedscope.app/>`_ (or, if you have Google Chrome, a viewer can also be
  accessed locally at  `<chrome://tracing>`_)
* Printed out in the console with :ref:`sdprof`

Using LIKWID for Profiling
------------------------------------------
`Likwid <https://github.com/RRZE-HPC/likwid>`_ is an easy-to-use tool suite that allows to instrument C/C++ programs and measure performance hardware counters.
Different regions of the code can be marked by calls to the `Likwid Marker API <https://github.com/RRZE-HPC/likwid/wiki/likwid-perfctr#using-the-marker-api>`_.
DaCe can automatically inject the setup code as well as the region markers for specific SDFG elements during code generation.
Currently, supported elements for instrumentation with Likwid are states and maps (top-level only).

Setup
~~~~~

In order to use Likwid instrumentation with DaCe, Likwid >v5.0 must be installed on your machine or available on your cluster. Furthermore, the DaCe implementation only supports the backends *direct* and *accessdaemon*. The implementation can switch between both backends by setting the environment variable *LIKWID_MODE* to "0" (direct) or "1" (accessdaemon; default).

Usage
~~~~~

Instrumentation with Likwid is almost as simple as instrumentation with regular timers.
For this, we look at this minimal matmul example. A detailed version can be found in the DaCe samples.

.. code-block:: python

    import dace
    import dace.transformation.helpers as xfh

    M = dace.symbol('M')
    K = dace.symbol('K')
    N = dace.symbol('N')

    @dace.program
    def matmul(A: dace.float32[M, K], B: dace.float32[K, N], C: dace.float32[M, N]):
        tmp = np.ndarray([M, N, K], dtype=A.dtype)
        for i, j, k in dace.map[0:M, 0:N, 0:K]:
            with dace.tasklet:
                in_A << A[i, k]
                in_B << B[k, j]
                out >> tmp[i, j, k]

                out = in_A * in_B
        dace.reduce(lambda a, b: a + b, tmp, C, axis=2, identity=0)

We convert the matmul program to its SDFG so that we can set the instrumentation flags to *LIKWID_Counters* on states and maps. The measured counters are accessible through the performance report after compilation and execution of the SDFG.

.. code-block:: python

    sdfg = matmul.to_sdfg()

    for state in sdfg.nodes():
        state.instrument = dace.InstrumentationType.LIKWID_Counters
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry) and xfh.get_parent_map(state, node) is None:
                node.instrument = dace.InstrumentationType.LIKWID_Counters
    
    csdfg = sdfg.compile()
    csdfg(A=A, B=B, C=C)
    report = sdfg.get_latest_report()

The only thing that changes is how we run the script: 

.. code-block:: console

  OMP_NUM_THREADS=2 LIKWID_EVENTS="FLOPS_SP" python matmul.py    

where OMP_NUM_THREADS sets the number of threads and LIKWID_EVENTS specifies which hardware counters to measure. Those event sets are defined in the `likwid's groups <https://github.com/RRZE-HPC/likwid/tree/master/groups>`_ for different architectures.