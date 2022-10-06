.. _debugging:

Debugging
=========

There are several sources of potential issues with DaCe programs:

    * Frontend failures during parsing programs to SDFGs
    * SDFG validity before or during optimization
    * Transformations that impair the correctness of the SDFG
    * Failures during the :ref:`codegen` process
    * Segmentation faults or errors in the generated code


In general, DaCe tries to raise a Python exception and clearly print the origin of the issue. However, to shed more light
into the origin of the problem, it can be useful to set the :envvar:`debugprint` configuration entry to ``1`` or ``verbose``.

There are several other important configuration entries: for frontend and debugging why Python functions become callbacks,
use :envvar:`frontend.verbose_errors`. For transformations that fail during matching, use :envvar:`optimizer.match_exception`.
For issues with :ref:`properties`, enable :envvar:`testing.serialization` and :envvar:`testing.deserialize_exception`.

Below we provide a more detailed methodology for debugging particular issues. You can find common errors and solutions
:ref:`here <errors>`.

.. _sdfg-validation:

Graph Validation
----------------

SDFGs can be validated for soundness. This happens automatically during compilation, but can be triggered manually
using the :func:`dace.sdfg.sdfg.SDFG.validate` method. It can be useful to detect issues in the graph, examples include
Memlets that mismatch their context, out of bounds access, undefined symbol use, scopes that are not properly closed, 
and many more.

On validation failure, (unless specified) a copy of the failing SDFG will be saved in the current working directory,
under ``_dacegraphs/invalid.sdfg``, which includes the source of the error. Opening it in the Visual Studio Code 
extension even zooms in on the issue automatically!


Crashes and Compiled Programs
-----------------------------

.. note::
    For debugging the code generation process itself, see :ref:`debug_codegen`.

Compiled programs are compiled to a shared object (``.so`` / ``.dll`` file) that is linked to the host process. If using
a DaCe program within Python, debugging it requires simply calling any debugger (such as ``gdb``) on the Python process
and potentially setting breakpoints on the generated code (which can be found using the ``sdfg.build_folder`` property).
For example:

.. code-block:: sh

    gdb --args python myscript.py [args...]


In most cases, debugging in Release mode does not yield actionable results. To better debug compiled programs, set 
the :envvar:`compiler.build_type` configuration entry to ``Debug`` and rerun the program. The following example shows
a crashing program and how the process works:

.. code-block:: python

    import dace
    import numpy as np
    N = dace.symbol('N')

    @dace.program
    def example(a: dace.float32[N], b: dace.float32[N]):
        b[5000000] = a[0]

    n = 10
    a = np.random.rand(n).astype(np.float32)
    b = np.random.rand(n).astype(np.float32)

    example(a, b)  # Calling this function could trigger a segmentation fault

.. code-block:: sh

    $ python example.py
    ...
    sh: segmentation fault  python example.py

    $ gdb --args python example.py
    ...
    (gdb) r
    ...
    Thread 1 "python" received signal SIGSEGV, Segmentation fault.
    0x00007fffe7259186 in __program_example_internal(example_t*, float*, float*, int) () from /path/.dacecache/example/build/libexample.so
    
    # No further information is given on the source of the issue. Below we set debug mode:
    $ DACE_compiler_build_type=Debug gdb --args python example.py
    ...
    (gdb) r
    ...
    Thread 1 "python" received signal SIGSEGV, Segmentation fault.
    0x00007fffe7159186 in __program_example_internal (__state=0x5555574669a0, a=0x55555699efd0, b=0x555556f4c390, N=10)
    --Type <RET> for more, q to quit, c to continue without paging--
    at /path/.dacecache/example/src/cpu/example.cpp:27
    27                  b[5000000] = __out;


You can also use the Visual Studio Code extension to debug Python programs by using the ``DaCe debugger`` debug provider.
It even supports mapping breakpoints from the Python code to the generated code.

For low-level access of the CMake configuration, you could also access the build folder, go to the ``build/`` 
subdirectory, and call ``ccmake .`` to modify it. After that run ``make`` to rebuild.

.. _gpu-debugging:

GPU Debugging in DaCe
~~~~~~~~~~~~~~~~~~~~~

As GPU kernels cannot be debugged directly in ``gdb``, there are other tools that can be used to debug GPU programs.

The CUDA toolkit provides more tools to debug kernels: ``cuda-gdb`` can break and debug CUDA kernels, and ``cuda-memcheck``
can be used to track invalid memory accesses. 

Additional debugging features in DaCe include GPU stream synchronization debugging. Since GPU toolkits (CUDA, HIP, OpenCL)
mostly run asynchronously using nonblocking calls, it is sometimes hard to pinpoint the source of an issue. Since GPU
programs can be large and run for a while, ``Debug`` mode cannot always be enabled. For these reasons, DaCe provides
a mode that can run directly in ``Release`` mode, called *synchronous debugging*. The mode inserts device-synchronization
calls after every GPU-related operation (kernel, library call) and checks for errors. This helps debug both crashes
and stream-related data races. Enable it by setting :envvar:`compiler.cuda.syncdebug` to True.


Debugging Transformations
-------------------------

Transformation debugging can be used for multiple purposes: it can be used to understand why transformations fail to
match on a specific subgraph, debug exceptions on matching, and failures during application of transformations.

By default, exceptions during transformation matching emit a warning. To debugging exceptions on matching, enable the
:envvar:`optimizer.match_exception` configuration entry, which would turn them into errors.

If setting breakpoints, since transformations repeatedly try to apply on matching subgraphs on an SDFG, it is 
recommended to set conditional breakpoints including labels or any defining properties of the nodes/edges you want to 
debug the transformation for.

Another approach is to run the debugger on the Visual Studio Code extension's optimizer daemon. The daemon is a Python
script, so it can be debugged as such. Simply create a new debug configuration that starts the script 
(see :ref:`qa_vscode` on how to find the command) with the right port, kill the existing SDFG Optimizer, and debug the
script. Breakpoints should now work inside DaCe or your custom transformations.


Debugging Frontend Issues
-------------------------

When debugging frontend issues, it is important to make the distinction between the frontend itself and transformations
applied on the initial SDFG. Thus, if there is a suspected issue in the frontend, first try disabling automatic simplification
(through the :envvar:`optimizer.automatic_simplification` config entry or the API, see below) and validating the initial 
SDFG for soundness:

.. code-block:: python

    sdfg = bad_program.to_sdfg(simplify=False)
    sdfg.validate()

If this works but some programs fail, it might be a serialization issue. Try a save/load roundtrip:

.. code-block:: python

    sdfg.save('test.sdfg')
    sdfg = dace.SDFG.from_file('test.sdfg')
    sdfg.validate()
    # ...other validation methods...

Otherwise, the issue could be in the :ref:`simplify`. Try to simplify while validating every step:

.. code-block:: python

    sdfg.simplify(verbose=True, validate_all=True)

This helps understanding which component causes the issue.
