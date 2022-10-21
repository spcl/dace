.. _codegen:

Code Generation
===============

The DaCe code generator traverses an SDFG and generates matching code for the different supported platforms. 
It also contains components that interact with compilers (via `CMake <https://cmake.org/>`_) and invoke the
compiled dynamic library (.so/.dll/.dylib file) directly from Python.

.. _codegen_how_it_works:

How it Works
------------

Given the low-level nature of the SDFG IR, the code generator simply recursively traverses the graph and emits code for each part.
This is shown in the animation below:

.. image:: codegen.gif


The main code (called "frame code") is generated in C with externally-callable functions (see :ref:`integration_c`).
The rest of the backends may use different languages (CUDA, SystemVerilog, OpenCL for FPGA, etc.).

The process starts with inspecting the SDFG to find out which targets are necessary. Then, each target can **preprocess** the graph for its own analysis (which we assume will not be modified during generation).
After that, the control-flow graph is converted to structured control flow constructs in C (i.e., ``for/while/if/switch`` etc.). Then, the frame code generator (``targets/framecode.py``) **dispatches** the backends as necessary.
There are many features that are enabled by generating code from SDFGs:

  * Allocation management can be handled based on :class:`~dace.dtypes.AllocationLifetime`
  * Defined variables can be tracked with types
  * Concurrency is represented by, e.g., OpenMP parallel sections or GPU streams


.. note::

    You can also extend the code generator with new backends externally, see the `Customizing Code Generation tutorial <https://nbviewer.jupyter.org/github/spcl/dace/blob/master/tutorials/codegen.ipynb>`_ 
    and the `Tensor Core sample <https://github.com/spcl/dace/blob/master/samples/codegen/tensor_cores.py>`_ for more information.


After the code is generated, ``compiler.py`` will invoke CMake on the build folder (e.g., ``.dacecache/<program>/build``)
with the CMake file in ``dace/codegen/CMakelists.txt``. This will generate a Makefile or Visual Studio solution,
which is then built by the compiler Python interface. The resulting dynamic library is then loaded into a :class:`~dace.codegen.compiled_sdfg.CompiledSDFG`
object, which calls the three C functions ``initialize`` (on init), ``program`` (every SDFG invocation), and ``finalize``
(on destruction).

File Structure
--------------

The files in ``dace/codegen`` are organized into several categories:

  * The ``targets`` folder contains target-specific backends
  * ``codegen.py`` controls the code generation API and process, with ``codeobject.py, common.py, dispatcher.py, exceptions.py`` and the ``tools`` subfolder as helpers
  * ``prettycode.py`` prints nicely indented C/C++ code
  * ``control_flow.py`` contains control flow analysis to recover structured control flow (e.g., for loops, if conditions) from SDFG state machines
  * ``cppunparse.py`` takes (limited) Python codes and outputs matching C++ codes. This is used in Tasklets and inter-state edges
  * The ``instrumentation`` folder contains implementations of :ref:`SDFG instrumentation <profiling>` providers
  * ``compiler.py`` and ``CMakeLists.txt`` contain all the utilities necessary to interoperate with the CMake compiler
  * ``compiled_sdfg.py`` contains interfaces to invoke a compiled SDFG dynamic library
  * ``dace/runtime`` contains the :ref:`runtime`

Class Structure and Detailed Process
------------------------------------

:func:`~dace.codegen.codegen.generate_code` is the entry point to code generation. It takes an SDFG and returns a 
list of :class:`~dace.codegen.codeobject.CodeObject` objects, referring to each generated file.

This function then invokes several preprocessing analysis passes, such as concretizing default storage and schedule types
into their actual value based on their context (e.g., a scalar inside a GPU kernel will become a register), and expanding
all remaining library nodes.

The next step is to create the frame-code generator via the :class:`~dace.codegen.targets.framecode.DaCeCodeGenerator`, 
which will later traverse the graph and invoke the other targets. The targets are found by traversing the SDFG,
followed by each one preprocessing the SDFG for metadata collection as necessary. The frame-code generator then
calls its own :func:`~dace.codegen.targets.framecode.DaCeCodeGenerator.generate_code` method, which will generate the headers, footers, and frame, as well as dispatch the other targets.

To generate compiler (and user) friendly code, the state machine of the SDFG is first converted into a structured control-flow
tree by calling :func:`~dace.codegen.control_flow.structured_control_flow_tree`.
This tree is then traversed by the frame-code generator to create ``for`` loops, ``if`` conditions, and ``switch`` statements, 
among others.

Within the control flow tree, each state is visited by the frame-code generator, which will then dispatch the other targets
using the :class:`~dace.codegen.dispatcher.TargetDispatcher` class. Code generator targets register themselves 
with the dispatcher by extending the :class:`~dace.codegen.targets.target.TargetCodeGenerator` class, and then
the dispatcher via the ``register_*_dispatcher`` methods (e.g., :func:`~dace.codegen.dispatcher.TargetDispatcher.register_node_dispatcher`)
in their constructor. The dispatcher will then call the given predicate function to determine whether the target
should be invoked for the given node. For example, an excerpt from the GPU code generator is shown below: 

.. code-block:: python

    @registry.autoregister_params(name='cuda')
    class CUDACodeGen(TargetCodeGenerator):
        """ GPU (CUDA/HIP) code generator. """
        target_name = 'cuda'
        title = 'CUDA'

        def __init__(self, frame_codegen: DaCeCodeGenerator, sdfg: SDFG):
            # ...
            self.dispatcher = frame_codegen.dispatcher
            self.dispatcher.register_map_dispatcher(dtypes.GPU_SCHEDULES, self)
            self.dispatcher.register_node_dispatcher(self, self.node_dispatch_predicate)
            self.dispatcher.register_state_dispatcher(self, self.state_dispatch_predicate)

        def node_dispatch_predicate(self, sdfg: SDFG, state: SDFGState, node: nodes.Node):
            ...


The dispatcher will then invoke its ``dispatch_*`` methods (e.g., :func:`~dace.codegen.dispatcher.TargetDispatcher.dispatch_node`)
to invoke the target. Those will then call the ``generate_*`` methods (e.g., :func:`~dace.codegen.targets.target.TargetCodeGenerator.generate_node`).
On most targets, each node type has a matching ``_generate_<class>`` method, similarly to AST visitors, which are 
responsible for that node type. For example, see :func:`~dace.codegen.targets.cpu.CPUCodeGen._generate_MapEntry` in 
:class:`~dace.codegen.targets.cpu.CPUCodeGen`.

In the generation methods, there are several arguments that are passed to the target, for locating the element (i.e., 
SDFG, state, node), and handles to two or more :class:`~dace.codegen.prettycode.CodeIOStream` objects, which are used to write
the code itself (it is common to have a ``callsite_stream`` that point to the current place in the file, and a ``global_stream``
for global declarations). At this point, instrumentation providers are also invoked to insert profiling code, if set. The
exact methods that are invoked can be found in :class:`~dace.codegen.instrumentation.provider.InstrumentationProvider`.

After the graph is traversed, each target is invoked with two methods: 
:func:`~dace.codegen.targets.target.TargetCodeGenerator.get_generated_codeobjects` and :func:`~dace.codegen.targets.target.TargetCodeGenerator.cmake_options`
to retrieve any extra :class:`~dace.codegen.codeobject.CodeObject` files and CMake options, respectively.
The frame-code generator will then merge all code objects and return them, along with any environments/libraries that
were requested by the code generators (e.g., link with CUBLAS). The compiler interface then generates the ``.dacecache``
folders in :func:`~dace.codegen.compiler.generate_program_folder` and invokes the CMake compiler in :func:`~dace.codegen.compiler.configure_and_compile`.


.. _runtime:

C++ Runtime Headers
-------------------

The code generator uses a thin C++ runtime for support. The folder, which contains header files written for the different platforms, can
be found in the ``dace/runtime`` folder. The ``dace.h`` header file is the point of entry for the runtime, and it includes all the other
necessary headers. The runtime is used for:

  * **Target-specific runtime functions**: Header files inside the ``cuda``, ``intel_fpga``, and ``xilinx`` folders contain
    GPU (CUDA/HIP), Intel FPGA, and Xilinx-specific functions, respectively.
  * Memory management
  * **Profiling**: ``perf/reporting.h`` contains functions that create :ref:`instrumentation reports <instrumentation>`,
    ``perf/papi.h`` contains functions that use the `PAPI <http://icl.cs.utk.edu/papi/>`_ library to measure performance counters.
  * **Serialization**: Data instrumentation is provided by ``serialization.h``, which can be used to serialize and deserialize 
    versioned data containers (e.g., arrays) to and from files.
  * **Data movement**: copying, packing/unpacking, atomic operations and others are supported by ``{copy, reduction}.h``
    and target-specific files such as ``cuda/copy.cuh``
  * **Interoperability with Python**: ``{complex, intset, math, os, pi, pyinterop}.h`` and others provide functions that 
    match Python interfaces. This is especially useful to generate matching code when calling functions such as ``range``
    inside Tasklets.

The folder also contains other files and helper functions, refer to its contents `on GitHub <https://github.com/spcl/dace/tree/master/dace/runtime/include/dace>`_ 
for more information.


.. _debug_codegen:

Debugging Code Generation
-------------------------

.. note::

    Read :ref:`recompilation` first to understand how to recompile SDFGs.


If the code generator fails to generate code for a given SDFG, it will raise an exception. Along with the exception message,
the failing pre-processed SDFG will be saved to ``_dacegraphs/failing.sdfg``. This file can first be inspected for structural
issues.

The code generator itself can be debugged in different ways. If the code generation process is failing, 
the Python debugger would be enough to try and understand what is going on. However, if the code generation process
successfully finishes, but generates erroneous code, we can configure DaCe to generate line information for each line
in the generated code, pointing to the Python file and line where it was generated. This can be done by setting the
:envvar:`compiler.codegen_lineinfo` to ``1``. This will generate a source map in ``.dacecache/<program>/map_codegen.json``
that can be read directly, or used automatically in the Visual Studio Code extension to jump directly to the 
originating line.

Once the offending line is known, it can be tricky to understand the specific circumstances of the failure. To debug
this efficiently, we can use conditional breakpoints on the code generator with particular hints on the source node.
For example, if we want to debug the code generation of a specific node, we can set a breakpoint in the code generator
and add a condition to it, such as ``node.label == "my_node"``. This will stop the code generation process when the
code generator reaches the node with the label ``my_node``. This can be used to debug the code generation of a specific
node, or to debug the code generation of a specific node type (e.g., ``isinstance(node, dace.nodes.MapEntry)``).


FPGA Code Generation
--------------------
The FPGA Code Generation emits High-Level Synthesis device code and all the host code required to target either Xilinx or Intel FPGAs.

The FPGA code generation is implemented by different modules, organized hierarchically:

    * a generic FPGA backend (``dace/codegen/target/fpga.py``) is in charge of traversing the SDFG as shown in :ref:`codegen_how_it_works`;
    * two lower level components that are in charge of generating device-specific code for Vivado HLS (``dace/codegen/target/xilinx.py``) or Intel FPGA OpenCL (``dace/codegen/target/intel_fpga.py``).

Vendor-specific semantics and syntax are handled by the two lower-level components triggered by the generic FPGA backend.

The FPGA code generation relies on the `HLSLIB <https://github.com/definelicht/hlslib>`_ external library to facilitate host/device interaction and HLS code generation.


Maps: pipelined and unrolled parallelism
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Maps are used to express parallel scopes in SDFGs.
In the context of FPGAs, we exploit this parallelism in two ways: pipelined and unrolled parallelism.

.. rubric::
    Pipeline parallelism

By default. maps are code-generated as pipelined loops, where iterations are executed in sequence, with the lowest II that can 
be attained by the compiler.
With the Intel OpenCL compiler, loops are automatically pipelined. For the Xilinx backend, proper pragmas are generated (``#pragma HLS pipeline``).


.. rubric::
    Unrolled (or spatial) parallelism

If a map is explicitly unrolled, this will be code generated as a loop with unrolling hints.
In this case, the compiler will unroll the loop, replicating the hardware and exploiting the spatial parallelism of the device.



Streams
^^^^^^^

Streams are DaCe containers that represent first-in, first-out queues. 
In FPGAs, they can be implemented in hardware (FIFOs) to exploit the on-chip resources and allow fast 
communication between different program components.

These containers and their related operations are generated differently for Xilinx and Intel FPGA:

    * for Xilinx FPGAs, streams are emitted in the top-level kernel function as local objects.
      Then they are passed as arguments to the producer and consumer accessing them.

    * for Intel FPGAs, they must be emitted to the global kernel scope, where the
      producer and consumer will read them directly (i.e., rather than receiving them as arguments).
      This would require, among the others, considering the case where different streams are defined
      using the same name. In this case, the Intel FPGA Code generator will mangle their name so 
      they can be uniquely identified in the program.

Finally, we should also consider the presence of streams that connect different FPGA kernels (see the section about FPGA kernels and processing elements).
In this case, they are defined either in the connectivity configuration file (``link.cfg``) that is passed to the Vitis compiler (Xilinx),
or in a shared header that is then included by the different kernels (Intel OpenCL).



Decoupled Memory interfaces 
^^^^^^^^^^^^^^^^^^^^^^^^^^^

When a container stored in the FPGA Device Memory (off-chip memory) is both read and written, DaCe, by default,
creates a single memory interface for both types of accesses.

While this has no particular performance impact on Intel, for Xilinx this could impair place and route step, resulting in 
a lower synthesis frequency.

For this reason, the programmer can set to true the DaCe configuration option ``DACE_compiler_fpga_xilixn_decouple_array_interfaces``.
This has an effect on the code generated for Xilinx. Any time that an array is If an array is both read and written, this option decouples 
its accesses by creating a memory interface for reading and one for writing. The array name is qualified and code generated with a ``_in`` or
``_out`` suffix, indicating the access directionality. 


*Warning*: while decoupling memory interfaces can improve performance, it must be used carefully. This may hide potential Read-After-Write or
Write-After-Read dependencies to the Vitis compiler, resulting in erroneous hardware. In addition to this, enabling the configuration could create up to 2 times the number of interfaces,
possibly reaching the limits supported by the device/Vitis.


.. _codegen_fpga_kernels:

FPGA Kernels and Processing Elements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the DaCe code generator backend encounters a state that only accesses containers situated on the FPGA, it designates it as an *FPGA kernel*
and triggers FPGA code generation (:func:`~dace.codegen.targets.fpga.FPGACodeGen.generate_state`).

Before continuing the traversal to generate the hardware itself, the kernel *boundary* is detected.
Here, DaCe supports two options:
    
    * by default, it will infer the entire SDFG state as an FPGA kernel. The DaCe code generator will generate each weakly connected
      component found in an SDFG state in a different *Processing Element*. Being independent, these SDFG components can be executed in parallel. 
      The notion of partitioning the functionality of a kernel into multiple independently-scheduled modules 
      is central to designing large FPGA architectures. 
        
    * if the ``DACE_compiler_fpga_concurrent_kernel_detection`` configuration option is set to ``True``, 
      a heuristic will further inspect each independent component for other parallelism opportunities (e.g., branches of the SDFG
      that can be executed in parallel). With this, inside the same state there could be multiple FPGA Kernels, that may depending
      on each other (e.g., a kernel must wait for the completion of a previous one before it can be executed). 


Once kernel boundaries are identified, the code generator  infers the necessary arguments that must be passed and generate 
host code call for kernel launches and synchronizations.

Regarding processing elements, in the Vivado HLS toolflow, processing elements are expressed by annotating a scope in the 
generated C++ code with the ``DATAFLOW`` pragma, resulting in every loop and function call in the scope to be scheduled 
as a distinct processing element.
Intel OpenCL has no distinction between processing elements and kernels. Therefore every processing element must be expressed as a 
separate OpenCL kernel. Launching each processing element is thus done directly from the host code.




Systolic Arrays
^^^^^^^^^^^^^^^
Systolic arrays are used to express parametric parallelism, by using an array of communicating processing elements that can be programmed to perform a common operation.

In a SDFG, 1D systolic arrays can be represented by unrolled maps in the outermost FPGA kernel scope.
The map can have a symbolic, but compile-time specialized, number of iterations, and must be coupled with array(s) of stream objects. 

When the map is unrolled, its body get replicated, and each instance becomes a weakly connected component in the state, resulting in them being instantiated as separate processing elements (see  :ref:`codegen_fpga_kernels`).


The actual code generation varies between Xilinx and Intel FPGA. In the former case, it is sufficient to unroll a loop in the C++ kernel code with bounds known at compile tim. For Intel, the OpenCL kernel representing the processing element is replicated and specialized directly in the generated code.


.. TODO: adding figure/example may help understanding what's going on.
