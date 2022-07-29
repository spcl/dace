.. _codegen:

Code Generation
===============

File structure (framecode etc.)

.. _codegen_how_it_works:

How it Works
------------

How it works (accompany gif + links to modules)
    * preprocess
    * CFG
    * framecode
    * allocation management + decisions regarding AllocationLifetime
    * defined variables
    * traversal + delegation (dispatching with example)
    * concurrency (e.g., OMP sections/tasking, GPU streams)

Link to Tensor Core tutorial

.. image:: codegen.gif

Important features


.. _debug_codegen:

Debugging Code Generation
-------------------------

_dacegraphs/failing.sdfg codegen failure

Generated code but need to make a small change? Enable ``use_cache`` to stop re-generating code, make change, go to ``build`` and run ``make``.


conditional breakpoints

codegen lineinfo config

The code generator can be debugged in different ways. If the code generation process itself is failing, 
the Python debugger would be enough to try and understand what is going on. 

For compiler errors, 
there are multiple ways of approaching this.

If there is a particular hint in the error (array name, kernel) that can pinpoint which node / memlet might be
problematic, it would be advisable to place a conditional breakpoint on the function that generates that kind of node,
conditioned on the specific label you are looking for. For example, if a map named "kernel_113" in the Xilinx codegen
is suspected as the source of the issue, I would place a breakpoint in the MapEntry code generator, e.g., here.

If it is hard to understand from which line in the code generator the issue has emerged, we created a configuration entry (disabled by default) that adds, to each line in the generated code, the code generator file/line that generated it. It is compiler.codegen_lineinfo in .dace.conf, or enabled through setting the envvar DACE_compiler_codegen_lineinfo=1. It will then create a source map in the appropriate .dacecache subfolder (where compilation failed) that you can explore through reading it or through the Visual Studio Code extension.


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



FPGA Kernels and Processing Elements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. _codegen_fpga_kernel:


When the DaCe code generator backend encounters a state that only accesses containers situated on the FPGA, it designates it as an *FPGA kernel*
and triggers FPGA code generation (:func:`~dace.codegen.targets.fpga.FPGACodeGen.generate_state`).

Before continuing the traversal to generate the hardware itself, the kernel *boundary* is detected.
Here, DaCe supports two options:
    
    * by default, it will infer the entire SDFG state as an FPGA kernel. The DaCe code generator will generate each weakly connected
      component found in an SDFG state in a different *Processing Element*. Being independent, these SDFG components can be executed in parallel. 
      The notion of partitioning the functionality of a kernel into multiple independently-scheduled modules 
      is central to designing large FPGA architectures. 
        
    * if the ``DACE_compiler_fpga_concurrent_kernel_detection`` configuration option is set to True, 
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





