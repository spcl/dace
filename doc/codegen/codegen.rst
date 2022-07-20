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

The FPGA codegeneration is implemented by different modules, organized in a hierarchical way:

    * a generic backend (`dace/codegen/target/fpga.py`) is in charge of traversing the SDFG as shown in :ref:`codegen_how_it_works`;
    * two lower level components that are in charge of generating device specifc code for Vivado HLS, (`dace/codegen/target/xilinx.py`) or for Intel FPGA OpenCL (`dace/codegen/target/intel_fpga.py`).

All vendor specific semantic and syntax are handled by the two lower level components, which are triggered by the generic backend.

The FPGA code generation relies on the `HLSLIB https://github.com/definelicht/hlslib`_ external library to faciliate host/device interaction and HLS code generation.


FPGA Kernel Detection
^^^^^^^^^^^^^^^^^^^^^
When the DaCe code generator backend detects a state that only access containers situated on the FPGA, then designate it as an FPGA kernel and triggers FPGA code generation.






Memory interfaces (in/out)
^^^^^^^^^^^^^^^^^^^^^^^^^
