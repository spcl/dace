.. _codegen:

Code Generation
===============

File structure (framecode etc.)

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

Modules / kernels + illustration

Memory interfaces (in/out)
