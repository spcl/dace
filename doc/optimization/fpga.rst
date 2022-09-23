FPGA Optimization Best Practices
================================



Maps and parallelism
--------------------

In DaCe maps are used to express parallel scopes in SDFGs.
In the context of FPGAs, we distinguish between:

    * *pipelined* maps, where iterations are executed in sequence, exploiting pipeline parallelism in the mapped computation; 
    * *unrolled* maps, which represent parametrically replicated hardware, such as systolic arrays or SIMD-style vectorization.

By default, maps are code-generated as pipeline-loops. The user can switch to unrolled maps by changing their schedule (either
programmatically or through the VSCode plugin).



Exploit FPGA Memory hierarchy
-----------------------------


Streams and how to exploit them
-------------------------------
In FPGAs, streams are implemented in hardware (FIFOs) to exploit the on-chip resources and allow fast 
communication between different program components.


.. Talk more about streams, how to define them, what characterizes them and how to transform


Library Node and FPGA specialization
------------------------------------


FPGA Kernels and Processing Elements
------------------------------------

In DaCe, a state that only accesses containers situated on the FPGA will trigger FPGA code generation.

In DaCe we hierachically organize the code in *FPGA Kernels*, that can be further dividided in multiple *Processing elements*.
These concepts will be mapped to different entities depending on the used FPGA backend (see :ref:`codegen_fpga_kernels`).


By default, an SDFG state with only FPGA containers is inferred as an FPGA kernel. Then, each of the weakly connected component
found in the state are treated as different Processing Elements, that can be executed in parallel.
The notion of partitioning the functionality of a kernel into multiple independently-scheduled modules is central to designing large FPGA architectures, and can be exploited to write systolic arrays.

If the ``DACE_compiler_fpga_concurrent_kernel_detection`` configuration option is set to ``True``, 
a heuristic will further inspect each independent component for other parallelism opportunities (e.g., branches of the SDFG
that can be executed in parallel). If this is the case, multiple, possibly depending, FPGA Kernels are generated for the same state.



