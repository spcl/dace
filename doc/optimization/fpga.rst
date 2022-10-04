FPGA Optimization Best Practices
================================

In the following, we provide guidance on leveraging DaCe functionalities to optimize DaCe programs targeting FPGAs.
Once that the user program is parsed into an SDFG, the performance engineer can optimize (transform) it for the sake of improving
performance. In the case of FPGA programs, best-practice and transformations can be applied to reduce data-movements, and to
increase spatial parallelism. 


Maps and parallelism
--------------------

In DaCe maps are used to express parallel scopes in SDFGs.
In the context of FPGAs, we distinguish between:

    * *pipelined* maps, where iterations are executed in sequence, exploiting pipeline parallelism in the mapped computation; 
    * *unrolled* maps, which represent parametrically replicated hardware, such as systolic arrays or SIMD-style vectorization.

By default, maps are code-generated as pipeline-loops. The user can switch to unrolled maps by changing their schedule (either
programmatically or through the VSCode plugin).



FPGA Memory hierarchy
-----------------------------

Modern FPGAs are characterized by having small fast on-chip memory, and large, but slower, off-chip memory.

DaCe allows to specify for each FPGA container, where it should be allocated by specifying its :py:data:`~dace.dtypes.StorageType`, either programmatically
or through the VSCode plugin. We can distinguish between:

  * *global* memory (:py:data:`~dace.dtypes.StorageType.FPGA_Global`), which represents data present in off-chip, memory-mapped storage such as DDR or HBM. 
    Containers in global memory can be created/accessed from both the host and the device side;
  * *local* memory (:py:data:`~dace.dtypes.StorageType.FPGA_Local`), representing any on-chip memory implementation such as registers, BRAM/M20K, 
    LUTRAM, or UltraRAM. Which one will be actually used is left up to the HLS compiler;
  * *register* memory (:py:data:`~dace.dtypes.StorageType.FPGA_Register`), which is a subset of local memory, but forces the compiler to implement it 
    as register (LUT), allowing parallel read/write to the container. This can be useful in the presence of unrolled maps.


.. TODO: introduce also Shift Register

Streams and how to exploit them
-------------------------------
In FPGAs, streams are implemented in hardware (FIFOs) to exploit the on-chip resources and allow fast 
communication between different program components.


.. Talk more about streams, how to define them, what characterizes them and how to transform and the requirements


Library Nodes and FPGA specialization
-------------------------------------


FPGA Kernels and Processing Elements
------------------------------------

In DaCe, a state that only accesses containers situated on the FPGA will trigger FPGA code generation.

In DaCe we hierarchically organize the code in *FPGA Kernels*, that can be further dividided in multiple *Processing elements*.
These concepts will be mapped to different entities depending on the used FPGA backend (see :ref:`codegen_fpga_kernels`).


By default, an SDFG state with only FPGA containers is inferred as an FPGA kernel. Then, each of the weakly connected component
found in the state are treated as different Processing Elements, that can be executed in parallel.
The notion of partitioning the functionality of a kernel into multiple independently-scheduled modules is central to designing large FPGA architectures, and can be exploited to write systolic arrays.

If the ``DACE_compiler_fpga_concurrent_kernel_detection`` configuration option is set to ``True``, 
a heuristic will further inspect each independent component for other parallelism opportunities (e.g., branches of the SDFG
that can be executed in parallel). If this is the case, multiple, possibly depending, FPGA Kernels are generated for the same state.


Suggested transformation for FPGA programs
------------------------------------------
.. Streaming, MapFusion and coalescing, auto-opt