## DaCe Samples

This folder contains various samples that showcase the capabilities of the DaCe framework.

There are several sub-folders here:
* **simple**: Simple examples using the Python/NumPy interface
* **explicit**: Examples that use the explicit data-centric API (`dace.map`, `dace.tasklet`), giving fine-grained control over hardware mapping
* **sdfg_api**: Programs that use the SDFG API to create the DaCe intermediate representation directly. Useful if you are writing your own frontend.
* **optimization**: Examples that use the transformation and instrumentation API to optimize programs to run fast on CPUs and GPUs
* **fpga**: FPGA programs with explicit circuit design patterns (e.g., systolic arrays), mostly using the SDFG API
* **distributed**: Python/NumPy and explicit applications that run on multiple machines
* **codegen**: Samples showing how to extend the code generator of DaCe to support new platforms (e.g., Tensor Cores)
