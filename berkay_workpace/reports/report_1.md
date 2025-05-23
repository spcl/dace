# Master's Thesis Report

**Thesis Title:** Code-generation for Modern GPUs in DaCe  
**Student:** Berkay Aydogdu  
**Supervisor:** Yakup Koray Budanaz  
**Date:** 2025-05-23  
**Short description:** The objectives of this Master's thesis are to refactor the CUDA code generator in DaCe and to extend it with new features. The refactoring focuses on improving the structure, readability, and maintainability of the code.

## Progress Overview

By inspecting the source code of the CUDA code generator, we identified several poor coding 
practices. These included, among others, intertwined functionality, non-descriptive variable 
and function names, and numerous code fragments that appeared more like quick fixes or hacks 
than thoughtfully designed solutions.

To address these issues, we implemented a new CUDA code generator class `ExperimentalCUDACodeGen`, which can be enabled via configuration settings. We began by
running simple programs using the new generator, reusing parts of the existing code to get 
minimal examples working.

We deliberately chose not to build a completely new generator from scratch, as improving code 
quality is only one part of the overall goal. Moreover, the existing implementation contains 
well-designed components that are worth preserving—there is no need to reinvent the wheel.

The following section highlights the notable aspects of the new implementation:

- Only simple features are supported for now, in order to eliminate the complexity introduced 
  by rarely used features such as dynamic parallelism.
- The generation of scopes — specifically GPU maps— has been almost completely reworked.
  In the existing CUDA code generator, this component has major issues, with several hundred 
  lines of dense code packed into just a few functions, even though it could be logically 
  split. For example, the generation of different map types (based on schedule types), the 
  kernel launch, and the kernel wrapper function are now implemented in separate functions. 
  We also improved naming throughout the code by replacing vague variable names with more 
  meaningful ones.
- The existing CUDA code generator opens and closes brackets in inconsistent   
  locations—sometimes even at another file. This is not only error-prone, but also makes
  the code appear more complex than necessary. To address this, we implemented a Python
  class (`KernelScopeManager`) that uses the `with` construct to clearly define when scopes 
  are  entered and exited, making bracket management more structured and easier to control.
- In our view, the existing CUDA code generator class relies on too many attributes, some of 
  which are specific to individual kernels—such as inputs, block and grid dimensions. These 
  are currently derived ad hoc and stored directly on the generator, leading to clutter and 
  reduced clarity. To address this, we introduced a `KernelSpec` class that encapsulates all 
  kernel-specific information. This allows such attributes to be accessed cleanly from a 
  KernelSpec instance, reducing the number of attributes in the code generator and improving 
  structure and maintainability.
- We also implemented a first extension, namely the support of WarpLevel schedules, by
  introducing a new GPU schedule type called `GPU_Warp`. With this, the we can specify which
  warps are selected to perform a task.


## Next Steps

The next steps include enabling asynchronous memory copies and continuing to refactor the
remaining parts of the code generator. This will require support for shared memory and
further discussions around key design decisions.




