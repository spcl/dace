# Recipes

## 1. How to generate and use the type injector module?

> TLDR; 1) We generate a module `ti.F90` from _patched_ ECRAD, 2) put it in the
_release_ ECRAD, 3) manually instrument `radiation_interface.F90` to use it, 4) then run ECRAD.

- Suppose we have the C++ preprocessed (but otherwise not transformed or prune) code in
  `~/gitspace/icon-dace-cpp-preprocessed/externals/ecrad` directory.
- Suppose we have the "real" ECRAD code, that is a released version that we can build and run, in `~/gitspace/icon-dace`
  directory.
- Suppose we want to generate a module `ti.F90` that can write down `config` etc. objects for "configuration injection"
  in SDFG optimisation for ECRAD, and we will keep this module in `externals/ecrad/utilities` directory of the real
  ECRAD.
- We then need to run from the root of DaCe repository:
  ```sh
  python -m dace.frontend.fortran.tools.generate_type_injectors \
      -i ~/gitspace/icon-dace-cpp-preprocessed/externals/ecrad \
      -f ~/gitspace/icon-dace/externals/ecrad/utilities/ti.F90
  ```

> NOTE: This module relies on the `serde.F90` module for some basic serialisation functionality. Such functionality
> is present in any serialisation module as a baseline, so anything generated (at the same version of DaCe) will work.

## 2. How to generate and use the serde Fortran module and C++ library?

> TLDR; 1) We generate a Fortran module `serde.F90` and a C++ header `serde.h` from _patched_ ECRAD, 2) put `serde.F90`
> in the _release_ ECRAD, 3) manually instrument `radiation_interface.F90` to use it, 4) then run ECRAD to have the data
> files, 5) put `serde.h` together with SDFG's C++ code, 6) use it to read those data files.

- Suppose we have the C++ preprocessed (but otherwise not transformed or prune) code in
  `~/gitspace/icon-dace-cpp-preprocessed/externals/ecrad` directory.
- Suppose we have an SDFG that is prudced out of this preprocessed code.
- Suppose we have the "real" ECRAD code, that is a released version that we can build and run, in `~/gitspace/icon-dace`
  directory.
- Suppose we want to generate a Fortran module `serde.F90` that can serialise various data objects that is relevant for
  that SDFG into files, and we will keep this module in `externals/ecrad/utilities` directory of the real ECRAD.
- Suppose we also want to generate C++ header `serde.h` that can deserialise those data objects from the files, which we
  will use later to pass to the SDFG as arguments for testing numerically. And we want to put this module in
  `~/radiation/include`, the same directory where the generated header from the SDFG is located.
- We then need to run from the root of DaCe repository:
  ```sh
  python -m dace.frontend.fortran.tools.generate_serde_f90_and_cpp \
      -i ~/gitspace/icon-dace-for-serde/icon-dace/externals/ecrad \
      -g ~/radiation/radiation.sdfg \
      -f ~/gitspace/icon-dace/externals/ecrad/utilities/serde.F90 \
      -c ~/radiation/include/serde.h
  ```
