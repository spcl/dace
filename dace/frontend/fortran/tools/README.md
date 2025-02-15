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
  ```commandline
  python -m dace.frontend.fortran.tools.generate_type_injectors \
      -i ~/gitspace/icon-dace-cpp-preprocessed/externals/ecrad \
      -f ~/gitspace/icon-dace/externals/ecrad/utilities/ti.F90
  ```

> NOTE: This module relies on the `serde.F90` module for some basic serialisation functionality. Such functionality
> is present in any serialisation module as a baseline, so anything generated (at the same version of DaCe) will work.

## 2. How to instrument the type injection module in the real ECRAD?

It necessary changes should look like the following:

```commandline
~/g/icon-dace (working-release|✚2) $ git diff externals/ecrad/radiation/radiation_interface.F90
diff --git a/externals/ecrad/radiation/radiation_interface.F90 b/externals/ecrad/radiation/radiation_interface.F90
index ab4e6fdc..4f76f6c9 100644
--- a/externals/ecrad/radiation/radiation_interface.F90
+++ b/externals/ecrad/radiation/radiation_interface.F90
@@ -204,6 +204,8 @@ contains
   subroutine radiation(ncol, nlev, istartcol, iendcol, config, &
        &  single_level, thermodynamics, gas, cloud, aerosol, flux)
 
+    use serde
+    use type_injection
     use parkind1,                 only : jprb
     use ecradhook,                  only : lhook, dr_hook, jphook
 
@@ -313,6 +315,17 @@ contains
 
     if (lhook) call dr_hook('radiation_interface:radiation',0,hook_handle)
 
+    call tic()
+    if (generation == 1) then
+      call type_inject(at("config.ti", asis=.true.), config)
+      call type_inject(at("single_level.ti", asis=.true.), single_level)
+      call type_inject(at("thermodynamics.ti", asis=.true.), thermodynamics)
+      call type_inject(at("gas.ti", asis=.true.), gas)
+      call type_inject(at("cloud.ti", asis=.true.), cloud)
+      call type_inject(at("aerosol.ti", asis=.true.), aerosol)
+      call type_inject(at("flux.ti", asis=.true.), flux)
+    endif
+
     !$ACC DATA CREATE(od_lw, ssa_lw, g_lw, od_lw_cloud, ssa_lw_cloud, g_lw_cloud, &
     !$ACC             od_sw, ssa_sw, g_sw, od_sw_cloud, ssa_sw_cloud, g_sw_cloud, &
     !$ACC             planck_hl, lw_emission, lw_albedo, sw_albedo_direct, &
```

Explanation:

- We called `tic()` to increase the version counter (which is a hidden integer `generation`). Both `tic()` and
  `generation` are from the `serde.F90` module.
- If this is the first generation (i.e., `generation == 1`), we call open a file with
  `at("some/where/config.ti", asis=.true.)`, and then write the "type injections" for `config_type` on the basis of
  `config` object (which is of that type). the `asis = .true.` argument is to preserve the file name exactly as given,
  without
  adding a version number to its name.
- Since we **ASSUME** that the relevant data in the `config` object do not change over the runtime of the program (
  sufficient for "instance injection"), and that the relevant data for _any_ object of `config_type` type will be the
  same for the runtime of the program.

## 3. How to generate and use the serde Fortran module and C++ library?

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
  ```commandline
  python -m dace.frontend.fortran.tools.generate_serde_f90_and_cpp \
      -i ~/gitspace/icon-dace-for-serde/icon-dace/externals/ecrad \
      -g ~/radiation/radiation.sdfg \
      -f ~/gitspace/icon-dace/externals/ecrad/utilities/serde.F90 \
      -c ~/radiation/include/serde.h
  ```
