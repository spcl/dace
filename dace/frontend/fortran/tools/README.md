# Recipes

Before attempting any validation or integration of ICON with DaCe, please first make sure you can build ICON
and execute a simple experiment using ECRAD (step 1).
Then, proceed to generate the serialization header (step 2), instrument ICON and generate data (step 3),
and validate the selected DaCe implementation (step 4).

Additionally, type injection can be generated (step 42), but these are not always needed for serialization.

## 0. Get ICON input data.

Clone `icon-dace` and switch to branch `yakup/ICON_24_10_merge_v2`.

## 1. How to build ICON?

Briefly speaking, there are three options:

1. Dockerized build and dedicated build pipeline.
2. Out-of-build runs of ICON which do not currently work.
3. In-source build and run.

To perform the third type of the build, install NetCDF and configure ICON:

```
./configure CC=/usr/bin/gcc-13 FC=/usr/bin/gfortran-13 --disable-mpi FCFLAGS="-O2 -fallow-argument-mismatch -g -I/usr/include" LIBS="-lnetcdff -lnetcdf -llapack -lblas" --disable-coupling --enable-ecrad
make -j16
```

Then, modify the experiment script `run/exp.exclaim_ape_R2B09.run`:

- In `atmo_dyn_grid`, change the hardcoded path to refer to
  `icon-model/grids/icon_grid_${atmos_gridID}_${atmos_refinement}_G.nc` inside the `icon-dace` from previous step.
- To perform a quick experiment, we decrease the simulation to four timestamps:
  `end_date=${end_date:="2000-01-01T00:03:00Z"}`.

We execute the experiment by entering `run` directory and running `bash exp.exclaim_ape_R2B09.run`.

It should take around 1-2 minutes to execute the experiment, which should end without any errors and validate.

This step has been validated on the `release-2024.10-public` branch of `icon-dace` repository.

## 2. How to generate and use the serde Fortran module and C++ library?

> TLDR; 1) We generate a Fortran module `serde.F90` and a C++ header `serde.h` from _patched_ ECRAD, 2) put `serde.F90`
> in the _release_ ECRAD, 3) manually instrument `radiation_interface.F90` to use it, 4) then run ECRAD to have the data
> files, 5) put `serde.h` together with SDFG's C++ code, 6) use it to read those data files.

By "patched" ECRAD, we mean here branch `patched` from `icon-dace` repository.
By "release" ECRAD, we mean the ICON repository from step 1.

- Suppose we have the C++ preprocessed (but otherwise not transformed or prune) code in
  `~/gitspace/icon-dace-cpp-preprocessed/externals/ecrad` directory.
- Suppose we have an SDFG that is produced out of this preprocessed code.
- Suppose we have the "real" ECRAD code, that is a released version that we can build and run, in `~/gitspace/icon-dace`
  directory.
- Suppose we want to generate a Fortran module `serde.F90` that can serialise various data objects that is relevant for
  that SDFG into files, and we will keep this module in `externals/ecrad/utilities` directory of the real ECRAD.
- Suppose we also want to generate C++ header `serde.h` that can deserialise those data objects from the files, which we
  will use later to pass to the SDFG as arguments for testing numerically. And we want to put this module in
  `~/radiation/include`, the same directory where the generated header from the SDFG is located.
- We then need to run from the root of DaCe repository:
  ```shell
  python -m dace.frontend.fortran.tools.generate_serde_f90_and_cpp \
      -i ~/gitspace/icon-dace-for-serde/icon-dace/externals/ecrad \
      -g ~/radiation/radiation.sdfg \
      -f ~/gitspace/icon-dace/externals/ecrad/utilities/serde.F90 \
      -c ~/radiation/include/serde.h
  ```

## 3. How to generate input and output data of ICON?

Once the serialization module is placed inside ICON, we modify the code to serialize and output data necessary to
execute our DaCe implementation.

First, we insert imports:

```fortran
subroutine radiation(ncol, nlev, istartcol, iendcol, config, &
        &  single_level, thermodynamics, gas, cloud, aerosol, flux)

    use serde
    use type_injection
```

Then, we generate the input arguments needed for a selected ICON function:

```fortran
if (lhook) call dr_hook('radiation_interface:radiation', 0, hook_handle)

call tic()
if (generation ==1) then
    call serialize(at("config.data", asis = .true.), config)
    call serialize(at("single_level.data", asis = .true.), single_level)
    call serialize(at("istartcol.data", asis = .true.), istartcol)
    call serialize(at("iendcol.data", asis = .true.), iendcol)
    call serialize(at("sw_albedo_direct.data", asis = .true.), sw_albedo_direct)
    call serialize(at("sw_albedo_diffuse.data", asis = .true.), sw_albedo_diffuse)
    call serialize(at("lw_albedo.data", asis = .true.), lw_albedo)
endif
```

Finally, we serialize and save the output data of the function once it has been executed.

```fortran
if (generation ==1) then
    call serialize(at("sw_albedo_direct.after.data", asis = .true.), sw_albedo_direct)
    call serialize(at("sw_albedo_diffuse.after.data", asis = .true.), sw_albedo_diffuse)
    call serialize(at("lw_albedo.after.data", asis = .true.), lw_albedo)
endif
```

We execute the experiment using `bash exp.exclaim_ape_R2B09.run` inside `run`, and the results
can be found in `experiments/exclaim_ape_R2B09`.

## 4. How to valdate SDFG?

Once we have input and output data of the ICON function, we have to compile SDFG and build
the tester. The tester will load input arguments, execute the DaCe-generated implementation,
and compare the obtained results with the outputs from ICON.

Below, you can find an example of the `get_albedos` function of the `radiation_single` interface.

Build it as:

```
g++ -Iinclude -I .dacecache/get_albedos/include/ -I<dace-dir>/dace/runtime/include test.cpp -L .dacecache/get_albedos/build/ -lget_albedos -o test
```

where `include` contains `serde.h` from step 1, and `.dacecache` refers to the DaCe cache of a
directory where SDFG was built.

We execute it as:

```
LD_LIBRARY_PATH=.dacecache/get_albedos/build/ ./test icon-dace/experiments/exclaim_ape_R2B09/
```

Source file

```cpp
#include <fstream>
#include <iostream>
#include <filesystem>

#include "serde.h"

#include "get_albedos.h"

int main(int argc, char** argv)
{
  if(argc != 2)
    return 1;

  std::filesystem::path ROOT{argv[1]};

  int istartcol, iendcol;
  {
    std::ifstream data(ROOT / "istartcol.data");
    serde::deserialize(&istartcol, data);
  }

  {
    std::ifstream data(ROOT / "iendcol.data");
    serde::deserialize(&iendcol, data);
  }

  config_type config;
  {
    std::ifstream data(ROOT / "config.data");
    serde::deserialize(&config, data);
  }

  single_level_type single_level;
  {
    std::ifstream data(ROOT / "single_level.data");
    serde::deserialize(&single_level, data);
  }

  std::ifstream data_albedo_direct(ROOT / "sw_albedo_direct.data");
  auto [sw_alb_direct_meta, sw_albedo_direct] = serde::read_array<double>(data_albedo_direct);

  std::ifstream data_albedo_diffuse(ROOT / "sw_albedo_diffuse.data");
  auto [sw_alb_diff_meta, sw_albedo_diffuse] = serde::read_array<double>(data_albedo_diffuse);

  std::ifstream data_lw_albedo(ROOT / "lw_albedo.data");
  auto [lw_alb_meta, lw_albedo] = serde::read_array<double>(data_lw_albedo);

  auto* state = __dace_init_get_albedos(
      &config,
      lw_albedo,
      sw_albedo_diffuse,
      sw_albedo_direct,
      &single_level,
      1,
      iendcol,
      istartcol,
      iendcol,
      istartcol
  );

  __program_get_albedos(
      state,
      &config,
      lw_albedo,
      sw_albedo_diffuse,
      sw_albedo_direct,
      &single_level,
      1,
      iendcol,
      istartcol,
      iendcol,
      istartcol
  );

  __dace_exit_get_albedos(state);


  std::ifstream data_albedo_direct_after(ROOT / "sw_albedo_direct.after.data");
  auto [_, sw_albedo_direct_after] = serde::read_array<double>(data_albedo_direct_after);

  std::cerr << "Validate sw_albedo_direct " << sw_alb_direct_meta.size[0] << "x" << sw_alb_direct_meta.size[1] << std::endl;
  for(int i = 0; i < sw_alb_direct_meta.size[0]; ++i) {
    for(int j = 0; j < sw_alb_direct_meta.size[1]; ++j) {

      double original = sw_albedo_direct_after[sw_alb_direct_meta.size[1]*i + j];
      double ours = sw_albedo_direct[sw_alb_direct_meta.size[1]*i + j];

      if(fabs(ours - original) > 10e-9) {
        std::cerr << "Error! " << original << " " << ours << std::endl;
      }
    }
  }

  std::ifstream data_albedo_diffuse_after(ROOT / "sw_albedo_diffuse.after.data");
  auto [__, sw_albedo_diffuse_after] = serde::read_array<double>(data_albedo_diffuse_after);

  std::cerr << "Validate sw_albedo_diffuse " << sw_alb_diff_meta.size[0] << "x" << sw_alb_diff_meta.size[1] << std::endl;
  for(int i = 0; i < sw_alb_diff_meta.size[0]; ++i) {
    for(int j = 0; j < sw_alb_diff_meta.size[1]; ++j) {

      double original = sw_albedo_diffuse_after[sw_alb_diff_meta.size[1]*i + j];
      double ours = sw_albedo_diffuse[sw_alb_diff_meta.size[1]*i + j];

      if(fabs(ours - original) > 10e-9) {
        std::cerr << "Error! " << original << " " << ours << std::endl;
      }
    }
  }

  std::ifstream data_lw_albedo_after(ROOT / "lw_albedo.after.data");
  auto [___, lw_albedo_after] = serde::read_array<double>(data_lw_albedo_after);

  std::cerr << "Validate lw_albedo " << lw_alb_meta.size[0] << "x" << lw_alb_meta.size[1] << std::endl;
  for(int i = 0; i < lw_alb_meta.size[0]; ++i) {
    for(int j = 0; j < lw_alb_meta.size[1]; ++j) {

      double original = lw_albedo_after[lw_alb_meta.size[1]*i + j];
      double ours = lw_albedo[lw_alb_meta.size[1]*i + j];

      if(fabs(ours - original) > 10e-9) {
        std::cerr << "Error! " << original << " " << ours << std::endl;
      }
    }
  }

  return 0;
}


```

## 42. How to generate and use the type injector module?

> [!NOTE]
> This step only needs to be done once and is not necessary for standard serialization.

> TLDR; 1) We generate a module `ti.F90` from _patched_ ECRAD, 2) put it in the
_release_ ECRAD, 3) manually instrument `radiation_interface.F90` to use it, 4) then run ECRAD.

- Suppose we have the C++ preprocessed (but otherwise not transformed or prune) code in
  `~/gitspace/icon-dace-cpp-preprocessed/externals/ecrad` directory.
- Suppose we have the "real" ECRAD code, that is a released version that we can build and run, in `~/gitspace/icon-dace`
  directory.
- Suppose we want to generate a module `ti.F90` that can write down `config` etc. objects for "configuration injection"
  in SDFG optimisation for ECRAD, and we will keep this module in `externals/ecrad/utilities` directory of the real
  ECRAD.
- Suppose we want the generated module to have the name `type_injection`.
- We then need to run from the root of DaCe repository:
  ```shell
  python -m dace.frontend.fortran.tools.generate_type_injectors \
      -i ~/gitspace/icon-dace-cpp-preprocessed/externals/ecrad \
      -f ~/gitspace/icon-dace/externals/ecrad/utilities/ti.F90 \
      -m type_injection
  ```

> NOTE: This module relies on the `serde.F90` module for some basic serialisation functionality. Such functionality
> is present in any serialisation module as a baseline, so anything generated (at the same version of DaCe) will work.

## 42a. How to instrument the type injection module in the real ECRAD?

Its necessary changes should look like the following:

```
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

## 4. How to generate an FParser-preprocessed AST?

> TLDR; 1) We pick one or more functions to preserve in the preprocessed AST, 2) We pick some functions (or none) to
> make no-op (even if we cannot prune them with static analysis), 3) We generate a Fortran representation of the
> preprocessed AST that simplifies and prunes the code.

- Suppose we have the C++ preprocessed (but otherwise not transformed or prune) code in
  `~/gitspace/icon-dace-cpp-preprocessed/` directory.
- Suppose we want to _eventually_ create an SDFG from this code with some function as an entry point, for example,
  the function `velocity_tendencies` from the module `mo_velocity_advection`.
    - But for now we would just like to generate a preprocessed AST with a Fortran representation in
      `~/velocity_tendencies.f90`. This AST would be a functional Fortran library, and can be consumed to build an "
      internal AST".
- Suppose we know that only the following directories are needed for this entry point to build:
    - `~/gitspace/icon-dace-cpp-preprocessed/src`
    - `~/gitspace/icon-dace-cpp-preprocessed/support`
    - `~/gitspace/icon-dace-cpp-preprocessed/externals/fortran-support/src`
    - `~/gitspace/icon-dace-cpp-preprocessed/externals/cdi/src`
    - `~/gitspace/icon-dace-cpp-preprocessed/externals/mtime/src`
- Suppose we also know that certain functions (or subroutines) are not needed in the final SDFG, but it is
  not possible to prune them in the compile-time (yet). So we want to make them "no-op" during the preprocessing. For
  example, the following functions:
    - From the `mo_exception` module, the function `finish`.
    - From the `mo_real_timer` module, the functions `timer_start`, `timer_stop` and `new_timer`.
- We then need to run from the root of DaCe repository:
  ```shell
  python -m dace.frontend.fortran.tools.create_preprocessed_ast \
         -i ~/gitspace/icon-dace-cpp-preprocessed/src \
         -i ~/gitspace/icon-dace-cpp-preprocessed/externals/fortran-support/src \
         -i ~/gitspace/icon-dace-cpp-preprocessed/externals/cdi/src \
         -i ~/gitspace/icon-dace-cpp-preprocessed/externals/mtime/src \
         -i ~/gitspace/icon-dace-cpp-preprocessed/support \
         -o ~/velocity_tendencies.f90 \
         -k mo_velocity_advection.velocity_tendencies \
         --noop mo_exception.finish \
         --noop mo_real_timer.timer_start \
         --noop mo_real_timer.timer_stop \
         --noop mo_real_timer.new_timer
  ```

## 5. How to generate an SDFG from an FParser-preprocessed AST?

> TLDR; 1) We already have a preprocessed AST, 2) And we also have the config injection files, 3) We generate an SDFG
> using these.

- Suppose we have a self-contained FParser-preprocessed AST in `~/velocity_tendencies.f90`.
- Suppose we want to create a single SDFG in `~/velocity_tendencies.sdfg` from this AST with some entry point, for
  example, the function `velocity_tendencies` from the module `mo_velocity_advection`.
- Suppose we have already generated the config injection data (i.e., the `.ti` files) using the method described
  earlier, and all these files are now in the `~/dace/dace/frontend/fortran/conf_files` directory.
- We then need to run from the root of DaCe repository:
  ```shell
  python -m dace.frontend.fortran.tools.create_singular_sdfg_from_ast \
         -i ~/velocity_tendencies.f90 \
         -k mo_velocity_advection.velocity_tendencies \
         -o ~/velocity_tendencies.sdfg \
         -c ~/dace/dace/frontend/fortran/conf_files
  ```
