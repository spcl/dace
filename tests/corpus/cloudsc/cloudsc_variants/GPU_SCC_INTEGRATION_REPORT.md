# CloudSC GPU SCC k-caching — frontend integration report

Integrating the dwarf-p-cloudsc GPU SCC k-caching files into the pipeline e2e tests. Sources,
module dependencies and includes are co-located in `cloudsc_variants/`.

**Lowered entry:** `cloudsc_scc_k_caching` — the computational body, post-IO/init. The multistep
(HDF5/MPI I/O) and the driver's timer/MPI wrapper are NOT lowered.

**Result:** the kernel lowers cleanly — `maps=0 tasklets=772 args=209`, validates. OpenACC pragmas
are ignored (flang `-U_OPENMP -U_OPENACC`), so the sequential body lowers and the
parallelize/canonicalize variants re-discover parallelism.

## What the frontend needed (none were frontend *bugs* — all missing/wrong imports)

Resolved by `build_sdfg_from_files([kernel, cloudsc_modules_clean.F90, *.func.h], entry=...)`.

1. **`fccld.func.h` / `fcttre.func.h` not found** — saturation statement-functions. Co-located.
2. **`parkind1.mod` not found → "must be a constant value" cascade** — the real/int KINDs. Supplied.
3. **`hdf5.mod` not found (via `file_io_mod`)** — `common/module` YOMCST/YOETHF/YOECLDP `USE FILE_IO_MOD`
   → `hdf5_file_mod`. The compute kernel needs only the module *constants*. Fixed by STRIPPING
   `USE FILE_IO_MOD` + the `CONTAINS` load routines, keeping declarations + the `TECLDP` type.
4. **`jprl` not found in parkind1** — the CPU `cloudsc.F90` PARKIND1 is `JPIM/JPRB`-only; the SC2026
   GPU variant `USE`s `JPRL/JPRM/JPRD`. Fixed with `common/module/parkind1.F90` (all kinds, no I/O).
5. **`rg/rd/... not found in yomcst`** — CPU `cloudsc.F90` passes constants as arguments, so its
   inlined YOMCST is empty of them. Fixed by the stripped `common/module` versions (step 3).
6. **`state_type` (YOMPHYDER → FIELD_MODULE)** — `USE`d but never referenced. One-type stub.

## Clean import set (`cloudsc_variants/cloudsc_modules_clean.F90`)

Full `PARKIND1` (all kinds) + I/O-stripped `YOMCST`/`YOETHF`/`YOECLDP` (declarations + `TECLDP` type)
+ `YOMPHYDER` stub. The module variables become SDFG free symbols → config-propagated.

## Not lowered, by design

`dwarf_cloudsc_gpu_multistep.F90` (HDF5/MPI I/O) and the driver's timer/MPI wrapper — the I/O and
init the user excluded ("lower the computational body post IO/init").
