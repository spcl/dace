# npbench corpus port notes

Each `npbench/<dwarf>/<name>.py` is a self-contained fusion of four files from the
optarena source tree (`/home/primrose/Work/optarena/optarena/benchmarks/<group>/.../<name>/`):

- `<name>.py`            -> `initialize` (+ helpers like `rng_complex`)
- `<name>_numpy.py`      -> the numpy `reference` (+ helpers like `relu`, `conv2d`)
- `<name>_dace.py`       -> the `@dc.program` kernel (+ dace helpers + symbol defs)
- `<name>.yaml`          -> `sizes` (S preset), `input_args`, `array_args`,
                            `output_args`, `scalars`, dwarf

The loader (`npbench.py`) resolves every function parameter by NAME from the
initialized arrays + dataset scalars, runs the numpy reference and a compiled SDFG
of the kernel on identical inputs, and compares.

## Recurring porting pitfalls (fixed in this pass)

1. **Missing module-level helpers.** `initialize`/`reference` call helpers that live
   in the source `<name>.py` / `<name>_numpy.py` and must be copied in verbatim
   (`rng_complex`; numpy `relu`/`conv2d`/`maxpool2d`/`batchnorm2d`;
   `getAcc`/`getEnergy`; `build_up_b`/`pressure_poisson[_periodic]`).

2. **Numpy-reference vs dace-kernel helper name collisions.** Several kernels share a
   helper name between the numpy reference and the dace `@dc.program` (`conv2d`,
   `getAcc`, `build_up_b`, ...). The numpy reference must use the NUMPY versions --
   give them distinct `*_np` names so the reference doesn't pick up the dace program.

3. **dtype must match between `initialize` and the kernel.** The kernels declare
   `dc_float = float32` / `complex64`; if `initialize`'s default `datatype` is
   `float64`/`complex128`, DaCe reinterprets the bytes -> garbage (silent BASE!=REF).
   Keep `initialize`'s dtype consistent with the kernel (spmv, permute_3d), or widen
   the kernel to match the numpy reference (contour_integral, stockham_fft).

4. **Derived integers used as free SDFG symbols** (`C_before_fc1` in lenet,
   `Nt` in nbody, `N = R**K` in stockham_fft) must be bound explicitly (added to
   `sizes`) rather than carried as "arrays", because the kernel uses them in
   `np.reshape`/`np.ndarray` shapes that cannot be inferred from a transient.

5. **Output mapping.** `output_args` come from the program's return value (in order)
   or, when the slot's return is `None`, from the in-place-mutated array. Kernels that
   `return` a diagnostic (KE/PE, stepcount) instead of the validated fields must
   return the validated fields or nothing (nbody, channel_flow).

6. **Float dataset scalars.** The loader now includes float-valued `sizes` (and the
   `scalars` dict) in `params` so the reference/kernel can resolve them by name
   (mandelbrot `xmin`/`horizon`, nbody `dt`/`G`/`softening`, cavity/channel
   `rho`/`nu`/`F`). SDFG symbol binding still filters floats out separately.

7. **Size-cap collisions.** The harness caps int dataset symbols to 16. Where two
   distinct dims both exceed the cap and collapse to an equal value, set
   distinct sub-cap sizes so the reference takes the same code path as the kernel
   (contour_integral NR!=NM). Chaotic / slowly-converging kernels (nbody, channel_flow)
   need a small step/iteration count to keep the fp32 correctness check stable.
