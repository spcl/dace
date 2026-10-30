# Handoff: running the velocity experiment on Daint

For whoever submits the job. It compares `velocity_tendencies` with and without the megakernel on one GH200 node, on the
GPU and on the 72 Grace cores. `run_daint_ranks.sbatch`, in the README, replaces the job below.

## Run it

Follow "Reproducing on Daint" in `README.md`: two checkouts, the Python environment, the 40 GB data set, then

```bash
cd dace/samples/megakernel_velocity
export DACE_FORTRAN_ROOT=<dace-fortran checkout> DATA_DIR=$SCRATCH/data_r02b05
export SPACK_ROOT=<a spack with gcc 16, llvm 22 and cuda for aarch64>
sbatch -A <project> run_daint.sbatch
```

The job asks for 8 hours. To try it first, use `DEVICES=gpu REPS=5`, which builds fewer variants and repeats less.

## Which code to run

This branch uses the old CPU codegen and the older offloading. The numbers in hand come from the port to `extended`, so to
match them use dace branch `megakernel-extended` with dace-fortran branch `extended-compat`, and install both with
`uv pip install --no-deps -e` for dace-fortran, because its lock pins another DaCe.

On that port, `export DACE_compiler_cpu_implementation=legacy` before `sbatch` gave the fastest megakernels on Beverin, on
CPU and GPU. Leaving it out uses the new CPU codegen. Set `MACHINE=daint-legacy-cpu` for that run, so that both can be
plotted side by side.

## Expect to adapt

- The `spack load` line in `run_daint.sbatch`: its package versions are the Beverin ones.
- The GPU backend. All GPU runs so far used HIP; check that DaCe picks CUDA and that `sdfg/src/` holds CUDA kernels.
- A GPU build may end in a segmentation fault at Python exit. That is harmless if the library exists, and the script
  checks for the library.

## Check the result

Results land in `results/daint-<job>/`.

1. `verify.txt` must show `n_differ=0` for every kind and field. Do not quote times of a kind that fails.
2. `times.csv` has one row per repetition. Use `compute_s` on the GPU, since `call_s` is mostly copies.
3. "Plotting the results" in `README.md` draws it, alone or beside the Beverin CSV.
