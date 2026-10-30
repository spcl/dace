# Megakernel on ICON velocity tendencies

Compares `velocity_tendencies` with one launch per kernel against one megakernel, on the CPU and the GPU, on real ICON
R02B05 data (nproma 20480, 90 levels). The Fortran source comes from dace-fortran.

## Reproducing on Daint

```bash
# 1. Checkouts, next to each other
git clone -b megakernel git@github.com:spcl/dace.git
git clone git@github.com:spcl/dace-fortran.git

# 2. Python environment: dace-fortran's locked packages with this DaCe over them
cd dace-fortran
uv sync --frozen --no-default-groups --group testing
uv pip install -e ../dace
source .venv/bin/activate
export DACE_FORTRAN_ROOT=$PWD

# 3. The data set: 5.8 GB to download, about 40 GB unpacked
mkdir -p "$SCRATCH/data_r02b05"
curl -fL https://polybox.ethz.ch/index.php/s/WoxiditKJjgdEfR/download/nproma20480_data_files.tar.xz |
  tar -xJ -C "$SCRATCH/data_r02b05"
export DATA_DIR=$SCRATCH/data_r02b05

# 4. The experiment
cd ../dace/samples/megakernel_velocity
export SPACK_ROOT=<a spack with gcc, llvm 22 and cuda for aarch64>
sbatch -A <project> run_daint.sbatch
```

`run_beverin.sbatch` is the same on an MI300A node. `run_beverin_ranks.sbatch` splits the work over the four APUs of one
node, one rank each: CPU base, CPU megakernel, GPU base, GPU megakernel; it lands in the same place, with `rank*/` beside
the merged files. Results land in `results/<machine>-<job>/`: `verify.txt` is the
numerics check, `time.txt` the timings, and `times.csv` every repetition, one row each. `python plot.py
results/<machine>-<job>/times.csv --out plot.png` draws them (it needs matplotlib, and takes several CSVs to put machines
side by side). `OUT`, `REPS` (100), `DEVICES` and `CHUNKS` override the defaults.

`run_daint.sbatch` has not been run yet; everything here was tested on Beverin. On Daint, expect to adapt the `spack
load` line first, and check that DaCe picks the CUDA backend.

## Plotting the results

`times.csv` needs nothing done to it first. Any machine with Python and matplotlib can draw it:

```bash
python plot.py results/<machine>-<job>/times.csv --out plot.png
```

Each panel is one device and `(lvn_only, istep)`, with one group of bars per CPU schedule. Bars are the median per call
and whiskers the quartiles: `compute_s` on the GPU, `call_s` on the CPU. Pass several CSVs to compare machines. The
columns are `machine, device, kind, lvn_only, istep, schedule, timestep, rep, call_s, compute_s`, one row per repetition.

Check `verify.txt` before reading the plot: it shows times whether or not a kind reproduced the dumps.

## Reusing the SDFGs

`save_baselines.py --out <dir>` stops before compiling and writes one set per `(lvn_only, istep)`:

| file | state |
|------|-------|
| `velocity_l<lvn>i<istep>_optimized.sdfgz` | parallelized, on no device yet |
| `velocity_l<lvn>i<istep>_cpu.sdfgz` | levels shared by the team |
| `velocity_l<lvn>i<istep>_gpu.sdfgz` | offloaded |

None has a megakernel or a timer. Each has a `.bindings.json` beside it, with the Fortran interface `dace.SDFG.from_file`
does not restore. Load them with the DaCe that saved them.

```bash
python build_baseline.py velocity_l0i2_cpu.sdfgz --out build_cpu
python build_baseline.py velocity_l0i2_gpu.sdfgz --out build_gpu --chiplets 6  # on the compute node
```

`build_baseline.py` compiles a file into the library `velocity_driver.F90` links. Run it on the node that will run the
kernels: a login node targets the wrong GPU.

## What it builds

Every configuration is built twice per `(lvn_only, istep)`: `--strict`, for the numerics, and with optimizing flags, for
the times. The data set holds (0, 1), (0, 2) and (1, 1); (1, 2) never occurs.

| kind | meaning |
|------|---------|
| `base` | one launch per kernel, or one `omp parallel` per map on the CPU |
| `megakernel` | `MegaKernelOptimization` with its defaults |
| `tuned` | `--fewer-syncs`: the megakernel with grid barriers only between dependent stages |

Every GPU kernel uses thread blocks of 256,1,1. On the MI300A, `CHIPLETS=6` spreads the blocks of each `base` kernel over
the XCDs; a cooperative grid is launched as it is, so the megakernels are not spread.

The pipeline is dace-fortran's frontend, `specialize_variant` for the configuration, `optimize` (loops to maps,
`MoveIfIntoMap`, fusion), then offloading with the block maps on the host, then the megakernel.

## How the data gets in

The dumps hold ICON's derived types, field by field, as text. `velocity_driver.F90` loads them into those same types and
calls `velocity_tendencies_dace`, the binding dace-fortran generates: it has the kernel's signature and does the AoS to
SoA unpacking. The untouched Fortran kernel is linked next to it and runs on the same state. `dace_vs_t1` compares with
the dump after the step, and `n_differ=0` means bit-identical.

## Reading the times

`call` is the whole binding call. `compute` is the part between the last copy to the device and the first copy back. The
benchmark starts with its fields on the host, so every call copies about 60 containers, which ICON would not do; those
copies take most of `call` and are the same with and without the megakernel. Compare `compute`.

A base GPU build can end with a segmentation fault in `__call_tls_dtors` when Python exits, after its libraries are
written. `run_experiment.sh` therefore checks for the library, not the exit code.
