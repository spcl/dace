# CloudSC optimization-pipeline e2e

Prove the three optimization pipelines keep CloudSC **numerically correct**, on CloudSC built three
different ways, checking after **every** subphase.

The matrix is **3 SDFGs × 3 pipelines = 9 cases**:

|              | `parallelize` | `canon_cpu` | `canon_gpu` (no offload) |
|--------------|:-:|:-:|:-:|
| `python`     | ✓ | ✓ | ✓ |
| `fortran`    | ✓ | ✓ | ✓ |
| `gpu_scc`    | ✓ | ✓ | ✓ |

Every case runs each pipeline **one subphase at a time**: apply → `validate()` → save the
post-subphase SDFGz. One numeric run at the end compares the fully-transformed graph against the
untransformed reference on the same inputs. The saved per-subphase SDFGz let you bisect any
divergence to the exact stage that caused it.

## Ownership split

- **dace-fortran side** (delivered separately, copied into this folder when ready): the `.F90`
  sources, `input file → SDFG` lowering, the physical **data initialization**, and the numeric
  **e2e runner** that proves the *generated* SDFG is e2e-correct.
- **this folder** — [`pipelines.py`](pipelines.py): the three pipelines and the
  apply/validate/save/cache **driver**. Frontend-agnostic; the data init and runner plug in through
  the `numeric_check` hook.

## 1. Input file → SDFG (dace-fortran) → verify

- **Input**: `cloudsc.F90` (CPU) or `cloudsc_gpu_scc_k_caching_mod.F90` (GPU SCC k-caching), plus
  its module dependencies and the `.func.h` saturation-function includes — all **co-located** so the
  frontend resolves everything from one place.
- **Generate**: lower the **compute body** `cloudsc_outer` (everything after I/O and init), *not* the
  I/O driver. flang runs with `-U_OPENMP -U_OPENACC`, so the pragmas are ignored and the sequential
  body lowers; the pipelines re-discover the parallelism.

  ```python
  from dace_fortran.build import build_sdfg_from_files
  files = ['cloudsc_gpu_scc_k_caching_mod.F90', 'cloudsc_modules_clean.F90', 'fcttre.func.h', 'fccld.func.h']
  sdfg = build_sdfg_from_files(files, entry='cloudsc_outer', out_dir='<tmp>')
  ```

- **Verify** (the dace-fortran unit test, copied here): run the generated SDFG on physical inputs and
  compare against the Fortran reference, single-core IEEE. This test asserts the *lowering* is
  correct **before** any pipeline touches it.

## 2. Load the data

The data init comes from the dace-fortran side; it reads the HDF5 physical profiles (far better than
uniform-random) and marshals each argument **by descriptor type**:

```python
from <data_init_module> import get_inputs_physical
inputs = get_inputs_physical(sdfg, seed=0)
```

- `dace.data.Scalar` → a Python scalar; a length-1 `Array` → a 1-element ndarray (marshal by
  descriptor type, **not** by `size == 1` — those two collide).
- `nblocks = 1` (single block). Level axis is keyed on the symbolic `klev` dimension (never on size —
  `klev == klon == 32` makes a size test pick the wrong axis).
- Shape/physics symbols (`klev`, `klon`, `nclv`, …) are resolved from the SDFG before marshaling.

## 3. Pipelines → post-pipeline SDFGz → verify

```python
from tests.corpus.cloudsc.pipelines import run_pipeline, uniquely_named, dump_root

sdfg = uniquely_named(sdfg, 'cloudsc_fortran')          # distinct name per frontend, no baseline collision
run_pipeline(sdfg, 'canon_cpu', dump_root(),
             constants={'nclv': 5},                     # config propagation (see below)
             numeric_check=runner)                      # from the dace-fortran e2e runner; None = structural-only
```

- Each subphase is applied, `validate()`-d (validate-all), and saved as
  `<tag>_<idx>_<label>.sdfgz` under `$CLOUDSC_E2E_DUMP` (default `$HOME/.cache/cloudsc_e2e`).
  Re-saving is skipped when an identical-hash SDFGz is already there — the **per-phase cache**.
- **Verify**: pass the dace-fortran runner as `numeric_check`. It runs the fully-transformed SDFG and
  the untransformed reference on the *same* physical inputs, single-core IEEE (`-O0 -fno-fast-math
  -ffp-contract=off`), and asserts they match to tolerance. Leave `numeric_check=None` to prove a
  pipeline merely *applies + validates* structurally without compiling.
- **Config propagation**: the `constants` argument bakes the parameter constants into the SDFG as the
  first `specialize` subphase (via `specialize_symbol`), so the pipelines optimize against fixed
  shapes/physics instead of free symbols.

## 4. What each pipeline does

Every variant starts with the `specialize` config-prop subphase, then:

- **`parallelize`** — `ParallelizePipeline`, 12 stages: peel/unroll, simplify, propagate
  memlets/symbols/constants, privatize scalars, then loop-to-reduce / accumulator-to-map+reduce and
  a final pattern-match sweep. Turns sequential loops into parallel maps.
- **`canon_cpu`** — `canonicalize(target='cpu')`, 164 stages: the full canonicalization + optimization
  stack (clean → loop/reduction normalization → … → codegen-ready), tuned for CPU.
- **`canon_gpu`** — `canonicalize(target='gpu')`, 165 stages: same stack plus the GPU knob, but it
  **structurally stops before GPU offload** — `offload_to_gpu`/`finalize_for_target` is *not* part of
  `_build_stages`, so the graph stays CPU-runnable and every subphase is validated on CPU. This
  exercises the GPU optimization path without needing a GPU to verify.

## Run

```bash
# structural, python column only (fortran/gpu SDFGs + runner arrive from the dace-fortran side):
OMP_NUM_THREADS=1 python -m tests.corpus.cloudsc.pipelines --variant parallelize

# full 9-case matrix, once the dace-fortran sources + data init + runner are copied in:
pytest tests/corpus/cloudsc/ -m integration -v -s -n1
```

Constraints: single-core (`OMP_NUM_THREADS=1`, IEEE flags above), 8 GB memory cap, MPI env vars set
for the dace scripts. SDFGz land under `$CLOUDSC_E2E_DUMP`.
