# CloudSC optimization-pipeline e2e

Prove the three optimization pipelines keep CloudSC **numerically correct**, on CloudSC built three
different ways, checking after **every** subphase.

The matrix is **3 SDFGs × 3 pipelines = 9 cases**:

|              | `parallelize` | `canon_cpu` | `canon_gpu` (no offload) |
|--------------|:-:|:-:|:-:|
| `python`     | ✓ | ✓ | ✓ |
| `fortran`    | ✓ | ✓ | ✓ |
| `gpu_scc`    | ✓ | ✓ | ✓ |

Each pipeline is grouped into **phases** (consecutive stages sharing a stage label — so canon's
`loop_to_x` Loop2X lifts and its `parallelize` Loop2Map each form one phase; the `parallelize`
variant splits into `prep` / `loop_to_x` / `parallelize`). At **every phase boundary**: apply the
phase's stages → `validate()` → `numeric_check` against the untransformed reference → save a phase
SDFGz **checkpoint**. A checkpoint on disk therefore means "this phase passed"; a re-run loads the
furthest good checkpoint and resumes past it, so the multi-minute build/`simplify` is not repeated.
Every variant drives the loops to **maximum (sound) parallelism** (peeling + anti-dep breaking on).

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

- Each phase is applied, `validate()`-d, numeric-checked, then saved as
  `<tag>__<plansig>__p<idx>__<label>.sdfgz` under `$CLOUDSC_E2E_DUMP` (default `$HOME/.cache/cloudsc_e2e`).
  `<plansig>` digests the phase plan, so a changed pipeline mints fresh checkpoints instead of
  resuming from a stale one; re-saving is skipped when the identical-hash SDFGz is already there.
  Resume loads the furthest checkpoint present and continues past it (`resume=False` to force a
  full rerun).
- **Verify**: `numeric_check(sdfg, phase_name)` runs at every phase boundary. The self-contained
  python check comes from `make_numeric_check(reference)` (or the reusable
  `build_reference_outputs` + `numeric_check_from` pair to share one reference run across variants) —
  it runs the candidate and the untransformed reference on the *same* physical inputs, single-core
  IEEE (`-O0 -fno-fast-math -ffp-contract=off`), bit-exact on value-preserving phases and relaxed
  from the first reassociating phase onward. The dace-fortran runner plugs into the same
  `(sdfg, phase_name)` hook. Leave `numeric_check=None` to prove a pipeline merely *applies +
  validates* structurally without compiling.
- **Config propagation**: the `constants` argument bakes the parameter constants into the SDFG as the
  first `specialize` subphase (via `specialize_symbol`), so the pipelines optimize against fixed
  shapes/physics instead of free symbols.

## 4. What each pipeline does

Every variant starts with the `specialize` config-prop phase and a `pretreat` phase (simplify +
StateFusionExtended), then:

- **`parallelize`** — `ParallelizePipeline` in 3 phases: `prep` (peel/unroll, simplify, propagate
  memlets/symbols/constants, privatize scalars, aug-assign→WCR), `loop_to_x` (loop-to-reduce /
  accumulator-to-map+reduce), `parallelize` (loop-to-map). Turns sequential loops into parallel maps
  (peeling on for max parallelism).
- **`canon_cpu`** — `canonicalize(target='cpu')`, the full stack grouped by stage label into ~49
  phases (clean → loop/reduction normalization → `loop_to_x` → `parallelize` → fuse → … →
  codegen-ready), tuned for CPU.
- **`canon_gpu`** — `canonicalize(target='gpu')`: same stack plus the GPU knob, but it **structurally
  stops before GPU offload** — `offload_to_gpu`/`finalize_for_target` is *not* part of `_build_stages`,
  so the graph stays CPU-runnable and every phase is validated + numeric-checked on CPU. Exercises the
  GPU optimization path without needing a GPU to verify.

## Run

```bash
# structural only (fast, no compile), python column:
OMP_NUM_THREADS=1 python -m tests.corpus.cloudsc.pipelines --variant parallelize

# with the self-contained python numeric check at every phase boundary (compiles+runs per phase);
# --no-resume forces a full rerun, otherwise it resumes from saved checkpoints:
OMP_NUM_THREADS=1 python -m tests.corpus.cloudsc.pipelines --variant canon_cpu --numeric --regime ieee

# full 9-case matrix, once the dace-fortran sources + data init + runner are copied in:
pytest tests/corpus/cloudsc/ -m integration -v -s -n1
```

Constraints: single-core (`OMP_NUM_THREADS=1`, IEEE flags above), 8 GB memory cap, MPI env vars set
for the dace scripts. SDFGz land under `$CLOUDSC_E2E_DUMP`.
