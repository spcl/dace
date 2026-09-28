# CloudSC quickstart

Reproduce the four optimized CloudSC SDFGs and run each one against the un-transformed reference.

Status: canon_cpu, canon_gpu and vec_cpu match the reference; vec_gpu builds and runs but does not match
yet (see `doc/design/cloudsc_pipelines_and_vectorizer_perf.md`).

Tested on dace branch `extended` at commit `a57fab95228e82e216e1fc3db93ef20b02b4fe17`.

## Pipeline

```
cloudsc.py --to_sdfg(simplify=False)--> reference
reference --canonicalize(target='cpu')--------------------------> canon_cpu --VectorizeCPUMultiDim(w=8, host ISA)--> vec_cpu
reference --canonicalize(target='gpu') + offload_cloudsc_to_gpu--> canon_gpu --VectorizeGPU(w=2)-------------------> vec_gpu
```

Both canonicalizations bake in the species constants (`nclv=5`, `ncldq*`) and the two run-time flags
(`yrecldp_nssopt`, `yrecldp_laericesed`); `klev` / `klon` stay symbolic. Recipe: [reproduce.py](reproduce.py).

## Setup

```bash
git clone -b extended https://github.com/spcl/dace.git && cd dace
git checkout a57fab95228e82e216e1fc3db93ef20b02b4fe17
pip install -e .          # Python >= 3.10; islpy and z3-solver are required deps
export PYTHONHASHSEED=0 OMP_STACKSIZE=64M DACE_compiler_max_stack_array_size=65536
ulimit -s 65536           # CloudSC keeps large arrays on the stack
```

## Build the SDFGs

Each step writes `<out>/<step>.sdfgz` and reloads it on the next run, so the parse, the canonicalization
and the vectorization run once. Delete a file to rebuild that step (and everything built from it).

```bash
python -m tests.corpus.cloudsc.reproduce --out ~/.cache/cloudsc_repro canon_cpu vec_cpu canon_gpu vec_gpu
```

Outputs: `reference.sdfgz`, `canon_cpu.sdfgz`, `vec_cpu.sdfgz`, `canon_gpu.sdfgz`, `vec_gpu.sdfgz`.
Open any of them in the DaCe VS Code extension or load with `dace.SDFG.from_file`.

## Run a simulation

`--run` compiles each step, runs it on the physical input set of the dwarf (`klev = klon = 32`), and
compares every output array to the reference run sequentially under an IEEE build (`-O0`, no
fast-math, no FP contraction). Tolerance `1e-10`: parallel maps reorder reductions.

```bash
python -m tests.corpus.cloudsc.reproduce --out ~/.cache/cloudsc_repro --run canon_cpu vec_cpu   # host
python -m tests.corpus.cloudsc.reproduce --out ~/.cache/cloudsc_repro --run canon_gpu vec_gpu   # NVIDIA GPU
```

Each line reports build+run time, worst absolute error and `OK` / `MISMATCH`. The reference inputs and
outputs are cached in `<out>/reference_io.pkl`.

To call an SDFG yourself:

```python
import dace, pickle
inputs, expected = pickle.load(open('/path/to/out/reference_io.pkl', 'rb'))
sdfg = dace.SDFG.from_file('/path/to/out/vec_cpu.sdfgz')
args = {k: v for k, v in inputs.items() if k in sdfg.arglist() or k in map(str, sdfg.free_symbols)}
sdfg(**args)   # outputs are written into the arrays in args
```

## CI

`tests/corpus/cloudsc/cloudsc_target_pipelines_test.py` runs the same recipes with the same numeric
check (host legs in `integration-tests-ci.yml`, `cloudsc-host-pipelines`).
