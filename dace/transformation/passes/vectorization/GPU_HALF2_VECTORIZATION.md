# Vectorizing an fp16 SDFG for GPU (half2 / FP16x2)

`VectorizeGPU` lowers an fp16 kernel to NVIDIA **half2** (FP16x2) SIMD: two fp16
lanes per instruction (`__hadd2`, `__hmul2`, `h2sin`, ...) inside a GPU kernel.
It is the K=1 multi-dim tile pipeline with three GPU choices fixed —
`target_isa="CUDA"`, `widths=(2,)` (a half2 is 2 lanes), and `assume_even=True`
(no remainder loop).

## Recipe

```python
import dace
from dace.transformation.interstate import LoopToMap
from dace.transformation.passes.vectorization.vectorize_gpu import VectorizeGPU

sdfg = my_program.to_sdfg(simplify=True)   # 1. build + simplify
sdfg.apply_transformations_repeated(LoopToMap)  # 2. loops -> maps
sdfg.simplify()                            # 3. (map fusion happens in simplify)

VectorizeGPU().apply_pass(sdfg, {})        # 4. half2 vectorize (GPU-schedules if needed)

sdfg.expand_library_nodes()                # 5. lower the tile lib nodes to cuda.h calls
sdfg.compile()                             # (or just sdfg.compile(), which expands too)
```

`VectorizeGPU` GPU-schedules the SDFG (`apply_gpu_transformations`) itself when no
`GPU_Device` map is present yet, so step 4 works on the CPU-scheduled maps that
steps 1–3 produce.

## fp16 scalars are NOT allowed as program inputs

Do **not** pass an fp16 scalar as a (non-transient) `@dace.program` argument:

```python
@dace.program
def bad(A: dace.float16[N], alpha: dace.float16, C: dace.float16[N]):  # ✗ alpha is a scalar input
    ...
```

An fp16 scalar has no stable by-value ABI across the host/device boundary. Pass
fp16 data as **arrays** only; keep fp16 scalars as **transients** (computed
inside the program), or pass the scalar as fp32/fp64 and cast it inside:

```python
@dace.program
def good(A: dace.float16[N], C: dace.float16[N]):
    for i in dace.map[0:N]:
        alpha = dace.float16(0.5)   # ✓ transient fp16 scalar
        C[i] = A[i] * alpha
```

## Constants

Write a numeric constant with an explicit cast to the tile precision —
`dace.float16(0.5)`. The pipeline keeps it at the **input precision** (never
fp64) and, for GPU, **duplicates** it into both half2 lanes (`__half2half2`) when
it is a broadcast operand. A bare `0.5` is cast to fp16 as well, but the explicit
cast documents intent and guarantees the half2 fast path fires (it needs a
`__half` tile).

## Even extents (`assume_even`)

The GPU path emits **no remainder loop**: every tiled map extent is assumed a
multiple of 2, so the map is a single `0:N:2` strided `GPU_Device` map (no masked
tail, no mismatched thread-block sizes). Ensure your innermost tiled dimension is
even. (If you need odd extents, run the CPU-style `VectorizeCPUMultiDim` with a
`masked_tail` remainder instead.)

## What lowers to half2

`cuda.h` provides native FP16x2 for: arithmetic `+ - * /`, `min`/`max`, all six
comparisons (`< <= > >= == !=`, each lane 1.0/0.0), `neg`/`abs`, and the
transcendentals/rounding `exp log sqrt sin cos floor ceil`. `tanh` (no `h2tanh`
intrinsic) and the logical `and`/`or` fall back to the per-lane scalar path;
masked/odd-width tiles also use the scalar path. fp8 has no native arithmetic and
computes through `float`.

## Deferred expansion

By default `VectorizeGPU` leaves the tile lib nodes (`TileBinop`, `TileLoad`, ...)
in the SDFG, each stamped with the `cuda` implementation, so you can inspect or
further-transform the vectorized form. Call `sdfg.expand_library_nodes()` (or
`sdfg.compile()`) to lower them to the cuda.h calls.
