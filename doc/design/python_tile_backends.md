# Python and tile DSL backends

An SDFG can compile to one Python function over NumPy/CuPy arrays instead of C++. A `GPU_Device` map over BLOCK tile
nodes (the output of `VectorizeGPU` / `VectorizeMultiDim(device=GPU, tile_group=BLOCK)`) becomes one kernel of a tile
DSL: Triton, cuTile or TileLang. The same tile kernel walker also writes CUDA Tile C++ kernels inside the C++/CUDA
generator. One program (tile block) runs per map iteration. Each tile node is one DSL statement.

## Pieces

- `codegen/python/prepare.py`: lowers the SDFG to the form the emitter takes. Expands library nodes with the `CuPy`
  implementation and runs `ControlFlowRaising`. It raises on control flow that stays unstructured or a tasklet that
  is not Python. Then it inlines host nested SDFGs and runs `InlineTaskletConnectors` and `UniqueNames`.
- `transformation/passes/unique_names.py`: `UniqueNames`. One Python function is one namespace, so a map parameter,
  connector, tasklet local or nested-SDFG name that would shadow an outer binding, a builtin or a module (`np`, `tl`,
  `ct`, `T`) is renamed `<name>_<N>`.
- `codegen/python/emit.py`: one recursive walker. Loops become `for`/`while`, conditional blocks become
  `if/elif/else`, maps become nested `for`, tasklet bodies are written verbatim, and transients are allocated at the
  top of the function. A GPU map is written as a tile kernel. A GPU map without tile nodes raises.
- `codegen/python/kernel.py`: `KernelEmitter` and the `Dialect` base. Register tiles are values, global memory moves
  only through `MaskedCopy`/`Gather`/`Scatter` nodes, and elementwise ops render from the `python` template of
  `libraries/tileops/ops.py` (`BINARY_OPS`, `UNARY_OPS`), one table for C++ and Python.
- `codegen/python/dialects/{triton,cutile,tilelang}.py`: per DSL, the imports, decorator, `program_id`, launch, and
  the function names that differ from the base.
- `codegen/python/dialects/cuda_tile_cpp.py`: CUDA Tile C++ (`__tile_global__`, launch `<<<grid, 1>>>`).
  `CppKernelEmitter` runs the shared walker and turns each statement into C++ through `cppunparse`.
  `TileKernelScopeGenerator` in `codegen/targets/experimental_cuda_helpers/scope_strategies.py` places the kernel.
  The GPU preprocessing passes skip these kernels (`is_tile_kernel`). Library expansion skips their nodes
  (`expand_outside_tile_kernels`).
- `libraries/python_expansions.py`: the `CuPy` expansion of Gemm, Gemv, Dot, Axpy, Ger, BatchedMatMul, Einsum,
  Cholesky, Solve, TensorTranspose, Reduce, ScatterConflictCheck, Copy and Fill. Each one is a single Python tasklet
  that uses `cupy` or `np`, depending on where the operands are stored.
- `codegen/python/compiled.py`: `PythonCompiledSDFG` imports the written module and calls it.

## Rules

- Tile kernels are BLOCK group only; THREAD and WARP tile nodes raise.
- Unstructured control flow and C++ tasklets raise; nothing falls back to C++.
- The default `compiler.backend: cpp` path is unchanged.
- Toolchain flags (`compiler.cuda.tile_args`) reach only builds that hold tile code (`CudaTileCppFlags`).

## How to use

```python
with dace.config.set_temporary("compiler", "backend", value="python"), \
     dace.config.set_temporary("compiler", "python", "tile_dialect", value="triton"):  # or cutile, tilelang
    sdfg.compile()(x=x, y=y, N=n)
```

For CUDA Tile C++ (needs CUDA 13.3 or newer, sm_80 or newer, C++20), keep `compiler.backend: cpp` and set
`compiler.cuda.tile_backend: cuda_tile`. Tests: `tests/codegen/python_backend/`, `tests/codegen/cuda_tile_cpp_test.py`
(pytest marks `triton`, `cutile`, `tilelang`, `cuda_tile`).
