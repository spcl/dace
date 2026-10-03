.. _torch_dynamo:

PyTorch (TorchDynamo) Frontend
==============================

The TorchDynamo frontend compiles PyTorch modules and functions with DaCe through ``torch.compile``. It captures the
program with TorchDynamo and AOTAutograd, lowers the resulting ATen graph into a DaCe schedule tree, converts that
tree into an SDFG, and compiles the SDFG **once** per set of Dynamo guards. Tensor sizes and strides become DaCe
symbols, so the compiled program is reused for every input shape and layout that satisfies the guards.

.. note::

    This frontend requires PyTorch (``pip install dace[ml]``). It is independent of the ONNX-based
    :class:`~dace.frontend.ml.torch.module.DaceModule` and does not require ``onnx``.

Usage
-----

.. code-block:: python

    import torch
    import dace.ml

    model = MyModule().eval()
    compiled = dace.ml.compile(model)          # same as torch.compile(model, backend='dace', dynamic=True)

    with torch.no_grad():
        y = compiled(torch.randn(8, 128))       # compiles the SDFG
        y = compiled(torch.randn(13, 128))      # reuses it: the batch size is a symbol

``dace.ml.compile`` can also be used as a decorator on functions. Importing ``dace.ml`` registers the backend under
the name ``'dace'``, so ``torch.compile(fn, backend='dace', dynamic=True)`` works as well. Options (simplification,
auto-optimization, decomposition overrides, saving the generated SDFGs) are keyword arguments of
:func:`dace.frontend.ml.torch.dynamo.compile`; the backend instance used for a compiled object is available as
``compiled._dace_backend`` and exposes ``last_sdfg`` and ``compile_count``.

How programs are lowered
------------------------

* **Operators.** Composite ATen operators are decomposed (Inductor-style) into a small primitive set. Pointwise
  operators become maps with broadcasting, reductions use the standard-library ``Reduce`` node, matrix products use
  the BLAS ``MatMul`` node, convolutions and pooling become nested maps, and ``view``/``permute``/``slice``/``expand``
  become DaCe views that reuse the FakeTensor's strides. Type promotion follows ATen.
* **Dynamic shapes.** Every symbolic size and stride produced by Dynamo (``s0``, ``s1``, ...) is a DaCe symbol with
  the same name. Non-contiguous inputs therefore do not trigger recompilation either.
* **Control flow.** The higher-order operators ``torch.cond``, ``torch.while_loop``, ``torch._higher_order_ops.scan``
  and ``map`` are lowered to native DaCe control flow (``ConditionalBlock`` and ``LoopRegion``), including loops with
  symbolic trip counts and nested control flow. Plain Python ``if``/``for``/``while`` on tensor values is handled by
  Dynamo itself (a graph break, or an error with ``fullgraph=True``); capturing it natively is work in progress.
* **Outputs.** Output tensors are allocated by torch with the strides torch expects, and the SDFG writes into them
  directly; inputs are passed by pointer without copies.

Limitations
-----------

* Inference only for now; the backward graph produced by AOTAutograd runs eagerly.
* Dynamo specializes sizes equal to 0 or 1 and unifies equal sizes on the first call ("duck shaping"); a later call
  with a different pattern recompiles. Use ``torch._dynamo.mark_dynamic`` to avoid this where needed.
* Data-dependent shapes (``nonzero``, ``.item()``) are not captured.
* Operators without a native lowering raise an error naming the operator; ``extra_decompositions`` and
  ``native_ops`` can be used to steer the decomposition table.
