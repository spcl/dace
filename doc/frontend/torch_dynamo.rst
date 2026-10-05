.. _torch_dynamo:

PyTorch (TorchDynamo) Frontend
==============================

The TorchDynamo frontend compiles PyTorch modules and functions with DaCe through ``torch.compile``. It captures the
program with TorchDynamo and AOTAutograd, lowers the resulting ATen graph into a DaCe schedule tree, converts that
tree into an SDFG, and compiles the SDFG **once** per set of Dynamo guards. Tensor sizes and strides become DaCe
symbols, so the compiled program is reused for every input shape and layout that satisfies the guards.

.. note::

    This frontend requires PyTorch 2.13 or newer (``pip install dace[ml]``). It is independent of the ONNX-based
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

Choosing what is symbolic
-------------------------

With ``dynamic=True`` alone, Dynamo decides which sizes become symbols: sizes that are equal in the first call share
one symbol ("duck shaping"), sizes 0 and 1 are specialized, and the symbols are named ``s0``, ``s1``, .... The
``dynamic_shapes`` argument gives this control to the user, in the vocabulary of ``torch.export``:

.. code-block:: python

    # Everything but the weights is symbolic: each dimension of each argument gets its own symbol (x_dim0, x_dim1, ...)
    compiled = dace.ml.compile(model, dynamic_shapes='all')

    # Named dimensions; strings or torch.export.Dim objects (whose bounds are honored)
    batch = torch.export.Dim('batch', min=2)
    compiled = dace.ml.compile(model, dynamic_shapes={'x': {0: batch, 1: 'seq'}, 'cache': {0: batch, 1: 'cache_len'}})

The specification mirrors the arguments (a dict keyed by argument name or a tuple in positional order; nested lists
and dicts of tensors are mirrored; an integer argument takes a single name). The names become the names of the DaCe
symbols, which matters when the SDFG is used from a ``@dace.program`` or inspected. Dimensions of size 0 or 1 in the
first call stay static under ``'all'`` (PyTorch specializes them and treats size-1 operands as broadcasting), and two
dimensions only share a symbol if the program constrains them to be equal.

Modules inside DaCe programs
----------------------------

A ``torch.nn.Module`` can be called from a ``@dace.program`` directly. The module is captured with TorchDynamo for
the argument types of the call (including symbolic sizes, which keep their names) and nested into the program's SDFG,
so DaCe optimizes across the boundary:

.. code-block:: python

    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3)).eval()
    N = dace.symbol('N')

    @dace.program
    def prog(x: dace.float32[N, 4]):
        return model(x) * 2

Parameters and buffers are passed by reference, so updating them does not require parsing the program again. The
assumptions TorchDynamo made about the module (attribute values such as ``training``, submodule types, global state)
are part of the program's cache: if one changes, the program is parsed and compiled again. The module must be
captured as a single graph (graph breaks raise an error), and the captured graph is an inference graph.

To obtain the captured graph without compiling it, use :func:`dace.frontend.ml.torch.dynamo.capture`, which returns
the ATen graph, the source of every input (argument, parameter, buffer, ...), and TorchDynamo's guards.

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
  Dynamo itself (a graph break, or an error with ``fullgraph=True``). The experimental backend
  ``dace.frontend.ml.torch.dynamo.cfg.ControlFlowBackend`` instead captures data-dependent ``if``/``while`` and
  ``for`` loops over symbolic ranges or tensors (with ``break``, ``continue``, ``else`` clauses, and early returns) as a
  control-flow graph of traced blocks and compiles it into the SDFG. When gradients are required, such graphs are
  differentiated by DaCe's automatic differentiation (a forward SDFG, and a backward SDFG that recomputes it), which
  reverses branches and counting loops; loops that exit depending on tensor values cannot be differentiated yet.
* **Outputs.** Output tensors are allocated by torch with the strides torch expects, and the SDFG writes into them
  directly; inputs are passed by pointer without copies.
* **Training.** When gradients are required, AOTAutograd traces a joint forward and backward graph. By default the
  backend compiles it into a single SDFG with a forward and a backward phase (selected by the ``aot_phase`` symbol),
  so that one compiled library serves both calls. The values the forward saves for the backward are chosen by an
  AOTAutograd partition function (``partitioner=``; by default values are saved rather than recomputed, and
  ``torch._functorch.partitioners.min_cut_rematerialization_partition`` recomputes cheap operators instead). Saved
  values are returned to PyTorch between the two calls, so several forward calls may precede their backward calls.
  ``joint=False`` compiles separate forward and backward SDFGs.

  .. code-block:: python

      model = torch.compile(model, backend='dace', dynamic=True)
      loss = criterion(model(x), y)
      loss.backward()   # runs the backward phase of the same SDFG

* **Training inside programs.** A ``@dace.program`` that uses modules and returns a scalar loss can be differentiated
  with DaCe's automatic differentiation instead (``dace.autodiff``, which also differentiates control flow such as
  loops in the program). ``dace.ml.training_step(program)`` returns a callable that computes the loss and the
  gradients of the parameters (and of tensor arguments that require gradients) in one SDFG call and accumulates them
  into ``.grad``:

  .. code-block:: python

      @dace.program
      def loss_program(x: dace.float32[N, 8], y: dace.float32[N, 1]):
          difference = model(x) - y
          return np.sum(difference * difference)

      step = dace.ml.training_step(loss_program)
      for x, y in batches:
          optimizer.zero_grad()
          loss = step(x, y)
          optimizer.step()

  ``dace.ml.differentiable(program)`` instead makes a program a differentiable PyTorch function with a forward and a
  backward SDFG (``dace.autodiff.make_backward_pass``): its outputs can feed further PyTorch operations, and
  ``backward()`` runs the backward SDFG, accumulating gradients into the parameters of the modules it uses.

  .. code-block:: python

      block = dace.ml.differentiable(block_program)
      loss = torch.nn.functional.mse_loss(head(block(x)), y)
      loss.backward()

  See ``samples/ml`` for complete examples of both ways of training.

Limitations
-----------

* Dynamo specializes sizes equal to 0 or 1 and unifies equal sizes on the first call ("duck shaping"); a later call
  with a different pattern recompiles. ``dynamic_shapes`` (above) avoids the duck shaping; sizes 0 and 1 remain
  special.
* Data-dependent shapes (``nonzero``, ``.item()``) are not captured.
* Operators without a native lowering raise an error naming the operator; ``extra_decompositions`` and
  ``native_ops`` can be used to steer the decomposition table.
