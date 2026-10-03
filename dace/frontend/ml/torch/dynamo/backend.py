# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The ``dace`` TorchDynamo backend."""
import contextlib
import dataclasses
import hashlib
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch

from .decompositions import build_decomposition_table


@dataclasses.dataclass
class DaceBackendOptions:
    sdfg_name: Optional[str] = None  #: Base name of the generated SDFGs (defaults to the compiled function's name)
    simplify: bool = True  #: Run ``SDFG.simplify`` after conversion
    validate: bool = True  #: Validate the generated SDFG
    auto_optimize: bool = False  #: Run ``dace.transformation.auto.auto_optimize`` before compiling
    onnx_fallback: bool = True  #: Fall back to ``dace.libraries.onnx`` expansions for unsupported operators
    extra_decompositions: Optional[Iterable] = None  #: Additional operators to decompose
    native_ops: Optional[Iterable] = None  #: Operators to keep intact (user-provided lowerings)
    save_sdfg: Optional[str] = None  #: Directory to save generated SDFGs into (for debugging)
    verbose: bool = False
    print_ops: bool = False  #: Print the post-decomposition operator histogram of each graph


class DaceBackend:
    """
    A TorchDynamo backend that lowers ATen graphs to DaCe.

    Instances are callables accepted by ``torch.compile(backend=...)``. The module-level :data:`dace_backend` instance
    is registered under the name ``'dace'``.
    """

    def __init__(self, **options):
        self.options = DaceBackendOptions(**options)
        # Configuration defaults for options not given explicitly
        from dace.config import Config
        for key in ('simplify', 'onnx_fallback'):
            if key not in options:
                try:
                    setattr(self.options, key, Config.get_bool('frontend', 'torch_dynamo', key))
                except (KeyError, TypeError, ValueError):  # pragma: no cover - missing schema entry
                    pass
        self._decomp_table: Optional[Dict] = None
        self.compile_count = 0  #: Number of SDFGs compiled by this backend (test oracle for compile-once behavior)
        self.last_sdfg = None  #: The most recently generated SDFG (before compilation)
        self.last_result = None  #: The most recent :class:`~dace.frontend.ml.torch.dynamo.importer.ImportResult`
        self._name_counter: Dict[str, int] = {}

    # Dynamo enters this context manager around tracing of frames compiled with this backend. The bytecode-level
    # control-flow capture (``cfg.goto_capture``) installs its translator patches here.
    backend_ctx_ctor = contextlib.nullcontext

    @property
    def decomposition_table(self) -> Dict:
        if self._decomp_table is None:
            self._decomp_table = build_decomposition_table(self.options.extra_decompositions, self.options.native_ops)
        return self._decomp_table

    def __call__(self, gm: torch.fx.GraphModule, example_inputs: List[Any]) -> Callable:
        from torch._dynamo.backends.common import aot_autograd
        return aot_autograd(fw_compiler=self._compile_forward,
                            bw_compiler=self._compile_backward,
                            decompositions=self.decomposition_table)(gm, example_inputs)

    # ------------------------------------------------------------------ compilers
    def _graph_name(self, gm: torch.fx.GraphModule) -> str:
        base = self.options.sdfg_name or 'dace_dynamo'
        digest = hashlib.sha1(gm.code.encode()).hexdigest()[:8]
        name = f'{base}_{digest}'
        n = self._name_counter.get(name, 0)
        self._name_counter[name] = n + 1
        return name if n == 0 else f'{name}_{n}'

    def _compile_forward(self, gm: torch.fx.GraphModule, example_inputs: List[Any]) -> Callable:
        from .importer import GraphImporter
        from .runtime import CompiledGraph

        if self.options.print_ops:
            _print_op_histogram(gm)
        importer = GraphImporter(self.options)
        result = importer.import_graph(gm, example_inputs, self._graph_name(gm))
        sdfg = result.sdfg
        self.last_sdfg = sdfg
        self.last_result = result
        if self.options.save_sdfg:
            import os
            os.makedirs(self.options.save_sdfg, exist_ok=True)
            sdfg.save(os.path.join(self.options.save_sdfg, sdfg.name + '.sdfgz'))
        if self.options.auto_optimize:
            from dace.transformation.auto import auto_optimize as aopt
            aopt.auto_optimize(sdfg, sdfg_device(sdfg))
        csdfg = sdfg.compile()
        self.compile_count += 1
        return CompiledGraph(csdfg, result)

    def _compile_backward(self, gm: torch.fx.GraphModule, example_inputs: List[Any]) -> Callable:
        # For now, backward graphs run eagerly. They are ATen graphs like the forward ones and can be lowered with the
        # same importer once training support is enabled.
        return gm.forward


def sdfg_device(sdfg):
    from dace import dtypes
    for desc in sdfg.arrays.values():
        if desc.storage == dtypes.StorageType.GPU_Global:
            return dtypes.DeviceType.GPU
    return dtypes.DeviceType.CPU


def _print_op_histogram(gm: torch.fx.GraphModule) -> None:
    from collections import Counter
    hist = Counter()
    for node in gm.graph.nodes:
        if node.op == 'call_function':
            hist[str(node.target)] += 1
        for sub in gm.children():
            pass
    print('Operator histogram:')
    for op, count in sorted(hist.items()):
        print(f'  {count:5d}  {op}')


#: Default backend instance, registered with Dynamo under the name ``'dace'``.
dace_backend = DaceBackend()

try:
    from torch._dynamo import register_backend as _register_backend
    _register_backend(compiler_fn=dace_backend, name='dace')
except Exception:  # pragma: no cover - registration is best-effort (e.g., already registered via entry point)
    pass
