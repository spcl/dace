# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The ``dace`` TorchDynamo backend."""
import contextlib
import dataclasses
import functools
import hashlib
import os
from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch
from torch._dynamo.backends.common import aot_autograd
from torch.nn.attention import SDPBackend, sdpa_kernel

from dace import dtypes
from dace.config import Config

from .decompositions import build_decomposition_table
from .importer import GraphImporter
from .runtime import CompiledGraph
from .sources import GraphDescription, describe_graph


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
    #: Normalized ``dynamic_shapes`` specification (see :mod:`~dace.frontend.ml.torch.dynamo.shapes`); used to name
    #: the DaCe symbols after the user's dimension names. Argument marking is done by :func:`~.interface.compile`.
    dynamic_shapes: Any = None
    signature: Any = None  #: ``inspect.Signature`` of the compiled callable (for mapping arguments to symbols)


class DaceBackend:
    """
    A TorchDynamo backend that lowers ATen graphs to DaCe.

    Instances are callables accepted by ``torch.compile(backend=...)``. The module-level :data:`dace_backend` instance
    is registered under the name ``'dace'``.
    """

    def __init__(self, **options):
        self.options = DaceBackendOptions(**options)
        # Configuration defaults for options not given explicitly
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
        #: Sources, symbol names, and guards of the most recent Dynamo-level graph
        self.last_description: Optional[GraphDescription] = None
        self.last_dynamo_graph: Optional[torch.fx.GraphModule] = None  #: The most recent Dynamo-level graph

    @property
    def symbol_names(self) -> Dict[str, str]:
        """Dynamo symbol -> DaCe symbol name of the most recent graph."""
        return dict(self.last_description.symbol_names) if self.last_description is not None else {}

    # Dynamo enters this context manager around tracing of frames compiled with this backend. The bytecode-level
    # control-flow capture (``cfg.goto_capture``) installs its translator patches here.
    backend_ctx_ctor = contextlib.nullcontext

    @property
    def decomposition_table(self) -> Dict:
        if self._decomp_table is None:
            self._decomp_table = build_decomposition_table(self.options.extra_decompositions, self.options.native_ops)
        return self._decomp_table

    def __call__(self, gm: torch.fx.GraphModule, example_inputs: List[Any]) -> Callable:
        # The Dynamo-level graph knows which argument, parameter, or buffer each placeholder came from; the AOT graphs
        # below do not (their placeholders correspond to the Dynamo-level ones positionally)
        description = describe_graph(gm, self.options.signature, self.options.dynamic_shapes)
        self.last_description = description
        self.last_dynamo_graph = gm
        # AOTAutograd may compile the backward graph lazily (on the first backward call), after other graphs.
        # Attention is traced with the math kernel, so that its backward consists of operators the frontend lowers.
        compiler = aot_autograd(fw_compiler=functools.partial(self._compile_forward, description=description),
                                bw_compiler=functools.partial(self._compile_backward, description=description),
                                decompositions=self.decomposition_table)
        with sdpa_kernel(SDPBackend.MATH):
            return compiler(gm, example_inputs)

    # ------------------------------------------------------------------ compilers
    def _graph_name(self, gm: torch.fx.GraphModule, suffix: str = '') -> str:
        base = self.options.sdfg_name or 'dace_dynamo'
        digest = hashlib.sha1(gm.code.encode()).hexdigest()[:8]
        name = f'{base}_{suffix}_{digest}' if suffix else f'{base}_{digest}'
        n = self._name_counter.get(name, 0)
        self._name_counter[name] = n + 1
        return name if n == 0 else f'{name}_{n}'

    def _compile_forward(self,
                         gm: torch.fx.GraphModule,
                         example_inputs: List[Any],
                         description: Optional[GraphDescription] = None) -> Callable:
        description = description or self.last_description
        input_names = description.input_names() if len(description.inputs) == len(example_inputs) else None
        return self._compile_graph(gm, example_inputs, self._graph_name(gm), description.symbol_names, input_names)

    def _compile_backward(self,
                          gm: torch.fx.GraphModule,
                          example_inputs: List[Any],
                          description: Optional[GraphDescription] = None) -> Callable:
        # The inputs of a backward graph are saved values and output gradients, which have no user-facing names
        description = description or self.last_description
        return self._compile_graph(gm, example_inputs, self._graph_name(gm, 'backward'), description.symbol_names, None)

    def _compile_graph(self, gm: torch.fx.GraphModule, example_inputs: List[Any], name: str,
                       symbol_names: Dict[str, str], input_names: Optional[List[Optional[str]]]) -> Callable:
        if self.options.print_ops:
            _print_op_histogram(gm)
        importer = GraphImporter(self.options)
        result = importer.import_graph(gm, example_inputs, name, symbol_names=symbol_names, input_names=input_names)
        sdfg = result.sdfg
        self.last_sdfg = sdfg
        self.last_result = result
        if self.options.save_sdfg:
            os.makedirs(self.options.save_sdfg, exist_ok=True)
            sdfg.save(os.path.join(self.options.save_sdfg, sdfg.name + '.sdfgz'))
        if self.options.auto_optimize:
            from dace.transformation.auto import auto_optimize as aopt  # Deferred: imports the transformation library
            aopt.auto_optimize(sdfg, sdfg_device(sdfg))
        csdfg = sdfg.compile()
        self.compile_count += 1
        return CompiledGraph(csdfg, result)


def sdfg_device(sdfg):
    for desc in sdfg.arrays.values():
        if desc.storage == dtypes.StorageType.GPU_Global:
            return dtypes.DeviceType.GPU
    return dtypes.DeviceType.CPU


def _print_op_histogram(gm: torch.fx.GraphModule) -> None:
    hist = Counter()
    for node in gm.graph.nodes:
        if node.op == 'call_function':
            hist[str(node.target)] += 1
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
