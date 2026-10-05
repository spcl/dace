# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The ``dace`` TorchDynamo backend."""
import collections
import contextlib
import dataclasses
import functools
import hashlib
import os
import warnings
from collections import Counter
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Tuple

import torch
from torch._dynamo.backends.common import aot_autograd
from torch._functorch.aot_autograd import make_boxed_func
from torch._functorch.partitioners import default_partition
from torch.nn.attention import SDPBackend, sdpa_kernel

from dace import data, dtypes
from dace.autodiff import make_backward_pass
from dace.config import Config

from .decompositions import build_decomposition_table
from . import joint as jt
from .context import UnsupportedOpError
from .importer import GraphImporter, PhaseInterface
from .runtime import CompiledGraph, DifferentiableGraph
from .training import compile_pair
from .sources import GraphDescription, describe_graph

#: Dynamo settings for tracing programs that DaCe compiles: ``.item()`` stays in the graph as a data-dependent symbol
#: (assigned from data in the SDFG) instead of breaking the graph
CAPTURE_CONFIG = {'capture_scalar_outputs': True}


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
    #: Compile each training graph into one SDFG with a forward and a backward phase (see :mod:`.joint`), instead
    #: of one SDFG per partitioned graph
    joint: bool = True
    #: AOTAutograd partition function that picks the values the forward saves for the backward (defaults to
    #: ``default_partition``, which saves instead of recomputing)
    partitioner: Optional[Callable] = None


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
        #: Joint plans by forward graph code, recorded by the partitioner until the forward graph is compiled
        self._joint_plans: Dict[str, Deque[jt.JointPlan]] = collections.defaultdict(collections.deque)
        #: Backward phases of compiled joint SDFGs by backward graph code, until AOTAutograd asks for them
        self._joint_backward: Dict[str, Deque[CompiledGraph]] = collections.defaultdict(collections.deque)

    @property
    def symbol_names(self) -> Dict[str, str]:
        """Dynamo symbol -> DaCe symbol name of the most recent graph."""
        return dict(self.last_description.symbol_names) if self.last_description is not None else {}

    # Dynamo enters this context manager around tracing of frames compiled with this backend. (It is not reachable
    # through ``torch.compile`` in torch 2.13+, which wraps the backend, so ``cfg.blocks`` patches Dynamo globally.)
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
        if _has_captured_control_flow(gm) and _requires_gradients(example_inputs):
            # AOTAutograd cannot differentiate captured control flow (an opaque operator): DaCe's autodiff can
            return self._compile_with_dace_autodiff(gm, example_inputs, description)
        # AOTAutograd may compile the backward graph lazily (on the first backward call), after other graphs.
        # Attention is traced with the math kernel, so that its backward consists of operators the frontend lowers.
        compiler = aot_autograd(fw_compiler=functools.partial(self._compile_forward, description=description),
                                bw_compiler=functools.partial(self._compile_backward, description=description),
                                partition_fn=self._partition,
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

    def _partition(self, joint: torch.fx.GraphModule, joint_inputs: Any,
                   **kwargs) -> Tuple[torch.fx.GraphModule, torch.fx.GraphModule]:
        """AOTAutograd's partition function: partitions the joint graph and records it for joint compilation."""
        forward, backward = (self.options.partitioner or default_partition)(joint, joint_inputs, **kwargs)
        if self.options.joint:
            plan = jt.plan_from_partition(joint, forward, backward, kwargs['num_fwd_outputs'])
            if plan is None:
                warnings.warn('The partitioned training graph cannot be compiled jointly; compiling the forward and '
                              'backward graphs separately')
            else:
                self._joint_plans[forward.code].append(plan)
        return forward, backward

    def _compile_forward(self,
                         gm: torch.fx.GraphModule,
                         example_inputs: List[Any],
                         description: Optional[GraphDescription] = None) -> Callable:
        description = description or self.last_description
        input_names = description.input_names() if len(description.inputs) == len(example_inputs) else None
        plans = self._joint_plans.get(gm.code)
        if plans:
            try:
                return self._compile_joint(plans.popleft(), self._graph_name(gm), description.symbol_names, input_names)
            except UnsupportedOpError as ex:
                warnings.warn(f'Compiling the forward and backward graphs separately: {ex}')
        return self._compile_graph(gm, example_inputs, self._graph_name(gm), description.symbol_names, input_names)

    def _compile_backward(self,
                          gm: torch.fx.GraphModule,
                          example_inputs: List[Any],
                          description: Optional[GraphDescription] = None) -> Callable:
        compiled = self._joint_backward.get(gm.code)
        if compiled:  # The backward phase of a joint SDFG
            return compiled.popleft()
        # The inputs of a backward graph are saved values and output gradients, which have no user-facing names
        description = description or self.last_description
        return self._compile_graph(gm, example_inputs, self._graph_name(gm, 'backward'), description.symbol_names, None)

    def _compile_joint(self, plan: jt.JointPlan, name: str, symbol_names: Dict[str, str],
                       input_names: Optional[List[Optional[str]]]) -> Callable:
        """Compiles a training graph into one SDFG; returns its forward phase and keeps its backward phase."""
        if self.options.print_ops:
            _print_op_histogram(plan.joint)
        result = GraphImporter(self.options).import_joint(plan, name, symbol_names, input_names)
        csdfg = self._compile_sdfg(result.sdfg)
        self.last_result = result
        backward = CompiledGraph(csdfg, result.backward.inputs, result.backward.outputs,
                                 _phase_arguments(result.sdfg, result.backward, result.forward))
        self._joint_backward[plan.backward.code].append(backward)
        return CompiledGraph(csdfg, result.forward.inputs, result.forward.outputs,
                             _phase_arguments(result.sdfg, result.forward, result.backward))

    def _compile_graph(self, gm: torch.fx.GraphModule, example_inputs: List[Any], name: str,
                       symbol_names: Dict[str, str], input_names: Optional[List[Optional[str]]]) -> Callable:
        if self.options.print_ops:
            _print_op_histogram(gm)
        importer = GraphImporter(self.options)
        result = importer.import_graph(gm, example_inputs, name, symbol_names=symbol_names, input_names=input_names)
        csdfg = self._compile_sdfg(result.sdfg)
        self.last_result = result
        return CompiledGraph(csdfg, result.inputs, result.outputs)

    def _compile_with_dace_autodiff(self, gm: torch.fx.GraphModule, example_inputs: List[Any],
                                    description: GraphDescription) -> Callable:
        """
        Compiles a training graph into a forward and a backward SDFG with DaCe's automatic differentiation (instead of
        AOTAutograd's backward graph), and returns a function that records them for PyTorch's autograd.
        """
        graphs = []

        def record(aten_gm, inputs):
            graphs.append((aten_gm, inputs))
            return make_boxed_func(aten_gm.forward)

        # The forward ATen graph (AOTAutograd in inference mode)
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            aot_autograd(fw_compiler=record, decompositions=self.decomposition_table)(gm, example_inputs)
        aten_gm, aten_inputs = graphs[0]
        input_names = description.input_names() if len(description.inputs) == len(aten_inputs) else None
        name = self._graph_name(aten_gm)
        result = GraphImporter(self.options).import_graph(aten_gm,
                                                          aten_inputs,
                                                          name,
                                                          symbol_names=description.symbol_names,
                                                          input_names=input_names)
        self.last_result = result
        if any(spec.kind != 'tensor' for spec in result.outputs):
            raise UnsupportedOpError('autodiff', 'the graph returns values other than new tensors')
        outputs = [spec.name for spec in result.outputs]
        differentiated = [
            spec.name for spec in result.inputs if spec.kind == 'tensor'
            and isinstance(example_inputs[spec.position], torch.Tensor) and example_inputs[spec.position].requires_grad
        ]
        self.last_sdfg = result.sdfg
        # The backward SDFG recomputes the forward pass: the data the captured control flow decides on (e.g., branch
        # predicates) need not be forwarded between the two SDFGs
        backward_pass = make_backward_pass(result.sdfg, outputs=outputs, inputs=differentiated, recompute_forward=True)
        pair = compile_pair(backward_pass, outputs, differentiated)
        self.compile_count += 2
        return DifferentiableGraph(pair, result.inputs, result.outputs)

    def _compile_sdfg(self, sdfg) -> Any:
        self.last_sdfg = sdfg
        if self.options.save_sdfg:
            os.makedirs(self.options.save_sdfg, exist_ok=True)
            sdfg.save(os.path.join(self.options.save_sdfg, sdfg.name + '.sdfgz'))
        if self.options.auto_optimize:
            from dace.transformation.auto import auto_optimize as aopt  # Deferred: imports the transformation library
            aopt.auto_optimize(sdfg, sdfg_device(sdfg))
        csdfg = sdfg.compile()
        self.compile_count += 1
        return csdfg


def _has_captured_control_flow(gm: torch.fx.GraphModule) -> bool:
    # By name: the operator is only registered once control-flow capture (``cfg``) is imported
    return any(node.op == 'call_function' and str(node.target) == 'dace.cfg.default' for node in gm.graph.nodes)


def _requires_gradients(example_inputs: List[Any]) -> bool:
    return torch.is_grad_enabled() and any(isinstance(x, torch.Tensor) and x.requires_grad for x in example_inputs)


def _phase_arguments(sdfg, interface: PhaseInterface, other: PhaseInterface) -> Dict[str, Any]:
    """
    The arguments of a joint SDFG that one phase does not get from its caller: the phase itself, ``None`` for the
    arrays of the other phase, and a placeholder value for the symbols only the other phase defines (and uses).
    """
    given = {spec.name
             for spec in interface.inputs} | {spec.name
                                              for spec in interface.outputs if spec.kind == 'tensor'}
    other_symbols = {spec.name for spec in other.inputs if spec.kind == 'sym'}
    arguments: Dict[str, Any] = {jt.PHASE_SYMBOL: interface.phase}
    for name, desc in sdfg.arglist().items():
        if name in given or name in arguments:
            continue
        if isinstance(desc, data.Array):
            arguments[name] = None
        elif name in other_symbols:
            arguments[name] = 0
    return arguments


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
