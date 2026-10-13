# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
ATen operators as DaCe library nodes.

The TorchDynamo frontend turns every ATen operator that it does not lower itself (views, pointwise maps, reductions,
...) into one :class:`ATenOp`: the tensor operands are input connectors named after the operator schema (``input``,
``weight``, ...; ``<name>_<i>`` for tensor lists), the results are output connectors ``output`` (``output_<i>`` for
tuples), and the remaining arguments are kept in :attr:`ATenOp.arguments`. An operator without a registered expansion
stays in the SDFG and fails only when the SDFG is expanded, naming the operator. Implementations are added per
operator by subclassing (see :class:`Convolution`) and registering expansions with
:func:`dace.library.register_expansion`.

This module does not import torch, so SDFGs containing these nodes load and expand without it.
"""

import copy
from typing import Any

from dace import SDFG, SDFGState, library, nodes, properties, symbolic
from dace.memlet import Memlet
from dace.transformation import transformation as xf

#: Node class by operator name (``aten.convolution.default``); operators not listed use :class:`ATenOp`
_NODE_CLASSES: dict[str, type["ATenOp"]] = {}


def aten_node(*targets: str):
    """Class decorator: a library node (see :func:`dace.library.node`) that stands for the given ATen operators."""

    def decorator(cls):
        cls = library.node(cls)
        for target in targets:
            _NODE_CLASSES[target] = cls
        return cls

    return decorator


def node_class(target: str) -> type["ATenNode"]:
    """The library node class that stands for an ATen operator name."""
    return _NODE_CLASSES.get(target, ATenOp)


def registered_targets() -> list[str]:
    """ATen operator names with a dedicated library node class."""
    return list(_NODE_CLASSES)


@properties.make_properties
class ATenNode(nodes.LibraryNode):
    """
    Base class of the ATen library nodes (see the module documentation). Not a library node itself: the library
    registry keys nodes by class, so the generic :class:`ATenOp` and the operator-specific nodes are siblings.
    """

    target = properties.Property(dtype=str, default="", desc="Operator name, e.g. 'aten.convolution.default'")
    arguments = properties.Property(
        dtype=dict,
        default=None,
        allow_none=True,
        to_json=lambda d: d,
        from_json=lambda d, context=None: d,
        desc="Non-tensor arguments by schema name (numbers, booleans, strings, lists; symbolic values as strings)",
    )

    def __init__(self, name: str, target: str = "", arguments: dict[str, Any] | None = None, **kwargs):
        super().__init__(name, **kwargs)
        self.target = target
        self.arguments = dict(arguments or {})

    def expand(self, *args, **kwargs) -> str:
        if not type(self).implementations:
            raise NotImplementedError(
                f"No DaCe implementation is registered for the ATen operator {self.target} (library node "
                f"{type(self).__name__} '{self.name}'). Register an expansion with dace.library.register_expansion "
                "or decompose the operator (extra_decompositions)."
            )
        return super().expand(*args, **kwargs)


@library.node
class ATenOp(ATenNode):
    """A call of an ATen operator that has no dedicated node class (see the module documentation)."""

    implementations = {}
    default_implementation = None


# ---------------------------------------------------------------------------------------------- expansion helpers
def as_symbolic(value: Any) -> Any:
    """Parses the strings in an argument value (recursively) as symbolic expressions."""
    if isinstance(value, str):
        return symbolic.pystr_to_symbolic(value)
    if isinstance(value, (list, tuple)):
        return [as_symbolic(v) for v in value]
    return value


def int_list(value: Any, length: int) -> list[int]:
    """Normalizes an ``int`` or ``List[int]`` argument (e.g. ``stride``) to a list of ``length`` integers."""
    value = as_symbolic(value)
    if not isinstance(value, (list, tuple)):
        value = [value]
    value = [int(v) for v in value]
    return value * length if len(value) == 1 and length > 1 else value


def expansion_sdfg(node: ATenNode, parent_state: SDFGState, parent_sdfg: SDFG) -> tuple[SDFG, SDFGState]:
    """
    An SDFG for an expansion of ``node`` with one array per connector, named after the connector and shaped like the
    memlet connected to it (strides and storage of the connected container).
    """
    sdfg = SDFG(node.name + "_expansion")
    state = sdfg.add_state()
    for edge in parent_state.in_edges(node):
        _add_connector_array(sdfg, parent_sdfg, edge.dst_conn, edge.data, edge.data.src_subset)
    for edge in parent_state.out_edges(node):
        _add_connector_array(sdfg, parent_sdfg, edge.src_conn, edge.data, edge.data.dst_subset)
    return sdfg, state


def _add_connector_array(sdfg: SDFG, parent_sdfg: SDFG, name: str, memlet: Memlet, subset) -> None:
    desc = parent_sdfg.arrays[memlet.data]
    subset = copy.deepcopy(subset)
    if subset.dims() == len(desc.shape):
        strides = list(desc.strides)
    else:  # The memlet addresses a slice of lower rank: keep the strides of the remaining dimensions
        kept = subset.squeeze()
        strides = [s for i, s in enumerate(desc.strides) if i in kept]
    sdfg.add_array(name, subset.size(), desc.dtype, strides=strides, storage=desc.storage)


def connect(state: SDFGState, libnode: nodes.LibraryNode, inputs: dict[str, str], outputs: dict[str, str]) -> None:
    """
    Adds ``libnode`` to ``state`` and connects it to the arrays named in ``inputs``/``outputs`` (array name ->
    connector name); arrays that do not exist in the SDFG (optional operands) are skipped.
    """
    state.add_node(libnode)
    arrays = state.sdfg.arrays
    for array, conn in inputs.items():
        if array in arrays:
            libnode.add_in_connector(conn)
            state.add_edge(state.add_read(array), None, libnode, conn, Memlet.from_array(array, arrays[array]))
    for array, conn in outputs.items():
        if array in arrays:
            libnode.add_out_connector(conn)
            state.add_edge(libnode, conn, state.add_write(array), None, Memlet.from_array(array, arrays[array]))


# ---------------------------------------------------------------------------------------------- convolution
@aten_node("aten.convolution.default", "aten._convolution.default")
class Convolution(ATenNode):
    """
    ``aten.convolution(input, weight, bias?, stride, padding, dilation, transposed, output_padding, groups)``.

    Connectors ``input``, ``weight``, optional ``bias`` and ``output``; the remaining arguments are in
    :attr:`~ATenOp.arguments`.
    """

    implementations = {}
    default_implementation = "ml"


@library.register_expansion(Convolution, "ml")
class ExpandConvolutionML(xf.ExpandTransformation):
    """Delegates to :class:`dace.libraries.ml.Convolution` (pure, cuDNN or MIOpen implementations)."""

    environments = []

    @staticmethod
    def expansion(node: Convolution, parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        from dace.libraries import ml  # Deferred: only needed to expand, not to load SDFGs

        sdfg, state = expansion_sdfg(node, parent_state, parent_sdfg)
        args = node.arguments
        nd = len(sdfg.arrays["weight"].shape) - 2
        conv = ml.Convolution(node.name)
        conv.strides = int_list(args.get("stride", 1), nd)
        conv.pads = int_list(args.get("padding", 0), nd)
        conv.dilations = int_list(args.get("dilation", 1), nd)
        conv.output_padding = int_list(args.get("output_padding", 0), nd)
        conv.groups = int(as_symbolic(args.get("groups", 1)))
        conv.transposed = bool(args.get("transposed", False))
        connect(state, conv, {"input": "_x", "weight": "_w", "bias": "_b"}, {"output": "_y"})
        return sdfg


# ---------------------------------------------------------------------------------------------- dropout
@aten_node("aten.native_dropout.default")
class NativeDropout(ATenNode):
    """
    ``aten.native_dropout(input, p, train) -> (output, mask)``.

    Connectors ``input``, ``output_0`` (the output) and ``output_1`` (the mask; absent when unused). The random
    numbers come from a counter-based generator keyed by :attr:`seed` (a symbol lets every call draw a fresh stream)
    and :attr:`offset` (distinct per node of a program).
    """

    implementations = {}
    default_implementation = "ml"

    seed = properties.SymbolicProperty(default=0, desc="Seed (key) of the random stream")
    offset = properties.SymbolicProperty(default=0, desc="Offset of this node in the random stream")


@library.register_expansion(NativeDropout, "ml")
class ExpandNativeDropoutML(xf.ExpandTransformation):
    """Delegates to :class:`dace.libraries.ml.Dropout`; with ``train=False`` copies the input and sets the mask."""

    environments = []

    @staticmethod
    def expansion(node: NativeDropout, parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        sdfg, state = expansion_sdfg(node, parent_state, parent_sdfg)
        train = node.arguments.get("train", True)
        if train is not None and not train:
            desc = sdfg.arrays["input"]
            state.add_nedge(state.add_read("input"), state.add_write("output_0"), Memlet.from_array("input", desc))
            if "output_1" in sdfg.arrays:
                mask = sdfg.arrays["output_1"]
                params = {f"__i{k}": f"0:{s}" for k, s in enumerate(mask.shape)}
                index = ", ".join(params)
                state.add_mapped_tasklet(
                    "mask", params, {}, "__out = 1", {"__out": Memlet(f"output_1[{index}]")}, external_edges=True
                )
            return sdfg

        from dace.libraries import ml  # Deferred: only needed to expand, not to load SDFGs

        drop = ml.Dropout(node.name)
        drop.p = as_symbolic(node.arguments.get("p", 0.5))
        drop.seed = node.seed
        drop.offset = node.offset
        connect(state, drop, {"input": "_x"}, {"output_0": "_y", "output_1": "_mask"})
        return sdfg


# ---------------------------------------------------------------------------------------------- pooling
def _pool_params(args: dict, nd: int) -> tuple[list[int], list[int], list[int]]:
    """Normalizes ``kernel_size``, ``stride`` (empty means ``kernel_size``) and ``padding`` to lists of ``nd`` ints."""
    kernel = int_list(args.get("kernel_size", 1), nd)
    stride = args.get("stride", [])
    stride = list(kernel) if not stride else int_list(stride, nd)
    pads = int_list(args.get("padding", 0), nd)
    return kernel, stride, pads


@aten_node(
    "aten.max_pool1d_with_indices.default",
    "aten.max_pool2d_with_indices.default",
    "aten.max_pool3d_with_indices.default",
)
class MaxPoolWithIndices(ATenNode):
    """
    ``aten.max_poolNd_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode) -> (values, indices)``.

    Connector ``self`` (input), ``output_0`` (values) and ``output_1`` (indices; absent when unused).
    """

    implementations = {}
    default_implementation = "ml"


@library.register_expansion(MaxPoolWithIndices, "ml")
class ExpandMaxPoolWithIndicesML(xf.ExpandTransformation):
    """Delegates to :class:`dace.libraries.ml.MaxPool`."""

    environments = []

    @staticmethod
    def expansion(node: MaxPoolWithIndices, parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        from dace.libraries import ml  # Deferred: only needed to expand, not to load SDFGs

        sdfg, state = expansion_sdfg(node, parent_state, parent_sdfg)
        args = node.arguments
        nd = len(sdfg.arrays["self"].shape) - 2
        kernel, stride, pads = _pool_params(args, nd)
        pool = ml.MaxPool(node.name)
        pool.kernel_shape = kernel
        pool.strides = stride
        pool.pads = pads
        pool.dilations = int_list(args.get("dilation", 1), nd)
        pool.ceil_mode = bool(args.get("ceil_mode", False))
        connect(state, pool, {"self": "_x"}, {"output_0": "_y", "output_1": "_indices"})
        return sdfg


@aten_node("aten.avg_pool1d.default", "aten.avg_pool2d.default", "aten.avg_pool3d.default")
class AvgPool(ATenNode):
    """
    ``aten.avg_poolNd(self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override) -> output``.

    Connector ``self`` (input) and ``output``.
    """

    implementations = {}
    default_implementation = "ml"


@library.register_expansion(AvgPool, "ml")
class ExpandAvgPoolML(xf.ExpandTransformation):
    """Delegates to :class:`dace.libraries.ml.AvgPool`."""

    environments = []

    @staticmethod
    def expansion(node: AvgPool, parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        from dace.libraries import ml  # Deferred: only needed to expand, not to load SDFGs

        sdfg, state = expansion_sdfg(node, parent_state, parent_sdfg)
        args = node.arguments
        nd = len(sdfg.arrays["self"].shape) - 2
        kernel, stride, pads = _pool_params(args, nd)
        divisor = args.get("divisor_override", None)
        pool = ml.AvgPool(node.name)
        pool.kernel_shape = kernel
        pool.strides = stride
        pool.pads = pads
        pool.ceil_mode = bool(args.get("ceil_mode", False))
        pool.count_include_pad = bool(args.get("count_include_pad", True))
        pool.divisor_override = int(as_symbolic(divisor)) if divisor is not None else 0
        connect(state, pool, {"self": "_x"}, {"output": "_y"})
        return sdfg


# ---------------------------------------------------------------------------------------------- max-pool backward
@aten_node("aten.max_pool2d_with_indices_backward.default", "aten.max_pool3d_with_indices_backward.default")
class MaxPoolWithIndicesBackward(ATenNode):
    """
    ``aten.max_poolNd_with_indices_backward(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode,
    indices) -> grad_input``.

    Connectors ``grad_output``, ``self`` (shape only, unused), ``indices`` and ``output``.
    """

    implementations = {}
    default_implementation = "ml"


@library.register_expansion(MaxPoolWithIndicesBackward, "ml")
class ExpandMaxPoolWithIndicesBackwardML(xf.ExpandTransformation):
    """Delegates to :class:`dace.libraries.ml.MaxPoolBackward` (``self`` is only needed for the output shape)."""

    environments = []

    @staticmethod
    def expansion(node: MaxPoolWithIndicesBackward, parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        from dace.libraries import ml  # Deferred: only needed to expand, not to load SDFGs

        sdfg, state = expansion_sdfg(node, parent_state, parent_sdfg)
        bwd = ml.MaxPoolBackward(node.name)
        connect(state, bwd, {"grad_output": "_dy", "indices": "_indices"}, {"output": "_dx"})
        return sdfg


# ---------------------------------------------------------------------------------------------- convolution backward
@aten_node("aten.convolution_backward.default")
class ConvolutionBackward(ATenNode):
    """
    ``aten.convolution_backward(grad_output, input, weight, bias_sizes, stride, padding, dilation, transposed,
    output_padding, groups, output_mask) -> (grad_input, grad_weight, grad_bias)``.

    Connectors ``grad_output``, ``input``, ``weight`` and ``output_0`` (grad_input), ``output_1`` (grad_weight),
    ``output_2`` (grad_bias); each output is absent when the corresponding gradient is not used.
    """

    implementations = {}
    default_implementation = "ml"


@library.register_expansion(ConvolutionBackward, "ml")
class ExpandConvolutionBackwardML(xf.ExpandTransformation):
    """One nested SDFG with the backward-data, backward-filter and (bias) reduction nodes, each built only when used."""

    environments = []

    @staticmethod
    def expansion(node: ConvolutionBackward, parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        from dace.libraries import ml  # Deferred: only needed to expand, not to load SDFGs
        from dace.libraries.standard import Reduce

        sdfg, state = expansion_sdfg(node, parent_state, parent_sdfg)
        args = node.arguments
        if bool(args.get("transposed", False)):
            raise NotImplementedError("Transposed convolution backward is not implemented")
        arrays = sdfg.arrays
        nd = len(arrays["weight"].shape) - 2
        strides = int_list(args.get("stride", 1), nd)
        pads = int_list(args.get("padding", 0), nd)
        dils = int_list(args.get("dilation", 1), nd)
        groups = int(as_symbolic(args.get("groups", 1)))

        def _wire(libnode, inputs: dict[str, str], outputs: dict[str, str]) -> None:
            state.add_node(libnode)
            for array, conn in inputs.items():
                state.add_edge(state.add_read(array), None, libnode, conn, Memlet.from_array(array, arrays[array]))
            for array, conn in outputs.items():
                state.add_edge(libnode, conn, state.add_write(array), None, Memlet.from_array(array, arrays[array]))

        if "output_0" in arrays:  # grad_input
            bwd = ml.ConvolutionBackwardData(
                node.name + "_data", strides=strides, pads=pads, dilations=dils, groups=groups
            )
            _wire(bwd, {"grad_output": "_dy", "weight": "_w"}, {"output_0": "_dx"})
        if "output_1" in arrays:  # grad_weight
            bwf = ml.ConvolutionBackwardFilter(
                node.name + "_filter", strides=strides, pads=pads, dilations=dils, groups=groups
            )
            _wire(bwf, {"input": "_x", "grad_output": "_dy"}, {"output_1": "_dw"})
        if "output_2" in arrays:  # grad_bias: sum of grad_output over all but the channel dimension
            g_rank = len(arrays["grad_output"].shape)
            axes = tuple([0] + list(range(2, g_rank)))
            red = Reduce(node.name + "_bias", "lambda a, b: a + b", axes, 0)
            red.add_in_connector("_in")
            red.add_out_connector("_out")
            _wire(red, {"grad_output": "_in"}, {"output_2": "_out"})
        return sdfg
