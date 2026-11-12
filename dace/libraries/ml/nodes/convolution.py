# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Convolution library node with PyTorch ``aten.convolution`` forward semantics.

Data layout follows PyTorch: inputs and outputs are ``(N, C, spatial...)`` and
weights are ``(Cout, Cin // groups, k...)``. Arbitrary input/output strides are
taken from the connected data descriptors, so channels-last tensors are
supported transparently through their strides.
"""

import copy

from dace import SDFG, Memlet, SDFGState, dtypes, library, nodes, properties, symbolic
from dace.libraries.ml import environments as env
from dace.transformation.transformation import ExpandTransformation


def _spatial_output_size(h, k, pad, dilation, stride):
    """Floor((H + 2*pad - dilation*(k-1) - 1) / stride) + 1 (PyTorch semantics)."""
    return symbolic.int_floor(h + 2 * pad - dilation * (k - 1) - 1, stride) + 1


#: Activations a convolution node can apply to its result
ACTIVATIONS = ("none", "relu")


def _activated(value: str, activation: str) -> str:
    """Python expression applying ``activation`` to ``value`` with torch's semantics (ReLU propagates NaN)."""
    if activation == "none":
        return value
    if activation == "relu":
        return f"({value} if (({value}) != ({value})) or (({value}) > 0) else 0)"
    raise ValueError(f"Unknown activation {activation!r}")


def _edges_by_connector(state, node):
    ins = {e.dst_conn: e for e in state.in_edges(node) if e.dst_conn}
    outs = {e.src_conn: e for e in state.out_edges(node) if e.src_conn}
    return ins, outs


def _operand(sdfg, edge):
    """
    The descriptor of the container connected by ``edge``, sized by the memlet's subset: a convolution may read or
    write a slice of a larger container (the channels of a concatenation), whose strides are kept.
    """
    desc = copy.deepcopy(sdfg.arrays[edge.data.data])
    subset = edge.data.subset
    if subset is not None and subset.dims() == len(desc.shape):
        desc.shape = tuple(subset.size())
    return desc


@library.node
class Convolution(nodes.LibraryNode):
    """
    N-dimensional (1-3 spatial dims) convolution, PyTorch ``aten.convolution``
    forward semantics. Connectors: inputs ``_x`` (data), ``_w`` (weights), and
    optional ``_b`` (bias, shape ``[Cout]``); output ``_y``.
    """

    implementations = {}
    default_implementation = "pure"

    strides = properties.ListProperty(
        element_type=int, default=[], category="Semantics", desc="Convolution stride per spatial dimension"
    )
    pads = properties.ListProperty(
        element_type=int,
        default=[],
        category="Semantics",
        desc="Symmetric zero-padding per spatial dimension (PyTorch semantics)",
    )
    dilations = properties.ListProperty(
        element_type=int, default=[], category="Semantics", desc="Kernel dilation per spatial dimension"
    )
    output_padding = properties.ListProperty(
        element_type=int,
        default=[],
        category="Semantics",
        desc="Extra size added to the output (transposed convolutions only)",
    )
    groups = properties.Property(dtype=int, default=1, category="Semantics", desc="Number of convolution groups")
    transposed = properties.Property(
        dtype=bool,
        default=False,
        category="Semantics",
        desc="Whether this is a transposed (fractionally-strided) convolution",
    )
    activation = properties.Property(
        dtype=str,
        default="none",
        choices=ACTIVATIONS,
        category="Semantics",
        desc="Activation applied to the result after the bias ('none' or 'relu'); cuDNN and MIOpen fuse it into the "
        "convolution kernel where they support it",
    )

    def __init__(
        self,
        name,
        strides=None,
        pads=None,
        dilations=None,
        output_padding=None,
        groups=1,
        transposed=False,
        activation="none",
        *args,
        **kwargs,
    ):
        super().__init__(name, *args, inputs={"_x", "_w"}, outputs={"_y"}, **kwargs)
        self.activation = activation
        if strides is not None:
            self.strides = list(strides)
        if pads is not None:
            self.pads = list(pads)
        if dilations is not None:
            self.dilations = list(dilations)
        if output_padding is not None:
            self.output_padding = list(output_padding)
        self.groups = groups
        self.transposed = transposed

    def validate(self, sdfg: SDFG, state: SDFGState):
        """
        Validate ranks and consistency, and confirm that the output spatial
        shape matches the PyTorch formula. Returns a tuple of descriptors
        ``(x, w, b, y)`` (``b`` is ``None`` when no bias is connected).
        """
        ins, outs = _edges_by_connector(state, self)
        if "_x" not in ins or "_w" not in ins:
            raise ValueError("Convolution requires '_x' and '_w' input connectors")
        if "_y" not in outs:
            raise ValueError("Convolution requires a '_y' output connector")

        x = _operand(sdfg, ins["_x"])
        w = _operand(sdfg, ins["_w"])
        y = _operand(sdfg, outs["_y"])
        b = _operand(sdfg, ins["_b"]) if "_b" in ins else None

        ndim = len(w.shape) - 2  # number of spatial dims
        if ndim < 1 or ndim > 3:
            raise ValueError("Convolution supports 1-3 spatial dimensions")
        if len(x.shape) != ndim + 2 or len(y.shape) != ndim + 2:
            raise ValueError("Input/output rank must equal (number of spatial dims) + 2")

        for lst, lname in ((self.strides, "strides"), (self.pads, "pads"), (self.dilations, "dilations")):
            if len(lst) != ndim:
                raise ValueError(f"'{lname}' must have one entry per spatial dimension ({ndim})")

        if self.groups < 1:
            raise ValueError("'groups' must be at least 1")

        if self.transposed:
            # Only forward (non-transposed) convolution is validated/implemented here.
            return x, w, b, y

        cout, cin_g = w.shape[0], w.shape[1]
        if symbolic.equal(y.shape[1], cout) is False:
            raise ValueError("Output channel dimension must match the number of weight output channels")
        if symbolic.equal(x.shape[1], cin_g * self.groups) is False:
            raise ValueError("Input channels must equal (weight input channels) * groups")
        if symbolic.equal(cout % self.groups, 0) is False:
            raise ValueError("Output channels must be divisible by 'groups'")
        if symbolic.equal(x.shape[0], y.shape[0]) is False:
            raise ValueError("Batch dimension of input and output must match")

        if b is not None:
            if len(b.shape) != 1 or symbolic.equal(b.shape[0], cout) is False:
                raise ValueError("Bias must be a 1-D array of length Cout")

        # Spatial shape check.
        for i in range(ndim):
            expected = _spatial_output_size(
                x.shape[2 + i], w.shape[2 + i], self.pads[i], self.dilations[i], self.strides[i]
            )
            if symbolic.equal(y.shape[2 + i], expected) is False:
                raise ValueError(
                    f"Output spatial dimension {i} ({y.shape[2 + i]}) does not match the expected size {expected}"
                )

        return x, w, b, y


@library.register_expansion(Convolution, "pure")
class ExpandConvolutionPure(ExpandTransformation):
    """
    Backend-agnostic native SDFG expansion. One outer map iterates over output
    elements ``(n, co, o...)``; an inner map reduces over ``(ci-in-group, k...)``
    with kernel ranges clipped per output element so that no out-of-bounds input
    read is ever generated (no padded copy of the input). Products accumulate
    into a transient of the output shape and ``_y`` is written exactly once
    (with the bias added when ``_b`` is connected).
    """

    environments = []

    @staticmethod
    def expansion(node: "Convolution", parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        x, w, b, y = node.validate(parent_sdfg, parent_state)
        if node.transposed:
            raise NotImplementedError("Transposed convolution is not implemented in the 'pure' expansion")

        ins, outs = _edges_by_connector(parent_state, node)
        ndim = len(w.shape) - 2
        has_bias = b is not None

        sdfg = SDFG("conv_pure")
        sdfg.add_array("_x", x.shape, x.dtype, strides=x.strides, storage=x.storage)
        sdfg.add_array("_w", w.shape, w.dtype, strides=w.strides, storage=w.storage)
        sdfg.add_array("_y", y.shape, y.dtype, strides=y.strides, storage=y.storage)
        if has_bias:
            sdfg.add_array("_b", b.shape, b.dtype, strides=b.strides, storage=b.storage)
        # Transient accumulator with the output shape (default/contiguous strides).
        sdfg.add_transient("_acc", y.shape, y.dtype, storage=y.storage)

        dtype = y.dtype
        cout = w.shape[0]
        cin_g = w.shape[1]
        n_batch = y.shape[0]
        cout_per_group = cout // node.groups
        out_spatial = [y.shape[2 + i] for i in range(ndim)]
        ksize = [w.shape[2 + i] for i in range(ndim)]
        in_spatial = [x.shape[2 + i] for i in range(ndim)]

        # Outer map parameter names.
        n, oc = "__n", "__oc"
        ox = [f"__o{i}" for i in range(ndim)]
        ic = "__ic"
        kk = [f"__k{i}" for i in range(ndim)]

        out_index = ", ".join([n, oc] + ox)
        out_map_ranges = {p: f"0:{symbolic.symstr(s)}" for p, s in zip([n, oc] + ox, [n_batch, cout] + out_spatial)}

        # --- State 1: zero the accumulator ---
        init_state = sdfg.add_state("init")
        init_state.add_mapped_tasklet(
            "conv_init", out_map_ranges, {}, "__out = 0", {"__out": Memlet(f"_acc[{out_index}]")}, external_edges=True
        )

        # --- State 2: accumulate products with clipped kernel ranges ---
        acc_state = sdfg.add_state_after(init_state, "accumulate")
        rx = acc_state.add_read("_x")
        rw = acc_state.add_read("_w")
        wacc = acc_state.add_write("_acc")

        ome, omx = acc_state.add_map("conv_outer", out_map_ranges)

        # Clipped inner ranges: for each spatial dim i the valid kernel taps are
        #   k in [max(0, ceil((pad - o*stride)/dil)), min(K, ceil((H + pad - o*stride)/dil)))
        inner_ranges = {ic: f"0:{symbolic.symstr(cin_g)}"}
        for i in range(ndim):
            p, s, d = node.pads[i], node.strides[i], node.dilations[i]
            lo = f"max(0, int_ceil({p} - ({ox[i]})*{s}, {d}))"
            hi = f"min({symbolic.symstr(ksize[i])}, int_ceil({symbolic.symstr(in_spatial[i])} + {p} - ({ox[i]})*{s}, {d}))"
            inner_ranges[kk[i]] = f"{lo}:{hi}"
        ime, imx = acc_state.add_map("conv_inner", inner_ranges)

        # Input channel index accounts for grouping.
        in_channel = f"({oc}) // {symbolic.symstr(cout_per_group)} * {symbolic.symstr(cin_g)} + ({ic})"
        x_pos = [f"({ox[i]})*{node.strides[i]} - {node.pads[i]} + ({kk[i]})*{node.dilations[i]}" for i in range(ndim)]
        x_index = ", ".join([n, in_channel] + x_pos)
        w_index = ", ".join([oc, ic] + kk)

        tasklet = acc_state.add_tasklet("conv_mul", {"__x", "__w"}, {"__o"}, "__o = __x * __w")

        acc_state.add_memlet_path(rx, ome, ime, tasklet, dst_conn="__x", memlet=Memlet(f"_x[{x_index}]"))
        acc_state.add_memlet_path(rw, ome, ime, tasklet, dst_conn="__w", memlet=Memlet(f"_w[{w_index}]"))
        acc_state.add_memlet_path(
            tasklet, imx, omx, wacc, src_conn="__o", memlet=Memlet(f"_acc[{out_index}]", wcr="lambda a, b: a + b")
        )

        # --- State 3: write the output exactly once (adding the bias and applying the activation) ---
        fin_state = sdfg.add_state_after(acc_state, "finalize")
        if has_bias:
            fin_state.add_mapped_tasklet(
                "conv_finalize",
                out_map_ranges,
                {
                    "__acc": Memlet(f"_acc[{out_index}]"),
                    "__b": Memlet(f"_b[{oc}]"),
                },
                f"__v = __acc + __b\n__y = {_activated('__v', node.activation)}",
                {"__y": Memlet(f"_y[{out_index}]")},
                external_edges=True,
            )
        else:
            fin_state.add_mapped_tasklet(
                "conv_finalize",
                out_map_ranges,
                {"__acc": Memlet(f"_acc[{out_index}]")},
                f"__y = {_activated('__acc', node.activation)}",
                {"__y": Memlet(f"_y[{out_index}]")},
                external_edges=True,
            )

        return sdfg


# ---------------------------------------------------------------------------------------------- backward data
@library.node
class ConvolutionBackwardData(nodes.LibraryNode):
    """
    Gradient of a (non-transposed) N-d convolution with respect to its input. Connectors: inputs ``_dy`` (output
    gradient) and ``_w`` (weights); output ``_dx`` (input gradient). Same semantics/properties as :class:`Convolution`.
    """

    implementations = {}
    default_implementation = "pure"

    strides = properties.ListProperty(element_type=int, default=[], category="Semantics", desc="Stride per spatial dim")
    pads = properties.ListProperty(element_type=int, default=[], category="Semantics", desc="Symmetric padding")
    dilations = properties.ListProperty(element_type=int, default=[], category="Semantics", desc="Dilation per dim")
    groups = properties.Property(dtype=int, default=1, category="Semantics", desc="Number of convolution groups")
    transposed = properties.Property(dtype=bool, default=False, category="Semantics", desc="Transposed convolution")

    def __init__(self, name, strides=None, pads=None, dilations=None, groups=1, transposed=False, **kwargs):
        super().__init__(name, inputs={"_dy", "_w"}, outputs={"_dx"}, **kwargs)
        if strides is not None:
            self.strides = list(strides)
        if pads is not None:
            self.pads = list(pads)
        if dilations is not None:
            self.dilations = list(dilations)
        self.groups = groups
        self.transposed = transposed

    def validate(self, sdfg: SDFG, state: SDFGState):
        ins, outs = _edges_by_connector(state, self)
        if "_dy" not in ins or "_w" not in ins:
            raise ValueError("ConvolutionBackwardData requires '_dy' and '_w' input connectors")
        if "_dx" not in outs:
            raise ValueError("ConvolutionBackwardData requires a '_dx' output connector")
        dy = _operand(sdfg, ins["_dy"])
        w = _operand(sdfg, ins["_w"])
        dx = _operand(sdfg, outs["_dx"])
        ndim = len(w.shape) - 2
        if ndim < 1 or ndim > 3:
            raise ValueError("ConvolutionBackwardData supports 1-3 spatial dimensions")
        if len(dy.shape) != ndim + 2 or len(dx.shape) != ndim + 2:
            raise ValueError("Input/output rank must equal (number of spatial dims) + 2")
        for lst, lname in ((self.strides, "strides"), (self.pads, "pads"), (self.dilations, "dilations")):
            if len(lst) != ndim:
                raise ValueError(f"'{lname}' must have one entry per spatial dimension ({ndim})")
        if symbolic.equal(dx.shape[1], w.shape[1] * self.groups) is False:
            raise ValueError("Input-gradient channels must equal (weight input channels) * groups")
        if symbolic.equal(dy.shape[1], w.shape[0]) is False:
            raise ValueError("Output-gradient channels must equal the number of weight output channels")
        if not self.transposed:
            for i in range(ndim):
                expected = _spatial_output_size(
                    dx.shape[2 + i], w.shape[2 + i], self.pads[i], self.dilations[i], self.strides[i]
                )
                if symbolic.equal(dy.shape[2 + i], expected) is False:
                    raise ValueError(f"Output-gradient spatial dim {i} ({dy.shape[2 + i]}) does not match {expected}")
        return dy, w, dx


@library.register_expansion(ConvolutionBackwardData, "pure")
class ExpandConvolutionBackwardDataPure(ExpandTransformation):
    """
    Gather formulation: one outer map over ``_dx`` elements, an inner reduction over the output channels of the group
    and the output positions that could have read this input element (ranges clipped so every read is in bounds, plus a
    divisibility guard for strided/dilated windows). No padded copy; accumulation is write-conflict-resolved.
    """

    environments = []

    @staticmethod
    def expansion(node: "ConvolutionBackwardData", parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        dy, w, dx = node.validate(parent_sdfg, parent_state)
        if node.transposed:
            raise NotImplementedError("Transposed convolution is not implemented in the 'pure' expansion")
        ndim = len(w.shape) - 2

        sdfg = SDFG("conv_backward_data_pure")
        sdfg.add_array("_dy", dy.shape, dy.dtype, strides=dy.strides, storage=dy.storage)
        sdfg.add_array("_w", w.shape, w.dtype, strides=w.strides, storage=w.storage)
        sdfg.add_array("_dx", dx.shape, dx.dtype, strides=dx.strides, storage=dx.storage)
        sdfg.add_transient("_acc", dx.shape, dx.dtype, storage=dx.storage)

        cout = w.shape[0]
        cin_g = w.shape[1]
        cout_pg = cout // node.groups
        out_spatial = [dy.shape[2 + i] for i in range(ndim)]
        ksize = [w.shape[2 + i] for i in range(ndim)]

        n, ci = "__n", "__ci"
        ipos = [f"__i{i}" for i in range(ndim)]
        cog = "__cog"
        o = [f"__o{i}" for i in range(ndim)]

        dx_index = ", ".join([n, ci] + ipos)
        dx_ranges = {
            p: f"0:{symbolic.symstr(s)}"
            for p, s in zip([n, ci] + ipos, [dx.shape[0], dx.shape[1]] + [dx.shape[2 + i] for i in range(ndim)])
        }

        init = sdfg.add_state("init")
        init.add_mapped_tasklet(
            "cbd_init", dx_ranges, {}, "__out = 0", {"__out": Memlet(f"_acc[{dx_index}]")}, external_edges=True
        )

        acc_state = sdfg.add_state_after(init, "accumulate")
        rdy = acc_state.add_read("_dy")
        rw = acc_state.add_read("_w")
        wacc = acc_state.add_write("_acc")
        ome, omx = acc_state.add_map("cbd_outer", dx_ranges)

        inner_ranges = {cog: f"0:{symbolic.symstr(cout_pg)}"}
        for i in range(ndim):
            p, s, d = node.pads[i], node.strides[i], node.dilations[i]
            lo = f"max(0, int_ceil(({ipos[i]}) + {p} - ({symbolic.symstr(ksize[i])} - 1)*{d}, {s}))"
            hi = f"min({symbolic.symstr(out_spatial[i])}, int_floor(({ipos[i]}) + {p}, {s}) + 1)"
            inner_ranges[o[i]] = f"{lo}:{hi}"
        ime, imx = acc_state.add_map("cbd_inner", inner_ranges)

        co = f"(({ci}) // ({symbolic.symstr(cin_g)})) * {symbolic.symstr(cout_pg)} + ({cog})"
        ci_g = f"({ci}) % ({symbolic.symstr(cin_g)})"
        kpos = [
            f"(({ipos[i]}) + {node.pads[i]} - ({o[i]})*{node.strides[i]}) // {node.dilations[i]}" for i in range(ndim)
        ]
        dy_index = ", ".join([n, co] + o)
        w_index = ", ".join([co, ci_g] + kpos)
        valid = " and ".join(
            f"((({ipos[i]}) + {node.pads[i]} - ({o[i]})*{node.strides[i]}) % {node.dilations[i]} == 0)"
            for i in range(ndim)
        )
        tasklet = acc_state.add_tasklet("cbd_mul", {"__g", "__w"}, {"__o"}, f"__o = (__g * __w) if ({valid}) else 0")
        acc_state.add_memlet_path(rdy, ome, ime, tasklet, dst_conn="__g", memlet=Memlet(f"_dy[{dy_index}]"))
        acc_state.add_memlet_path(rw, ome, ime, tasklet, dst_conn="__w", memlet=Memlet(f"_w[{w_index}]"))
        acc_state.add_memlet_path(
            tasklet, imx, omx, wacc, src_conn="__o", memlet=Memlet(f"_acc[{dx_index}]", wcr="lambda a, b: a + b")
        )

        fin = sdfg.add_state_after(acc_state, "finalize")
        fin.add_mapped_tasklet(
            "cbd_copy",
            dx_ranges,
            {"__acc": Memlet(f"_acc[{dx_index}]")},
            "__dx = __acc",
            {"__dx": Memlet(f"_dx[{dx_index}]")},
            external_edges=True,
        )
        return sdfg


# ---------------------------------------------------------------------------------------------- backward filter
@library.node
class ConvolutionBackwardFilter(nodes.LibraryNode):
    """
    Gradient of a (non-transposed) N-d convolution with respect to its weights. Connectors: inputs ``_x`` (data) and
    ``_dy`` (output gradient); output ``_dw`` (weight gradient). Same semantics/properties as :class:`Convolution`.
    """

    implementations = {}
    default_implementation = "pure"

    strides = properties.ListProperty(element_type=int, default=[], category="Semantics", desc="Stride per spatial dim")
    pads = properties.ListProperty(element_type=int, default=[], category="Semantics", desc="Symmetric padding")
    dilations = properties.ListProperty(element_type=int, default=[], category="Semantics", desc="Dilation per dim")
    groups = properties.Property(dtype=int, default=1, category="Semantics", desc="Number of convolution groups")
    transposed = properties.Property(dtype=bool, default=False, category="Semantics", desc="Transposed convolution")

    def __init__(self, name, strides=None, pads=None, dilations=None, groups=1, transposed=False, **kwargs):
        super().__init__(name, inputs={"_x", "_dy"}, outputs={"_dw"}, **kwargs)
        if strides is not None:
            self.strides = list(strides)
        if pads is not None:
            self.pads = list(pads)
        if dilations is not None:
            self.dilations = list(dilations)
        self.groups = groups
        self.transposed = transposed

    def validate(self, sdfg: SDFG, state: SDFGState):
        ins, outs = _edges_by_connector(state, self)
        if "_x" not in ins or "_dy" not in ins:
            raise ValueError("ConvolutionBackwardFilter requires '_x' and '_dy' input connectors")
        if "_dw" not in outs:
            raise ValueError("ConvolutionBackwardFilter requires a '_dw' output connector")
        x = _operand(sdfg, ins["_x"])
        dy = _operand(sdfg, ins["_dy"])
        dw = _operand(sdfg, outs["_dw"])
        ndim = len(dw.shape) - 2
        if ndim < 1 or ndim > 3:
            raise ValueError("ConvolutionBackwardFilter supports 1-3 spatial dimensions")
        if len(x.shape) != ndim + 2 or len(dy.shape) != ndim + 2:
            raise ValueError("Input/output rank must equal (number of spatial dims) + 2")
        for lst, lname in ((self.strides, "strides"), (self.pads, "pads"), (self.dilations, "dilations")):
            if len(lst) != ndim:
                raise ValueError(f"'{lname}' must have one entry per spatial dimension ({ndim})")
        if symbolic.equal(dw.shape[0], dy.shape[1]) is False:
            raise ValueError("Weight-gradient output channels must equal the output-gradient channels")
        if symbolic.equal(x.shape[1], dw.shape[1] * self.groups) is False:
            raise ValueError("Input channels must equal (weight input channels) * groups")
        return x, dy, dw


@library.register_expansion(ConvolutionBackwardFilter, "pure")
class ExpandConvolutionBackwardFilterPure(ExpandTransformation):
    """
    Reduction formulation: one outer map over ``_dw`` elements, an inner reduction over the batch and output positions
    (clipped so the correlated input index stays in bounds). No padded copy; accumulation is write-conflict-resolved.
    """

    environments = []

    @staticmethod
    def expansion(node: "ConvolutionBackwardFilter", parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        x, dy, dw = node.validate(parent_sdfg, parent_state)
        if node.transposed:
            raise NotImplementedError("Transposed convolution is not implemented in the 'pure' expansion")
        ndim = len(dw.shape) - 2

        sdfg = SDFG("conv_backward_filter_pure")
        sdfg.add_array("_x", x.shape, x.dtype, strides=x.strides, storage=x.storage)
        sdfg.add_array("_dy", dy.shape, dy.dtype, strides=dy.strides, storage=dy.storage)
        sdfg.add_array("_dw", dw.shape, dw.dtype, strides=dw.strides, storage=dw.storage)
        sdfg.add_transient("_acc", dw.shape, dw.dtype, storage=dw.storage)

        cout = dw.shape[0]
        cin_g = dw.shape[1]
        cout_pg = cout // node.groups
        n_batch = x.shape[0]
        out_spatial = [dy.shape[2 + i] for i in range(ndim)]
        in_spatial = [x.shape[2 + i] for i in range(ndim)]
        ksize = [dw.shape[2 + i] for i in range(ndim)]

        co, cig = "__co", "__cig"
        kpos = [f"__k{i}" for i in range(ndim)]
        n = "__n"
        o = [f"__o{i}" for i in range(ndim)]

        dw_index = ", ".join([co, cig] + kpos)
        dw_ranges = {p: f"0:{symbolic.symstr(s)}" for p, s in zip([co, cig] + kpos, [cout, cin_g] + ksize)}

        init = sdfg.add_state("init")
        init.add_mapped_tasklet(
            "cbf_init", dw_ranges, {}, "__out = 0", {"__out": Memlet(f"_acc[{dw_index}]")}, external_edges=True
        )

        acc_state = sdfg.add_state_after(init, "accumulate")
        rx = acc_state.add_read("_x")
        rdy = acc_state.add_read("_dy")
        wacc = acc_state.add_write("_acc")
        ome, omx = acc_state.add_map("cbf_outer", dw_ranges)

        inner_ranges = {n: f"0:{symbolic.symstr(n_batch)}"}
        for i in range(ndim):
            p, s, d = node.pads[i], node.strides[i], node.dilations[i]
            lo = f"max(0, int_ceil({p} - ({kpos[i]})*{d}, {s}))"
            hi = f"min({symbolic.symstr(out_spatial[i])}, int_ceil({symbolic.symstr(in_spatial[i])} + {p} - ({kpos[i]})*{d}, {s}))"
            inner_ranges[o[i]] = f"{lo}:{hi}"
        ime, imx = acc_state.add_map("cbf_inner", inner_ranges)

        ci = f"(({co}) // {symbolic.symstr(cout_pg)}) * {symbolic.symstr(cin_g)} + ({cig})"
        xpos = [f"({o[i]})*{node.strides[i]} - {node.pads[i]} + ({kpos[i]})*{node.dilations[i]}" for i in range(ndim)]
        dy_index = ", ".join([n, co] + o)
        x_index = ", ".join([n, ci] + xpos)
        tasklet = acc_state.add_tasklet("cbf_mul", {"__g", "__x"}, {"__o"}, "__o = __g * __x")
        acc_state.add_memlet_path(rdy, ome, ime, tasklet, dst_conn="__g", memlet=Memlet(f"_dy[{dy_index}]"))
        acc_state.add_memlet_path(rx, ome, ime, tasklet, dst_conn="__x", memlet=Memlet(f"_x[{x_index}]"))
        acc_state.add_memlet_path(
            tasklet, imx, omx, wacc, src_conn="__o", memlet=Memlet(f"_acc[{dw_index}]", wcr="lambda a, b: a + b")
        )

        fin = sdfg.add_state_after(acc_state, "finalize")
        fin.add_mapped_tasklet(
            "cbf_copy",
            dw_ranges,
            {"__acc": Memlet(f"_acc[{dw_index}]")},
            "__dw = __acc",
            {"__dw": Memlet(f"_dw[{dw_index}]")},
            external_edges=True,
        )
        return sdfg


# ---------------------------------------------------------------------------------------------- cuDNN and MIOpen
# One tasklet per node that owns a ``ConvolutionPlan`` (see ``include/dace_cudnn.h`` and ``dace_miopen.h``): the
# tensor descriptors, the chosen algorithm and a workspace that grows on demand. The plan is rebuilt only when the
# shape key (sizes and strides, which may be symbolic) changes, so the per-call cost is the library call itself.

_CUDNN_DTYPES = {
    dtypes.float16: "CUDNN_DATA_HALF",
    dtypes.bfloat16: "CUDNN_DATA_BFLOAT16",
    dtypes.float32: "CUDNN_DATA_FLOAT",
    dtypes.float64: "CUDNN_DATA_DOUBLE",
}
_MIOPEN_DTYPES = {
    dtypes.float16: "miopenHalf",
    dtypes.bfloat16: "miopenBFloat16",
    dtypes.float32: "miopenFloat",
}

#: Per kind of convolution: which connector feeds the plan's ``input``/``output``/``filter`` descriptors.
_PLAN_ROLES = {
    "forward": {"input": "_x", "output": "_y", "filter": "_w"},
    "backward_data": {"input": "_dx", "output": "_dy", "filter": "_w"},
    "backward_filter": {"input": "_x", "output": "_dy", "filter": "_dw"},
}

# cuDNN heuristics (no data needed), algorithm enumerations and the call of each kind
_CUDNN_CALLS = {
    "forward": dict(
        perf="cudnnConvolutionFwdAlgoPerf_t",
        count="CUDNN_CONVOLUTION_FWD_ALGO_COUNT",
        algo="cudnnConvolutionFwdAlgo_t",
        query="cudnnGetConvolutionForwardAlgorithm_v7(__h, {input}, {filter}, {conv}, {output}, __c, __n, __p)",
        call="cudnnConvolutionForward(__h, &__alpha, {input}, _x, {filter}, _w, {conv}, ({algo})__plan.algo, "
        "__plan.Workspace(), __plan.workspace_size, &__beta, {output}, _y)",
    ),
    "backward_data": dict(
        perf="cudnnConvolutionBwdDataAlgoPerf_t",
        count="CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT",
        algo="cudnnConvolutionBwdDataAlgo_t",
        query="cudnnGetConvolutionBackwardDataAlgorithm_v7(__h, {filter}, {output}, {conv}, {input}, __c, __n, __p)",
        call="cudnnConvolutionBackwardData(__h, &__alpha, {filter}, _w, {output}, _dy, {conv}, ({algo})__plan.algo, "
        "__plan.Workspace(), __plan.workspace_size, &__beta, {input}, _dx)",
    ),
    "backward_filter": dict(
        perf="cudnnConvolutionBwdFilterAlgoPerf_t",
        count="CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT",
        algo="cudnnConvolutionBwdFilterAlgo_t",
        query="cudnnGetConvolutionBackwardFilterAlgorithm_v7(__h, {input}, {output}, {conv}, {filter}, __c, __n, __p)",
        call="cudnnConvolutionBackwardFilter(__h, &__alpha, {input}, _x, {output}, _dy, {conv}, ({algo})__plan.algo, "
        "__plan.Workspace(), __plan.workspace_size, &__beta, {filter}, _dw)",
    ),
}

# MIOpen: workspace query, the search (which needs the buffers; done once per shape) and the call of each kind
_MIOPEN_CALLS = {
    "forward": dict(
        workspace="miopenConvolutionForwardGetWorkSpaceSize(__h, {filter}, {input}, {conv}, {output}, &__ws)",
        find="miopenFindConvolutionForwardAlgorithm(__h, {input}, _x, {filter}, _w, {conv}, {output}, _y, 1, &__n, "
        "&__perf, __wsp, __ws, false)",
        algo="miopenConvFwdAlgorithm_t",
        found="fwd_algo",
        call="miopenConvolutionForward(__h, &__alpha, {input}, _x, {filter}, _w, {conv}, ({algo})__plan.algo, &__beta, "
        "{output}, _y, __plan.Workspace(), __plan.workspace_size)",
    ),
    "backward_data": dict(
        workspace="miopenConvolutionBackwardDataGetWorkSpaceSize(__h, {output}, {filter}, {conv}, {input}, &__ws)",
        find="miopenFindConvolutionBackwardDataAlgorithm(__h, {output}, _dy, {filter}, _w, {conv}, {input}, _dx, 1, "
        "&__n, &__perf, __wsp, __ws, false)",
        algo="miopenConvBwdDataAlgorithm_t",
        found="bwd_data_algo",
        call="miopenConvolutionBackwardData(__h, &__alpha, {output}, _dy, {filter}, _w, {conv}, ({algo})__plan.algo, "
        "&__beta, {input}, _dx, __plan.Workspace(), __plan.workspace_size)",
    ),
    "backward_filter": dict(
        workspace="miopenConvolutionBackwardWeightsGetWorkSpaceSize(__h, {output}, {input}, {conv}, {filter}, &__ws)",
        find="miopenFindConvolutionBackwardWeightsAlgorithm(__h, {output}, _dy, {input}, _x, {conv}, {filter}, _dw, "
        "1, &__n, &__perf, __wsp, __ws, false)",
        algo="miopenConvBwdWeightsAlgorithm_t",
        found="bwd_weights_algo",
        call="miopenConvolutionBackwardWeights(__h, &__alpha, {output}, _dy, {input}, _x, {conv}, ({algo})__plan.algo, "
        "&__beta, {filter}, _dw, __plan.Workspace(), __plan.workspace_size)",
    ),
}


def _c_list(values) -> str:
    """The values as a C initializer list of ``int`` (symbolic sizes are cast, as they may be 64-bit symbols)."""
    from dace.codegen.targets import cpp

    def c_int(v):
        if isinstance(v, int) or getattr(v, "is_Integer", False):
            return str(int(v))
        return f"(int)({cpp.sym2cpp(v)})"

    return ", ".join(c_int(v) for v in values)


def _gpu_operands(node, parent_sdfg, parent_state, kind: str):
    """Validates the node for a GPU expansion and returns its operand descriptors by connector name."""
    if node.transposed:
        raise NotImplementedError("Transposed convolution is not implemented in the cuDNN/MIOpen expansions")
    node.validate(parent_sdfg, parent_state)
    ins, outs = _edges_by_connector(parent_state, node)
    operands = {conn: _operand(parent_sdfg, e) for conn, e in list(ins.items()) + list(outs.items())}
    for conn, desc in operands.items():
        if desc.storage != dtypes.StorageType.GPU_Global:
            raise ValueError(f"cuDNN/MIOpen convolution requires '{conn}' to be in GPU_Global storage")
    return operands


def _library_dtype(mapping: dict, dtype, library: str) -> str:
    if dtype not in mapping:
        raise NotImplementedError(f"{library} convolution does not support dtype {dtype}")
    return mapping[dtype]


#: Output connector of each kind of convolution
_OUTPUT_CONNECTOR = {"forward": "_y", "backward_data": "_dx", "backward_filter": "_dw"}

_plan_count = 0


def _plan_tasklet(node, kind: str, operands: dict, plan_type: str, code: str, field: str) -> nodes.Tasklet:
    """A tasklet that owns one convolution plan (``field`` of the program's state struct) and runs ``code``."""
    out_conn = _OUTPUT_CONNECTOR[kind]
    return nodes.Tasklet(
        node.name + "_" + plan_type.lower(),
        set(operands) - {out_conn},
        {out_conn},
        code,
        language=dtypes.Language.CPP,
        state_fields=[f"dace::ml::{plan_type}ConvolutionPlan {field};"],
    )


def _new_plan_field() -> str:
    """Name of a plan in the state struct, unique per program (nested SDFGs share the struct)."""
    global _plan_count
    _plan_count += 1
    return f"conv_plan_{_plan_count}"


class _Geometry:
    """
    Sizes and strides of the operands and the convolution parameters as the libraries take them: both need at least
    two spatial dimensions, so a 1-D convolution is promoted to a 2-D one with a unit height (the inserted dimension
    gets a stride that keeps the dense layouts recognizable).
    """

    def __init__(self, node, operands: dict, kind: str):
        roles = _PLAN_ROLES[kind]
        ndim = len(operands[roles["filter"]].shape) - 2
        self.promoted = ndim == 1
        self.ndim = max(ndim, 2)
        self.x, self.y, self.w = (self._dims(operands[r]) for r in (roles["input"], roles["output"], roles["filter"]))
        cout = self.w[0][0]
        self.bias = ([1, cout] + [1] * self.ndim, [cout] + [1] * (self.ndim + 1))
        unit = [0] if self.promoted else []
        one = [1] if self.promoted else []
        self.pads = unit + list(node.pads)
        self.strides = one + list(node.strides)
        self.dilations = one + list(node.dilations)

    def _dims(self, desc) -> tuple[list, list]:
        shape, strides = list(desc.shape), list(desc.strides)
        if self.promoted:
            shape.insert(2, 1)
            strides.insert(2, strides[2] * shape[3])
        return shape, strides

    def key(self, with_filter_strides: bool) -> str:
        """The shape key of the plan: everything that can change between calls of a program with symbolic sizes."""
        values = self.x[0] + self.x[1] + self.y[0] + self.y[1] + self.w[0]
        if with_filter_strides:
            values = values + self.w[1]
        return _c_list(values)


def _is_dense_in_order(shape: list, strides: list, order: list[int]) -> bool:
    """Whether the strides are those of a dense tensor whose dimensions are laid out in ``order`` (outermost first)."""
    expected = 1
    for dim in reversed(order):
        if symbolic.equal(strides[dim], expected) is False:
            return False
        expected = expected * shape[dim]
    return True


def _layout(dims: tuple[list, list]) -> str | None:
    """``"NCHW"`` (contiguous), ``"NHWC"`` (``torch.channels_last``: channels innermost) or ``None`` (other)."""
    shape, strides = dims
    rank = len(shape)
    if _is_dense_in_order(shape, strides, list(range(rank))):
        return "NCHW"
    if _is_dense_in_order(shape, strides, [0] + list(range(2, rank)) + [1]):
        return "NHWC"
    return None


def _require_dense_bias(b, library: str) -> None:
    if symbolic.equal(b.strides[0], 1) is False:
        raise NotImplementedError(f"{library} requires a contiguous bias (stride {b.strides[0]} given)")


_CUDNN_FILTER_FORMATS = {"NCHW": "CUDNN_TENSOR_NCHW", "NHWC": "CUDNN_TENSOR_NHWC"}
_CUDNN_ACTIVATIONS = {"relu": "CUDNN_ACTIVATION_RELU"}
_MIOPEN_ACTIVATIONS = {"relu": "miopenActivationRELU"}
_MIOPEN_LAYOUTS = {
    ("NCHW", 4): "miopenTensorNCHW",
    ("NHWC", 4): "miopenTensorNHWC",
    ("NCHW", 5): "miopenTensorNCDHW",
    ("NHWC", 5): "miopenTensorNDHWC",
}


def _cudnn_code(node, kind: str, operands: dict, field: str) -> str:
    roles = _PLAN_ROLES[kind]
    y, w = operands[roles["output"]], operands[roles["filter"]]
    b = operands.get("_b")
    geo = _Geometry(node, operands, kind)
    ndim = geo.ndim
    # cuDNN filter descriptors carry a layout instead of strides
    filter_format = _CUDNN_FILTER_FORMATS.get(_layout(geo.w))
    if filter_format is None:
        raise NotImplementedError(
            f"cuDNN supports contiguous (NCHW) or channels-last (NHWC) weights; strides {list(w.strides)} given"
        )
    if b is not None:
        _require_dense_bias(b, "cuDNN")
    dt = _library_dtype(_CUDNN_DTYPES, y.dtype, "cuDNN")
    compute_dt = "CUDNN_DATA_DOUBLE" if y.dtype == dtypes.float64 else "CUDNN_DATA_FLOAT"
    scalar = "double" if y.dtype == dtypes.float64 else "float"
    # Reduced-precision data may use the tensor cores (what torch does); single precision stays exact (no TF32)
    math_type = "CUDNN_TENSOR_OP_MATH" if y.dtype in (dtypes.float16, dtypes.bfloat16) else "CUDNN_FMA_MATH"
    plan = {role: f"__plan.{role}" for role in ("input", "output", "filter", "conv", "bias", "activation")}
    plan["algo"] = _CUDNN_CALLS[kind]["algo"]
    activation = getattr(node, "activation", "none")
    shape_key = geo.key(with_filter_strides=False)

    def tensor(desc_name, shape, strides):
        return f"""
        {{
            int __dims[] = {{{_c_list(shape)}}};
            int __strides[] = {{{_c_list(strides)}}};
            dace::ml::CheckCudnnError(cudnnSetTensorNdDescriptor({desc_name}, {dt}, {ndim + 2}, __dims, __strides));
        }}"""

    setup = tensor(plan["input"], *geo.x) + tensor(plan["output"], *geo.y)
    setup += f"""
        {{
            int __dims[] = {{{_c_list(geo.w[0])}}};
            dace::ml::CheckCudnnError(cudnnSetFilterNdDescriptor({plan["filter"]}, {dt}, {filter_format}, {ndim + 2},
                __dims));
            int __pads[] = {{{_c_list(geo.pads)}}};
            int __strides[] = {{{_c_list(geo.strides)}}};
            int __dilations[] = {{{_c_list(geo.dilations)}}};
            dace::ml::CheckCudnnError(cudnnSetConvolutionNdDescriptor({plan["conv"]}, {ndim}, __pads, __strides,
                __dilations, CUDNN_CROSS_CORRELATION, {compute_dt}));
            dace::ml::CheckCudnnError(cudnnSetConvolutionGroupCount({plan["conv"]}, {node.groups}));
            dace::ml::CheckCudnnError(cudnnSetConvolutionMathType({plan["conv"]}, {math_type}));
        }}"""
    if b is not None or activation != "none":  # the fused convolution+bias+activation call requires a bias
        setup += tensor(plan["bias"], *geo.bias)
    if activation != "none":
        setup += f"""
        dace::ml::CheckCudnnError(cudnnSetActivationDescriptor({plan["activation"]}, {_CUDNN_ACTIVATIONS[activation]},
            CUDNN_NOT_PROPAGATE_NAN, 0.0));"""
    calls = _CUDNN_CALLS[kind]
    query = calls["query"].format(**plan)
    code = f"""
    cudnnHandle_t &__h = __dace_cudnn_handle;
    auto &__plan = __state->{field};
    if (__plan.Configure({{{shape_key}}})) {{{setup}
        auto __choice = dace::ml::SelectAlgorithm<{calls["perf"]}>(
            [&](int __c, int *__n, {calls["perf"]} *__p) {{ return {query}; }}, {calls["count"]});
        __plan.algo = __choice.algo;
        __plan.workspace_size = __choice.workspace;
    }}
    const {scalar} __alpha = 1, __beta = 0;
"""
    separate = f"""        dace::ml::CheckCudnnError({calls["call"].format(**plan)});
"""
    if b is not None:
        separate += f"""        dace::ml::CheckCudnnError(cudnnAddTensor(__h, &__alpha, {plan["bias"]}, _b, &__alpha, {plan["output"]}, _y));
"""
    if activation == "none":
        code += separate.replace("\n        ", "\n    ")[4:]
    else:
        bias_ptr = "_b" if b is not None else f"__plan.ZeroBias(sizeof({y.dtype.ctype}) * ({_c_list([geo.w[0][0]])}))"
        code += f"""    // One kernel for convolution, bias and activation where cuDNN supports it; separate calls otherwise
    const void *__bias = {bias_ptr};
    if (__plan.fused) {{
        cudnnStatus_t __status = cudnnConvolutionBiasActivationForward(__h, &__alpha, {plan["input"]}, _x,
            {plan["filter"]}, _w, {plan["conv"]}, ({plan["algo"]})__plan.algo, __plan.Workspace(), __plan.workspace_size,
            &__beta, {plan["output"]}, _y, {plan["bias"]}, __bias, {plan["activation"]}, {plan["output"]}, _y);
        if (__status == CUDNN_STATUS_NOT_SUPPORTED) __plan.fused = false; else dace::ml::CheckCudnnError(__status);
    }}
    if (!__plan.fused) {{
{separate}        dace::ml::CheckCudnnError(cudnnActivationForward(__h, {plan["activation"]}, &__alpha, {plan["output"]}, _y,
            &__beta, {plan["output"]}, _y));
    }}
"""
    return env.cudnn.cuDNN.handle_setup_code(node) + code


def _miopen_code(node, kind: str, operands: dict, field: str) -> str:
    roles = _PLAN_ROLES[kind]
    y = operands[roles["output"]]
    b = operands.get("_b")
    geo = _Geometry(node, operands, kind)
    ndim = geo.ndim
    if b is not None:
        _require_dense_bias(b, "MIOpen")
    dt = _library_dtype(_MIOPEN_DTYPES, y.dtype, "MIOpen")
    plan = {role: f"__plan.{role}" for role in ("input", "output", "filter", "conv", "bias", "activation")}
    plan["algo"] = _MIOPEN_CALLS[kind]["algo"]
    activation = getattr(node, "activation", "none")
    shape_key = geo.key(with_filter_strides=True)

    def tensor(desc_name, shape, strides):
        return f"""
        {{
            int __dims[] = {{{_c_list(shape)}}};
            int __strides[] = {{{_c_list(strides)}}};
            dace::ml::CheckMiopenError(miopenSetTensorDescriptor({desc_name}, {dt}, {ndim + 2}, __dims, __strides));
        }}"""

    def tensor_with_layout(desc_name, shape, layout):
        return f"""
        {{
            int __dims[] = {{{_c_list(shape)}}};
            dace::ml::CheckMiopenError(miopenSetNdTensorDescriptorWithLayout({desc_name}, {dt}, {layout}, __dims,
                {ndim + 2}));
        }}"""

    # MIOpen's convolution solvers are selected by the layout the descriptors imply. When the operands share a
    # standard layout (NCHW or channels-last, as torch produces them) it is stated explicitly, as torch does; any other
    # combination is described by its strides and MIOpen decides whether it supports it.
    layouts = {_layout(geo.x), _layout(geo.y), _layout(geo.w)}
    layout = _MIOPEN_LAYOUTS.get((next(iter(layouts)), ndim + 2)) if len(layouts) == 1 else None
    if layout is not None:
        setup = (
            tensor_with_layout(plan["input"], geo.x[0], layout)
            + tensor_with_layout(plan["output"], geo.y[0], layout)
            + tensor_with_layout(plan["filter"], geo.w[0], layout)
        )
    else:
        setup = tensor(plan["input"], *geo.x) + tensor(plan["output"], *geo.y) + tensor(plan["filter"], *geo.w)
    setup += f"""
        {{
            int __pads[] = {{{_c_list(geo.pads)}}};
            int __strides[] = {{{_c_list(geo.strides)}}};
            int __dilations[] = {{{_c_list(geo.dilations)}}};
            dace::ml::CheckMiopenError(miopenInitConvolutionNdDescriptor({plan["conv"]}, {ndim}, __pads, __strides,
                __dilations, miopenConvolution));
            dace::ml::CheckMiopenError(miopenSetConvolutionGroupCount({plan["conv"]}, {node.groups}));
        }}"""
    if b is not None:
        setup += tensor(plan["bias"], *geo.bias)
    fusion = ""
    if activation != "none":
        setup += f"""
        dace::ml::CheckMiopenError(miopenSetActivationDescriptor({plan["activation"]},
            {_MIOPEN_ACTIVATIONS[activation]}, 0.0, 0.0, 0.0));"""
        fusion = f"""
        __plan.SetupFusion(__h, {"true" if b is not None else "false"}, {_MIOPEN_ACTIVATIONS[activation]});"""
    calls = _MIOPEN_CALLS[kind]
    code = f"""
    miopenHandle_t &__h = __dace_miopen_handle;
    auto &__plan = __state->{field};
    if (__plan.Configure({{{shape_key}}})) {{{setup}
        size_t __ws = 0;
        dace::ml::CheckMiopenError({calls["workspace"].format(**plan)});
        void *__wsp = __plan.Workspace(__ws);
        miopenConvAlgoPerf_t __perf;
        int __n = 0;
        dace::ml::CheckMiopenError({calls["find"].format(**plan)});
        if (__n < 1) throw std::runtime_error("MIOpen found no applicable convolution algorithm");
        __plan.algo = __perf.{calls["found"]};
        __plan.workspace_size = __perf.memory;{fusion}
    }}
    const float __alpha = 1.0f, __beta = 0.0f;
"""
    separate = f"""        dace::ml::CheckMiopenError({calls["call"].format(**plan)});
"""
    if b is not None:
        separate += f"""        dace::ml::CheckMiopenError(miopenConvolutionForwardBias(__h, &__alpha, {plan["bias"]}, _b, &__beta, {plan["output"]}, _y));
"""
    if activation == "none":
        code += separate.replace("\n        ", "\n    ")[4:]
    else:
        bias_args = ""
        if b is not None:
            bias_args = """
        dace::ml::CheckMiopenError(miopenSetOpArgsBiasForward(__plan.fusion_args, __plan.bias_op, &__alpha, &__beta, _b));"""
        code += f"""    // One fusion plan for convolution, bias and activation where MIOpen supports it; separate calls otherwise
    if (__plan.fused) {{
        dace::ml::CheckMiopenError(miopenSetOpArgsConvForward(__plan.fusion_args, __plan.conv_op, &__alpha, &__beta, _w));{bias_args}
        dace::ml::CheckMiopenError(miopenSetOpArgsActivForward(__plan.fusion_args, __plan.activation_op, &__alpha, &__beta,
            0.0, 0.0, 0.0));
        dace::ml::CheckMiopenError(miopenExecuteFusionPlan(__h, __plan.fusion, {plan["input"]}, _x, {plan["output"]}, _y,
            __plan.fusion_args));
    }} else {{
{separate}        dace::ml::CheckMiopenError(miopenActivationForward(__h, {plan["activation"]}, &__alpha, {plan["output"]}, _y,
            &__beta, {plan["output"]}, _y));
    }}
"""
    return env.miopen.MIOpen.handle_setup_code(node) + code


def _register_gpu_expansions(node_class, kind: str) -> None:
    """Registers the ``cuDNN`` and ``MIOpen`` expansions of one convolution node class."""

    class ExpandCuDNN(ExpandTransformation):
        __doc__ = f"""cuDNN implementation ({kind}): the algorithm is chosen by cuDNN's heuristics once per shape, the
        descriptors and workspace live in the program's state (see ``include/dace_cudnn.h``)."""
        environments = [env.cudnn.cuDNN]

        @staticmethod
        def expansion(node, parent_state: SDFGState, parent_sdfg: SDFG):
            operands = _gpu_operands(node, parent_sdfg, parent_state, kind)
            field = _new_plan_field()
            return _plan_tasklet(node, kind, operands, "Cudnn", _cudnn_code(node, kind, operands, field), field)

    class ExpandMIOpen(ExpandTransformation):
        __doc__ = f"""MIOpen implementation ({kind}): the algorithm is searched (``miopenFind*``) once per shape, the
        descriptors and workspace live in the program's state (see ``include/dace_miopen.h``)."""
        environments = [env.miopen.MIOpen]

        @staticmethod
        def expansion(node, parent_state: SDFGState, parent_sdfg: SDFG):
            operands = _gpu_operands(node, parent_sdfg, parent_state, kind)
            field = _new_plan_field()
            return _plan_tasklet(node, kind, operands, "Miopen", _miopen_code(node, kind, operands, field), field)

    for cls, name in ((ExpandCuDNN, "cuDNN"), (ExpandMIOpen, "MIOpen")):
        cls.__name__ = cls.__qualname__ = f"Expand{node_class.__name__}{name}"
        library.register_expansion(node_class, name)(cls)


_register_gpu_expansions(Convolution, "forward")
_register_gpu_expansions(ConvolutionBackwardData, "backward_data")
_register_gpu_expansions(ConvolutionBackwardFilter, "backward_filter")
