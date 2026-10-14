# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Pooling library nodes with PyTorch ``aten`` forward and backward semantics.

Data layout follows PyTorch: inputs and outputs are ``(N, C, spatial...)``. Arbitrary strides are taken from the
connected data descriptors, so channels-last tensors are supported transparently. The ``pure`` expansions evaluate one
map over the output elements with the window clipped to the input bounds per output element, so no padded copy of the
input is ever materialised. Max-pool indices are the flat row-major index into the unpadded spatial plane of the input.
"""

from dace import SDFG, Memlet, SDFGState, dtypes, library, nodes, properties, symbolic
from dace.transformation.transformation import ExpandTransformation


def _edges_by_connector(state, node):
    ins = {e.dst_conn: e for e in state.in_edges(node) if e.dst_conn}
    outs = {e.src_conn: e for e in state.out_edges(node) if e.src_conn}
    return ins, outs


def _pool_output_size(h, k, pad, stride, dilation, ceil_mode):
    """PyTorch pooling output size, applying the ceil-mode correction that drops a window starting in the right pad."""
    base = h + 2 * pad - dilation * (k - 1) - 1
    if not ceil_mode:
        return symbolic.int_floor(base, stride) + 1
    out = symbolic.int_ceil(base, stride) + 1
    # torch drops the last window if it starts entirely in the right/bottom padding.
    if all(isinstance(v, int) for v in (h, k, pad, stride, dilation)):
        out = int(out)
        if (out - 1) * stride >= h + pad:
            out -= 1
    return out


def _clipped_inner_ranges(k_params, o_params, kernel, pads, strides, dilations, in_spatial):
    """Kernel-tap ranges clipped so every input read ``o*stride - pad + k*dilation`` stays within ``[0, H)``."""
    ranges = {}
    for i, kp in enumerate(k_params):
        p, s, d = pads[i], strides[i], dilations[i]
        o = o_params[i]
        lo = f"max(0, int_ceil({p} - ({o})*{s}, {d}))"
        hi = f"min({symbolic.symstr(kernel[i])}, int_ceil({symbolic.symstr(in_spatial[i])} + {p} - ({o})*{s}, {d}))"
        ranges[kp] = f"{lo}:{hi}"
    return ranges


# ---------------------------------------------------------------------------------------------- max pooling
@library.node
class MaxPool(nodes.LibraryNode):
    """
    N-dimensional (1-3 spatial dims) max pooling, PyTorch ``aten.max_poolNd_with_indices`` forward semantics. Input
    connector ``_x``; output connectors ``_y`` and optional ``_indices`` (flat row-major index into the unpadded
    spatial plane of the input; first occurrence wins on ties).
    """

    implementations = {}
    default_implementation = "pure"

    kernel_shape = properties.ListProperty(
        element_type=int, default=[], category="Semantics", desc="Pooling window size per spatial dimension"
    )
    strides = properties.ListProperty(
        element_type=int, default=[], category="Semantics", desc="Pooling stride per spatial dimension"
    )
    pads = properties.ListProperty(
        element_type=int, default=[], category="Semantics", desc="Symmetric padding per spatial dimension"
    )
    dilations = properties.ListProperty(
        element_type=int, default=[], category="Semantics", desc="Window dilation per spatial dimension"
    )
    ceil_mode = properties.Property(
        dtype=bool, default=False, category="Semantics", desc="Use ceil instead of floor for the output size"
    )

    def __init__(self, name, kernel_shape=None, strides=None, pads=None, dilations=None, ceil_mode=False, **kwargs):
        super().__init__(name, inputs={"_x"}, outputs={"_y"}, **kwargs)
        if kernel_shape is not None:
            self.kernel_shape = list(kernel_shape)
        if strides is not None:
            self.strides = list(strides)
        if pads is not None:
            self.pads = list(pads)
        if dilations is not None:
            self.dilations = list(dilations)
        self.ceil_mode = ceil_mode

    def validate(self, sdfg: SDFG, state: SDFGState):
        ins, outs = _edges_by_connector(state, self)
        if "_x" not in ins:
            raise ValueError("MaxPool requires an '_x' input connector")
        if "_y" not in outs:
            raise ValueError("MaxPool requires a '_y' output connector")
        x = sdfg.arrays[ins["_x"].data.data]
        y = sdfg.arrays[outs["_y"].data.data]
        indices = sdfg.arrays[outs["_indices"].data.data] if "_indices" in outs else None
        ndim = len(self.kernel_shape)
        if ndim < 1 or ndim > 3:
            raise ValueError("MaxPool supports 1-3 spatial dimensions")
        if len(x.shape) != ndim + 2 or len(y.shape) != ndim + 2:
            raise ValueError("Input/output rank must equal (number of spatial dims) + 2")
        for lst, lname in ((self.strides, "strides"), (self.pads, "pads"), (self.dilations, "dilations")):
            if len(lst) != ndim:
                raise ValueError(f"'{lname}' must have one entry per spatial dimension ({ndim})")
        if indices is not None and len(indices.shape) != ndim + 2:
            raise ValueError("Indices must have the same rank as the output")
        for i in range(ndim):
            expected = _pool_output_size(
                x.shape[2 + i], self.kernel_shape[i], self.pads[i], self.strides[i], self.dilations[i], self.ceil_mode
            )
            if symbolic.equal(y.shape[2 + i], expected) is False:
                raise ValueError(f"Output spatial dimension {i} ({y.shape[2 + i]}) does not match expected {expected}")
        return x, y, indices


@library.register_expansion(MaxPool, "pure")
class ExpandMaxPoolPure(ExpandTransformation):
    """One map over the output elements; an inner map reduces the window (clipped per output element, no padded copy)."""

    environments = []

    @staticmethod
    def expansion(node: "MaxPool", parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        x, y, indices = node.validate(parent_sdfg, parent_state)
        ndim = len(node.kernel_shape)
        has_indices = indices is not None

        sdfg = SDFG("maxpool_pure")
        sdfg.add_array("_x", x.shape, x.dtype, strides=x.strides, storage=x.storage)
        sdfg.add_array("_y", y.shape, y.dtype, strides=y.strides, storage=y.storage)
        if has_indices:
            sdfg.add_array("_indices", indices.shape, indices.dtype, strides=indices.strides, storage=indices.storage)

        n, c = "__n", "__c"
        o_params = [f"__o{i}" for i in range(ndim)]
        k_params = [f"__k{i}" for i in range(ndim)]
        in_spatial = [x.shape[2 + i] for i in range(ndim)]
        out_spatial = [y.shape[2 + i] for i in range(ndim)]

        out_index = ", ".join([n, c] + o_params)
        out_ranges = {
            p: f"0:{symbolic.symstr(s)}" for p, s in zip([n, c] + o_params, [y.shape[0], y.shape[1]] + out_spatial)
        }
        inner_ranges = _clipped_inner_ranges(
            k_params, o_params, node.kernel_shape, node.pads, node.strides, node.dilations, in_spatial
        )
        x_pos = [
            f"({o_params[i]})*{node.strides[i]} - {node.pads[i]} + ({k_params[i]})*{node.dilations[i]}"
            for i in range(ndim)
        ]
        x_index = ", ".join([n, c] + x_pos)

        neg = (
            repr(float(dtypes.min_value(y.dtype)))
            if y.dtype in (dtypes.float16, dtypes.float32, dtypes.float64)
            else str(dtypes.min_value(y.dtype))
        )

        # State 1: initialise the output to the type minimum.
        init = sdfg.add_state("init")
        init.add_mapped_tasklet(
            "maxpool_init", out_ranges, {}, f"__out = {neg}", {"__out": Memlet(f"_y[{out_index}]")}, external_edges=True
        )

        # State 2: window maximum (write-conflict-resolved).
        red = sdfg.add_state_after(init, "reduce")
        rx = red.add_read("_x")
        wy = red.add_write("_y")
        ome, omx = red.add_map("maxpool_outer", out_ranges)
        ime, imx = red.add_map("maxpool_inner", inner_ranges)
        tasklet = red.add_tasklet("maxpool_max", {"__x"}, {"__o"}, "__o = __x")
        red.add_memlet_path(rx, ome, ime, tasklet, dst_conn="__x", memlet=Memlet(f"_x[{x_index}]"))
        red.add_memlet_path(
            tasklet, imx, omx, wy, src_conn="__o", memlet=Memlet(f"_y[{out_index}]", wcr="lambda a, b: max(a, b)")
        )

        if has_indices:
            # Flat row-major index into the unpadded spatial plane.
            plane_stride = ["1"] * ndim
            for i in range(ndim - 2, -1, -1):
                plane_stride[i] = f"({plane_stride[i + 1]}) * ({symbolic.symstr(in_spatial[i + 1])})"
            flat = " + ".join(f"({x_pos[i]}) * ({plane_stride[i]})" for i in range(ndim))
            sentinel = str(dtypes.max_value(indices.dtype))

            init_idx = sdfg.add_state_after(red, "init_idx")
            init_idx.add_mapped_tasklet(
                "maxpool_init_idx",
                out_ranges,
                {},
                f"__out = {sentinel}",
                {"__out": Memlet(f"_indices[{out_index}]")},
                external_edges=True,
            )
            arg = sdfg.add_state_after(init_idx, "argmax")
            rx2 = arg.add_read("_x")
            ry = arg.add_read("_y")
            wi = arg.add_write("_indices")
            ome2, omx2 = arg.add_map("argmax_outer", out_ranges)
            ime2, imx2 = arg.add_map("argmax_inner", inner_ranges)
            t2 = arg.add_tasklet("argmax", {"__x", "__y"}, {"__o"}, f"__o = ({flat}) if (__x == __y) else ({sentinel})")
            arg.add_memlet_path(rx2, ome2, ime2, t2, dst_conn="__x", memlet=Memlet(f"_x[{x_index}]"))
            arg.add_memlet_path(ry, ome2, ime2, t2, dst_conn="__y", memlet=Memlet(f"_y[{out_index}]"))
            arg.add_memlet_path(
                t2,
                imx2,
                omx2,
                wi,
                src_conn="__o",
                memlet=Memlet(f"_indices[{out_index}]", wcr="lambda a, b: min(a, b)"),
            )

        return sdfg


# ---------------------------------------------------------------------------------------------- average pooling
@library.node
class AvgPool(nodes.LibraryNode):
    """
    N-dimensional (1-3 spatial dims) average pooling, PyTorch ``aten.avg_poolNd`` forward semantics. Input connector
    ``_x``; output connector ``_y``.
    """

    implementations = {}
    default_implementation = "pure"

    kernel_shape = properties.ListProperty(
        element_type=int, default=[], category="Semantics", desc="Pooling window size per spatial dimension"
    )
    strides = properties.ListProperty(
        element_type=int, default=[], category="Semantics", desc="Pooling stride per spatial dimension"
    )
    pads = properties.ListProperty(
        element_type=int, default=[], category="Semantics", desc="Symmetric padding per spatial dimension"
    )
    ceil_mode = properties.Property(
        dtype=bool, default=False, category="Semantics", desc="Use ceil instead of floor for the output size"
    )
    count_include_pad = properties.Property(
        dtype=bool, default=True, category="Semantics", desc="Whether the averaging divisor counts padded elements"
    )
    divisor_override = properties.Property(
        dtype=int, default=0, category="Semantics", desc="Fixed averaging divisor (0 means use the pool size)"
    )

    def __init__(
        self,
        name,
        kernel_shape=None,
        strides=None,
        pads=None,
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=0,
        **kwargs,
    ):
        super().__init__(name, inputs={"_x"}, outputs={"_y"}, **kwargs)
        if kernel_shape is not None:
            self.kernel_shape = list(kernel_shape)
        if strides is not None:
            self.strides = list(strides)
        if pads is not None:
            self.pads = list(pads)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def validate(self, sdfg: SDFG, state: SDFGState):
        ins, outs = _edges_by_connector(state, self)
        if "_x" not in ins:
            raise ValueError("AvgPool requires an '_x' input connector")
        if "_y" not in outs:
            raise ValueError("AvgPool requires a '_y' output connector")
        x = sdfg.arrays[ins["_x"].data.data]
        y = sdfg.arrays[outs["_y"].data.data]
        ndim = len(self.kernel_shape)
        if ndim < 1 or ndim > 3:
            raise ValueError("AvgPool supports 1-3 spatial dimensions")
        if len(x.shape) != ndim + 2 or len(y.shape) != ndim + 2:
            raise ValueError("Input/output rank must equal (number of spatial dims) + 2")
        for lst, lname in ((self.strides, "strides"), (self.pads, "pads")):
            if len(lst) != ndim:
                raise ValueError(f"'{lname}' must have one entry per spatial dimension ({ndim})")
        for i in range(ndim):
            expected = _pool_output_size(
                x.shape[2 + i], self.kernel_shape[i], self.pads[i], self.strides[i], 1, self.ceil_mode
            )
            if symbolic.equal(y.shape[2 + i], expected) is False:
                raise ValueError(f"Output spatial dimension {i} ({y.shape[2 + i]}) does not match expected {expected}")
        return x, y


@library.register_expansion(AvgPool, "pure")
class ExpandAvgPoolPure(ExpandTransformation):
    """Clipped window sum into an accumulator, then a divide whose divisor follows PyTorch's count/clip rules."""

    environments = []

    @staticmethod
    def expansion(node: "AvgPool", parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        x, y = node.validate(parent_sdfg, parent_state)
        ndim = len(node.kernel_shape)

        sdfg = SDFG("avgpool_pure")
        sdfg.add_array("_x", x.shape, x.dtype, strides=x.strides, storage=x.storage)
        sdfg.add_array("_y", y.shape, y.dtype, strides=y.strides, storage=y.storage)
        sdfg.add_transient("_acc", y.shape, y.dtype, storage=y.storage)

        n, c = "__n", "__c"
        o_params = [f"__o{i}" for i in range(ndim)]
        k_params = [f"__k{i}" for i in range(ndim)]
        in_spatial = [x.shape[2 + i] for i in range(ndim)]
        out_spatial = [y.shape[2 + i] for i in range(ndim)]
        dilations = [1] * ndim

        out_index = ", ".join([n, c] + o_params)
        out_ranges = {
            p: f"0:{symbolic.symstr(s)}" for p, s in zip([n, c] + o_params, [y.shape[0], y.shape[1]] + out_spatial)
        }
        inner_ranges = _clipped_inner_ranges(
            k_params, o_params, node.kernel_shape, node.pads, node.strides, dilations, in_spatial
        )
        x_pos = [f"({o_params[i]})*{node.strides[i]} - {node.pads[i]} + ({k_params[i]})" for i in range(ndim)]
        x_index = ", ".join([n, c] + x_pos)

        # State 1: zero accumulator.
        init = sdfg.add_state("init")
        init.add_mapped_tasklet(
            "avgpool_init", out_ranges, {}, "__out = 0", {"__out": Memlet(f"_acc[{out_index}]")}, external_edges=True
        )

        # State 2: clipped window sum.
        acc = sdfg.add_state_after(init, "accumulate")
        rx = acc.add_read("_x")
        wacc = acc.add_write("_acc")
        ome, omx = acc.add_map("avgpool_outer", out_ranges)
        ime, imx = acc.add_map("avgpool_inner", inner_ranges)
        tasklet = acc.add_tasklet("avgpool_add", {"__x"}, {"__o"}, "__o = __x")
        acc.add_memlet_path(rx, ome, ime, tasklet, dst_conn="__x", memlet=Memlet(f"_x[{x_index}]"))
        acc.add_memlet_path(
            tasklet, imx, omx, wacc, src_conn="__o", memlet=Memlet(f"_acc[{out_index}]", wcr="lambda a, b: a + b")
        )

        # State 3: divide by the pool size.
        fin = sdfg.add_state_after(acc, "finalize")
        if node.divisor_override:
            divisor = str(int(node.divisor_override))
        else:
            factors = []
            for i in range(ndim):
                p, s, k = node.pads[i], node.strides[i], node.kernel_shape[i]
                h = symbolic.symstr(in_spatial[i])
                start = f"({o_params[i]})*{s} - {p}"
                if node.count_include_pad:
                    factors.append(f"(min(({start}) + {k}, ({h}) + {p}) - ({start}))")
                else:
                    factors.append(f"(min(({start}) + {k}, ({h})) - max(({start}), 0))")
            divisor = " * ".join(factors)
        fin.add_mapped_tasklet(
            "avgpool_div",
            out_ranges,
            {"__acc": Memlet(f"_acc[{out_index}]")},
            f"__y = __acc / ({divisor})",
            {"__y": Memlet(f"_y[{out_index}]")},
            external_edges=True,
        )
        return sdfg


# ---------------------------------------------------------------------------------------------- max-pool backward
@library.node
class MaxPoolBackward(nodes.LibraryNode):
    """
    Gradient of max pooling with respect to its input: scatter-add of each output gradient to the input element its
    window selected. Input connectors ``_dy`` and ``_indices`` (flat row-major index into the unpadded spatial plane);
    output connector ``_dx`` (zero-initialised).
    """

    implementations = {}
    default_implementation = "pure"

    def __init__(self, name, **kwargs):
        super().__init__(name, inputs={"_dy", "_indices"}, outputs={"_dx"}, **kwargs)

    def validate(self, sdfg: SDFG, state: SDFGState):
        ins, outs = _edges_by_connector(state, self)
        if "_dy" not in ins or "_indices" not in ins:
            raise ValueError("MaxPoolBackward requires '_dy' and '_indices' input connectors")
        if "_dx" not in outs:
            raise ValueError("MaxPoolBackward requires a '_dx' output connector")
        dy = sdfg.arrays[ins["_dy"].data.data]
        indices = sdfg.arrays[ins["_indices"].data.data]
        dx = sdfg.arrays[outs["_dx"].data.data]
        if len(dy.shape) != len(dx.shape):
            raise ValueError("MaxPoolBackward '_dy' and '_dx' must have the same rank")
        if len(indices.shape) != len(dy.shape):
            raise ValueError("MaxPoolBackward '_indices' must have the same rank as '_dy'")
        return dy, indices, dx


@library.register_expansion(MaxPoolBackward, "pure")
class ExpandMaxPoolBackwardPure(ExpandTransformation):
    """Zero ``_dx``, then scatter-add each output gradient to the (flat) input index its window selected."""

    environments = []

    @staticmethod
    def expansion(node: "MaxPoolBackward", parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        dy, indices, dx = node.validate(parent_sdfg, parent_state)
        ndim = len(dx.shape) - 2

        sdfg = SDFG("maxpool_backward_pure")
        sdfg.add_array("_dy", dy.shape, dy.dtype, strides=dy.strides, storage=dy.storage)
        sdfg.add_array("_indices", indices.shape, indices.dtype, strides=indices.strides, storage=indices.storage)
        sdfg.add_array("_dx", dx.shape, dx.dtype, strides=dx.strides, storage=dx.storage)

        n, c = "__n", "__c"
        o_params = [f"__o{i}" for i in range(ndim)]
        dx_spatial = [dx.shape[2 + i] for i in range(ndim)]
        dy_spatial = [dy.shape[2 + i] for i in range(ndim)]

        dx_full = ", ".join([n, c] + [f"0:{symbolic.symstr(s)}" for s in dx_spatial])
        out_index = ", ".join([n, c] + o_params)
        dy_ranges = {
            p: f"0:{symbolic.symstr(s)}" for p, s in zip([n, c] + o_params, [dy.shape[0], dy.shape[1]] + dy_spatial)
        }

        # State 1: zero the input gradient.
        init = sdfg.add_state("init")
        full_ranges = {
            p: f"0:{symbolic.symstr(s)}" for p, s in zip([n, c] + [f"__i{i}" for i in range(ndim)], dx.shape)
        }
        init.add_mapped_tasklet(
            "maxpool_bw_init",
            full_ranges,
            {},
            "__out = 0",
            {"__out": Memlet(f"_dx[{', '.join([n, c] + [f'__i{i}' for i in range(ndim)])}]")},
            external_edges=True,
        )

        # State 2: scatter-add.
        plane_stride = ["1"] * ndim
        for i in range(ndim - 2, -1, -1):
            plane_stride[i] = f"({plane_stride[i + 1]}) * ({symbolic.symstr(dx_spatial[i + 1])})"
        coords = ", ".join(f"(__ix // ({plane_stride[i]})) % ({symbolic.symstr(dx_spatial[i])})" for i in range(ndim))

        sc = sdfg.add_state_after(init, "scatter")
        rg = sc.add_read("_dy")
        rix = sc.add_read("_indices")
        wdx = sc.add_write("_dx")
        me, mx = sc.add_map("maxpool_bw", dy_ranges)
        out_memlet = Memlet(f"_dx[{dx_full}]", wcr="lambda a, b: a + b")
        out_memlet.dynamic = True
        t = sc.add_tasklet("scatter", {"__g", "__ix"}, {"__o"}, f"__o[{coords}] = __g")
        sc.add_memlet_path(rg, me, t, dst_conn="__g", memlet=Memlet(f"_dy[{out_index}]"))
        sc.add_memlet_path(rix, me, t, dst_conn="__ix", memlet=Memlet(f"_indices[{out_index}]"))
        sc.add_memlet_path(t, mx, wdx, src_conn="__o", memlet=out_memlet)
        return sdfg
