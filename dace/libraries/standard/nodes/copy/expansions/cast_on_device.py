# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Host/device copy that changes dtype: the bytes cross the boundary unchanged, the GPU casts."""

import copy

import dace
from dace import dtypes, library
from dace.libraries.standard.helper import GPU_RESIDENT_STORAGES
from dace.libraries.standard.nodes.copy.common import INPUT_CONNECTOR_NAME, OUTPUT_CONNECTOR_NAME, _make_expansion_sdfg
from dace.libraries.standard.nodes.copy.node import CopyLibraryNode
from dace.transformation.transformation import ExpandTransformation


@library.register_expansion(CopyLibraryNode, "CastOnDevice")
class ExpandCastOnDevice(ExpandTransformation):
    """Two chained copies through a device buffer in the host side's dtype and shape.

    Host to device: ``src -> stage`` (memcpy) then ``stage -> dst`` (GPU cast).
    Device to host: ``src -> stage`` (GPU cast) then ``stage -> dst`` (memcpy).

    The buffer is pooled (``cudaMallocAsync`` on the copy's stream), so it costs no device-wide
    synchronization and is held only while the copy runs.
    """

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        ctx = _make_expansion_sdfg(node, parent_state, allow_cross_storage=True)
        to_device = ctx.out.storage in GPU_RESIDENT_STORAGES
        host, shape = (ctx.inp, ctx.in_shape_collapsed) if to_device else (ctx.out, ctx.out_shape_collapsed)

        # The inner copies' connectors are named like the wrapper's arrays, so they live one level down.
        inner = dace.SDFG(f"{ctx.sdfg.name}_cast")
        inner.add_datadesc("src", copy.deepcopy(ctx.sdfg.arrays[ctx.inp_name]))
        inner.add_datadesc("dst", copy.deepcopy(ctx.sdfg.arrays[ctx.out_name]))
        _, stage = inner.add_transient("stage", shape, host.dtype, dtypes.StorageType.GPU_Global)
        stage.pool = True
        state = inner.add_state(is_start_block=True)
        src, mid, dst = state.add_read("src"), state.add_access("stage"), state.add_write("dst")
        for label, a, b in ((f"{node.label}_in", src, mid), (f"{node.label}_out", mid, dst)):
            cpy = CopyLibraryNode(label)
            state.add_edge(a, None, cpy, INPUT_CONNECTOR_NAME, inner.make_array_memlet(a.data))
            state.add_edge(cpy, OUTPUT_CONNECTOR_NAME, b, None, inner.make_array_memlet(b.data))

        nsdfg = ctx.state.add_nested_sdfg(inner, {"src": None}, {"dst": None})
        ctx.state.add_edge(
            ctx.state.add_read(ctx.inp_name), None, nsdfg, "src", ctx.sdfg.make_array_memlet(ctx.inp_name)
        )
        ctx.state.add_edge(
            nsdfg, "dst", ctx.state.add_write(ctx.out_name), None, ctx.sdfg.make_array_memlet(ctx.out_name)
        )
        return ctx.sdfg
