# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""One BLAS node per iteration of a GPU kernel, as the ``blas_level*_in_kernel`` tests build and check it."""

import numpy as np

import dace
from dace.libraries.standard.block_reduce import in_kernel_block_implementation
from dace.sdfg import nodes
from dace.transformation.passes.offloading.taskloop import is_device_wide_libnode, is_kernel_refused_libnode

BATCH = 3
BLOCK = "CUDA (block strided)"

#: An operand: the node's connector, the array, its shape per batch element, and whether the node writes it.
Operand = tuple[str, str, tuple[int, ...], bool]


def per_block(name: str, node: nodes.LibraryNode, operands: list[Operand], dtype=dace.float64):
    """``map b (GPU_Device) { node(X[b] for every operand X) }``, every array ``[BATCH, *shape]`` on the device."""
    sdfg = dace.SDFG(name)
    for conn, array, shape, written in operands:
        if array not in sdfg.arrays:
            sdfg.add_array(array, [BATCH, *shape], dtype, dace.StorageType.GPU_Global)
    state = sdfg.add_state()
    entry, leave = state.add_map("batch", {"b": f"0:{BATCH}"}, schedule=dace.ScheduleType.GPU_Device)
    state.add_node(node)
    for conn, array, shape, written in operands:
        memlet = dace.Memlet(f"{array}[b, {', '.join(f'0:{s}' for s in shape)}]")
        if written:
            state.add_memlet_path(node, leave, state.add_write(array), src_conn=conn, memlet=memlet)
        else:
            state.add_memlet_path(state.add_read(array), entry, node, dst_conn=conn, memlet=memlet)
    return sdfg, state


def lowered_on_the_block(sdfg: dace.SDFG, state: dace.SDFGState, node: nodes.LibraryNode) -> str:
    """Asserts that the kernel keeps ``node`` and lowers it to one block of lanes; returns the lowered C++."""
    assert not is_device_wide_libnode(node, state) and not is_kernel_refused_libnode(node, state)
    assert in_kernel_block_implementation(node, state) == BLOCK
    node.implementation = BLOCK
    sdfg.expand_library_nodes()
    sdfg.validate()
    lanes = [
        n.map
        for n, parent in sdfg.all_nodes_recursive()
        if isinstance(n, nodes.MapEntry) and n.map.schedule == dace.ScheduleType.GPU_ThreadBlock
    ]
    assert [lane.range.size()[0] for lane in lanes] == [256]
    (code,) = [n.code.as_string for n, parent in sdfg.all_nodes_recursive() if isinstance(n, nodes.Tasklet)]
    assert "threadIdx.x" in code and "__syncthreads();" in code
    generated = "\n".join(c.clean_code for c in sdfg.generate_code())
    assert "__global__" in generated and "cublas" not in generated.lower()
    return code


def random_inputs(arrays: dict[str, tuple[int, ...]], seed: int = 7) -> dict[str, np.ndarray]:
    """``[BATCH, *shape]`` random float64 arrays, by name."""
    rng = np.random.default_rng(seed)
    return {name: rng.random((BATCH, *shape)) for name, shape in arrays.items()}


def run_on_the_block(
    built: tuple[dace.SDFG, dace.SDFGState, nodes.LibraryNode], inputs: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Lowers the node of ``built`` (:func:`per_block`) to the block, runs the SDFG on the device, and returns every
    array afterwards."""
    import cupy

    sdfg, state, node = built
    node.implementation = BLOCK
    device = {name: cupy.asarray(value) for name, value in inputs.items()}
    sdfg(**device)
    return {name: cupy.asnumpy(value) for name, value in device.items()}
