# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The block-strided reduction every in-kernel reduce-shaped library node lowers to.

``gpucub::BlockReduce`` reduces ONE value per thread. The shape a kernel actually presents is ``M``
elements and ``B`` threads with no relation between them -- ``cross_entropy_loss`` reduces 46,341
classes with a 256-wide block -- and CUB has no single primitive for that. The documented
composition is a block-strided loop into a register accumulator, then one ``BlockReduce`` over the
``B`` partials, which is what this module emits.

One emitter rather than one per node: ``Reduce`` and ``Dot`` differ only in the expression that
produces an element (``_in[i*s]`` against ``_x[i*sx] * _y[i*sy]``). Duplicating the surrounding
loop, the shared-memory declaration and the broadcast is how two copies drift apart on the barrier
placement, which is the part that goes silently wrong rather than loudly broken.
"""
from dace import dtypes

#: Threads per block for the in-kernel collectives. Four wavefronts on CDNA (64 wide), eight warps
#: on NVIDIA (32 wide). Matches ``scan.BLOCK_COLLECTIVE_THREADS``; a kernel holding both takes the
#: max of its thread-block maps, so keeping them equal avoids specializing the block for one.
BLOCK_COLLECTIVE_THREADS = 256


def block_reduce_code(idstr: str, ctype: str, lanes: int, count_expr: str, element_expr: str, redop: str, identity: str,
                      out_expr: str) -> str:
    """C++ for ``out_expr = reduce(element_expr(i) for i in range(count_expr))``, by one thread block.

    :param idstr: Unique suffix for the emitted type and shared-storage names. Two collectives in
                  one kernel must not share ``__shared__`` storage.
    :param ctype: The accumulator's C type.
    :param lanes: Threads in the block; must match the enclosing thread-block map.
    :param count_expr: Number of elements, as C++.
    :param element_expr: The i-th element, as C++ over the loop variable ``__bri``.
    :param redop: A CUB-compatible binary functor EXPRESSION (e.g. ``dace::_wcr_fixed<...>()``).
    :param identity: The op's identity, at ``ctype``. Lanes past the end fold this, so a short final
                     chunk needs no special case -- and every lane must still reach the collective
                     below, which carries a barrier.
    :param out_expr: The C++ lvalue the result is written to.
    """
    return f'''{{
    typedef gpucub::BlockReduce<{ctype}, {lanes}> BlockReduceT_{idstr};
    __shared__ typename BlockReduceT_{idstr}::TempStorage tmp_{idstr};
    __shared__ {ctype} bcast_{idstr};
    const long __brn_{idstr} = (long)({count_expr});
    {ctype} __bracc_{idstr} = {identity};
    for (long __bri = (long)threadIdx.x; __bri < __brn_{idstr}; __bri += {lanes}) {{
        __bracc_{idstr} = ({redop})(__bracc_{idstr}, ({element_expr}));
    }}
    {ctype} __brtot_{idstr} = BlockReduceT_{idstr}(tmp_{idstr}).Reduce(__bracc_{idstr}, {redop});
    // BlockReduce leaves the total on thread 0 ONLY. Everything downstream in this block reads the
    // result, so it is broadcast through shared memory; the barrier after the store is what makes
    // it visible, and the one before it keeps a second collective from reusing ``tmp`` unfenced.
    if (threadIdx.x == 0) bcast_{idstr} = __brtot_{idstr};
    __syncthreads();
    {out_expr} = bcast_{idstr};
    __syncthreads();
}}'''


def add_block_lane_map(state, label: str, lanes: int = BLOCK_COLLECTIVE_THREADS):
    """The ``GPU_ThreadBlock`` map that supplies a collective's threads.

    Its parameter is deliberately unused by the emitted code: the collective indexes threads through
    ``threadIdx`` the way CUB itself does. The map is here to tell the code generator two things it
    can learn no other way -- that the enclosing device map runs one iteration per BLOCK rather than
    per thread, and how wide the block is (``get_kernel_dimensions`` reads the block size off the
    thread-block maps a kernel contains).
    """
    return state.add_map(label, {'__lane': f'0:{lanes}'}, schedule=dtypes.ScheduleType.GPU_ThreadBlock)


#: The in-kernel lowering key a library node registers when it can run as a BLOCK collective, most
#: specific first. Having one is the whole heuristic for what belongs inside a GPU kernel: a
#: reduce-shaped node (``Reduce``, ``Scan``, ``Dot``) reduces along one axis whose extent is bounded
#: by the problem's feature width, so a thread block is the right amount of machine to point at it.
#: A node with none -- ``Gemm``, ``BatchedMatMul``, ``TensorTranspose`` -- does device-scale work per
#: invocation (measured: 39.8M elements for one in-kernel TensorTranspose, ~5e9 FLOP for one
#: BatchedMatMul) and belongs in a device-wide vendor call issued from the host, not in one block.
#:
#: Deliberately capability-based rather than a list of class names: giving a node a block expansion
#: is what opts it in, in ONE place, and no size is read. Extents here are symbolic
#: (``num_classes``, ``out_channels``, ``dim``) and unreadable at compile time anyway.
GPU_BLOCK_IMPLEMENTATIONS = ('CUDA (block strided)', 'CUDA (block)')


def gpu_block_implementation(node) -> str:
    """The block lowering ``node`` registers, or ``None`` when it has none.

    ``Reduce`` registers both keys and they are NOT interchangeable: ``'CUDA (block)'`` is the
    one-element-per-thread form (register in, register out, ``M == B``), which the in-kernel shape
    does not satisfy. Most specific first is what picks the general one.
    """
    impls = type(node).implementations
    return next((impl for impl in GPU_BLOCK_IMPLEMENTATIONS if impl in impls), None)
