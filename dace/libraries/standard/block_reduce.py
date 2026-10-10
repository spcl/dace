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

from dace import SDFG, dtypes, nodes
from dace.sdfg.state import SDFGState

#: Threads per block for the in-kernel collectives. Four wavefronts on CDNA (64 wide), eight warps
#: on NVIDIA (32 wide). Matches ``scan.BLOCK_COLLECTIVE_THREADS``; a kernel holding both takes the
#: max of its thread-block maps, so keeping them equal avoids specializing the block for one.
BLOCK_COLLECTIVE_THREADS = 256

#: The fold a block collective spells itself, over ``{a}`` and ``{b}``. Min and max keep ``std::min`` /
#: ``std::max``'s operand order, so ties and NaNs resolve as the runtime functor resolves them.
FOLDS = {
    dtypes.ReductionType.Sum: "{a} + {b}",
    dtypes.ReductionType.Product: "{a} * {b}",
    dtypes.ReductionType.Min: "({b} < {a} ? {b} : {a})",
    dtypes.ReductionType.Max: "({a} < {b} ? {b} : {a})",
    dtypes.ReductionType.Logical_And: "{a} && {b}",
    dtypes.ReductionType.Logical_Or: "{a} || {b}",
    dtypes.ReductionType.Bitwise_And: "{a} & {b}",
    dtypes.ReductionType.Bitwise_Or: "{a} | {b}",
    dtypes.ReductionType.Bitwise_Xor: "{a} ^ {b}",
}


def block_redop(redtype: dtypes.ReductionType, ctype: str) -> str:
    """A CUB-compatible binary functor EXPRESSION folding ``redtype`` at ``ctype``.

    A lambda rather than ``dace::_wcr_fixed``: it needs no runtime header, so the same text builds against the
    DaCe runtime and in a standalone CPF unit. A reduction without a fold here keeps the runtime functor.
    """
    fold = FOLDS.get(redtype)
    if fold is None:
        return f"dace::_wcr_fixed<dace::ReductionType::{redtype.name}, {ctype}>()"
    return (
        f"[] (const {ctype} &__fold_l, const {ctype} &__fold_r) {{ return {fold.format(a='__fold_l', b='__fold_r')}; }}"
    )


def block_reduce_code(
    idstr: str, ctype: str, lanes: int, count_expr: str, element_expr: str, redop: str, identity: str, out_expr: str
) -> str:
    """C++ for ``out_expr = reduce(element_expr(i) for i in range(count_expr))``, by one thread block.

    :param idstr: Unique suffix for the emitted type and shared-storage names. Two collectives in
                  one kernel must not share ``__shared__`` storage.
    :param ctype: The accumulator's C type.
    :param lanes: Threads in the block; must match the enclosing thread-block map.
    :param count_expr: Number of elements, as C++.
    :param element_expr: The i-th element, as C++ over the loop variable ``__bri``.
    :param redop: A CUB-compatible binary functor EXPRESSION (:func:`block_redop`).
    :param identity: The op's identity, at ``ctype``. Lanes past the end fold this, so a short final
                     chunk needs no special case -- and every lane must still reach the collective
                     below, which carries a barrier.
    :param out_expr: The C++ lvalue the result is written to.
    """
    # ``redop`` is called UNPARENTHESISED. Wrapping it, as ``(redop)(a, b)``, is read as a C-style
    # cast of the comma expression ``(a, b)`` to the function type ``redop``, and the fold never
    # compiles -- which is what kept every ``CUDA (block strided)`` reduce off the device.
    fold = block_allreduce_code(idstr, ctype, lanes, f"__bracc_{idstr}", redop, out_expr)
    return f"""{{
    const long __brn_{idstr} = (long)({count_expr});
    {ctype} __bracc_{idstr} = {identity};
    for (long __bri = (long)threadIdx.x; __bri < __brn_{idstr}; __bri += {lanes}) {{
        __bracc_{idstr} = {redop}(__bracc_{idstr}, ({element_expr}));
    }}
{fold}
}}"""


def block_allreduce_code(idstr: str, ctype: str, lanes: int, value_expr: str, redop: str, out_expr: str) -> str:
    """C++ that folds every thread's ``value_expr`` across the block and hands EVERY thread the total.

    ``gpucub::BlockReduce`` leaves the total on thread 0 only, so it is broadcast through shared
    memory: the barrier after the store makes it visible, and the one after the read keeps a second
    collective from reusing the temporary storage unfenced. Every thread must reach this code.

    :param idstr: Unique suffix for the emitted type and shared-storage names.
    :param ctype: The value's C type.
    :param lanes: Threads in the block; must match the enclosing thread-block map.
    :param value_expr: This thread's partial.
    :param redop: A CUB-compatible binary functor EXPRESSION, called unparenthesised.
    :param out_expr: The C++ lvalue every thread receives the total in.
    """
    return f"""{{
    typedef gpucub::BlockReduce<{ctype}, {lanes}> BlockReduceT_{idstr};
    __shared__ typename BlockReduceT_{idstr}::TempStorage tmp_{idstr};
    __shared__ {ctype} bcast_{idstr};
    {ctype} __brtot_{idstr} = BlockReduceT_{idstr}(tmp_{idstr}).Reduce(({value_expr}), {redop});
    if (threadIdx.x == 0) bcast_{idstr} = __brtot_{idstr};
    __syncthreads();
    {out_expr} = bcast_{idstr};
    __syncthreads();
}}"""


def add_block_lane_map(state, label: str, lanes: int = BLOCK_COLLECTIVE_THREADS):
    """The ``GPU_ThreadBlock`` map that supplies a collective's threads.

    Its parameter is deliberately unused by the emitted code: the collective indexes threads through
    ``threadIdx`` the way CUB itself does. The map is here to tell the code generator two things it
    can learn no other way -- that the enclosing device map runs one iteration per BLOCK rather than
    per thread, and how wide the block is (``get_kernel_dimensions`` reads the block size off the
    thread-block maps a kernel contains).
    """
    return state.add_map(label, {"__lane": f"0:{lanes}"}, schedule=dtypes.ScheduleType.GPU_ThreadBlock)


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
GPU_BLOCK_IMPLEMENTATIONS = ("CUDA (block strided)", "CUDA (block)")


def gpu_block_implementation(node: nodes.Node, state: SDFGState | None = None, sdfg: SDFG | None = None) -> str | None:
    """The block lowering ``node`` registers and can take, or ``None`` when it has none.

    ``Reduce`` registers both keys and they are NOT interchangeable: ``'CUDA (block)'`` is the
    one-element-per-thread form (register in, register out, ``M == B``), which the in-kernel shape
    does not satisfy. Most specific first is what picks the general one. Given the node's ``state``
    and ``sdfg``, a lowering that would refuse the node's shape is not offered.
    """
    from dace.libraries.standard.nodes.reduce import Reduce, block_strided_refusal
    from dace.libraries.standard.nodes.scan import Scan, block_refusal

    impls = type(node).implementations
    block = next((impl for impl in GPU_BLOCK_IMPLEMENTATIONS if impl in impls), None)
    if block is None or state is None:
        return block
    if (
        isinstance(node, Reduce)
        and block_strided_refusal(node, state, sdfg if sdfg is not None else state.sdfg) is not None
    ):
        return None
    if isinstance(node, Scan) and block_refusal(node) is not None:
        return None
    return block
