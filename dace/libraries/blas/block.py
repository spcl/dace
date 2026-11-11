# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""In-kernel BLAS by ONE thread block: each lane computes output elements, strided over the block.

``Gemv``, ``Ger``, ``Axpy``, ``Scal``, the symmetric routines, ``BatchedMatMul`` and ``Einsum`` all compute every
output element as an independent sum of products over the contracted indices. One emitter spells that for all of
them: the lanes split the flattened output, and each lane runs the contraction of its elements serially, in a fixed
order, so the result does not depend on the block size. Plain C++ over ``threadIdx`` and ``__syncthreads``, which
CUDA and HIP both provide.
"""

from collections.abc import Sequence

import dace
from dace import SDFG, SDFGState, dtypes, symbolic
from dace.libraries.blas.blas_helpers import scalar_arguments
from dace.libraries.standard.block_reduce import BLOCK_COLLECTIVE_THREADS, add_block_lane_map
from dace.sdfg import nodes

#: An index of the emitted loops: its C++ variable and its extent.
Loop = tuple[str, object]


def element(pointer: str, indices: Sequence[str], strides: Sequence) -> str:
    """``pointer`` at ``indices``, as C++."""
    offset = " + ".join(f"({i}) * ({symbolic.symstr(s)})" for i, s in zip(indices, strides)) or "0"
    return f"{pointer}[{offset}]"


def at(sdfg: SDFG, name: str, *indices: str) -> str:
    """Array ``name`` of ``sdfg`` at ``indices``, through its tasklet connector ``_<name>``."""
    return element(f"_{name}", indices, sdfg.arrays[name].strides)


def symmetric_at(sdfg: SDFG, name: str, uplo: bool, row: str, col: str) -> str:
    """The symmetric matrix ``name`` at ``(row, col)``, read from the upper (``uplo``) or the lower triangle."""
    from dace.libraries.blas.nodes.symv import symmetric_element  # Import cycle

    return f"(({symmetric_element(uplo, row, col)}) ? {at(sdfg, name, row, col)} : {at(sdfg, name, col, row)})"


def scaled_sum(sdfg: SDFG, prior: str, alpha, beta, *indices: str) -> tuple[str, dict]:
    """``alpha * __acc + beta * prior[indices]`` as C++, and the coefficients it names. As in BLAS, ``beta == 0``
    does not read ``prior``, so NaN there does not reach the result."""
    if symbolic.equal_valued(0, beta):
        return "__alpha * __acc", {"alpha": alpha}
    return f"__alpha * __acc + __beta * {at(sdfg, prior, *indices)}", {"alpha": alpha, "beta": beta}


def block_contraction_code(free: Sequence[Loop], reduced: Sequence[Loop], term: str, store: str, ctype: str) -> str:
    """C++ that runs ``store`` for every point of ``free``, the points strided over the block's lanes.

    :param free: The output's indices, outermost first; the last varies fastest across the lanes.
    :param reduced: The contracted indices: ``__acc`` sums ``term`` over them before ``store`` runs. Without any,
                    ``__acc`` is ``term``, if given.
    :param term: The C++ summand, over the variables of both index lists.
    :param store: The C++ statement writing one output element.
    :param ctype: The accumulator's C type.
    """
    lanes = BLOCK_COLLECTIVE_THREADS
    total = " * ".join(f"(long)({symbolic.symstr(extent)})" for var, extent in free) or "1"
    split = []
    for var, extent in reversed(free[1:]):
        split.append(f"const long {var} = __brem % (long)({symbolic.symstr(extent)});")
        split.append(f"__brem /= (long)({symbolic.symstr(extent)});")
    if free:
        split.insert(0, "long __brem = __brf;")
        split.append(f"const long {free[0][0]} = __brem;")
    body = store
    if reduced:
        summed = f"__acc = __acc + {term};"
        for var, extent in reversed(reduced):
            summed = f"for (long {var} = 0; {var} < (long)({symbolic.symstr(extent)}); ++{var}) {{ {summed} }}"
        body = f"{ctype} __acc = static_cast<{ctype}>(0);\n        {summed}\n        {store}"
    elif term:
        body = f"const {ctype} __acc = {term};\n        {store}"
    indices = "\n        ".join(split)
    return f"""{{
    const long __brn = {total};
    for (long __brf = (long)threadIdx.x; __brf < __brn; __brf += {lanes}) {{
        {indices}
        {body}
    }}
}}
__syncthreads();"""


def coefficients(dtype: dtypes.typeclass, **values) -> str:
    """Declarations of ``__<name>``, each of ``values`` at ``dtype``."""
    return scalar_arguments(dtype, **values)[0]


def block_sdfg(
    node: nodes.LibraryNode,
    state: SDFGState,
    nested: SDFG,
    code: str,
    values: Sequence = (),
    reads: dict[str, str] | None = None,
    writes: dict[str, str] | None = None,
) -> SDFG:
    """``nested``, whose arrays the caller added, running ``code`` on the lanes of one block.

    :param values: The coefficients ``code`` names. Their symbols are declared, so the nested SDFG maps them.
    :param reads: Array per tasklet input connector; every input array is read, so the outer read stays. By
                  default each connector of ``node`` is its array, read as ``_<connector>`` (``__A`` for ``_A``).
    :param writes: Array per tasklet output connector, by default as ``reads``.
    """
    reads = reads if reads is not None else {f"_{c}": c for c in node.in_connectors}
    writes = writes if writes is not None else {f"_{c}": c for c in node.out_connectors}
    wanted = {str(s) for v in values for s in symbolic.pystr_to_symbolic(v).free_symbols} - nested.free_symbols
    if wanted:
        defined = state.symbols_defined_at(node)
        for name in sorted(wanted):
            nested.add_symbol(name, defined.get(name, dtypes.int64))
    body = nested.add_state(f"{node.label}_block_state")
    pointers = {conn: dace.pointer(nested.arrays[name].dtype.base_type) for conn, name in reads.items()}
    outputs = {conn: dace.pointer(nested.arrays[name].dtype.base_type) for conn, name in writes.items()}
    tasklet = body.add_tasklet(f"{node.label}_block", pointers, outputs, code, language=dace.Language.CPP)
    entry, exit_node = add_block_lane_map(body, f"{node.label}_block_lanes")
    for conn, name in reads.items():
        memlet = dace.Memlet.from_array(name, nested.arrays[name])
        body.add_memlet_path(body.add_read(name), entry, tasklet, dst_conn=conn, memlet=memlet)
    for conn, name in writes.items():
        memlet = dace.Memlet.from_array(name, nested.arrays[name])
        body.add_memlet_path(tasklet, exit_node, body.add_write(name), src_conn=conn, memlet=memlet)
    return nested


def vector_types(sdfg: SDFG) -> bool:
    """Whether an array of ``sdfg`` holds vector elements, which the scalar lanes do not address."""
    return any(desc.dtype.veclen > 1 for desc in sdfg.arrays.values())
