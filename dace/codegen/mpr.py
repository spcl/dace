# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""MPR: rendering an SDFG as one self-contained translation unit.

``mpr(sdfg)`` returns C++ that a bare host compiler accepts -- ``g++ -std=c++20 -fopenmp`` with no
``-I``, no ``libdace``, no BLAS -- and that computes what the SDFG computes. ``mpr(sdfg,
language='c')`` returns C23 under the same terms; the two are one semantics in two spellings, held
together by ``tests/codegen/mpr/test_lowering_table.py``. The point is not portability for its own
sake: it is that the result can be read, diffed and edited on its own, so a maximally parallel
rendering of a program can be compared against the original the way Pluto's output is.

Nothing here re-implements code generation. The readable CPU generator
(``compiler.cpu.implementation = experimental_readable``) already emits the shape MPR wants --
``<array>_idx`` index helpers, connector-free tasklets, ``#pragma omp parallel for`` on parallel
maps. MPR is that generator run under :attr:`~dace.mpr_lowering.Dialect.STANDALONE`, which changes
three things and no more:

* the printers spell runtime functions with the standard library or MPR's own inline definitions
  (:mod:`dace.mpr_lowering`),
* the frame emits ``extern "C" void <name>(<arglist>)`` instead of the state-carrying
  ``__program_<name>_internal`` plus its ``__dace_init`` / ``__dace_exit`` pair
  (``framecode.generate_standalone_footer``),
* this module prepends the preamble -- system headers, then exactly the inline definitions the
  finished text calls.

The preamble is assembled LAST, from the emitted text, because that is the only point that knows
which helpers were used. Deriving it from the SDFG instead would mean predicting the printers'
output, and a helper reached through a tasklet body (not a memlet subset) would be missed.

What MPR refuses, it refuses loudly -- see :func:`prepare`, and the standalone paths in
``framecode``. A rendering that quietly dropped an initializer, a caller-supplied buffer or a GPU
kernel would still compile and still produce numbers, just not the SDFG's.
"""
import copy
import re
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

from dace.ordered import OrderedSet

from dace import data as dt, dtypes, mpr_lowering
from dace.codegen import codegen
from dace.codegen.codeobject import CodeObject
from dace.config import Config, set_temporary
from dace.sdfg import SDFG, nodes
from dace.transformation.passes.scalar_promotion import PromoteScalarOutputsToArrays

#: The storage types MPR can render, as an ALLOWLIST. Ordinary host memory and plain locals, and
#: nothing else: ``CPU_Pinned`` is host memory but is allocated through the CUDA API, and the
#: accelerator storages (GPU, SVE, Snitch) each need a compiler MPR does not invoke. An allowlist
#: rather than a list of the refused ones because ``StorageType`` is extensible -- a storage
#: registered by a backend MPR has never heard of must refuse, not slip through.
HOST_STORAGE = frozenset({
    dtypes.StorageType.Default, dtypes.StorageType.Register, dtypes.StorageType.CPU_Heap,
    dtypes.StorageType.CPU_ThreadLocal
})

#: The schedules MPR can render, likewise an allowlist. ``CPU_Multicore`` is the OpenMP loop that
#: makes the rendering parallel in the first place; the rest are the sequential and unspecified
#: forms. Anything else (GPU device/thread-block, FPGA, SVE) needs another compiler.
HOST_SCHEDULES = frozenset({
    dtypes.ScheduleType.Default, dtypes.ScheduleType.Sequential, dtypes.ScheduleType.CPU_Multicore,
    dtypes.ScheduleType.CPU_Persistent
})

#: Lifetimes whose buffer the ordinary generators park in the state struct, and what MPR turns each
#: into. ``Persistent`` and ``Global`` outlive one call only so a repeated invocation can reuse the
#: allocation; a single self-contained entry point has no second invocation to reuse it, so SDFG
#: lifetime -- allocate on entry, free on return -- computes the same values. ``External`` is NOT
#: here: its buffer comes from the caller through an init handshake, so demoting it would allocate
#: a private buffer and silently discard what the caller passed. framecode refuses it instead.
#: What each library node computes, as one line MPR writes above the code its expansion produced.
#:
#: A pure expansion is loops and tasklets; nothing in it says the loops were a Cholesky
#: factorization. The comment is what keeps the rendering readable as the program it came from --
#: which is the point of MPR, since the output exists to be read and edited rather than linked.
#:
#: Keyed by class name. Every library node registered in the process must appear here (the suite
#: asserts it), so a new node arrives with a description instead of rendering as anonymous loops.
LIBRARY_NODE_DESCRIPTIONS: Dict[str, str] = {
    'Abort': 'MPI_Abort: terminate the communicator',
    'AllNode': 'all: true where every element along the reduced axes is true',
    'Allgather': 'MPI_Allgather: gather from every rank to every rank',
    'Allreduce': 'MPI_Allreduce: reduce across ranks, result on every rank',
    'Alltoall': 'MPI_Alltoall: every rank exchanges a block with every rank',
    'AnyNode': 'any: true where any element along the reduced axes is true',
    'ArgMax': 'argmax: index (and value) of the largest element along the reduced axis',
    'ArgMin': 'argmin: index (and value) of the smallest element along the reduced axis',
    'ArgReduce': 'argument reduction: index of the element the reduction selected',
    'Asum': 'BLAS asum: sum of absolute values of a vector',
    'Axpy': 'BLAS axpy: y = alpha * x + y',
    'BackwardPass': 'autodiff backward pass: the reverse-mode derivative of the forward subgraph',
    'Barrier': 'MPI_Barrier: synchronize the communicator',
    'BatchedMatMul': 'batched matrix product: one gemm per batch index',
    'Bcast': 'MPI_Bcast: broadcast from the root rank',
    'BlacsGridInit': 'Cblacs_gridinit: build a BLACS process grid on a communicator',
    'BlockCyclicGather': 'gather block-cyclic (ScaLAPACK) distributed data',
    'BlockCyclicScatter': 'scatter data in the block-cyclic (ScaLAPACK) distribution',
    'BlockGather': 'gather block-distributed data onto one rank',
    'BlockScatter': 'scatter data to ranks in blocks',
    'Broadcast': 'broadcast: expand operands to a common shape',
    'CSRMM': 'sparse CSR matrix times dense matrix',
    'CSRMV': 'sparse CSR matrix times dense vector',
    'CShift': 'circular shift along an axis',
    'Cholesky': 'Cholesky factorization: A = L @ L^T',
    'CodeLibraryNode': 'verbatim code supplied by the SDFG author',
    'CommF2c': 'MPI_Comm_f2c: convert a Fortran communicator handle',
    'CommRank': 'MPI_Comm_rank: this rank in the communicator',
    'CommSize': 'MPI_Comm_size: number of ranks in the communicator',
    'CommSplit': 'MPI_Comm_split: split the communicator',
    'Copy': 'BLAS copy: y = x',
    'CopyLibraryNode': 'copy: write one buffer into another',
    'CountLibraryNode': 'count: number of elements satisfying the predicate',
    'Dot': 'BLAS dot: inner product of two vectors',
    'Dummy': 'MPI placeholder node carrying an ordering dependency',
    'Einsum': 'einsum: a contraction over the index expression',
    'FFT': 'discrete Fourier transform',
    'FFTInterpolate': 'Fourier interpolation: resample through the frequency domain',
    'FillLibraryNode': 'fill: set every element to a constant',
    'FindFirst': 'find-first: index of the first element satisfying the predicate',
    'FortranIONode': 'Fortran I/O statement',
    'Gather': 'MPI_Gather: collect from every rank onto the root',
    'Gatherv': 'MPI_Gatherv: collect variable-sized blocks onto the root',
    'Gearbox': 'gearbox: change the element width of a stream',
    'Gemm': 'BLAS gemm: C = alpha * A @ B + beta * C',
    'Gemv': 'BLAS gemv: y = alpha * A @ x + beta * y',
    'Geqrf': 'LAPACK geqrf: QR factorization, Householder form',
    'Ger': 'BLAS ger: A = alpha * x @ y^T + A (rank-1 update)',
    'Getrf': 'LAPACK getrf: LU factorization with partial pivoting',
    'Getri': 'LAPACK getri: matrix inverse from an LU factorization',
    'Getrs': 'LAPACK getrs: solve A @ X = B from an LU factorization',
    'IFFT': 'inverse discrete Fourier transform',
    'Iamax': 'BLAS iamax: index of the largest absolute value in a vector',
    'IntegerSort': 'integer sort: counting/radix sort of an integer key array',
    'Inv': 'matrix inverse',
    'Irecv': 'MPI_Irecv: non-blocking receive',
    'Isend': 'MPI_Isend: non-blocking send',
    'LayoutChange': 'layout change: rewrite the data into a different memory layout',
    'MPINode': 'MPI operation',
    'MatMul': 'matrix product, dispatched to gemm / gemv / batched gemm by operand rank',
    'MergeLibraryNode': 'merge: select elementwise between operands by a condition',
    'NamelistRead': 'Fortran namelist read',
    'Norm2': 'norm2: Euclidean norm',
    'Nrm2': 'BLAS nrm2: Euclidean norm of a vector',
    'ONNXOp': 'ONNX operator',
    'Orgqr': 'LAPACK orgqr: form Q explicitly from a Householder QR factorization',
    'Pgemm': 'PBLAS pgemm: distributed matrix product',
    'Pgemv': 'PBLAS pgemv: distributed matrix-vector product',
    'Potrf': 'LAPACK potrf: Cholesky factorization of a positive-definite matrix',
    'Potrs': 'LAPACK potrs: solve A @ X = B from a Cholesky factorization',
    'Read': 'Fortran read statement',
    'Recv': 'MPI_Recv: blocking receive',
    'Redistribute': 'redistribute an array between two process grids',
    'Scal': 'BLAS scal: x = alpha * x',
    'Scan': 'scan: running (prefix) fold along an axis',
    'Scatter': 'MPI_Scatter: distribute from the root to every rank',
    'ScatterConflictCheck': 'scatter conflict check: detect indices written more than once',
    'Send': 'MPI_Send: blocking send',
    'Sendrecv': 'MPI_Sendrecv: paired send and receive',
    'Solve': 'solve the linear system A @ X = B',
    'Stencil': 'stencil: apply the given neighbourhood expression at every point',
    'Swap': 'BLAS swap: exchange two vectors',
    'Symm': 'BLAS symm: C = alpha * A @ B + beta * C with A symmetric',
    'Symmetrize': 'symmetrize: average a matrix with its transpose',
    'Symv': 'BLAS symv: y = alpha * A @ x + beta * y with A symmetric',
    'Syr2k': 'BLAS syr2k: C = alpha * (A @ B^T + B @ A^T) + beta * C (symmetric rank-2k update)',
    'Syrk': 'BLAS syrk: C = alpha * A @ A^T + beta * C (symmetric rank-k update)',
    'TensorDot': 'tensor contraction over the given axis pairs',
    'TensorTranspose': 'tensor transpose: permute the axes',
    'TileBinop': 'tile binary operation, elementwise over a tile',
    'TileFMA': 'tile fused multiply-add',
    'TileITE': 'tile select: elementwise choice between two tiles',
    'TileIota': 'tile iota: fill a tile with its own indices',
    'TileLoad': 'tile load: read a tile from memory',
    'TileMMA': 'tile matrix multiply-accumulate',
    'TileMaskGen': 'tile mask: the predicate for a partial tile',
    'TileReduce': 'tile reduction',
    'TileStore': 'tile store: write a tile to memory',
    'TileUnop': 'tile unary operation, elementwise over a tile',
    'Transpose': 'matrix transpose',
    'Trmm': 'BLAS trmm: B = alpha * op(A) @ B with A triangular',
    'Trmv': 'BLAS trmv: x = op(A) @ x with A triangular',
    'Trsm': 'BLAS trsm: solve op(A) @ X = alpha * B with A triangular',
    'Trsv': 'BLAS trsv: solve op(A) @ x = b with A triangular',
    'UnregisteredLibraryNode': 'a library node whose implementing module is not installed',
    'Wait': 'MPI_Wait: complete one non-blocking request',
    'Waitall': 'MPI_Waitall: complete every non-blocking request',
    'Write': 'Fortran write statement',
}

#: Descriptions for the one class name two libraries share. Looked up as ``<module>.<class>``,
#: before :data:`LIBRARY_NODE_DESCRIPTIONS` -- which deliberately has NO ``Reduce`` entry, so
#: neither meaning can be served to the other by a bare-name lookup.
QUALIFIED_DESCRIPTIONS: Dict[str, str] = {
    'dace.libraries.mpi.nodes.reduce.Reduce': 'MPI_Reduce: reduce across ranks onto the root',
    'dace.libraries.standard.nodes.reduce.Reduce': 'reduction over the given axes with the given operator',
}

#: Library-node implementations MPR selects, best first. ``pure`` is an SDFG made of maps and
#: tasklets -- which is the whole point: it renders as loops, and it is the PARALLEL form.
#: ``pure-seq`` is the sequential fallback for a node that has no parallel pure expansion.
#: ``MappedTasklet`` is the copy library node's spelling of the same idea -- a map that reads one
#: buffer and writes the other. Its ``Auto`` default would pick ``dace::CopyND`` for a strided
#: copy, which is a runtime template, so the choice is made here rather than left to the node.
#:
#: The alternative is what the node would pick on its own: a BLAS call, ``dace::reduce``, cuBLAS.
#: Every one of those is a library MPR cannot link, so leaving the choice to the node's default
#: would render a translation unit that names a symbol nothing defines.
PURE_IMPLEMENTATIONS = ('pure', 'pure-seq', 'MappedTasklet')

#: Slack in the expand-and-reselect loop of :func:`force_pure_expansions`, on top of the one round
#: per library node a single state holds. A node may expand into further library nodes (``MatMul``
#: -> ``Gemm`` -> its own expansion), so each generation needs a round of its own; the slack covers
#: that nesting depth. The bound exists only so a node that expands to itself fails with a message
#: instead of looping forever.
MAX_EXPANSION_ROUNDS = 16

LIFETIME_DEMOTIONS = {
    dtypes.AllocationLifetime.Persistent: dtypes.AllocationLifetime.SDFG,
    dtypes.AllocationLifetime.Global: dtypes.AllocationLifetime.SDFG,
}


def uses_device_code(sdfg: SDFG) -> List[str]:
    """Names of the constructs in ``sdfg`` that would require a device compiler.

    :param sdfg: the SDFG to inspect (recursively).
    :returns: a description per offending construct, empty when the SDFG is host-only.
    :seealso: :data:`HOST_STORAGE`, :data:`HOST_SCHEDULES`.
    """
    found: List[str] = []
    for subsdfg, name, desc in sdfg.arrays_recursive():
        if desc.storage not in HOST_STORAGE:
            found.append(f'{subsdfg.name}.{name} is in {desc.storage.name} storage')
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, nodes.EntryNode) and node.schedule not in HOST_SCHEDULES:
                found.append(f'{state.label}/{node.label} has the {node.schedule.name} schedule')
            elif isinstance(node, nodes.Tasklet) and node.language not in (dtypes.Language.Python, dtypes.Language.CPP):
                found.append(f'{state.label}/{node.label} is a {node.language.name} tasklet')
    return found


def description_of(node) -> Optional[str]:
    """The one-line description of what library node ``node`` computes.

    :param node: the library node, or its class. Both are accepted because the emitter has a node
                 and the coverage test has a class, and they must agree on the lookup -- a second
                 spelling of it is a second thing to keep in step.
    :returns: the description, or ``None`` if the node's class has none recorded.
    """
    cls = node if isinstance(node, type) else type(node)
    qualified = f'{cls.__module__}.{cls.__name__}'
    return QUALIFIED_DESCRIPTIONS.get(qualified) or LIBRARY_NODE_DESCRIPTIONS.get(cls.__name__)


def subtree_guids(node, state) -> Set[str]:
    """GUIDs of ``node`` and, if it is a nested SDFG, of every node inside it.

    An expansion usually lands as a single nested SDFG, and the code the reader sees comes from the
    tasklets and maps INSIDE it -- so those are the GUIDs the comment has to be able to attach to.
    """
    guids = {node.guid}
    if isinstance(node, nodes.NestedSDFG):
        guids.update(inner.guid for inner, _ in node.sdfg.all_nodes_recursive())
    return guids


def force_pure_expansions(sdfg: SDFG, provenance: Optional[Dict[str, Tuple[str, str]]] = None) -> None:
    """Expand every library node in ``sdfg`` through its pure implementation, in place.

    Done here rather than left to code generation because the choice has to be made GENERATION BY
    GENERATION: a node's expansion can introduce further library nodes, and those arrive carrying
    their own default implementation (a BLAS call, the reduction runtime) which nothing would
    re-point afterwards.

    At most ONE node per state is expanded per round. Expansion leaves no trace of which node
    produced what, so the nodes an expansion added are found by diffing the state -- and a diff can
    only be attributed when a single expansion happened in that state. Expanding two at once would
    hand every new node to whichever origin was recorded first, and a program with two matrix
    products would render with one of them commented and the other anonymous. States expand in
    parallel with each other, so the number of rounds is the deepest single state's library-node
    count, not the SDFG's.

    A node with no pure implementation at all is left alone: it may still expand to something
    renderable, and if it does not, the ``dace::`` symbol it emits is reported against its name by
    :func:`verify`, which says more than a refusal from here could.

    :param sdfg: the SDFG to expand. Call on a COPY.
    :param provenance: filled in with ``node GUID -> (origin GUID, description)`` for the code each
                       expansion produced, so the rendering can say what the loops used to be.
                       Descriptions come from :data:`LIBRARY_NODE_DESCRIPTIONS`.
    :raises NotImplementedError: if expansion has not converged (see :data:`MAX_EXPANSION_ROUNDS`).
    """
    rounds = 0
    while True:
        pending: Dict[int, List] = {}
        for node, state in sdfg.all_nodes_recursive():
            if isinstance(node, nodes.LibraryNode):
                pending.setdefault(id(state), []).append((node, state))
        if not pending:
            return
        rounds += 1
        if rounds > MAX_EXPANSION_ROUNDS + max(len(group) for group in pending.values()):
            remaining = sorted({type(node).__name__ for group in pending.values() for node, _ in group})
            raise NotImplementedError(f'MPR could not expand {", ".join(remaining)} after {rounds} rounds; '
                                      'a library node appears to expand into itself')

        # One per state, by GUID so the pick does not depend on graph iteration order.
        chosen = [sorted(group, key=lambda pair: pair[0].guid)[0] for group in pending.values()]
        described: Dict[int, Tuple[str, str, set]] = {}
        for node, state in chosen:
            available = type(node).implementations
            for candidate in PURE_IMPLEMENTATIONS:
                if candidate in available:
                    node.implementation = candidate
                    break
            if provenance is not None:
                description = description_of(node)
                # A library node's specialization hint has to be captured here for the same reason
                # its description does: the expansion consumes the node, and the loops it leaves
                # behind carry no memory of what chose their shape. Folded into the description so
                # the emitter's once-per-origin dedupe covers both -- a Scan expands into several
                # maps, and the trade is one trade, not one per map.
                if description is not None and node.specialization_hint:
                    description = f'{description}\n{node.specialization_hint}'
                if description is not None:
                    # Recorded BEFORE the expansion: afterwards the node is gone, and with it any
                    # way to ask what it was.
                    described[id(state)] = (node.guid, description, {existing.guid for existing in state.nodes()})

        selected = {id(node) for node, _ in chosen}
        sdfg.expand_library_nodes(recursive=False, predicate=lambda node: id(node) in selected)

        for _, state in chosen:
            record = described.get(id(state))
            if record is None:
                continue
            origin, description, before = record
            for produced in state.nodes():
                if produced.guid in before:
                    continue
                for guid in subtree_guids(produced, state):
                    provenance.setdefault(guid, (origin, description))


#: The prefix DaCe gives a data container that carries a program's return value. A single return is
#: ``__return``; a returned tuple is ``__return_0``, ``__return_1``, ... (``parser.py`` builds both).
RETURN_PREFIX = '__return'


def is_return_name(name: str) -> bool:
    """Whether ``name`` is a return container's name."""
    return name == RETURN_PREFIX or name.startswith(RETURN_PREFIX + '_')


def return_containers(sdfg: SDFG) -> List[Tuple[SDFG, str]]:
    """Every return container in ``sdfg``'s whole tree, as ``(owning SDFG, name)``.

    Both the DECLARATIONS and the ACCESS NODES are walked, because they answer different questions
    and MPR needs both. ``arrays_recursive`` finds a container that exists, including one a nested
    SDFG declared and never wired out; the access nodes are where a name is actually read or
    written, which is what lets a refusal name the state a reader can go and look at.

    Nested SDFGs are included on purpose. A nested ``__return`` is ordinary -- it is the nested
    SDFG's out-connector, and the value leaves through a memlet rather than through the entry
    signature -- but a nested one that is TRANSIENT leaves nowhere, and checking only the top level
    would not see it.

    :param sdfg: the outermost SDFG.
    :returns: ``(owner, name)`` pairs, deduplicated, in a deterministic order.
    """
    found: Dict[Tuple[int, str], Tuple[SDFG, str]] = {}
    for owner, name, _ in sdfg.arrays_recursive():
        if is_return_name(name):
            found.setdefault((owner.cfg_id, name), (owner, name))
    for node, parent in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.AccessNode) and is_return_name(node.data):
            owner = parent.sdfg
            found.setdefault((owner.cfg_id, node.data), (owner, node.data))
    return [found[key] for key in sorted(found)]


def refuse_by_value_returns(sdfg: SDFG) -> None:
    """Refuse a return container whose value could not reach the caller.

    At the TOP LEVEL the container is an entry-point parameter, and the descriptor decides whether
    the caller can read it. An ``Array`` is spelled ``T * __restrict__`` -- an out-parameter, which
    works. A ``Scalar`` is spelled ``T``, a BY-VALUE parameter, so the rendering would compute the
    result into the callee's own copy and the caller would read back whatever it passed in. That is
    a wrong answer rather than a compile error, which is why it is refused here and not left to the
    host compiler. The Python frontend widens a scalar return to ``Array(dtype, [1])`` before it
    gets this far, so the case is reachable only from a hand-built or non-Python-frontend SDFG --
    exactly where nothing else would catch it.

    Inside a NESTED SDFG the same name means something else: it is the nested SDFG's out-connector,
    and the value leaves through a memlet, so a scalar one is fine. What is not fine is a transient
    one, which is written into a buffer local to that nested SDFG and read by nobody.

    The fix in every case is a promotion pass -- rewrite the descriptor to a one-element array and
    re-subscript its accesses -- which is not written yet. Until it is, refuse: a rendering that
    silently discards the program's result is worse than no rendering.

    :param sdfg: the outermost SDFG.
    :raises NotImplementedError: naming the container and the SDFG that declares it.
    """
    for owner, name in return_containers(sdfg):
        desc = owner.arrays[name]
        where = f'{sdfg.name}: the return container {name!r}'
        if owner is not sdfg:
            where = f'{sdfg.name}: the return container {name!r} of the nested SDFG {owner.name!r}'
        if desc.transient:
            raise NotImplementedError(f'MPR cannot render {where} is transient, so nothing outside the SDFG that '
                                      'declares it can read the value it holds.')
        if owner is not sdfg:
            continue
        if isinstance(desc, dt.Scalar):
            raise NotImplementedError(f'MPR cannot render {where} is a Scalar, which the entry signature passes BY '
                                      "VALUE, so the result would be computed into the callee's copy and discarded. "
                                      f'Promote {name!r} to a one-element array before rendering.')
        if not isinstance(desc, dt.Array):
            raise NotImplementedError(f'MPR cannot render {where} is a {type(desc).__name__}, which has no '
                                      'plain-pointer spelling in the entry signature.')


def refuse_runtime_scopes(sdfg: SDFG) -> None:
    """Refuse the constructs whose only implementation is a DaCe runtime class.

    A ``Stream`` descriptor is emitted as ``dace::Stream<T>`` and a consume scope drives it through
    ``dace::Consume``: both are runtime templates carrying a lock-free queue, and neither has a
    standalone spelling that MPR could inline. Without this check a stream still fails, but as the
    self-containment assertion on the finished text, which names ``dace::Stream`` rather than the
    container it came from.

    The consume scope is checked FIRST because one always reads a stream, so the other order would
    report the stream it happens to drain and never the scope itself.

    :param sdfg: the outermost SDFG.
    :raises NotImplementedError: naming the consume scope or the stream, and the SDFG holding it.
    """
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, nodes.ConsumeEntry):
                raise NotImplementedError(f'MPR cannot render {state.label}/{node.label}: a consume scope is driven '
                                          'by the runtime class dace::Consume, whose queue and quiescence detection '
                                          'MPR does not provide. Express the work as a map before rendering.')
    for subsdfg, name, desc in sdfg.arrays_recursive():
        if isinstance(desc, dt.Stream):
            raise NotImplementedError(f'MPR cannot render {subsdfg.name}.{name}: a Stream is the runtime class '
                                      'dace::Stream, a lock-free queue with no standalone spelling. Rewrite the '
                                      'producer and consumer around an array before rendering.')


def prepare(sdfg: SDFG, provenance: Optional[Dict[str, Tuple[str, str]]] = None) -> None:
    """Make ``sdfg`` renderable as one host translation unit, in place.

    Four things happen: every written signature scalar is promoted to a length-1 array so it is
    addressable (:class:`~dace.transformation.passes.scalar_promotion.PromoteScalarOutputsToArrays`),
    what that could not make renderable is refused
    (:func:`refuse_by_value_returns`), every library node is pointed at its pure implementation and
    expanded (:func:`force_pure_expansions`), and lifetimes that would need a state struct are
    demoted (see :data:`LIFETIME_DEMOTIONS`). Anything MPR cannot express raises here rather than
    at compile time, where the message would be a C++ diagnostic about a name this module chose.

    The promotion runs in BOTH dialects, not only C. A by-value scalar out-parameter discards the
    result just as silently in C++; the C rendering merely also fails to compile, because a written
    scalar connector on a nested SDFG binds as ``T &`` there.

    :param sdfg: the SDFG to prepare. Call on a COPY -- :func:`mpr` does.
    :param provenance: filled in with the library-node descriptions the rendering will comment
                       with (see :func:`force_pure_expansions`).
    :raises NotImplementedError: if the SDFG needs a device compiler, holds a stream or a consume
                                 scope (:func:`refuse_runtime_scopes`), or carries a return container
                                 the entry signature cannot pass back (:func:`refuse_by_value_returns`).
    """
    device = uses_device_code(sdfg)
    if device:
        raise NotImplementedError('MPR renders one host translation unit, but ' + '; '.join(device) +
                                  '. Render the CPU form of this SDFG instead.')
    refuse_runtime_scopes(sdfg)
    PromoteScalarOutputsToArrays().apply_pass(sdfg, {})
    refuse_by_value_returns(sdfg)
    force_pure_expansions(sdfg, provenance)
    for _, _, desc in sdfg.arrays_recursive():
        demoted = LIFETIME_DEMOTIONS.get(desc.lifetime)
        if demoted is not None:
            desc.lifetime = demoted


def frame_object(objects: List[CodeObject], name: str) -> CodeObject:
    """The one translation unit MPR renders, out of what code generation produced.

    A second LINKABLE object means the SDFG was split across files -- a ``.cu`` for a GPU kernel, or
    a separate unit per nest under ``codegen_params.split_nsdfg_translation_units``. Either way the
    single-file contract is broken, and returning just the frame would return a unit that does not
    contain the computation. Non-linkable objects (the call header, the sample ``main``) are
    generated for every SDFG and are not part of the build, so they do not count.

    :param objects: what :func:`dace.codegen.codegen.generate_code` returned.
    :param name: the SDFG's name, for the message.
    :returns: the frame code object.
    :raises NotImplementedError: if the code was split across translation units.
    """
    linkable = [obj for obj in objects if obj.linkable]
    if len(linkable) != 1:
        extra = ', '.join(f'{obj.name}.{obj.language}' for obj in linkable if obj.target_type != 'Frame')
        raise NotImplementedError(f'MPR renders one translation unit, but {name} generated {len(linkable)}: '
                                  f'{extra}. Turn off the split-translation-unit codegen parameters.')
    return linkable[0]


def written_containers(sdfg: SDFG) -> OrderedSet:
    """The container names some state WRITES, at any nesting depth.

    An ``AccessNode`` with an incoming edge is a write. Nested SDFGs are walked too: a nested
    transient that happens to share an outer name is then reported as written, which is the SAFE
    direction -- it only ever withholds a ``const``, never grants one wrongly.

    :param sdfg: the SDFG to scan.
    :returns: the written names.
    """
    written: OrderedSet = OrderedSet()
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, nodes.AccessNode) and state.in_degree(node) > 0:
                written.add(node.data)
            elif isinstance(node, nodes.NestedSDFG):
                written |= written_containers(node.sdfg)
    return written


def readonly_entry_arrays(sdfg: SDFG) -> OrderedSet:
    """The entry point's ARRAY parameters that nothing writes -- the ones whose pointee is const.

    Read off the same SDFG the signature is generated from, so the qualifier and the argument list
    cannot disagree. Callers that publish a binding for the rendering (the ABI ``const`` flag)
    should derive it from HERE rather than recomputing it, which is what let a binding say
    ``const: true`` while the rendered signature said otherwise.

    :param sdfg: the PREPARED SDFG -- the one whose ``arglist()`` is the signature.
    :returns: the names to qualify.
    """
    written = written_containers(sdfg)
    return OrderedSet(name for name, desc in sdfg.arglist().items()
                      if isinstance(desc, dt.Array) and name not in written)


def entry_parameter_name(param: str) -> str:
    """The declared name in one entry-signature parameter (``float * __restrict__ a`` -> ``a``)."""
    return param.strip().split()[-1].lstrip('*')


def qualify_readonly_pointers(code: str, sdfg: SDFG, entry: str) -> str:
    """Add ``const`` to the entry point's read-only pointer parameters.

    The signature is built by ``Data.as_arg``, which is shared with every other DaCe backend and
    has no notion of a read-only parameter, so the qualifier is applied here instead -- the same
    place the ctype names are re-spelled, and for the same reason. Without it a rendering hands a
    non-const pointer to a buffer it only reads: every C and C++ linter reports it
    (cppcheck ``constParameterPointer``), and a published binding that derived ``const`` from the
    written-set disagreed with the signature it was supposed to describe.

    Adding ``const`` cannot break a caller: a ``T *`` converts to ``const T *`` implicitly in both
    languages, and the parameter is still passed as one pointer, so the ABI is unchanged.

    :param code: the rendered unit.
    :param sdfg: the PREPARED SDFG.
    :param entry: the entry point's name.
    :returns: the unit with the read-only parameters qualified.
    """
    readonly = readonly_entry_arrays(sdfg)
    if not readonly:
        return code
    pattern = re.compile(r'\bvoid\s+%s\s*\(' % re.escape(entry))
    out, cursor = [], 0
    for match in pattern.finditer(code):
        opened = match.end() - 1
        closed = code.index(')', opened)
        # The parameter list is split on commas and terminated at the first ``)``, which is exact
        # for pointers and by-value scalars and wrong for anything nested (a function-pointer
        # parameter). MPR emits neither today; refuse rather than mangle the signature if it ever
        # does.
        if '(' in code[opened + 1:closed]:
            raise NotImplementedError(f'MPR cannot qualify the entry signature of {entry}: a parameter carries a '
                                      'nested parameter list, which this rewrite cannot split.')
        params = [p.strip() for p in code[opened + 1:closed].split(',')]
        params = [f'const {p}' if entry_parameter_name(p) in readonly else p for p in params]
        out.append(code[cursor:opened + 1] + ', '.join(params))
        cursor = closed
    out.append(code[cursor:])
    return ''.join(out)


#: ``language`` argument -> the dialect that renders it. ``'c++'`` is the default and stays the
#: historical behaviour exactly.
LANGUAGES: Dict[str, mpr_lowering.Dialect] = {
    'c++': mpr_lowering.Dialect.STANDALONE,
    'c': mpr_lowering.Dialect.STANDALONE_C,
}


def dialect_for(language: str) -> mpr_lowering.Dialect:
    """The dialect ``language`` names.

    :param language: ``'c++'`` or ``'c'``.
    :raises ValueError: for any other value, naming what is available.
    """
    try:
        return LANGUAGES[language]
    except KeyError:
        raise ValueError(f'MPR renders {sorted(LANGUAGES)}, not {language!r}') from None


def preamble(code: str, dialect: mpr_lowering.Dialect = mpr_lowering.Dialect.STANDALONE) -> str:
    """The include block and inline definitions ``code`` needs, in the order they must appear.

    Derived from the finished text (:func:`~dace.mpr_lowering.helpers_used`) rather than from the
    SDFG: helpers arrive from two printers -- symbolic expressions and tasklet bodies -- and only
    the emitted unit has seen both.

    :param code: the emitted translation unit, without its preamble.
    :param dialect: which standalone dialect emitted it.
    :returns: the preamble, ending in a blank line.
    """
    used = mpr_lowering.helpers_used(code, dialect)
    definitions = mpr_lowering.definitions_for(used, dialect)
    headers = mpr_lowering.headers_for(used, dialect)
    lines = ['// Rendered by DaCe MPR (maximal parallel rendering): self-contained, no DaCe runtime.']
    lines += [f'#include {header}' for header in headers]
    if dialect is mpr_lowering.Dialect.STANDALONE_C:
        lines.append(mpr_lowering.C_UNDEF_LINE)
    if definitions:
        lines.append('')
        lines.append('// Functions the DaCe runtime headers would otherwise provide.')
        lines.extend(definitions)
    lines.append('')
    return '\n'.join(lines)


#: What a finished rendering must not contain, and what each one means. MPR's own gate, checked on
#: the emitted text before it is handed back: every one of these is a construct that BUILDS inside
#: the DaCe tree (where the runtime headers are on the include path) and fails only once the output
#: is used the way MPR promises it can be. Failing here names the SDFG construct that caused it,
#: which a link error against ``libdace`` never would.
#:
#: The test harness (``tests/codegen/mpr/conftest.py``) states the same contract independently and
#: on purpose -- it is the acceptance spec, written from outside, and it also compiles the result
#: with no include path at all, which is the only check that cannot be fooled by a table that
#: forgot an entry.
BANNED: Tuple[Tuple[re.Pattern, str], ...] = (
    (re.compile(r'#\s*include\s*[<"][^>"]*dace/'), 'a DaCe runtime header'),
    (re.compile(r'#\s*include\s*"'), 'a quoted (build-tree-relative) include'),
    (re.compile(r'CopyND'), 'a dace::CopyND copy -- insert explicit copies before rendering'),
    (re.compile(r'__dace_(init|exit)\w*'), 'a DaCe init/exit entry point'),
    (re.compile(r'\bdace\s*::'), 'a DaCe runtime symbol'),
    (re.compile(r'\bDACE_[A-Z]'), 'a DaCe preprocessor macro'),
    (re.compile(r'__state\b'), 'a state-struct dereference'),
)

#: The scalar type spellings a code generator writes a declarator with. Used to anchor the
#: reference-parameter pattern below: a bare ``\w+\s*&\s*\w+`` would also match the bitwise
#: ``exponent & 1)`` in MPR's own ``ipow``, and a gate with a false positive gets disabled.
_C_DECLARED_TYPES = (r'(?:const\s+)?(?:unsigned\s+|signed\s+)?'
                     r'(?:long\s+double|long\s+long|u?int(?:8|16|32|64)_t|double|float|bool|char|short|int|long)'
                     r'(?:\s+_Complex)?')

#: What a finished C rendering must not contain, on top of :data:`BANNED`. Every one of these is
#: valid C++ that the C++ dialect emits on purpose, so a leak is a dialect branch that was missed
#: rather than a construct that should never exist.
BANNED_C: Tuple[Tuple[re.Pattern, str], ...] = BANNED + (
    (re.compile(r'\bstd\s*::'), 'a C++ standard-library symbol'),
    (re.compile(r'\btemplate\s*<'), 'a C++ template'),
    (re.compile(r'extern\s*"C"'), 'a C++ language linkage specifier'),
    (re.compile(r'\bstatic_cast\s*<'), 'a C++ static_cast'),
    (re.compile(r'\bnew\s'), 'a C++ new-expression'),
    (re.compile(r'\bdelete\b'), 'a C++ delete-expression'),
    # ``constexpr`` on an OBJECT is C23 and is how MPR emits an SDFG constant, so only the FUNCTION
    # form is banned: a qualifier run ending in a declarator with a parameter list.
    (re.compile(r'\b(?:constexpr|consteval)\b[^;=\n]*\b\w+\s*\([^;]*\)\s*\{'), 'a constexpr/consteval function'),
    (re.compile(_C_DECLARED_TYPES + r'\s*&\s*\w+\s*[,)]'), 'a C++ reference parameter'),
)


def verify(code: str, name: str, dialect: mpr_lowering.Dialect = mpr_lowering.Dialect.STANDALONE) -> None:
    """Assert ``code`` is self-contained, or raise naming what leaked.

    :param code: the rendered translation unit.
    :param name: the SDFG's name, for the message.
    :param dialect: which standalone dialect rendered it, choosing the table to check against.
    :raises RuntimeError: on the first banned construct found.
    """
    banned = BANNED_C if dialect is mpr_lowering.Dialect.STANDALONE_C else BANNED
    for pattern, meaning in banned:
        match = pattern.search(code)
        if match is None:
            continue
        start = code.rfind('\n', 0, match.start()) + 1
        end = code.find('\n', match.end())
        line = code[start:end if end != -1 else len(code)].strip()
        raise RuntimeError(f'MPR rendered {name} with {meaning} ({match.group(0)!r}), so the result is not '
                           f'self-contained:\n    {line}')


class Rendering(NamedTuple):
    """A rendered SDFG: the C++ text, and the SDFG that text was generated from.

    The second field is not a convenience. MPR renders a PREPARED COPY -- library nodes expanded
    through their pure implementations, lifetimes demoted -- and preparation can change the
    ARGUMENT LIST: expanding a ``Reduce`` into a map introduces the extent symbol the library node
    had kept to itself, so the entry point takes an argument the original SDFG's ``arglist()``
    never mentions. Calling the rendered code means calling it with THIS SDFG's arglist; the
    original's would silently drop that symbol and run the kernel on an uninitialized extent.
    """
    #: The self-contained translation unit.
    code: str
    #: The prepared copy that was rendered. Its ``arglist()`` is the entry point's signature.
    sdfg: SDFG


def render(sdfg: SDFG, validate: bool = True, language: str = 'c++') -> Rendering:
    """Render ``sdfg`` and return the text together with the SDFG it describes.

    :param sdfg: the SDFG to render. Not modified -- a copy is prepared and rendered.
    :param validate: validate the SDFG during code generation.
    :param language: ``'c++'`` (the default, C++20) or ``'c'`` (C23). Both are self-contained: the
                     result builds with a bare host compiler, no ``-I``, no libdace, no BLAS.
    :returns: the :class:`Rendering`.
    :raises NotImplementedError: if the SDFG needs anything a single host unit cannot hold; the
                                 message names the construct.
    :raises ValueError: if ``language`` is neither.
    """
    dialect = dialect_for(language)
    prepared = copy.deepcopy(sdfg)
    provenance: Dict[str, Tuple[str, str]] = {}
    prepare(prepared, provenance)
    # DACE_* environment variables outrank set_temporary, so a shell that pins the CPU generator to
    # ``legacy`` would silently render through the wrong one -- and the legacy generator emits
    # ``dace::CopyND`` and state-struct accesses that no dialect switch can take back. Refuse.
    with set_temporary('compiler', 'cpu', 'implementation', value='experimental_readable'):
        selected = Config.get('compiler', 'cpu', 'implementation')
        if selected != 'experimental_readable':
            raise RuntimeError('MPR builds on the readable CPU code generator, but '
                               f'compiler.cpu.implementation is pinned to {selected!r} (a DACE_* environment '
                               'variable outranks the in-process setting). Unset it to render.')
        with mpr_lowering.dialect_scope(dialect):
            with mpr_lowering.provenance_scope(provenance):
                objects = codegen.generate_code(prepared, validate=validate)
                body = frame_object(objects, sdfg.name).clean_code
                # Type names reach the text from the entry signature and from declarations, neither
                # of which goes through an expression printer, so the rename runs over the whole unit.
                body = mpr_lowering.rewrite_ctypes(body, dialect)
                body = qualify_readonly_pointers(body, prepared, sdfg.name)
                if dialect is mpr_lowering.Dialect.STANDALONE_C:
                    # ``__restrict__`` is the GNU spelling ``Data.as_arg`` emits because C++ has no
                    # ``restrict`` keyword. C does, and it is the one a C23 unit should carry.
                    body = re.sub(r'\b__restrict__\b', 'restrict', body)
    code = preamble(body, dialect) + body
    verify(code, sdfg.name, dialect)
    return Rendering(code, prepared)


def mpr(sdfg: SDFG, validate: bool = True, language: str = 'c++') -> str:
    """Render ``sdfg`` as one self-contained translation unit.

    The SDFG is copied first, so neither the lifetime demotions nor the code generator's own
    in-place lowering (library expansion, inlining, explicit copies) is visible to the caller.

    Use :func:`render` instead where the code will actually be CALLED: preparation can add an
    argument, and :class:`Rendering` carries the SDFG that says which.

    :param sdfg: the SDFG to render.
    :param validate: validate the SDFG during code generation.
    :param language: ``'c++'`` (the default) or ``'c'``.
    :returns: the translation unit, defining ``extern "C" void <sdfg.name>(<arglist>)`` in C++ and
              ``void <sdfg.name>(<arglist>)`` in C, whose ABI is the same.
    :raises NotImplementedError: if the SDFG needs anything a single host unit cannot hold; the
                                 message names the construct.
    :raises ValueError: if ``language`` is neither.
    """
    return render(sdfg, validate=validate, language=language).code
