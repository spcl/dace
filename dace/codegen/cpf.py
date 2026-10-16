# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""CPF: rendering an SDFG as self-contained source.

``cpf(sdfg)`` returns C++ that a bare host compiler accepts -- ``g++ -std=c++20 -fopenmp`` with no
``-I``, no ``libdace``, no BLAS -- and that computes what the SDFG computes. ``cpf(sdfg,
language='c')`` returns C23 under the same terms; the two are one semantics in two spellings, held
together by ``tests/codegen/cpf/test_lowering_table.py``. The point is not portability for its own
sake: it is that the result can be read, diffed and edited on its own, so a maximally parallel
rendering of a program can be compared against the original the way Pluto's output is.

Nothing here re-implements code generation. The readable CPU generator
(``compiler.cpu.implementation = experimental_readable``) already emits the shape CPF wants --
``<array>_idx`` index helpers, connector-free tasklets, ``#pragma omp parallel for`` on parallel
maps. CPF is that generator run under :attr:`~dace.cpf_lowering.Dialect.STANDALONE`, which changes
three things and no more:

* the printers spell runtime functions with the standard library or CPF's own inline definitions
  (:mod:`dace.cpf_lowering`),
* the frame emits ``extern "C" void <name>(<arglist>)`` instead of the state-carrying
  ``__program_<name>_internal`` plus its ``__dace_init`` / ``__dace_exit`` pair
  (``framecode.generate_standalone_footer``),
* this module prepends the preamble -- system headers, then exactly the inline definitions the
  finished text calls.

The preamble is assembled LAST, from the emitted text, because that is the only point that knows
which helpers were used. Deriving it from the SDFG instead would mean predicting the printers'
output, and a helper reached through a tasklet body (not a memlet subset) would be missed.

A device dialect (``'hip'``, ``'cuda'``) renders the SDFG the way DaCe builds it: TWO units, the host
unit (``.cpp``, the ``extern "C"`` entry, which only launches) and the device unit (``.hip`` / ``.cu``,
the ``__global__`` kernels and the launchers the host unit calls), both compiled by the device compiler.

What CPF refuses, it refuses loudly -- see :func:`prepare`, and the standalone paths in
``framecode``. A rendering that quietly dropped an initializer, a caller-supplied buffer or a GPU
kernel would still compile and still produce numbers, just not the SDFG's.
"""

import contextlib
import copy
import dataclasses
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from typing import NamedTuple

from dace import cpf_lowering, dtypes, symbolic, tile_functions
from dace import data as dt
from dace.codegen import codegen
from dace.codegen.codeobject import CodeObject
from dace.config import Config, set_temporary
from dace.libraries.blas.nodes.gemm import Gemm
from dace.libraries.fft.environments import fftw3
from dace.libraries.tileops.dispatch import FUNCTION_ISAS, ISA, isa_implementation, widest_function_isa
from dace.libraries.tileops.nodes.tile_op import TileOp
from dace.library import get_environment
from dace.memlet import Memlet
from dace.optionals import required
from dace.ordered import OrderedSet
from dace.sdfg import SDFG, SDFGState, nodes
from dace.sdfg.replace import replace_properties_dict
from dace.sdfg.scope import is_devicelevel_gpu
from dace.transformation.passes.canonicalize.annotate_loop_kinds import AnnotateLoopKinds
from dace.transformation.passes.canonicalize.finalize import blas_addresses, explicit_accumulating_copies
from dace.transformation.passes.scalar_promotion import PromoteScalarOutputsToArrays

#: The storage types CPF can render, as an ALLOWLIST. Ordinary host memory and plain locals, and
#: nothing else: ``CPU_Pinned`` is host memory but is allocated through the CUDA API, and the
#: accelerator storages (GPU, SVE, Snitch) each need a compiler CPF does not invoke. An allowlist
#: rather than a list of the refused ones because ``StorageType`` is extensible -- a storage
#: registered by a backend CPF has never heard of must refuse, not slip through.
HOST_STORAGE = frozenset(
    {
        dtypes.StorageType.Default,
        dtypes.StorageType.Register,
        dtypes.StorageType.CPU_Heap,
        dtypes.StorageType.CPU_ThreadLocal,
    }
)

#: The schedules CPF can render, likewise an allowlist. ``CPU_Multicore`` is the OpenMP loop that
#: makes the rendering parallel in the first place; the rest are the sequential and unspecified
#: forms. Anything else (GPU device/thread-block, FPGA, SVE) needs another compiler.
HOST_SCHEDULES = frozenset(
    {
        dtypes.ScheduleType.Default,
        dtypes.ScheduleType.Sequential,
        dtypes.ScheduleType.CPU_Multicore,
        dtypes.ScheduleType.CPU_Persistent,
    }
)

#: Lifetimes whose buffer the ordinary generators park in the state struct, and what CPF turns each
#: into. ``Persistent`` and ``Global`` outlive one call only so a repeated invocation can reuse the
#: allocation; a single self-contained entry point has no second invocation to reuse it, so SDFG
#: lifetime -- allocate on entry, free on return -- computes the same values. ``External`` is NOT
#: here: its buffer comes from the caller through an init handshake, so demoting it would allocate
#: a private buffer and silently discard what the caller passed. framecode refuses it instead.
#: What each library node computes, as one line CPF writes above the code its expansion produced.
#:
#: A pure expansion is loops and tasklets; nothing in it says the loops were a Cholesky
#: factorization. The comment is what keeps the rendering readable as the program it came from --
#: which is the point of CPF, since the output exists to be read and edited rather than linked.
#:
#: Keyed by class name. Every library node registered in the process must appear here (the suite
#: asserts it), so a new node arrives with a description instead of rendering as anonymous loops.
LIBRARY_NODE_DESCRIPTIONS: dict[str, str] = {
    "Abort": "MPI_Abort: terminate the communicator",
    "AllNode": "all: true where every element along the reduced axes is true",
    "Allgather": "MPI_Allgather: gather from every rank to every rank",
    "Allreduce": "MPI_Allreduce: reduce across ranks, result on every rank",
    "Alltoall": "MPI_Alltoall: every rank exchanges a block with every rank",
    "AnyNode": "any: true where any element along the reduced axes is true",
    "ArgMax": "argmax: index (and value) of the largest element along the reduced axis",
    "ArgMin": "argmin: index (and value) of the smallest element along the reduced axis",
    "ArgReduce": "argument reduction: index of the element the reduction selected",
    "Asum": "BLAS asum: sum of absolute values of a vector",
    "Axpy": "BLAS axpy: y = alpha * x + y",
    "BackwardPass": "autodiff backward pass: the reverse-mode derivative of the forward subgraph",
    "Barrier": "MPI_Barrier: synchronize the communicator",
    "BatchedMatMul": "batched matrix product: one gemm per batch index",
    "Bcast": "MPI_Bcast: broadcast from the root rank",
    "BlacsGridInit": "Cblacs_gridinit: build a BLACS process grid on a communicator",
    "BlockCyclicGather": "gather block-cyclic (ScaLAPACK) distributed data",
    "BlockCyclicScatter": "scatter data in the block-cyclic (ScaLAPACK) distribution",
    "BlockGather": "gather block-distributed data onto one rank",
    "BlockScatter": "scatter data to ranks in blocks",
    "Broadcast": "broadcast: expand operands to a common shape",
    "CSRMM": "sparse CSR matrix times dense matrix",
    "CSRMV": "sparse CSR matrix times dense vector",
    "CShift": "circular shift along an axis",
    "Cholesky": "Cholesky factorization: A = L @ L^T",
    "CodeLibraryNode": "verbatim code supplied by the SDFG author",
    "CommF2c": "MPI_Comm_f2c: convert a Fortran communicator handle",
    "CommRank": "MPI_Comm_rank: this rank in the communicator",
    "CommSize": "MPI_Comm_size: number of ranks in the communicator",
    "CommSplit": "MPI_Comm_split: split the communicator",
    "Copy": "BLAS copy: y = x",
    "CopyLibraryNode": "copy: write one buffer into another",
    "CountLibraryNode": "count: number of elements satisfying the predicate",
    "Dot": "BLAS dot: inner product of two vectors",
    "Dummy": "MPI placeholder node carrying an ordering dependency",
    "Eigh": "eigh: eigenvalues and eigenvectors of a Hermitian matrix",
    "Einsum": "einsum: a contraction over the index expression",
    "ExternalCall": "external call: a nest run from its own SDFG or called in a separately compiled library",
    "ExtremeIndex": "extreme index: position of the smallest or largest element along the reduced axis",
    "FFT": "discrete Fourier transform",
    "FFTInterpolate": "Fourier interpolation: resample through the frequency domain",
    "FillLibraryNode": "fill: set every element to a constant",
    "FindFirst": "find-first: index of the first element satisfying the predicate",
    "FortranIONode": "Fortran I/O statement",
    "FourierTransform": "discrete Fourier transform, forward or inverse",
    "Gather": "MPI_Gather: collect from every rank onto the root",
    "Gatherv": "MPI_Gatherv: collect variable-sized blocks onto the root",
    "Gearbox": "gearbox: change the element width of a stream",
    "Gemm": "BLAS gemm: C = alpha * A @ B + beta * C",
    "Gemv": "BLAS gemv: y = alpha * A @ x + beta * y",
    "Ger": "BLAS ger: A = alpha * x @ y^T + A (rank-1 update)",
    "Getrf": "LAPACK getrf: LU factorization with partial pivoting",
    "Getri": "LAPACK getri: matrix inverse from an LU factorization",
    "Getrs": "LAPACK getrs: solve A @ X = B from an LU factorization",
    "IFFT": "inverse discrete Fourier transform",
    "Iamax": "BLAS iamax: index of the largest absolute value in a vector",
    "IntegerSort": "integer sort: counting/radix sort of an integer key array",
    "Inv": "matrix inverse",
    "Irecv": "MPI_Irecv: non-blocking receive",
    "Isend": "MPI_Isend: non-blocking send",
    "MPINode": "MPI operation",
    "MaskReduction": "mask reduction: fold of a mask along the reduced axes",
    "MaskedCopyLibraryNode": "masked tile copy: load or store a tile under a per-lane mask",
    "MatMul": "matrix product, dispatched to gemm / gemv / batched gemm by operand rank",
    "MergeLibraryNode": "merge: select elementwise between operands by a condition",
    "NamelistRead": "Fortran namelist read",
    "Norm2": "norm2: Euclidean norm",
    "Nrm2": "BLAS nrm2: Euclidean norm of a vector",
    "ONNXOp": "ONNX operator",
    "Pgemm": "PBLAS pgemm: distributed matrix product",
    "Pgemv": "PBLAS pgemv: distributed matrix-vector product",
    "Potrf": "LAPACK potrf: Cholesky factorization of a positive-definite matrix",
    "Read": "Fortran read statement",
    "Recv": "MPI_Recv: blocking receive",
    "Redistribute": "redistribute an array between two process grids",
    "Scal": "BLAS scal: x = alpha * x",
    "Scan": "scan: running (prefix) fold along an axis",
    "StateScan": "state scan: a finite-state or linear recurrence run as a parallel prefix over blocks",
    "Scatter": "MPI_Scatter: distribute from the root to every rank",
    "ScatterConflictCheck": "scatter conflict check: detect indices written more than once",
    "Send": "MPI_Send: blocking send",
    "Solve": "solve the linear system A @ X = B",
    "Stencil": "stencil: apply the given neighbourhood expression at every point",
    "Swap": "BLAS swap: exchange two vectors",
    "Syevd": "LAPACK syevd: eigendecomposition of a Hermitian matrix, divide and conquer",
    "Symm": "BLAS symm: C = alpha * A @ B + beta * C with A symmetric",
    "Symmetrize": "symmetrize: average a matrix with its transpose",
    "Symv": "BLAS symv: y = alpha * A @ x + beta * y with A symmetric",
    "Syr2k": "BLAS syr2k: C = alpha * (A @ B^T + B @ A^T) + beta * C (symmetric rank-2k update)",
    "Syrk": "BLAS syrk: C = alpha * A @ A^T + beta * C (symmetric rank-k update)",
    "TensorDot": "tensor contraction over the given axis pairs",
    "TensorTranspose": "tensor transpose: permute the axes",
    "TileBinop": "tile binary operation, elementwise over a tile",
    "TileFMA": "tile fused multiply-add",
    "TileITE": "tile select: elementwise choice between two tiles",
    "TileIota": "tile iota: fill a tile with its own indices",
    "TileGather": "tile gather: read a tile through index tiles, replicated, transposed or broadcast",
    "TileMMA": "tile matrix multiply-accumulate",
    "TileMaskGen": "tile mask: the predicate for a partial tile",
    "TileOp": "tile operation: a loop over the lanes of a tile",
    "TileReduce": "tile reduction",
    "TileScan": "tile scan: running (prefix) fold over the lanes of a tile",
    "TileScatter": "tile scatter: write a tile through index tiles, transposed or broadcast",
    "TileUnop": "tile unary operation, elementwise over a tile",
    "Transpose": "matrix transpose",
    "Trmm": "BLAS trmm: B = alpha * op(A) @ B with A triangular",
    "Trmv": "BLAS trmv: x = op(A) @ x with A triangular",
    "Trsm": "BLAS trsm: solve op(A) @ X = alpha * B with A triangular",
    "Trsv": "BLAS trsv: solve op(A) @ x = b with A triangular",
    "UnregisteredLibraryNode": "a library node whose implementing module is not installed",
    "Wait": "MPI_Wait: complete one non-blocking request",
    "Waitall": "MPI_Waitall: complete every non-blocking request",
    "Write": "Fortran write statement",
}

#: Descriptions for the one class name two libraries share. Looked up as ``<module>.<class>``,
#: before :data:`LIBRARY_NODE_DESCRIPTIONS` -- which deliberately has NO ``Reduce`` entry, so
#: neither meaning can be served to the other by a bare-name lookup.
QUALIFIED_DESCRIPTIONS: dict[str, str] = {
    "dace.libraries.mpi.nodes.reduce.Reduce": "MPI_Reduce: reduce across ranks onto the root",
    "dace.libraries.standard.nodes.reduce.Reduce": "reduction over the given axes with the given operator",
}

#: Library-node implementations CPF selects, best first. The criterion is NOT the name ``pure`` and
#: NOT an empty ``environments`` list -- it is "expands to something a standalone unit can compile",
#: which means SDFG content (maps and tasklets) or a call into the C++ standard library.
#:
#: ``pure`` is an SDFG made of maps and tasklets -- which is the whole point: it renders as loops,
#: and it is the PARALLEL form. ``pure-seq`` is the sequential fallback for a node with no parallel
#: pure expansion, ``MappedTasklet`` the spelling used by a copy node that has no ``Auto``.
#:
#: DELIBERATELY ABSENT, and the reason this list is a checked allowlist rather than a filter on
#: ``environments``: DaCe's faster-sounding implementations declare no environment and still name a
#: symbol nothing here defines. ``Reduce``'s ``vectorized`` lowers onto ``horizontal_reduce_*`` from
#: ``dace/horizontal_reduce.h`` (its ``CPU`` is re-spelled, see :data:`RENDERABLE_BY_NODE`). ``Auto`` is a
#: per-node dispatcher that can land on those, so it is listed only for the nodes whose ``Auto``
#: always renders (:data:`RENDERABLE_BY_NODE`).
RENDERABLE_IMPLEMENTATIONS = ("pure", "pure-seq", "MappedTasklet")

#: Per-node-type implementations that ARE renderable, tried ahead of the global list. The criterion
#: is unchanged -- "expands to something a standalone unit can compile" -- but it is a property of
#: the expansion, not of the name, so a name absent from the global list can still qualify for one
#: node and not for another.
#:
#: CPF is a parallel form, so a node renders through its OpenMP ``CPU`` lowering whenever CPF can
#: spell what that lowering reaches. ``ArgReduce``'s emits a self-contained ``omp declare
#: reduction`` over a (value, index) pair; ``Reduce``'s ``dace::reduce`` entry points, ``FindFirst``'s
#: ``dace::find_first_index``, ``Scan``'s
#: ``dace::scan`` entry points and ``ScatterConflictCheck``'s ``dace::detect_collision`` are
#: re-spelled by :mod:`dace.cpf_lowering` as the OpenMP ``reduction`` loop, the cancelling search, the
#: ``inscan`` scan and the tagged-write check. ``StateScan``'s ``CPU`` lowering reaches no runtime at
#: all -- it is its ``pure`` SDFG with the two block maps on ``CPU_Multicore`` instead of ``Sequential`` --
#: and ``pure`` would run the scan's extra work (``(count + 1) * N`` steps for a finite scan) on one
#: thread, slower than the loop the scan replaced. ``Reduce``'s ``pure`` would be a parallel loop with a
#: ``critical`` section per element (argmax_value under canonicalize: ~0.5 s for 6M doubles). Without them a node falls to ``pure``, a SEQUENTIAL loop, and the rendered
#: unit hands its reader a serial loop for work the canonicalize pipeline itself parallelizes --
#: ``argmax_with_index``, ``tsvc_2_s318`` and ``tsvc_2_s3110`` all measured 9-11x over numpy in the
#: parallel form and rendered with no ``omp`` at all.
#:
#: ``Auto`` is the copy and fill nodes' own selector, preferred over any fixed spelling because it
#: picks BY SIZE AND LAYOUT: one ``std::memcpy``/``memset`` for a contiguous copy that runs once, a
#: parallel mapped tasklet past the threshold where the map is worth its overhead. Its
#: ``dace::CopyND`` branch needs a ``GPU_Shared`` endpoint, which :func:`prepare` has already
#: refused by the time this runs.
RENDERABLE_BY_NODE: dict[str, tuple[str, ...]] = {
    "ArgReduce": ("CPU",),
    "CopyLibraryNode": ("Auto",),
    "FillLibraryNode": ("Auto",),
    "FindFirst": ("CPU",),
    "IntegerSort": ("isocpp",),
    "Reduce": ("CPU",),
    "Scan": ("CPU",),
    "ScatterConflictCheck": ("CPU",),
    "StateScan": ("CPU",),
}

#: Per-node-type implementations tried ahead of everything else when a DEVICE dialect renders a
#: node that READS OR WRITES DEVICE MEMORY from host level. The criterion is the same one the two
#: tables above use -- "expands to something this unit can compile" -- with the machine added: a
#: host expansion over device pointers compiles and then reads the wrong memory.
#:
#: Only the node types whose host expansion is a FREE TASKLET are here. That is the shape that
#: cannot survive: a tasklet holding ``scan_incl_sum(...)`` or the cancelling OpenMP search stays
#: host code and dereferences a device pointer, and ``FindFirst`` refuses outright rather than let
#: it. A ``pure`` expansion made of MAPS (``Reduce``, ``ArgReduce``, the copy and fill nodes) is not
#: in that shape: the maps inherit the library node's GPU schedule and become a kernel, which is
#: both correct and the rendering CPF wants -- so ``Reduce`` takes ``pure`` here, ahead of its host
#: ``CPU`` lowering, its ``GPUAuto`` being a ``dace::`` cub call where its ``pure`` is loops.
#:
#: So are the node types whose ``pure`` expansion is maps that do NOT take the node's schedule.
#: ``StateScan``'s ``pure`` pins every map ``Sequential`` (it is the form for readers of the SDFG), so
#: at host level over device memory its step runs as host code and its loop headers read the device
#: tables (crc16's ``data``, dfa's ``__symbols``). Its ``CUDA`` expansion is the same SDFG with the
#: maps on ``GPU_Device`` -- kernels the device unit holds, no library call -- and it is the
#: implementation canonicalize already chose for the node.
#:
#: The device expansions named here reach the runtime the same way the host ones do, and CPF spells
#: what they reach: ``get_scratch``, ``find_first_index_device`` and ``inclusive_affine`` are
#: inline definitions in :mod:`dace.cpf_lowering`, exactly as ``find_first_index`` and the scans are
#: for the host.
RENDERABLE_BY_NODE_DEVICE: dict[str, tuple[str, ...]] = {
    "FindFirst": ("CUDA",),
    "Reduce": ("pure",),
    "Scan": ("CUDA",),
    "StateScan": ("CUDA",),
}

#: The one vendor library family each architecture renders through, best first. A BLAS, LAPACK, linear-algebra or
#: FFT node with one of these implementations becomes that library call rather than CPF loops: the libraries ship
#: with the toolkit (CUDA, ROCm) or are the one CPU BLAS the environment provides (OpenBLAS with its CBLAS and
#: LAPACKE interfaces, FFTW3 for transforms), so the unit only adds them to its link line (:attr:`Rendering.link`).
#: One variant per architecture: MKL, cuTENSOR/hipTENSOR, TBLIS and HPTT are not offered. A host-level node over
#: device memory takes its device dialect's family; every other node, the host one.
VENDOR_IMPLEMENTATIONS_HOST: tuple[str, ...] = ("OpenBLAS", "FFTW3")
#: Library nodes that move data rather than compute, kept as CPF loops although a vendor offers them: a transpose
#: is a copy the rest of the unit can fuse with, where ``cblas_domatcopy`` would be an opaque OpenBLAS extension.
VENDOR_EXCLUDED_NODES: frozenset[str] = frozenset({"Transpose"})
VENDOR_IMPLEMENTATIONS_DEVICE: dict[cpf_lowering.Dialect, tuple[str, ...]] = {
    cpf_lowering.Dialect.STANDALONE_CUDA: ("cuBLAS", "cuSolverDn", "cuFFT"),
    cpf_lowering.Dialect.STANDALONE_HIP: ("rocBLAS", "rocSOLVER", "hipFFT"),
}


#: Consecutive rounds of :func:`force_renderable_expansions` that may leave the library-node census
#: unchanged before it refuses. NOT a bound on total rounds: a state needs one round per node.
MAX_EXPANSION_STALLED_ROUNDS = 16

LIFETIME_DEMOTIONS = {
    dtypes.AllocationLifetime.Persistent: dtypes.AllocationLifetime.SDFG,
    dtypes.AllocationLifetime.Global: dtypes.AllocationLifetime.SDFG,
}

#: What a DEVICE dialect admits ON TOP of the host allowlists. One unit still, but one the device
#: compiler builds, so a kernel and its global memory belong in it. ``GPU_Shared`` is here because
#: a thread-block map's scratch is part of the kernel it lives in, not a separate allocation.
DEVICE_STORAGE = frozenset({dtypes.StorageType.GPU_Global, dtypes.StorageType.GPU_Shared})

#: Likewise for schedules: the device map, its thread-block map, and the persistent form.
DEVICE_SCHEDULES = frozenset(
    {
        dtypes.ScheduleType.GPU_Device,
        dtypes.ScheduleType.GPU_ThreadBlock,
        dtypes.ScheduleType.GPU_ThreadBlock_Dynamic,
        dtypes.ScheduleType.GPU_Persistent,
    }
)


def uses_device_code(sdfg: SDFG, dialect: cpf_lowering.Dialect | None = None) -> list[str]:
    """Names of the constructs in ``sdfg`` that ``dialect`` cannot render.

    Under a host dialect that is every device construct, because CPF would have to invoke a
    compiler it does not have. Under a device dialect the kernels are the POINT, so the allowlists
    widen by :data:`DEVICE_STORAGE` and :data:`DEVICE_SCHEDULES` -- and what is left refused is
    what no single unit can hold whichever compiler builds it (FPGA, SVE, Snitch).

    :param sdfg: the SDFG to inspect (recursively).
    :param dialect: the dialect being rendered; the ambient one when omitted.
    :returns: a description per offending construct, empty when the SDFG fits the dialect.
    :seealso: :data:`HOST_STORAGE`, :data:`HOST_SCHEDULES`.
    """
    on_device = (dialect in cpf_lowering.DEVICE_DIALECTS) if dialect is not None else cpf_lowering.device()
    storages = HOST_STORAGE | DEVICE_STORAGE if on_device else HOST_STORAGE
    schedules = HOST_SCHEDULES | DEVICE_SCHEDULES if on_device else HOST_SCHEDULES
    # Device tasklets are written in CPP -- there is no separate device language in the enum -- so
    # the language allowlist does not widen; what widens is where the CPP runs.
    languages = (dtypes.Language.Python, dtypes.Language.CPP)

    found: list[str] = []
    for subsdfg, name, desc in sdfg.arrays_recursive():
        if desc.storage not in storages:
            found.append(f"{subsdfg.name}.{name} is in {desc.storage.name} storage")
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, nodes.EntryNode) and node.schedule not in schedules:
                found.append(f"{state.label}/{node.label} has the {node.schedule.name} schedule")
            elif isinstance(node, nodes.Tasklet) and node.language not in languages:
                found.append(f"{state.label}/{node.label} is a {node.language.name} tasklet")
    return found


def description_of(node: nodes.LibraryNode | type[nodes.LibraryNode]) -> str | None:
    """The one-line description of what library node ``node`` computes.

    :param node: the library node, or its class. Both are accepted because the emitter has a node
                 and the coverage test has a class, and they must agree on the lookup -- a second
                 spelling of it is a second thing to keep in step.
    :returns: the description, or ``None`` if the node's class has none recorded.
    """
    cls = node if isinstance(node, type) else type(node)
    qualified = f"{cls.__module__}.{cls.__name__}"
    return QUALIFIED_DESCRIPTIONS.get(qualified) or LIBRARY_NODE_DESCRIPTIONS.get(cls.__name__)


def subtree_guids(node: nodes.Node, state: SDFGState) -> set[str]:
    """GUIDs of ``node`` and, if it is a nested SDFG, of every node inside it.

    An expansion usually lands as a single nested SDFG, and the code the reader sees comes from the
    tasklets and maps INSIDE it -- so those are the GUIDs the comment has to be able to attach to.
    """
    guids = {node.guid}
    if isinstance(node, nodes.NestedSDFG):
        guids.update(inner.guid for inner, _ in node.sdfg.all_nodes_recursive())
    return guids


def on_device_at_host_level(node: nodes.LibraryNode, state: SDFGState) -> bool:
    """Whether ``node`` issues a DEVICE library call: device operands, host level.

    Both halves matter. Device operands are what makes a host expansion wrong -- it would
    dereference a device pointer -- and host level is what makes a device call POSSIBLE: a node
    already inside a kernel has no launch to issue and must lower to device code instead, which is
    the split :func:`~dace.transformation.passes.canonicalize.finalize.libnode_is_device_code`
    makes for the GPU pipeline and this reuses so the two cannot drift apart.
    """
    from dace.transformation.passes.canonicalize.finalize import libnode_is_device_code

    owner = state.sdfg
    touches = any(
        owner.arrays[edge.data.data].storage in DEVICE_STORAGE
        for edge in state.all_edges(node)
        if edge.data is not None and edge.data.data in owner.arrays
    )
    return touches and not libnode_is_device_code(node, state, owner)


#: Floating types no vendor library CPF calls takes: BLAS, LAPACK and the FFTs come in single and double
#: precision, real and complex, and their bfloat16 / half entry points (``rocblas_gemm_ex``) are not what the
#: expansions call. A node over one of these takes the loops (HB's bf16 ML kernels).
VENDOR_UNTYPED_FLOATS: frozenset = frozenset(dtypes.FLOAT_TYPES) - {dtypes.float32, dtypes.float64}


def narrow_float_operands(node: nodes.LibraryNode, state: SDFGState) -> bool:
    """Whether ``node`` reads or writes a floating type no vendor library takes (:data:`VENDOR_UNTYPED_FLOATS`)."""
    arrays = state.sdfg.arrays
    return any(
        arrays[edge.data.data].dtype in VENDOR_UNTYPED_FLOATS
        for edge in state.all_edges(node)
        if edge.data is not None and edge.data.data in arrays
    )


def renderable_implementations(node: nodes.LibraryNode, state: SDFGState) -> tuple[str, ...]:
    """The implementations to try for ``node``, best first.

    :seealso: :data:`RENDERABLE_BY_NODE_DEVICE`, :data:`RENDERABLE_BY_NODE`,
              :data:`RENDERABLE_IMPLEMENTATIONS`.
    """
    if isinstance(node, TileOp) and node.implementation in tile_functions.ISAS:
        # Its tile functions are inline definitions of the unit, as :func:`target_tile_functions` chose them.
        return (node.implementation,)
    if is_devicelevel_gpu(state.sdfg, state, node):
        # Inside a kernel: the tabled lowerings are host runtime (OpenMP reductions and scans, the
        # cancelling search, ``std::sort``) and a vendor call is host code, so a ``__global__`` function
        # cannot call any of them (cp2k_grid_integrate's per-thread ``Scan`` reached ``scan_incl_product``).
        # Every tabled node has a loop expansion among the global ones, and loops are device code.
        return RENDERABLE_IMPLEMENTATIONS
    preferred = RENDERABLE_BY_NODE.get(type(node).__name__, ())
    # A GEMM whose operands no leading dimension names (a view strided on both matrix axes) is one finalize already
    # gave the loop expansion; a vendor call in front of it would raise ``sAM or sAK should be 1`` on expansion.
    vendored = (
        type(node).__name__ not in VENDOR_EXCLUDED_NODES
        and not (isinstance(node, Gemm) and not blas_addresses(node, state))
        and not narrow_float_operands(node, state)
    )
    if cpf_lowering.device() and on_device_at_host_level(node, state):
        vendor = VENDOR_IMPLEMENTATIONS_DEVICE[cpf_lowering.active_dialect()] if vendored else ()
        preferred = vendor + RENDERABLE_BY_NODE_DEVICE.get(type(node).__name__, ()) + preferred
    elif vendored and not is_devicelevel_gpu(state.sdfg, state, node):
        preferred = VENDOR_IMPLEMENTATIONS_HOST + preferred
    return preferred + RENDERABLE_IMPLEMENTATIONS


def target_tile_functions(sdfg: SDFG, isa: ISA) -> None:
    """Lower every host tile node through the tile functions of ONE ISA, in place.

    ``AUTO`` is the widest ISA with tile functions the host executes (AVX-512, AVX2, NEON), else the
    scalar ``omp simd`` lane loop. A node those functions cannot express lowers ``pure``; a node in a
    GPU kernel keeps its device lowering.

    :raises ValueError: if ``isa`` has no tile functions (SVE, CUDA).
    """
    resolved = widest_function_isa() if isa is ISA.AUTO else isa
    if resolved not in FUNCTION_ISAS:
        raise ValueError(f"CPF renders tile nodes for {[isa.name for isa in FUNCTION_ISAS]} or AUTO, not {isa.name}")
    for node, state in sdfg.all_nodes_recursive():
        if (
            isinstance(node, TileOp)
            and node.target_isa is not ISA.CUDA
            and not is_devicelevel_gpu(state.sdfg, state, node)
        ):
            node.target_isa = resolved
            node.implementation = isa_implementation(node, state, resolved)


def schedule_host_level_device_node(node: nodes.LibraryNode, state: SDFGState) -> None:
    """Make a host-level node over device memory expand into a KERNEL rather than a host map.

    A library node whose operands live in device memory and which sits at host level
    (:func:`on_device_at_host_level`) is left ``Sequential`` by
    :func:`~dace.transformation.passes.canonicalize.finalize.canonicalize_set_fast_implementations`,
    because it was pointed at a device LIBRARY CALL (cuBLAS/cuSolverDn/CUB) and a call needs no
    schedule of its own. CPF cannot render that call, so :func:`renderable_implementations` re-points
    the node at ``pure``/``Auto``, whose expansion is MAPS -- and those maps inherit ``node.schedule``.
    Left ``Sequential`` they are host code indexing ``GPU_Global`` operands, which validation rejects
    outright ("stored as StorageType.GPU_Global but accessed on host"): measured on the
    scientific_computing Gemm/Dot/Reduce/TensorDot nodes (lulesh, cholesky, minife, channel_flow,
    ls3df_scf), every one of which refused to render as ``hip``.

    Setting the schedule here is the same correction
    :func:`~dace.transformation.passes.canonicalize.finalize.canonicalize_set_fast_implementations`
    already makes for a node with NO device expansion; CPF reaches the same state by a different
    route -- a device expansion exists, but not one a standalone unit can compile.

    A node already ``GPU_Device``, a host-dialect rendering and a node inside a kernel are all left
    alone: only the host-level device-memory case is wrong, and only in a device dialect. So is a
    node whose implementation CPF did not choose -- its caller leaves that one's expansion alone. Any
    other device schedule at host level is corrected too: an offloaded node's expansion hands the
    nodes it produces the thread-block schedule of the scope they would be in (doitgen's matmul).
    """
    if not cpf_lowering.device() or node.schedule == dtypes.ScheduleType.GPU_Device:
        return
    if on_device_at_host_level(node, state):
        node.schedule = dtypes.ScheduleType.GPU_Device


def force_renderable_expansions(sdfg: SDFG, provenance: dict[str, tuple[str, str]] | None = None) -> None:
    """Expand every library node in ``sdfg`` through a renderable implementation, in place.

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

    A node with no renderable implementation at all is left alone: it may still expand to something
    renderable, and if it does not, the ``dace::`` symbol it emits is reported against its name by
    :func:`verify`, which says more than a refusal from here could.

    :param sdfg: the SDFG to expand. Call on a COPY.
    :param provenance: filled in with ``node GUID -> (origin GUID, description)`` for the code each
                       expansion produced, so the rendering can say what the loops used to be.
                       Descriptions come from :data:`LIBRARY_NODE_DESCRIPTIONS`.
    :raises NotImplementedError: if expansion has stalled (see :data:`MAX_EXPANSION_STALLED_ROUNDS`).
    """
    census: Counter | None = None
    stalled = 0
    while True:
        pending: dict[int, list] = {}
        for node, state in sdfg.all_nodes_recursive():
            if isinstance(node, nodes.LibraryNode):
                pending.setdefault(id(state), []).append((node, state))
        if not pending:
            return
        # A budget taken from the CURRENT population shrinks as the round number grows, so the
        # two cross part way through any long drain. Progress is what can actually be observed.
        current = Counter(type(node).__name__ for group in pending.values() for node, _ in group)
        stalled = stalled + 1 if current == census else 0
        census = current
        if stalled > MAX_EXPANSION_STALLED_ROUNDS:
            # Observations only: naming a cause sent readers after a cycle that did not exist.
            counted = ", ".join(f"{name} x{count}" for name, count in sorted(census.items()))
            raise NotImplementedError(
                f"CPF stopped expanding library nodes: {stalled} consecutive rounds left "
                f"the same nodes pending ({counted}). Expansion is making no progress; "
                f"check whether one of these expands into a node of its own type."
            )

        # One per state, by graph insertion order -- GUIDs are fresh uuid4() per node
        # (dace/sdfg/graph.py) so sorting by them reorders randomly run to run.
        chosen = []
        for group in pending.values():
            state = group[0][1]
            order = {id(node): index for index, node in enumerate(state.nodes())}
            chosen.append(min(group, key=lambda pair: order[id(pair[0])]))
        described: dict[int, tuple[nodes.LibraryNode, str, set]] = {}
        for node, state in chosen:
            available = type(node).implementations
            selected = next((c for c in renderable_implementations(node, state) if c in available), None)
            if selected is not None:
                node.implementation = selected
                # A device expansion (``CUDA``) is a host call that launches its own kernel. Every
                # other one, ``Reduce``'s device ``pure`` included, is maps that take the node's schedule.
                if selected != "CUDA":
                    schedule_host_level_device_node(node, state)
            if provenance is not None:
                description = description_of(node)
                if description is not None:
                    # Recorded BEFORE the expansion: afterwards the node is gone from the graph, and
                    # with it any way to ask what it was.
                    described[id(state)] = (node, description, {existing.guid for existing in state.nodes()})

        expanded = {id(node) for node, _ in chosen}
        sdfg.expand_library_nodes(recursive=False, predicate=lambda node: id(node) in expanded)

        for _, state in chosen:
            record = described.get(id(state))
            if record is None:
                continue
            node, description, before = record
            origin = node.guid
            # The specialization hint is read AFTER the expansion, which may have corrected it (a
            # sequential Scan expansion says so). Folded into the description so the emitter's
            # once-per-origin dedupe covers both: a Scan expands into several maps, one trade.
            if node.specialization_hint:
                description = f"{description}\n{node.specialization_hint}"
            for produced in state.nodes():
                if produced.guid in before:
                    continue
                for guid in subtree_guids(produced, state):
                    required(provenance).setdefault(guid, (origin, description))


#: The prefix DaCe gives a data container that carries a program's return value. A single return is
#: ``__return``; a returned tuple is ``__return_0``, ``__return_1``, ... (``parser.py`` builds both).
RETURN_PREFIX = "__return"


def is_return_name(name: str) -> bool:
    """Whether ``name`` is a return container's name."""
    return name == RETURN_PREFIX or name.startswith(RETURN_PREFIX + "_")


def return_containers(sdfg: SDFG) -> list[tuple[SDFG, str]]:
    """Every return container in ``sdfg``'s whole tree, as ``(owning SDFG, name)``.

    Both the DECLARATIONS and the ACCESS NODES are walked, because they answer different questions
    and CPF needs both. ``arrays_recursive`` finds a container that exists, including one a nested
    SDFG declared and never wired out; the access nodes are where a name is actually read or
    written, which is what lets a refusal name the state a reader can go and look at.

    Nested SDFGs are included on purpose. A nested ``__return`` is ordinary -- it is the nested
    SDFG's out-connector, and the value leaves through a memlet rather than through the entry
    signature -- but a nested one that is TRANSIENT leaves nowhere, and checking only the top level
    would not see it.

    :param sdfg: the outermost SDFG.
    :returns: ``(owner, name)`` pairs, deduplicated, in a deterministic order.
    """
    found: dict[tuple[int, str], tuple[SDFG, str]] = {}
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
        where = f"{sdfg.name}: the return container {name!r}"
        if owner is not sdfg:
            where = f"{sdfg.name}: the return container {name!r} of the nested SDFG {owner.name!r}"
        if desc.transient:
            raise NotImplementedError(
                f"CPF cannot render {where} is transient, so nothing outside the SDFG that "
                "declares it can read the value it holds."
            )
        if owner is not sdfg:
            continue
        if isinstance(desc, dt.Scalar):
            raise NotImplementedError(
                f"CPF cannot render {where} is a Scalar, which the entry signature passes BY "
                "VALUE, so the result would be computed into the callee's copy and discarded. "
                f"Promote {name!r} to a one-element array before rendering."
            )
        if not isinstance(desc, dt.Array):
            raise NotImplementedError(
                f"CPF cannot render {where} is a {type(desc).__name__}, which has no "
                "plain-pointer spelling in the entry signature."
            )


def refuse_runtime_scopes(sdfg: SDFG) -> None:
    """Refuse the constructs whose only implementation is a DaCe runtime class.

    A ``Stream`` descriptor is emitted as ``dace::Stream<T>`` and a consume scope drives it through
    ``dace::Consume``: both are runtime templates carrying a lock-free queue, and neither has a
    standalone spelling that CPF could inline. Without this check a stream still fails, but as the
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
                raise NotImplementedError(
                    f"CPF cannot render {state.label}/{node.label}: a consume scope is driven "
                    "by the runtime class dace::Consume, whose queue and quiescence detection "
                    "CPF does not provide. Express the work as a map before rendering."
                )
    for subsdfg, name, desc in sdfg.arrays_recursive():
        if isinstance(desc, dt.Stream):
            raise NotImplementedError(
                f"CPF cannot render {subsdfg.name}.{name}: a Stream is the runtime class "
                "dace::Stream, a lock-free queue with no standalone spelling. Rewrite the "
                "producer and consumer around an array before rendering."
            )


def prepare(sdfg: SDFG, provenance: dict[str, tuple[str, str]] | None = None) -> None:
    """Make ``sdfg`` renderable as one host translation unit, in place.

    Six things happen: every library node is pointed at the best implementation a standalone unit
    can compile and expanded (:func:`force_renderable_expansions`), every accumulating copy becomes a
    tasklet or a map (:func:`~dace.transformation.passes.canonicalize.finalize.explicit_accumulating_copies`), every written signature scalar,
    the expansions' included, is promoted to a length-1 array so it is addressable
    (:class:`~dace.transformation.passes.scalar_promotion.PromoteScalarOutputsToArrays`), what that
    could not make renderable is refused (:func:`refuse_by_value_returns`), every loop still
    unlabelled is given its kind (``AnnotateLoopKinds``), and lifetimes that would need a state struct are
    demoted (see :data:`LIFETIME_DEMOTIONS`). Anything CPF cannot express raises here rather than
    at compile time, where the message would be a C++ diagnostic about a name this module chose.

    The promotion runs in BOTH dialects, not only C. A by-value scalar out-parameter discards the
    result just as silently in C++; the C rendering merely also fails to compile, because a written
    scalar connector on a nested SDFG binds as ``T &`` there.

    :param sdfg: the SDFG to prepare. Call on a COPY -- :func:`cpf` does.
    :param provenance: filled in with the library-node descriptions the rendering will comment
                       with (see :func:`force_renderable_expansions`).
    :raises NotImplementedError: if the SDFG needs a device compiler, holds a stream or a consume
                                 scope (:func:`refuse_runtime_scopes`), or carries a return container
                                 the entry signature cannot pass back (:func:`refuse_by_value_returns`).
    """
    cpf_lowering.refuse_unspellable_containers(sdfg)
    device = uses_device_code(sdfg)
    if device:
        raise NotImplementedError(
            "CPF renders one translation unit, but "
            + "; ".join(device)
            + ". Render the CPU form of this SDFG, or the 'cuda' or 'hip' language for a device one."
        )
    refuse_runtime_scopes(sdfg)
    force_renderable_expansions(sdfg, provenance)
    # After the expansions, which make accumulating copies of their own.
    explicit_accumulating_copies(sdfg)
    # After the expansions: a Dot's or a pure Reduce's nested function writes its result through a
    # scalar connector the promotion has to see (gramschmidt, durbin, trmm, addusxx_g).
    PromoteScalarOutputsToArrays().apply_pass(sdfg, {})
    refuse_by_value_returns(sdfg)
    # Canonicalize labelled the loops it left; specialization and the expansions above made more.
    AnnotateLoopKinds().apply_pass(sdfg, {})
    # Before the demotion: a buffer every thread of a team shares must reach the entry first.
    lift_team_shared_transients(sdfg)
    for _, _, desc in sdfg.arrays_recursive():
        demoted = LIFETIME_DEMOTIONS.get(desc.lifetime)
        if demoted is not None:
            desc.lifetime = demoted


def team_called(node: nodes.NestedSDFG, state: SDFGState) -> bool:
    """Whether every thread of a persistent team calls ``node``'s function: a ``CPU_Persistent`` map scope (the
    ``#pragma omp parallel`` a hoisted loop runs in) encloses it, in its own state or around the nested SDFG that
    holds it.

    :param node: the nested SDFG node.
    :param state: the state holding ``node``.
    :returns: ``True`` if a team calls it.
    """
    scope = state.entry_node(node)
    while scope is not None:
        if isinstance(scope, nodes.MapEntry) and scope.map.schedule == dtypes.ScheduleType.CPU_Persistent:
            return True
        scope = state.entry_node(scope)
    outer = state.sdfg
    caller = (
        next(
            (
                (candidate, holder)
                for candidate, holder in outer.parent_sdfg.all_nodes_recursive()
                if isinstance(candidate, nodes.NestedSDFG) and candidate.sdfg is outer
            ),
            None,
        )
        if outer.parent_sdfg is not None
        else None
    )
    return caller is not None and team_called(*caller)


def lift_team_shared_transients(sdfg: SDFG) -> int:
    """Move every ``Persistent`` and ``Global`` transient of a nested SDFG a team calls into the SDFG that calls
    it, wired through the nested node's connectors, innermost first so a buffer climbs to the entry.

    The ordinary generators allocate such a buffer ONCE, in the state struct, shared by every thread of the
    team. :data:`LIFETIME_DEMOTIONS` turns it into an allocation on entry of the function that declares it,
    which is right for the entry and wrong for a nested function every thread of an ``omp parallel`` region
    calls: each thread then allocated a private copy, filled only its own ``omp for`` share of it, and read
    the rest uninitialized (bfs's ``frontier``). A buffer whose extent names a symbol the enclosing scopes
    define per iteration stays where it is.

    :param sdfg: the SDFG to rewrite in place.
    :returns: how many buffers were lifted, counting each level of a climb.
    """
    lifted = 0
    # Walked, not read off ``parent_nsdfg_node``: an outlining re-homes nested SDFGs without refreshing every
    # back pointer. Pre-order lists a caller before its callees; reversed, a callee is lifted first.
    calls = [(node, state) for node, state in sdfg.all_nodes_recursive() if isinstance(node, nodes.NestedSDFG)]
    for node, state in reversed(calls):
        inner = node.sdfg
        if inner is None or not team_called(node, state):
            continue
        outer = state.sdfg
        scopes = []
        scope = state.entry_node(node)
        while scope is not None:
            scopes.append(scope)
            scope = state.entry_node(scope)
        per_iteration = {param for entry in scopes for param in entry.map.params}
        symrepl = {
            symbolic.symbol(name): symbolic.pystr_to_symbolic(value)
            for name, value in node.symbol_mapping.items()
            if name != str(value)
        }
        for name, desc in list(inner.arrays.items()):
            if (
                not desc.transient
                or desc.lifetime not in LIFETIME_DEMOTIONS
                or type(desc) is not dt.Array
                or name in node.in_connectors
                or name in node.out_connectors
            ):
                continue
            outer_desc = copy.deepcopy(desc)
            if symrepl:
                replace_properties_dict(outer_desc, {}, symrepl)
            if {str(s) for s in outer_desc.free_symbols} & per_iteration:
                continue
            accesses = [(st, n) for st in inner.states() for n in st.data_nodes() if n.data == name]
            read = any(st.out_degree(n) > 0 for st, n in accesses)
            written = any(st.in_degree(n) > 0 for st, n in accesses)
            outer_name = outer.add_datadesc(name, outer_desc, find_new_name=True)
            desc.transient = False
            memlet = Memlet.from_array(outer_name, outer_desc)
            if read:
                node.add_in_connector(name)
                state.add_memlet_path(
                    state.add_read(outer_name), *reversed(scopes), node, dst_conn=name, memlet=copy.deepcopy(memlet)
                )
            if written:
                node.add_out_connector(name, force=True)
                exits = [state.exit_node(entry) for entry in scopes]
                state.add_memlet_path(
                    node, *exits, state.add_write(outer_name), src_conn=name, memlet=copy.deepcopy(memlet)
                )
            lifted += 1
    return lifted


#: Source languages a GPU target emits its own translation unit in. CUDA emits ``cu``; HIP emits
#: plain ``cpp``, because hipcc compiles ``.cpp`` -- the SAME language the frame carries. So the
#: language alone cannot tell the two apart there, and :data:`DEVICE_TARGET_TYPES` is what does.
DEVICE_LANGUAGES = ("cu", "hip", "hip.cpp")

#: Build subdirectories a GPU target emits into. MEASURED on gfx942: generate_code returns the
#: frame at ``target_type=''`` and the device object at ``target_type='hip'``, both carrying
#: ``language='cpp'``.
DEVICE_TARGET_TYPES = ("cuda", "hip")


def is_device_object(obj: CodeObject) -> bool:
    """Whether ``obj`` is the unit a GPU target emitted, rather than the frame that calls it.

    Either label is enough, and neither is enough alone: a HIP device object shares the frame's
    ``language``, while a target that leaves ``target_type`` at its default is separated only by
    the language. Answering to EITHER covers both backends; the frame answers to neither.
    """
    return obj.target_type in DEVICE_TARGET_TYPES or obj.language in DEVICE_LANGUAGES


def split_units(objects: list[CodeObject], name: str) -> tuple[CodeObject, CodeObject | None]:
    """The units CPF renders, out of what code generation produced: the frame, and under a device dialect
    the one device object.

    The frame holds the entry and, on a device dialect, calls ``__cpf_runkernel_*``; the device object
    defines those launchers and the kernels. Any other split -- a device object under a host dialect,
    a second device object, a separate unit per nest under
    ``codegen_params.split_nsdfg_translation_units`` -- means a unit CPF does not render would hold
    part of the computation. Non-linkable objects (the call header, the sample ``main``) are
    generated for every SDFG and are not part of the build, so they do not count.

    :param objects: what :func:`dace.codegen.codegen.generate_code` returned.
    :param name: the SDFG's name, for the message.
    :returns: the frame object, and the device object (``None`` under a host dialect).
    :raises NotImplementedError: if the code was split any other way.
    """
    linkable = [obj for obj in objects if obj.linkable]
    # Neither label separates the two on every backend, so is_device_object takes either: hipcc
    # compiles .cpp, so both HIP objects come back ``language='cpp'``.
    frame = [obj for obj in linkable if not is_device_object(obj)]
    device = [obj for obj in linkable if is_device_object(obj)]
    if len(frame) == 1 and len(device) <= (1 if cpf_lowering.device() else 0):
        return frame[0], (device[0] if device else None)
    extra = ", ".join(f"{obj.name}.{obj.language}" for obj in linkable)
    raise NotImplementedError(
        f"CPF renders the frame and at most one device unit, but {name} generated {len(linkable)}: "
        f"{extra}. Turn off the split-translation-unit codegen parameters."
    )


def state_struct(frame: str, name: str) -> str:
    """The frame's ``<name>_state_t`` definition, which the device unit's launchers take a pointer to.

    The frame defines it (it carries the stream context); the device object only names it. Repeating
    the identical definition in the device unit is what separate compilation needs, and the one
    definition rule allows. Empty when the frame defines no state.
    """
    match = re.search(rf"^struct {re.escape(name)}_state_t \{{.*?^\}};\n", frame, re.MULTILINE | re.DOTALL)
    return match.group(0) + "\n" if match else ""


def written_containers(sdfg: SDFG) -> OrderedSet[str]:
    """The container names some state WRITES, at any nesting depth.

    An ``AccessNode`` with a data-carrying incoming edge is a write; an empty-memlet edge only orders
    the node after its source (an inlined callee leaves one into every later read). Nested SDFGs are walked too: a nested
    transient that happens to share an outer name is then reported as written, which is the SAFE
    direction -- it only ever withholds a ``const``, never grants one wrongly.

    :param sdfg: the SDFG to scan.
    :returns: the written names.
    """
    written: OrderedSet[str] = OrderedSet()
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, nodes.AccessNode) and any(not edge.data.is_empty() for edge in state.in_edges(node)):
                written.add(node.data)
            elif isinstance(node, nodes.NestedSDFG):
                written |= written_containers(node.sdfg)
    return written


def readonly_entry_arrays(sdfg: SDFG, arglist: dict[str, dt.Data] | None = None) -> OrderedSet[str]:
    """The entry point's ARRAY parameters that nothing writes -- the ones whose pointee is const.

    Read off the same SDFG the signature is generated from, so the qualifier and the argument list
    cannot disagree. Callers that publish a binding for the rendering (the ABI ``const`` flag)
    should derive it from HERE rather than recomputing it, which is what let a binding say
    ``const: true`` while the rendered signature said otherwise.

    :param sdfg: the PREPARED SDFG -- the one whose ``arglist()`` is the signature.
    :param arglist: ``sdfg.arglist()`` when the caller already holds it.
    :returns: the names to qualify.
    """
    written = written_containers(sdfg)
    entry_arguments = sdfg.arglist() if arglist is None else arglist
    return OrderedSet(
        name for name, desc in entry_arguments.items() if isinstance(desc, dt.Array) and name not in written
    )


def entry_parameter_name(param: str) -> str:
    """The declared name in one entry-signature parameter (``float * __restrict__ a`` -> ``a``)."""
    return param.strip().split()[-1].lstrip("*")


def qualify_readonly_pointers(
    code: str,
    sdfg: SDFG,
    entry: str,
    arglist: dict[str, dt.Data] | None = None,
    writable: Sequence[str] = (),
) -> str:
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
    :param arglist: ``sdfg.arglist()`` when the caller already holds it.
    :param writable: parameters the caller's own declaration spells writable, kept so even when nothing writes them.
    :returns: the unit with the read-only parameters qualified.
    """
    readonly = readonly_entry_arrays(sdfg, arglist) - OrderedSet(writable)
    if not readonly:
        return code
    return rewrite_entry_parameters(
        code, entry, lambda params: [f"const {p}" if entry_parameter_name(p) in readonly else p for p in params]
    )


def rewrite_entry_parameters(code: str, entry: str, rewrite: Callable[[list[str]], list[str]]) -> str:
    """Apply ``rewrite`` to the entry point's parameter list wherever the unit declares it.

    One splitter for every signature rewrite, so the qualifier pass and the ordering pass can never
    disagree about where a parameter begins or which declarations they reach.

    The list is split on commas and terminated at the first ``)``, which is exact for pointers and
    by-value scalars and wrong for anything nested (a function-pointer parameter). CPF emits
    neither today; a nested list is refused rather than mangled.

    :param code: the rendered unit.
    :param entry: the entry point's name.
    :param rewrite: takes the stripped parameter declarations in order, returns the new ones.
    :returns: the unit with every declaration of ``entry`` rewritten.
    :raises NotImplementedError: if a parameter carries a nested parameter list.
    """
    pattern = re.compile(r"\bvoid\s+%s\s*\(" % re.escape(entry))
    out, cursor = [], 0
    for match in pattern.finditer(code):
        opened = match.end() - 1
        closed = code.index(")", opened)
        if "(" in code[opened + 1 : closed]:
            raise NotImplementedError(
                f"CPF cannot rewrite the entry signature of {entry}: a parameter carries a "
                "nested parameter list, which this rewrite cannot split."
            )
        params = [p.strip() for p in code[opened + 1 : closed].split(",")]
        out.append(code[cursor : opened + 1] + ", ".join(rewrite(params)))
        cursor = closed
    out.append(code[cursor:])
    return "".join(out)


def reorder_entry_parameters(code: str, entry: str, order: Sequence[str]) -> str:
    """Rewrite the entry point's parameter list into ``order``, matching declarations by name.

    CPF's own order is ``SDFG.arglist()``: every array sorted by name, then every scalar sorted by
    name. A caller whose calling convention is fixed elsewhere needs the same body under a
    different parameter order -- an ABI that reserves a trailing scratch pair puts a pointer behind
    the scalars, which no name sort reaches. Nothing in the body depends on the order, and the unit
    is self-contained (no prototype, no header), so the entry is declared here and nowhere else.

    Only the ORDER moves. Each declaration keeps the type and the qualifiers CPF gave it, including
    the ``const`` :func:`qualify_readonly_pointers` added: C linkage ignores qualifiers, so
    re-spelling them to match a caller's own declaration would change nothing a compiler can see.

    ``order`` must name each parameter the entry takes exactly once. A disagreement is refused
    rather than resolved by dropping, duplicating or inventing one: the result would link and be called with its
    arguments shifted, which no compiler catches across a rename.

    :param code: the rendered unit.
    :param entry: the entry point's name.
    :param order: the parameter names, in the order the caller will pass them.
    :returns: the unit with every declaration of ``entry`` reordered.
    :raises ValueError: if ``order`` does not name each of the entry's parameters exactly once.
    """
    wanted = list(order)

    def to_order(params: list[str]) -> list[str]:
        by_name = {entry_parameter_name(p): p for p in params}
        if sorted(by_name) != sorted(wanted):
            raise ValueError(
                f"CPF cannot render {entry} in the requested order: the entry takes "
                f"{sorted(by_name)} but the order names {sorted(wanted)}."
            )
        return [by_name[name] for name in wanted]

    return rewrite_entry_parameters(code, entry, to_order)


@dataclasses.dataclass(frozen=True)
class EntrySignature:
    """The parameter list a caller requires of the entry point, when CPF's own will not do.

    ``order`` names every parameter exactly once, in the order the caller passes them; it is
    applied by :func:`reorder_entry_parameters`, so each parameter keeps CPF's type and qualifiers.
    ``workspace`` / ``workspace_size`` add a caller-owned scratch buffer: a non-transient ``uint8_t``
    array of ``workspace_size`` (``int64_t``) bytes, which ``order`` must name too. It exists only so the
    entry matches the caller's prototype: the body never touches it, and it stays a writable
    ``uint8_t *`` as the caller declares it.
    """

    order: Sequence[str]
    workspace: str | None = None
    workspace_size: str | None = None

    def __post_init__(self) -> None:
        if (self.workspace is None) != (self.workspace_size is None):
            raise ValueError("EntrySignature takes workspace and workspace_size together or neither")


def add_workspace(sdfg: SDFG, workspace: str, size: str) -> None:
    """Add the caller-owned scratch pair to ``sdfg``'s entry, device-resident when any argument is.

    :raises ValueError: if either name is already one of ``sdfg``'s data or symbols.
    """
    taken = [name for name in (workspace, size) if name in sdfg.arrays or name in sdfg.symbols]
    if taken:
        raise ValueError(f"CPF cannot add the workspace pair to {sdfg.name}: {taken} already name something in it")
    on_device = any(
        not desc.transient and desc.storage is dtypes.StorageType.GPU_Global for desc in sdfg.arrays.values()
    )
    sdfg.add_symbol(size, dtypes.int64)
    sdfg.add_array(
        workspace,
        [symbolic.symbol(size, dtypes.int64)],
        dtypes.uint8,
        storage=dtypes.StorageType.GPU_Global if on_device else dtypes.StorageType.Default,
    )


#: ``language`` argument -> the dialect that renders it. ``'c++'`` is the default and stays the
#: historical behaviour exactly.
LANGUAGES: dict[str, cpf_lowering.Dialect] = {
    "c++": cpf_lowering.Dialect.STANDALONE,
    "c": cpf_lowering.Dialect.STANDALONE_C,
    "hip": cpf_lowering.Dialect.STANDALONE_HIP,
    "cuda": cpf_lowering.Dialect.STANDALONE_CUDA,
}


def dialect_for(language: str) -> cpf_lowering.Dialect:
    """The dialect ``language`` names.

    :param language: a key of :data:`LANGUAGES`.
    :raises ValueError: for any other value, naming what is available.
    """
    try:
        return LANGUAGES[language]
    except KeyError:
        raise ValueError(f"CPF renders {sorted(LANGUAGES)}, not {language!r}") from None


def preamble(
    code: str, dialect: cpf_lowering.Dialect = cpf_lowering.Dialect.STANDALONE, environments: Iterable[str] = ()
) -> str:
    """The include block and inline definitions ``code`` needs, in the order they must appear.

    Derived from the finished text (:func:`~dace.cpf_lowering.helpers_used`) rather than from the
    SDFG: helpers arrive from two printers -- symbolic expressions and tasklet bodies -- and only
    the emitted unit has seen both.

    :param code: the emitted translation unit, without its preamble.
    :param dialect: which standalone dialect emitted it.
    :param environments: the DaCe environments the unit's code generator used; a vendor library among them
                         (:data:`~dace.cpf_lowering.VENDOR_LIBRARIES`) adds its headers and definitions.
    :returns: the preamble, ending in a blank line.
    """
    used = cpf_lowering.helpers_used(code, dialect)
    definitions = cpf_lowering.definitions_for(used, dialect)
    headers = cpf_lowering.headers_for(used, dialect)
    vendors = cpf_lowering.vendor_libraries(environments, code)
    lines = ["// Rendered by DaCe CPF (canonical parallel form): self-contained, no DaCe runtime."] + CONTRACT_LINES
    lines += [f"#include {header}" for header in headers]
    vendor_headers = [
        header
        for vendor in vendors
        for header in vendor.headers + (vendor.cxx_headers if dialect is not cpf_lowering.Dialect.STANDALONE_C else ())
    ]
    seen_wrappers: set[str] = set()
    wrappers: list[str] = []
    for vendor in vendors:
        # The wrappers are C++; a C unit's expansions call the vendor directly.
        if vendor.wrapper and dialect is not cpf_lowering.Dialect.STANDALONE_C:
            wrapper_headers, wrapper = cpf_lowering.vendor_wrapper(vendor.wrapper, seen_wrappers)
            vendor_headers += wrapper_headers
            wrappers.append(wrapper)
    lines += [f"#include {header}" for header in dict.fromkeys(vendor_headers) if header not in headers]
    if dialect is cpf_lowering.Dialect.STANDALONE_C:
        lines.append(cpf_lowering.C_UNDEF_LINE)
    if dialect in cpf_lowering.DEVICE_DIALECTS:
        lines.append("")
        lines.append("// What dace/dace.h would define for the device side.")
        # code PLUS the definitions: a scan expansion's inline helper is emitted BELOW this block
        # and calls ``gpucub::DeviceScan`` itself, so gating on ``code`` alone dropped the namespace
        # alias out from under it and the unit failed to compile on an undeclared ``gpucub``.
        lines.append(cpf_lowering.device_preamble(code + "\n".join(definitions), dialect))
    if definitions:
        lines.append("")
        lines.append("// Functions the DaCe runtime headers would otherwise provide.")
        lines.extend(definitions)
    for vendor in vendors:
        if vendor.definitions:
            lines.append("")
            lines.append(vendor.definitions.rstrip("\n"))
    for wrapper in filter(None, wrappers):
        lines.append("")
        lines.append("// DaCe's wrapper of the vendor library, which the library calls below go through.")
        lines.append(wrapper)
    lines.append("")
    return "\n".join(lines)


#: What the rendering already did and what is left, ahead of the per-loop verdicts it explains.
CONTRACT_LINES = [
    "// Already parallelized, with basic heuristics applied. Every loop names its class first:",
    "//   parallel, sequential -- settled; their parallelism needs no further reasoning.",
    "//   unsure               -- open; the only loops whose parallelism is worth reasoning about.",
    "// Spend the effort on heuristic optimizations and restructuring.",
]

#: What a finished rendering must not contain, and what each one means. CPF's own gate, checked on
#: the emitted text before it is handed back: every one of these is a construct that BUILDS inside
#: the DaCe tree (where the runtime headers are on the include path) and fails only once the output
#: is used the way CPF promises it can be. Failing here names the SDFG construct that caused it,
#: which a link error against ``libdace`` never would.
#:
#: The test harness (``tests/codegen/cpf/conftest.py``) states the same contract independently and
#: on purpose -- it is the acceptance spec, written from outside, and it also compiles the result
#: with no include path at all, which is the only check that cannot be fooled by a table that
#: forgot an entry.
BANNED: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r'#\s*include\s*[<"][^>"]*dace/'), "a DaCe runtime header"),
    (re.compile(r'#\s*include\s*"'), "a quoted (build-tree-relative) include"),
    (re.compile(r"CopyND"), "a dace::CopyND copy -- insert explicit copies before rendering"),
    (re.compile(r"__dace_(init|exit)\w*"), "a DaCe init/exit entry point"),
    (re.compile(r"\bdace\s*::"), "a DaCe runtime symbol"),
    (re.compile(r"\bDACE_[A-Z]"), "a DaCe preprocessor macro"),
    (re.compile(r"^[ \t]*#[ \t]*define\b", re.MULTILINE), "a preprocessor macro definition"),
    (re.compile(r"__state\b"), "a state-struct dereference"),
)

#: The scalar type spellings a code generator writes a declarator with. Used to anchor the
#: reference-parameter pattern below: a bare ``\w+\s*&\s*\w+`` would also match the bitwise
#: ``exponent & 1)`` in CPF's own ``ipow``, and a gate with a false positive gets disabled.
_C_DECLARED_TYPES = (
    r"(?:const\s+)?(?:unsigned\s+|signed\s+)?"
    r"(?:long\s+double|long\s+long|u?int(?:8|16|32|64)_t|double|float|bool|char|short|int|long)"
    r"(?:\s+_Complex)?"
)

#: What a finished C rendering must not contain, on top of :data:`BANNED`. Every one of these is
#: valid C++ that the C++ dialect emits on purpose, so a leak is a dialect branch that was missed
#: rather than a construct that should never exist.
#: What a finished DEVICE rendering must not contain. The state-struct entry is dropped from
#: :data:`BANNED` because the unit DECLARES that struct, which carries the stream, rather than
#: borrowing it (:func:`~dace.cpf_lowering.device_entry_prologue`). Every ``DACE_*`` spelling is
#: written out by :func:`~dace.cpf_lowering.device_spell_out`, so one surviving is a leak here too.
BANNED_DEVICE: tuple[tuple[re.Pattern, str], ...] = tuple(
    entry for entry in BANNED if entry[1] != "a state-struct dereference"
)

BANNED_C: tuple[tuple[re.Pattern, str], ...] = BANNED + (
    (re.compile(r"\bstd\s*::"), "a C++ standard-library symbol"),
    (re.compile(r"\btemplate\s*<"), "a C++ template"),
    (re.compile(r'extern\s*"C"'), "a C++ language linkage specifier"),
    (re.compile(r"\bstatic_cast\s*<"), "a C++ static_cast"),
    (re.compile(r"\bnew\s"), "a C++ new-expression"),
    (re.compile(r"\bdelete\b"), "a C++ delete-expression"),
    # ``constexpr`` on an OBJECT is C23 and is how CPF emits an SDFG constant, so only the FUNCTION
    # form is banned: a qualifier run ending in a declarator with a parameter list.
    (re.compile(r"\b(?:constexpr|consteval)\b[^;=\n]*\b\w+\s*\([^;]*\)\s*\{"), "a constexpr/consteval function"),
    (re.compile(_C_DECLARED_TYPES + r"\s*&\s*\w+\s*[,)]"), "a C++ reference parameter"),
)


def verify(code: str, name: str, dialect: cpf_lowering.Dialect = cpf_lowering.Dialect.STANDALONE) -> None:
    """Assert ``code`` is self-contained, or raise naming what leaked.

    :param code: the rendered translation unit.
    :param name: the SDFG's name, for the message.
    :param dialect: which standalone dialect rendered it, choosing the table to check against.
    :raises RuntimeError: on the first banned construct found.
    """
    if dialect is cpf_lowering.Dialect.STANDALONE_C:
        banned = BANNED_C
    elif dialect in cpf_lowering.DEVICE_DIALECTS:
        banned = BANNED_DEVICE
    else:
        banned = BANNED
    for pattern, meaning in banned:
        match = pattern.search(code)
        if match is None:
            continue
        start = code.rfind("\n", 0, match.start()) + 1
        end = code.find("\n", match.end())
        line = code[start : end if end != -1 else len(code)].strip()
        raise RuntimeError(
            f"CPF rendered {name} with {meaning} ({match.group(0)!r}), so the result is not self-contained:\n    {line}"
        )


#: Per HOST dialect: the language standard its output is defined against, the source suffix, the
#: environment variable naming its compiler, and the compilers to try when that is unset.
#:
#: The DEVICE dialects are deliberately absent. Their compiler is not the host compiler and is not
#: installed on every box a host render runs on, so gating them here would turn "hipcc is not
#: installed" into "this SDFG cannot be rendered" -- and a gate that quietly skips instead would be
#: the very shape this one exists to close.
COMPILE_CHECK_TOOLCHAINS: dict[cpf_lowering.Dialect, tuple[str, str, str, tuple[str, ...]]] = {
    cpf_lowering.Dialect.STANDALONE: ("c++20", ".cpp", "CXX", ("g++", "c++")),
    cpf_lowering.Dialect.STANDALONE_C: ("c23", ".c", "CC", ("gcc", "cc")),
}

#: What a ``'cuda'`` unit is built with besides the output flags. ``--expt-relaxed-constexpr`` is what
#: DaCe's own CUDA build adds (``dace/codegen/CMakeLists.txt``): kernels call the ``constexpr`` index
#: and runtime helpers, which nvcc otherwise treats as host-only. hipcc needs no such flag.
CUDA_BUILD_FLAGS: tuple[str, ...] = ("-std=c++20", "--expt-relaxed-constexpr")

#: What the gate compiles with. ``-fsyntax-only`` because the question is whether the TEXT is a
#: valid translation unit, not how fast its object code is: it parses, resolves every name and
#: type-checks, and skips optimization and object emission, which is where the time goes. NO
#: ``-I``: a header the unit names must be a system header, which is half of what self-contained
#: means. ``-fopenmp`` because the parallel form is OpenMP and its pragmas must parse.
COMPILE_CHECK_FLAGS: tuple[str, ...] = ("-fsyntax-only", "-fopenmp")


def compile_check_compiler(dialect: cpf_lowering.Dialect) -> str | None:
    """The compiler the gate builds ``dialect``'s output with, or ``None`` if none is installed.

    :param dialect: which standalone dialect rendered the unit.
    :returns: an executable path, or ``None``.
    """
    _standard, _suffix, variable, fallbacks = COMPILE_CHECK_TOOLCHAINS[dialect]
    candidate = os.environ.get(variable)
    if not candidate and dialect is cpf_lowering.Dialect.STANDALONE:
        candidate = Config.get("compiler", "cpu", "executable")
    resolved = shutil.which(candidate) if candidate else None
    for fallback in fallbacks:
        if resolved is not None:
            break
        resolved = shutil.which(fallback)
    return resolved


def vendor_include_flags(environments: Iterable[str]) -> tuple[str, ...]:
    """``-I`` flags for the headers of the vendor libraries among ``environments``, where the environment finds
    them off the default search path (a spack or conda OpenBLAS, FFTW3). Only for :func:`compile_check`: the
    rendered unit names system headers, and the machine that builds it provides the library."""
    flags: list[str] = []
    for env in cpf_lowering.vendor_environments(environments):
        includes = getattr(env, "cmake_includes", None)
        flags += [f"-I{path}" for path in (includes() if callable(includes) else includes or ())]
    return tuple(dict.fromkeys(flags))


def compile_check(code: str, name: str, dialect: cpf_lowering.Dialect, extra_flags: Sequence[str] = ()) -> None:
    """Assert the rendered unit is one a bare host compiler accepts, or raise with its first error.

    :func:`verify` answers "does this text name anything CPF may not name", which is a different
    question from "is this text a translation unit". A form that passes the first and fails the
    second is the worst outcome CPF has: it is written to disk, served as the canonical parallel
    form, and read as good by whoever gets it -- and nothing between here and the compiler that
    never runs says otherwise. So the render does not return text it has not compiled.

    A no-op for the device dialect, which has no entry in :data:`COMPILE_CHECK_TOOLCHAINS`.

    :param code: the rendered translation unit.
    :param name: the SDFG's name, for the message.
    :param dialect: which standalone dialect rendered it, choosing the compiler and the standard.
    :param extra_flags: further compiler flags, the include paths of the vendor libraries it calls.
    :raises RuntimeError: if the unit does not compile, or if its compiler is not installed --
                          "could not be checked" must not read the same as "was checked".
    """
    toolchain = COMPILE_CHECK_TOOLCHAINS.get(dialect)
    if toolchain is None:
        return
    standard, suffix, variable, fallbacks = toolchain
    compiler = compile_check_compiler(dialect)
    if compiler is None:
        raise RuntimeError(
            f"CPF cannot check that {name} compiles: no {suffix[1:]} compiler found (tried "
            f"${variable} and {fallbacks}). A form that was never compiled must not be served as "
            "one that was, so the render refuses rather than hand back unchecked text."
        )
    workdir = tempfile.mkdtemp(prefix=f"cpf_check_{name}_")
    try:
        source = os.path.join(workdir, name + suffix)
        with open(source, "w") as handle:
            handle.write(code)
        command = [compiler, "-std=" + standard, *COMPILE_CHECK_FLAGS, *extra_flags, source]
        proc = subprocess.run(command, cwd=workdir, capture_output=True, text=True)
        if proc.returncode == 0:
            return
        first = next((line for line in proc.stderr.splitlines() if " error" in line), proc.stderr.strip())
        raise RuntimeError(
            f"CPF rendered {name} as text that does not compile, so the form would have been "
            f"served as good and found broken by whoever built it:\n    {first.strip()}"
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def device_backend(dialect: cpf_lowering.Dialect) -> contextlib.AbstractContextManager[None]:
    """Generate for ``dialect``'s own GPU backend (:data:`~dace.cpf_lowering.DEVICE_BACKENDS`), not the machine's.

    :param dialect: the dialect being rendered.
    :returns: a context pinning ``compiler.cuda.backend`` for a device dialect; a no-op for a host one.
    """
    backend = cpf_lowering.DEVICE_BACKENDS.get(dialect)
    if backend is None:
        return contextlib.nullcontext()
    return set_temporary("compiler", "cuda", "backend", value=backend)


class Rendering(NamedTuple):
    """A rendered SDFG: its source, and the SDFG that source was generated from.

    The second field is not a convenience. CPF renders a PREPARED COPY -- library nodes expanded
    through renderable implementations, lifetimes demoted -- and preparation can change the
    ARGUMENT LIST: expanding a ``Reduce`` into a map introduces the extent symbol the library node
    had kept to itself, so the entry point takes an argument the original SDFG's ``arglist()``
    never mentions. Calling the rendered code means calling it with THIS SDFG's arglist; the
    original's would silently drop that symbol and run the kernel on an uninitialized extent.

    ``arguments`` is the second half of that: it says what ORDER the rendered signature takes those
    parameters in, which is the arglist's order unless the caller supplied an :class:`EntrySignature`.
    """

    #: The self-contained translation unit; under a device dialect the HOST unit (``<name>.cpp``):
    #: the ``extern "C"`` entry, which only launches.
    code: str
    #: The prepared copy that was rendered. Its ``arglist()`` names the entry point's parameters
    #: and gives their types; ``arguments`` is the ORDER the rendered text declares them in.
    sdfg: SDFG
    #: The entry point's parameter names, in the order the rendered signature takes them. Equal to
    #: ``tuple(sdfg.arglist())`` unless the caller asked for an order of its own, which is exactly
    #: when a consumer that read the arglist instead would call with its arguments shifted.
    arguments: tuple[str, ...]
    #: Under a device dialect, the DEVICE unit (``<name>.hip`` / ``<name>.cu``): the ``__global__``
    #: kernels and the launchers ``code`` calls. Empty under a host dialect.
    device_code: str = ""
    #: What the caller links the units against, in order: the libraries of every environment that only links
    #: (an ``ExternCall``'s archive and its link flags) and of every vendor library the units call
    #: (:data:`~dace.cpf_lowering.VENDOR_LIBRARIES`). Empty when the units need nothing beyond the system's.
    link: tuple[str, ...] = ()
    #: The vendor libraries the units call, by family name (``blas``, ``lapack``, ``fftw``, ``cublas``,
    #: ``cusolver``, ``cufft``, ``rocblas``, ``rocsolver``, ``hipfft``): what a caller that loads libraries by
    #: name has to load for :attr:`link` to resolve (:func:`~dace.cpf_lowering.vendor_families`).
    libraries: tuple[str, ...] = ()


def render(
    sdfg: SDFG,
    validate: bool = True,
    language: str = "c++",
    signature: EntrySignature | None = None,
    check_compiles: bool = True,
    isa: ISA = ISA.AUTO,
) -> Rendering:
    """Render ``sdfg`` and return the text together with the SDFG it describes.

    :param sdfg: the SDFG to render. Not modified -- a copy is prepared and rendered.
    :param validate: validate the SDFG during code generation.
    :param language: ``'c++'`` (the default, C++20) or ``'c'`` (C23), which build with a bare host
                     compiler, no ``-I``, no libdace, no BLAS; or a device dialect, ``'cuda'`` (nvcc) or
                     ``'hip'`` (hipcc), which renders a host unit and a device unit and adds only the
                     toolkit's own headers. A device entry has the ``'c++'`` prototype, takes device
                     pointers for ``GPU_Global`` arrays (the caller copies to and from the device) and
                     synchronizes the device before it returns.
    :param signature: the entry point's parameter order, and optionally a workspace pair, for a
                      caller whose calling convention is fixed elsewhere (:class:`EntrySignature`).
                      ``None`` keeps CPF's own order (``arglist()``: arrays by name, then scalars by
                      name). The order must name each parameter of the prepared SDFG exactly once.
    :param check_compiles: compile the finished unit before returning it (:func:`compile_check`).
                           Default on: a form that renders clean text and does not compile is
                           served as good and read as good, which is the one failure CPF has no
                           other gate for. Pass ``False`` only where the caller compiles the
                           result itself.
    :param isa: the ISA whose tile functions the tile nodes call (:func:`target_tile_functions`);
                ``AUTO`` is the widest one the host executes.
    :returns: the :class:`Rendering`.
    :raises NotImplementedError: if the SDFG needs anything the dialect's units cannot hold; the
                                 message names the construct.
    :raises RuntimeError: if a finished unit is not self-contained, or does not compile.
    :raises ValueError: if ``language`` is neither, or if ``signature`` does not fit the entry
                        (see :class:`EntrySignature`).
    """
    dialect = dialect_for(language)
    prepared = copy.deepcopy(sdfg)
    if signature is not None and signature.workspace is not None and signature.workspace_size is not None:
        add_workspace(prepared, signature.workspace, signature.workspace_size)
    target_tile_functions(prepared, isa)
    provenance: dict[str, tuple[str, str]] = {}
    # Under the dialect, because ``prepare`` EXPANDS library nodes and an expansion bakes its
    # tasklet text once and for good. A node that spells its own element transform -- ArgReduce
    # writes ``std::abs`` -- had no way to know which dialect was being rendered and always chose
    # the C++ one, so every argmax over a transformed element failed the C rendering at the
    # self-containment check rather than at anything a caller could act on.
    with cpf_lowering.dialect_scope(dialect), device_backend(dialect):
        # ``prepare`` expands library nodes into loops and tasklets on this throwaway copy; nothing
        # CPF emits reads node.debuginfo, so the inspect.stack() walk behind it is pure overhead. No
        # history either: the first recorded expansion deep-copies the whole copy into ``orig_sdfg``.
        # A unit is built where the rendering machine's FFTW3 companion libraries may not be: plan single-threaded.
        with (
            set_temporary("compiler", "lineinfo", value="none"),
            set_temporary("store_history", value=False),
            fftw3.single_threaded_planner(),
        ):
            prepare(prepared, provenance)
    # DACE_* environment variables outrank set_temporary, so a shell that pins the CPU generator to
    # ``legacy`` would silently render through the wrong one -- and the legacy generator emits
    # ``dace::CopyND`` and state-struct accesses that no dialect switch can take back. Refuse.
    with set_temporary("compiler", "cpu", "implementation", value="experimental_readable"), device_backend(dialect):
        selected = Config.get("compiler", "cpu", "implementation")
        if selected != "experimental_readable":
            raise RuntimeError(
                "CPF builds on the readable CPU code generator, but "
                f"compiler.cpu.implementation is pinned to {selected!r} (a DACE_* environment "
                "variable outranks the in-process setting). Unset it to render."
            )
        with cpf_lowering.dialect_scope(dialect):
            with cpf_lowering.provenance_scope(provenance):
                # Same reasoning as the prepare() call above: codegen's own lowering passes (copy
                # lifting, library expansion) add nodes whose debuginfo nothing here reads.
                with set_temporary("compiler", "lineinfo", value="none"), fftw3.single_threaded_planner():
                    objects = codegen.generate_code(prepared, validate=validate)
                frame, device = split_units(objects, sdfg.name)
                body = frame.clean_code
                # Type names reach the text from the entry signature and from declarations, neither
                # of which goes through an expression printer, so the rename runs over the whole unit.
                body = cpf_lowering.rewrite_ctypes(body, dialect)
                device_body = ""
                if device is not None:
                    device_body = state_struct(body, sdfg.name) + cpf_lowering.rewrite_ctypes(
                        device.clean_code, dialect
                    )
                # Nothing below changes the prepared SDFG, so one argument list serves both uses.
                entry_arglist = prepared.arglist()
                caller_writable = (signature.workspace,) if signature is not None and signature.workspace else ()
                body = qualify_readonly_pointers(body, prepared, sdfg.name, entry_arglist, caller_writable)
                # After the qualifier pass, so each declaration carries the ``const`` CPF decided
                # on before it moves; only the order changes here.
                if signature is not None:
                    body = reorder_entry_parameters(body, sdfg.name, signature.order)
                if dialect is cpf_lowering.Dialect.STANDALONE_C:
                    # ``__restrict__`` is the GNU spelling ``Data.as_arg`` emits because C++ has no
                    # ``restrict`` keyword. C does, and it is the one a C23 unit should carry.
                    body = re.sub(r"\b__restrict__\b", "restrict", body)
                # Last, because the passes above match the DaCe spellings the generator wrote.
                if dialect in cpf_lowering.DEVICE_DIALECTS:
                    body = cpf_lowering.device_spell_out(body, dialect)
                    device_body = cpf_lowering.respell_vendor_namespaces(
                        cpf_lowering.device_spell_out(device_body, dialect)
                    )
                body = cpf_lowering.respell_vendor_namespaces(body)
    code = preamble(body, dialect, frame.environments) + body
    # The device unit repeats the frame's state struct, whose vendor handles and plans need their declarations.
    device_environments = (set(device.environments) | set(frame.environments)) if device is not None else set()
    device_code = preamble(device_body, dialect, device_environments) + device_body if device_body else ""
    for unit in filter(None, (code, device_code)):
        verify(unit, sdfg.name, dialect)
        if check_compiles:
            compile_check(unit, sdfg.name, dialect, vendor_include_flags(set(frame.environments) | device_environments))
    arguments = tuple(signature.order) if signature is not None else tuple(entry_arglist)
    return Rendering(
        code,
        prepared,
        arguments,
        device_code,
        link_libraries(frame, dialect, code + device_code),
        cpf_lowering.vendor_families(code + device_code),
    )


def link_libraries(frame: CodeObject, dialect: cpf_lowering.Dialect, code: str = "") -> tuple[str, ...]:
    """The libraries of the frame's environments, by environment name, without those CPF supplies itself. Only
    link-only environments carry any here: the standalone footer refuses every other kind."""
    libraries: list[str] = []
    for vendor in cpf_lowering.vendor_libraries(frame.environments, code):
        libraries += vendor.link_tokens(code)
    for name in sorted(frame.environments):
        env = get_environment(name)
        if env.__name__ in cpf_lowering.VENDOR_LIBRARIES:
            continue
        provided = cpf_lowering.PROVIDED_ENVIRONMENTS
        if dialect in cpf_lowering.DEVICE_DIALECTS:
            provided = provided | cpf_lowering.DEVICE_PROVIDED_ENVIRONMENTS
        if env.__name__ not in provided:
            libraries += env.cmake_libraries
    return tuple(libraries)


def cpf(
    sdfg: SDFG,
    validate: bool = True,
    language: str = "c++",
    signature: EntrySignature | None = None,
    check_compiles: bool = True,
    isa: ISA = ISA.AUTO,
) -> str:
    """Render ``sdfg`` as one self-contained host translation unit.

    The SDFG is copied first, so neither the lifetime demotions nor the code generator's own
    in-place lowering (library expansion, inlining, explicit copies) is visible to the caller.

    Use :func:`render` instead where the code will actually be CALLED: preparation can add an
    argument, and :class:`Rendering` carries the SDFG that says which.

    :param sdfg: the SDFG to render.
    :param validate: validate the SDFG during code generation.
    :param language: ``'c++'`` (the default) or ``'c'``. A device dialect renders two units, which
                     :func:`render` returns.
    :param signature: the entry point's parameter order and workspace pair; ``None`` keeps CPF's
                      (see :func:`render`).
    :param check_compiles: compile the finished unit before returning it (see :func:`render`).
    :param isa: the ISA of the tile functions (see :func:`render`).
    :returns: the translation unit, defining ``extern "C" void <sdfg.name>(<arglist>)`` in C++ and
              ``void <sdfg.name>(<arglist>)`` in C, whose ABI is the same.
    :raises NotImplementedError: if the SDFG needs anything a single host unit cannot hold; the
                                 message names the construct.
    :raises ValueError: if ``language`` is neither, or if ``signature`` does not fit the entry
                        (see :class:`EntrySignature`).
    """
    if dialect_for(language) in cpf_lowering.DEVICE_DIALECTS:
        raise ValueError(f"CPF renders {language!r} as a host and a device unit: call render(), not cpf()")
    rendering = render(
        sdfg, validate=validate, language=language, signature=signature, check_compiles=check_compiles, isa=isa
    )
    return rendering.code
