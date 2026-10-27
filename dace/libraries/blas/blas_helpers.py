# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from copy import deepcopy as dc
from typing import Any, NamedTuple

import numpy as np

import dace
from dace import data, dtypes
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil


def matrix_view(subset) -> tuple[list[Any], list[int]]:
    """
    Returns an operand's matrix view: the raw subset if it is already 2D, otherwise the squeezed one.

    Squeezing unconditionally rejects a genuine ``(N, 1)`` column as "not a matrix"; not squeezing at
    all rejects an ``(NQ, 1, NP)`` reshape. Callers that read a size, a stride or a dimension index
    must all use this view, or they disagree about which dimensions the operand has.

    :param subset: The subset of the memlet accessing the operand.
    :return: The size in the matrix view, and the subset dimensions it kept.
    """
    size = subset.size()
    if len(size) == 2:
        return size, list(range(len(size)))
    squeezed = dc(subset)
    dims = squeezed.squeeze()
    return squeezed.size(), dims


def to_blastype(dtype):
    """Returns a BLAS character that corresponds to the input type.
    Used in MKL/CUBLAS calls."""

    if dtype == np.float16:
        return "H"
    elif dtype == np.float32:
        return "S"
    elif dtype == np.float64:
        return "D"
    elif dtype == np.complex64:
        return "C"
    elif dtype == np.complex128:
        return "Z"
    else:
        raise TypeError(f"Type {dtype.__name__} not supported in BLAS operations")


def cublas_type_metadata(dtype: dtypes.typeclass) -> tuple[str, str, str]:
    """
    Returns type metadata on a given dace dtype.

    :return: A 3 tuple of (BLAS letter, CUDA C type, Name in dace runtime).
    """

    if dtype == dtypes.float16:
        return "H", "__half", "Half"
    elif dtype == dtypes.float32:
        return "S", "float", "Float"
    elif dtype == dtypes.float64:
        return "D", "double", "Double"
    elif dtype == dtypes.complex64:
        return "C", "cuComplex", "Complex64"
    elif dtype == dtypes.complex128:
        return "Z", "cuDoubleComplex", "Complex128"
    else:
        raise TypeError(f"Type {str(dtype)} not supported in BLAS operations")


def dtype_to_cudadatatype(dtype: dtypes.typeclass) -> str:
    types = {
        dtypes.float16: "CUDA_R_16F",
        dtypes.float32: "CUDA_R_32F",
        dtypes.complex64: "CUDA_C_32F",
        dtypes.float64: "CUDA_R_64F",
        dtypes.complex128: "CUDA_C_64F",
        dtypes.int8: "CUDA_R_8I",
        dtypes.uint8: "CUDA_R_8U",
        dtypes.int32: "CUDA_R_32I",
        # dtypes.complex32: 'CUDA_C_16F',
        # dtypes.bfloat16: 'CUDA_R_16BF',
        # dtypes.bcomplex32: 'CUDA_C_16BF',
        # dtypes.icomplex16: 'CUDA_C_8I',
        # dtypes.iucomplex16: 'CUDA_C_8U',
        # dtypes.icomplex64: 'CUDA_C_32I',
    }
    return types[dtype]


def to_cublas_computetype(dtype: dtypes.typeclass) -> str:
    types = {
        dtypes.float16: "16F",
        dtypes.float32: "32F",
        dtypes.float64: "64F",
        dtypes.complex64: "32F",
        dtypes.complex128: "64F",
        dtypes.int8: "32I",
        dtypes.int16: "32I",
        dtypes.int32: "32I",
        dtypes.int64: "32I",
        dtypes.uint8: "32I",
        dtypes.uint16: "32I",
        dtypes.uint32: "32I",
        dtypes.uint64: "32I",
    }
    return types[dtype]


def get_gemm_opts(a_strides, b_strides, c_strides) -> dict[str, Any]:
    """
    Returns GEMM argument order, transposition, and leading dimensions
    based on column-major storage from dace arrays.

    :param a: Data descriptor for the first matrix.
    :param b: Data descriptor for the second matrix.
    :param c: Data descriptor for the output matrix.
    :return: A dictionary with the following keys: swap (if True, a and b
             should be swapped); lda, ldb, ldc (leading dimensions); ta, tb
             (whether GEMM should be called with OP_N or OP_T).
    """
    # possible order (C, row based) of dimensions in input array
    # and computed result based on
    # 1. N/T - transpose flag in cublas
    # 2. LR/RL - order in which A and B are passed into cublas
    #     k m, n k -> n m (LR, N, N)
    #     m k, n k -> n m (LR, T, N)
    #     k m, k n -> n m (LR, N, T)
    #     m k, k n -> n m (LR, T, T)
    #     m k, k n -> m n (RL, N, N)
    #     m k, n k -> m n (RL, N, T)
    #     k m, k n -> m n (RL, T, N)
    #     k m, n k -> m n (RL, T, T)
    #       |    |      |
    #     use these 3 to detect correct option

    sAM, sAK = a_strides[-2:]
    sBK, sBN = b_strides[-2:]
    sCM, sCN = c_strides[-2:]

    opts = {
        "mkm": {"swap": False, "lda": sAK, "ldb": sBN, "ldc": sCN, "ta": "N", "tb": "N"},
        "kkm": {"swap": False, "lda": sAM, "ldb": sBN, "ldc": sCN, "ta": "T", "tb": "N"},
        "mnm": {"swap": False, "lda": sAK, "ldb": sBK, "ldc": sCN, "ta": "N", "tb": "T"},
        "knm": {"swap": False, "lda": sAM, "ldb": sBK, "ldc": sCN, "ta": "T", "tb": "T"},
        "knn": {"swap": True, "lda": sAM, "ldb": sBK, "ldc": sCM, "ta": "N", "tb": "N"},
        "kkn": {"swap": True, "lda": sAM, "ldb": sBN, "ldc": sCM, "ta": "N", "tb": "T"},
        "mnn": {"swap": True, "lda": sAK, "ldb": sBK, "ldc": sCM, "ta": "T", "tb": "N"},
        "mkn": {"swap": True, "lda": sAK, "ldb": sBN, "ldc": sCM, "ta": "T", "tb": "T"},
    }

    if sAM == 1 and sAK != 1:
        optA = "m"
    elif sAK == 1:
        optA = "k"
    else:
        raise Exception("sAM or sAK should be 1")

    if sBN == 1:
        optB = "n"
    elif sBK == 1:
        optB = "k"
    else:
        raise Exception("sBK or sBN should be 1")

    if sCM == 1:
        optC = "m"
    elif sCN == 1:
        optC = "n"
    else:
        raise Exception("sCM or sCN should be 1")

    return opts[optA + optB + optC]


def check_one_device(*descs: data.Data):
    """Raise unless one device can access every operand: a register vector and a heap matrix both live on the
    host, although their storage types differ.

    :param descs: the operand descriptors.
    """
    if not any(
        all(dtypes.can_access(schedule, desc.storage) for desc in descs)
        for schedule in (dtypes.ScheduleType.Default, dtypes.ScheduleType.GPU_Device)
    ):
        raise ValueError("Input matrices must be accessible from one device")


def check_access(schedule: dtypes.ScheduleType, *descs: data.Data):
    """If schedule cannot access all passed descriptors, through an error.

    :param schedule: the schedule.
    :param descs: the descriptors to check.
    """
    for desc in descs:
        if not dtypes.can_access(schedule, desc.storage):
            raise ValueError(f"Schedule mismatch: {schedule} cannot access {desc.storage}")


class Operand(NamedTuple):
    """The data on a BLAS connector, with its unit dimensions squeezed out."""

    desc: data.Data
    shape: list
    strides: list


def operand(node: nodes.LibraryNode, state: "dace.SDFGState", sdfg: "dace.SDFG", connector: str) -> Operand:
    if connector in node.in_connectors:
        edge = sdutil.in_edge_with_name(node, state, connector)
    else:
        edge = sdutil.out_edge_with_name(node, state, connector)
    desc = sdfg.arrays[edge.data.data]
    subset = dc(edge.data.subset)
    dims = subset.squeeze()
    return Operand(desc, subset.size(), [desc.strides[d] * edge.data.subset[d][2] for d in dims])


def require_one_dtype(node: nodes.LibraryNode, state: "dace.SDFGState", sdfg: "dace.SDFG", *connectors: str) -> None:
    """Raise unless the data on every one of ``connectors`` has one element type: one BLAS routine reads them all."""
    types = {connector: operand(node, state, sdfg, connector).desc.dtype for connector in connectors}
    if len(set(types.values())) > 1:
        raise ValueError(f"{node.label}: every operand needs one element type, got {types}")


class MatrixLayout(NamedTuple):
    """How BLAS reads matrices: all in row-major or all in column-major order, each with its leading dimension."""

    row_major: bool
    leading: list


def matrix_layout(*matrices: Operand) -> MatrixLayout | None:
    """The common layout of ``matrices``, or ``None`` if one has no contiguous dimension or they differ."""
    if all(m.strides[1] == 1 for m in matrices):
        return MatrixLayout(True, [m.strides[0] for m in matrices])
    if all(m.strides[0] == 1 for m in matrices):
        return MatrixLayout(False, [m.strides[1] for m in matrices])
    return None


def blas_prefix(dtype: dtypes.typeclass) -> str | None:
    """The BLAS letter of ``dtype`` (``D`` for ``double``), or ``None`` if BLAS has no routines for it."""
    try:
        func, _, _ = cublas_type_metadata(dtype)
    except TypeError:
        return None
    return None if func == "H" else func


def scalar_arguments(dtype: dtypes.typeclass, **values: Any) -> tuple[str, dict[str, str]]:
    """Declares each of ``values`` as a scalar of ``dtype``; returns the declarations and, per name, the argument
    passing it to CBLAS: the value itself to a real routine, its address to a complex one."""
    declarations = "".join(f"{dtype.ctype} __{name} = {dtype.ctype}({value});\n" for name, value in values.items())
    if dtype in (dtypes.complex64, dtypes.complex128):
        return declarations, {name: f"&__{name}" for name in values}
    return declarations, {name: f"__{name}" for name in values}


def connector_sdfg(node: nodes.LibraryNode, state: "dace.SDFGState", sdfg: "dace.SDFG") -> "dace.SDFG":
    """A nested SDFG with one array per connector of ``node``, in the squeezed shape and strides of its data."""
    nsdfg = dace.SDFG(f"{node.label}_sdfg")
    for connector in list(node.in_connectors) + list(node.out_connectors):
        desc, shape, strides = operand(node, state, sdfg, connector)
        nsdfg.add_array(connector, shape, desc.dtype, desc.storage, strides=strides)
    return nsdfg


def same_memory(node: nodes.LibraryNode, state: "dace.SDFGState", source: str, target: str) -> bool:
    """Whether connectors ``source`` and ``target`` of ``node`` carry the same data and subset."""
    read = sdutil.in_edge_with_name(node, state, source).data
    write = sdutil.out_edge_with_name(node, state, target).data
    return read.data == write.data and read.subset == write.subset


def in_place_call(
    node: nodes.LibraryNode,
    state: "dace.SDFGState",
    sdfg: "dace.SDFG",
    source: str,
    target: str,
    code: str,
    overwrites: bool = False,
) -> "dace.SDFG":
    """A nested SDFG for a BLAS routine that updates an operand in place: copies connector ``source`` to ``target``,
    then runs ``code``, which reads the other inputs and updates ``target``, each by its connector's name with one
    more leading underscore (``__A`` for ``_A``): a tasklet connector named as the array would shadow it.

    The copy is skipped when ``source`` and ``target`` already share memory, or when the routine ``overwrites``
    ``target`` without reading it (``beta == 0``).
    """
    nsdfg = connector_sdfg(node, state, sdfg)
    # The source stays a tasklet input even when unused, so the outer graph keeps its read
    reads = {f"_{c}": c for c in node.in_connectors}
    if overwrites or same_memory(node, state, source, target):
        call_state = nsdfg.add_state(f"{node.label}_call")
    else:
        copy_state = nsdfg.add_state(f"{node.label}_copy")
        copy_state.add_nedge(
            copy_state.add_read(source),
            copy_state.add_write(target),
            dace.Memlet.from_array(source, nsdfg.arrays[source]),
        )
        call_state = nsdfg.add_state_after(copy_state, f"{node.label}_call")
        # The call reads ``target`` too, so the copy into it is not dead
        reads[f"_{target}_before"] = target
    tasklet = call_state.add_tasklet(node.label, list(reads), [f"_{target}"], code, language=dtypes.Language.CPP)
    for connector, name in reads.items():
        call_state.add_edge(
            call_state.add_read(name), None, tasklet, connector, dace.Memlet.from_array(name, nsdfg.arrays[name])
        )
    call_state.add_edge(
        tasklet, f"_{target}", call_state.add_write(target), None, dace.Memlet.from_array(target, nsdfg.arrays[target])
    )
    return nsdfg
