# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Handles of communicators created outside a program, for the ``_comm`` connector of communication nodes (see
:class:`dace.libraries.comm.nodes.node.CommNode`). A program that takes its communicators as arguments can run inside
an application that owns the processes (e.g., a framework that bootstraps NCCL over its own process group), and an
NCCL program needs no MPI then.

A handle is an integer: pass it to the program as an ``int64`` scalar argument connected to ``_comm``.
"""

import ctypes
import ctypes.util
import functools
import glob
import os
from typing import Any

from dace.libraries.comm.environments.nccl import NCCL

#: Size in bytes of an ``ncclUniqueId`` (``NCCL_UNIQUE_ID_BYTES``)
NCCL_UNIQUE_ID_BYTES = 128


class _NcclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_char * NCCL_UNIQUE_ID_BYTES)]


@functools.cache
def _nccl_library() -> ctypes.CDLL:
    """The NCCL (or RCCL) shared library that programs link (see :class:`dace.libraries.comm.environments.NCCL`)."""
    name = "rccl" if NCCL.uses_rccl() else "nccl"
    path = None
    root = None if NCCL.uses_rccl() else NCCL.nccl_root()
    if root:
        libraries = sorted(glob.glob(os.path.join(root, "lib", "libnccl.so*")))
        path = libraries[0] if libraries else None
    path = path or ctypes.util.find_library(name)
    if path is None:
        raise OSError(f"Cannot find the {name.upper()} library (see dace.libraries.comm.environments.NCCL)")
    lib = ctypes.CDLL(path)
    lib.ncclGetUniqueId.argtypes = [ctypes.POINTER(_NcclUniqueId)]
    lib.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, _NcclUniqueId, ctypes.c_int]
    lib.ncclCommDestroy.argtypes = [ctypes.c_void_p]
    lib.ncclGetErrorString.argtypes = [ctypes.c_int]
    lib.ncclGetErrorString.restype = ctypes.c_char_p
    return lib


def _check(status: int, call: str) -> None:
    if status != 0:
        message = _nccl_library().ncclGetErrorString(status).decode()
        raise RuntimeError(f"{call} failed: {message}")


def nccl_unique_id() -> bytes:
    """
    A new NCCL unique ID, which one process creates and sends to every process of the communicator to create (e.g.,
    over the process group of the application).

    :return: The ``ncclUniqueId`` as ``NCCL_UNIQUE_ID_BYTES`` bytes.
    """
    uid = _NcclUniqueId()
    _check(_nccl_library().ncclGetUniqueId(ctypes.byref(uid)), "ncclGetUniqueId")
    return bytes(uid)  # All bytes: the ``internal`` field would stop at the first NUL


def nccl_init_rank(unique_id: bytes, rank: int, nranks: int) -> int:
    """
    Creates this process's member of an NCCL communicator on the current GPU (``ncclCommInitRank``). Collective: every
    process of the communicator calls it with the same unique ID.

    :param unique_id: The unique ID (see :func:`nccl_unique_id`).
    :param rank: The rank of this process in the communicator.
    :param nranks: The number of processes in the communicator.
    :return: The handle of the ``ncclComm_t``. Destroy it with :func:`nccl_destroy`.
    """
    if len(unique_id) != NCCL_UNIQUE_ID_BYTES:
        raise ValueError(f"An NCCL unique ID has {NCCL_UNIQUE_ID_BYTES} bytes, got {len(unique_id)}")
    uid = _NcclUniqueId.from_buffer_copy(unique_id)
    comm = ctypes.c_void_p()
    _check(_nccl_library().ncclCommInitRank(ctypes.byref(comm), nranks, uid, rank), "ncclCommInitRank")
    return comm.value


def nccl_destroy(handle: int) -> None:
    """Destroys an NCCL communicator created by :func:`nccl_init_rank`."""
    _check(_nccl_library().ncclCommDestroy(ctypes.c_void_p(handle)), "ncclCommDestroy")


def mpi_handle(comm: Any) -> int:
    """
    The handle of an ``mpi4py`` communicator.

    :param comm: An ``mpi4py.MPI.Comm``.
    :return: Its ``MPI_Comm`` as an integer.
    """
    # Imported here: mpi4py is optional, and importing it initializes MPI
    from mpi4py import MPI

    return MPI._handleof(comm)
