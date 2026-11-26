# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Process groups of ``torch.distributed`` as communicators of DaCe programs.

The functional collectives in a captured graph (``_c10d_functional.*``) name their process group by a string. The
importer resolves it to a :class:`GroupInfo` while lowering, and the program takes one communicator handle per group
and backend as an argument (the ``_comm`` connector of :mod:`dace.libraries.comm` nodes, see
:func:`communicator_argument`). The runtime fills it in on every call from :func:`communicator`, which creates the
communicator once per process:

- NCCL (CUDA tensors): bootstrapped over the torch process group itself. Group rank 0 creates an NCCL unique ID and
  broadcasts it over the group, then every member creates its NCCL communicator on the current CUDA device. No MPI is
  involved, so programs run under ``torchrun``.
- MPI (CPU tensors): the MPI communicator of the group's ranks. The processes must be MPI processes whose world ranks
  equal their ``torch.distributed`` ranks (e.g., launched with ``mpirun`` and a process group initialized from the
  ``OMPI_COMM_WORLD_*`` environment).

Every rank compiles its own program by default. With distributed compilation, rank 0 compiles and the other ranks load
its build (:func:`compile_sdfg`, through :func:`dace.sdfg.utils.distributed_compile`), and a failure on any rank
raises on every rank instead of leaving the others waiting in the program's first collective.
"""

import atexit
import dataclasses
from typing import Any

import torch
import torch.distributed as dist
from torch._subclasses.fake_tensor import unset_fake_temporarily

from dace.codegen.compiled_sdfg import CompiledSDFG
from dace.libraries.comm import communicators
from dace.sdfg import SDFG
from dace.sdfg import utils as sdfg_utils

#: Implementation of the communication nodes by device type of the communicated tensors
IMPLEMENTATIONS = {"cuda": "NCCL", "cpu": "MPI"}


@dataclasses.dataclass(frozen=True)
class GroupInfo:
    """A ``torch.distributed`` process group as seen from this process."""

    name: str  #: Group name, as the functional collectives give it
    ranks: tuple[int, ...]  #: Global ranks of the members, in group rank order
    rank: int  #: Group rank of this process

    @property
    def size(self) -> int:
        return len(self.ranks)


def resolve_group(group_name: Any) -> GroupInfo:
    """
    Resolves the process group a functional collective names.

    :param group_name: The group name of a ``_c10d_functional`` operator (or a ``ProcessGroup``).
    :return: The members of the group and the rank of this process in it.
    """
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("Lowering a torch.distributed collective requires an initialized default process group")
    group = group_name if isinstance(group_name, dist.ProcessGroup) else _process_group(str(group_name))
    ranks = tuple(dist.get_process_group_ranks(group))
    return GroupInfo(name=group.group_name, ranks=ranks, rank=group.rank())


def implementation_for(device: Any) -> str:
    """The communication library implementation for tensors on ``device`` (``NCCL`` on CUDA/ROCm, ``MPI`` on CPU)."""
    device_type = torch.device(device).type if device is not None else "cpu"
    if device_type not in IMPLEMENTATIONS:
        raise NotImplementedError(f"Collectives on {device_type} tensors are not supported")
    return IMPLEMENTATIONS[device_type]


class ProcessGroupCommunicator:
    """
    The communicator interface :func:`dace.sdfg.utils.distributed_compile` uses (that of ``mpi4py``), over a
    ``torch.distributed`` process group, so that programs launched without MPI (e.g., by ``torchrun``) can share a
    build.
    """

    def __init__(self, group: "dist.ProcessGroup | None" = None):
        self.group = group  #: The process group (``None`` for the default group)

    def Get_rank(self) -> int:
        return dist.get_rank(self.group)

    # The object collectives communicate real tensors: compilation runs under Dynamo's fake tensor mode
    def bcast(self, obj: Any, root: int = 0) -> Any:
        objects = [obj]
        with unset_fake_temporarily():
            dist.broadcast_object_list(objects, group_src=root, group=self.group)
        return objects[0]

    def allgather(self, obj: Any) -> list[Any]:
        objects: list[Any] = [None] * dist.get_world_size(self.group)
        with unset_fake_temporarily():
            dist.all_gather_object(objects, obj, group=self.group)
        return objects


def compile_sdfg(sdfg: SDFG, distributed_compilation: bool = False) -> CompiledSDFG:
    """
    Compiles the SDFG of a graph.

    :param sdfg: The SDFG.
    :param distributed_compilation: Compile on rank 0 of the default process group and load the build on the other
                                    ranks, which must hold the same SDFG (the program may not depend on the rank) and
                                    share a file system. Every rank raises if any rank fails. Without an initialized
                                    default process group, compiles locally.
    :return: The compiled SDFG.
    """
    if distributed_compilation and dist.is_available() and dist.is_initialized():
        return sdfg_utils.distributed_compile(sdfg, ProcessGroupCommunicator())
    return sdfg.compile()


def _process_group(name: str) -> "dist.ProcessGroup":
    return torch._C._distributed_c10d._resolve_process_group(name)


@dataclasses.dataclass
class _Communicator:
    group: "dist.ProcessGroup"
    handle: int
    owner: Any = None  #: Object that keeps the communicator alive (an ``mpi4py`` communicator)
    destroy: Any = None  #: Callable that destroys the communicator at exit


#: Communicators of this process by (group name, implementation)
_COMMUNICATORS: dict[tuple[str, str], _Communicator] = {}


def communicator(group_name: str, implementation: str) -> int:
    """
    The handle of this process's communicator of a process group, created on first use. Collective over the group
    on first use (as every collective of the program is).

    :param group_name: The name of the process group.
    :param implementation: ``NCCL`` or ``MPI``.
    :return: The handle to pass to the program's communicator argument.
    """
    group = _process_group(group_name)
    key = (group_name, implementation)
    cached = _COMMUNICATORS.get(key)
    if cached is not None and cached.group is group:
        return cached.handle
    if implementation == "NCCL":
        created = _create_nccl(group)
    elif implementation == "MPI":
        created = _create_mpi(group)
    else:
        raise ValueError(f"Unknown communication implementation {implementation}")
    if cached is not None and cached.destroy is not None:  # The group was destroyed and its name reused
        cached.destroy()
    _COMMUNICATORS[key] = created
    return created.handle


def _create_nccl(group: "dist.ProcessGroup") -> _Communicator:
    unique_id = [communicators.nccl_unique_id() if group.rank() == 0 else None]
    dist.broadcast_object_list(unique_id, group_src=0, group=group)
    handle = communicators.nccl_init_rank(unique_id[0], group.rank(), group.size())
    return _Communicator(group, handle, destroy=lambda: communicators.nccl_destroy(handle))


def _create_mpi(group: "dist.ProcessGroup") -> _Communicator:
    # Imported here: mpi4py is optional, and importing it initializes MPI
    from mpi4py import MPI

    world = MPI.COMM_WORLD
    if world.Get_rank() != dist.get_rank() or world.Get_size() != dist.get_world_size():
        raise RuntimeError(
            "CPU collectives run over MPI, whose world must match the torch.distributed world: MPI rank "
            f"{world.Get_rank()} of {world.Get_size()}, torch rank {dist.get_rank()} of {dist.get_world_size()}"
        )
    ranks = dist.get_process_group_ranks(group)
    if list(ranks) == list(range(world.Get_size())):
        comm = world
    else:
        comm = world.Create_group(world.Get_group().Incl(list(ranks)))
    return _Communicator(group, communicators.mpi_handle(comm), owner=comm)


@atexit.register
def _destroy_communicators() -> None:
    for entry in _COMMUNICATORS.values():
        if entry.destroy is not None:
            entry.destroy()
    _COMMUNICATORS.clear()
