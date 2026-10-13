// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

// NCCL and RCCL share one API (RCCL exports the ``nccl*`` symbols), so the same generated code
// serves both GPU backends. DaCe defines WITH_HIP for every translation unit of a HIP build.
#ifdef WITH_HIP
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#else
#include <cuda_runtime.h>
#include <nccl.h>
#endif

#include <mpi.h>

#include "dace_ccl_fwd.h"

#include <map>
#include <set>
#include <stdexcept>
#include <string>

namespace dace {

namespace comm {

#ifdef WITH_HIP
typedef hipStream_t GpuStream;
#else
typedef cudaStream_t GpuStream;
#endif

static void CheckNcclError(ncclResult_t status, const char* expr) {
  if (status != ncclSuccess) {
    throw std::runtime_error(std::string("NCCL call ") + expr + " failed: " + ncclGetErrorString(status));
  }
}

#define DACE_NCCL_CHECK(expr) ::dace::comm::CheckNcclError((expr), #expr)

static void SynchronizeStream(GpuStream stream) {
#ifdef WITH_HIP
  hipError_t status = hipStreamSynchronize(stream);
  if (status != hipSuccess) {
    throw std::runtime_error(std::string("hipStreamSynchronize failed: ") + hipGetErrorString(status));
  }
#else
  cudaError_t status = cudaStreamSynchronize(stream);
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("cudaStreamSynchronize failed: ") + cudaGetErrorString(status));
  }
#endif
}

// NCCL added ncclAlltoAll, ncclGather and ncclScatter in 2.28. RCCL has long had ncclGather and
// ncclScatter (announced by RCCL_GATHER_SCATTER) and spelled the all-to-all ncclAllToAll until it
// took the NCCL name with the 2.28 API. Without a native call, the helpers below fall back to
// grouped point-to-point operations.
#if defined(NCCL_VERSION_CODE) && NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
#define DACE_NCCL_NATIVE_ALLTOALL
#define DACE_NCCL_NATIVE_GATHER_SCATTER
#elif defined(WITH_HIP)
#define DACE_NCCL_NATIVE_ALLTOALL
#ifdef RCCL_GATHER_SCATTER
#define DACE_NCCL_NATIVE_GATHER_SCATTER
#endif
#endif

#if !defined(DACE_NCCL_NATIVE_ALLTOALL) || !defined(DACE_NCCL_NATIVE_GATHER_SCATTER)
/// The size in bytes of one element of an NCCL type, to address the per-rank chunks of a buffer.
static size_t NcclTypeSize(ncclDataType_t type) {
  switch (type) {
    case ncclInt8:
    case ncclUint8:
      return 1;
    case ncclFloat16:
#if defined(WITH_HIP) || NCCL_VERSION_CODE >= NCCL_VERSION(2, 10, 0)
    case ncclBfloat16:
#endif
      return 2;
    case ncclInt32:
    case ncclUint32:
    case ncclFloat32:
      return 4;
    case ncclInt64:
    case ncclUint64:
    case ncclFloat64:
      return 8;
    default:
      throw std::runtime_error("Unsupported NCCL data type");
  }
}

/// The ``index``-th chunk of ``bytes`` bytes of ``buffer``.
static const void* Chunk(const void* buffer, int index, size_t bytes) {
  return static_cast<const char*>(buffer) + index * bytes;
}

static void* Chunk(void* buffer, int index, size_t bytes) { return static_cast<char*>(buffer) + index * bytes; }
#endif

/// Exchanges ``count`` elements with every rank, as ``MPI_Alltoall``: the chunk at ``send + i * count``
/// goes to rank ``i``, and the chunk from rank ``i`` lands at ``recv + i * count``.
static void AlltoAll(const void* send, void* recv, size_t count, ncclDataType_t type, ncclComm_t comm,
                     GpuStream stream) {
#if defined(NCCL_VERSION_CODE) && NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
  DACE_NCCL_CHECK(ncclAlltoAll(send, recv, count, type, comm, stream));
#elif defined(DACE_NCCL_NATIVE_ALLTOALL)
  DACE_NCCL_CHECK(ncclAllToAll(send, recv, count, type, comm, stream));
#else
  int size;
  DACE_NCCL_CHECK(ncclCommCount(comm, &size));
  const size_t bytes = count * NcclTypeSize(type);
  DACE_NCCL_CHECK(ncclGroupStart());
  for (int peer = 0; peer < size; ++peer) {
    DACE_NCCL_CHECK(ncclSend(Chunk(send, peer, bytes), count, type, peer, comm, stream));
    DACE_NCCL_CHECK(ncclRecv(Chunk(recv, peer, bytes), count, type, peer, comm, stream));
  }
  DACE_NCCL_CHECK(ncclGroupEnd());
#endif
}

/// Gathers ``count`` elements of every rank to ``root``, as ``MPI_Gather``: the elements of rank ``i``
/// land at ``recv + i * count``. Only the root uses ``recv``.
static void Gather(const void* send, void* recv, size_t count, ncclDataType_t type, int root, ncclComm_t comm,
                   GpuStream stream) {
#ifdef DACE_NCCL_NATIVE_GATHER_SCATTER
  DACE_NCCL_CHECK(ncclGather(send, recv, count, type, root, comm, stream));
#else
  int size, rank;
  DACE_NCCL_CHECK(ncclCommCount(comm, &size));
  DACE_NCCL_CHECK(ncclCommUserRank(comm, &rank));
  const size_t bytes = count * NcclTypeSize(type);
  DACE_NCCL_CHECK(ncclGroupStart());
  DACE_NCCL_CHECK(ncclSend(send, count, type, root, comm, stream));
  if (rank == root) {
    for (int peer = 0; peer < size; ++peer) {
      DACE_NCCL_CHECK(ncclRecv(Chunk(recv, peer, bytes), count, type, peer, comm, stream));
    }
  }
  DACE_NCCL_CHECK(ncclGroupEnd());
#endif
}

/// Scatters ``count`` elements to every rank from ``root``, as ``MPI_Scatter``: rank ``i`` receives the
/// elements at ``send + i * count``. Only the root uses ``send``.
static void Scatter(const void* send, void* recv, size_t count, ncclDataType_t type, int root, ncclComm_t comm,
                    GpuStream stream) {
#ifdef DACE_NCCL_NATIVE_GATHER_SCATTER
  DACE_NCCL_CHECK(ncclScatter(send, recv, count, type, root, comm, stream));
#else
  int size, rank;
  DACE_NCCL_CHECK(ncclCommCount(comm, &size));
  DACE_NCCL_CHECK(ncclCommUserRank(comm, &rank));
  const size_t bytes = count * NcclTypeSize(type);
  DACE_NCCL_CHECK(ncclGroupStart());
  if (rank == root) {
    for (int peer = 0; peer < size; ++peer) {
      DACE_NCCL_CHECK(ncclSend(Chunk(send, peer, bytes), count, type, peer, comm, stream));
    }
  }
  DACE_NCCL_CHECK(ncclRecv(recv, count, type, root, comm, stream));
  DACE_NCCL_CHECK(ncclGroupEnd());
#endif
}

/**
 * The NCCL communicators of a DaCe program, one per MPI communicator it communicates over, and the
 * NCCL group that collects its nonblocking point-to-point operations.
 *
 * NCCL has no bootstrap of its own, so every communicator is created from an MPI communicator.
 * NCCL also has no requests: an ``Isend``/``Irecv`` joins an open NCCL group, which the next
 * blocking operation or ``Wait`` closes. That fuses an exchange of nonblocking operations into
 * one launch, which is what keeps a ring of sends and receives from deadlocking.
 **/
class NcclContext {
 public:
  NcclContext() = default;
  NcclContext(const NcclContext&) = delete;
  NcclContext& operator=(const NcclContext&) = delete;
  // A destructor that throws terminates the process, so teardown does not check for errors.
  ~NcclContext() { Destroy(); }

  /// Creates the NCCL communicator of ``mpi_comm``. Collective over the processes of ``mpi_comm``;
  /// does nothing on a process outside it (``MPI_COMM_NULL``) or when the communicator exists.
  void Create(MPI_Comm mpi_comm) {
    if (mpi_comm == MPI_COMM_NULL || comms_.count(mpi_comm) > 0) return;
    int rank, size;
    MPI_Comm_rank(mpi_comm, &rank);
    MPI_Comm_size(mpi_comm, &size);
    ncclUniqueId id;
    if (rank == 0) DACE_NCCL_CHECK(ncclGetUniqueId(&id));
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, mpi_comm);
    ncclComm_t comm;
    DACE_NCCL_CHECK(ncclCommInitRank(&comm, size, id, rank));
    comms_[mpi_comm] = comm;
  }

  /// Returns the NCCL communicator created for ``mpi_comm``.
  ncclComm_t Get(MPI_Comm mpi_comm) const {
    auto it = comms_.find(mpi_comm);
    if (it == comms_.end()) {
      throw std::runtime_error("No NCCL communicator was created for this MPI communicator");
    }
    return it->second;
  }

  /// Opens the NCCL group that the following operations join, unless one is open already.
  void Begin() {
    if (group_open_) return;
    DACE_NCCL_CHECK(ncclGroupStart());
    group_open_ = true;
  }

  /// Remembers that a nonblocking operation was enqueued on ``stream``, for ``Wait`` to synchronize.
  void Track(GpuStream stream) { pending_streams_.insert(stream); }

  /// Closes the open NCCL group, which launches every operation in it.
  void Launch() {
    if (!group_open_) return;
    group_open_ = false;
    DACE_NCCL_CHECK(ncclGroupEnd());
  }

  /// Completes every nonblocking operation: launches them and synchronizes their streams.
  void Wait() {
    Launch();
    for (GpuStream stream : pending_streams_) SynchronizeStream(stream);
    pending_streams_.clear();
  }

  void Destroy() {
    for (auto& entry : comms_) ncclCommDestroy(entry.second);
    comms_.clear();
  }

 private:
  std::map<MPI_Comm, ncclComm_t> comms_;
  std::set<GpuStream> pending_streams_;
  bool group_open_ = false;
};

}  // namespace comm

}  // namespace dace
