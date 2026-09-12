# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The vendor vocabulary a GPU sparse expansion needs, one object per backend.

hipSPARSE mirrors the cuSPARSE GENERIC API call for call -- ``hipsparseCreateCsr``,
``hipsparseSpMV``, ``hipsparseDnVecDescr_t`` -- so the two expansions differ only in a prefix, an
enum prefix, a handle name, an error check and the data-type enum. Those five live here and the
expansion body is shared, rather than a second copy of the body per vendor.

Not rocSPARSE: that is the NATIVE library, spelled differently throughout, and hipSPARSE dispatches
to it anyway -- so targeting it would mean maintaining a second body for the same work.
"""
from typing import NamedTuple


class GpuSparseDialect(NamedTuple):
    """How one vendor sparse library spells what an expansion emits."""

    #: Function-name prefix: ``cusparse`` / ``hipsparse``.
    prefix: str
    #: Enum prefix: ``CUSPARSE`` / ``HIPSPARSE``.
    upper: str
    #: The handle variable the environment's setup code puts in scope.
    handle: str
    #: Error-check function wrapping every call.
    check: str
    #: Scalar-type enum prefix: cuSPARSE takes the CUDA-wide ``cudaDataType`` (``CUDA_R_64F``),
    #: hipSPARSE takes HIP's, which is the same set under a different spelling (``HIP_R_64F``).
    datatype_prefix: str


CUSPARSE = GpuSparseDialect(
    prefix="cusparse",
    upper="CUSPARSE",
    handle="__dace_cusparse_handle",
    check="dace::sparse::CheckCusparseError",
    datatype_prefix="CUDA",
)

HIPSPARSE = GpuSparseDialect(
    prefix="hipsparse",
    upper="HIPSPARSE",
    handle="__dace_hipsparse_handle",
    check="dace::sparse::CheckHipsparseError",
    datatype_prefix="HIP",
)
