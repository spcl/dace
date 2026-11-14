# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The CPU (CBLAS) and GPU (cuBLAS, hipBLAS) expansions every vendor-backed BLAS node shares."""

from collections.abc import Callable

from dace import library
from dace.libraries import hip_dialect
from dace.libraries.blas import environments
from dace.transformation.transformation import ExpandTransformation


def vendor_expansions(node_name: str, call: Callable) -> dict[str, type[ExpandTransformation]]:
    """The OpenBLAS, MKL, cuBLAS and hipBLAS expansions of a node, each validating the node and returning
    ``call(node, state, sdfg, gpu_environment)``; the CPU libraries get ``None``, as both are CBLAS. hipBLAS is the
    cuBLAS expansion renamed; rocBLAS is its older name."""
    backends = {
        "OpenBLAS": (environments.openblas.OpenBLAS, False),
        "MKL": (environments.intel_mkl.IntelMKL, False),
        "cuBLAS": (environments.cublas.cuBLAS, True),
    }
    expansions = {}
    for implementation, (environment, gpu) in backends.items():

        def expansion(node, parent_state, parent_sdfg, *, bound_environment=environment if gpu else None, **kwargs):
            node.validate(parent_sdfg, parent_state)
            return call(node, parent_state, parent_sdfg, bound_environment)

        name = f"Expand{node_name}{implementation[0].upper()}{implementation[1:]}"
        attributes = {
            "environments": [environment],
            "expansion": staticmethod(expansion),
            "__module__": call.__module__,
        }
        expansions[implementation] = library.expansion(type(name, (ExpandTransformation,), attributes))
    expansions["hipBLAS"] = hip_dialect.hip_expansion(
        expansions["cuBLAS"], environments.hipblas.hipBLAS, hip_dialect.BLAS
    )
    expansions["rocBLAS"] = expansions["hipBLAS"]
    return expansions
