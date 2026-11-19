# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``MemoryEngineWait`` library node: wait for asynchronous memory engine (TMA) copies to complete."""

from dace import SDFG, SDFGState, dtypes, library, nodes, properties, symbolic
from dace.codegen.common import sym2cpp
from dace.libraries.standard.environments.memory_engine import MemoryEngine
from dace.sdfg.scope import is_devicelevel_gpu
from dace.transformation.transformation import ExpandTransformation

BARRIER_INPUT_NAME = "_bar"
"""Connector of the mbarrier (``uint64`` in ``GPU_Shared`` memory) the copies complete on."""
TOKEN_INPUT_NAME = "_token"
"""Connector of the arrival token of a copy in token mode."""


@library.expansion
class ExpandMemoryEngineWaitNative(ExpandTransformation):
    """Spin on the mbarrier with the ``mbarrier.try_wait`` instruction of the GPU, by token or by phase parity."""

    environments = [MemoryEngine]

    @staticmethod
    def expansion(node: "MemoryEngineWait", parent_state: SDFGState, parent_sdfg: SDFG) -> nodes.Tasklet:
        if not is_devicelevel_gpu(parent_sdfg, parent_state, node):
            raise ValueError(f"{node.label}: MemoryEngineWait must be placed inside a GPU kernel")
        inputs = {BARRIER_INPUT_NAME: dtypes.pointer(dtypes.uint64)}
        if node.parity is None:
            if not any(e.dst_conn == TOKEN_INPUT_NAME for e in parent_state.in_edges(node)):
                raise ValueError(f"{node.label}: MemoryEngineWait needs either a parity or a token input")
            inputs[TOKEN_INPUT_NAME] = dtypes.uint64
            code = f"dace::cuda::mbarrier_wait_token({BARRIER_INPUT_NAME}, {TOKEN_INPUT_NAME});"
        else:
            code = f"dace::cuda::mbarrier_wait_parity({BARRIER_INPUT_NAME}, static_cast<uint32_t>({sym2cpp(node.parity)}));"
        return nodes.Tasklet(node.label, inputs, {}, code, language=dtypes.Language.CPP, side_effects=True)


@library.node
class MemoryEngineWait(nodes.LibraryNode):
    """Every thread waits until the memory engine copies tracked by an mbarrier have completed, after which it sees
    the data they wrote.

    The copies are tracked in one of two ways (see ``InsertAsyncWaits``):
    - by **token**: the ``_token`` input carries the arrival token a copy returned (``parity`` is ``None``).
    - by **phase parity**: ``parity`` is ``k & 1`` for the ``k``-th phase of the barrier since its initialization.
    """

    implementations = {"native": ExpandMemoryEngineWaitNative}
    default_implementation = "native"

    parity = properties.SymbolicProperty(
        default=None, allow_none=True, desc="Parity (0 or 1) of the barrier phase to wait for. None waits on a token."
    )

    def __init__(self, name: str, parity: symbolic.SymbolicType | int | str | None = None, **kwargs):
        inputs = {BARRIER_INPUT_NAME} if parity is not None else {BARRIER_INPUT_NAME, TOKEN_INPUT_NAME}
        super().__init__(name, inputs=inputs, **kwargs)
        self.parity = parity

    @property
    def free_symbols(self) -> set[str]:
        if self.parity is None:
            return set()
        return {str(s) for s in symbolic.pystr_to_symbolic(self.parity).free_symbols}

    def has_side_effects(self, sdfg: SDFG) -> bool:
        return True
