# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Transport of captured control flow through Dynamo and AOTAutograd: an opaque custom operator, ``dace::cfg``, stands
for a whole control-flow graph of traced blocks. AOTAutograd only needs its output metadata (the fake implementation);
the DaCe importer looks the graph up by id and lowers it to schedule-tree control flow.
"""

import dataclasses
import itertools
from typing import Any, Dict, List, Optional

import torch

#: Kinds of block exits
GOTO, BRANCH, RETURN = "goto", "branch", "return"


@dataclasses.dataclass
class Binding:
    """Where a block input comes from: an operator input (``tensor``/``sym`` by index) or a constant."""

    kind: str  #: ``'tensor'``, ``'sym'``
    index: int


@dataclasses.dataclass
class BlockRecord:
    """One traced block (one specialization of a block start)."""

    id: int
    start: int  #: Instruction index (leader) where the block starts
    graph: torch.fx.Graph  #: Torch-level graph traced by Dynamo
    input_names: List[str]  #: CFG variables given as the first placeholders, in order
    input_examples: List[Any]  #: Example (fake) values of those placeholders
    lifted: List[Binding] = dataclasses.field(default_factory=list)  #: Remaining placeholders: operator inputs
    exit_kind: str = GOTO
    #: Successor block ids: one for ``goto``, ``{True: id, False: id}`` for ``branch``, none for ``return``
    successors: Any = None
    #: Names of the CFG variables the block outputs (after the predicate for ``branch``)
    output_names: List[str] = dataclasses.field(default_factory=list)
    #: Python integers the block passes on for symbolic inputs of its successors (e.g., a loop counter's start)
    output_constants: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class CfgRecord:
    """
    A captured control-flow graph: an entry (a branch, or a goto into a loop) into traced blocks, returning flat
    tensors.
    """

    id: int
    blocks: List[BlockRecord]
    entry_predicate: Optional[Binding]  #: Operator input holding the entry predicate (``None``: goto)
    entry_successors: Dict[Any, int]  #: Entry block ids by predicate value (``None`` for a goto)
    entry_bindings: Dict[str, Binding]  #: Operator inputs of the CFG variables live at the entry
    output_examples: List[Any]  #: Example (fake) values of the flat outputs
    code_name: str = ""
    entry_constants: Dict[str, Any] = dataclasses.field(default_factory=dict)  #: Python integers for symbolic inputs
    #: The entry predicate as an expression over symbols (e.g., a comparison of sizes), instead of an operator input
    entry_condition: Any = None

    def block(self, block_id: int) -> BlockRecord:
        return self.blocks[block_id]


_REGISTRY: Dict[int, CfgRecord] = {}
_IDS = itertools.count()


def register(record_factory) -> CfgRecord:
    """Registers the record created by ``record_factory(id)`` and returns it."""
    record = record_factory(next(_IDS))
    _REGISTRY[record.id] = record
    return record


def lookup(cfg_id: int) -> CfgRecord:
    return _REGISTRY[cfg_id]


@torch.library.custom_op("dace::cfg", mutates_args=())
def cfg(cfg_id: int, tensors: List[torch.Tensor], symints: List[int]) -> List[torch.Tensor]:
    """A captured control-flow graph (see :mod:`.transport`). It only exists in graphs compiled by DaCe."""
    raise RuntimeError("dace::cfg only runs as part of a graph compiled with the DaCe backend")


@cfg.register_fake
def _cfg_fake(cfg_id: int, tensors: List[torch.Tensor], symints: List[int]) -> List[torch.Tensor]:
    record = lookup(cfg_id)
    return [torch.empty_strided(e.size(), e.stride(), dtype=e.dtype, device=e.device) for e in record.output_examples]
