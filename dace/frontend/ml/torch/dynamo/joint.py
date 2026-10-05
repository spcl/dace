# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Joint forward/backward compilation of training graphs.

AOTAutograd traces one *joint* graph per training graph: primals and output gradients (tangents) in, forward outputs
and input gradients out. A partitioner then decides which values the forward saves for the backward (the *cut*).
Instead of compiling the two partitioned graphs separately, the DaCe backend lowers the joint graph into one SDFG with
two phases (``if aot_phase == 0: forward else: backward``), see :meth:`.importer.GraphImporter.import_joint`.

A :class:`JointPlan` describes one such graph. The phases are derived from the joint graph and the cut alone: the
forward phase computes everything the forward outputs and the saved values need from the primals, and the backward
phase computes the gradients from the saved values and the tangents (recomputing anything else it needs). The calling
conventions of the two phases follow AOTAutograd's partitioned graphs, which the runtime calls.
"""
import dataclasses
from typing import Dict, List, Optional, Sequence, Set

import torch.fx

#: Name of the SDFG symbol that selects the phase of a joint SDFG
PHASE_SYMBOL = 'aot_phase'
FORWARD, BACKWARD = 0, 1


@dataclasses.dataclass
class JointPlan:
    """A training graph to compile into one SDFG."""
    joint: torch.fx.GraphModule  #: The joint graph (placeholders: primals, then tangents)
    forward: torch.fx.GraphModule  #: The partitioned forward graph (defines the forward calling convention)
    backward: torch.fx.GraphModule  #: The partitioned backward graph (defines the backward calling convention)
    num_fwd_outputs: int  #: Number of joint outputs that are forward outputs (the rest are gradients)
    saved: List[str]  #: Names of the joint nodes passed from the forward to the backward phase (the cut)

    def nodes(self) -> Dict[str, torch.fx.Node]:
        return {n.name: n for n in self.joint.graph.nodes}

    def forward_nodes(self) -> Set[torch.fx.Node]:
        """The joint nodes the forward phase computes: ancestors of the forward outputs and the saved values."""
        nodes = self.nodes()
        roots = [nodes[n] for n in _output_names(self.forward)]
        return _ancestors(roots, stop=set())

    def backward_nodes(self) -> Set[torch.fx.Node]:
        """The joint nodes the backward phase computes: ancestors of the gradients, up to (excluding) saved values."""
        nodes = self.nodes()
        stop = {nodes[n] for n in self.saved}
        roots = [r for r in flat_outputs(self.joint)[self.num_fwd_outputs:] if isinstance(r, torch.fx.Node)]
        return _ancestors(roots, stop)


def plan_from_partition(joint: torch.fx.GraphModule, forward: torch.fx.GraphModule, backward: torch.fx.GraphModule,
                        num_fwd_outputs: int) -> Optional[JointPlan]:
    """
    Builds the plan of a partitioned joint graph, or returns ``None`` if the partitioned graphs cannot be expressed
    in terms of the joint graph (e.g., the partitioner introduced operators of its own).
    """
    names = {n.name for n in joint.graph.nodes}
    for gm in (forward, backward):
        for node in gm.graph.nodes:
            if node.op in ('placeholder', 'call_function') and node.name not in names:
                return None
    saved = [n.name for n in backward.graph.nodes if n.op == 'placeholder' and not _is_tangent(n)]
    # Every saved value is a primal or returned by the forward graph
    available = set(_output_names(forward)) | {n.name for n in joint.graph.nodes if n.op == 'placeholder'}
    if any(name not in available for name in saved):
        return None
    return JointPlan(joint, forward, backward, num_fwd_outputs, saved)


def placeholder_names(gm: torch.fx.GraphModule) -> List[str]:
    return [n.name for n in gm.graph.nodes if n.op == 'placeholder']


def _is_tangent(node: torch.fx.Node) -> bool:
    return node.op == 'placeholder' and node.name.startswith('tangents')


def flat_outputs(gm: torch.fx.GraphModule) -> List:
    """The values (nodes or constants) a graph returns."""
    output = next(n for n in gm.graph.nodes if n.op == 'output')
    values = output.args[0]
    return list(values) if isinstance(values, (list, tuple)) else [values]


def _output_names(gm: torch.fx.GraphModule) -> List[str]:
    return [v.name for v in flat_outputs(gm) if isinstance(v, torch.fx.Node)]


def _ancestors(roots: Sequence[torch.fx.Node], stop: Set[torch.fx.Node]) -> Set[torch.fx.Node]:
    """``roots`` and their transitive inputs, not crossing (and not including) ``stop``."""
    result: Set[torch.fx.Node] = set()
    work = list(roots)
    while work:
        node = work.pop()
        if node in result or node in stop:
            continue
        result.add(node)
        work.extend(node.all_input_nodes)
    return result
