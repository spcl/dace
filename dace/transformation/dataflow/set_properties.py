from collections import deque
from typing import Any
from dace.sdfg import SDFG
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.state import SDFGState
from dace.transformation import transformation
from dace.properties import DictProperty, make_properties, Property


@make_properties
class SetProperties(transformation.SingleStateTransformation):

    node = transformation.PatternNode(nodes.Node)

    ps = Property(dtype=dict, default={})

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.node)]

    def can_be_applied(self, state, expr_index, sdfg, permissive=False):
        return True

    def apply(self, state: SDFGState, sdfg: SDFG):
        for key, val in self.ps.items():
            pts = str(key).split('.')
            if len(pts) == 1:
                if hasattr(self.node, key):
                    setattr(self.node, key, val)
            else:
                pivot = self.node
                for k in pts[:-1]:
                    if hasattr(pivot, k):
                        pivot = getattr(pivot, k)
                if hasattr(pivot, pts[-1]):
                    setattr(pivot, pts[-1], val)
