from collections import deque
from typing import Any
from dace.sdfg import SDFG
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.state import SDFGState
from dace.transformation import transformation
from dace.properties import DictProperty, make_properties, Property


@make_properties
class SetArrayProperties(transformation.SingleStateTransformation):

    node = transformation.PatternNode(nodes.AccessNode)

    ps = Property(dtype=dict, default={})

    @staticmethod
    def annotates_memlets():
        return True

    @classmethod
    def expressions(cls):
        return [SDFG('_')]

    def can_be_applied(self, state, expr_index, sdfg, permissive=False):
        return True

    def apply(self, _, sdfg: SDFG):
        arr = sdfg._arrays[self.node.data]
        for key, val in self.ps.items():
            pts = str(key).split('.')
            if len(pts) == 1:
                if hasattr(arr, key):
                    setattr(arr, key, val)
            else:
                pivot = arr
                for k in pts[:-1]:
                    if hasattr(pivot, k):
                        pivot = getattr(pivot, k)
                if hasattr(pivot, pts[-1]):
                    setattr(pivot, pts[-1], val)
