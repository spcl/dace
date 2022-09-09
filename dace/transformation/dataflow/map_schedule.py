import dace

from dace.sdfg import SDFG
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.state import SDFGState
from dace.transformation import transformation
from dace.properties import make_properties, Property, EnumProperty


@make_properties
class MapSchedule(transformation.SingleStateTransformation):
    map_entry = transformation.PatternNode(nodes.MapEntry)

    schedule_type = EnumProperty(dtype=dace.ScheduleType,
                                 default=dace.ScheduleType.Default,
                                 desc="Schedule type of the map")
    collapse = Property(dtype=int, default=1)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry)]

    def can_be_applied(self, state, expr_index, sdfg, permissive=False):
        return True

    def apply(self, state: SDFGState, sdfg: SDFG):
        self.map_entry.map.schedule = self.schedule_type
        self.map_entry.map.collapse = self.collapse
