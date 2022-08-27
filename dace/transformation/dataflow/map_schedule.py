import dace

from dace.sdfg import SDFG
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.state import SDFGState
from dace.transformation import transformation
from dace.properties import make_properties, Property


@make_properties
class MapSchedule(transformation.SingleStateTransformation):
    map_entry = transformation.PatternNode(nodes.MapEntry)

    schedule_type = Property(dtype=str, default="Sequential", allow_none=False)
    collapse = Property(dtype=int, default=0)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry)]

    def can_be_applied(self, state, expr_index, sdfg, permissive=False):
        return True

    def apply(self, state: SDFGState, sdfg: SDFG):
        if self.schedule_type == "ScheduleType.Sequential":
            self.map_entry.map.schedule = dace.ScheduleType.Sequential
            self.map_entry.map.collapse = self.collapse
        elif self.schedule_type == "ScheduleType.CPU_Multicore":
            self.map_entry.map.schedule = dace.ScheduleType.CPU_Multicore
            self.map_entry.map.collapse = self.collapse
