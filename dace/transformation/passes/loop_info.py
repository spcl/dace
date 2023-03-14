from dace import sdfg
from dace.codegen import control_flow
from dace.transformation import helpers
from typing import cast, Dict, List, Optional

class LoopInfo:

    def __init__(self, cf_scope: control_flow.ForScope, parent_loop: Optional["LoopInfo"] = None):

        self._cf_scope = cf_scope
        self._loop_guard = cf_scope.guard
        self._parent_loop = parent_loop
        self._body_states: List[sdfg.SDFGState] = []
        self._nested_loops: List[LoopInfo] = []

    def add_nested_loop(self, loop: "LoopInfo"):
        self._nested_loops.append(loop)

    def add_state(self, state: sdfg.SDFGState):
        self._body_states.append(state)

    @property
    def cf_scope(self) -> control_flow.ForScope:
        return self._cf_scope

    @property
    def guard(self) -> sdfg.SDFGState:
        return self._loop_guard

    @property
    def name(self) -> str:
        return self._loop_guard.label

    @property
    def body(self) -> List[sdfg.SDFGState]:
        return self._body_states

    @property
    def parent_loop(self) -> Optional["LoopInfo"]:
        return self._parent_loop

    @property
    def is_nested(self) -> bool:
        return self._parent_loop is not None

    @property
    def nested_loops(self) -> List["LoopInfo"]:
        return self._nested_loops

    @staticmethod
    def from_loop(loop: control_flow.ForScope) -> List["LoopInfo"]:

        loops: List[LoopInfo] = []

        def process_state(state: sdfg.SDFGState, loop: LoopInfo):
            loop.add_state(state=state)

        def process_if(scope: control_flow.IfScope, loop: LoopInfo):

            process_state(scope.branch_state, loop)
            for state in scope.body.elements:
                process_scope(state, loop)

        def process_for_loop(scope: control_flow.ForScope, loop: Optional[LoopInfo]):

            nonlocal loops

            newloop = LoopInfo(cf_scope=scope, parent_loop=loop)
            if loop is not None:
                loop.add_nested_loop(newloop)
            loops.append(newloop)

            for body_state in scope.body.elements:
                process_scope(body_state, newloop)

        def process_scope(scope: control_flow.ControlFlow, loop: Optional[LoopInfo]):

            match type(scope):
                case control_flow.ForScope:
                    process_for_loop(cast(control_flow.ForScope, scope), loop)
                case control_flow.IfScope:
                    process_if(cast(control_flow.IfScope, scope), loop)
                case control_flow.SingleState:
                    process_state(cast(control_flow.SingleState, scope).state, loop)

        process_scope(loop, None)

        return loops

class Loops:

    def __init__(self):
        self._loops = []
        self._states_to_loops: Dict[sdfg.SDFGState, Optional[LoopInfo]] = {}

    @property
    def loops(self) -> List[LoopInfo]:
        return self._loops

    def state_inside_loop(self, state: sdfg.SDFGState) -> Optional[LoopInfo]:
        return self._states_to_loops[state] if state in self._states_to_loops else None

    @staticmethod
    def from_sdfg(sdfg: sdfg.SDFG) -> "Loops":

        loops = Loops()

        sdfg_loops = helpers.find_sdfg_control_flow(sdfg)
        for _, v in sdfg_loops.items():

            cf_scope = v[1]
            # FIXME: support other loop types
            if isinstance(cf_scope, control_flow.ForScope):
                loops._loops.extend(LoopInfo.from_loop(cf_scope))

        for loop in loops._loops:
            for loop_state in loop.body:
                loops._states_to_loops[loop_state] = loop

        return loops

