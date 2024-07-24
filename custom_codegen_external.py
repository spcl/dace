import dace
from dace import registry
from dace.sdfg.scope import ScopeSubgraphView
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.targets.target import TargetCodeGenerator
from dace.codegen.targets.framecode import DaCeCodeGenerator
from dace.codegen.targets.cpp import sym2cpp

@dace.program
def custom_kernel(A: dace.float64[20, 30]):
    for i, j in dace.map[0:20:2, 0:30]:
        A[i, j] += A[i, j]



dace.ScheduleType.register('LoopyLoop')
dace.SCOPEDEFAULT_SCHEDULE[dace.ScheduleType.LoopyLoop] = dace.ScheduleType.Sequential
dace.SCOPEDEFAULT_STORAGE[dace.ScheduleType.LoopyLoop] = dace.StorageType.CPU_Heap


@registry.autoregister_params(name='loopy')
class MyCustomLoop(TargetCodeGenerator):
    def __init__(self, frame_codegen: DaCeCodeGenerator, sdfg: dace.SDFG):
        ################################################################
        # Define some locals:
        # Can be used to call back to the frame-code generator
        self.frame = frame_codegen
        # Can be used to dispatch other code generators for allocation/nodes
        self.dispatcher = frame_codegen.dispatcher
        
        ################################################################
        # Register handlers/hooks through dispatcher: Can be used for 
        # nodes, memory copy/allocation, scopes, states, and more.
        
        # In this case, register scopes
        self.dispatcher.register_map_dispatcher(dace.ScheduleType.LoopyLoop, self)
        
        # You can similarly use register_{array,copy,node,state}_dispatcher
        
    # A scope dispatcher will trigger a method called generate_scope whenever 
    # an SDFG has a scope with that schedule
    def generate_scope(self, sdfg: dace.SDFG, scope: ScopeSubgraphView,
                       state_id: int, function_stream: CodeIOStream,
                       callsite_stream: CodeIOStream):
        # The parameters here are:
        # sdfg: The SDFG we are currently generating.
        # scope: The subgraph of the state containing only the scope (map contents)
        #        we want to generate the code for.
        # state_id: The state in the SDFG the subgraph is taken from (i.e., 
        #           `sdfg.node(state_id)` is the same as `scope.graph`)
        # function_stream: A cursor to the global code (which can be used to define
        #                  functions, hence the name).
        # callsite_stream: A cursor to the current location in the code, most of
        #                  the code is generated here.
        
        # We can get the map entry node from the scope graph
        entry_node = scope.source_nodes()[0]
        
        # First, generate an opening brace (for instrumentation and dynamic map ranges)
        callsite_stream.write('{', sdfg, state_id, entry_node)
        
        ################################################################
        # Generate specific code: We will generate a reversed loop with a 
        # comment for each dimension of the map. For the sake of simplicity,
        # dynamic map ranges are not supported.
        
        for param, rng in zip(entry_node.map.params, entry_node.map.range):
            # We use the sym2cpp function from the cpp support functions
            # to convert symbolic expressions to proper C++
            begin, end, stride = (sym2cpp(r) for r in rng)
            
            # Every write is optionally (but recommended to be) tagged with
            # 1-3 extra arguments, serving as line information to match
            # SDFG, state, and graph nodes/edges to written code.
            callsite_stream.write(f'''// Loopy-loop {param}
            for (auto {param} = {end}; {param} >= {begin}; {param} -= {stride}) {{''',
                                  sdfg, state_id, entry_node
            )
        
            # NOTE: CodeIOStream will automatically take care of indentation for us.
        
        
        # Now that the loops have been defined, use the dispatcher to invoke any
        # code generator (including this one) that is registered to deal with
        # the internal nodes in the subgraph. We skip the MapEntry node.
        self.dispatcher.dispatch_subgraph(sdfg, scope, state_id,
                                          function_stream, callsite_stream,
                                          skip_entry_node=True)
        
        # NOTE: Since skip_exit_node above is set to False, closing braces will
        #       be automatically generated

# Preview SDFG
sdfg = custom_kernel.to_sdfg()

# Change schedule
for node, _ in sdfg.all_nodes_recursive():
    if isinstance(node, dace.nodes.MapEntry):
        node.schedule = dace.ScheduleType.LoopyLoop

# Code(sdfg.generate_code()[0].clean_code, language='cpp')


# display
from IPython.display import Code
from IPython.display import display
display(Code(sdfg.generate_code()[0].clean_code, language='cpp'))
