import dace.library
import dace.properties
import dace.sdfg.nodes
from dace import dtypes
from dace.symbolic import symstr
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from dace.codegen.targets.ipu_files import ipu_utils as ipu_utils


@dace.library.expansion
class ExpandMMPopLib(ExpandTransformation):

    environments = [environments.poplar.IPU]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        (adesc, bdesc, cdesc) = node.validate(parent_sdfg, parent_state)
        
        A_poplar_type = ipu_utils.TYPE_TO_IPU[adesc.dtype]
        B_poplar_type = ipu_utils.TYPE_TO_IPU[bdesc.dtype]
        C_poplar_type = ipu_utils.TYPE_TO_IPU[cdesc.dtype]
        

        init = f""" 
            {{
                {A_poplar_type} A = {node.A_scalar_param};
                Tensor B = {node.B_scalar_param};
                Tensor C = {node.C_scalar_param};
            }}
            // Add variables to the graph
            Tensor m1 = __state->graph.addVariable(FLOAT, {{900, 600}}, "m1");
            Tensor m2 = __state->graph.addVariable(FLOAT, {{600, 300}}, "m2");
            Tensor m3 = __state->graph.addVariable(FLOAT, {{300, 200}}, "m3");
            poputil::mapTensorLinearly(__state->graph, m1);
            poputil::mapTensorLinearly(__state->graph, m2);
            poputil::mapTensorLinearly(__state->graph, m3);
            Tensor m4 = poplin::matMul(__state->graph, m1, m2, __state->prog, "m4");
        """
        
        code = f"""
            {init}
        """
            
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dtypes.Language.CPP)
        return tasklet


@dace.library.node
class IPUMatMul(dace.sdfg.nodes.LibraryNode):
    """Executes poplin::matMul.
    """
    # Global properties
    implementations = {
        "MM": ExpandMMPopLib,
    }
    default_implementation = None
    
    A_scalar_param = dace.properties.Property(allow_none=False, default=0, desc="A scalar")
    B_scalar_param = dace.properties.Property(allow_none=False, default=0, desc="B scalar")
    C_scalar_param = dace.properties.Property(allow_none=False, default=0, desc="C scalar")
    
    def __init__(self, name, A_scalar_param, B_scalar_param, C_scalar_param):
        super().__init__(name, inputs={"_inbufferA", "_inbufferB"}, outputs={"_outbufferC"})
        self.A_scalar_param = A_scalar_param
        self.B_scalar_param = B_scalar_param
        self.C_scalar_param = C_scalar_param    

    def validate(self, sdfg, state):
        """
        :return: A three-tuple (buffer) of the three data descriptors in the
                 parent SDFG.
        """

        inbufferA, inbufferB, outbufferC = None, None, None
        for e in state.out_edges(self):
            if e.src_conn == "_outbufferC":
                outbufferC = sdfg.arrays[e.data.data]
        for e in state.in_edges(self):
            if e.dst_conn == "_inbufferA":
                inbufferA = sdfg.arrays[e.data.data]
            if e.dst_conn == "_inbufferB":
                inbufferB = sdfg.arrays[e.data.data]


        return (inbufferA, inbufferB, outbufferC)
