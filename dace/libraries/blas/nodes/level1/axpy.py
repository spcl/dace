import dace.library
import dace.properties
import dace.graph.nodes
from dace.transformation.pattern_matching import ExpandTransformation
from ... import environments


@dace.library.expansion
class Expand_AXPY_Pure(ExpandTransformation):

    environments = []

    @staticmethod
    def make_sdfg(dtype):

        n = dace.symbol("n")

        # name graph and state
        vecAdd_sdfg = dace.SDFG('vecAdd_graph')
        vecAdd_state = vecAdd_sdfg.add_state()

        # add data descriptors
        vecAdd_sdfg.add_array('_x', shape=[n], dtype=dtype)
        vecAdd_sdfg.add_array('_y', shape=[n], dtype=dtype)
        vecAdd_sdfg.add_scalar('_a', dtype=dtype)
        vecAdd_sdfg.add_array('_res', shape=[n], dtype=dtype)

        x_in = vecAdd_state.add_read('_x')
        y_in = vecAdd_state.add_read('_y')
        a_in = vecAdd_state.add_read('_a')
        z_out = vecAdd_state.add_write('_res')

        vecMap_entry, vecMap_exit = vecAdd_state.add_map(
            'vecAdd_map',
            dict(i='0:n')   
        )

        vecAdd_tasklet = vecAdd_state.add_tasklet(
            'vecAdd_task',
            ['x_con', 'y_con', 'a_con'],
            ['z_con'],
            '''
z_con = a_con * x_con + y_con
            '''
        )

        vecAdd_state.add_memlet_path(
            x_in, vecMap_entry, vecAdd_tasklet,
            dst_conn='x_con',
            memlet=dace.Memlet.simple(x_in.data, 'i')
        )

        vecAdd_state.add_memlet_path(
            y_in, vecMap_entry, vecAdd_tasklet,
            dst_conn='y_con',
            memlet=dace.Memlet.simple(y_in.data, 'i')
        )

        vecAdd_state.add_memlet_path(
            a_in, vecMap_entry, vecAdd_tasklet,
            dst_conn='a_con',
            memlet=dace.Memlet.simple(a_in.data, '0')
        )

        vecAdd_state.add_memlet_path(
            vecAdd_tasklet, vecMap_exit, z_out,
            src_conn='z_con',
            memlet=dace.Memlet.simple(z_out.data, 'i')
        )

        return vecAdd_sdfg

    @staticmethod
    def expansion(node, state, sdfg): #TODO: ask Johannes what this is how to enforce single prec here
        node.validate(sdfg, state)
        if node.dtype is None:
            raise ValueError("Data type must be set to expand " + str(node) +
                             ".")
        return Expand_AXPY_Pure.make_sdfg(node.dtype)


@dace.library.node
class Axpy(dace.graph.nodes.LibraryNode):

    # Global properties
    implementations = {
        "pure": Expand_AXPY_Pure
    }
    default_implementation = 'pure'

    # Object fields
    # TODO: ask what this is
    dtype = dace.properties.TypeClassProperty(allow_none=True)

    def __init__(self, name, dtype=dace.float32, *args, **kwargs):
        super().__init__(
            name, *args, inputs={"_x", "_y", "_a"}, outputs={"_res"}, **kwargs
        )
        self.dtype = dtype

    def validate(self, sdfg, state):
        
        # TODO: implement validation
        return True
