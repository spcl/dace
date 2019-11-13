from dace import data
from dace.graph import dot
from dace.graph.nodes import CodeNode
from dace.properties import Property, make_properties


@make_properties
class BlasNode(CodeNode):
    ''' A node that represents a basic linear-algebra operation. '''

    label = Property(dtype=str, desc="Blas op name")

    def __init__(self, label, inputs=None, outputs=None):
        self._label = label
        super(BlasNode, self).__init__(inputs or set(), outputs or set())

    @property
    def name(self):
        return self._label

    def draw_node(self, sdfg, graph):
        return dot.draw_node(sdfg, graph, self, shape="octagon")

    def validate(self, sdfg, state):
        if not data.validate_name(self.label):
            raise NameError('Invalid tasklet name "%s"' % self.label)
        for in_conn in self.in_connectors:
            if not data.validate_name(in_conn):
                raise NameError('Invalid input connector "%s"' % in_conn)
        for out_conn in self.out_connectors:
            if not data.validate_name(out_conn):
                raise NameError('Invalid output connector "%s"' % out_conn)

    def __str__(self):
        if not self.label:
            return "--Empty--"
        else:
            return self.label


class CopyNode(BlasNode):

    def __init__(self, inputs=None, outputs=None):
        super(CopyNode, self).__init__('copy', inputs, outputs)


class ArrayScalarMulNode(BlasNode):

    def __init__(self, inputs=None, outputs=None):
        super(ArrayScalarMulNode, self).__init__('arrayscalarmul', inputs, outputs)


class AddNode(BlasNode):

    def __init__(self, inputs=None, outputs=None):
        super(AddNode, self).__init__('add', inputs, outputs)


class MatMulNode(BlasNode):

    def __init__(self, inputs=None, outputs=None):
        super(MatMulNode, self).__init__('matmul', inputs, outputs)
