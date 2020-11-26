# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import copy
from dace.symbolic import symstr
from dace.properties import Property
import dace.library as library
from dace.transformation.transformation import ExpandTransformation
from dace.sdfg.nodes import LibraryNode
from .. import environments


@library.expansion
class ExpandGerPure(ExpandTransformation):

    environments = []

    @staticmethod
    def make_sdfg(node, parent_state, parent_sdfg):
        sdfg = dace.SDFG(node.label + "_sdfg")
        raise NotImplementedError("NYI")
        return sdfg

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        if node.dtype is None:
            raise ValueError("Data type must be set to expand " + str(node) +
                             ".")
        return ExpandGerPure.make_sdfg(node, state, sdfg)


@library.node
class Ger(LibraryNode):

    # Global properties
    implementations = {
        "pure": ExpandGerPure,
    }
    default_implementation = None

    # Object fields
    dtype = dace.properties.TypeClassProperty(allow_none=True)
    transA = Property(dtype=bool,
                      desc="Whether to transpose A before multiplying")
    alpha = Property(
        dtype=tuple(dace.dtypes._CONSTANT_TYPES),
        default=1,
        desc="A scalar which will be multiplied with A @ x before adding y")
    beta = Property(dtype=tuple(dace.dtypes._CONSTANT_TYPES),
                    default=1,
                    desc="A scalar which will be multiplied with y")

    def __init__(self,
                 name,
                 dtype=None,
                 location=None,
                 alpha=1):
        super().__init__(name,
                         location=location,
                         inputs={"_x", "_y", "_a"},
                         outputs={"_res"})
        self.dtype = dtype
        self.alpha = alpha

    def validate(self, sdfg, state):
        raise NotImplementedError("NYI")
