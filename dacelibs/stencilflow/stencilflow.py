import dace.library
import dace.properties
import enum
import json
import typing


class Access:
    """ Representation of a single access to a given offset of a given field.
    """

    def __init__(self, field: str, offset: typing.List[int], alias: str):
        self.field = field
        self.offset = offset
        self.alias = alias

    def dims(self):
        return len(self.__offset__)

    @staticmethod
    def from_dict(d):
        field = d["field"]
        offset = list(map(int, d["offset"]))
        alias = d["alias"]
        return Access(field, offset, alias)

    def to_dict(self):
        d = {
            "field": self.field,
            "offset": self.offset,
            "alias": self.alias
        }
        return d

    @staticmethod
    def from_json(s):
        d = json.loads(s)
        return Access.from_dict(d)

    def to_json(self):
        d = self.to_dict()
        return json.dumps(d)


@dace.library.node
class Stencil(dace.library.LibraryNode):

    accesses = dace.properties.ListProperty(Access)
    code = dace.properties.CodeProperty()
    boundary_conditions = dace.properties.Property(dtype=dict)

    implementations = {}
    default_implementation = None

    def __init__(self, name, accesses, code, boundary_conditions):
        super().__init__(name)
        self.accesses = accesses
        self.code = code
        self.boundary_conditions = boundary_conditions

    @staticmethod
    def fromJSON_object(as_json, context=None):
        stencil = Stencil("", [], "", {})
        dace.properties.Property.set_properties_from_json(
            stencil, as_json, context=context)
        return stencil


@dace.library.library
class StencilFlow:

    nodes = [Stencil]
    transformations = []
