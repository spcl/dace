import ast
from functools import reduce
import operator
from typing import List, Set

import dace
import dace.serialize
from dace import subsets, dtypes
from dace.frontend.operations import detect_reduction_type
from dace.frontend.python.astutils import unparse
from dace.properties import (Property, make_properties, DataProperty,
                             SubsetProperty, SymbolicProperty,
                             DebugInfoProperty, LambdaProperty)


@make_properties
class Memlet(object):
    """ Data movement object. Represents the data, the subset moved, and the
        manner it is reindexed (`other_subset`) into the destination.
        If there are multiple conflicting writes, this object also specifies
        how they are resolved with a lambda function.
    """

    # Properties
    veclen = Property(dtype=int, desc="Vector length", default=1)
    num_accesses = SymbolicProperty(default=0)
    subset = SubsetProperty(default=subsets.Range([]))
    other_subset = SubsetProperty(allow_none=True)
    data = DataProperty()
    debuginfo = DebugInfoProperty()
    wcr = LambdaProperty(allow_none=True)
    wcr_conflict = Property(dtype=bool, default=True)
    allow_oob = Property(dtype=bool,
                         default=False,
                         desc='Bypass out-of-bounds validation')

    def __init__(self,
                 data,
                 num_accesses,
                 subset,
                 vector_length,
                 wcr=None,
                 other_subset=None,
                 debuginfo=None,
                 wcr_conflict=True):
        """ Constructs a Memlet.
            :param data: The data object or name to access. B{Note:} this
                         parameter will soon be deprecated.
            @type data: Either a string of the data descriptor name or an
                        AccessNode.
            :param num_accesses: The number of times that the moved data
                                 will be subsequently accessed. If
                                 `dace.dtypes.DYNAMIC` (-1),
                                 designates that the number of accesses is
                                 unknown at compile time.
            :param subset: The subset of `data` that is going to be accessed.
            :param vector_length: The length of a single unit of access to
                                  the data (used for vectorization
                                  optimizations).
            :param wcr: A lambda function specifying how write-conflicts
                        are resolved. The syntax of the lambda function receives two elements: `current` value and `new` value,
                        and returns the value after resolution. For example,
                        summation is `lambda cur, new: cur + new`.
            :param other_subset: The reindexing of `subset` on the other
                                 connected data.
            :param debuginfo: Source-code information (e.g., line, file)
                              used for debugging.
            :param wcr_conflict: If False, forces non-locked conflict
                                 resolution when generating code. The default
                                 is to let the code generator infer this
                                 information from the SDFG.
        """

        # Properties
        self.num_accesses = num_accesses  # type: sympy.expr.Expr
        self.subset = subset  # type: subsets.Subset
        self.veclen = vector_length  # type: int
        if hasattr(data, 'data'):
            data = data.data
        self.data = data  # type: str

        # Annotates memlet with _how_ writing is performed in case of conflict
        self.wcr = wcr
        self.wcr_conflict = wcr_conflict

        # The subset of the other endpoint we are copying from/to (note:
        # carries the dimensionality of the other endpoint too!)
        self.other_subset = other_subset

        self.debuginfo = debuginfo

    def to_json(self, parent_graph=None):
        attrs = dace.serialize.all_properties_to_json(self)

        retdict = {"type": "Memlet", "attributes": attrs}

        return retdict

    @staticmethod
    def from_json(json_obj, context=None):
        if json_obj['type'] != "Memlet":
            raise TypeError("Invalid data type")

        # Create dummy object
        ret = Memlet("", dace.dtypes.DYNAMIC, None, 1)
        dace.serialize.set_properties_from_json(ret, json_obj, context=context)

        return ret

    @staticmethod
    def simple(data,
               subset_str,
               veclen=1,
               wcr_str=None,
               other_subset_str=None,
               wcr_conflict=True,
               num_accesses=None,
               debuginfo=None):
        """ Constructs a Memlet from string-based expressions.
            :param data: The data object or name to access. B{Note:} this
                         parameter will soon be deprecated.
            @type data: Either a string of the data descriptor name or an
                        AccessNode.
            :param subset_str: The subset of `data` that is going to
                               be accessed in string format. Example: '0:N'.
            :param veclen: The length of a single unit of access to
                           the data (used for vectorization optimizations).
            :param wcr_str: A lambda function (as a string) specifying
                            how write-conflicts are resolved. The syntax
                            of the lambda function receives two elements:
                            `current` value and `new` value,
                            and returns the value after resolution. For
                            example, summation is
                            `'lambda cur, new: cur + new'`.
            :param other_subset_str: The reindexing of `subset` on the other
                                     connected data (as a string).
            :param wcr_conflict: If False, forces non-locked conflict
                                 resolution when generating code. The default
                                 is to let the code generator infer this
                                 information from the SDFG.
            :param num_accesses: The number of times that the moved data
                                 will be subsequently accessed. If
                                 `dace.dtypes.DYNAMIC` (-1),
                                 designates that the number of accesses is
                                 unknown at compile time.
            :param debuginfo: Source-code information (e.g., line, file)
                              used for debugging.

        """
        subset = SubsetProperty.from_string(subset_str)
        if num_accesses is not None:
            na = num_accesses
        else:
            na = subset.num_elements()

        if wcr_str is not None:
            wcr = LambdaProperty.from_string(wcr_str)
        else:
            wcr = None

        if other_subset_str is not None:
            other_subset = SubsetProperty.from_string(other_subset_str)
        else:
            other_subset = None

        # If it is an access node or another memlet
        if hasattr(data, 'data'):
            data = data.data

        return Memlet(data,
                      na,
                      subset,
                      veclen,
                      wcr=wcr,
                      other_subset=other_subset,
                      wcr_conflict=wcr_conflict,
                      debuginfo=debuginfo)

    @staticmethod
    def from_array(dataname, datadesc, wcr=None):
        """ Constructs a Memlet that transfers an entire array's contents.
            :param dataname: The name of the data descriptor in the SDFG.
            :param datadesc: The data descriptor object.
            :param wcr: The conflict resolution lambda.
            @type datadesc: Data.
        """
        range = subsets.Range.from_array(datadesc)
        return Memlet(dataname, range.num_elements(), range, 1, wcr=wcr)

    def __hash__(self):
        return hash((self.data, self.num_accesses, self.subset, self.veclen,
                     str(self.wcr), self.other_subset))

    def __eq__(self, other):
        return all([
            self.data == other.data, self.num_accesses == other.num_accesses,
            self.subset == other.subset, self.veclen == other.veclen,
            self.wcr == other.wcr, self.other_subset == other.other_subset
        ])

    def num_elements(self):
        """ Returns the number of elements in the Memlet subset. """
        return self.subset.num_elements()

    def bounding_box_size(self):
        """ Returns a per-dimension upper bound on the maximum number of
            elements in each dimension.

            This bound will be tight in the case of Range.
        """
        return self.subset.bounding_box_size()

    def validate(self, sdfg, state):
        if self.data is not None and self.data not in sdfg.arrays:
            raise KeyError('Array "%s" not found in SDFG' % self.data)

    @property
    def free_symbols(self) -> Set[str]:
        """ Returns a set of symbols used in this edge's properties. """
        # Symbolic properties are in num_accesses, and the two subsets
        result = set()
        result |= set(map(str, self.num_accesses.free_symbols))
        if self.subset:
            result |= self.subset.free_symbols
        if self.other_subset:
            result |= self.other_subset.free_symbols
        return result

    def __label__(self, sdfg, state):
        """ Returns a string representation of the memlet for display in a
            graph.

            :param sdfg: The SDFG in which the memlet resides.
            :param state: An SDFGState object in which the memlet resides.
        """
        if self.data is None:
            return self._label(None)
        return self._label(sdfg.arrays[self.data].shape)

    def __str__(self):
        return self._label(None)

    def _label(self, shape):
        result = ''
        if self.data is not None:
            result = self.data

        if self.subset is None:
            return result

        num_elements = self.subset.num_elements()
        if self.num_accesses != num_elements:
            if self.num_accesses == -1:
                result += '(dyn) '
            else:
                result += '(%s) ' % SymbolicProperty.to_string(
                    self.num_accesses)
        arrayNotation = True
        try:
            if shape is not None and reduce(operator.mul, shape, 1) == 1:
                # Don't mention array if we're accessing a single element and it's zero
                if all(s == 0 for s in self.subset.min_element()):
                    arrayNotation = False
        except TypeError:
            # Will fail if trying to check the truth value of a sympy expr
            pass
        if arrayNotation:
            result += '[%s]' % str(self.subset)
        if self.wcr is not None and str(self.wcr) != '':
            # Autodetect reduction type
            redtype = detect_reduction_type(self.wcr)
            if redtype == dtypes.ReductionType.Custom:
                wcrstr = unparse(ast.parse(self.wcr).body[0].value.body)
            else:
                wcrstr = str(redtype)
                wcrstr = wcrstr[wcrstr.find('.') + 1:]  # Skip "ReductionType."

            result += ' (CR: %s)' % wcrstr

        if self.other_subset is not None:
            result += ' -> [%s]' % str(self.other_subset)
        return result

    def __repr__(self):
        return "Memlet (" + self.__str__() + ")"


class EmptyMemlet(Memlet):
    """ A memlet without data. Primarily used for connecting nodes to scopes
        without transferring data to them. """
    def __init__(self):
        super(EmptyMemlet, self).__init__(None, 0, None, 1)


class MemletTree(object):
    """ A tree of memlet edges.

        Since memlets can form paths through scope nodes, and since these
        paths can split in "OUT_*" connectors, a memlet edge can be extended
        to a memlet tree. The tree is always rooted at the outermost-scope node,
        which can mean that it forms a tree of directed edges going forward
        (in the case where memlets go through scope-entry nodes) or backward
        (through scope-exit nodes).

        Memlet trees can be used to obtain all edges pertaining to a single
        memlet using the `memlet_tree` function in SDFGState. This collects
        all siblings of the same edge and their children, for instance if
        multiple inputs from the same access node are used.
    """
    def __init__(
        self,
        edge,
        parent=None,
        children=None
    ):  # type: (dace.sdfg.graph.MultiConnectorEdge, MemletTree, List[MemletTree]) -> None
        self.edge = edge
        self.parent = parent
        self.children = children or []

    def __iter__(self):
        if self.parent is not None:
            yield from self.parent
            return

        def traverse(node):
            yield node.edge
            for child in node.children:
                yield from traverse(child)

        yield from traverse(self)

    def root(self) -> 'MemletTree':
        node = self
        while node.parent is not None:
            node = node.parent
        return node

    def traverse_children(self, include_self=False):
        if include_self:
            yield self
        for child in self.children:
            yield from child.traverse_children(include_self=True)
