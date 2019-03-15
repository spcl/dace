import ast
from functools import reduce
import operator
import copy as cp

import dace
from dace import data as dt, subsets, symbolic, types
from dace.frontend.operations import detect_reduction_type
from dace.frontend.python.astutils import unparse
from dace.properties import (
    Property, make_properties, DataProperty, ShapeProperty, SubsetProperty,
    SymbolicProperty, TypeClassProperty, DebugInfoProperty, LambdaProperty)


@make_properties
class Memlet(object):
    """ Data movement object. Represents the data, the subset moved, and the
        manner it is reindexed (`other_subset`) into the destination.
        If there are multiple conflicting writes, this object also specifies
        how they are resolved with a lambda function.
    """

    # Properties
    veclen = Property(dtype=int, desc="Vector length")
    num_accesses = SymbolicProperty()
    subset = SubsetProperty()
    other_subset = SubsetProperty(allow_none=True)
    data = DataProperty()
    debuginfo = DebugInfoProperty()
    wcr = LambdaProperty(allow_none=True)
    wcr_identity = Property(dtype=object, default=None, allow_none=True)
    wcr_conflict = Property(dtype=bool, default=True)

    def __init__(self,
                 data,
                 num_accesses,
                 subset,
                 vector_length,
                 wcr=None,
                 wcr_identity=None,
                 other_subset=None,
                 debuginfo=None,
                 wcr_conflict=True):
        """ Constructs a Memlet.
            @param data: The data object or name to access. B{Note:} this
                         parameter will soon be deprecated.
            @type data: Either a string of the data descriptor name or an
                        AccessNode.
            @param num_accesses: The number of times that the moved data
                                 will be subsequently accessed. If
                                 `dace.types.DYNAMIC` (-1),
                                 designates that the number of accesses is
                                 unknown at compile time.
            @param subset: The subset of `data` that is going to be accessed.
            @param vector_length: The length of a single unit of access to
                                  the data (used for vectorization 
                                  optimizations).
            @param wcr: A lambda function specifying how write-conflicts
                        are resolved. The syntax of the lambda function receives two elements: `current` value and `new` value,
                        and returns the value after resolution. For example,
                        summation is `lambda cur, new: cur + new`.
            @param wcr_identity: Identity value used for the first write 
                                 conflict. B{Note:} this parameter will soon
                                 be deprecated.
            @param other_subset: The reindexing of `subset` on the other 
                                 connected data.
            @param debuginfo: Source-code information (e.g., line, file) 
                              used for debugging.
            @param wcr_conflict: If False, forces non-locked conflict 
                                 resolution when generating code. The default
                                 is to let the code generator infer this 
                                 information from the SDFG.
        """

        # Properties
        self.num_accesses = num_accesses  # type: sympy math
        self.subset = subset  # type: subsets.Subset
        self.veclen = vector_length  # type: int (in elements, default 1)
        if hasattr(data, 'data'):
            data = data.data
        self.data = data  # type: str

        # Annotates memlet with _how_ writing is performed in case of conflict
        self.wcr = wcr
        self.wcr_identity = wcr_identity
        self.wcr_conflict = wcr_conflict

        # The subset of the other endpoint we are copying from/to (note:
        # carries the dimensionality of the other endpoint too!)
        self.other_subset = other_subset

        self.debuginfo = debuginfo

    def toJSON(self, indent=0):
        json = " " * indent + "{\n"
        indent += 2
        json += " " * indent + "\"type\" : \"" + type(self).__name__ + "\",\n"
        json += " " * indent + "\"label\" : \"" + str(self) + "\"\n"
        indent -= 2
        json += " " * indent + "}\n"
        return json

    @staticmethod
    def simple(data,
               subset_str,
               veclen=1,
               wcr_str=None,
               wcr_identity=None,
               other_subset_str=None,
               wcr_conflict=True,
               num_accesses=None,
               debuginfo=None):
        """ Constructs a Memlet from string-based expressions.
            @param data: The data object or name to access. B{Note:} this
                         parameter will soon be deprecated.
            @type data: Either a string of the data descriptor name or an
                        AccessNode.
            @param subset_str: The subset of `data` that is going to 
                               be accessed in string format. Example: '0:N'.
            @param veclen: The length of a single unit of access to
                           the data (used for vectorization optimizations).
            @param wcr_str: A lambda function (as a string) specifying 
                            how write-conflicts are resolved. The syntax 
                            of the lambda function receives two elements:
                            `current` value and `new` value,
                            and returns the value after resolution. For 
                            example, summation is 
                            `'lambda cur, new: cur + new'`.
            @param wcr_identity: Identity value used for the first write 
                                 conflict. B{Note:} this parameter will soon
                                 be deprecated.
            @param other_subset_str: The reindexing of `subset` on the other 
                                     connected data (as a string).
            @param wcr_conflict: If False, forces non-locked conflict 
                                 resolution when generating code. The default
                                 is to let the code generator infer this 
                                 information from the SDFG.
            @param num_accesses: The number of times that the moved data
                                 will be subsequently accessed. If
                                 `dace.types.DYNAMIC` (-1),
                                 designates that the number of accesses is
                                 unknown at compile time.
            @param debuginfo: Source-code information (e.g., line, file) 
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

        return Memlet(
            data,
            na,
            subset,
            veclen,
            wcr=wcr,
            wcr_identity=wcr_identity,
            other_subset=other_subset,
            wcr_conflict=wcr_conflict,
            debuginfo=debuginfo)

    @staticmethod
    def from_array(dataname, datadesc):
        """ Constructs a Memlet that transfers an entire array's contents.
            @param dataname: The name of the data descriptor in the SDFG.
            @param datadesc: The data descriptor object.
            @type datadesc: Data.
        """
        range = subsets.Range.from_array(datadesc)
        return Memlet(dataname, range.num_elements(), range, 1)

    def __hash__(self):
        return hash((self.data, self.num_accesses, self.subset, self.veclen,
                     str(self.wcr), self.wcr_identity, self.other_subset))

    def __eq__(self, other):
        return all([
            self.data == other.data, self.num_accesses == other.num_accesses,
            self.subset == other.subset, self.veclen == other.veclen,
            self.wcr == other.wcr, self.wcr_identity == other.wcr_identity,
            self.other_subset == other.other_subset
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
        if self.data not in sdfg.arrays:
            raise KeyError('Array "%s" not found in SDFG' % self.data)

    def __label__(self, sdfg, state):
        """ Returns a string representation of the memlet for display in a 
            graph.

            @param sdfg: The SDFG in which the memlet resides.
            @param state: An SDFGState object in which the memlet resides.
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
            result += '(%s) ' % str(self.num_accesses)
        arrayNotation = True
        try:
            if shape is not None and reduce(operator.mul, shape, 1) == 1:
                # Don't draw array if we're accessing a single element
                arrayNotation = False
        except TypeError:
            # Will fail if trying to check the truth value of a sympy expr
            pass
        if arrayNotation:
            result += '[%s]' % str(self.subset)
        if self.wcr is not None and str(self.wcr) != '':
            # Autodetect reduction type
            redtype = detect_reduction_type(self.wcr)
            if redtype == types.ReductionType.Custom:
                wcrstr = unparse(ast.parse(self.wcr).body[0].value.body)
            else:
                wcrstr = str(redtype)
                wcrstr = wcrstr[wcrstr.find('.') + 1:]  # Skip "ReductionType."

            result += ' (CR: %s' % wcrstr
            if self.wcr_identity is not None:
                result += ', id: %s' % str(self.wcr_identity)
            result += ')'

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
