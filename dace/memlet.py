import ast
from functools import reduce
import operator
from typing import List, Set, Union
import warnings

import dace
import dace.serialize
from dace import subsets, dtypes, symbolic
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
    volume = SymbolicProperty(default=0,
                              desc='The exact number of elements moved '
                              'using this memlet, or the maximum number '
                              'if dynamic=True (with 0 as unbounded)')
    dynamic = Property(default=False,
                       desc='Is the number of elements moved determined at '
                       'runtime (e.g., data dependent)')
    src_subset = SubsetProperty(allow_none=True,
                                desc='Subset of elements to move from the '
                                'source of this edge. If None, all elements '
                                'participate')
    dst_subset = SubsetProperty(allow_none=True,
                                desc='Subset of elements to move to the '
                                'destination of this edge. If None, all '
                                'elements participate')
    wcr = LambdaProperty(allow_none=True,
                         desc='If set, defines a write-conflict resolution '
                         'lambda function. The syntax of the lambda function '
                         'receives two elements: `current` value and `new` '
                         'value, and returns the value after resolution')

    # Code generation and validation hints
    debuginfo = DebugInfoProperty(desc='Line information to track source and '
                                  'generated code')
    wcr_nonatomic = Property(dtype=bool,
                             default=False,
                             desc='If True, always generates non-conflicting '
                             '(non-atomic) writes in resulting code')
    allow_oob = Property(dtype=bool,
                         default=False,
                         desc='Bypass out-of-bounds validation')

    def __init__(self,
                 expr: str = None,
                 src_subset: Union[str, subsets.Subset] = None,
                 dst_subset: Union[str, subsets.Subset] = None,
                 volume: Union[int, str, symbolic.SymbolicType] = None,
                 dynamic: bool = False,
                 wcr: Union[str, ast.AST] = None,
                 debuginfo: dtypes.DebugInfo = None,
                 wcr_nonatomic: bool = False,
                 allow_oob: bool = False):
        """ 
        Constructs a Memlet.
        :param expr: A string expression of the this memlet, given as an ease
                     of use API. Must follow one of the following forms:
                     1. ``SRC_SUBSET -> DST_SUBSET``,
                     2. ``ARRAY[SUBSET]``,
                     3. `ARRAY[SUBSET] -> OTHER_SUBSET``.
                     Note that modes 2 and 3 are deprecated and will leave 
                     the memlet uninitialized until inserted into an SDFG.
        :param src_subset: The subset to take from the source of the edge,
                           represented either as a string or a Subset object.
        :param dst_subset: The subset to offset into the destination of the
                           edge, represented either as a string or a Subset 
                           object.
        :param volume: The exact number of elements moved using this
                       memlet, or the maximum number of elements if
                       ``dynamic`` is set to True. If dynamic and this
                       value is set to zero, the number of elements moved
                       is runtime-defined and unbounded.
        :param dynamic: If True, the number of elements moved in this memlet
                        is defined dynamically at runtime.
        :param wcr: A lambda function (represented as a string or Python AST) 
                    specifying how write-conflicts are resolved. The syntax
                    of the lambda function receives two elements: ``current`` 
                    value and `new` value, and returns the value after 
                    resolution. For example, summation is represented by
                    ``'lambda cur, new: cur + new'``.
        :param debuginfo: Line information from the generating source code.
        :param wcr_nonatomic: If True, overrides the automatic code generator 
                              decision and treat all write-conflict resolution
                              operations as non-atomic, which might cause race
                              conditions in the general case.
        :param allow_oob: If True, bypasses the checks in SDFG validation for
                          out-of-bounds accesses in memlet subsets.
        """
        self._initialized = False

        # Deprecated fields kept for API and ease of use
        self._data = None
        self._subset = None
        self._other_subset = None
        self._is_data_src = None
        # Will be set once memlet is added into an SDFG (in try_initialize)
        self._sdfg = None
        self._state = None
        self._edge = None

        self.src_subset = None
        self.dst_subset = None

        # Initialize first by string expression
        if expr is not None:
            self._parse_memlet_from_str(expr)

        # Set properties
        self.src_subset = self.src_subset or src_subset
        self.dst_subset = self.dst_subset or dst_subset

        if volume is not None:
            self.volume = volume
        else:
            if self.src_subset is not None:
                self.volume = self.src_subset.num_elements()
            elif self.dst_subset is not None:
                self.volume = self.dst_subset.num_elements()
            else:
                self.volume = 1

        self.dynamic = dynamic
        self.wcr = wcr
        self.wcr_nonatomic = wcr_nonatomic
        self.debuginfo = debuginfo

        # If the new fields are set, mark memlet as initialized
        if (self._data is None and self._subset is None
                and self._other_subset is None):
            self._initialized = True

    def to_json(self):
        if not self._initialized:
            warnings.warn('Saving an uninitialized memlet')

        attrs = dace.serialize.all_properties_to_json(self)

        # Fill in legacy values
        attrs['data'] = self.data
        attrs['subset'] = self.subset
        attrs['other_subset'] = self.other_subset

        return {
            "type": "Memlet",
            "initialized": self._initialized,
            "attributes": attrs
        }

    @staticmethod
    def from_json(json_obj, context=None):
        ret = Memlet()
        dace.serialize.set_properties_from_json(ret, json_obj, context=context)
        if context:
            ret._sdfg = context['sdfg']
            ret._state = context['state']
            ret._edge = context['edge']
        return ret

    def is_empty(self) -> bool:
        """ 
        Returns True if this memlet carries no data. Memlets without data are
        primarily used for connecting nodes to scopes without transferring 
        data to them. 
        """
        return (self.data is None and self.src_subset is None
                and self.dst_subset is None)

    @property
    def num_accesses(self):
        """ 
        Returns the total memory movement volume (in elements) of this memlet.
        """
        return self.volume

    @num_accesses.setter
    def num_accesses(self, value):
        self.volume = value

    @staticmethod
    def simple(data,
               subset_str,
               veclen=1,
               wcr_str=None,
               other_subset_str=None,
               wcr_conflict=True,
               num_accesses=None,
               debuginfo=None):
        """ DEPRECATED: Constructs a Memlet from string-based expressions.
            :param data: The data object or name to access. 
            :type data: Either a string of the data descriptor name or an
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
                                 -1, designates that the number of accesses is
                                 unknown at compile time.
            :param debuginfo: Source-code information (e.g., line, file)
                              used for debugging.

        """
        warnings.warn(
            'This function is deprecated, please use the Memlet '
            'constructor instead', DeprecationWarning)

        result = Memlet()

        if isinstance(subset_str, subsets.Subset):
            result._subset = subset_str
        else:
            result._subset = SubsetProperty.from_string(subset_str)

        if num_accesses is not None:
            if num_accesses == -1:
                result.dynamic = True
                result.volume = 0
            else:
                result.volume = num_accesses
        else:
            result.volume = result._subset.num_elements()

        if wcr_str is not None:
            if isinstance(wcr_str, ast.AST):
                result.wcr = wcr_str
            else:
                result.wcr = LambdaProperty.from_string(wcr_str)

        if other_subset_str is not None:
            if isinstance(other_subset_str, subsets.Subset):
                result._other_subset = other_subset_str
            else:
                result._other_subset = SubsetProperty.from_string(
                    other_subset_str)
        else:
            result._other_subset = None

        # If it is an access node or another memlet
        if hasattr(data, 'data'):
            result._data = data.data
        else:
            result._data = data

        result.wcr_nonatomic = not wcr_conflict

        # Ensure memlet is not set as initialized
        result._initialized = False

        return result

    def _parse_from_subexpr(self, expr: str):
        if expr[-1] != ']':  # No subset given, try to use whole array
            if not dtypes.validate_name(expr):
                raise SyntaxError('Invalid memlet syntax "%s"' % expr)
            return expr, None

        # array[subset] syntax
        arrname, subset_str = expr[:-1].split('[')
        if not dtypes.validate_name(arrname):
            raise SyntaxError('Invalid array name "%s" in memlet' % arrname)
        return arrname, SubsetProperty.from_string(subset_str)

    def _parse_memlet_from_str(self, expr: str):
        """
        Parses a memlet and fills in either the src_subset,dst_subset fields
        or the _data,_subset fields.
        :param expr: A string expression of the this memlet, given as an ease
                of use API. Must follow one of the following forms:
                1. ``SRC_SUBSET -> DST_SUBSET``,
                2. ``ARRAY[SUBSET]``,
                3. `ARRAY[SUBSET] -> OTHER_SUBSET``.
                Note that modes 2 and 3 are deprecated and will leave 
                the memlet uninitialized until inserted into an SDFG.
        """
        expr = expr.strip()
        if '->' not in expr:  # Option 2
            self._data, self._subset = self._parse_from_subexpr(expr)
            self._initialized = False
            return

        # Options 1 or 3
        src_expr, dst_expr = expr.split('->')
        src_expr = src_expr.strip()
        dst_expr = dst_expr.strip()
        # Option 1
        if '[' not in src_expr and not dtypes.validate_name(src_expr):
            self.src_subset = SubsetProperty.from_string(src_expr)
            self.dst_subset = SubsetProperty.from_string(dst_expr)
            self._initialized = True
            return

        # Option 3
        self._data, self._subset = self._parse_from_subexpr(src_expr)
        self._other_subset = SubsetProperty.from_string(dst_expr)
        self._initialized = False

    def try_initialize(self, sdfg: 'dace.sdfg.SDFG',
                       state: 'dace.sdfg.SDFGState',
                       edge: 'dace.sdfg.graph.MultiConnectorEdge'):
        """ 
        Tries to initialize the internal fields of the memlet (e.g., src/dst 
        subset) once it is added to an SDFG as an edge.
        """
        from dace.sdfg.nodes import AccessNode  # Avoid import loop
        self._sdfg = sdfg
        self._state = state
        self._edge = edge
        if self._initialized:
            return

        assert self._data is not None

        if self._subset is None:
            self._subset = subsets.Range.from_array(sdfg.arrays[self._data])

        # Find source/destination of memlet
        try:
            path = state.memlet_path(edge)
        except (ValueError, AssertionError, StopIteration):
            # Cannot initialize yet
            return

        is_data_src = True
        if isinstance(path[-1].dst, AccessNode):
            if path[-1].dst.data == self._data:
                is_data_src = False

        self.src_subset = self._subset if is_data_src else self._other_subset
        self.dst_subset = self._other_subset if is_data_src else self._subset
        self._is_data_src = is_data_src

        # Mark memlet as initialized
        self._initialized = True

    @staticmethod
    def from_array(dataname, datadesc, wcr=None):
        """ Constructs a Memlet that transfers an entire array's contents.
            :param dataname: The name of the data descriptor in the SDFG.
            :param datadesc: The data descriptor object.
            :param wcr: The conflict resolution lambda.
            :type datadesc: Data
        """
        rng = subsets.Range.from_array(datadesc)
        return Memlet.simple(dataname, rng, wcr=wcr)

    def __hash__(self):
        return hash(
            (self.volume, self.src_subset, self.dst_subset, str(self.wcr)))

    def __eq__(self, other):
        return all([
            self.volume == other.volume, self.src_subset == other.src_subset,
            self.dst_subset == other.dst_subset, self.wcr == other.wcr
        ])

    def num_elements(self):
        """ Returns the number of elements in the Memlet subset. """
        if self.src_subset:
            return self.src_subset.num_elements()
        elif self.dst_subset:
            return self.dst_subset.num_elements()
        return 0

    def bounding_box_size(self):
        """ Returns a per-dimension upper bound on the maximum number of
            elements in each dimension.

            This bound will be tight in the case of Range.
        """
        if self.src_subset:
            return self.src_subset.bounding_box_size()
        elif self.dst_subset:
            return self.dst_subset.bounding_box_size()
        return []

    # Legacy fields
    @property
    def subset(self):
        if not self._initialized:
            return self._subset
        elif self._is_data_src is not None:
            return self.src_subset if self._is_data_src else self.dst_subset
        return self.src_subset

    @property
    def other_subset(self):
        if not self._initialized:
            return self._other_subset
        elif self._is_data_src is not None:
            return self.dst_subset if self._is_data_src else self.src_subset
        return self.dst_subset

    @property
    def data(self):
        if self._data is not None:
            return self._data
        elif not self._initialized:
            return self._data
        elif self._state is not None and self._edge is not None:
            # Try to obtain data from ends of memlet path
            from dace.sdfg.nodes import AccessNode  # Avoid import loop
            path = self._state.memlet_path(self._edge)
            path_src = path[0].src
            path_dst = path[-1].dst
            # Cache and return result
            if self._is_data_src is not None:
                self._data = (path_src.data
                              if self._is_data_src else path_dst.data)
                return self._data
            if isinstance(path_src, AccessNode):
                self._data = path_src.data
                return self._data
            elif isinstance(path_dst, AccessNode):
                self._data = path_dst.data
                return self._data
        return None

    # End of legacy fields

    def validate(self, sdfg, state):
        pass

    @property
    def free_symbols(self) -> Set[str]:
        """ Returns a set of symbols used in this edge's properties. """
        # Symbolic properties are in volume, and the two subsets
        result = set()
        result |= set(map(str, self.volume.free_symbols))
        if self.src_subset:
            result |= self.src_subset.free_symbols
        if self.dst_subset:
            result |= self.dst_subset.free_symbols
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
        if self.dynamic:
            result += '(dyn) '
        elif self.volume != num_elements:
            result += '(%s) ' % SymbolicProperty.to_string(self.num_accesses)
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
