# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import ast
from copy import deepcopy as dcpy, copy
from functools import reduce
import operator
from typing import TYPE_CHECKING, List, Optional, Set, Union
import warnings

import dace
import dace.serialize
from dace import subsets, dtypes, symbolic
from dace.frontend.operations import detect_reduction_type
from dace.frontend.python.astutils import unparse
from dace.properties import (Property, make_properties, DataProperty, SubsetProperty, SymbolicProperty,
                             DebugInfoProperty, LambdaProperty)
from dace.sdfg import nodes

if TYPE_CHECKING:
    import dace.sdfg.graph

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
                       dtype=bool,
                       desc='Is the number of elements moved determined at '
                       'runtime (e.g., data dependent)')
    subset = SubsetProperty(allow_none=True, desc='Subset of elements to move from the data attached to this edge.')
    other_subset = SubsetProperty(allow_none=True,
                                  desc='Subset of elements after reindexing to the data not attached '
                                  'to this edge (e.g., for offsets and reshaping).')
    data = DataProperty(desc='Data descriptor attached to this memlet')
    wcr = LambdaProperty(allow_none=True,
                         desc='If set, defines a write-conflict resolution '
                         'lambda function. The syntax of the lambda function '
                         'receives two elements: `current` value and `new` '
                         'value, and returns the value after resolution')

    # Code generation and validation hints
    debuginfo = DebugInfoProperty(desc='Line information to track source and generated code')
    wcr_nonatomic = Property(dtype=bool,
                             default=False,
                             desc='If True, always generates non-conflicting '
                             '(non-atomic) writes in resulting code')
    allow_oob = Property(dtype=bool, default=False, desc='Bypass out-of-bounds validation')

    def __init__(self,
                 expr: Optional[str] = None,
                 data: Optional[str] = None,
                 subset: Union[str, subsets.Subset, None] = None,
                 other_subset: Union[str, subsets.Subset, None] = None,
                 volume: Union[int, str, symbolic.SymbolicType, None] = None,
                 dynamic: bool = False,
                 wcr: Union[str, ast.AST, None] = None,
                 debuginfo: Optional[dtypes.DebugInfo] = None,
                 wcr_nonatomic: bool = False,
                 allow_oob: bool = False):
        """ 
        Constructs a Memlet.
        
        :param expr: A string expression of the this memlet, given as an ease
                     of use API. Must follow one of the following forms:
                     1. ``ARRAY``,
                     2. ``ARRAY[SUBSET]``,
                     3. ``ARRAY[SUBSET] -> OTHER_SUBSET``.
        :param data: Data descriptor name attached to this memlet.
        :param subset: The subset to take from the data attached to the edge,
                       represented either as a string or a Subset object.
        :param other_subset: The subset to offset into the other side of the
                             memlet, represented either as a string or a Subset 
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

        # Will be set once memlet is added into an SDFG (in try_initialize)
        self._sdfg = None
        self._state = None
        self._edge = None

        # Field caching which subset belongs to source or destination of memlet
        self._is_data_src = None

        # Initialize first by string expression
        self.data = None
        self.subset = None
        self.other_subset = None
        if expr is not None:
            self._parse_memlet_from_str(expr)

        # Set properties
        self.data = self.data or data
        self.subset = self.subset or subset
        self.other_subset = self.other_subset or other_subset

        if volume is not None:
            self.volume = volume
        else:
            if self.subset is not None:
                self.volume = self.subset.num_elements()
            elif self.other_subset is not None:
                self.volume = self.other_subset.num_elements()
            else:
                self.volume = 1

        self.dynamic = dynamic
        self.wcr = wcr
        self.wcr_nonatomic = wcr_nonatomic
        self.debuginfo = debuginfo
        self.allow_oob = allow_oob

    @staticmethod
    def from_memlet(memlet: 'Memlet') -> 'Memlet':
        sbs = subsets.Range(memlet.subset.ndrange()) if memlet.subset is not None else None
        osbs = subsets.Range(memlet.other_subset.ndrange()) if memlet.other_subset is not None else None
        result = Memlet(data=memlet.data,
                        subset=sbs,
                        other_subset=osbs,
                        volume=memlet.volume,
                        dynamic=memlet.dynamic,
                        wcr=memlet.wcr,
                        debuginfo=copy(memlet.debuginfo),
                        wcr_nonatomic=memlet.wcr_nonatomic,
                        allow_oob=memlet.allow_oob)
        result._is_data_src = memlet._is_data_src
        return result

    def to_json(self):
        attrs = dace.serialize.all_properties_to_json(self)

        # Fill in new values
        if self.src_subset is not None:
            attrs['src_subset'] = self.src_subset.to_json()
        else:
            attrs['src_subset'] = None
        if self.dst_subset is not None:
            attrs['dst_subset'] = self.dst_subset.to_json()
        else:
            attrs['dst_subset'] = None

        attrs['is_data_src'] = self._is_data_src

        # Fill in legacy (DEPRECATED) values for backwards compatibility
        attrs['num_accesses'] = \
            str(self.volume) if not self.dynamic else -1

        return {"type": "Memlet", "attributes": attrs}

    @staticmethod
    def from_json(json_obj, context=None):
        ret = Memlet()
        dace.serialize.set_properties_from_json(ret,
                                                json_obj,
                                                context=context,
                                                ignore_properties={'src_subset', 'dst_subset', 'num_accesses', 'is_data_src'})
        
        # Allow serialized memlet to override src/dst_subset to disambiguate self-copies
        if 'is_data_src' in json_obj['attributes']:
            ret._is_data_src = json_obj['attributes']['is_data_src']
        
        if context:
            ret._sdfg = context['sdfg']
            ret._state = context['sdfg_state']
        return ret

    def __deepcopy__(self, memo):
        node = object.__new__(Memlet)

        # Set properties
        node._volume = dcpy(self._volume, memo=memo)
        node._dynamic = self._dynamic
        node._subset = dcpy(self._subset, memo=memo)
        node._other_subset = dcpy(self._other_subset, memo=memo)
        node._data = dcpy(self._data, memo=memo)
        node._wcr = dcpy(self._wcr, memo=memo)
        node._wcr_nonatomic = dcpy(self._wcr_nonatomic, memo=memo)
        node._debuginfo = dcpy(self._debuginfo, memo=memo)
        node._wcr_nonatomic = self._wcr_nonatomic
        node._allow_oob = self._allow_oob
        node._is_data_src = self._is_data_src

        # Nullify graph references
        node._sdfg = None
        node._state = None
        node._edge = None

        return node

    def is_empty(self) -> bool:
        """ 
        Returns True if this memlet carries no data. Memlets without data are
        primarily used for connecting nodes to scopes without transferring 
        data to them. 
        """
        return (self.data is None and self.src_subset is None and self.dst_subset is None)

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
               wcr_str=None,
               other_subset_str=None,
               wcr_conflict=True,
               num_accesses=None,
               debuginfo=None,
               dynamic=False):
        """
        DEPRECATED: Constructs a Memlet from string-based expressions.

        :param data: The data object or name to access. 
        :param subset_str: The subset of `data` that is going to
                            be accessed in string format. Example: '0:N'.
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
        :param dynamic: If True, the number of elements moved in this memlet
                        is defined dynamically at runtime.
        """
        # warnings.warn(
        #     'This function is deprecated, please use the Memlet '
        #     'constructor instead', DeprecationWarning)

        result = Memlet()

        if isinstance(subset_str, subsets.Subset):
            result.subset = subset_str
        else:
            result.subset = SubsetProperty.from_string(subset_str)

        result.dynamic = dynamic

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
                result.other_subset = other_subset_str
            else:
                result.other_subset = SubsetProperty.from_string(other_subset_str)
        else:
            result.other_subset = None

        # If it is an access node or another memlet
        if hasattr(data, 'data'):
            result.data = data.data
        else:
            result.data = data

        result.wcr_nonatomic = not wcr_conflict

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
                1. ``ARRAY``,
                2. ``ARRAY[SUBSET]``,
                3. ``ARRAY[SUBSET] -> OTHER_SUBSET``.
                Note that modes 2 and 3 are deprecated and will leave 
                the memlet uninitialized until inserted into an SDFG.
        """
        expr = expr.strip()
        if '->' not in expr:  # Options 1 and 2
            self.data, self.subset = self._parse_from_subexpr(expr)
            return

        # Option 3
        src_expr, dst_expr = expr.split('->')
        src_expr = src_expr.strip()
        dst_expr = dst_expr.strip()
        if '[' not in src_expr and not dtypes.validate_name(src_expr):
            raise SyntaxError('Expression without data name not yet allowed')

        self.data, self.subset = self._parse_from_subexpr(src_expr)
        self.other_subset = SubsetProperty.from_string(dst_expr)

    def try_initialize(self, sdfg: 'dace.sdfg.SDFG', state: 'dace.sdfg.SDFGState',
                       edge: 'dace.sdfg.graph.MultiConnectorEdge'):
        """ 
        Tries to initialize the internal fields of the memlet (e.g., src/dst 
        subset) once it is added to an SDFG as an edge.
        """
        from dace.sdfg.nodes import AccessNode, CodeNode  # Avoid import loops
        self._sdfg = sdfg
        self._state = state
        self._edge = edge

        # If memlet is code->code, ensure volume=1
        if (isinstance(edge.src, CodeNode) and isinstance(edge.dst, CodeNode) and self.volume == 0):
            self.volume = 1

        # Find source/destination of memlet
        try:
            path = state.memlet_path(edge)
        except (ValueError, StopIteration):
            # Cannot initialize yet
            return

        is_data_src = False
        is_data_dst = False
        if isinstance(path[0].src, AccessNode):
            if path[0].src.data == self._data:
                is_data_src = True
        if isinstance(path[-1].dst, AccessNode):
            if path[-1].dst.data == self._data:
                is_data_dst = True
        if is_data_src and is_data_dst:
            # In case both point to the same array,
            # prefer existing setting or is_data_src=True
            if self._is_data_src is None:
                self._is_data_src = True
        else:
            self._is_data_src = is_data_src

        # If subset is None, fill in with entire array
        if (self.data is not None and self.subset is None):
            self.subset = subsets.Range.from_array(sdfg.arrays[self.data])

        other_data = None
        if self._is_data_src and isinstance(path[-1].dst, nodes.AccessNode):
            other_data = path[-1].dst.data
        elif not self._is_data_src and isinstance(path[0].src, nodes.AccessNode):
            other_data = path[0].src.data
        # If there is "other_data" (and "data") but there is no "other_subset", then set "other_subset"
        # to the same range as "subset", if the shape of "other_data" covers the "subset" or the shape of "data".
        if other_data is not None and self.data is not None and self.other_subset is None:
            data_desc = sdfg.arrays[self.data]
            other_data_desc = sdfg.arrays[other_data]
            if len(data_desc.shape) == len(other_data_desc.shape):
                data_subset = subsets.Range.from_array(data_desc)
                other_data_subset = subsets.Range.from_array(other_data_desc)
                if other_data_subset.covers(self.subset) or other_data_subset.covers(data_subset):
                    self.other_subset = dcpy(self.subset)

    def get_src_subset(self, edge: 'dace.sdfg.graph.MultiConnectorEdge', state: 'dace.sdfg.SDFGState'):
        self.try_initialize(state.parent, state, edge)
        return self.src_subset

    def get_dst_subset(self, edge: 'dace.sdfg.graph.MultiConnectorEdge', state: 'dace.sdfg.SDFGState'):
        self.try_initialize(state.parent, state, edge)
        return self.dst_subset

    @staticmethod
    def from_array(dataname, datadesc, wcr=None):
        """ 
        Constructs a Memlet that transfers an entire array's contents.

        :param dataname: The name of the data descriptor in the SDFG.
        :param datadesc: The data descriptor object.
        :param wcr: The conflict resolution lambda.
        :type datadesc: Data
        """
        rng = subsets.Range.from_array(datadesc)
        return Memlet.simple(dataname, rng, wcr_str=wcr)

    def __hash__(self):
        return hash((self.volume, self.src_subset, self.dst_subset, str(self.wcr)))

    def __eq__(self, other):
        return all([
            self.volume == other.volume, self.src_subset == other.src_subset, self.dst_subset == other.dst_subset,
            self.wcr == other.wcr
        ])

    def replace(self, repl_dict):
        """
        Substitute a given set of symbols with a different set of symbols.
        
        :param repl_dict: A dict of string symbol names to symbols with
                          which to replace them.
        """
        repl_to_intermediate = {}
        repl_to_final = {}
        for symbol in repl_dict:
            if str(symbol) != str(repl_dict[symbol]):
                intermediate = symbolic.symbol('__dacesym_' + str(symbol))
                repl_to_intermediate[symbolic.symbol(symbol)] = intermediate
                repl_to_final[intermediate] = repl_dict[symbol]

        if len(repl_to_intermediate) > 0:
            if self.volume is not None and symbolic.issymbolic(self.volume):
                self.volume = self.volume.subs(repl_to_intermediate)
                self.volume = self.volume.subs(repl_to_final)
            if self.subset is not None:
                self.subset.replace(repl_to_intermediate)
                self.subset.replace(repl_to_final)
            if self.other_subset is not None:
                self.other_subset.replace(repl_to_intermediate)
                self.other_subset.replace(repl_to_final)

    def num_elements(self):
        """ Returns the number of elements in the Memlet subset. """
        if self.subset:
            return self.subset.num_elements()
        elif self.other_subset:
            return self.other_subset.num_elements()
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

    # New fields
    @property
    def src_subset(self):
        if self._is_data_src is not None:
            return self.subset if self._is_data_src else self.other_subset
        return self.subset

    @src_subset.setter
    def src_subset(self, new_src_subset):
        if self._is_data_src is not None:
            if self._is_data_src:
                self.subset = new_src_subset
            else:
                self.other_subset = new_src_subset
        else:
            self.subset = new_src_subset

    @property
    def dst_subset(self):
        if self._is_data_src is not None:
            return self.other_subset if self._is_data_src else self.subset
        return self.other_subset

    @dst_subset.setter
    def dst_subset(self, new_dst_subset):
        if self._is_data_src is not None:
            if self._is_data_src:
                self.other_subset = new_dst_subset
            else:
                self.subset = new_dst_subset
        else:
            self.other_subset = new_dst_subset

    def validate(self, sdfg, state):
        if self.data is not None and self.data not in sdfg.arrays:
            raise KeyError('Array "%s" not found in SDFG' % self.data)

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

    def get_stride(self, sdfg: 'dace.sdfg.SDFG', map: 'dace.sdfg.nodes.Map', dim: int = -1) -> 'dace.symbolic.SymExpr':
        """ Returns the stride of the underlying memory when traversing a Map.
            
            :param sdfg: The SDFG in which the memlet resides.
            :param map: The map in which the memlet resides.
            :param dim: The dimension that is incremented. By default it is the innermost.
        """
        if self.data is None:
            return symbolic.pystr_to_symbolic('0')

        param = symbolic.symbol(map.params[dim])
        array = sdfg.arrays[self.data]

        # Flatten the subset to a 1D-offset (using the array strides) at some iteration
        curr = self.subset.at([0] * len(array.strides), array.strides)

        # Substitute the param with the next (possibly strided) value
        next = curr.subs(param, param + map.range[dim][2])

        # The stride is the difference between both
        return (next - curr).simplify()

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
            result += '(%s) ' % SymbolicProperty.to_string(self.volume)
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
            if self._is_data_src is False:
                result += ' <- [%s]' % str(self.other_subset)
            else:
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
    def __init__(self,
                 edge: 'dace.sdfg.graph.MultiConnectorEdge[Memlet]',
                 downwards: bool = True,
                 parent: 'MemletTree' = None,
                 children: Optional[List['MemletTree']] = None) -> None:
        self.edge = edge
        self.parent = parent
        self.children = children or []
        self._downwards = downwards

    @property
    def downwards(self):
        """ If True, this memlet tree points downwards (rooted at the source node). """
        return self._downwards

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

    def leaves(self) -> 'List[dace.sdfg.graph.MultiConnectorEdge[Memlet]]':
        """ Returns a list of all the leaves of this MemletTree, i.e., the innermost edges. """
        if not self.children:
            return [self.edge]

        result = []
        for child in self.children:
            result.extend(child.leaves())
        return result

    def traverse_children(self, include_self=False):
        if include_self:
            yield self
        for child in self.children:
            yield from child.traverse_children(include_self=True)
