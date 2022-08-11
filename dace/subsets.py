# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace.serialize
from dace import data, symbolic, dtypes
import re
import sympy as sp
from functools import reduce
import sympy.core.sympify
from typing import List, Optional, Sequence, Set, Union
import warnings
from dace.config import Config


class Subset(object):
    """ Defines a subset of a data descriptor. """
    def covers(self, other):
        """ Returns True if this subset covers (using a bounding box) another
            subset. """
        def nng(expr):
            # When dealing with set sizes, assume symbols are non-negative
            try:
                # TODO: Fix in symbol definition, not here
                for sym in list(expr.free_symbols):
                    expr = expr.subs({sym: sp.Symbol(sym.name, nonnegative=True)})
                return expr
            except AttributeError:  # No free_symbols in expr
                return expr

        symbolic_positive = Config.get('optimizer', 'symbolic_positive')

        if not symbolic_positive:
            try:
                return all([(symbolic.simplify_ext(nng(rb)) <= symbolic.simplify_ext(nng(orb))) == True
                            and (symbolic.simplify_ext(nng(re)) >= symbolic.simplify_ext(nng(ore))) == True
                            for rb, re, orb, ore in zip(self.min_element_approx(), self.max_element_approx(),
                                                        other.min_element_approx(), other.max_element_approx())])
            except TypeError:
                return False

        else:
            try:
                for rb, re, orb, ore in zip(self.min_element_approx(), self.max_element_approx(),
                                            other.min_element_approx(), other.max_element_approx()):

                    # lower bound: first check whether symbolic positive condition applies
                    if not (len(rb.free_symbols) == 0 and len(orb.free_symbols) == 1):
                        if not symbolic.simplify_ext(nng(rb)) <= symbolic.simplify_ext(nng(orb)):
                            return False

                    # upper bound: first check whether symbolic positive condition applies
                    if not (len(re.free_symbols) == 1 and len(ore.free_symbols) == 0):
                        if not symbolic.simplify_ext(nng(re)) >= symbolic.simplify_ext(nng(ore)):
                            return False
            except TypeError:
                return False

            return True

    def __repr__(self):
        return '%s (%s)' % (type(self).__name__, self.__str__())

    def offset(self, other, negative, indices=None):
        raise NotImplementedError

    def offset_new(self, other, negative, indices=None):
        raise NotImplementedError

    def at(self, i, strides):
        """ Returns the absolute index (1D memory layout) of this subset at
            the given index tuple.

            For example, the range [2:10:2] at index 2 would return 6 (2+2*2).

            :param i: A tuple of the same dimensionality as subset.dims() or
                      subset.data_dims().
            :param strides: The strides of the array we are subsetting.
            :return: Absolute 1D index at coordinate i.
        """
        raise NotImplementedError

    def coord_at(self, i):
        """ Returns the offseted coordinates of this subset at
            the given index tuple.

            For example, the range [2:10:2] at index 2 would return 6 (2+2*2).

            :param i: A tuple of the same dimensionality as subset.dims() or
                      subset.data_dims().
            :return: Absolute coordinates for index i (length equal to
                     `data_dims()`, may be larger than `dims()`).
        """
        raise NotImplementedError

    @property
    def free_symbols(self) -> Set[str]:
        """ Returns a set of undefined symbols in this subset. """
        raise NotImplementedError('free_symbols not implemented by "%s"' % type(self).__name__)


def _simplified_str(val):
    val = _expr(val)
    try:
        return str(int(val))
    except TypeError:
        return str(val)


def _expr(val):
    if isinstance(val, symbolic.SymExpr):
        return val.expr
    return val


def _approx(val):
    if isinstance(val, symbolic.SymExpr):
        return val.approx
    elif isinstance(val, sp.Basic):
        return val
    return symbolic.pystr_to_symbolic(val)


def _tuple_to_symexpr(val):
    return (symbolic.SymExpr(val[0], val[1]) if isinstance(val, tuple) else symbolic.pystr_to_symbolic(val))


@dace.serialize.serializable
class Range(Subset):
    """ Subset defined in terms of a fixed range. """
    def __init__(self, ranges):
        parsed_ranges = []
        parsed_tiles = []
        for r in ranges:
            if len(r) != 3 and len(r) != 4:
                raise ValueError("Expected 3-tuple or 4-tuple")
            parsed_ranges.append((_tuple_to_symexpr(r[0]), _tuple_to_symexpr(r[1]), _tuple_to_symexpr(r[2])))
            if len(r) == 3:
                parsed_tiles.append(symbolic.pystr_to_symbolic(1))
            else:
                parsed_tiles.append(symbolic.pystr_to_symbolic(r[3]))
        self.ranges = parsed_ranges
        self.tile_sizes = parsed_tiles

    @staticmethod
    def from_indices(indices: 'Indices'):
        return Range([(i, i, 1) for i in indices.indices])

    def to_json(self):
        ret = []

        def a2s(obj):
            if isinstance(obj, symbolic.SymExpr):
                return {'main': str(obj.expr), 'approx': str(obj.approx)}
            else:
                return _simplified_str(obj)

        for (start, end, step), tile in zip(self.ranges, self.tile_sizes):
            ret.append({'start': a2s(start), 'end': a2s(end), 'step': a2s(step), 'tile': a2s(tile)})

        return {'type': 'Range', 'ranges': ret}

    @staticmethod
    def from_json(obj, context=None):
        if not isinstance(obj, dict):
            raise TypeError("Expected dict, got {}".format(type(obj)))
        if obj['type'] != 'Range':
            raise TypeError("from_json of class \"Range\" called on json "
                            "with type %s (expected 'Range')" % obj['type'])

        ranges = obj['ranges']
        tuples = []

        def p2s(x):
            pts = symbolic.pystr_to_symbolic
            if isinstance(x, str):
                return pts(x)
            else:
                return symbolic.SymExpr(pts(x['main']), pts(x['approx']))

        for r in ranges:
            tuples.append((p2s(r['start']), p2s(r['end']), p2s(r['step']), p2s(r['tile'])))

        return Range(tuples)

    @staticmethod
    def from_array(array: 'dace.data.Data'):
        """ Constructs a range that covers the full array given as input. """
        return Range([(-o, s - 1 -o , 1) for s, o in zip(array.shape, array.offset)])
        if any(o != 0 for o in array.offset):
            result.offset(array.offset, True)
        return result

    def __hash__(self):
        return hash(tuple(r for r in self.ranges))

    def __add__(self, other):
        sum_ranges = self.ranges + other.ranges
        return Range(sum_ranges)

    def num_elements(self):
        return reduce(sp.Mul, self.size(), 1)

    def num_elements_exact(self):
        return reduce(sp.Mul, self.bounding_box_size(), 1)

    def size(self, for_codegen=False):
        """ Returns the number of elements in each dimension. """
        offset = [-1 if (s < 0) == True else 1 for _, _, s in self.ranges]

        if for_codegen:
            int_ceil = symbolic.int_ceil
            return [
                ts * int_ceil(((iMax.approx if isinstance(iMax, symbolic.SymExpr) else iMax) + off -
                               (iMin.approx if isinstance(iMin, symbolic.SymExpr) else iMin)),
                              (step.approx if isinstance(step, symbolic.SymExpr) else step))
                for (iMin, iMax, step), off, ts in zip(self.ranges, offset, self.tile_sizes)
            ]
        else:
            return [
                ts * sp.ceiling(((iMax.approx if isinstance(iMax, symbolic.SymExpr) else iMax) + off -
                                 (iMin.approx if isinstance(iMin, symbolic.SymExpr) else iMin)) /
                                (step.approx if isinstance(step, symbolic.SymExpr) else step))
                for (iMin, iMax, step), off, ts in zip(self.ranges, offset, self.tile_sizes)
            ]

    def size_exact(self):
        """ Returns the number of elements in each dimension. """
        return [
            ts * sp.ceiling(((iMax.expr if isinstance(iMax, symbolic.SymExpr) else iMax) + 1 -
                             (iMin.expr if isinstance(iMin, symbolic.SymExpr) else iMin)) /
                            (step.expr if isinstance(step, symbolic.SymExpr) else step))
            for (iMin, iMax, step), ts in zip(self.ranges, self.tile_sizes)
        ]

    def bounding_box_size(self):
        """ Returns the size of a bounding box around this range. """
        return [
            # sp.floor((iMax - iMin) / step) - iMin
            ts * ((iMax.approx if isinstance(iMax, symbolic.SymExpr) else iMax) -
                  (iMin.approx if isinstance(iMin, symbolic.SymExpr) else iMin) + 1)
            for (iMin, iMax, step), ts in zip(self.ranges, self.tile_sizes)
        ]

    def min_element(self):
        return [_expr(x[0]) for x in self.ranges]

    def max_element(self):
        return [_expr(x[1]) for x in self.ranges]

    def max_element_approx(self):
        return [_approx(x[1]) for x in self.ranges]

    def min_element_approx(self):
        return [_approx(x[0]) for x in self.ranges]

    def coord_at(self, i):
        """ Returns the offseted coordinates of this subset at
            the given index tuple.

            For example, the range [2:10:2] at index 2 would return 6 (2+2*2).

            :param i: A tuple of the same dimensionality as subset.dims() or
                      subset.data_dims().
            :return: Absolute coordinates for index i (length equal to
                     `data_dims()`, may be larger than `dims()`).
        """
        tiles = sum(1 if ts != 1 else 0 for ts in self.tile_sizes)
        if len(i) != len(self.ranges) and len(i) != len(self.ranges) + tiles:
            raise ValueError('Invalid dimensionality of input tuple (expected'
                             ' %d, got %d)' % (len(self.ranges), len(i)))

        # Pad with zeros for tiles
        ts_len = len(i) - len(self.ranges)
        ti = i[len(self.ranges):] + [0] * (tiles - ts_len)

        i = i[:len(self.ranges)]

        return tuple(_expr(rb) + k * _expr(rs) for k, (rb, _, rs) in zip(i, self.ranges)) + tuple(ti)

    def at(self, i, strides):
        """ Returns the absolute index (1D memory layout) of this subset at
            the given index tuple.

            For example, the range [2:10:2] at index 2 would return 6 (2+2*2).

            :param i: A tuple of the same dimensionality as subset.dims() or
                      subset.data_dims().
            :param strides: The strides of the array we are subsetting.
            :return: Absolute 1D index at coordinate i.
        """
        coord = self.coord_at(i)

        # Return i0 + i1*size0 + i2*size1*size0 + ....
        # Cancel out stride since we determine the initial offset only here
        return sum(
            _expr(s) * _expr(astr) / _expr(rs)
            for s, (_, _, rs), astr in zip(coord, self.ranges, self.absolute_strides(strides)))

    def data_dims(self):
        return (sum(1 if (re - rb + 1) != 1 else 0 for rb, re, _ in self.ranges) + sum(1 if ts != 1 else 0
                                                                                       for ts in self.tile_sizes))

    def offset(self, other, negative, indices=None):
        if not isinstance(other, Subset):
            if isinstance(other, (list, tuple)):
                other = Indices(other)
            else:
                other = Indices([other for _ in self.ranges])
        mult = -1 if negative else 1
        if indices is None:
            indices = set(range(len(self.ranges)))
        off = other.min_element()
        for i in indices:
            rb, re, rs = self.ranges[i]
            if rb == re and rb!=0 :
                rbmult=0
                remult=0
            else: 
                rbmult=1    
                remult=1   
            self.ranges[i] = (rb + rbmult * mult * off[i], re + remult * mult * off[i], rs)

    def offset_new(self, other, negative, indices=None):
        if not isinstance(other, Subset):
            if isinstance(other, (list, tuple)):
                other = Indices(other)
            else:
                other = Indices([other for _ in self.ranges])
        mult = -1 if negative else 1
        if indices is None:
            indices = set(range(len(self.ranges)))
        off = other.min_element()
        return Range([(self.ranges[i][0] + mult * off[i], self.ranges[i][1] + mult * off[i], self.ranges[i][2])
                      for i in indices])

    def dims(self):
        return len(self.ranges)

    def absolute_strides(self, global_shape):
        """ Returns a list of strides for advancing one element in each
            dimension. Size of the list is equal to `data_dims()`, which may
            be larger than `dims()` depending on tile sizes. """
        # ..., stride2*size1*size0, stride1*size0, stride0, ..., tile strides
        return [rs * global_shape[i] for i, (_, _, rs) in enumerate(self.ranges)
                ] + [global_shape[i] for i, ts in enumerate(self.tile_sizes) if ts != 1]

    def strides(self):
        return [rs for _, _, rs in self.ranges]

    @staticmethod
    def _range_pystr(range):
        return "(" + ", ".join(map(str, range)) + ")"

    def pystr(self):
        return "[" + ", ".join(map(Range._range_pystr, self.ranges)) + "]"

    @property
    def free_symbols(self) -> Set[str]:
        result = set()
        for dim in self.ranges:
            for d in dim:
                result |= symbolic.symlist(d).keys()
        return result

    def reorder(self, order):
        """ Re-orders the dimensions in-place according to a permutation list.
            :param order: List or tuple of integers from 0 to self.dims() - 1,
                          indicating the desired order of the dimensions.
        """
        new_ranges = [self.ranges[o] for o in order]
        self.ranges = new_ranges

    @staticmethod
    def dim_to_string(d, t=1):
        if isinstance(d, tuple):
            dres = _simplified_str(d[0])
            if d[1] is not None:
                if d[1] - d[0] != 0:
                    off = 1
                    if d[2] is not None and (d[2] < 0) == True:
                        off = -1
                    dres += ':' + _simplified_str(d[1] + off)
            if d[2] != 1:
                if d[1] is None:
                    dres += ':'
                dres += ':' + _simplified_str(d[2])
            if t != 1:
                if d[1] is None and d[2] == 1:
                    dres += '::'
                elif d[2] == 1:
                    dres += ':'
                dres += ':' + _simplified_str(t)
            return dres
        else:
            return _simplified_str(d)

    @staticmethod
    def from_string(string):

        # The following code uses regular expressions in order to support the
        # use of comma not only for separating range dimensions, but also
        # inside function calls.

        # Example (with 2 dimensions):
        # tile_i * ts_i : min(int_ceil(M, rs_i), tile_i * ts_i + ts_i),
        # regtile_j * rs_j : min(K, regtile_j * rs_j + rs_j)

        ranges = []

        # Split string to tokens separated by colons.
        # tokens = [
        #   'tile_i * ts_i ',
        #   'min(int_ceil(M, rs_i), tile_i * ts_i + ts_i), regtile_j * rs_j ',
        #   'min(K, regtile_j * rs_j + rs_j)'
        # ]
        tokens = string.split(':')

        # In the example, the second token must be split to 2 separate tokens.

        # List of list of tokens (one list per range dimension)
        multi_dim_tokens = []
        # List of tokens (single dimension)
        uni_dim_tokens = []

        for token in tokens:

            i = 0  # Character index in the token
            count = 0  # Number of open parenthesis

            while i < len(token):
                # Comma found while not in a function or any other expression
                # with parenthesis. This is a comma separating range dimensions.
                if token[i] == ',' and count == 0:
                    # Split the token to token[:i] and token[i+1:]
                    # Append token[:i] to the current range dimension
                    uni_dim_tokens.append(token[0:i])
                    # Append current range dimension to the list of lists
                    multi_dim_tokens.append(uni_dim_tokens)
                    # Start a new range dimension
                    uni_dim_tokens = []
                    # Adjust the token
                    token = token[i + 1:]
                    i = 0
                    continue
                # Open parenthesis found, increase count by 1
                if token[i] == '(':
                    count += 1
                # Closing parenthesis found, decrease cound by 1
                elif token[i] == ')':
                    count -= 1
                # Move to the next character
                i += 1

            # Append token to the current range dimension
            uni_dim_tokens.append(token)

        # Append current range dimension to the list of lists
        multi_dim_tokens.append(uni_dim_tokens)

        # Generate ranges
        for uni_dim_tokens in multi_dim_tokens:
            # If dimension has only 1 token, then it is an index (not a range),
            # treat as range of size 1
            if len(uni_dim_tokens) < 2:
                ranges.append(
                    (symbolic.pystr_to_symbolic(uni_dim_tokens[0]), symbolic.pystr_to_symbolic(uni_dim_tokens[0]), 1))
                continue
                #return Range(ranges)
            # If dimension has more than 4 tokens, the range is invalid
            if len(uni_dim_tokens) > 4:
                raise SyntaxError("Invalid range: {}".format(multi_dim_tokens))
            # Support for SymExpr
            tokens = []
            for token in uni_dim_tokens:
                expr = token.split('|')
                if len(expr) == 1:
                    tokens.append(expr[0])
                elif len(expr) == 2:
                    tokens.append((expr[0], expr[1]))
                else:
                    raise SyntaxError("Invalid range: {}".format(multi_dim_tokens))
            # Parse tokens
            try:
                if isinstance(tokens[0], tuple):
                    begin = symbolic.SymExpr(tokens[0][0], tokens[0][1])
                else:
                    begin = symbolic.pystr_to_symbolic(tokens[0])
                if len(tokens) >= 3:
                    if isinstance(tokens[2], tuple):
                        step = symbolic.SymExpr(tokens[2][0], tokens[2][1])
                    else:
                        step = symbolic.SymExpr(tokens[2])
                else:
                    step = 1
                eoff = -1
                if (step < 0) == True:
                    eoff = 1
                if isinstance(tokens[1], tuple):
                    end = symbolic.SymExpr(tokens[1][0], tokens[1][1]) + eoff
                else:
                    end = symbolic.pystr_to_symbolic(tokens[1]) + eoff
                if len(tokens) >= 4:
                    if isinstance(tokens[3], tuple):
                        tsize = tokens[3][0]
                    else:
                        tsize = tokens[3]
                else:
                    tsize = 1
            except sympy.SympifyError:
                raise SyntaxError("Invalid range: {}".format(string))
            # Append range
            ranges.append((begin, end, step, tsize))

        return Range(ranges)

    @staticmethod
    def ndslice_to_string(slice, tile_sizes=None):
        if tile_sizes is None:
            return ", ".join([Range.dim_to_string(s) for s in slice])
        return ", ".join([Range.dim_to_string(s, t) for s, t in zip(slice, tile_sizes)])

    @staticmethod
    def ndslice_to_string_list(slice, tile_sizes=None):
        if tile_sizes is None:
            return [Range.dim_to_string(s) for s in slice]
        return [Range.dim_to_string(s, t) for s, t in zip(slice, tile_sizes)]

    def ndrange(self):
        return [(rb, re, rs) for rb, re, rs in self.ranges]

    def __str__(self):
        return Range.ndslice_to_string(self.ranges, self.tile_sizes)

    def __iter__(self):
        return iter(self.ranges)

    def __len__(self):
        return len(self.ranges)

    def __getitem__(self, key):
        return self.ranges.__getitem__(key)

    def __setitem__(self, key, value):
        return self.ranges.__setitem__(key, value)

    def __eq__(self, other):
        if not isinstance(other, Range):
            return False
        if len(self.ranges) != len(other.ranges):
            return False
        return all([(rb == orb and re == ore and rs == ors)
                    for (rb, re, rs), (orb, ore, ors) in zip(self.ranges, other.ranges)])

    def __ne__(self, other):
        return not self.__eq__(other)

    def compose(self, other) -> 'Range':
        if not isinstance(other, Subset):
            raise TypeError("Cannot compose ranges with non-subsets")

        new_subset = []
        if self.data_dims() == other.dims():
            # case 1: subsets may differ in dimensions, but data_dims correspond
            #         to other dims -> all non-data dims are cut out
            idx = 0
            off = self.min_element()
            for (rb, re, rs), rt in zip(self.ranges, self.tile_sizes):
                if re - rb == 0:
                    if isinstance(other, Indices):
                        new_subset.append(rb)
                    else:
                        new_subset.append((rb, re, rs, rt))
                else:
                    if isinstance(other[idx], tuple):
                        new_subset.append((rb + rs * other[idx][0], rb + rs * other[idx][1], rs * other[idx][2], rt))
                    else:
                        new_subset.append(rb + rs * other[idx])
                    idx += 1
        elif self.dims() == other.dims():
            # case 2: subsets have the same dimensions (but possibly different
            # data_dims) -> all non-data dims remain
            for idx, ((rb, re, rs), rt) in enumerate(zip(self.ranges, self.tile_sizes)):
                if re - rb == 0:
                    if isinstance(other, Indices):
                        new_subset.append(rb)
                    else:
                        new_subset.append((rb, re, rs, rt))
                else:
                    if isinstance(other[idx], tuple):
                        new_subset.append((rb + rs * other[idx][0], rb + rs * other[idx][1], rs * other[idx][2], rt))
                    else:
                        new_subset.append(rb + rs * other[idx])
        elif (other.data_dims() == 0 and all([r == (0, 0, 1) if isinstance(other, Range) else r == 0 for r in other])):
            # NOTE: This is a special case where the other subset is the
            # (potentially multidimensional) index zero.
            # For example, A[i, j] -> tmp[0]. The result of such a
            # composition should be equal to the first subset.
            if isinstance(other, Range):
                new_subset.extend(self.ranges)
            else:
                new_subset.extend([rb for rb, _, _ in self.ranges])
        else:
            raise ValueError("Dimension mismatch in composition: "
                             "Subset composed must be either completely "
                             "stripped of all non-data dimensions "
                             "or be not stripped of latter at all.")

        if isinstance(other, Range):
            return Range(new_subset)
        elif isinstance(other, Indices):
            return Indices(new_subset)
        else:
            raise NotImplementedError

    def squeeze(self, ignore_indices=None, offset=True):
        ignore_indices = ignore_indices or []
        shape = self.size()
        non_ones = []
        offset_indices = []
        sqz_idx = 0
        for i, d in enumerate(shape):
            if i in ignore_indices:
                non_ones.append(i)
                sqz_idx += 1
            elif d != 1:
                non_ones.append(i)
                offset_indices.append(sqz_idx)
                sqz_idx += 1
            else:
                pass
        squeezed_ranges = [self.ranges[i] for i in non_ones]
        squeezed_tsizes = [self.tile_sizes[i] for i in non_ones]
        if not squeezed_ranges:
            squeezed_ranges = [(0, 0, 1)]
            squeezed_tsizes = [1]
        self.ranges = squeezed_ranges
        self.tile_sizes = squeezed_tsizes
        if offset:
            self.offset(self, True, indices=offset_indices)
        return non_ones

    def unsqueeze(self, axes: Sequence[int]) -> List[int]:
        """ Adds 0:1 ranges to the subset, in the indices contained in axes.
        
        The method is mostly used to restore subsets that had their length-1
        ranges removed (i.e., squeezed subsets). Hence, the method is
        called 'unsqueeze'.

        Examples (initial subset, axes -> result subset, output):
        - [i:i+10], [0] -> [0:1, i], [0]
        - [i:i+10], [0, 1] -> [0:1, 0:1, i:i+10], [0, 1]
        - [i:i+10], [0, 2] -> [0:1, i:i+10, 0:1], [0, 2]
        - [i:i+10], [0, 1, 2, 3] -> [0:1, 0:1, 0:1, 0:1, i:i+10], [0, 1, 2, 3]
        - [i:i+10], [0, 2, 3, 4] -> [0:1, i:i+10, 0:1, 0:1, 0:1], [0, 2, 3, 4]
        - [i:i+10], [0, 1, 1] -> [0:1, 0:1, 0:1, i:i+10], [0:1, 1, 2]

        :param axes: The axes where the 0:1 ranges should be added.
        :return: A list of the actual axes where the 0:1 ranges were added.
        """
        result = []
        for axis in sorted(axes):
            self.ranges.insert(axis, (0, 0, 1))
            self.tile_sizes.insert(axis, 1)

            if len(result) > 0 and result[-1] >= axis:
                result.append(result[-1] + 1)
            else:
                result.append(axis)
        return result

    def pop(self, dimensions):
        new_ranges = []
        new_tsizes = []
        for i in range(len(self.ranges)):
            if i not in dimensions:
                new_ranges.append(self.ranges[i])
                new_tsizes.append(self.tile_sizes[i])
        if not new_ranges:
            new_ranges = [(symbolic.pystr_to_symbolic(0), symbolic.pystr_to_symbolic(0), symbolic.pystr_to_symbolic(1))]
            new_tsizes = [symbolic.pystr_to_symbolic(1)]
        self.ranges = new_ranges
        self.tile_sizes = new_tsizes

    def string_list(self):
        return Range.ndslice_to_string_list(self.ranges, self.tile_sizes)

    def replace(self, repl_dict):
        for i, ((rb, re, rs), ts) in enumerate(zip(self.ranges, self.tile_sizes)):
            self.ranges[i] = (rb.subs(repl_dict) if symbolic.issymbolic(rb) else rb,
                              re.subs(repl_dict) if symbolic.issymbolic(re) else re,
                              rs.subs(repl_dict) if symbolic.issymbolic(rs) else rs)
            self.tile_sizes[i] = (ts.subs(repl_dict) if symbolic.issymbolic(ts) else ts)

    def intersects(self, other: 'Range'):
        type_error = False
        for i, (rng, orng) in enumerate(zip(self.ranges, other.ranges)):
            if (rng[2] != 1 or orng[2] != 1 or self.tile_sizes[i] != 1 or other.tile_sizes[i] != 1):
                # TODO: This function does not consider strides or tiles
                return None

            # Special case: ranges match
            if rng[0] == orng[0] or rng[1] == orng[1]:
                continue

            # Since conditions can be indeterminate, we check them separately
            # for being False, then make a check that may raise a TypeError
            cond1 = (rng[0] <= orng[1])
            cond2 = (orng[0] <= rng[1])
            # NOTE: We have to use the "==" operator because of SymPy returning
            #       a special boolean type!
            try:
                if cond1 == False or cond2 == False:
                    return False
                if not (cond1 and cond2):
                    return False
            except TypeError:  # cannot determine truth value of Relational
                type_error = True

        if type_error:
            raise TypeError("cannot determine truth value of Relational")

        return True


@dace.serialize.serializable
class Indices(Subset):
    """ A subset of one element representing a single index in an
        N-dimensional data descriptor. """
    def __init__(self, indices):
        if indices is None or len(indices) == 0:
            raise TypeError('Expected an array of index expressions: got empty' ' array or None')
        if isinstance(indices, str):
            raise TypeError("Expected collection of index expression: got str")
        elif isinstance(indices, symbolic.SymExpr):
            self.indices = indices
        else:
            self.indices = [symbolic.pystr_to_symbolic(i) for i in indices]
        self.tile_sizes = [1]

    def to_json(self):
        def a2s(obj):
            if isinstance(obj, symbolic.SymExpr):
                return str(obj.expr)
            else:
                return str(obj)

        return {'type': 'Indices', 'indices': list(map(a2s, self.indices))}

    @staticmethod
    def from_json(obj, context=None):
        if obj['type'] != 'Indices':
            raise TypeError("from_json of class \"Indices\" called on json "
                            "with type %s (expected 'Indices')" % obj['type'])

        #return Indices(symbolic.SymExpr(obj['indices']))
        return Indices([*map(symbolic.pystr_to_symbolic, obj['indices'])])

    def __hash__(self):
        return hash(tuple(i for i in self.indices))

    def num_elements(self):
        return 1

    def num_elements_exact(self):
        return 1

    def bounding_box_size(self):
        return [1] * len(self.indices)

    def size(self):
        return [1] * len(self.indices)

    def size_exact(self):
        return self.size()

    def min_element(self):
        return self.indices

    def max_element(self):
        return self.indices

    def max_element_approx(self):
        return [_approx(ind) for ind in self.indices]

    def min_element_approx(self):
        return [_approx(ind) for ind in self.indices]

    def data_dims(self):
        return 0

    def dims(self):
        return len(self.indices)

    def strides(self):
        return [1] * len(self.indices)

    def absolute_strides(self, global_shape):
        return [1] * len(self.indices)

    def offset(self, other, negative, indices=None):
        if not isinstance(other, Subset):
            if isinstance(other, (list, tuple)):
                other = Indices(other)
            else:
                other = Indices([other for _ in self.indices])
        mult = -1 if negative else 1
        for i, off in enumerate(other.min_element()):
            self.indices[i] += mult * off

    def offset_new(self, other, negative, indices=None):
        if not isinstance(other, Subset):
            if isinstance(other, (list, tuple)):
                other = Indices(other)
            else:
                other = Indices([other for _ in self.indices])
        mult = -1 if negative else 1
        return Indices([self.indices[i] + mult * off for i, off in enumerate(other.min_element())])

    def coord_at(self, i):
        """ Returns the offseted coordinates of this subset at
            the given index tuple.
            For example, the range [2:10:2] at index 2 would return 6 (2+2*2).
            :param i: A tuple of the same dimensionality as subset.dims().
            :return: Absolute coordinates for index i.
        """
        if len(i) != len(self.indices):
            raise ValueError('Invalid dimensionality of input tuple (expected'
                             ' %d, got %d)' % (len(self.indices), len(i)))
        if any([k != 0 for k in i]):
            raise ValueError('Value out of bounds')

        return tuple(r for r in self.indices)

    def at(self, i, strides):
        """ Returns the absolute index (1D memory layout) of this subset at
            the given index tuple.
            For example, the range [2:10::2] at index 2 would return 6 (2+2*2).
            :param i: A tuple of the same dimensionality as subset.dims().
            :param strides: The strides of the array we are subsetting.
            :return: Absolute 1D index at coordinate i.
        """
        coord = self.coord_at(i)

        # Return i0 + i1*size0 + i2*size1*size0 + ....
        return sum(s * strides[i] for i, s in enumerate(coord))

    def pystr(self):
        return str(self.indices)

    def __str__(self):
        return ", ".join(map(str, self.indices))

    @property
    def free_symbols(self) -> Set[str]:
        result = set()
        for dim in self.indices:
            result |= symbolic.symlist(dim).keys()
        return result

    @staticmethod
    def from_string(s):
        return Indices([symbolic.pystr_to_symbolic(m.group(0)) for m in re.finditer("[^,;:]+", s)])

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, key):
        return self.indices.__getitem__(key)

    def __setitem__(self, key, value):
        return self.indices.__setitem__(key, value)

    def __eq__(self, other):
        if not isinstance(other, Indices):
            return False
        if len(self.indices) != len(other.indices):
            return False
        return all([i == o_i for i, o_i in zip(self.indices, other.indices)])

    def reorder(self, order):
        """ Re-orders the dimensions in-place according to a permutation list.
            :param order: List or tuple of integers from 0 to self.dims() - 1,
                          indicating the desired order of the dimensions.
        """
        new_indices = [self.indices[o] for o in order]
        self.indices = new_indices

    def __ne__(self, other):
        return not self.__eq__(other)

    def ndrange(self):
        return [(i, i, 1) for i in self.indices]

    def compose(self, other):
        raise TypeError('Index subsets cannot be composed with other subsets')

    def squeeze(self, ignore_indices=None):
        ignore_indices = ignore_indices or []
        non_ones = []
        for i in range(len(self.indices)):
            if i in ignore_indices:
                non_ones.append(i)
        squeezed_indices = [self.indices[i] for i in non_ones]
        if not squeezed_indices:
            squeezed_indices = [0]
        self.indices = squeezed_indices
        return non_ones

    def unsqueeze(self, axes: Sequence[int]) -> List[int]:
        """ Adds zeroes to the subset, in the indices contained in axes.
        
        The method is mostly used to restore subsets that had their
        zero-indices removed (i.e., squeezed subsets). Hence, the method is
        called 'unsqueeze'.

        Examples (initial subset, axes -> result subset, output):
        - [i], [0] -> [0, i], [0]
        - [i], [0, 1] -> [0, 0, i], [0, 1]
        - [i], [0, 2] -> [0, i, 0], [0, 2]
        - [i], [0, 1, 2, 3] -> [0, 0, 0, 0, i], [0, 1, 2, 3]
        - [i], [0, 2, 3, 4] -> [0, i, 0, 0, 0], [0, 2, 3, 4]
        - [i], [0, 1, 1] -> [0, 0, 0, i], [0, 1, 2]

        :param axes: The axes where the zero-indices should be added.
        :return: A list of the actual axes where the zero-indices were added.
        """
        result = []
        for axis in sorted(axes):
            self.indices.insert(axis, 0)

            if len(result) > 0 and result[-1] >= axis:
                result.append(result[-1] + 1)
            else:
                result.append(axis)
        return result

    def replace(self, repl_dict):
        for i, ind in enumerate(self.indices):
            self.indices[i] = (ind.subs(repl_dict) if symbolic.issymbolic(ind) else ind)

    def pop(self, dimensions):
        new_indices = []
        for i in range(len(self.indices)):
            if i not in dimensions:
                new_indices.append(self.indices[i])
        self.indices = new_indices

    def intersects(self, other: 'Indices'):
        return all(ind == oind for ind, oind in zip(self.indices, other.indices))

    def intersection(self, other: 'Indices'):
        if self.intersects(other):
            return self
        return None


def _union_special_cases(arb: symbolic.SymbolicType, brb: symbolic.SymbolicType, are: symbolic.SymbolicType,
                         bre: symbolic.SymbolicType):
    """ 
    Special cases of subset unions. If case found, returns pair of 
    (min,max), otherwise returns None.
    """
    if are + 1 == brb:
        return (arb, bre)
    elif bre + 1 == arb:
        return (brb, are)
    return None


def bounding_box_union(subset_a: Subset, subset_b: Subset) -> Range:
    """ Perform union by creating a bounding-box of two subsets. """
    if subset_a.dims() != subset_b.dims():
        raise ValueError('Dimension mismatch between %s and %s' % (str(subset_a), str(subset_b)))

    # Check whether all expressions containing a symbolic value should
    # always be evaluated to positive. If so, union will yield
    # a different result respectively.
    symbolic_positive = Config.get('optimizer', 'symbolic_positive')

    result = []
    for arb, brb, are, bre in zip(subset_a.min_element_approx(), subset_b.min_element_approx(),
                                  subset_a.max_element_approx(), subset_b.max_element_approx()):
        # Special case
        spcase = _union_special_cases(arb, brb, are, bre)
        if spcase is not None:
            minrb, maxre = spcase
            result.append((minrb, maxre, 1))
            continue

        try:
            minrb = min(arb, brb)
        except TypeError:
            if symbolic_positive:
                if len(arb.free_symbols) == 0:
                    minrb = arb
                elif len(brb.free_symbols) == 0:
                    minrb = brb
                else:
                    raise
            else:
                raise

        try:
            maxre = max(are, bre)
        except TypeError:
            if symbolic_positive:
                if len(are.free_symbols) == 0:
                    maxre = bre
                elif len(bre.free_symbols) == 0:
                    maxre = are
                else:
                    raise
            else:
                raise

        result.append((minrb, maxre, 1))

    return Range(result)


def union(subset_a: Subset, subset_b: Subset) -> Subset:
    """ Compute the union of two Subset objects.
        If the subsets are not of the same type, degenerates to bounding-box
        union.
        :param subset_a: The first subset.
        :param subset_b: The second subset.
        :return: A Subset object whose size is at least the union of the two
                 inputs. If union failed, returns None.
    """
    try:

        if subset_a is not None and subset_b is None:
            return subset_a
        elif subset_b is not None and subset_a is None:
            return subset_b
        elif subset_a is None and subset_b is None:
            raise TypeError('Both subsets cannot be None')
        elif type(subset_a) != type(subset_b):
            return bounding_box_union(subset_a, subset_b)
        elif isinstance(subset_a, Indices):
            # Two indices. If they are adjacent, returns a range that contains both,
            # otherwise, returns a bounding box of the two
            return bounding_box_union(subset_a, subset_b)
        elif isinstance(subset_a, Range):
            # TODO(later): More involved Strided-Tiled Range union
            return bounding_box_union(subset_a, subset_b)
        else:
            warnings.warn('Unrecognized Subset type %s in union, degenerating to'
                          ' bounding box' % type(subset_a).__name__)
            return bounding_box_union(subset_a, subset_b)
    except TypeError:  # cannot determine truth value of Relational
        return None


def intersects(subset_a: Subset, subset_b: Subset) -> Union[bool, None]:
    """
    Returns True if two subsets intersect, False if they do not, or
    None if the answer cannot be determined.

    :param subset_a: The first subset.
    :param subset_b: The second subset.
    :return: True if subsets intersect, False if not, None if indeterminate.
    """
    try:
        if subset_a is None or subset_b is None:
            return False
        if isinstance(subset_a, Indices):
            subset_a = Range.from_indices(subset_a)
        if isinstance(subset_b, Indices):
            subset_b = Range.from_indices(subset_b)
        if type(subset_a) is type(subset_b):
            return subset_a.intersects(subset_b)
        return None
    except TypeError:  # cannot determine truth value of Relational
        return None
