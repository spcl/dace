from dace import data, symbolic, types
import re
import sympy as sp
from functools import reduce
from sympy.core.sympify import SympifyError


class Subset(object):
    """ Defines a subset of a data descriptor. """

    def covers(self, other):
        """ Returns True if this subset covers (using a bounding box) another
            subset. """
        try:
            return all([
                rb <= orb and re >= ore for rb, re, orb, ore in zip(
                    self.min_element(), self.max_element(),
                    other.min_element(), other.max_element())
            ])
        except TypeError:
            return False

    def __repr__(self):
        return '%s (%s)' % (type(self).__name__, self.__str__())


def _simplified_str(val):
    try:
        return str(int(val))
    except TypeError:
        return str(val)


def _expr(val):
    if isinstance(val, symbolic.SymExpr):
        return val.expr
    return val


def _tuple_to_symexpr(val):
    return (symbolic.SymExpr(val[0], val[1])
            if isinstance(val, tuple) else symbolic.pystr_to_symbolic(val))


class Range(Subset):
    """ Subset defined in terms of a fixed range. """

    def __init__(self, ranges):
        parsed_ranges = []
        parsed_tiles = []
        for r in ranges:
            if len(r) != 3 and len(r) != 4:
                raise ValueError("Expected 3-tuple or 4-tuple")
            parsed_ranges.append((_tuple_to_symexpr(r[0]),
                                  _tuple_to_symexpr(r[1]),
                                  _tuple_to_symexpr(r[2])))
            if len(r) == 3:
                parsed_tiles.append(symbolic.pystr_to_symbolic(1))
            else:
                parsed_tiles.append(symbolic.pystr_to_symbolic(r[3]))
        self.ranges = parsed_ranges
        self.tile_sizes = parsed_tiles

    @staticmethod
    def from_array(array):
        """ Constructs a range that covers the full array given as input. 
            @type array: dace.data.Data """
        return Range([(0, s - 1, 1) for s in array.shape])

    def __hash__(self):
        return hash(tuple(r for r in self.ranges))

    def __add__(self, other):
        sum_ranges = self.ranges + other.ranges
        return Range(sum_ranges)

    def num_elements(self):
        return reduce(sp.mul.Mul, self.size(), 1)

    def size(self, for_codegen=False):
        """ Returns the number of elements in each dimension. """
        if for_codegen == True:
            int_ceil = sp.Function('int_ceil')
            return [
                ts * int_ceil(
                    ((iMax.approx
                      if isinstance(iMax, symbolic.SymExpr) else iMax) + 1 -
                     (iMin.approx
                      if isinstance(iMin, symbolic.SymExpr) else iMin)),
                    (step.approx
                     if isinstance(step, symbolic.SymExpr) else step))
                for (iMin, iMax, step), ts in zip(self.ranges, self.tile_sizes)
            ]
        else:
            return [
                ts * sp.ceiling(
                    ((iMax.approx
                      if isinstance(iMax, symbolic.SymExpr) else iMax) + 1 -
                     (iMin.approx
                      if isinstance(iMin, symbolic.SymExpr) else iMin)) /
                    (step.approx
                     if isinstance(step, symbolic.SymExpr) else step))
                for (iMin, iMax, step), ts in zip(self.ranges, self.tile_sizes)
            ]

    def bounding_box_size(self):
        """ Returns the size of a bounding box around this range. """
        return [
            # sp.floor((iMax - iMin) / step) - iMin
            ts * ((iMax.approx
                   if isinstance(iMax, symbolic.SymExpr) else iMax) -
                  (iMin.approx
                   if isinstance(iMin, symbolic.SymExpr) else iMin) + 1)
            for (iMin, iMax, step), ts in zip(self.ranges, self.tile_sizes)
        ]

    def min_element(self):
        return [_expr(x[0]) for x in self.ranges]

    def max_element(self):
        return [_expr(x[1]) for x in self.ranges]
        # return [(sp.floor((iMax - iMin) / step) - 1) * step
        #        for iMin, iMax, step in self.ranges]

    def coord_at(self, i):
        """ Returns the offseted coordinates of this subset at
            the given index tuple.
            
            For example, the range [2:10:2] at index 2 would return 6 (2+2*2).
            
            @param i: A tuple of the same dimensionality as subset.dims() or
                      subset.data_dims().
            @return: Absolute coordinates for index i (length equal to
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

        return tuple(
            _expr(rb) + k * _expr(rs)
            for k, (rb, _, rs) in zip(i, self.ranges)) + tuple(ti)

    def at(self, i, global_shape):
        """ Returns the absolute index (1D memory layout) of this subset at
            the given index tuple.
            
            For example, the range [2:10:2] at index 2 would return 6 (2+2*2).
            
            @param i: A tuple of the same dimensionality as subset.dims() or
                      subset.data_dims().
            @param global_shape: The full size of the set that we are 
                                 subsetting (e.g., full array strides/padded 
                                 shape).
            @return: Absolute 1D index at coordinate i.
        """
        coord = self.coord_at(i)

        # Return i0 + i1*size0 + i2*size1*size0 + ....
        # Cancel out stride since we determine the initial offset only here
        return sum(
            _expr(s) * _expr(astr) / _expr(rs) for s, (_, _, rs), astr in zip(
                coord, self.ranges, self.absolute_strides(global_shape)))

    def data_dims(self):
        return (
            sum(1 if (re - rb + 1) != 1 else 0
                for rb, re, _ in self.ranges) + sum(1 if ts != 1 else 0
                                                    for ts in self.tile_sizes))

    def offset(self, other, negative):
        if not isinstance(other, Subset):
            other = Indices([other for _ in self.ranges])
        mult = -1 if negative else 1
        for i, off in enumerate(other.min_element()):
            rb, re, rs = self.ranges[i]
            self.ranges[i] = (rb + mult * off, re + mult * off, rs)

    def dims(self):
        return len(self.ranges)

    def absolute_strides(self, global_shape):
        """ Returns a list of strides for advancing one element in each
            dimension. Size of the list is equal to `data_dims()`, which may
            be larger than `dims()` depending on tile sizes. """
        # ..., stride2*size1*size0, stride1*size0, stride0, ..., tile strides
        return [
            rs * reduce(sp.mul.Mul, global_shape[i + 1:], 1)
            for i, (_, _, rs) in enumerate(self.ranges)
        ] + [
            reduce(sp.mul.Mul, global_shape[i + 1:], 1)
            for i, ts in enumerate(self.tile_sizes) if ts != 1
        ]

    def strides(self):
        return [rs for _, _, rs in self.ranges]

    @staticmethod
    def _range_pystr(range):
        return "(" + ", ".join(map(str, range)) + ")"

    def pystr(self):
        return "[" + ", ".join(map(Range._range_pystr, self.ranges)) + "]"

    @property
    def free_symbols(self):
        result = set()
        for dim in self.ranges:
            for d in dim:
                result.update(set(symbolic.symlist(d)))
        return result

    def reorder(self, order):
        """ Re-orders the dimensions in-place according to a permutation list.
            @param order: List or tuple of integers from 0 to self.dims() - 1,
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
                    dres += ':' + _simplified_str(d[1] + 1)
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
                ranges.append((symbolic.pystr_to_symbolic(uni_dim_tokens[0]),
                               symbolic.pystr_to_symbolic(uni_dim_tokens[0]),
                               1))
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
                    raise SyntaxError(
                        "Invalid range: {}".format(multi_dim_tokens))
            # Parse tokens
            try:
                if isinstance(tokens[0], tuple):
                    begin = symbolic.SymExpr(tokens[0][0], tokens[0][1])
                else:
                    begin = symbolic.pystr_to_symbolic(tokens[0])
                if isinstance(tokens[1], tuple):
                    end = symbolic.SymExpr(tokens[1][0], tokens[1][1]) - 1
                else:
                    end = symbolic.pystr_to_symbolic(tokens[1]) - 1
                if len(tokens) >= 3:
                    if isinstance(tokens[2], tuple):
                        step = symbolic.SymExpr(tokens[2][0], tokens[2][1])
                    else:
                        step = symbolic.SymExpr(tokens[2])
                else:
                    step = 1
                if len(tokens) >= 4:
                    if isinstance(tokens[3], tuple):
                        tsize = tokens[3][0]
                    else:
                        tsize = tokens[3]
                else:
                    tsize = 1
            except SympifyError:
                raise SyntaxError("Invalid range: {}".format(string))
            # Append range
            ranges.append((begin, end, step, tsize))

        return Range(ranges)

    @staticmethod
    def ndslice_to_string(slice, tile_sizes=None):
        if tile_sizes is None:
            return ", ".join([Range.dim_to_string(s) for s in slice])
        return ", ".join(
            [Range.dim_to_string(s, t) for s, t in zip(slice, tile_sizes)])

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
                    for (rb, re, rs), (orb, ore,
                                       ors) in zip(self.ranges, other.ranges)])

    def __ne__(self, other):
        return not self.__eq__(other)

    def compose(self, other):
        if not isinstance(other, Subset):
            raise TypeError("Cannot compose ranges with non-subsets")
        if self.data_dims() != other.dims():
            raise ValueError("Dimension mismatch in composition")
        new_subset = []
        idx = 0
        for (rb, re, rs), rt in zip(self.ranges, self.tile_sizes):
            if re - rb == 0:
                if isinstance(other, Indices):
                    new_subset.append(rb)
                else:
                    new_subset.append((rb, re, rs, rt))
            else:
                if isinstance(other[idx], tuple):
                    new_subset.append(
                        (rb + rs * other[idx][0], rb + rs * other[idx][1],
                         rs * other[idx][2], rt))
                else:
                    new_subset.append(rb + rs * other[idx])
                idx += 1
        if isinstance(other, Range):
            return Range(new_subset)
        elif isinstance(other, Indices):
            return Indices(new_subset)
        else:
            raise NotImplementedError


class Indices(Subset):
    """ A subset of one element representing a single index in an
        N-dimensional data descriptor. """

    def __init__(self, indices):
        if indices is None or len(indices) == 0:
            raise TypeError('Expected an array of index expressions: got empty'
                            ' array or None')
        if isinstance(indices, str):
            raise TypeError("Expected collection of index expression: got str")
        if isinstance(indices, tuple):
            self.indices = symbolic.SymExpr(indices[0], indices[1])
        else:
            self.indices = symbolic.pystr_to_symbolic(indices)
        self.tile_sizes = [1]

    def __hash__(self):
        return hash(tuple(i for i in self.indices))

    def num_elements(self):
        return 1

    def bounding_box_size(self):
        return [1] * len(self.indices)

    def size(self):
        return [1] * len(self.indices)

    def min_element(self):
        return self.indices

    def max_element(self):
        return self.indices

    def data_dims(self):
        return 0

    def dims(self):
        return len(self.indices)

    def strides(self):
        return [1] * len(self.indices)

    def absolute_strides(self, global_shape):
        return [1] * len(self.indices)

    def offset(self, other, negative):
        if not isinstance(other, Subset):
            other = Indices([other for _ in self.indices])
        mult = -1 if negative else 1
        for i, off in enumerate(other.min_element()):
            self.indices[i] += mult * off

    def coord_at(self, i):
        """ Returns the offseted coordinates of this subset at
            the given index tuple.
            For example, the range [2:10:2] at index 2 would return 6 (2+2*2).
            @param i: A tuple of the same dimensionality as subset.dims().
            @return: Absolute coordinates for index i.
        """
        if len(i) != len(self.indices):
            raise ValueError('Invalid dimensionality of input tuple (expected'
                             ' %d, got %d)' % (len(self.indices), len(i)))
        if any([k != 0 for k in i]):
            raise ValueError('Value out of bounds')

        return tuple(r for r in self.indices)

    def at(self, i, global_shape):
        """ Returns the absolute index (1D memory layout) of this subset at
            the given index tuple.
            For example, the range [2:10::2] at index 2 would return 6 (2+2*2).
            @param i: A tuple of the same dimensionality as subset.dims().
            @param global_shape: The full size of the set that we are 
                                 subsetting (e.g., full array strides/padded 
                                 shape).
            @return: Absolute 1D index at coordinate i.
        """
        coord = self.coord_at(i)

        # Return i0 + i1*size0 + i2*size1*size0 + ....
        return sum(s * reduce(sp.mul.Mul, global_shape[i + 1:], 1)
                   for i, s in enumerate(coord))

    def pystr(self):
        return str(self.indices)

    def __str__(self):
        return ", ".join(map(str, self.indices))

    @property
    def free_symbols(self):
        result = set()
        for dim in self.indices:
            result.update(set(symbolic.symlist(d)))
        return result

    @staticmethod
    def from_string(s):
        return Indices([
            symbolic.pystr_to_symbolic(m.group(0))
            for m in re.finditer("[^,;:]+", s)
        ])

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
            @param order: List or tuple of integers from 0 to self.dims() - 1,
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


def bounding_box_union(subset_a: Subset, subset_b: Subset) -> Range:
    """ Perform union by creating a bounding-box of two subsets. """
    if subset_a.dims() != subset_b.dims():
        raise ValueError('Dimension mismatch between %s and %s' %
                         (str(subset_a), str(subset_b)))

    result = [(min(arb, brb), max(are, bre), 1) for arb, brb, are, bre in zip(
        subset_a.min_element(), subset_b.min_element(), subset_a.max_element(),
        subset_b.max_element())]
    return Range(result)
