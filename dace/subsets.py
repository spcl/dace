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


def nng(expr):
    # When dealing with set sizes, assume symbols are non-negative
    if hasattr(expr, 'free_symbols'):
        # TODO: Fix in symbol definition, not here
        return expr.subs(((sym, sp.Symbol(sym.name, nonnegative=True)) for sym in list(expr.free_symbols)))
    return expr


def bounding_box_cover_exact(subset_a, subset_b, approximation=False) -> bool:
    """Test if `subset_a` covers `subset_b`.

    The function uses a bounding box to test if `subset_a` covers `subset_b`,
    i.e. that `subset_a` is at least as big as `subset_b`. By default the
    box is constructed using `{min, max}_element()` or if `approximation` is
    `True` `{min, max}_element_approx()`. The most important difference compared
    to `bounding_box_cover_exact()` is that this function does not assume
    that the symbols are positive.

    The function returns `True` if it _can be shown_ that `subset_a` covers `subset_b`
    and `False` otherwise.

    :param subset_a: The first subset, the one that should cover.
    :param subset_b: The second subset, the one that should be convered.
    :param approximation: If `True` then use the approximated bounds.
    """
    min_elements_a = subset_a.min_element_approx() if approximation else subset_a.min_element()
    max_elements_a = subset_a.max_element_approx() if approximation else subset_a.max_element()
    min_elements_b = subset_b.min_element_approx() if approximation else subset_b.min_element()
    max_elements_b = subset_b.max_element_approx() if approximation else subset_b.max_element()

    # Covering only make sense if the two subsets have the same number of dimensions.
    if len(min_elements_a) != len(min_elements_b):
        return ValueError(f"A bounding box of dimensionality {len(min_elements_a)} cannot"
                          f" test covering a bounding box of dimensionality {len(min_elements_b)}.")

    # NOTE: The original implementation always called `nng()`. However, it was decided that
    #   this is an error and the call was removed in PR#2093.
    simplify = lambda expr: symbolic.simplify_ext(expr)
    no_simplify = lambda expr: expr

    # NOTE: Just doing the check is very fast, compared to simplify. Thus we first try to do the
    #   matching without running if this does not work, then we try again with simplify.
    for simp_fun in [no_simplify, simplify]:
        if all((simp_fun(rb) <= simp_fun(orb)) == True and (simp_fun(re) >= simp_fun(ore)) == True
               for rb, re, orb, ore in zip(min_elements_a, max_elements_a, min_elements_b, max_elements_b)):
            return True
    return False


def bounding_box_symbolic_positive(subset_a, subset_b, approximation=False) -> bool:
    """Checks if `subset_a` covers `subset_b` using positivity assumption.

    The function uses a bounding box to test if `subset_a` covers `subset_b`,
    i.e. that `subset_a` is at least as big as `subset_b`. By default the
    box is constructed using `{min, max}_element()` or if `approximation` is
    `True` `{min, max}_element_approx()`. The function will perform the
    covering check under the assumption that all symbols are positive,
    which is the main difference to `bounding_box_cover_exact()`.

    The function returns `True` if it _can be shown_ that `subset_a` covers `subset_b`
    and `False` otherwise.

    :param subset_a: The first subset, the one that should cover.
    :param subset_b: The second subset, the one that should be convered.
    :param approximation: If `True` then use the approximated bounds.

    :note: In previous versions this function raised `TypeError` in some cases
        when a truth value could not be determined. This behaviour was removed,
        since the `bounding_box_cover_exact()` does not show this behaviour.
    """
    min_elements_a = subset_a.min_element_approx() if approximation else subset_a.min_element()
    max_elements_a = subset_a.max_element_approx() if approximation else subset_a.max_element()
    min_elements_b = subset_b.min_element_approx() if approximation else subset_b.min_element()
    max_elements_b = subset_b.max_element_approx() if approximation else subset_b.max_element()

    # Covering only make sense if the two subsets have the same number of dimensions.
    if len(min_elements_a) != len(min_elements_b):
        return ValueError(f"A bounding box of dimensionality {len(min_elements_a)} cannot"
                          f" test covering a bounding box of dimensionality {len(min_elements_b)}.")

    # NOTE: `nng()` is applied inside the loop.
    simplify = lambda expr: symbolic.simplify_ext(expr)
    no_simplify = lambda expr: expr

    for rb, re, orb, ore in zip(min_elements_a, max_elements_a, min_elements_b, max_elements_b):
        # NOTE: Applying simplify takes a lot of time, thus we try to avoid it and try to do the test
        #   first with the symbols we get and if we are unable to figuring out something, we run
        #   simplify. Furthermore, we also try to postpone `nng()` as long as we can.
        # NOTE: We use first `==` in the hope that it is much faster than `<=`.
        # NOTE: We have to use the `== True` test because of SymPy's behaviour. Otherwise we would
        #   get an expression resulting in a `TypeError`.

        # lower bound: first check whether symbolic positive condition applies
        if not (len(rb.free_symbols) == 0 and len(orb.free_symbols) == 1):
            rb, orb = nng(rb), nng(orb)
            for simp_fun in [no_simplify, simplify]:
                simp_rb, simp_orb = simp_fun(rb), simp_fun(orb)
                if (simp_rb == simp_orb) == True:
                    break
                elif (simp_rb <= simp_orb) == True:
                    break
            else:
                # We were unable to determine covering for that dimension.
                #  Thus we assume that there is no covering.
                return False

        # upper bound: first check whether symbolic positive condition applies
        if not (len(re.free_symbols) == 1 and len(ore.free_symbols) == 0):
            re, ore = nng(re), nng(ore)
            for simp_fun in [no_simplify, simplify]:
                simp_re, simp_ore = simp_fun(re), simp_fun(ore)
                if (simp_re == simp_ore) == True:
                    break
                elif (simp_re >= simp_ore) == True:
                    break
            else:
                return False

    return True


class Subset(object):
    """ Defines a subset of a data descriptor. """

    def covers(self, other):
        """ Returns True if this subset covers (using a bounding box) another
            subset. """

        # Subsets of different dimensionality can never cover each other.
        if self.dims() != other.dims():
            return ValueError(
                f"A subset of dimensionality {self.dims()} cannot test covering a subset of dimensionality {other.dims()}"
            )

        if Config.get('optimizer', 'symbolic_positive'):
            return bounding_box_symbolic_positive(self, other, approximation=True)
        else:
            return bounding_box_cover_exact(self, other, approximation=True)

    def covers_precise(self, other):
        """ Returns True if self contains all the elements in other. """

        # Subsets of different dimensionality can never cover each other.
        if self.dims() != other.dims():
            return ValueError(
                f"A subset of dimensionality {self.dims()} cannot test covering a subset of dimensionality {other.dims()}"
            )

        # If self does not cover other with a bounding box union, return false.
        symbolic_positive = Config.get('optimizer', 'symbolic_positive')
        if symbolic_positive and (not bounding_box_cover_exact(self, other)):
            return False
        elif not bounding_box_symbolic_positive(self, other):
            return False

        # NOTE: The original implementation always called `nng()`. However, it was decided that
        #   and the application was made conditional on `symbolic_positive`, in PR#2093.
        simplify = (lambda expr: symbolic.simplify_ext(nng(expr))) if symbolic_positive else (
            lambda expr: symbolic.simplify_ext(expr))
        no_simplify = lambda expr: expr

        # In the following we will first perform the check as is, and if that fails try it again
        #   with simplify. We do it because simplify is a very expensive operation and we try to
        #   avoid calling it.
        try:
            # if self is an index no further distinction is needed
            if isinstance(self, Indices):
                return True

            elif isinstance(self, Range):
                # other is an index so we need to check if the step of self is such that other is covered
                # self.start % self.step == other.index % self.step
                if isinstance(other, Indices):
                    # TODO: Think if inverting the order is simpler.
                    for simp_fun in [no_simplify, simplify]:
                        for (start, _, step), i in zip(self.ranges, other.indices):
                            simp_step = simp_fun(step)
                            simp_start = simp_fun(start)
                            simp_i = simp_fun(i)
                            if not (((simp_start % simp_step) == (simp_i % simp_step)) == True):
                                return False
                    return True

                else:
                    assert isinstance(other, Range)
                    # other is a range so in every dimension self.step has to divide other.step and
                    # self.start % self.step = other.start % other.step
                    self_steps = [r[2] for r in self.ranges]
                    other_steps = [r[2] for r in other.ranges]
                    starts = self.min_element()
                    ostarts = other.min_element()

                    for i, simp_fun in enumerate([no_simplify, simplify]):
                        try:
                            for start, step, ostart, ostep in zip(starts, self_steps, ostarts, other_steps):
                                simp_start = simp_fun(start)
                                simp_ostart = simp_fun(ostart)
                                if not (ostep % step == 0 and
                                        ((simp_start == simp_ostart) or
                                         (simp_start % simp_fun(step) == simp_ostart % simp_fun(ostep)) == True)):
                                    return False
                        except TypeError:
                            # If a `TypeError happens during the "no simplify" phase, we immediately
                            #   go to the simplify phase, in the hope that it might be possible to
                            #   simplify the expression more. If we are already using simplify, then
                            #   we return `False`.
                            if i == 0:
                                continue
                            return False
                    return True
            else:
                raise ValueError(
                    f'Does not know how to compare a `{type(self).__name__}` with a `{type(other).__name__}`.')

        except TypeError:
            return False

    def __repr__(self):
        return '%s (%s)' % (type(self).__name__, self.__str__())

    def offset(self, other, negative, indices=None, offset_end=True):
        raise NotImplementedError

    def offset_new(self, other, negative, indices=None, offset_end=True):
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
        result = Range([(0, s - 1, 1) for s in array.shape])
        if any(o != 0 for o in array.offset):
            result.offset(array.offset, True)
        return result

    def __hash__(self):
        return hash(tuple(r for r in self.ranges))

    def __add__(self, other):
        sum_ranges = self.ranges + other.ranges
        return Range(sum_ranges)

    def __deepcopy__(self, memo) -> 'Range':
        """Performs a deepcopy of `self`.

        For performance reasons only the mutable parts are copied.
        """
        # Because SymPy expression and numbers and tuple in Python are immutable, it is enough
        #  to shallow copy the list that stores them.
        node = object.__new__(Range)
        node.ranges = self.ranges.copy()
        node.tile_sizes = self.tile_sizes.copy()

        return node

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

    def offset(self, other, negative, indices=None, offset_end=True):
        if other is None:
            return
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
            if offset_end:
                re = re + mult * off[i]
            self.ranges[i] = (rb + mult * off[i], re, rs)

    def offset_new(self, other, negative, indices=None, offset_end=True):
        if other is None:
            return Range(self.ranges)
        if not isinstance(other, Subset):
            if isinstance(other, (list, tuple)):
                other = Indices(other)
            else:
                other = Indices([other for _ in self.ranges])
        mult = -1 if negative else 1
        if indices is None:
            indices = set(range(len(self.ranges)))
        off = other.min_element()
        return Range([(self.ranges[i][0] + mult * off[i], self.ranges[i][1] if not offset_end else
                       (self.ranges[i][1] + mult * off[i]), self.ranges[i][2]) for i in indices])

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

    def get_free_symbols_by_indices(self, indices: List[int]) -> Set[str]:
        """
        Get set of free symbols by only looking at the dimension given by the indices list

        :param indices: The indices of the dimensions to look at
        :type indices: List[int]
        :return: The set of free symbols
        :rtype: Set[str]
        """
        result = set()
        for i, dim in enumerate(self.ranges):
            if i in indices:
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
                value = symbolic.pystr_to_symbolic(uni_dim_tokens[0].strip())
                ranges.append((value, value, 1))
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

    def compose(self, other):
        if not isinstance(other, Subset):
            raise TypeError("Cannot compose ranges with non-subsets")

        new_subset = []
        if self.data_dims() == other.dims():
            # case 1: subsets may differ in dimensions, but data_dims correspond
            #         to other dims -> all non-data dims are cut out
            idx = 0
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

    def squeeze(self, ignore_indices: Optional[List[int]] = None, offset: bool = True) -> List[int]:
        """
        Removes size-1 ranges from the subset and returns a list of dimensions that remain.

        For example, ``[i:i+10, j]`` will change the range to ``[i:i+10]`` and return ``[0]``.
        If ``offset`` is True, the subset will become ``[0:10]``.

        :param ignore_indices: An iterable of dimensions to not include in squeezing.
        :param offset: If True, will offset the non-ignored indices back so that they start with 0.
        :return: A list of dimension indices in the original subset, which remain in the squeezed result.
        """
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
            raise TypeError('Expected an array of index expressions: got empty'
                            ' array or None')
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

    def __deepcopy__(self, memo) -> 'Indices':
        """Performs a deepcopy of `self`.

        For performance reasons only the mutable parts are copied.
        """
        # Because SymPy expression and numbers and tuple in Python are immutable, it is enough
        #  to shallow copy the list that stores them.
        node = object.__new__(Indices)
        node.indices = self.indices.copy()
        return node

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

    def offset(self, other, negative, indices=None, offset_end=True):
        if not isinstance(other, Subset):
            if isinstance(other, (list, tuple)):
                other = Indices(other)
            else:
                other = Indices([other for _ in self.indices])
        mult = -1 if negative else 1
        for i, off in enumerate(other.min_element()):
            self.indices[i] += mult * off

    def offset_new(self, other, negative, indices=None, offset_end=True):
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


class SubsetUnion(Subset):
    """
    Wrapper subset type that stores multiple Subsets in a list.
    """

    def __init__(self, subset):
        self.subset_list: list[Subset] = []
        if isinstance(subset, SubsetUnion):
            self.subset_list = subset.subset_list
        elif isinstance(subset, list):
            for subset in subset:
                if not subset:
                    break
                if isinstance(subset, (Range, Indices)):
                    self.subset_list.append(subset)
                else:
                    raise NotImplementedError
        elif isinstance(subset, (Range, Indices)):
            self.subset_list = [subset]

    def covers(self, other):
        """
        Returns True if this SubsetUnion covers another subset (using a bounding box).
        If other is another SubsetUnion then self and other will
        only return true if self is other. If other is a different type of subset
        true is returned when one of the subsets in self is equal to other.
        """

        if isinstance(other, SubsetUnion):
            for subset in self.subset_list:
                # check if ther is a subset in self that covers every subset in other
                if all(subset.covers(s) for s in other.subset_list):
                    return True
            # return False if that's not the case for any of the subsets in self
            return False
        else:
            return any(s.covers(other) for s in self.subset_list)

    def covers_precise(self, other):
        """
        Returns True if this SubsetUnion covers another
        subset. If other is another SubsetUnion then self and other will
        only return true if self is other. If other is a different type of subset
        true is returned when one of the subsets in self is equal to other
        """

        if isinstance(other, SubsetUnion):
            for subset in self.subset_list:
                # check if ther is a subset in self that covers every subset in other
                if all(subset.covers_precise(s) for s in other.subset_list):
                    return True
            # return False if that's not the case for any of the subsets in self
            return False
        else:
            return any(s.covers_precise(other) for s in self.subset_list)

    def __str__(self):
        string = ''
        for subset in self.subset_list:
            if not string == '':
                string += " "
            string += subset.__str__()
        return string

    def dims(self):
        if not self.subset_list:
            return 0
        return next(iter(self.subset_list)).dims()

    def union(self, other: Subset):
        """In place union of self with another Subset"""
        try:
            if isinstance(other, SubsetUnion):
                self.subset_list += other.subset_list
            elif isinstance(other, Indices) or isinstance(other, Range):
                self.subset_list.append(other)
            else:
                raise TypeError
        except TypeError:  # cannot determine truth value of Relational
            return None

    @property
    def free_symbols(self) -> Set[str]:
        result = set()
        for subset in self.subset_list:
            result |= subset.free_symbols
        return result

    def replace(self, repl_dict):
        for subset in self.subset_list:
            subset.replace(repl_dict)

    def num_elements(self):
        # TODO: write something more meaningful here
        min = 0
        for subset in self.subset_list:
            try:
                if subset.num_elements() < min or min == 0:
                    min = subset.num_elements()
            except:
                continue

        return min


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
                    minrb = sympy.Min(arb, brb)
            else:
                minrb = sympy.Min(arb, brb)

        try:
            maxre = max(are, bre)
        except TypeError:
            if symbolic_positive:
                if len(are.free_symbols) == 0:
                    maxre = bre
                elif len(bre.free_symbols) == 0:
                    maxre = are
                else:
                    maxre = sympy.Max(are, bre)
            else:
                maxre = sympy.Max(are, bre)

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
        elif isinstance(subset_a, SubsetUnion) or isinstance(subset_b, SubsetUnion):
            return list_union(subset_a, subset_b)
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


def list_union(subset_a: Subset, subset_b: Subset) -> Subset:
    """
    Returns the union of two Subset lists.

    :param subset_a: The first subset.
    :param subset_b: The second subset.
    :return: A SubsetUnion object that contains all elements of subset_a and subset_b.
    """
    # TODO(later): Merge subsets in both lists if possible
    try:
        if subset_a is not None and subset_b is None:
            return subset_a
        elif subset_b is not None and subset_a is None:
            return subset_b
        elif subset_a is None and subset_b is None:
            raise TypeError('Both subsets cannot be None')
        elif type(subset_a) != type(subset_b):
            if isinstance(subset_b, SubsetUnion):
                return SubsetUnion(subset_b.subset_list.append(subset_a))
            else:
                return SubsetUnion(subset_a.subset_list.append(subset_b))
        elif isinstance(subset_a, SubsetUnion):
            return SubsetUnion(subset_a.subset_list + subset_b.subset_list)
        else:
            return SubsetUnion([subset_a, subset_b])

    except TypeError:
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
