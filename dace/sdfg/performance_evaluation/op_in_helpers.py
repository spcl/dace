# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains class CacheLineTracker which keeps track of all arrays of an SDFG and their cache line position
and class AccessStack which which corresponds to the stack used to compute the stack distance.
Further, provides a curve fitting method and plotting function. """

import warnings
from dace.data import Array
import sympy as sp
from collections import deque
from scipy.optimize import curve_fit
import numpy as np
from dace import symbol


class CacheLineTracker:
    """ A CacheLineTracker maps data container accesses to the corresponding accessed cache line. """

    def __init__(self, L) -> None:
        self.array_info = {}
        self.start_lines = {}
        self.next_free_line = 0
        self.L = L

    def add_array(self, name: str, a: Array, mapping):
        if name not in self.start_lines:
            # new array encountered
            self.array_info[name] = a
            self.start_lines[name] = self.next_free_line
            # increase next_free_line
            self.next_free_line += (a.total_size.subs(mapping) * a.dtype.bytes + self.L - 1) // self.L  # ceil division

    def cache_line_id(self, name: str, access: [int], mapping):
        arr = self.array_info[name]
        one_d_index = 0
        for dim in range(len(access)):
            i = access[dim]
            one_d_index += (i + sp.sympify(arr.offset[dim]).subs(mapping)) * sp.sympify(arr.strides[dim]).subs(mapping)

        # divide by L to get the cache line id
        return self.start_lines[name] + (one_d_index * arr.dtype.bytes) // self.L

    def copy(self):
        new_clt = CacheLineTracker(self.L)
        new_clt.array_info = dict(self.array_info)
        new_clt.start_lines = dict(self.start_lines)
        new_clt.next_free_line = self.next_free_line
        return new_clt


class Node:

    def __init__(self, val: int, n=None) -> None:
        self.v = val
        self.next = n


class AccessStack:
    """ A stack of cache line ids. For each memory access, we search the corresponding cache line id
    in the stack, report its distance and move it to the top of the stack. If the id was not found,
    we report a distance of -1. """

    def __init__(self, C) -> None:
        self.top = None
        self.num_calls = 0
        self.length = 0
        self.C = C

    def touch(self, id):
        self.num_calls += 1
        curr = self.top
        prev = None
        found = False
        distance = 0
        while curr is not None:
            # check if we found id
            if curr.v == id:
                # take curr node out
                if prev is not None:
                    prev.next = curr.next
                    curr.next = self.top
                    self.top = curr

                found = True
                break

            # iterate further
            prev = curr
            curr = curr.next
            distance += 1

        if not found:
            # we accessed this cache line for the first time ever
            self.top = Node(id, self.top)
            self.length += 1
            distance = -1

        return distance

    def in_cache_as_list(self):
        """
        Returns a list of cache ids currently in cache. Index 0 is the most recently used.
        """
        res = deque()
        curr = self.top
        dist = 0
        while curr is not None and dist < self.C:
            res.append(curr.v)
            curr = curr.next
            dist += 1
        return res

    def debug_print(self):
        # prints the whole stack
        print('\n')
        curr = self.top
        while curr is not None:
            print(curr.v, end=', ')
            curr = curr.next
        print('\n')

    def copy(self):
        new_stack = AccessStack(self.C)
        cache_content = self.in_cache_as_list()
        if len(cache_content) > 0:
            new_top_value = cache_content.popleft()
            new_stack.top = Node(new_top_value)
            curr = new_stack.top
            for x in cache_content:
                curr.next = Node(x)
                curr = curr.next
        return new_stack


def plot(x, work_map, cache_misses, op_in_map, symbol_name, C, L, sympy_f, element, name):
    plt = None
    try:
        import matplotlib.pyplot as plt_import
        plt = plt_import
    except ModuleNotFoundError:
        pass

    if plt is None:
        warnings.warn('Plotting only possible with matplotlib installed')
        return

    work_map = work_map[element]
    cache_misses = cache_misses[element]
    op_in_map = op_in_map[element]
    sympy_f = sympy_f[element]

    a = np.linspace(1, max(x) + 5, max(x) * 4)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(x, cache_misses, label=f'C={C*L}, L={L}')
    b = []
    for curr in a:
        b.append(sp.N(sp.sympify(sympy_f).subs(symbol_name, curr)))
    ax[0].plot(a, b)

    c = []
    for i, curr in enumerate(x):
        if work_map[0].subs(symbol_name, curr) == 0:
            c.append(0)
        elif (cache_misses[i] * L) == 0:
            c.append(9999)
        else:
            c.append(work_map[0].subs(symbol_name, curr) / (cache_misses[i] * L))
    c = np.array(c).astype(np.float64)

    ax[1].scatter(x, c, label=f'C={C*L}, L={L}')
    b = []
    for curr in a:
        b.append(sp.N(sp.sympify(op_in_map).subs(symbol_name, curr)))
    ax[1].plot(a, b)

    ax[0].set_ylim(bottom=0, top=max(cache_misses) + max(cache_misses) / 10)
    ax[0].set_xlim(left=0, right=max(x) + 1)
    ax[0].set_xlabel(symbol_name)
    ax[0].set_ylabel('Number of Cache Misses')
    ax[0].set_title(name)
    ax[0].legend(fancybox=True, framealpha=0.5)

    ax[1].set_ylim(bottom=0, top=max(c) + max(c) / 10)
    ax[1].set_xlim(left=0, right=max(x) + 1)
    ax[1].set_xlabel(symbol_name)
    ax[1].set_ylabel('Operational Intensity')
    ax[1].set_title(name)

    fig.show()


def compute_mape(f, test_x, test_y, test_set_size):
    total_error = 0
    for i in range(test_set_size):
        pred = f(test_x[i])
        err = abs(test_y[i] - pred)
        total_error += err / test_y[i]
    return total_error / test_set_size


def r_squared(pred, y):
    if np.sum(np.square(y - y.mean())) <= 0.0001:
        return 1
    return 1 - np.sum(np.square(y - pred)) / np.sum(np.square(y - y.mean()))


def find_best_model(x, y, I, J, symbol_name):
    """ Find the best model out of all combinations of (i, j) from I and J via leave-one-out cross validation. """
    min_error = None
    for i in I:
        for j in J:
            # current model
            if i == 0 and j == 0:

                def f(x, b):
                    return b * np.ones_like(x)
            else:

                def f(x, c, b):
                    return c * np.power(x, i) * np.power(np.log2(x), j) + b

            error_sum = 0
            for left_out in range(len(x)):
                xx = np.delete(x, left_out)
                yy = np.delete(y, left_out)
                try:
                    param, _ = curve_fit(f, xx, yy)

                    # predict on left out sample
                    pred = f(x[left_out], *param)
                    squared_error = np.square(pred - y[left_out])
                    error_sum += squared_error
                except RuntimeError:
                    # triggered if no fit was found --> give huge error
                    error_sum += 999999

            mean_error = error_sum / len(x)
            if min_error is None or mean_error < min_error:
                # new best model found
                min_error = mean_error
                best_i_j = (i, j)
    if best_i_j[0] == 0 and best_i_j[1] == 0:

        def f_best(x, b):
            return b * np.ones_like(x)
    else:

        def f_best(x, c, b):
            return c * np.power(x, best_i_j[0]) * np.power(np.log2(x), best_i_j[1]) + b

    # fit best model to all data points
    final_p, _ = curve_fit(f_best, x, y)

    def final_f(x):
        return f_best(x, *final_p)

    if best_i_j[0] == 0 and best_i_j[1] == 0:
        sympy_f = final_p[0]
    else:
        sympy_f = sp.simplify(final_p[0] * symbol(symbol_name)**best_i_j[0] *
                              sp.log(symbol(symbol_name), 2)**best_i_j[1] + final_p[1])
    # compute r^2
    r_s = r_squared(final_f(x), y)
    return final_f, sympy_f, r_s


def fit_curve(x, y, symbol_name):
    """
    Fits a function throught the data set.

    :param x: The independent values.
    :param y: The dependent values.
    :param symbol_name: The name of the SDFG symbol.
    """
    x = np.array(x).astype(np.int32)
    y = np.array(y).astype(np.float64)

    # model search space
    I = [x / 4 for x in range(13)]
    J = [0, 1, 2]
    final_f, sympy_final_f, r_s = find_best_model(x, y, I, J, symbol_name)

    return final_f, sympy_final_f, r_s
