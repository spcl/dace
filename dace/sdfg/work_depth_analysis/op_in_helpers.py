# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains class CacheLineTracker which keeps track of all arrays of an SDFG and their cache line position. 
Further, contains class AccessStack which which corresponds to the stack used to compute the stack distance. """

from dace.data import Array
import sympy as sp
from collections import deque

class CacheLineTracker:

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
            self.next_free_line += (a.total_size.subs(mapping) * a.dtype.bytes + self.L - 1) // self.L    # ceil division

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

    # TODO: this can be optimised such that the stack is never larger than C, since all elements deeper than C are misses 
    # anyway. (then we cannot distinguish compulsory misses from capacity misses though)

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

            # shorten the stack if distance >= C
            # if distance >= self.C and curr is not None:
            #     curr.next = None

        if not found:
            # we accessed this cache line for the first time ever
            self.top = Node(id, self.top)
            self.length += 1
            distance = -1

        return distance
    
    def compare_cache(self, other):
        "Returns True if the same data resides in cache with the same LRU order"
        s = self.top
        o = other.top
        dist = 0
        while s is not None and o is not None and dist < self.C:
            dist += 1
            if s != o:
                return False
            s = s.next
            o = o.next
            if s is None and o is not None:
                return False
            if s is not None and o is None:
                return False
            
        return True
    
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
        return new_stack
