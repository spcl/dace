# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains class CacheLineTracker which keeps track of all arrays of an SDFG and their cache line position. 
Further, contains class AccessStack which which corresponds to the stack used to compute the stack distance. """

from dace.data import Array

class CacheLineTracker:

    def __init__(self, L) -> None:
        self.array_info = {}
        self.start_lines = {}
        self.next_free_line = 0
        self.L = L

    def add_array(self, name: str, a: Array):
        if name not in self.start_lines:
            # new array encountered
            self.array_info[name] = a
            self.start_lines[name] = self.next_free_line
            # increase next_free_line
            self.next_free_line += (a.total_size * a.dtype.bytes + self.L - 1) // self.L    # ceil division
    
    def cache_line_id(self, name: str, access: [int]):
        arr = self.array_info[name]
        one_d_index = 0
        for dim in range(len(access)):
            i = access[dim]
            one_d_index += (i + arr.offset[dim]) * arr.strides[dim]
        
        # divide by L to get the cache line id
        return self.start_lines[name] + (one_d_index * arr.dtype.bytes) // self.L


class Node:

    def __init__(self, val: int, n=None) -> None:
        self.v = val
        self.next = n


class AccessStack:
    """ A stack of cache line ids. For each memory access, we search the corresponding cache line id
    in the stack, report its distance and move it to the top of the stack. If the id was not found,
    we report a distance of -1. """

    def __init__(self) -> None:
        self.top = None

    def touch(self, id):

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
            distance = -1

        return distance

