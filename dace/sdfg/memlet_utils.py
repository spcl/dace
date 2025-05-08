# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

import ast
from collections import defaultdict
import copy
from dace.frontend.python import memlet_parser
from dace import data, Memlet, subsets
from typing import Callable, Dict, Iterable, Optional, Set, TypeVar, Tuple, Union


class MemletReplacer(ast.NodeTransformer):
    """
    Iterates over all memlet expressions (name or subscript with matching array in SDFG) in a code block.
    The callable can also return another memlet to replace the current one.
    """

    def __init__(self,
                 arrays: Dict[str, data.Data],
                 process: Callable[[Memlet], Union[Memlet, None]],
                 array_filter: Optional[Set[str]] = None) -> None:
        """
        Create a new memlet replacer.

        :param arrays: A mapping from array names to data descriptors.
        :param process: A callable that takes a memlet and returns a memlet or None.
        :param array_filter: An optional subset of array names to process.
        """
        self.process = process
        self.arrays = arrays
        self.array_filter = array_filter or self.arrays.keys()

    def _parse_memlet(self, node: Union[ast.Name, ast.Subscript]) -> Memlet:
        """
        Parses a memlet from a subscript or name node.

        :param node: The node to parse.
        :return: The parsed memlet.
        """
        # Get array name
        if isinstance(node, ast.Name):
            data = node.id
        elif isinstance(node, ast.Subscript):
            data = node.value.id
        else:
            raise TypeError('Expected Name or Subscript')

        # Parse memlet subset
        array = self.arrays[data]
        subset, newaxes, _ = memlet_parser.parse_memlet_subset(array, node, self.arrays)
        if newaxes:
            raise NotImplementedError('Adding new axes to memlets is not supported')

        return Memlet(data=data, subset=subset)

    def _memlet_to_ast(self, memlet: Memlet) -> ast.Subscript:
        """
        Converts a memlet to a subscript node.

        :param memlet: The memlet to convert.
        :return: The converted node.
        """
        return ast.parse(f'{memlet.data}[{memlet.subset}]').body[0].value

    def _replace(self, node: Union[ast.Name, ast.Subscript]) -> ast.Subscript:
        cur_memlet = self._parse_memlet(node)
        new_memlet = self.process(cur_memlet)
        if new_memlet is None:
            return node

        new_node = self._memlet_to_ast(new_memlet)
        return ast.copy_location(new_node, node)

    def visit_Name(self, node: ast.Name):
        if node.id in self.array_filter:
            return self._replace(node)
        return self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript):
        if isinstance(node.value, ast.Name) and node.value.id in self.array_filter:
            return self._replace(node)
        return self.generic_visit(node)


class MemletSet(Set[Memlet]):
    """
    Implements a set of memlets that considers subsets that intersect or are covered by its other memlets.
    Set updates and unions also perform unions on the contained memlet subsets.
    """

    def __init__(self, iterable: Optional[Iterable[Memlet]] = None, intersection_is_contained: bool = True):
        """
        Initializes a memlet set.

        :param iterable: An optional iterable of memlets to initialize the set with.
        :param intersection_is_contained: Whether the check ``m in memlet_set`` should return True if the memlet
                                          only intersects with the contents of the set. If False, only completely
                                          covered subsets would return True.
        """
        self.internal_set: Dict[str, Set[Memlet]] = {}
        self.intersection_is_contained = intersection_is_contained
        if iterable is not None:
            self.update(iterable)

    def __iter__(self):
        for subset in self.internal_set.values():
            yield from subset

    def __len__(self):
        return len(self.internal_set)

    def update(self, *iterable: Iterable[Memlet]):
        """
        Updates set of memlets via union of existing ranges.
        """
        if len(iterable) == 0:
            return
        if len(iterable) > 1:
            for i in iterable:
                self.update(i)
            return

        to_update, = iterable
        for elem in to_update:
            self.add(elem)

    def add(self, elem: Memlet):
        """
        Adds a memlet to the set, potentially performing a union of existing ranges.
        """
        if elem.data not in self.internal_set:
            self.internal_set[elem.data] = {elem}
            return

        # Memlet is in set, either perform a union (if possible) or add to internal set
        # TODO(later): Consider other_subset as well
        for existing_memlet in self.internal_set[elem.data]:
            try:
                if existing_memlet.subset.intersects(elem.subset) == True:  # Definitely intersects
                    if existing_memlet.subset.covers(elem.subset):
                        break  # Nothing to do

                    # Create a new union memlet
                    self.internal_set[elem.data].remove(existing_memlet)
                    new_memlet = copy.deepcopy(existing_memlet)
                    new_memlet.subset = subsets.union(existing_memlet.subset, elem.subset)
                    self.internal_set[elem.data].add(new_memlet)
                    break
            except TypeError:  # Indeterminate
                pass
        else:  # all intersections were False or indeterminate (may or does not intersect with existing memlets)
            self.internal_set[elem.data].add(elem)

    def __contains__(self, elem: Memlet) -> bool:
        """
        Returns True iff the memlet or a range superset thereof exists in this set.
        """
        if elem.data not in self.internal_set:
            return False
        for existing_memlet in self.internal_set[elem.data]:
            if existing_memlet.subset.covers(elem.subset):
                return True
            if self.intersection_is_contained:
                try:
                    if existing_memlet.subset.intersects(elem.subset) == False:
                        continue
                    else:  # May intersect or indeterminate
                        return True
                except TypeError:
                    return True

        return False

    def union(self, *s: Iterable[Memlet]) -> 'MemletSet':
        """
        Performs a set-union (with memlet union) over the given sets of memlets.

        :return: New memlet set containing the union of this set and the inputs.
        """
        newset = MemletSet(self)
        newset.update(s)
        return newset


T = TypeVar('T')


class MemletDict(Dict[Memlet, T]):
    """
    Implements a dictionary with memlet keys that considers subsets that intersect or are covered by its other memlets.
    """
    covers_cache: Dict[Tuple, bool] = {}

    def __init__(self, **kwargs):
        self.internal_dict: Dict[str, Dict[Memlet, T]] = defaultdict(dict)
        if kwargs:
            self.update(kwargs)

    def _getkey(self, elem: Memlet) -> Optional[Memlet]:
        """
        Returns the corresponding key (exact, covered, intersecting, or indeterminately intersecting memlet) if
        exists in the dictionary, or None if it does not.
        """
        if elem.data not in self.internal_dict:
            return None
        for existing_memlet in self.internal_dict[elem.data]:
            key = (existing_memlet.subset, elem.subset)
            is_covered = self.covers_cache.get(key, None)
            if is_covered is None:
                is_covered = existing_memlet.subset.covers(elem.subset)
                self.covers_cache[key] = is_covered
            if is_covered:
                return existing_memlet
            try:
                if existing_memlet.subset.intersects(elem.subset) == False:  # Definitely does not intersect
                    continue
            except TypeError:
                pass

            # May or will intersect
            return existing_memlet

        return None

    def _setkey(self, key: Memlet, value: T) -> None:
        self.internal_dict[key.data][key] = value

    def clear(self):
        self.internal_dict.clear()

    def update(self, mapping: Dict[Memlet, T]):
        for k, v in mapping.items():
            ak = self._getkey(k)
            if ak is None:
                self._setkey(k, v)
            else:
                self._setkey(ak, v)

    def __contains__(self, elem: Memlet) -> bool:
        """
        Returns True iff the memlet or a range superset thereof exists in this dictionary.
        """
        return self._getkey(elem) is not None

    def __getitem__(self, key: Memlet) -> T:
        actual_key = self._getkey(key)
        if actual_key is None:
            raise KeyError(key)
        return self.internal_dict[key.data][actual_key]

    def __setitem__(self, key: Memlet, value: T) -> None:
        actual_key = self._getkey(key)
        if actual_key is None:
            self._setkey(key, value)
        else:
            self._setkey(actual_key, value)
