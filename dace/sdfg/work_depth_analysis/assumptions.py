# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import sympy as sp
from typing import Dict


class UnionFind:
    """
    Simple, not really optimized UnionFind implementation.
    """

    def __init__(self, elements) -> None:
        self.ids = {e: e for e in elements}

    def add_element(self, e):
        if e in self.ids:
            return False
        self.ids.update({e: e})
        return True

    def find(self, e):
        prev = e
        curr = self.ids[e]
        while prev != curr:
            prev = curr
            curr = self.ids[curr]
        # shorten the path
        self.ids[e] = curr
        return curr

    def union(self, e, f):
        if f not in self.ids:
            self.add_element(f)
        self.ids[self.find(e)] = f


class ContradictingAssumptions(Exception):
    pass


class Assumptions:
    """
    Summarises the assumptions for a single symbol in three lists: equal, greater, lesser.
    """

    def __init__(self) -> None:
        self.greater = []
        self.lesser = []
        self.equal = []

    def add_greater(self, g):
        if isinstance(g, sp.Symbol):
            self.greater.append(g)
        else:
            self.greater = [x for x in self.greater if isinstance(x, sp.Symbol) or x > g]
            if len([y for y in self.greater if not isinstance(y, sp.Symbol)]) == 0:
                self.greater.append(g)
        self.check_consistency()

    def add_lesser(self, l):
        if isinstance(l, sp.Symbol):
            self.lesser.append(l)
        else:
            self.lesser = [x for x in self.lesser if isinstance(x, sp.Symbol) or x < l]
            if len([y for y in self.lesser if not isinstance(y, sp.Symbol)]) == 0:
                self.lesser.append(l)
        self.check_consistency()

    def add_equal(self, e):
        for x in self.equal:
            if not (isinstance(x, sp.Symbol) or isinstance(e, sp.Symbol)) and x != e:
                raise ContradictingAssumptions()
        self.equal.append(e)
        self.check_consistency()

    def check_consistency(self):
        if len(self.equal) > 0:
            # we know exact value
            for e in self.equal:
                for g in self.greater:
                    if (e <= g) == True:
                        raise ContradictingAssumptions()
                for l in self.lesser:
                    if (e >= l) == True:
                        raise ContradictingAssumptions()
        else:
            # check if any greater > any lesser
            for g in self.greater:
                for l in self.lesser:
                    if (g > l) == True:
                        raise ContradictingAssumptions()
        return True

    def num_assumptions(self):
        # returns the number of individual assumptions for this symbol
        return len(self.greater) + len(self.lesser) + len(self.equal)


def propagate_assumptions(x, y, condensed_assumptions):
    """
    Assuming x is equal to y, we propagate the assumptions on x to y. E.g. we have x==y and
    x<5. Then, this method adds y<5 to the assumptions.

    :param x: A symbol.
    :param y: Another symbol equal to x.
    :param condensed_assumptions: Current assumptions over all symbols.
    """
    if x == y:
        return
    assum_x = condensed_assumptions[x]
    if y not in condensed_assumptions:
        condensed_assumptions[y] = Assumptions()
    assum_y = condensed_assumptions[y]
    for e in assum_x.equal:
        if e is not sp.Symbol(y):
            assum_y.add_equal(e)
    for g in assum_x.greater:
        assum_y.add_greater(g)
    for l in assum_x.lesser:
        assum_y.add_lesser(l)
    assum_y.check_consistency()


def propagate_assumptions_equal_symbols(condensed_assumptions):
    """
    This method handles two things: 1) It generates the substitution dict for all equality assumptions.
    And 2) it propagates assumptions too all equal symbols. For each equivalence class, we find a unique
    representative using UnionFind. Then, all assumptions get propagates to this symbol using
    ``propagate_assumptions``.

    :param condensed_assumptions: Current assumptions over all symbols.
    :return: Returns a tuple consisting of 2 substitution dicts. The first one replaces each symbol with
    the unique representative of its equivalence class. The second dict replaces each symbol with its numeric 
    value (if we assume it to be equal some value, e.g. N==5).
    """
    # Make one set with unique identifier for each equality class
    uf = UnionFind(list(condensed_assumptions))
    for sym in condensed_assumptions:
        for other in condensed_assumptions[sym].equal:
            if isinstance(other, sp.Symbol):
                # we assume sym == other --> union these
                uf.union(sym, other.name)

    equality_subs1 = {}

    # For each equivalence class, we now have one unique identifier.
    # For each class, we give all the assumptions to this single symbol.
    # And we swap each symbol in class for this symbol.
    for sym in list(condensed_assumptions):
        for other in condensed_assumptions[sym].equal:
            if isinstance(other, sp.Symbol):
                propagate_assumptions(sym, uf.find(sym), condensed_assumptions)
                equality_subs1.update({sym: sp.Symbol(uf.find(sym))})

    equality_subs2 = {}
    # In a second step, each symbol gets replace with its equal number (if present)
    # using equality_subs2.
    for sym, assum in condensed_assumptions.items():
        for e in assum.equal:
            if not isinstance(e, sp.Symbol):
                equality_subs2.update({sym: e})

    # Imagine we have M>N and M==10. We need to deduce N<10 from that. Following code handles that:
    for sym, assum in condensed_assumptions.items():
        for g in assum.greater:
            if isinstance(g, sp.Symbol):
                for e in condensed_assumptions[g.name].equal:
                    if not isinstance(e, sp.Symbol):
                        condensed_assumptions[sym].add_greater(e)
                        assum.greater.remove(g)
        for l in assum.lesser:
            if isinstance(l, sp.Symbol):
                for e in condensed_assumptions[l.name].equal:
                    if not isinstance(e, sp.Symbol):
                        condensed_assumptions[sym].add_lesser(e)
                        assum.lesser.remove(l)
    return equality_subs1, equality_subs2


def parse_assumptions(assumptions, array_symbols):
    """
    Parses a list of assumptions into substitution dictionaries. Firstly, it gathers all assumptions and
    keeps only the strongest ones. Afterwards it constructs two substitution dicts for the equality
    assumptions: First dict for symbol==symbol assumptions; second dict for symbol==number assumptions.
    The other assumptions get handles by N tuples of substitution dicts (N = max number of concurrent
    assumptions for a single symbol). Each tuple is responsible for at most one assumption for each symbol. 
    First dict in the tuple substitutes the symbol with the assumption; second dict restores the initial symbol.

    :param assumptions: List of assumption strings.
    :param array_symbols: List of symbols we assume to be positive, since they are the size of a data container.
    :return: Tuple consisting of the 2 dicts responsible for the equality assumptions and the list of size N
    reponsible for all other assumptions.
    """

    # TODO: This assumptions system can be improved further, especially the deduction of further assumptions
    # from the ones we already have. An example of what is not working currently:
    # We have assumptions N>0 N<5 and M>5.
    # In the first substitution round we use N>0 and M>5.
    # In the second substitution round we use N<5.
    # Therefore, Max(M, N) will not be evaluated to M, even though from the input assumptions
    # one can clearly deduce M>N.
    # This happens since N<5 and M>5 are not in the same substitution round.
    # The easiest way to fix this is probably to actually deduce the M>N assumption.
    # This guarantees that in some substitution round, we will replace M with N + _p_M, where
    # _p_M is some positive symbol. Hence, we would resolve Max(M, N) to N + _p_M, which is M.

    # I suspect there to be many more cases where further assumptions will not be deduced properly.
    # But if the user enters assumptions as explicitly as possible, e.g. N<5 M>5 M>N, then everything
    # works fine.

    # For each symbol x appearing as a data container size, we can assume x>0.
    # TODO (later): Analyze size of shapes more, such that e.g. shape N + 1 --> We can assume N > -1.
    # For now we only extract assumptions out of shapes if shape consists of only a single symbol.
    for sym in array_symbols:
        assumptions.append(f'{sym.name}>0')

    if assumptions is None:
        return {}, [({}, {})]

    # Gather assumptions, keeping only the strongest ones for each symbol.
    condensed_assumptions: Dict[str, Assumptions] = {}
    for a in assumptions:
        if '==' in a:
            symbol, rhs = a.split('==')
            if symbol not in condensed_assumptions:
                condensed_assumptions[symbol] = Assumptions()
            try:
                condensed_assumptions[symbol].add_equal(int(rhs))
            except ValueError:
                condensed_assumptions[symbol].add_equal(sp.Symbol(rhs))
        elif '>' in a:
            symbol, rhs = a.split('>')
            if symbol not in condensed_assumptions:
                condensed_assumptions[symbol] = Assumptions()
            try:
                condensed_assumptions[symbol].add_greater(int(rhs))
            except ValueError:
                condensed_assumptions[symbol].add_greater(sp.Symbol(rhs))
                # add the opposite, i.e. for x>y, we add y<x
                if rhs not in condensed_assumptions:
                    condensed_assumptions[rhs] = Assumptions()
                condensed_assumptions[rhs].add_lesser(sp.Symbol(symbol))
        elif '<' in a:
            symbol, rhs = a.split('<')
            if symbol not in condensed_assumptions:
                condensed_assumptions[symbol] = Assumptions()
            try:
                condensed_assumptions[symbol].add_lesser(int(rhs))
            except ValueError:
                condensed_assumptions[symbol].add_lesser(sp.Symbol(rhs))
                # add the opposite, i.e. for x<y, we add y>x
                if rhs not in condensed_assumptions:
                    condensed_assumptions[rhs] = Assumptions()
                condensed_assumptions[rhs].add_greater(sp.Symbol(symbol))

    # Handle equal assumptions.
    equality_subs = propagate_assumptions_equal_symbols(condensed_assumptions)

    # How many assumptions does symbol with most assumptions have?
    curr_max = -1
    for _, assum in condensed_assumptions.items():
        if assum.num_assumptions() > curr_max:
            curr_max = assum.num_assumptions()

    all_subs = []
    for i in range(curr_max):
        all_subs.append(({}, {}))

    # Construct all the substitution dicts. In each substitution round we take at most one assumption for each
    # symbol. Each round has two dicts: First one swaps in the assumption and second one restores the initial
    # symbol.
    for sym, assum in condensed_assumptions.items():
        i = 0
        for g in assum.greater:
            replacement_symbol = sp.Symbol(f'_p_{sym}', positive=True, integer=True)
            all_subs[i][0].update({sp.Symbol(sym): replacement_symbol + g})
            all_subs[i][1].update({replacement_symbol: sp.Symbol(sym) - g})
            i += 1
        for l in assum.lesser:
            replacement_symbol = sp.Symbol(f'_n_{sym}', negative=True, integer=True)
            all_subs[i][0].update({sp.Symbol(sym): replacement_symbol + l})
            all_subs[i][1].update({replacement_symbol: sp.Symbol(sym) - l})
            i += 1

    return equality_subs, all_subs
