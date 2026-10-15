# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Finite-state and GF(2)-affine kernels: loops whose only carried dependence runs through one value.

The kernels mirror HPCAgent-Bench's ``finite_state_machine`` and ``combinational_logic`` kernels at a CI shape. The
algorithms are cited per kernel; the inputs are built here from a seed:

- ``dfa``: a random complete automaton over a symbol stream, counting state visits.
- ``kmp``: Knuth-Morris-Pratt search, counting pattern occurrences (Knuth, Morris and Pratt, SIAM J. Comput. 1977).
- ``crc16``: the bitwise CRC-16 register (via NPBench, BSD-3-Clause).
- ``huffman_decode``: bit-serial Huffman decoding, tallying symbols (Huffman, Proc. IRE 1952).
- ``aho_corasick``: dense Aho-Corasick multi-pattern matching (Aho and Corasick, CACM 1975).
- ``utf8_validate``: table-driven UTF-8 validation after Hoehrmann's decoder, tables from RFC 3629.
- ``csv_tokenize``: an RFC 4180 tokenizer counting fields and records.

Every one is lifted by ``LoopToStateScan`` (Mytkowicz, Musuvathi and Schulte, ASPLOS 2014, for the finite case).
"""

import heapq
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

import dace
from dace.frontend.python.parser import DaceProgram

N, M, NS, NA, NI = (dace.symbol(name, dace.int64) for name in ("N", "M", "NS", "NA", "NI"))

#: The stream length every kernel runs at.
LENGTH = 4096


@dace.program
def dfa(trans: dace.int64[NS, NA], symbols: dace.int64[N], counts: dace.int64[NS]):
    state = 0
    for i in range(N):
        state = trans[state, symbols[i]]
        counts[state] += 1


@dace.program
def kmp(text: dace.int64[N], pattern: dace.int64[M], fail: dace.int64[M], matches: dace.int64[1]):
    count = 0
    q = 0
    for i in range(N):
        while q > 0 and pattern[q] != text[i]:
            q = fail[q - 1]
        if pattern[q] == text[i]:
            q = q + 1
        if q == M:
            count = count + 1
            q = fail[q - 1]
    matches[0] = count


@dace.program
def crc16(data: dace.uint8[N], crc: dace.int64[1], polynomial: dace.int64):
    c = 0xFFFF
    for b in range(N):
        cur_byte = 0xFF & data[b]
        for bit in range(0, 8):
            if (c & 0x0001) ^ (cur_byte & 0x0001):
                c = (c >> 1) ^ polynomial
            else:
                c = c >> 1
            cur_byte = cur_byte >> 1
    crc[0] = (c ^ 0xFFFF) & 0xFFFF


@dace.program
def huffman_decode(child: dace.int64[NI, 2], bits: dace.int64[N], counts: dace.int64[NI + 1]):
    node = 0
    for i in range(N):
        nxt = child[node, bits[i]]
        if nxt >= NI:
            counts[nxt - NI] += 1
            node = 0
        else:
            node = nxt


@dace.program
def aho_corasick(delta: dace.int64[NS, 4], accept: dace.int64[NS], text: dace.int64[N], matches: dace.int64[1]):
    state = 0
    total = 0
    for i in range(N):
        state = delta[state, text[i]]
        total = total + accept[state]
    matches[0] = total


@dace.program
def utf8_validate(byte_class: dace.int64[256], trans: dace.int64[9, 12], data: dace.int64[N], counts: dace.int64[2]):
    state = 0
    for i in range(N):
        state = trans[state, byte_class[data[i]]]
        if state <= 1:
            counts[state] += 1


@dace.program
def csv_tokenize(char_class: dace.int64[256], trans: dace.int64[5, 4], text: dace.int64[N], counts: dace.int64[2]):
    state = 1
    for i in range(N):
        state = trans[state, char_class[text[i]]]
        if state <= 1:
            counts[state] += 1


def reference_dfa(trans, symbols, counts):
    state = 0
    for symbol in symbols:
        state = trans[state, symbol]
        counts[state] += 1


def reference_kmp(text, pattern, fail, matches):
    q = 0
    for symbol in text:
        while q > 0 and pattern[q] != symbol:
            q = fail[q - 1]
        if pattern[q] == symbol:
            q += 1
        if q == len(pattern):
            matches[0] += 1
            q = fail[q - 1]


def reference_crc16(data, crc, polynomial):
    c = 0xFFFF
    for byte in data:
        cur = int(byte)
        for bit in range(8):
            c = (c >> 1) ^ polynomial if (c ^ cur) & 1 else c >> 1
            cur >>= 1
    crc[0] = (c ^ 0xFFFF) & 0xFFFF


def reference_huffman_decode(child, bits, counts):
    inner = child.shape[0]
    node = 0
    for bit in bits:
        nxt = child[node, bit]
        if nxt >= inner:
            counts[nxt - inner] += 1
            node = 0
        else:
            node = nxt


def reference_aho_corasick(delta, accept, text, matches):
    state = 0
    for symbol in text:
        state = delta[state, symbol]
        matches[0] += accept[state]


def count_classified(classes, trans, stream, counts, start: int):
    state = start
    for symbol in stream:
        state = trans[state, classes[symbol]]
        if state <= 1:
            counts[state] += 1


def reference_utf8_validate(byte_class, trans, data, counts):
    count_classified(byte_class, trans, data, counts, 0)


def reference_csv_tokenize(char_class, trans, text, counts):
    count_classified(char_class, trans, text, counts, 1)


def failure_table(pattern: np.ndarray) -> np.ndarray:
    fail = np.zeros(len(pattern), np.int64)
    k = 0
    for i in range(1, len(pattern)):
        while k > 0 and pattern[k] != pattern[i]:
            k = fail[k - 1]
        if pattern[k] == pattern[i]:
            k += 1
        fail[i] = k
    return fail


def huffman(weights: list[int]) -> tuple[np.ndarray, list[list[int]]]:
    """The Huffman tree of ``weights`` as ``child[node, bit]`` (root 0, leaves ``NI + symbol``) and the codes."""
    inner = len(weights) - 1
    heap = [(weight, symbol, inner + symbol) for symbol, weight in enumerate(weights)]
    heapq.heapify(heap)
    merged: list[tuple[int, int]] = []
    while len(heap) > 1:
        first, second = heapq.heappop(heap), heapq.heappop(heap)
        merged.append((first[2], second[2]))
        heapq.heappush(heap, (first[0] + second[0], len(weights) + len(merged), inner - len(merged)))
    child = np.zeros((inner, 2), np.int64)
    for created, pair in enumerate(merged):
        child[inner - 1 - created] = pair
    codes: list[list[int]] = [[] for weight in weights]
    pending = [(0, [])]
    while pending:
        node, path = pending.pop()
        for bit in (0, 1):
            target = int(child[node, bit])
            if target >= inner:
                codes[target - inner] = [*path, bit]
            else:
                pending.append((target, [*path, bit]))
    return child, codes


def aho_automaton(patterns: list[list[int]], states: int) -> tuple[np.ndarray, np.ndarray]:
    """The dense Aho-Corasick automaton of ``patterns`` over four symbols, padded to ``states`` rows."""
    goto, ending = [[-1] * 4], [0]
    for pattern in patterns:
        node = 0
        for symbol in pattern:
            if goto[node][symbol] < 0:
                goto[node][symbol] = len(goto)
                goto.append([-1] * 4)
                ending.append(0)
            node = goto[node][symbol]
        ending[node] += 1
    delta, accept, fail = np.zeros((states, 4), np.int64), np.zeros(states, np.int64), [0] * len(goto)
    order = deque([0])
    while order:
        node = order.popleft()
        accept[node] = ending[node] + (accept[fail[node]] if node else 0)
        for symbol in range(4):
            target = goto[node][symbol]
            if target < 0:
                delta[node, symbol] = delta[fail[node], symbol] if node else 0
                continue
            fail[target] = delta[fail[node], symbol] if node else 0
            delta[node, symbol] = target
            order.append(target)
    return delta, accept


def utf8_tables() -> tuple[np.ndarray, np.ndarray]:
    """Byte classes and the 9-state transition table of RFC 3629's well-formed sequences."""
    spans = [
        [(0x00, 0x7F)],
        [(0x80, 0x8F)],
        [(0xC2, 0xDF)],
        [(0xE1, 0xEC), (0xEE, 0xEF)],
        [(0xED, 0xED)],
        [(0xF4, 0xF4)],
        [(0xF1, 0xF3)],
        [(0xA0, 0xBF)],
        [(0xC0, 0xC1), (0xF5, 0xFF)],
        [(0x90, 0x9F)],
        [(0xE0, 0xE0)],
        [(0xF0, 0xF0)],
    ]
    byte_class = np.zeros(256, np.int64)
    for cls, ranges in enumerate(spans):
        for low, high in ranges:
            byte_class[low : high + 1] = cls
    trans = np.full((9, 12), 1, np.int64)
    for start in (0, 1):
        trans[start, [0, 2, 3, 6, 10, 4, 11, 5]] = [0, 2, 3, 4, 5, 6, 7, 8]
    for state, classes, target in [
        (2, [1, 9, 7], 0),
        (3, [1, 9, 7], 2),
        (4, [1, 9, 7], 3),
        (5, [7], 2),
        (6, [1, 9], 2),
        (7, [9, 7], 3),
        (8, [1], 3),
    ]:
        trans[state, classes] = target
    return byte_class, trans


def csv_tables() -> tuple[np.ndarray, np.ndarray]:
    """Character classes (other, comma, quote, newline) and the 5-state RFC 4180 tokenizer."""
    char_class = np.zeros(256, np.int64)
    char_class[[ord(","), ord('"'), ord("\n")]] = [1, 2, 3]
    trans = np.array([[2, 0, 3, 1], [2, 0, 3, 1], [2, 0, 2, 1], [3, 3, 4, 3], [2, 0, 3, 1]], np.int64)
    return char_class, trans


def stream(tokens: list[bytes], rng: np.random.Generator) -> np.ndarray:
    """``LENGTH`` elements of random tokens laid end to end."""
    out: list[int] = []
    while len(out) < LENGTH:
        out += list(tokens[int(rng.integers(0, len(tokens)))])
    return np.array(out[:LENGTH], np.int64)


@dataclass(frozen=True, slots=True)
class Inputs:
    arrays: dict[str, np.ndarray]
    scalars: dict[str, object]
    sizes: dict[str, int]
    reference: Callable


def inputs(name: str, seed: int) -> Inputs:
    """The arrays, scalars, size symbols and oracle of kernel ``name``."""
    rng = np.random.default_rng(seed)
    if name == "dfa":
        trans, symbols = rng.integers(0, 16, (16, 4)), rng.integers(0, 4, LENGTH)
        return Inputs(
            dict(trans=trans, symbols=symbols, counts=np.zeros(16, np.int64)),
            {},
            dict(N=LENGTH, NS=16, NA=4),
            reference_dfa,
        )
    if name == "kmp":
        pattern = np.array([0, 1, 0, 1, 1, 0], np.int64)
        arrays = dict(
            text=rng.integers(0, 2, LENGTH), pattern=pattern, fail=failure_table(pattern), matches=np.zeros(1, np.int64)
        )
        return Inputs(arrays, {}, dict(N=LENGTH, M=len(pattern)), reference_kmp)
    if name == "crc16":
        arrays = dict(data=rng.integers(0, 256, LENGTH).astype(np.uint8), crc=np.zeros(1, np.int64))
        return Inputs(arrays, dict(polynomial=np.int64(0x8408)), dict(N=LENGTH), reference_crc16)
    if name == "huffman_decode":
        child, codes = huffman([int(rng.integers(1, 100)) ** 2 for symbol in range(16)])
        bits = stream([bytes(code) for code in codes], rng)
        return Inputs(
            dict(child=child, bits=bits, counts=np.zeros(16, np.int64)),
            {},
            dict(N=LENGTH, NI=15),
            reference_huffman_decode,
        )
    if name == "aho_corasick":
        delta, accept = aho_automaton([list(rng.integers(0, 4, 5)) for motif in range(8)], 41)
        arrays = dict(delta=delta, accept=accept, text=rng.integers(0, 4, LENGTH), matches=np.zeros(1, np.int64))
        return Inputs(arrays, {}, dict(N=LENGTH, NS=41), reference_aho_corasick)
    if name == "utf8_validate":
        byte_class, trans = utf8_tables()
        tokens = [
            chr(int(rng.integers(low, high))).encode()
            for low, high in [(0x20, 0x7F), (0x80, 0x800), (0x800, 0xD800), (0x10000, 0x110000)] * 8
        ]
        tokens += [b"\x80", b"\xc0", b"\xe2\x82", b"\xed\xa0\x80"]
        arrays = dict(byte_class=byte_class, trans=trans, data=stream(tokens, rng), counts=np.zeros(2, np.int64))
        return Inputs(arrays, {}, dict(N=LENGTH), reference_utf8_validate)
    if name == "csv_tokenize":
        char_class, trans = csv_tables()
        fields = ["abc", '"x,y"', '"a""b"', '"l1\nl2"', "12", ""]
        tokens = [(field + sep).encode() for field in fields for sep in (",", ",", "\n")]
        arrays = dict(char_class=char_class, trans=trans, text=stream(tokens, rng), counts=np.zeros(2, np.int64))
        return Inputs(arrays, {}, dict(N=LENGTH), reference_csv_tokenize)
    raise KeyError(name)


PROGRAMS: dict[str, DaceProgram] = {
    "dfa": dfa,
    "kmp": kmp,
    "crc16": crc16,
    "huffman_decode": huffman_decode,
    "aho_corasick": aho_corasick,
    "utf8_validate": utf8_validate,
    "csv_tokenize": csv_tokenize,
}


def collect() -> list[str]:
    """Every kernel's name, in corpus order."""
    return list(PROGRAMS)
