# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace


def test():
    s = dace.define_stream()
    S = dace.define_streamarray([2, 2])

    for i in range(6):
        s[0].append(i)
        for j in range(2):
            S[0, j].append(i + j)
            S[1, j].append(i + j * 10)

    while len(s[0]):
        print(s[0].popleft())
    while len(S[1, 1]):
        print(S[1, 1].popleft())


if __name__ == "__main__":
    test()
