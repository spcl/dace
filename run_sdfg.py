import dace
import numpy as np

from random import randint

# @dace.program
# def main(a, b):
#     cond: int = randint(0, 1)
#     bleh = myref(cond)
#     if cond:
#         i = 1
#         bleh = a5
#     else:
#         i = 7
#         bleh = b
    
#     # a = i
#     return bleh

@dace.program
def main(x: int, b):

    i = np.full((20,), 1)

    if x < 100:
        if x < 50:
            i = np.full((20,), 2)
    else:
        if x > 200:
            i = np.full((20,), 3)

    return b + i

# def f(x):
#     return x

# @dace.program
# def main(a, b=None):

#     if b is None:
#         b = a

#     return b

main.to_sdfg(1, np.full((20,), 2)).save('bla.sdfg')


# B = 7
# class A:

#     class B:
#         ...
