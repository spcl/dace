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

@dace
def refs(a, b, i, out):
    aa = a
    bb = b
    if i < 5:
        c = np.copy(aa)
    else:
        c = bb
    out[:] = c


@dace
def refs(A, B, i, out):
    if i < 5:
        ref = A
    else:
        ref = B
    out[:] = ref

@dace
def fors(A, B):
    
    a = 0
    for i in range(7):
        a = a + 1


a = np.random.rand(20)
b = np.random.rand(20)
i = 1
out = np.random.rand(20)
sdfg = refs.to_sdfg(a, b, i, out)
sdfg.save('bla.sdfg')

@dace.program
def main(x: int, b):

    i = np.full((20,), 0)

    if x < 100:
        if x < 50:
            i = np.full((20,), 50)
    else:
        if x > 200:
            i = np.full((20,), 200)

    return b + i

# @dace.program
# def main(x: int, b):

#     i = np.full((20,), 1)

#     if x < 100:
#         i = np.full((20,), 2)

#     return b + i

# def f(x):
#     return x

# @dace.program
# def main(a, b=None):

#     if b is None:
#         b = a

#     return b

# return ref fail

# a = np.random.rand(20)
# b = np.random.rand(20)
# i = 1
# out = np.random.rand(20)
# sdfg = refs.to_sdfg(a, b, i, out)
# sdfg.validate()
# sdfg.simplify()
# sdfg.save('bla.sdfg')
# sdfg.compile()

# print(main(201, np.full((20,), 2)))


# B = 7
# class A:

#     class B:
#         ...
