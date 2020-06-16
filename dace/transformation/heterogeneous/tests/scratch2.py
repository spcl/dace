import dace
from dace.subsets import *




range1 = Range([[0,0,1]])
range2 = Range([[0,0,1]])

print(range1.compose(range2))
