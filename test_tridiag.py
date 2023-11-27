import numpy as np
from mod4.utils import cyclic_tridiag

A = [   [1,2,0,0,1],
        [1,5,6,0,0],
        [0,4,3,6,0],
        [0,0,1,1,2],
        [1,0,0,1,1]
    ]
a = [1,4,1,1,7]
b = [1,5,3,1,1]
c = [2,6,6,2,7]
d = [1,2,3,4,5]

a,b,c,d = map(lambda x: np.array(x, dtype="float64"), (a,b,c,d))
A = np.array(A, dtype="float64")
x = cyclic_tridiag(a, b, c, 1.0, 1.0, d)

print(x)
print(A.dot(x))