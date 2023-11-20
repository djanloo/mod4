# distutils: language = c++
from mod4.tsai cimport tsai_1D_v

cpdef evolve_tsai(double [:,:] p, dict i_pars, dict phy_pars):
    """Uses 1D evolvers to evolve 2D systems"""

    cdef int N, M, i, j
    N, M = p.shape[0], p.shape[1]

    for i in range(N):
        for j in range(M):
            pass


    return
