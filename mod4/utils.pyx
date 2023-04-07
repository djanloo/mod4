import numpy as np

cpdef tridiag(double [:] lower, double [:] diag, double [:] upper, double [:] d):
    """Solve the tridiagonal system by Thomas Algorithm"""
    cdef int N = len(diag)
    cdef int i
    cdef double [:] x = np.zeros(N, dtype="float64")
    cdef double [:] a = lower.copy(), b = diag.copy(), c = upper.copy()

    A = np.zeros((N,N))

    for i in range(N):
        A[i,i] = b[i] 
        if i >= 1:
            A[i, i-1] = a[i-1]
        if i < N-1:
            A[i, i+1] = c[i]

    # print(f"tri: a = {np.array(a)}")
    # print(f"tri: b = {np.array(b)}")
    # print(f"tri: c = {np.array(c)}")

    # print(f"tri :A = {A}")


    for i in range(N-1):
        b[i+1] -= a[i]/b[i]*c[i]
        d[i+1] -= a[i]/b[i]*d[i]
    
    x[N-1] = d[N-1]/b[N-1]
    for i in range(N-2, -1, -1):
        x[i] = (d[i] - c[i]*x[i+1])/b[i]
    
    return(np.array(x))


cpdef cyclic_tridiag(double [:] lower, double [:] diag, double [:] upper, double c_up_right, double c_down_left, double [:] d):
    cdef int N = len(diag), i
    cdef double [:] u = np.zeros(N), v = np.zeros(N)
    cdef gamma = 1.0 # the mysterious parameter
    
    u[0]   = gamma
    u[N-1] = c_down_left

    v[0]    = 1
    v[N-1]  = c_up_right/gamma

    # Solution of the pure tridiag
    x0 = tridiag(lower, diag, upper, d)

    # solution for the auxiliary vector of Shermann-Morrison
    q = tridiag(lower, diag, upper, u)

    # Correction considering that v has only two nonzero entries
    # delta = (q outer v ) y
    cdef double [:] delta = np.zeros(N)
    for i in range(N):
        delta[i] = (v[0] + v[N-1])*q[i]*x0[i]
    return np.array( x0 - np.array(delta)/(1 + v[0]*q[0] + v[N-1]*q[N-1]))
    