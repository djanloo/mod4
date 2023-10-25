import numpy as np

cdef tridiag(double [:] lower, double [:] diag, double [:] upper, double [:] d):
    """Solve the tridiagonal system by Thomas Algorithm"""
    cdef int N = len(diag)
    cdef int i
    cdef double [:] x = np.zeros(N, dtype="float64")
    cdef double [:] a = lower.copy(), b = diag.copy(), c = upper.copy()

    # A = np.zeros((N,N))

    # for i in range(N):
    #     A[i,i] = b[i] 
    #     if i >= 1:
    #         A[i, i-1] = a[i-1]
    #     if i < N-1:
    #         A[i, i+1] = c[i]

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
    cdef double gamma = 1.0 # the mysterious parameter
    
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


cdef double [:,:] stupid_inverse(double [:,:] A):
    det = np.linalg.det(A)
    B = A.copy()
    B[0, 0] = A[1,1]/det 
    B[1, 1] = A[0,0]/det
    B[0,1] = -A[1,0]/det 
    B[1,0] = -A[0,1]/det
    return B

cpdef double [:,:] analytic(double [:, :] X, double [:,:] V, double time, 
              double x0, double v0, 
              physical_params):

    N, M = X.shape[0], X.shape[1]
    omega_squared, gamma, sigma = map(physical_params.get, ['omega_squared', 'gamma', 'sigma'])

    l1, l2 = 0.5 * gamma + np.sqrt( (0.5*gamma)**2 - omega_squared**2), 0.5 * gamma - np.sqrt( (0.5*gamma)**2 - omega_squared**2)
    exp1, exp2 = np.exp(-l1*time), np.exp(-l2*time)
    G = [[(l1*exp1 - l2*exp2)/(l1 - l2), (exp2 - exp1)/(l1 -l2)],
         [omega_squared**2*(exp1 - exp2)/(l1-l2), (l1*exp1 - l2*exp2)/(l1 -l2)]]
    G = np.array(G)

    Sigm11 = sigma**2/(l1 -l2)**2*( (l1 + l2)/l1/l2 - 4*(1- exp1*exp2)/(l1+l1)- exp1**2 - exp2**2)
    Sigm12 = sigma**2/(l1 + l2)**2*(exp1 - exp2)**2
    Sigm22 = sigma**2/(l1-l2)**2*(l1+l2 + 4*l1*l2/(l1+l2)*(exp1*exp2 - 1) -l1*exp1**2 -l2*exp2**2)

    S = np.array([  [Sigm11, Sigm12],
                    [Sigm12, Sigm22]])
    normalization = 2*np.pi*np.sqrt(np.linalg.det(S))
    S_inv = stupid_inverse(S)
    result = np.zeros((N, M))
    for i in range(N):
      for j in range(M):
        extended_vect = np.array([X[i,j], V[i,j]])
        extended_init = np.array([x0, v0])
        mu = extended_vect - np.dot(G, extended_init)
        exponent = -0.5*np.dot(mu, np.dot(S_inv, mu.time_index))
        result[i, j] = np.exp(exponent)/normalization
        
    cdef double norm = 0.0
    for i in range(N):
      for j in range(M):
        norm += result[i,j]
    print(f"norm of analytic is {norm*(X[0,1] - X[0, 0])*(V[1, 0] - V[0,0])}")
    return result
