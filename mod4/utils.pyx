import numpy as np
from scipy.linalg import expm 

cdef double [:] tridiag(double [:] lower, double [:] diag, double [:] upper, double [:] d):
    """Solves the tridiagonal system by Thomas Algorithm"""
    cdef int N = len(diag)
    cdef int i
    cdef double [:] x = np.zeros(N, dtype="float64")
    cdef double [:] a = lower.copy(), b = diag.copy(), c = upper.copy()

    for i in range(N-1):
        b[i+1] -= a[i]/b[i]*c[i]
        d[i+1] -= a[i]/b[i]*d[i]
    
    x[N-1] = d[N-1]/b[N-1]
    for i in range(N-2, -1, -1):
        x[i] = (d[i] - c[i]*x[i+1])/b[i]
    
    return x

cpdef get_tridiag(double [:] lower, double [:] diag, double [:] upper):
    cdef int N = len(diag) 
    cdef double [:,:] A = np.zeros((N,N))
    cdef int i

    for i in range(N):
        A[i,i] = diag[i] 
        if i >= 1:
            A[i, i-1] = lower[i-1]
        if i < N-1:
            A[i, i+1] = upper[i]
    return A

def get_quad_mesh(integration_params):
    cdef double Lx, Lv,dx,dv
    Lx, Lv, dx, dv = map(integration_params.get, ["Lx", "Lv", "dx", "dv"])
    cdef int N = int(Lx/dx), M = int(Lv/dv)
    cdef double [:] x = np.arange(-(N//2), N//2)*dx
    cdef double [:] v = np.arange(-(M//2), M//2)*dv
    return np.meshgrid(np.array(x), np.array(v))

def get_lin_mesh(integration_params):
    cdef double L, d

    L = integration_params.get('Lx', 0.0)
    if  L == 0.0:
        L = integration_params.get('Lv', 0.0)
    d = integration_params.get('dx', 0.0)
    if d == 0.0:
        d = integration_params.get('dv', 0.0)
    cdef int N = int(L/d)
    cdef int i = 0

    mesh = np.zeros(N+1)
    for i in range(N+1):
        mesh[i] = -L/2 + i*d
    return mesh

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

cdef complex det22(complex [:,:] A):
    return A[0,0]*A[1,1] - A[1,0]*A[0,1]

cdef complex [:,:] stupid_inverse(complex [:,:] A):
    det = det22(A)
    B = A.copy()
    B[0, 0] = A[1,1]/det 
    B[1, 1] = A[0,0]/det
    B[0,1] = -A[1,0]/det 
    B[1,0] = -A[0,1]/det
    return B

cpdef complex [:,:] analytic(double [:, :] X, double [:,:] V, double time, 
              double x0, double v0, 
              physical_params):
    cdef unsigned int N, M, i, j
    cdef double omega_squared, gamma, sigma_squared, under_root
    cdef complex [:,:] result, S
    cdef complex l1_plus_l2, l1_times_l2, l1_minus_l2, exp1, exp2
    cdef double [:,:] G

    N, M = X.shape[0], X.shape[1]
    omega_squared, gamma, sigma_squared = map(physical_params.get, ['omega_squared', 'gamma', 'sigma_squared'])

    under_root = (0.5*gamma)**2 - omega_squared 
    l1_times_l2 = omega_squared
    l1_plus_l2 = gamma

    if under_root < 0:
        # Underdamping case
        l1, l2 = 0.5*gamma + 1j*np.sqrt(-under_root),  0.5*gamma - 1j*np.sqrt(-under_root)
        l1_minus_l2 = 2j*np.sqrt(-under_root)
    else:
        # Overdamping case
        l1, l2 = 0.5 * gamma + np.sqrt(under_root), 0.5 * gamma - np.sqrt(under_root)
        l1_minus_l2 = 2*np.sqrt(under_root)
    
    Gamma = np.array([[0, -1], [omega_squared, gamma]])
    exp1, exp2 = np.exp(-l1*time), np.exp(-l2*time)
    G = expm(-Gamma*time)

    Sigm11 = ( l1_plus_l2/l1_times_l2 - 4*(1- exp1*exp2)/l1_plus_l2 - exp1**2/l1 - exp2**2/l2)
    Sigm12 = (exp1 - exp2)**2
    Sigm22 = ( l1_plus_l2 + 4*l1_times_l2/l1_plus_l2*(exp1*exp2 - 1) - l1*exp1**2 -l2*exp2**2)

    S = 0.5*sigma_squared/l1_minus_l2**2*np.array([ [Sigm11, Sigm12],
                                                    [Sigm12, Sigm22]], dtype=complex)


    normalization = 2*np.pi*np.sqrt(det22(S))
    S_inv = stupid_inverse(S)
    result = np.zeros((N, M), dtype=complex)
    
    for i in range(N):
      for j in range(M):
        extended_vect = np.array([X[i,j], V[i,j]])
        extended_init = np.array([x0, v0])
        mu = extended_vect - np.dot(G, extended_init)
        exponent = -0.5*np.dot(mu, np.dot(S_inv, mu.T))
        result[i, j] = np.exp(exponent)/normalization
        
    return result

cpdef quad_int(double [:,:] f, integration_params):
    cdef double dx,dv
    dx,dv = map(integration_params.get, [ "dx", "dv"])

    cdef unsigned int N = f.shape[0], M = f.shape[1]
    cdef double sum = 0.0
    cdef double [:] intermediate = np.zeros(N)
    cdef unsigned int i = 0, j = 0

    for i in range(N):
        for j in range(M-1):
            intermediate[i] += f[i, j] + f[i, j+1]
        intermediate[i] *= 0.5 * dv
    
    for i in range(N-1):
        sum += intermediate[i] + intermediate[i+1]
    sum *= 0.5 * dx

    return sum






    

