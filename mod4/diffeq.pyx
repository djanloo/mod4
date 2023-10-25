from cython.parallel import prange
import numpy as np

cimport cython
from cython.parallel import prange
cimport numpy as np

from time import perf_counter
from libc.math cimport sin

cdef double [:] tridiag(double [:] lower, double [:] diag, double [:] upper, double [:] d):
    """Solve the tridiagonal system by Thomas Algorithm"""
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
    
    return(np.array(x))

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

cdef double a(double x, double v, double time_index, dict physical_params):
  return physical_params['omega_squared']*x + physical_params['gamma']*v

cdef double d_a(double x, double v, double time_index, dict physical_params):
  '''Differential of a wrt v'''
  return physical_params['gamma']

def funker_plank( double [:,:] p0, 
                    double [:] x, double [:] v,
                    physical_params,
                    integration_params,
                    save_norm = False
                    ):
  print(f"physical_params: {physical_params}")
  print(f"integration_params: {integration_params}")
  cdef double dt   = integration_params['dt']
  cdef unsigned int n_steps = integration_params['n_steps']
  cdef float t0    = physical_params.get('t0', 0.0)

  cdef unsigned int N = len(x)
  cdef unsigned int M = len(v)
  cdef unsigned int time_index = 0, i = 0, j = 0

  cdef double [:,:] p = p0.copy(), p_intermediate = p0.copy()
  cdef double [:] norm = np.zeros(n_steps)

  cdef double dx = np.diff(x)[0]
  cdef double dv = np.diff(v)[0]

  cdef double theta = 0.5 * dt/dv
  cdef double alpha = 0.5 * dt/dx 
  cdef double eta = physical_params['sigma']*dt/dv**2
  
  # Declarations of the diagonals
  cdef double [:] lower_x, diagonal_x, upper_x, b_x
  cdef double [:] lower_v, diagonal_v, upper_v, b_v

  lower_v, upper_v, b_v = np.ones(M), np.ones(M), np.ones(M)
  lower_x, upper_x, b_x = np.ones(N), np.ones(N), np.ones(N)

  # Diagonal of systems does not change
  diagonal_v = np.ones(M) * (1 + 2*eta - 0.5*d_a(0,0,0, physical_params)*dt)
  diagonal_x = np.ones(N)

  # Support variables
  cdef double [:] row = np.zeros(max(N,M))

  for time_index in range(n_steps):
    # First evolution: differential wrt V
    # For each value of x, a tridiagonal system is solved to find values of v
    for i in range(N):

      # Prepares tridiagonal matrix and the constant term
      for j in range(M):
        upper_v[j]  = - eta
        upper_v[j] -= theta * a(x[i], v[j], t0 + time_index*dt, physical_params)
        upper_v[j] -= 0.25 * dt * d_a(x[i], v[j], t0 + time_index*dt, physical_params)


        if j < M-1:
          lower_v[j] =  - eta
          lower_v[j] += theta * a(x[i], v[j+1], t0 + time_index*dt, physical_params)
          lower_v[j] -= 0.25 * dt * d_a(x[i], v[j+1], t0 + time_index*dt, physical_params)
        b_v[j] =  p[j, i]

      # # Boundary conditions
      # b_v[0] = 0
      # diagonal_v[0] = 1
      # upper_v[0] = 0

      # b_v[M-1] = 0
      # diagonal_v[M-1] = 1
      # lower_v[M-2] = 0

      # Solves the tridiagonal system for the column
      row = tridiag(lower_v, diagonal_v, upper_v, b_v)
      
      for j in range(M):
        p[j,i] = row[j]

    # Second evolution: differential wrt x
    # For each value of v, a tridiagonal system is solved to find values of x
    for j in range(M):

      # Prepares tridiagonal matrix and constant term
      for i in range(N):
        lower_x[i] = - alpha * v[j]
        upper_x[i] =   alpha * v[j]

        b_x[i] = p_intermediate[j, i]

      # # Boundary conditions
      # b_x[0] = 0
      # diagonal_x[0] = 1
      # upper_x[0] = 0

      # b_x[M-1] = 0
      # diagonal_x[N-1] = 1
      # lower_x[N-2] = 0
      
      # Solves the tridiagonal system for the row
      row =  tridiag(lower_x, diagonal_x, upper_x, b_x)

      for i in range(N):
        p[j, i] = row[i]

    # Takes trace of normalization
    if save_norm:
      for i in range(N):
        for j in range(M):
          norm[time_index] += p[j,i]
      
      norm[time_index] *= dx * dv
  print(f"last time is {time_index*dt}")
  return p, norm
