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
  
def get_boundary_func(int type):
  if type == 0:
    cdef a(double [:] v):
      return v
    return a
  else:
    return None


cdef double d_a(double x, double v, double t, dict physical_params):
  '''Differential of a2 wrt v'''
  return physical_params['gamma']


def funker_plank( double [:,:] p0, 
                    double [:] x, double [:] v,
                    physical_params,
                    integration_params,
                    save_norm = False # This costs a lot of time
                    ):

  cdef double dt   = integration_params['dt']
  cdef unsigned int n_steps = integration_params['n_steps']
  cdef float t0    = physical_params.get('t0', 0)

  cdef unsigned int N = len(x)
  cdef unsigned int M = len(v)
  cdef unsigned int t, i, j

  cdef double [:,:] p = p0.copy(), p_intermediate = p0.copy()
  cdef double [:] norm = np.zeros(n_steps)

  cdef double dx = np.diff(x)[0]
  cdef double dv = np.diff(v)[0]

  cdef double theta = 0.5 * dt/dv
  cdef double omega = 0.5 * dt/dx 
  cdef double eta = physical_params['sigma']*dt/dv**2

  # Declarations of the diagonals
  cdef double [:] lower_x, diagonal_x, upper_x, b_x
  cdef double [:] lower_v, diagonal_v, upper_v, b_v

  lower_v, upper_v, b_v = np.ones(M), np.ones(M), np.ones(M)
  lower_x, upper_x, b_x = np.ones(N), np.ones(N), np.ones(N)

  # Diagonal of systems does not change
  diagonal_v = np.ones(M) * (1 + 2*eta - 0.5*d_a(0,0,0, physical_params)*dt)
  diagonal_x = np.ones(N)

  # test
  cdef double [:] sum_of_row = np.zeros(M)
  cdef int row_index

  for t in range(n_steps):
    # First evolution: differential wrt V
    # For each value of x, a tridiagonal system is solved to find values of v
    for i in range(N):

      # Prepares tridiagonal matrix and the constant term
      for j in range(M):
        upper_v[j]  = - eta
        upper_v[j] -= theta * a(x[i], v[j], t0 + t*dt, physical_params)
        upper_v[j] -= 0.25 * dt * d_a(x[i], v[j], t0 + t*dt, physical_params)


        if j < M-1:
          lower_v[j] =  - eta
          lower_v[j] += theta * a(x[i], v[j+1], t0 + t*dt, physical_params)
          lower_v[j] -= 0.25 * dt * d_a(x[i], v[j+1], t0 + t*dt, physical_params)
        b_v[j] =  p[j, i]

      # # Test stochastic matrix:

      # for row_index in range(M):
      #   sum_of_row[row_index] = diagonal_v[row_index] + upper_v[row_index]
      #   if row_index > 1:
      #     sum_of_row[row_index] += lower_v[row_index - 1]
      # print(np.array(sum_of_row))
      # print( np.array(sum_of_row) + dt * d_a2(x[0], v[0], t0 + t*dt, physical_params))

      # Boundary conditions
      b_v[0] = 0
      diagonal_v[0] = 1
      upper_v[0] = 0

      b_v[M-1] = 0
      diagonal_v[M-1] = 1
      lower_v[M-2] = 0

      # Solves the tridiagonal system for the column
      p[:, i] =  tridiag(lower_v, diagonal_v, upper_v, b_v)
    
    # Second evolution: differential wrt x
    # For each value of v, a tridiagonal system is solved to find values of x
    for j in range(M):

      # Prepares tridiagonal matrix and constant term
      for i in range(N):
        lower_x[i] = - omega * v[j]
        upper_x[i] =   omega * v[j]

        b_x[i] = p_intermediate[j, i]

      # Boundary conditions
      b_x[0] = 0
      diagonal_x[0] = 1
      upper_x[0] = 0

      b_x[M-1] = 0
      diagonal_x[N-1] = 1
      lower_x[N-2] = 0

      # Solves the tridiagonal system for the row
      p[j, :] = tridiag(lower_x, diagonal_x, upper_x, b_x)

    # Takes trace of normalization
    if save_norm:
      for i in range(N):
        for j in range(M):
          norm[t] += p[j,i]
      
      norm[t] *= dx * dv

  return p, norm
