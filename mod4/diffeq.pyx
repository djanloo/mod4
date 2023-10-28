from cython.parallel import prange
import numpy as np

cimport cython
from cython.parallel import prange
cimport numpy as np

from time import perf_counter
from libc.math cimport sin

from .utils import quad_int

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

cdef double a(double x, double v, double time, dict physical_params):
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
  cdef double eta = 0.5*physical_params['sigma']**2*dt/dv**2
  
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

      # Solves the tridiagonal system for the column
      p_intermediate[:, i]= tridiag(lower_v, diagonal_v, upper_v, b_v)

      # Boundary conditions
      p_intermediate[0, i] = 0.0
      p_intermediate[M-1, i] = 0.0

    # Second evolution: differential wrt x
    # For each value of v, a tridiagonal system is solved to find values of x
    for j in range(M):

      # Prepares tridiagonal matrix and constant term
      for i in range(N):
        lower_x[i] = - alpha * v[j]
        upper_x[i] =   alpha * v[j]

        b_x[i] = p_intermediate[j, i]
      
      # Solves the tridiagonal system for the row
      p[j, :] =  tridiag(lower_x, diagonal_x, upper_x, b_x)

      # Boundary conditions
      p[j, 0] = 0.0
      p[j, N-1] = 0.0

    # Takes trace of normalization
    if save_norm:
      norm[time_index] = quad_int(p, x, v)
  return p, norm
