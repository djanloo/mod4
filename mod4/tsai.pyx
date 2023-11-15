# distutils: language = c++
from libcpp cimport bool

from cython.parallel import prange
import numpy as np

cimport cython
from cython.parallel import prange
cimport numpy as np

from time import perf_counter
from libc.math cimport sin

cimport utils
from utils import quad_int, get_tridiag
from utils cimport tridiag

cimport diffeq
from diffeq cimport a, sigma_squared



cpdef tsai1d(double [:] p0, double x, dict physical_params, dict integration_params):
  ## Time
  cdef double dt   = integration_params['dt']
  cdef unsigned int n_steps = integration_params['n_steps']
  cdef double t0    = physical_params.get('t0', 0.0)

  ## Space
  cdef double Lv,dv
  Lv,dv = map(integration_params.get, ["Lv", "dv"])

  cdef unsigned int  M = int(Lv/dv)
  cdef double [:] v = np.arange(-int(M)//2, int(M)//2)*dv

  cdef unsigned int time_index = 0, j = 0

  cdef double [:] p = p0.copy()
  cdef double theta = dt/dv
  cdef double s = 0.0

  # Declarations of the diagonals
  cdef double [:] lower, diagonal, upper, b
  lower, diagonal, upper, b = np.ones(M), np.ones(M), np.ones(M), np.ones(M)

  for time_index in range(n_steps):
    time = t0 + time_index*dt
    s = 0.5*dt/dv**2*sigma_squared(x, time, physical_params)

    for j in range(M):
        a_now_right  = theta*a(x, v[j] + dv,  time,       physical_params)
        a_now_left   = theta*a(x, v[j] - dv,  time,       physical_params)
        
        a_next_right = theta*a(x, v[j] + dv,  time+dt,    physical_params)
        a_next_left  = theta*a(x, v[j] - dv,  time+dt,    physical_params)
        a_next_here  = theta*a(x, v[j] ,      time+dt,       physical_params)

        lower[j] = 1 - 1.5*a_next_here - 3*s 
        diagonal[j] = 4 + 1.5*(a_next_left - a_next_right)  + 6*s
        upper[j] = 1 + 1.5*a_next_right - 3*s
      
        if j != 0 and j != M-1:
            b[j] =   (1 - 1.5*a_now_right + 3*s)* p[j+1] 
            b[j] +=   (4 - 1.5*(a_now_left - a_now_right) - 6*s)* p[j]
            b[j] +=   ( 1 + 1.5*a_now_left + 3*s)* p[j-1]

    p = tridiag(lower, diagonal, upper, b)
  return p
