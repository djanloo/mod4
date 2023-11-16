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

cdef double sigma_squared_full(double x, double v, double t, dict physical_params):
    return physical_params['sigma_squared']

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

        a_next_here  = theta*a(x, v[j] ,      time+dt,    physical_params)

        lower[j]    = 1 - 1.5*a_next_here   - 3*s 
        diagonal[j] = 4 + 1.5*(a_next_left  - a_next_right) + 6*s
        upper[j]    = 1 + 1.5*a_next_right  - 3*s
      
        if j != 0 and j != M-1:
            b[j] =   (1 - 1.5*a_now_right + 3*s)* p[j+1] 
            b[j] +=  (4 - 1.5*(a_now_left - a_now_right) - 6*s)* p[j]
            b[j] +=  (1 + 1.5*a_now_left  + 3*s)* p[j-1]
        else:
            b[j] = 0

    p = tridiag(lower, diagonal, upper, b)
  return p


cpdef tsai_FV(double [:] p0, double x, dict physical_params, dict integration_params):
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

    cdef double [:] p = p0.copy(), p_new = p0.copy()
    cdef double [:] P = np.zeros(len(p0)-1)

    # Computation of initial cell averages
    for j in range(M-1):
        P[j] = 0.5*(p[j] + p[j+1])

    cdef double theta = dt/dv
    cdef double eta = dt/dv**2

    # Declarations of the diagonals
    cdef double [:] lower, diagonal, upper, b
    lower, diagonal, upper, b = np.ones(M), np.ones(M), np.ones(M), np.ones(M)

    for time_index in range(n_steps):
        time = t0 + time_index*dt
        s = eta*sigma_squared(x, time, physical_params)
        
        for j in range(M):
            # Diffusion coeffs
            s_next_right = eta*sigma_squared_full(x, v[j] + dv,  time+dt, physical_params)
            s_next_here  = eta*sigma_squared_full(x, v[j],       time+dt, physical_params)
            s_now_right  = eta*sigma_squared_full(x, v[j] + dv,  time,    physical_params)
            s_now_here   = eta*sigma_squared_full(x, v[j],       time,    physical_params)

            # Advection coeffs
            a_now_right  = theta*a(x, v[j] + dv,time,       physical_params)
            a_now_here   = theta*a(x, v[j],     time,       physical_params)
            a_next_right = theta*a(x, v[j] + dv,time + dt,  physical_params)
            a_next_here  = theta*a(x, v[j],     time + dt,  physical_params)

            denom = 1 + 3*s_next_right + 3*s_next_here
            a_minus =  0.5*(a_next_here + 2*s_next_right + 4*s_next_here)/denom
            a_plus  = -0.5*(a_next_right - 4*s_next_right - 2*s_next_here)/denom 
            b_minus = 0.5*(a_now_here + 2*s_now_right + 4*s_now_here)/denom 
            b_center = (1 - 3*s_now_right - 3*s_now_here)/denom
            b_plus = 0.5*(-a_now_right + 4*s_now_right + 2*s_now_here)/denom

            special = 
            lower[j] = 

        p_new = tridiag(lower, diagonal, upper, b)

        # Update of averages
        for j in range(M-1):
            # Diffusion coeffs
            s_next_right = eta*sigma_squared_full(x, v[j] + dv,  time+dt, physical_params)
            s_next_here  = eta*sigma_squared_full(x, v[j],       time+dt, physical_params)
            s_now_right  = eta*sigma_squared_full(x, v[j] + dv,  time,    physical_params)
            s_now_here   = eta*sigma_squared_full(x, v[j],       time,    physical_params)

            # Advection coeffs
            a_now_right  = theta*a(x, v[j] + dv,time,       physical_params)
            a_now_here   = theta*a(x, v[j],     time,       physical_params)
            a_next_right = theta*a(x, v[j] + dv,time + dt,  physical_params)
            a_next_here  = theta*a(x, v[j],     time + dt,  physical_params)

            denom = 1 + 3*s_next_right + 3*s_next_here
            a_minus =  0.5*(a_next_here + 2*s_next_right + 4*s_next_here)/denom
            a_plus  = -0.5*(a_next_right - 4*s_next_right - 2*s_next_here)/denom 
            b_minus = 0.5*(a_now_here + 2*s_now_right + 4*s_now_here)/denom 
            b_center = (1 - 3*s_now_right - 3*s_now_here)/denom
            b_plus = 0.5*(-a_now_right + 4*s_now_right + 2*s_now_here)/denom

            # Step for averages
            P[j] = a_minus*p_new[j] + a_plus*p_new[j+1] + b_minus*p[j] + b_center*P[j] + b_plus*p[j+1]
    return p