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

cpdef advect_LW(double [:] p0, double x, dict physical_params, dict integration_params):
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

  cdef double [:] p = p0.copy(), p_half = p0.copy()
  cdef double theta = 0.5 * dt/dv
  
  for time_index in range(n_steps):
    time = t0 + time_index*dt
      
    # Half step LW
    for j in range(M):

      # Note tha theta is absorbed in working variable
      a_plus  =  theta * a(x,v[j] + dv, time, physical_params)
      a_here =  theta * a(x,v[j], time, physical_params)

      if j != M-1:
        p_half[j] = 0.5*(p[j+1] + p[j]) - (a_plus*p[j+1] - a_here*p[j])

    # Half step LW
    for j in range(M):
      a_plus_half  =  theta * a(x,v[j] + 0.5*dv, time, physical_params)
      a_minus_half =  theta * a(x,v[j] - 0.5*dv, time, physical_params)
      if j!= 0:
        p[j] -= 2*(a_plus_half*p_half[j] - a_minus_half*p_half[j-1])

  return p

cpdef diffuse_CN(double [:] p0, double x, dict physical_params, dict integration_params):
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
  cdef double eta   = 0.5 * dt/dv**2
  
  # Declarations of the diagonals
  cdef double [:] lower, diagonal, upper, b
  lower, diagonal, upper, b = np.ones(M), np.ones(M), np.ones(M), np.ones(M)

  for time_index in range(n_steps):
    time = t0 + time_index*dt
    s = eta * sigma_squared(x, time, physical_params)
    b = p.copy()
    for j in range(M):
      
      diagonal[j] = 1 + 2 * s
      upper[j]    = - s
      lower[j]    = - s
      
      if j != 0 and j != M-1:
        b[j] +=   (   s)* p[j+1] 
        b[j] +=   (-2*s)* p[j]
        b[j] +=   (   s)* p[j-1]

    p = tridiag(lower, diagonal, upper, b)
  return p



cpdef advectLW_diffuseCN(double [:] p0, double x, dict physical_params, dict integration_params):
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
  cdef bool diffCN
  diffCN = integration_params.get('diffCN', False)
  
  cdef double [:] p = p0.copy(), p_half = p0.copy()
  cdef double theta = 0.5 * dt/dv
  cdef double eta   = 0.5 * dt/dv**2
  
  if diffCN:
    eta = 0.5*eta
  
  # Declarations of the diagonals
  cdef double [:] lower, diagonal, upper, b
  lower, diagonal, upper, b = np.ones(M), np.ones(M), np.ones(M), np.ones(M)

  for time_index in range(n_steps):
    time = t0 + time_index*dt

    # Half step LW
    for j in range(M):

      # Note tha theta is absorbed in working variable
      a_plus =  theta * a(x,v[j] + dv, time, physical_params)
      a_here =  theta * a(x,v[j], time, physical_params)

      if j != M-1:
        p_half[j] = 0.5*(p[j+1] + p[j]) - (a_plus*p[j+1] - a_here*p[j])

    # Half step LW
    for j in range(M):
      a_plus_half  =  theta * a(x,v[j] + 0.5*dv, time, physical_params)
      a_minus_half =  theta * a(x,v[j] - 0.5*dv, time, physical_params)
      if j!= 0:
        p[j] -= 2*(a_plus_half*p_half[j] - a_minus_half*p_half[j-1])

    s = eta * sigma_squared(x, time, physical_params)
    b = p.copy()
    # Cranck-Nicholson
    for j in range(M):

      diagonal[j] = 1 + 2 * s  
      upper[j]    = - s
      lower[j]    = - s
      
      if diffCN:
        if j != 0 and j != M-1:
          b[j] +=   (   s)* p[j+1] 
          b[j] +=   (-2*s)* p[j]
          b[j] +=   (   s)* p[j-1]

    p = tridiag(lower, diagonal, upper, b)
  return p

cpdef advect_diffuse_LW(double [:] p0, double x, dict physical_params, dict integration_params):
  """See the example at LeVeque eq. 4.10
  
  Form my case if sigma = sigma(x) only, then this is the standard explicit diffusion.
  See also eq. 6.4

  This doesn't work.
  """
  raise NotImplementedError("This method fails")
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

  cdef double [:] p = p0.copy(), p_half = p0.copy()
  cdef double theta = 0.5 * dt/dv
  cdef double eta   = 0.5 * dt/dv**2

  
  for time_index in range(n_steps):
    time = t0 + time_index*dt
      
    # Half step LW
    for j in range(M):

      # Note tha theta is absorbed in working variable
      a_plus  =  theta * a(x,v[j] + dv, time, physical_params)
      a_here =  theta * a(x,v[j], time, physical_params)

      if j != M-1:
        p_half[j] = 0.5*(p[j+1] + p[j]) - (a_plus*p[j+1] - a_here*p[j])

    # Half step LW
    s = eta * sigma_squared(x, time, physical_params)

    for j in range(M):
      a_plus_half  =  theta * a(x,v[j] + 0.5*dv, time, physical_params)
      a_minus_half =  theta * a(x,v[j] - 0.5*dv, time, physical_params)
      if j!= 0:
        p[j] -= 2*(a_plus_half*p_half[j] - a_minus_half*p_half[j-1])
      
      # Hardcore explicit diffusion
      if j!=0 and j!= M-1:
        p[j] += s*p[j-1] - 2*p[j] + s*p[j]

  return p
