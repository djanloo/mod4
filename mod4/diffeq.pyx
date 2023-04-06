from cython.parallel import prange
import numpy as np
cimport cython

cpdef FTCS(double [:] u0, double v=0.5, double dx=0.1, double dt=0.1, int n_steps = 10):
  """Implements a forward time centered space integrator 
  on a flux conservative PDE
  """
  cdef int N = len(u0)
  cdef double [:] u = u0.copy()
  cdef double [:] u_next_time = u0.copy()

  cdef int i, _
  cdef double left, right

  for _ in range(n_steps):
    for i in range(N):
      left = 0 if i == 0 else u[i-1]
      right = 0 if i == N-1 else u[i+1]
      u_next_time[i] = u[i] - v*dt/dx/2* (right - left) 
    
    u = u_next_time.copy()
  
  return np.array(u)

cpdef LAX(double [:] u0, double v=0.5, double dx=0.1, double dt=0.1, int n_steps = 10):
  """Implements a Lax integrator 
  on a flux conservative PDE
  """

  print(f"Conditioning number is {v*dt/dx}")
  cdef int N = len(u0)
  cdef double [:] u = u0.copy()
  cdef double [:] u_next_time = u0.copy()

  cdef int i, _
  cdef double left, right, alpha = v*dt/dx

  for _ in range(n_steps):
    for i in range(N):
      # PBC
      left  = u[i-1] if i != 0   else u[N-1]
      right = u[i+1] if i != N-1 else u[0]

      u_next_time[i] = 0.5*(right*(1-alpha) + left*(1+alpha))
    
    u = u_next_time.copy()
  
  return np.array(u)

cpdef LAX_WENDROFF(double [:] u0, double v=0.5, double dx=0.1, double dt=0.1, int n_steps = 10):
  """Implements a Lax-Wendroff integrator 
  on a flux conservative PDE
  """

  print(f"Conditioning number is {v*dt/dx}")
  cdef int N = len(u0)
  cdef double [:] u = u0.copy()
  cdef double [:] u_next_time = u0.copy()
  cdef double [:] u_intermediate = u0.copy()

  cdef int i, _
  cdef double left, right, alpha = v*dt/dx

  for _ in range(n_steps):

    # LAX STEP
    for i in range(N):
      # PBC
      left  = u[i-1] if i != 0   else u[N-1]
      right = u[i+1] if i != N-1 else u[0]

      u_intermediate[i] = 0.5*(right*(1-alpha) + left*(1+alpha))
  
    # LEAPFROG STEP
    for i in range(N):
      # PBC
      left = u_intermediate[i-1] if i != 0 else u_intermediate[N-1]
      right = u_intermediate[i+1] if i != N-1 else u_intermediate[0]

      u_next_time[i] = u[i] - v*dt/dx * (right - left) 

    u = u_next_time.copy()
  
  return np.array(u)

def burgers_lw(double [:] u0, double dx=0.1, double dt=0.1, int n_steps = 10):
  print(f"alpha {dt/dx}")
  cdef int N = len(u0)
  cdef double [:] u = u0.copy()
  cdef double [:] u_next_time = u0.copy()
  cdef double [:] u_intermediate = u0.copy()

  cdef int i, _
  cdef double left, right, alpha = dt/dx

  for _ in range(n_steps):

    # LAX STEP
    for i in range(N):
      # PBC
      left  = u[i-1] if i != 0   else u[N-1]
      right = u[i+1] if i != N-1 else u[0]

      u_intermediate[i] = 0.5*(left + right) - alpha*0.5*(left+right)*0.5*(right-left)
  
    # LEAPFROG STEP
    for i in range(N):
      # PBC
      left = u_intermediate[i-1] if i != 0 else u_intermediate[N-1]
      right = u_intermediate[i+1] if i != N-1 else u_intermediate[0]

      u_next_time[i] = u[i] -  0.5*(left+right) *   alpha*(right - left)

    u = u_next_time.copy()
  
  return np.array(u)


def burgers_lw_forcing_extremum(double [:] u0, double [:] f_t, double dx=0.1, double dt=0.1, int n_steps = 10):
  print(f"alpha {dt/dx}")
  cdef int N = len(u0)
  cdef double [:] u = u0.copy()
  cdef double [:] u_next_time = u0.copy()
  cdef double [:] u_intermediate = u0.copy()

  cdef int i, _
  cdef double left, right, alpha = dt/dx

  for _ in range(n_steps):

    # LAX STEP
    for i in range(N):
      # PBC
      left  = u[i-1] if i != 0   else f_t(_)
      right = u[i+1] if i != N-1 else u[0]

      u_intermediate[i] = 0.5*(left + right) - alpha*0.5*(left+right)*0.5*(right-left)
  
    # LEAPFROG STEP
    for i in range(N):
      # PBC
      left = u_intermediate[i-1] if i != 0 else u_intermediate[N-1]
      right = u_intermediate[i+1] if i != N-1 else u_intermediate[0]

      u_next_time[i] = u[i] -  0.5*(left+right) *   alpha*(right - left)

    u = u_next_time.copy()
  
  return np.array(u)
