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
  cdef double left, right

  for _ in range(n_steps):
    for i in range(N):
      left = 0 if i == 0 else u[i-1]
      right = 0 if i == N-1 else u[i+1]

      u_next_time[i] = 0.5*(right+left) - v*dt/dx/2 * (right - left) 
    
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
  cdef double left, right

  for _ in range(n_steps):

    # LAX STEP
    for i in range(N):
      left = u[N-1] if i == 0 else u[i-1]
      right = u[0] if i == N-1 else u[i+1]
      u_intermediate[i] = 0.5*(right+left) - v*dt/dx/2 * (right - left) 
  
    # LEAPFROG STEP
    for i in range(N):
      left = u_intermediate[N-1] if i == 0 else u_intermediate[i-1]
      right = u_intermediate[0] if i == N-1 else u_intermediate[i+1]
      u_next_time[i] = u[i] - v*dt/dx * (right - left) 

    u = u_next_time.copy()
  
  return np.array(u)


cpdef preburgers(double [:] u0, double v=0.5, double dx=0.1, double dt=0.1, int n_steps = 10):
  """
  Burgers solved with Hopf-Cole tranformation.

  Basically solves:
      
      du/dt = v d**2 u /dx**2      (Heat equation)

  with a dumb integration:

      u_next_time(i) = u(i) + r [ u(i-1) - 2 u(i) + u(i+1)  ]
  """

  print(f"Conditioning number is {v*dt/dx}")
  cdef int N = len(u0)
  cdef double [:] u = u0.copy()
  cdef double [:] u_next_time = u0.copy()

  cdef int i, _
  cdef double left, right

  for _ in range(n_steps):

    for i in range(N):
      if i == 0:
        u_next_time[i] = u[i] + 2*r*(u[i+1] - u[i-1])
      elif i == N-1:
        u_next_time[i] = u[i] + 2*r*(u[i-1] - u[i])
      else:
        u_next_time[i] = u[i] + r*(u[i-1] - 2*u[i] + u[i+1])
  
    u = u_next_time.copy()


  

  return np.array(u)