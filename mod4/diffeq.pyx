from cython.parallel import prange
import numpy as np

cimport cython
cimport numpy as np

from.utils import tridiag

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

def burgers_lw(double [:] u0, double nu =0.1, double dx=0.1, double dt=0.1, int n_steps = 10):
  print(f"alpha {dt/dx}")
  print(f"beta {nu*dt/dx**2}")

  cdef int N = len(u0)
  cdef double [:] u = u0.copy()
  cdef double [:] u_next_time = u0.copy()
  cdef double [:] g = u0.copy() # This is the intermediate layer

  cdef int i, _
  cdef double left_g, right_g, left_u, right_u, alpha = dt/dx

  for _ in range(n_steps):

    # LAX STEP
    for i in range(N):
      # PBC
      left_u  = u[i-1] if i != 0   else u[N-1]
      right_u = u[i+1] if i != N-1 else u[0]

      g[i] = 0.5*(left_u + right_u) - 0.5*(left_u + right_u)* alpha*(right_u - left_u) + 0.5*nu*dt/dx**2*(left_u + right_u - 2*u[i])
  
    # LEAPFROG STEP
    for i in range(N):
      # PBC
      left_g  = g[i-1] if i != 0   else g[N-1]
      right_g = g[i+1] if i != N-1 else g[0]

      u_next_time[i] = u[i] - 0.5*(left_g + right_g) * alpha*(right_g - left_g) + nu*dt/dx**2 * 0.5*(left_g + left_u + right_g + right_u - 2*g[i] - 2*u[i]) # poi prova ad approssimare 2*g_i con g_i + u_i

    u = u_next_time.copy()
  
  return np.array(u)

def heat_lw(double [:] u0, double nu =0.1, double dx=0.1, double dt=0.1, int n_steps = 10):
  print(f"alpha {dt/dx}")
  print(f"beta {nu*dt/dx**2}")

  cdef int N = len(u0)
  cdef double [:] u = u0.copy()
  cdef double [:] u_next_time = u0.copy()
  cdef double [:] g = u0.copy() # This is the intermediate layer

  cdef int i, _
  cdef double left_u, right_u, left_g, right_g,  alpha = dt/dx

  for _ in range(n_steps):

    # LAX STEP
    for i in range(N):
      # PBC
      left_u  = u[i-1] if i != 0   else u[N-1]
      right_u = u[i+1] if i != N-1 else u[0]

      g[i] = 0.5*(left_u + right_u) + 0.5*nu*dt/dx**2*(left_u  + right_u - 2*u[i])
  
    # LEAPFROG STEP
    for i in range(N):
      # PBC
      left_g  = g[i-1] if i != 0   else g[N-1]
      right_g = g[i+1] if i != N-1 else g[0]

      u_next_time[i] = u[i]  + nu*dt/dx**2*0.5*(left_g + left_u + right_g + right_u - 2*g[i] - 2*u[i]) # poi prova ad approssimare 2*g_i con g_i + u_i

    u = u_next_time.copy()
  
  return np.array(u)


def heat_cn(double [:] u0, double nu=0.1, double dx=0.1, double dt=0.1, int n_steps = 10):

  cdef int N = len(u0), i, _
  cdef double alpha = 0.5*nu*dt/dx**2, left, right
  print(f"Cranck-Nicolson - tridiagonal iterative: alpha = {alpha}")

  cdef double [:] u       = u0.copy()                       # current time
  cdef double [:] u_next  = u0.copy()
  cdef double [:] d       = np.zeros(N, dtype="float64")    # the vector of constants

  # Builds the tridiagonal matrix
  cdef double [:] diagonal        = (1+2*alpha)*np.ones(N, dtype="float64")   
  cdef double [:] second_diagonal = -alpha*np.ones(N, dtype="float64")

  second_diagonal[N-1] = 0.0

  evolve = lambda x: tridiag(second_diagonal, diagonal, second_diagonal, x)  # gets the next iteration with purely tridiag approx

  for _ in range(n_steps):

    # Builds the constant vector
    for i in range(N):
      left  = u[i-1] if i != 0   else u[N-1]
      right = u[i+1] if i != N-1 else u[0]
      d[i] =  alpha*left + (1-2*alpha)*u[i] + alpha*right

      if i == 0:
        d[i] += alpha*u[N-1] 
      if i == N-1:
        d[i] += alpha*u[0]

    # Solves the tridiagonal system for the next time
    u_next = evolve(d)

    u = u_next.copy()
  return np.array(u)

# def heat_cn_inv(double [:] u0, double nu =0.1, double dx=0.1, double dt=0.1, int n_steps = 10):

#   cdef int N = len(u0), i
#   cdef double alpha = 0.5*nu*dt/dx**2, left, right
#   print(f"Crank-Nicolson - exact inversion: alpha = {alpha}")

#   cdef double [:,:] A = np.zeros((N, N), dtype='float64'), B=np.zeros((N,N), dtype='float64')

#   # builds A and B
#   for i in range(N):
#     A[i, i] = 1 + 2*alpha
#     if i < N-1:
#       A[i, i+1] = -alpha
#     if i >= 1:
#       A[i, i-1] = -alpha 

#     B[i,i] = 1 - 2*alpha
#     if i < N-1:
#       B[i, i+1] = alpha
#     if i >= 1:
#       B[i, i-1] = alpha 

  
#   A[0, N-1] = -alpha 
#   A[N-1, 0] = -alpha

#   B[0, N-1] = alpha 
#   B[N-1, 0] = alpha

#   A , B = np.array(A), np.array(B)

#   cdef np.ndarray C = np.linalg.matrix_power( np.linalg.inv(A).dot(B), n_steps)

#   u = C.dot(u0)

#   return np.array(u)


def diff_advec(double [:] u0, double nu=0.1, double c=0.1, double dx=0.1, double dt=0.1, int n_steps = 10):
  cdef int N = len(u0), i, _
  cdef double alpha = 0.5*nu*dt/dx**2, beta = c*dt/4/dx, left = 0.0, right = 0.0
  print(f"Diffusion-Advection - tridiagonal iterative: alpha = {alpha}, beta = {beta}")

  cdef double [:] u       = u0.copy()                       # current time
  cdef double [:] u_next  = u0.copy()
  cdef double [:] d       = np.zeros(N, dtype="float64")    # the vector of constants

  # Builds the tridiagonal matrix
  cdef double [:] diagonal        =  (1+2*alpha) *np.ones(N, dtype="float64")   
  cdef double [:] upper_diagonal  = +(beta-alpha)*np.ones(N, dtype="float64")
  cdef double [:] lower_diagonal  = -(beta+alpha)*np.ones(N, dtype="float64")
  evolve = lambda x: tridiag(lower_diagonal, diagonal, upper_diagonal, x)  # gets the next iteration with purely tridiag approx

  for _ in range(n_steps):

    # Builds the constant vector
    for i in range(N):
      left  = u[i-1] if i != 0   else u[N-1]
      right = u[i+1] if i != N-1 else u[0]
      d[i]  = (beta + alpha)*left + (1-2*alpha)*u[i] + (alpha-beta)*right

      if i == 0:
        d[i] += (alpha + beta)*u[N-1] 
      if i == N-1:
        d[i] += (alpha - beta)*u[0]

    # Solves the tridiagonal system for the next time
    # print(f"lowdiag = {np.array(lower_diagonal)}")
    # print(f"updiag = {np.array(upper_diagonal)}")
    u_next = evolve(d)

    u = u_next.copy()
  return np.array(u)

