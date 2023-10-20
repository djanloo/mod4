from cython.parallel import prange
import numpy as np

cimport cython
cimport numpy as np

from time import perf_counter


cdef tridiag(double [:] lower, double [:] diag, double [:] upper, double [:] d):
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
  cdef double [:] d       = np.zeros(N)    # the vector of constants

  # Builds the tridiagonal matrix
  cdef double [:] diagonal        = (1+2*alpha)*np.ones(N)   
  cdef double [:] second_diagonal = -alpha*np.ones(N)

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


def diff_advec(double [:] u0, double nu=0.1, double c=0.1, double dx=0.1, double dt=0.1, int n_steps = 10):
  cdef int N = len(u0), i, _
  cdef double alpha = 0.5*nu*dt/dx**2, beta = c*dt/4/dx, left = 0.0, right = 0.0
  print(f"Diffusion-Advection - tridiagonal iterative: alpha = {alpha}, beta = {beta}")

  cdef double [:] u       = u0.copy()      # current time
  cdef double [:] d       = np.zeros(N)    # the vector of constants

  # Builds the tridiagonal matrix
  cdef double [:] diagonal        =  (1+2*alpha) *np.ones(N)   
  cdef double [:] upper_diagonal  = +(beta-alpha)*np.ones(N)
  cdef double [:] lower_diagonal  = -(beta+alpha)*np.ones(N)
  evolve = lambda x: tridiag(lower_diagonal, diagonal, upper_diagonal, x) 

  for _ in range(n_steps):

    # Builds the constant vector
    for i in range(N):
      left  = u[i-1] if i != 0   else 2*u[N-1]
      right = u[i+1] if i != N-1 else 2*u[0]
      d[i]  = (beta + alpha)*left + (1-2*alpha)*u[i] + (alpha-beta)*right

      # if i == 0:
      #   d[i] += (alpha + beta)*u[N-1] 
      # if i == N-1:
      #   d[i] += (alpha - beta)*u[0]

    u = evolve(d)

  return np.array(u)


def burgers_cn(double [:] u0, double nu=0.1, double dx=0.1, double dt=0.1, int n_steps = 10):
  cdef int N = len(u0), i, _
  cdef double alpha = 0.5*nu*dt/dx**2, beta = 0.5*dt/dx, left = 0.0, right = 0.0
  print(f"Burgers - Crank-Nicolson: alpha = {alpha}, beta = {beta}")

  cdef double [:] u = u0.copy()      # current time
  cdef double [:] d = np.zeros(N)    # the vector of constants

  # Builds the tridiagonal matrix
  cdef double [:] diagonal        = (1+2*alpha)*np.ones(N)   
  cdef double [:] upper_diagonal  = (-alpha)*np.ones(N)
  cdef double [:] lower_diagonal  = (-alpha)*np.ones(N)

  # Gets the next iteration with purely tridiag approx
  evolve = lambda x: tridiag(lower_diagonal, diagonal, upper_diagonal, x)  

  for _ in range(n_steps):

    # Builds the constant vector
    for i in range(N):
      left  = u[i-1] if i != 0   else u[N-1]
      right = u[i+1] if i != N-1 else u[0]
      d[i]  = (beta*u[i] + alpha)*left + (1-2*alpha)*u[i] + (alpha-beta*u[i])*right

      # Border conditions/tridiag approx
      if i == 0:
        d[i] += (alpha)*u[N-1] 
      if i == N-1:
        d[i] += (alpha)*u[0]

    u = evolve(d)

  return np.array(u)

cdef funker_plank_a1(x):
  return 1.0 * x

cdef funker_plank_a2(x, v):
  return 0.1 * v

cdef funker_plank_differential_a2(x,v):
 return 0

cpdef funker_plank(double [:,:] p0, 
                  double [:] x_values, double [:] v_values,
                  double dt= 0.1, # Integration parameters
                  double alpha=1.0, double gamma=0.1, double sigma=0.1, # Physical parameters
                  int n_steps = 10):
  ## Updates in Split-operator style

  cdef int N = len(x_values)
  cdef int M = len(v_values)
  # print(f"Matrix is {N} (x) times {M}(v)")
  cdef int t, i, j

  cdef double [:,:] p = p0.copy(), p_intermediate = p0.copy()
  cdef double [:] norm = np.zeros(n_steps)

  cdef double dx = np.diff(x_values)[0]
  cdef double dv = np.diff(v_values)[0]
  # print(f"dx = {dx}, dv = {dv}, dt = {dt}")

  # # ADI
  # dt /=2

  cdef double theta = dt/dv
  cdef double omega = dt/dx 
  cdef double eta = sigma*dt/dv**2
  # print(f"theta = {theta}, omega = {omega}, eta = {eta}, gamma*dt = {gamma*dt}")

  # Declarations of the diagonals
  cdef double [:] lower_x, diagonal_x, upper_x, b_x
  cdef double [:] lower_v, diagonal_v, upper_v, b_v

  lower_v, upper_v, b_v = np.ones(M), np.ones(M), np.ones(M)
  lower_x, upper_x, b_x = np.ones(N), np.ones(N), np.ones(N)

  # Diagonal of systems does not change
  diagonal_v = np.ones(M) * (1 + 2*eta - 0.5*gamma*dt)
  diagonal_x = np.ones(N)

  start = perf_counter()

  for t in range(n_steps):
    # First evolution: differential wrt V
    # For each value of x, a tridiagonal system is solved to find values of v
    for i in range(N):

      # Prepares tridiagonal matrix and the constant term
      for j in range(M):
        upper_v[j] = - eta - theta *  0.5 * (alpha * x_values[i] + gamma * v_values[j]) - 0.25*dt*gamma

        if j < M-1:
          lower_v[j] =  - eta + theta *  0.5 * (alpha * x_values[i] + gamma * v_values[j + 1]) - 0.25*dt*gamma

        # Standard operator split
        b_v[j] =  p[i,j]

        # # ADI: add the part of the other step
        # if i != 0:
        #   b_v[j] += - v_values[j] * 0.5 * omega *( - p[i-1, j]) 
        
        # if i != N-1:
        #   b_v[j] += - v_values[j] * 0.5 * omega * p[i+1, j]  
        
  
      # # # Boundary conditions
      # b_v[0]   -= (- eta + theta *  0.5 * (alpha * x_values[i] + gamma * v_values[0]) - 0.25*dt*gamma)*p[i, N-1]
      # b_v[M-1] -= (- eta - theta *  0.5 * (alpha * x_values[i] + gamma * v_values[M-1]) - 0.25*dt*gamma)*p[i, 0]
      b_v[0] = 0
      diagonal_v[0] = 1
      upper_v[0] = 0

      b_v[M-1] = 0
      diagonal_v[M-1] = 1
      lower_v[M-2] = 0

      # Performs the tridiag in the local i-value
      row = tridiag(lower_v, diagonal_v, upper_v, b_v)

      # Copies the row in p_ij
      for j in range(M):
        p_intermediate[i, j] = row[j]
    
    # Second evolution: differential wrt x
    # For each value of v, a tridiagonal system is solved to find values of x
    for j in range(M):

      # Prepares tridiagonal matrix and constant term
      for i in range(N):
        lower_x[i] = - 0.5 * omega * v_values[j]
        upper_x[i] =   0.5 * omega * v_values[j]

        # Standard split operator
        b_x[i] = p_intermediate[i,j]

        # # ADI
        # if j < M-1:
        #   b_x[i] -= (- eta - theta *  0.5 * (alpha * x_values[i] + gamma * v_values[j]) - 0.25*dt*gamma)*p_intermediate[i, j+1]
        # if j > 0 and j < M-1:
        #   b_x[i] -= (- eta + theta *  0.5 * (alpha * x_values[i] + gamma * v_values[j]) - 0.25*dt*gamma)*p_intermediate[i, j-1]

      # # Boundary conditions
      # b_x[0]   -= ( - 0.5 * omega * v_values[j])  * p[N-1, j]
      # b_x[N-1] -= ( + 0.5 * omega * v_values[j] ) * p[  0, j]

      b_x[0] = 0
      diagonal_x[0] = 1
      upper_x[0] = 0

      b_x[M-1] = 0
      diagonal_x[N-1] = 1
      lower_x[N-2] = 0

      row = tridiag(lower_x, diagonal_x, upper_x, b_x)
    
      for i in range(N):
        p[i,j] = row[i]

      # Takes trace of normalization
      sum = 0
      for i in range(N):
        for j in range(M):
          sum += p[i,j]
      
      norm[t] = sum * dx * dv

  print(f"Took {perf_counter() - start}")
  return p, norm

