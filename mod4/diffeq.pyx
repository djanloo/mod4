# distutils: language = c++
from libcpp cimport bool

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
  '''potential/ drag function'''
  return  -physical_params['omega_squared']*x - physical_params['gamma']*v

cdef double d_a(double x, double v, double time_index, dict physical_params):
  '''Derivative of potential wrt v'''
  return -physical_params['gamma']

cdef double sigma_squared(double x, double t, dict physical_params):
  '''Noise function'''
  return physical_params['sigma_squared']


def funker_plank( double [:,:] p0, 
                    double [:] x, double [:] v,
                    physical_params,
                    integration_params,
                    save_norm = False,
                    save_current=False
                    ):
  cdef double dt   = integration_params['dt']
  cdef unsigned int n_steps = integration_params['n_steps']
  cdef double t0    = physical_params.get('t0', 0.0)

  cdef unsigned int N = len(x)
  cdef unsigned int M = len(v)
  cdef unsigned int time_index = 0, i = 0, j = 0

  cdef double [:,:] p = p0.copy(), p_intermediate = p0.copy()
  cdef double [:] norm = np.zeros(n_steps)

  cdef double dx = np.diff(x)[0]
  cdef double dv = np.diff(v)[0]

  cdef double theta = 0.5 * dt/dv
  cdef double alpha = 0.5 * dt/dx 
  cdef double eta =   0.5 * dt/dv**2
  cdef double time = t0
  
  # Declarations of the diagonals
  cdef double [:] lower_x, diagonal_x, upper_x, b_x
  cdef double [:] lower_v, diagonal_v, upper_v, b_v

  lower_v, upper_v, b_v = np.ones(M), np.ones(M), np.ones(M)
  lower_x, upper_x, b_x = np.ones(N), np.ones(N), np.ones(N)

  diagonal_v = np.ones(M)
  diagonal_x = np.ones(N)

  cdef dict currents = dict(top=np.zeros(n_steps), 
                            bottom=np.zeros(n_steps), 
                            left=np.zeros(n_steps),
                            right=np.zeros(n_steps))


  for time_index in range(n_steps):
    time = t0 + time_index*dt
    # First evolution: differential wrt V
    # For each value of x, a tridiagonal system is solved to find values of v
    for i in range(N):

      # Prepares tridiagonal matrix and the constant term
      for j in range(M):
        
        diagonal_v[j] = 1  + 0.5 * dt * d_a(x[i],v[j], time, physical_params)
        diagonal_v[j] += 2 * eta * sigma_squared(x[i], time, physical_params)

        upper_v[j]  = - eta * sigma_squared(x[i], time, physical_params)
        upper_v[j] += theta * a(x[i], v[j], time, physical_params)
        upper_v[j] += 0.25 * dt * d_a(x[i], v[j], time, physical_params)
   
        lower_v[j] =  - eta * sigma_squared(x[i], time, physical_params)
        lower_v[j] -= theta * a(x[i], v[j] + dv, time, physical_params)
        lower_v[j] += 0.25 * dt * d_a(x[i], v[j] + dv, time, physical_params)

        b_v[j] =  p[j, i]  

      # Solves the tridiagonal system for the column
      p_intermediate[:, i]= tridiag(lower_v, diagonal_v, upper_v, b_v)

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
      
    # Takes trace of normalization
    if save_norm:
      norm[time_index] = quad_int(p, x, v)

    if save_current: 
      # Integral in v
      for j in range(M):
        currents['right'][time_index] += v[j]*p[j, N-2]*dv
        currents['left'][time_index] -= v[j]*p[j, 1]*dv

      # Integral in x
      for i in range(N):
        currents['top'][time_index] += a(x[i], v[M-2],  t0 + time_index*dt, physical_params)*p[M-2, i]*dx
        currents['top'][time_index] -= 0.5*physical_params['sigma_squared']*( (p[M-1,i] - p[M-2,i])/dv )*dx

        currents['bottom'][time_index] -= a(x[i], v[2],  t0 + time_index*dt, physical_params)*p[2, i]*dx
        currents['bottom'][time_index] += 0.5*physical_params['sigma_squared']**2*( (p[1,i] - p[0,i])/dv )*dx

  return p, norm, currents

def funker_plank_cn( double [:,:] p0, 
                    double [:] x, double [:] v,
                    physical_params,
                    integration_params,
                    save_norm = False,
                    save_current=False
                    ):
  cdef double dt   = integration_params['dt']
  cdef unsigned int n_steps = integration_params['n_steps']
  cdef double t0    = physical_params.get('t0', 0.0)

  cdef unsigned int N = len(x)
  cdef unsigned int M = len(v)
  cdef unsigned int time_index = 0, i = 0, j = 0

  cdef double [:,:] p = p0.copy(), p_intermediate = p0.copy()
  cdef double [:] norm = np.zeros(n_steps)

  cdef double dx = np.diff(x)[0]
  cdef double dv = np.diff(v)[0]

  cdef double theta = 0.25*dt/dv   ##CN
  cdef double alpha = 0.25 * dt/dx ##CN
  cdef double eta = 0.25*dt/dv**2  ## CN
  cdef double time = t0
  
  # Declarations of the diagonals
  cdef double [:] lower_x, diagonal_x, upper_x, b_x
  cdef double [:] lower_v, diagonal_v, upper_v, b_v

  lower_v, upper_v, b_v = np.ones(M), np.ones(M), np.ones(M)
  lower_x, upper_x, b_x = np.ones(N), np.ones(N), np.ones(N)

  diagonal_v = np.ones(M)
  diagonal_x = np.ones(N)

  cdef dict currents = dict(top=np.zeros(n_steps), 
                            bottom=np.zeros(n_steps), 
                            left=np.zeros(n_steps),
                            right=np.zeros(n_steps))

  for time_index in range(n_steps):
    time = t0 + time_index*dt
    # First evolution: differential wrt V
    # For each value of x, a tridiagonal system is solved to find values of v
    for i in range(N):
      
      # Prepares tridiagonal matrix and the constant term
      for j in range(M):
        
        diagonal_v[j] = 1  + 0.5* 0.5 * dt * d_a(x[i],v[j], time, physical_params)
        diagonal_v[j] += 2 * eta * sigma_squared(x[i], time, physical_params)

        upper_v[j]  = - eta * sigma_squared(x[i], time, physical_params)
        upper_v[j] += theta * a(x[i], v[j], time, physical_params)
        upper_v[j] += 0.5 * 0.25 * dt * d_a(x[i], v[j], time, physical_params)
   
        lower_v[j] =  - eta * sigma_squared(x[i], time, physical_params)
        lower_v[j] -= theta * a(x[i], v[j] + dv, time, physical_params)
        lower_v[j] += 0.5 * 0.25* dt * d_a(x[i], v[j] + dv, time, physical_params)

        b_v[j] =  p[j, i]  

        ## CN
        if j > 0 and j < M-1:
          # Diffusion
          b_v[j] +=   (eta *sigma_squared(x[i], time, physical_params) )   * p[j+1,i] 
          b_v[j] +=   (-2*eta *sigma_squared(x[i], time, physical_params) )* p[j,i]
          b_v[j] +=   (eta * sigma_squared(x[i], time, physical_params) )  * p[j-1, i]

          # Drift
          b_v[j] += (-0.25 * 0.5 * dt * d_a(x[i], v[j], time, physical_params) - theta * a(x[i], v[j], time, physical_params))*p[j+1, i]
          b_v[j] += (-0.5  * 0.5 * dt * d_a(x[i], v[j], time, physical_params))*p[j, i]
          b_v[j] += (-0.25 * 0.5 * dt * d_a(x[i], v[j], time, physical_params) + theta * a(x[i], v[j], time, physical_params))*p[j-1, i]

      ## Raw conservation of norm
      diagonal_v[0] = 1 - lower_v[0]
      diagonal_v[M-1] = 1- upper_v[M-2]

      # Solves the tridiagonal system for the column
      p_intermediate[:, i] = tridiag(lower_v, diagonal_v, upper_v, b_v)

    # Second evolution: differential wrt x
    # For each value of v, a tridiagonal system is solved to find values of x
    for j in range(M):

      # Prepares tridiagonal matrix and constant term
      for i in range(N):
        lower_x[i] = - alpha * v[j]
        upper_x[i] =   alpha * v[j] 

        b_x[i] = p_intermediate[j, i] 
        
        ## CN
        if i != 0 and i != N-1:
          b_x[i] += alpha * v[j] * (p_intermediate[j, i-1] - p_intermediate[j, i+1])

      ## Raw conservation of norm
      diagonal_x[0] = 1 - lower_x[0]
      diagonal_x[M-1] = 1 - upper_x[M-2]
      
      # Solves the tridiagonal system for the row
      p[j, :] =  tridiag(lower_x, diagonal_x, upper_x, b_x)

    # Takes trace of normalization
    if save_norm:
      norm[time_index] = quad_int(p, x, v)

    if save_current: 
      # Integral in v
      for j in range(M):
        currents['right'][time_index] += v[j]*p[j, N-2]*dv
        currents['left'][time_index]  -= v[j]*p[j, 1]*dv

      # Integral in x
      for i in range(N):
        currents['top'][time_index] += a(x[i], v[M-2],  t0 + time_index*dt, physical_params)*p[M-2, i]*dx
        currents['top'][time_index] -= 0.5*physical_params['sigma_squared']*( (p[M-1,i] - p[M-2,i])/dv )*dx

        currents['bottom'][time_index] -= a(x[i], v[2],  t0 + time_index*dt, physical_params)*p[2, i]*dx
        currents['bottom'][time_index] += 0.5*physical_params['sigma_squared']**2*( (p[1,i] - p[0,i])/dv )*dx

  return p, norm, currents

# def funker_plank_original( double [:,:] p0, 
#                     double [:] x, double [:] v,
#                     physical_params,
#                     integration_params,
#                     save_norm = False,
#                     save_current=False
#                     ):
#   cdef double dt   = integration_params['dt']
#   cdef unsigned int n_steps = integration_params['n_steps']
#   cdef float t0    = physical_params.get('t0', 0.0)

#   cdef unsigned int N = len(x)
#   cdef unsigned int M = len(v)
#   cdef unsigned int time_index = 0, i = 0, j = 0

#   cdef double [:,:] p = p0.copy(), p_intermediate = p0.copy()
#   cdef double [:] norm = np.zeros(n_steps)

#   cdef double dx = np.diff(x)[0]
#   cdef double dv = np.diff(v)[0]

#   cdef double theta = 0.5 * dt/dv
#   cdef double alpha = 0.5 * dt/dx 
#   cdef double eta = 0.5*physical_params['sigma_squared']*dt/dv**2
  
#   # Declarations of the diagonals
#   cdef double [:] lower_x, diagonal_x, upper_x, b_x
#   cdef double [:] lower_v, diagonal_v, upper_v, b_v

#   lower_v, upper_v, b_v = np.ones(M), np.ones(M), np.ones(M)
#   lower_x, upper_x, b_x = np.ones(N), np.ones(N), np.ones(N)

#   # Diagonal of systems does not change
#   diagonal_v = np.ones(M) * (1 + 2*eta - 0.5*d_a(0,0,0, physical_params)*dt)
#   diagonal_x = np.ones(N)

#   cdef dict currents = dict(top=np.zeros(n_steps), 
#                             bottom=np.zeros(n_steps), 
#                             left=np.zeros(n_steps),
#                             right=np.zeros(n_steps))
#   # Boundary: pre
#   for i in range(N):
#     p[0,i] = 0
#     p[M-1, i] = 0
  
#   for j in range(M):
#     p[j, 0] = 0
#     p[j, N-1] = 0

#   for time_index in range(n_steps):
#     # First evolution: differential wrt V
#     # For each value of x, a tridiagonal system is solved to find values of v
#     for i in range(N):

#       # Prepares tridiagonal matrix and the constant term
#       for j in range(M):
#         diagonal_v[j] = 1 + 2*eta - theta*(a(x[i], v[i] + 0.5*dv, time_index*dt, physical_params) - a(x[i], v[i] - 0.5*dv, time_index*dt, physical_params))
        
#         upper_v[j]  = - eta
#         upper_v[j] -= theta * a(x[i], v[j] + 0.5 * dv, t0 + time_index*dt, physical_params)

#         lower_v[j] =  - eta
#         lower_v[j] += theta * a(x[i], v[j] + dv - 0.5 * dv, t0 + time_index*dt, physical_params)

#         b_v[j] =  p[j, i]      


#       # Solves the tridiagonal system for the column
#       p_intermediate[:, i]= tridiag(lower_v, diagonal_v, upper_v, b_v)

#       # Boundary conditions
#       p_intermediate[0, i] = 0.0
#       p_intermediate[M-1, i] = 0.0

#     # Second evolution: differential wrt x
#     # For each value of v, a tridiagonal system is solved to find values of x
#     for j in range(M):

#       # Prepares tridiagonal matrix and constant term
#       for i in range(N):
#         lower_x[i] = - alpha * v[j]
#         upper_x[i] =   alpha * v[j]

#         b_x[i] = p_intermediate[j, i]

#       # Solves the tridiagonal system for the row
#       p[j, :] =  tridiag(lower_x, diagonal_x, upper_x, b_x)

#       # Boundary conditions
#       p[j, 0] = 0.0
#       p[j, N-1] = 0.0
      
#     # Takes trace of normalization
#     if save_norm:
#       norm[time_index] = quad_int(p, x, v)

#     if save_current: 
#       # Integral in v
#       for j in range(M):
#         currents['right'][time_index] += v[j]*p[j, N-2]*dv
#         currents['left'][time_index] -= v[j]*p[j, 1]*dv

#       # Integral in x
#       for i in range(N):
#         currents['top'][time_index] += a(x[i], v[M-2],  t0 + time_index*dt, physical_params)*p[M-2, i]*dx
#         currents['top'][time_index] += 0.5*physical_params['sigma']**2*( (p[M-1,i] - p[M-2,i])/dv )*dx

#         currents['bottom'][time_index] -= a(x[i], v[2],  t0 + time_index*dt, physical_params)*p[2, i]*dx
#         currents['bottom'][time_index] -= 0.5*physical_params['sigma']**2*( (p[1,i] - p[0,i])/dv )*dx

#   return p, norm, currents

def generic_3_step( double [:,:] p0, 
                    double [:] x, double [:] v,
                    physical_params,
                    integration_params,
                    save_norm = False,
                    save_current=False
                    ):
  cdef double dt   = integration_params['dt']
  cdef unsigned int n_steps = integration_params['n_steps']
  cdef double t0    = physical_params.get('t0', 0.0)

  cdef bool [:] CN_ized_steps = integration_params.get('CN', [])

  cdef unsigned int N = len(x)
  cdef unsigned int M = len(v)
  cdef unsigned int time_index = 0, i = 0, j = 0

  cdef double [:,:] p = p0.copy(), p_star = p0.copy(), p_dagger = p0.copy()
  cdef double [:] norm = np.zeros(n_steps)

  cdef double dx = np.diff(x)[0]
  cdef double dv = np.diff(v)[0]

  cdef double theta = 0.5 * dt/dv   
  cdef double alpha = 0.5 * dt/dx
  cdef double eta   = 0.5 * dt/dv**2 
  cdef double time  = t0

  # Halves the bros in case Crank-Nicholson is chosen
  if CN_ized_steps[0]:
    print("V-drift Crank-Nicholson-ized")
    theta = 0.5 * theta

  if CN_ized_steps[1]:
    print("X-drift Crank-Nicholson-ized")
    alpha = 0.5 * alpha

  if CN_ized_steps[2]:
    print("Diffusion Crank-Nicholson-ized")
    eta = 0.5 * eta

  # Declarations of the diagonals
  cdef double [:] lower_1, diagonal_1, upper_1, b_1
  cdef double [:] lower_2, diagonal_2, upper_2, b_2
  cdef double [:] lower_3, diagonal_3, upper_3, b_3

  diagonal_1, lower_1, upper_1, b_1 = np.ones(M), np.ones(M), np.ones(M), np.ones(M)
  diagonal_2, lower_2, upper_2, b_2 = np.ones(N), np.ones(N), np.ones(N), np.ones(N)
  diagonal_3, lower_3, upper_3, b_3 = np.ones(M), np.ones(M), np.ones(M), np.ones(M)

  cdef dict currents = dict(top=np.zeros(n_steps), 
                            bottom=np.zeros(n_steps), 
                            left=np.zeros(n_steps),
                            right=np.zeros(n_steps))

  for time_index in range(n_steps):
    time = t0 + time_index*dt
    ################################### First evolution: v-drift ######################################
    for i in range(N):
      
      b_1 =  p[:, i] 
      for j in range(M):
        
        diagonal_1[j] = 1 + theta * dv * d_a(x[i],v[j], time, physical_params)

        upper_1[j] =   theta * a(x[i], v[j], time, physical_params)
        upper_1[j] += 0.5 * theta * dv * d_a(x[i], v[j], time, physical_params)
   
        lower_1[j] = - theta * a(x[i], v[j] + dv, time, physical_params)
        lower_1[j] += 0.5 * theta * dv * d_a(x[i], v[j] + dv, time, physical_params)

        if CN_ized_steps[0]:
          if j > 0 and j < M-1:

            # Drift
            b_1[j] += (- 0.5 * theta * dv * d_a(x[i], v[j], time, physical_params) - theta * a(x[i], v[j], time, physical_params))*p[j+1, i]
            b_1[j] += (- theta * dv * d_a(x[i], v[j], time, physical_params))*p[j, i]
            b_1[j] += (- 0.5 * theta * dv * d_a(x[i], v[j], time, physical_params) + theta * a(x[i], v[j], time, physical_params))*p[j-1, i]


      # Solves the tridiagonal system for the column
      p_star[:, i] = tridiag(lower_1, diagonal_1, upper_1, b_1)
      print(f"step 1 ampl: {np.sum(p_star[:, i])/np.sum(p[:, i])}")

    ################################### Second evolution: x-drift ######################################
    for j in range(M):

      # Prepares tridiagonal matrix and constant term
      for i in range(N):
        lower_2[i] = - alpha * v[j]
        upper_2[i] =   alpha * v[j] 

        b_2[i] = p_star[j, i] 
        
        if CN_ized_steps[1]:
          if i != 0 and i != N-1:
            b_2[i] += alpha * v[j] * (p_star[j, i-1] - p_star[j, i+1])
      
      # Solves the tridiagonal system for the row
      p_dagger[j, :] =  tridiag(lower_2, diagonal_2, upper_2, b_2)
      print(f"step 2 ampl: {np.sum(p_dagger[j, :])/np.sum(p_star[j, :])}")

    ################################### Third evolution: diffusion ######################################
    for i in range(N):
      for j in range(M):
        
        diagonal_3[j] = 1 + 2 * eta * sigma_squared(x[i], time, physical_params)
        upper_3[j]    = - eta * sigma_squared(x[i], time, physical_params)
        lower_3[j]    =  - eta * sigma_squared(x[i], time, physical_params)
        b_3[j] =  p_dagger[j, i]  

        if CN_ized_steps[2]:
          if j > 0 and j < M-1:
            b_3[j] +=   (   eta * sigma_squared(x[i], time, physical_params))* p_dagger[j+1,i] 
            b_3[j] +=   (-2*eta * sigma_squared(x[i], time, physical_params))* p_dagger[j,i]
            b_3[j] +=   (   eta * sigma_squared(x[i], time, physical_params))* p_dagger[j-1, i]

      # ## Raw conservation of norm
      # diagonal_v[0] = 1 - lower_v[0]
      # diagonal_v[M-1] = 1- upper_v[M-2]

      # Solves the tridiagonal system for the column
      p[:, i] = tridiag(lower_3, diagonal_3, upper_3, b_3)
      print(f"step 3 ampl: {np.sum(p[:, i])/np.sum(p_dagger[:, i])}")

    ##################################### UTILS #####################################
    # Takes trace of normalization
    if save_norm:
      norm[time_index] = quad_int(p, x, v)

    if save_current: 
      # Integral in v
      for j in range(M):
        currents['right'][time_index] += v[j]*p[j, N-2]*dv
        currents['left'][time_index]  -= v[j]*p[j, 1]*dv

      # Integral in x
      for i in range(N):
        currents['top'][time_index] += a(x[i], v[M-2],  t0 + time_index*dt, physical_params)*p[M-2, i]*dx
        currents['top'][time_index] -= 0.5*physical_params['sigma_squared']*( (p[M-1,i] - p[M-2,i])/dv )*dx

        currents['bottom'][time_index] -= a(x[i], v[2],  t0 + time_index*dt, physical_params)*p[2, i]*dx
        currents['bottom'][time_index] += 0.5*physical_params['sigma_squared']**2*( (p[1,i] - p[0,i])/dv )*dx

  return p, norm, currents
