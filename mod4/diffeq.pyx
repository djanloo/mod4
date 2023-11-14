# distutils: language = c++
from libcpp cimport bool

from cython.parallel import prange
import numpy as np

cimport cython
from cython.parallel import prange
cimport numpy as np

from time import perf_counter
from libc.math cimport sin

from .utils import quad_int, get_tridiag

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
  return  -physical_params['omega_squared']*x - physical_params['gamma']*v*(x**2 - 1)

cdef double sigma_squared(double x, double t, dict physical_params):
  '''Noise function'''
  return physical_params['sigma_squared']

def generic_3_step( double [:,:] p0, 
                    physical_params,
                    integration_params,
                    save_norm = False,
                    save_current=False
                    ):
  ## Time
  cdef double dt   = integration_params['dt']
  cdef unsigned int n_steps = integration_params['n_steps']
  cdef double t0    = physical_params.get('t0', 0.0)

  ## Space
  cdef double Lx, Lv,dx,dv
  Lx, Lv,dx,dv = map(integration_params.get, ["Lx", "Lv", "dx", "dv"])
  cdef unsigned int N = int(Lx/dx), M = int(Lv/dv)
  cdef double [:] x = np.arange(-int(N)//2, int(N)//2)*dx

  cdef double [:] v = np.arange(-int(M)//2, int(M)//2)*dv

  ## Add-ons
  cdef bool [:] CN_ized_steps = integration_params.get('CN', np.array([False, False, False]))
  cdef bool ADI = integration_params.get("ADI", False)
  cdef bool upwind_mode = integration_params.get("upwind_mode", False)

  cdef unsigned int time_index = 0, i = 0, j = 0

  cdef double [:,:] p = p0.copy(), p_star = p0.copy(), p_dagger = p0.copy()
  cdef double [:] norm = np.zeros(n_steps)

  if ADI:
    print("Set to ADI mode")
    dt = dt/3.0

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

  # Working variables
  cdef double a_plus, a_minus, s, a_here

  cdef dict currents = dict(top=np.zeros(n_steps), 
                            bottom=np.zeros(n_steps), 
                            left=np.zeros(n_steps),
                            right=np.zeros(n_steps))

  for time_index in range(n_steps):
    time = t0 + time_index*dt
    ################################### First evolution: v-drift ######################################
    for i in range(N):
      
      # Constant part of coefficient vector does not depend on j
      b_1 =  p[:, i].copy()
      
      # Diffusion coefficient does not depend on j
      s = eta * sigma_squared(x[i], time, physical_params)

      for j in range(M):

        # Note tha theta is absorbed in working variable
        a_plus  =  theta * a(x[i],v[j] + 0.5*dv, time, physical_params)
        a_minus =  theta * a(x[i],v[j] - 0.5*dv, time, physical_params)

        diagonal_1[j] = 1 +  a_plus - a_minus
        upper_1[j]    =   + a_plus
        # Since lower has an offset of one 
        # (write the matrix down and you'll see)
        lower_1[j]    =   -  a_plus
        
        if ADI:
          # ADI-sytle
          if i != 0 and i != N-1:
            # X-drift
            b_1[j] += - alpha * v[j] * (p[j, i+1] - p[j, i-1])
          
          if j != 0 and j != M-1:
            # Diffusion
            b_1[j] +=   (   s)* p[j+1,i] 
            b_1[j] +=   (-2*s)* p[j,i]
            b_1[j] +=   (   s)* p[j-1, i]

        if CN_ized_steps[0]:
          if j > 0 and j < M-1:

            # Drift
            b_1[j] += - a_plus*p[j+1, i]
            b_1[j] += (a_minus - a_plus)*p[j, i]
            b_1[j] += a_minus*p[j-1, i]

      # Solves the tridiagonal system for the column
      p_star[:, i] = tridiag(lower_1, diagonal_1, upper_1, b_1)
      # amplification_average[0] += np.sum(p_star[:, i])/np.sum(p[:,i])/N/n_steps
      # print(f"Sum of V-rows\n{np.sum(get_tridiag(lower_1, diagonal_1, upper_1), axis=1)}")      
      # print(f"Sum of V-cols\n{np.sum(get_tridiag(lower_1, diagonal_1, upper_1), axis=0)}")

    ################################### Second evolution: x-drift ######################################
    for j in range(M):

      # Does not depend on i
      b_2 = p_star[j, :].copy()

      for i in range(N):
        lower_2[i] = - alpha * v[j]
        upper_2[i] =   alpha * v[j] 

        b_2[i] = p_star[j, i]

        if ADI:
          # ADI-style
          a_plus  = theta * a(x[i], v[j] + 0.5*dv, time, physical_params)
          a_minus = theta * a(x[i], v[j] - 0.5*dv, time, physical_params)
          s = eta * sigma_squared(x[i], time, physical_params)
          if j != 0 and j != M-1:
            
            # V-Drift
            b_2[j] += (- a_plus)*p_star[j+1, i]
            b_2[j] += ( a_minus - a_plus)*p_star[j, i]
            b_2[j] += (+ a_minus)*p_star[j-1, i]

            # Diffusion
            b_2[i] +=   (   s)* p_star[j+1,i] 
            b_2[i] +=   (-2*s)* p_star[j,i]
            b_2[i] +=   (   s)* p_star[j-1, i]
        
        if CN_ized_steps[1]:
          if i != 0 and i != N-1:
            b_2[i] += alpha * v[j] * (p_star[j, i-1] - p_star[j, i+1])
      
      # Solves the tridiagonal system for the row
      p_dagger[j, :] =  tridiag(lower_2, diagonal_2, upper_2, b_2)
      # amplification_average[1] += np.sum(p_dagger[j, :])/np.sum(p_star[j,:])/M/n_steps

    ################################### Third evolution: diffusion ######################################
    for i in range(N):
      s = eta * sigma_squared(x[i], time, physical_params)
      b_3 = p_dagger[:, i].copy()
      for j in range(M):
        
        diagonal_3[j] = 1 + 2 * s
        upper_3[j]    = - s
        lower_3[j]    = - s
        
        if ADI:
          a_plus  = theta * a(x[i], v[j] + 0.5*dv, time, physical_params)
          a_minus = theta * a(x[i], v[j] - 0.5*dv, time, physical_params)
          # V-drift
          if j!=0 and j!= M-1:
            b_3[j] += (- theta * a(x[i], v[j] + 0.5*dv, time, physical_params))*p_dagger[j+1, i]
            b_3[j] += (- theta *( a(x[i],v[j] + 0.5*dv, time, physical_params) - a(x[i],v[j] - 0.5*dv, time, physical_params)) )*p_dagger[j, i]
            b_3[j] += (+ theta * a(x[i], v[j] - 0.5*dv, time, physical_params))*p_dagger[j-1, i]
          if i != 0 and i != N-1:
            # X-drift
            b_3[j] += alpha * v[j] * (p_dagger[j, i-1] - p_dagger[j, i+1])

        if CN_ized_steps[2]:
          if j > 0 and j < M-1:
            b_3[j] +=   (   s)* p_dagger[j+1,i] 
            b_3[j] +=   (-2*s)* p_dagger[j,i]
            b_3[j] +=   (   s)* p_dagger[j-1, i]

      # ## Raw conservation of norm
      # diagonal_v[0] = 1 - lower_v[0]
      # diagonal_v[M-1] = 1- upper_v[M-2]

      # Solves the tridiagonal system for the column
      p[:, i] = tridiag(lower_3, diagonal_3, upper_3, b_3)
      # amplification_average[2] += np.sum(p[:, i])/np.sum(p_dagger[:,i])/N/n_steps

    ##################################### UTILS #####################################
    # Takes trace of normalization
    if save_norm:
      norm[time_index] = quad_int(p, integration_params)

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


# def funker_plank( double [:,:] p0,
#                     physical_params,
#                     integration_params,
#                     save_norm = False,
#                     save_current=False
#                     ):
#   ## Time
#   cdef double dt   = integration_params['dt']
#   cdef unsigned int n_steps = integration_params['n_steps']
#   cdef double t0    = physical_params.get('t0', 0.0)

#   ## Space
#   cdef double Lx, Lv,dx,dv
#   Lx, Lv,dx,dv = map(integration_params.get, ["Lx", "Lv", "dx", "dv"])
#   cdef unsigned int N = int(Lx/dx), M = int(Lv/dv)
#   cdef double [:] x = np.arange(-int(N)//2, int(N)//2)*dx

#   cdef double [:] v = np.arange(-int(M)//2, int(M)//2)*dv

#   cdef unsigned int time_index = 0, i = 0, j = 0

#   cdef double [:,:] p = p0.copy(), p_intermediate = p0.copy()
#   cdef double [:] norm = np.zeros(n_steps)

#   cdef double theta = 0.5 * dt/dv
#   cdef double alpha = 0.5 * dt/dx 
#   cdef double eta =   0.5 * dt/dv**2
#   cdef double time = t0
  
#   # Declarations of the diagonals
#   cdef double [:] lower_x, diagonal_x, upper_x, b_x
#   cdef double [:] lower_v, diagonal_v, upper_v, b_v

#   lower_v, upper_v, b_v = np.ones(M), np.ones(M), np.ones(M)
#   lower_x, upper_x, b_x = np.ones(N), np.ones(N), np.ones(N)

#   diagonal_v = np.ones(M)
#   diagonal_x = np.ones(N)

#   cdef dict currents = dict(top=np.zeros(n_steps), 
#                             bottom=np.zeros(n_steps), 
#                             left=np.zeros(n_steps),
#                             right=np.zeros(n_steps))


#   for time_index in range(n_steps):
#     time = t0 + time_index*dt
#     # First evolution: differential wrt V
#     # For each value of x, a tridiagonal system is solved to find values of v
#     for i in range(N):

#       # Prepares tridiagonal matrix and the constant term
#       for j in range(M):
        
#         diagonal_v[j] = 1  + 0.5 * dt * d_a(x[i],v[j], time, physical_params)
#         diagonal_v[j] += 2 * eta * sigma_squared(x[i], time, physical_params)

#         upper_v[j]  = - eta * sigma_squared(x[i], time, physical_params)
#         upper_v[j] += theta * a(x[i], v[j], time, physical_params)
#         upper_v[j] += 0.25 * dt * d_a(x[i], v[j], time, physical_params)
   
#         lower_v[j] =  - eta * sigma_squared(x[i], time, physical_params)
#         lower_v[j] -= theta * a(x[i], v[j] + dv, time, physical_params)
#         lower_v[j] += 0.25 * dt * d_a(x[i], v[j] + dv, time, physical_params)

#         b_v[j] =  p[j, i]  

#       # Solves the tridiagonal system for the column
#       p_intermediate[:, i]= tridiag(lower_v, diagonal_v, upper_v, b_v)

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
      
#     # Takes trace of normalization
#     if save_norm:
#       norm[time_index] = quad_int(p, integration_params)

#     if save_current: 
#       # Integral in v
#       for j in range(M):
#         currents['right'][time_index] += v[j]*p[j, N-2]*dv
#         currents['left'][time_index] -= v[j]*p[j, 1]*dv

#       # Integral in x
#       for i in range(N):
#         currents['top'][time_index] += a(x[i], v[M-2],  t0 + time_index*dt, physical_params)*p[M-2, i]*dx
#         currents['top'][time_index] -= 0.5*physical_params['sigma_squared']*( (p[M-1,i] - p[M-2,i])/dv )*dx

#         currents['bottom'][time_index] -= a(x[i], v[2],  t0 + time_index*dt, physical_params)*p[2, i]*dx
#         currents['bottom'][time_index] += 0.5*physical_params['sigma_squared']**2*( (p[1,i] - p[0,i])/dv )*dx

#   return p, norm, currents

# def funker_plank_cn( double [:,:] p0,
#                     physical_params,
#                     integration_params,
#                     save_norm = False,
#                     save_current=False
#                     ):
#   ## Time
#   cdef double dt   = integration_params['dt']
#   cdef unsigned int n_steps = integration_params['n_steps']
#   cdef double t0    = physical_params.get('t0', 0.0)

#   ## Space
#   cdef double Lx, Lv,dx,dv
#   Lx, Lv,dx,dv = map(integration_params.get, ["Lx", "Lv", "dx", "dv"])
#   cdef unsigned int N = int(Lx/dx), M = int(Lv/dv)
#   cdef double [:] x = np.arange(-int(N)//2, int(N)//2)*dx

#   cdef double [:] v = np.arange(-int(M)//2, int(M)//2)*dv

#   cdef unsigned int time_index = 0, i = 0, j = 0

#   cdef double [:,:] p = p0.copy(), p_intermediate = p0.copy()
#   cdef double [:] norm = np.zeros(n_steps)

#   cdef double theta = 0.25*dt/dv   ##CN
#   cdef double alpha = 0.25 * dt/dx ##CN
#   cdef double eta = 0.25*dt/dv**2  ## CN
#   cdef double time = t0
  
#   # Declarations of the diagonals
#   cdef double [:] lower_x, diagonal_x, upper_x, b_x
#   cdef double [:] lower_v, diagonal_v, upper_v, b_v

#   lower_v, upper_v, b_v = np.ones(M), np.ones(M), np.ones(M)
#   lower_x, upper_x, b_x = np.ones(N), np.ones(N), np.ones(N)

#   diagonal_v = np.ones(M)
#   diagonal_x = np.ones(N)

#   cdef dict currents = dict(top=np.zeros(n_steps), 
#                             bottom=np.zeros(n_steps), 
#                             left=np.zeros(n_steps),
#                             right=np.zeros(n_steps))

#   for time_index in range(n_steps):
#     time = t0 + time_index*dt
#     # First evolution: differential wrt V
#     # For each value of x, a tridiagonal system is solved to find values of v
#     for i in range(N):
      
#       # Prepares tridiagonal matrix and the constant term
#       for j in range(M):
        
#         diagonal_v[j] = 1  + 0.5* 0.5 * dt * d_a(x[i],v[j], time, physical_params)
#         diagonal_v[j] += 2 * eta * sigma_squared(x[i], time, physical_params)

#         upper_v[j]  = - eta * sigma_squared(x[i], time, physical_params)
#         upper_v[j] += theta * a(x[i], v[j], time, physical_params)
#         upper_v[j] += 0.5 * 0.25 * dt * d_a(x[i], v[j], time, physical_params)
   
#         lower_v[j] =  - eta * sigma_squared(x[i], time, physical_params)
#         lower_v[j] -= theta * a(x[i], v[j] + dv, time, physical_params)
#         lower_v[j] += 0.5 * 0.25* dt * d_a(x[i], v[j] + dv, time, physical_params)

#         b_v[j] =  p[j, i]  

#         ## CN
#         if j > 0 and j < M-1:
#           # Diffusion
#           b_v[j] +=   (eta *sigma_squared(x[i], time, physical_params) )   * p[j+1,i] 
#           b_v[j] +=   (-2*eta *sigma_squared(x[i], time, physical_params) )* p[j,i]
#           b_v[j] +=   (eta * sigma_squared(x[i], time, physical_params) )  * p[j-1, i]

#           # Drift
#           b_v[j] += (-0.25 * 0.5 * dt * d_a(x[i], v[j], time, physical_params) - theta * a(x[i], v[j], time, physical_params))*p[j+1, i]
#           b_v[j] += (-0.5  * 0.5 * dt * d_a(x[i], v[j], time, physical_params))*p[j, i]
#           b_v[j] += (-0.25 * 0.5 * dt * d_a(x[i], v[j], time, physical_params) + theta * a(x[i], v[j], time, physical_params))*p[j-1, i]

#       ## Raw conservation of norm
#       diagonal_v[0] = 1 - lower_v[0]
#       diagonal_v[M-1] = 1- upper_v[M-2]

#       # Solves the tridiagonal system for the column
#       p_intermediate[:, i] = tridiag(lower_v, diagonal_v, upper_v, b_v)

#     # Second evolution: differential wrt x
#     # For each value of v, a tridiagonal system is solved to find values of x
#     for j in range(M):

#       # Prepares tridiagonal matrix and constant term
#       for i in range(N):
#         lower_x[i] = - alpha * v[j]
#         upper_x[i] =   alpha * v[j] 

#         b_x[i] = p_intermediate[j, i] 
        
#         ## CN
#         if i != 0 and i != N-1:
#           b_x[i] += alpha * v[j] * (p_intermediate[j, i-1] - p_intermediate[j, i+1])

#       ## Raw conservation of norm
#       diagonal_x[0] = 1 - lower_x[0]
#       diagonal_x[M-1] = 1 - upper_x[M-2]
      
#       # Solves the tridiagonal system for the row
#       p[j, :] =  tridiag(lower_x, diagonal_x, upper_x, b_x)

#     # Takes trace of normalization
#     if save_norm:
#       norm[time_index] = quad_int(p, integration_params)

#     if save_current: 
#       # Integral in v
#       for j in range(M):
#         currents['right'][time_index] += v[j]*p[j, N-2]*dv
#         currents['left'][time_index]  -= v[j]*p[j, 1]*dv

#       # Integral in x
#       for i in range(N):
#         currents['top'][time_index] += a(x[i], v[M-2],  t0 + time_index*dt, physical_params)*p[M-2, i]*dx
#         currents['top'][time_index] -= 0.5*physical_params['sigma_squared']*( (p[M-1,i] - p[M-2,i])/dv )*dx

#         currents['bottom'][time_index] -= a(x[i], v[2],  t0 + time_index*dt, physical_params)*p[2, i]*dx
#         currents['bottom'][time_index] += 0.5*physical_params['sigma_squared']**2*( (p[1,i] - p[0,i])/dv )*dx

#   return p, norm, currents

def funker_plank_original( double [:,:] p0,
                    physical_params,
                    integration_params,
                    save_norm = False,
                    save_current=False
                    ):
  ## Time
  cdef double dt   = integration_params['dt']
  cdef unsigned int n_steps = integration_params['n_steps']
  cdef double t0    = physical_params.get('t0', 0.0)

  ## Space
  cdef double Lx, Lv,dx,dv
  Lx, Lv,dx,dv = map(integration_params.get, ["Lx", "Lv", "dx", "dv"])
  cdef unsigned int N = int(Lx/dx), M = int(Lv/dv)
  cdef double [:] x = np.arange(-int(N)//2, int(N)//2)*dx

  cdef double [:] v = np.arange(-int(M)//2, int(M)//2)*dv

  cdef unsigned int time_index = 0, i = 0, j = 0

  cdef double [:,:] p = p0.copy(), p_intermediate = p0.copy()
  cdef double [:] norm = np.zeros(n_steps)

  cdef double theta = 0.5 * dt/dv
  cdef double alpha = 0.5 * dt/dx 
  cdef double eta = 0.5*physical_params['sigma_squared']*dt/dv**2
  
  # Declarations of the diagonals
  cdef double [:] lower_x, diagonal_x, upper_x, b_x
  cdef double [:] lower_v, diagonal_v, upper_v, b_v

  lower_v, upper_v, b_v = np.ones(M), np.ones(M), np.ones(M)
  lower_x, upper_x, b_x = np.ones(N), np.ones(N), np.ones(N)

  # Diagonal of systems does not change
  diagonal_v = np.ones(M)
  diagonal_x = np.ones(N)

  cdef dict currents = dict(top=np.zeros(n_steps), 
                            bottom=np.zeros(n_steps), 
                            left=np.zeros(n_steps),
                            right=np.zeros(n_steps))

  for time_index in range(n_steps):
    # First evolution: differential wrt V
    # For each value of x, a tridiagonal system is solved to find values of v
    for i in range(N):

      # Prepares tridiagonal matrix and the constant term
      for j in range(M):
        diagonal_v[j] = 1 + 2*eta + theta*(a(x[i], v[i] + 0.5*dv, time_index*dt, physical_params) - a(x[i], v[i] - 0.5*dv, time_index*dt, physical_params))
        
        upper_v[j]  = - eta
        upper_v[j] += theta * a(x[i], v[j] + 0.5 * dv, t0 + time_index*dt, physical_params)

        lower_v[j] =  - eta
        lower_v[j] -= theta * a(x[i], v[j] + dv - 0.5 * dv, t0 + time_index*dt, physical_params)

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
      norm[time_index] = quad_int(p, integration_params)

    if save_current: 
      # Integral in v
      for j in range(M):
        currents['right'][time_index] += v[j]*p[j, N-2]*dv
        currents['left'][time_index] -= v[j]*p[j, 1]*dv

      # Integral in x
      for i in range(N):
        currents['top'][time_index] += a(x[i], v[M-2],  t0 + time_index*dt, physical_params)*p[M-2, i]*dx
        currents['top'][time_index] += 0.5*physical_params['sigma']**2*( (p[M-1,i] - p[M-2,i])/dv )*dx

        currents['bottom'][time_index] -= a(x[i], v[2],  t0 + time_index*dt, physical_params)*p[2, i]*dx
        currents['bottom'][time_index] -= 0.5*physical_params['sigma']**2*( (p[1,i] - p[0,i])/dv )*dx

  return p, norm, currents


cpdef advect_v(double [:] p0, double x, dict physical_params, dict integration_params):
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
  cdef double theta = 0.5 * dt/dv
  
  # Declarations of the diagonals
  cdef double [:] lower, diagonal, upper, b
  lower, diagonal, upper, b = np.ones(M), np.ones(M), np.ones(M), np.ones(M)

  for time_index in range(n_steps):
    time = t0 + time_index*dt
      
    # Constant part of coefficient vector does not depend on j
    b =  p.copy()
    
    for j in range(M):

      # Note tha theta is absorbed in working variable
      a_plus  =  theta * a(x,v[j] + 0.5*dv, time, physical_params)
      a_minus =  theta * a(x,v[j] - 0.5*dv, time, physical_params)

      diagonal[j] = 1 +  a_plus - a_minus
      upper[j]    =   + a_plus
      lower[j]    =   - a_plus
      
    # Solves the tridiagonal system for the column
    p = tridiag(lower, diagonal, upper, b)
  return p

cpdef diffuse_v(double [:] p0, double x, dict physical_params, dict integration_params):
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
  cdef double theta = 0.5 * dt/dv
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

cpdef advect_diffuse_IMPL(double [:] p0, double x, dict physical_params, dict integration_params):
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
  
  cdef double [:] p = p0.copy()
  cdef double theta = 0.5 * dt/dv
  cdef double eta   = 0.5 * dt/dv**2
  
  if diffCN:
    eta = 0.5*eta

  # Declarations of the diagonals
  cdef double [:] lower, diagonal, upper, b
  lower, diagonal, upper, b = np.ones(M), np.ones(M), np.ones(M), np.ones(M)

  for time_index in range(n_steps):
    time = t0 + time_index*dt
    s = eta * sigma_squared(x, time, physical_params)
    b = p.copy()
    for j in range(M):

      # Note tha theta is absorbed in working variable
      a_plus  =  theta * a(x,v[j] + 0.5*dv, time, physical_params)
      a_minus =  theta * a(x,v[j] - 0.5*dv, time, physical_params)

      diagonal[j] = 1 + 2 * s +  a_plus - a_minus
      upper[j]    = - s + a_plus
      lower[j]    = - s - a_plus
      
      if diffCN:
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
  diffCN = integration_params.get('diffCN', False), integration_params.get('advCN', False)
  
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

  """
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

# cpdef advect_BW(double [:] p0, double x, dict physical_params, dict integration_params):
#   ## Time
#   cdef double dt   = integration_params['dt']
#   cdef unsigned int n_steps = integration_params['n_steps']
#   cdef double t0    = physical_params.get('t0', 0.0)

#   ## Space
#   cdef double Lv,dv
#   Lv,dv = map(integration_params.get, ["Lv", "dv"])

#   cdef unsigned int  M = int(Lv/dv)
#   cdef double [:] v = np.arange(-int(M)//2, int(M)//2)*dv

#   cdef unsigned int time_index = 0, j = 0

#   cdef double [:] p = p0.copy(), p_half = p0.copy()
#   cdef double theta = 0.5 * dt/dv
  
#   for time_index in range(n_steps):
#     time = t0 + time_index*dt
      
#     # Half step LW
#     for j in range(M):

#       # Note tha theta is absorbed in working variable
#       a_plus  =  theta * a(x,v[j] + dv, time, physical_params)
#       a_here =  theta * a(x,v[j], time, physical_params)

#       if a_here > 0:
      
#       else:


#   return p